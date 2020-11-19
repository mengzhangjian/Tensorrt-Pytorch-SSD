#include <opencv4/opencv2/opencv.hpp>
#include <experimental/filesystem>
#include "BatchStream.h"
#include "EntropyCalibrator.h"
#include "argsParser.h"
#include "buffers.h"
#include "logger.h"
#include "parserOnnxConfig.h"
#include "gpu_nms.hpp"
#include "NvInfer.h"
#include <cuda_runtime_api.h>
#include <cstdlib>
#include <fstream>
#include <iostream>
#include <sstream>

#include <map>
#include <random>
#include "tracker.h"
#include "utils.h"

std::string get_tegra_pipeline(int width, int height, int fps) {
    return "nvcamerasrc ! video/x-raw(memory:NVMM), width=(int)" + std::to_string(width) + ", height=(int)" +
           std::to_string(height) + ", format=(string)I420, framerate=(fraction)" + std::to_string(fps) +
           "/1 ! nvvidconv flip-method=2 ! video/x-raw, format=(string)BGRx ! videoconvert ! video/x-raw, format=(string)BGR ! appsink";
}

bool comp(const samplesCommon::Bbox &a, const samplesCommon::Bbox &b)
{
    return a.score > b.score;
}

const std::string gSampleName = "TensorRT.sample_onnx_ssd";

bool fileExists(const std::string filename)
{
    if (!std::experimental::filesystem::exists(std::experimental::filesystem::path(filename)))
    {
        std::cout<<"File does not exist: "<< filename << std::endl;
        return false;
    }
    return true;
}

//!
//! \brief The OnnxSSDParams structure groups the additional parameters required by
//!         the Uff SSD sample.
//!
struct SampleOnnxSSDParams : public samplesCommon::SampleParams
{
    std::string onnxFileName;    //!< The file name of the Onnx model to use
    std::string labelsFileName; //!< The file namefo the class labels
    int outputClsSize;          //!< The number of output classes
    int calBatchSize;           //!< The size of calibration batch
    int nbCalBatches;           //!< The number of batches for calibration
    int keepTopK;               //!< The maximum number of detection post-NMS
    float visualThreshold;      //!< The minimum score threshold to consider a detection
};

//! \brief  The OnnxSSD class implements the SSD sample
//!
//! \details It creates the network using an UFF model
//!
class OnnxSSD
{
    template <typename T>
    using SampleUniquePtr = std::unique_ptr<T, samplesCommon::InferDeleter>;

public:
    OnnxSSD(const SampleOnnxSSDParams & params)
        : mParams(params)
        , mEngine(nullptr)
    {
    }

    //!
    //! \brief Function builds the network engine
    //!
    bool build();

    //!
    //! \brief Runs the TensorRT inference engine for this sample
    //!
    bool infer();

    //!
    //! \brief Cleans up any state created in the sample class
    //!
    bool teardown();

private:
    SampleOnnxSSDParams mParams; //!< The parameters for the sample.

    nvinfer1::Dims mInputDims; //!< The dimensions of the input to the network.
    cv::Mat SsdImg;

    std::shared_ptr<nvinfer1::ICudaEngine> mEngine; //!< The TensorRT engine used to run the network

    //!
    //! \brief Parses an Onnx model for SSD and creates a TensorRT network
    //!
    bool constructNetwork(SampleUniquePtr<nvinfer1::IBuilder>& builder,
        SampleUniquePtr<nvinfer1::INetworkDefinition>& network, SampleUniquePtr<nvinfer1::IBuilderConfig>& config,
        SampleUniquePtr<nvonnxparser::IParser>& parser);

    //!
    //! \brief Reads the input and mean data, preprocesses, and stores the result in a managed buffer
    //!
    bool processInput(const samplesCommon::BufferManager& buffers, cv::Mat&img);

    //!
    //! \brief Filters output detections and verify results
    //!
    std::vector<cv::Rect> verifyOutput(const samplesCommon::BufferManager& buffers, cv::Mat&img);
};

//!
//! \return Returns true if the engine was created successfully and false otherwise
//!
bool OnnxSSD::build()
{
    initLibNvInferPlugins(&sample::gLogger.getTRTLogger(), "");

    auto builder = SampleUniquePtr<nvinfer1::IBuilder>(nvinfer1::createInferBuilder(sample::gLogger.getTRTLogger()));
    if (!builder)
    {
        return false;
    }
    const auto explicitBatch = 1U << static_cast<uint32_t>(nvinfer1::NetworkDefinitionCreationFlag::kEXPLICIT_BATCH);
    auto network = SampleUniquePtr<nvinfer1::INetworkDefinition>(builder->createNetworkV2(explicitBatch));
    if (!network)
    {
        return false;
    }

    auto config = SampleUniquePtr<nvinfer1::IBuilderConfig>(builder->createBuilderConfig());
    if (!config)
    {
        return false;
    }

    auto parser = SampleUniquePtr<nvonnxparser::IParser>(nvonnxparser::createParser(*network, sample::gLogger.getTRTLogger()));
    if (!parser)
    {
        return false;
    }

    auto constructed = constructNetwork(builder, network, config, parser);
    if (!constructed)
    {
        return false;
    }

    assert(network->getNbInputs() == 1);
    mInputDims = network->getInput(0)->getDimensions();
    auto dims =  network->getOutput(0)->getDimensions();
    assert(mInputDims.nbDims == 4);

    assert(network->getNbOutputs() == 2);

    return true;
}
//!
//! \param builder Pointer to the engine builder
//!
bool OnnxSSD::constructNetwork(SampleUniquePtr<nvinfer1::IBuilder>& builder,
    SampleUniquePtr<nvinfer1::INetworkDefinition>& network, SampleUniquePtr<nvinfer1::IBuilderConfig>& config,
    SampleUniquePtr<nvonnxparser::IParser>& parser)
{
    auto parsed = parser->parseFromFile(
        locateFile(mParams.onnxFileName, mParams.dataDirs).c_str(), static_cast<int>(sample::gLogger.getReportableSeverity()));
    if (!parsed)
    {
        return false;
    }

    builder->setMaxBatchSize(mParams.batchSize);
    config->setMaxWorkspaceSize(1_GiB);
    if (mParams.fp16)
    {
        config->setFlag(BuilderFlag::kFP16);
    }

    // Calibrator life time needs to last until after the engine is built.
    std::unique_ptr<IInt8Calibrator> calibrator;
    std::string precision = ((mParams.fp16==1) ? "fp16" : "fp32");
    
    std::string name_engine = "ssd_onnx." + precision + ".engine";
    if (mParams.int8)
    {
        sample::gLogInfo << "Using Entropy Calibrator 2" << std::endl;
        const std::string listFileName = "list.txt";
        const int imageC = 3;
        const int imageH = 300;
        const int imageW = 300;
        nvinfer1::DimsNCHW imageDims{};
        imageDims = nvinfer1::DimsNCHW{mParams.calBatchSize, imageC, imageH, imageW};
        BatchStream calibrationStream(
            mParams.calBatchSize, mParams.nbCalBatches, imageDims, listFileName, mParams.dataDirs);
        calibrator.reset(new Int8EntropyCalibrator2<BatchStream>(
            calibrationStream, 0, "UffSSD", mParams.inputTensorNames[0].c_str()));
        config->setFlag(BuilderFlag::kINT8);
        config->setInt8Calibrator(calibrator.get());
    }
    if(!fileExists(name_engine))
    {
        mEngine = std::shared_ptr<nvinfer1::ICudaEngine>(
            builder->buildEngineWithConfig(*network, *config), samplesCommon::InferDeleter());
        if(!mEngine)
            return false;
        IHostMemory* trtModelStream = mEngine->serialize();
        std::ofstream p(name_engine, std::ios::binary);
        p.write((const char*)trtModelStream->data(),trtModelStream->size());
        p.close();
        trtModelStream->destroy();

    }
    else{
      // cout << "Loading TensorRT engine from plan file..." << endl;
        sample::gLogInfo << "Loading TensorRT engine from plan file " << name_engine << std::endl;

        std::ifstream planFile(name_engine);

        if (!planFile.is_open())
        {
            sample::gLogError << "Could not open plan file: " << name_engine << std::endl;
            return false;
        }

        std::stringstream planBuffer;
        planBuffer << planFile.rdbuf();
        std::string plan = planBuffer.str();
        IRuntime *runtime = createInferRuntime(sample::gLogger.getTRTLogger());
        ICudaEngine *engine = runtime->deserializeCudaEngine((void*)plan.data(), plan.size(), nullptr);

        mEngine = std::shared_ptr<nvinfer1::ICudaEngine>(engine, samplesCommon::InferDeleter());
        }

        if (!mEngine)
        {
            return false;
        }

    return true;
}

//!
//! \brief Runs the TensorRT inference engine for this sample
//!
//! \details This function is the main execution function of the sample. It allocates the buffer,
//!          sets inputs and executes the engine.
//!

/*
bool OnnxSSD::infer()
{
    samplesCommon::BufferManager buffers(mEngine);

    auto context = SampleUniquePtr<nvinfer1::IExecutionContext>(mEngine->createExecutionContext());

    if (!context)
    {
        return false;
    }

    // Read the input data into the managed buffers
    assert(mParams.inputTensorNames.size() == 1);
    int WIDTH = 1920;
    int HEIGHT = 1080;
    int FPS = 25;
    std::string pipeline = get_tegra_pipeline(WIDTH, HEIGHT, FPS);
    
    cv::VideoCapture cap("4.h264"); 


    if(!cap.isOpened())
    {
        std::cout<<"Error opening video stream or file" <<std::endl;
        return -1;
    }
        // Generate random colors to visualize different bbox
    std::random_device rd;  //Will be used to obtain a seed for the random number engine
    std::mt19937 gen(rd()); //Standard mersenne_twister_engine seeded with rd()
    constexpr int max_random_value = 20;
    std::uniform_int_distribution<> dis(0, max_random_value);
    constexpr int factor = 255 / max_random_value;
    std::vector<cv::Scalar> colors;
    for (int n = 0; n < kNumColors; ++n) 
    {
        //Use dis to transform the random unsigned int generated by gen into an int in [0, 7]
        colors.emplace_back(cv::Scalar(dis(gen) * factor, dis(gen) * factor, dis(gen) * factor));
    }
    Tracker tracker;
    std::vector<cv::Point> pig_area;
    cv::FileStorage fs("config.yaml", cv::FileStorage::READ);
    // for(int i = 0; i< 100; i ++)
    // {
    //     s.emplace_back(cv::Point(i, i + 1));
    // }
    // fs << "Area" << s;
    fs["Area"] >> pig_area;
    // std::cout << pig_area.size() << std::endl;
    // // if(!(pig_area.size() % 2))
    //     // std::cout>>"please check pig area point number"<<std::endl;
    // for(int i =0; i < pig_area.size(); i++)
    // {
    //     std::cout << pig_area[i].x << ' ' << pig_area[i].y << std::endl;
    // }
    const cv::Point *pts = (const cv::Point*)cv::Mat(pig_area).data;
    int npts = cv::Mat(pig_area).rows;
    std::set<int> previous_tracker_id;
    int count=0;
    while(1)
    {
        cv::Mat frame;
        cap >> frame;
        if(frame.empty())
            break;
        cv::polylines(frame, &pts, &npts, 1, false, cv::Scalar(0, 255, 0), 3);
        cv::line(frame, pig_area[0], pig_area[pig_area.size() - 1], cv::Scalar(0, 0, 255), 3);
        
        if (!processInput(buffers, frame))
            {
                return false;
            }

            // Memcpy from host input buffers to device input buffers
            buffers.copyInputToDevice();

            bool status = context->execute(mParams.batchSize, buffers.getDeviceBindings().data());
            if (!status)
            {
                return false;
            }
            // Memcpy from device output buffers to host output buffers
            buffers.copyOutputToHost();

            // Post-process detections and verify results
        std::vector<cv::Rect> detections = verifyOutput(buffers, frame);
        
        tracker.Run(detections);
        const auto tracks = tracker.GetTracks();
        for (auto &trk : tracks) 
        {
        // only draw tracks which meet certain criteria
        if (trk.second.coast_cycles_ < kMaxCoastCycles &&
            (trk.second.hit_streak_ >= kMinHits)) {
            const auto &bbox = trk.second.GetStateAsBbox();
            cv::putText(frame, std::to_string(trk.first), cv::Point(bbox.tl().x, bbox.tl().y - 10),
                        cv::FONT_HERSHEY_DUPLEX, 2, cv::Scalar(255, 255, 255), 2);
            cv::rectangle(frame, bbox, colors[trk.first % kNumColors], 3);
             cv::Point p0 = cv::Point(bbox.tl().x, bbox.tl().y);
            cv::Point p1 = cv::Point(bbox.tl().x + bbox.width, bbox.tl().y);

            if(doIntersect(p0, p1, pig_area[0], pig_area[pig_area.size() - 1]))
            {
                if(previous_tracker_id.count(trk.first)==0)
                {
                    count++;
                    previous_tracker_id.emplace(trk.first);
                    std::cout<<"pig number: "<<count<<std::endl;
            }
        }
            }
        }
       
        cv::imshow("img", frame);
        cv::waitKey(1);

    }

    return true;
}
*/
bool OnnxSSD::infer()
{
    samplesCommon::BufferManager buffers(mEngine);

    auto context = SampleUniquePtr<nvinfer1::IExecutionContext>(mEngine->createExecutionContext());

    if (!context)
    {
        return false;
    }

    // Read the input data into the managed buffers
    assert(mParams.inputTensorNames.size() == 1);
    
    cv::Mat frame = cv::imread("test.jpg");
    if (!processInput(buffers, frame))
        {
            return false;
        }

    // Memcpy from host input buffers to device input buffers
    buffers.copyInputToDevice();

    bool status = context->execute(mParams.batchSize, buffers.getDeviceBindings().data());
    if (!status)
    {
        return false;
    }
    // Memcpy from device output buffers to host output buffers
    buffers.copyOutputToHost();

    // Post-process detections and verify results
    std::vector<cv::Rect> detections = verifyOutput(buffers, frame);
        
    cv::imshow("img", frame);
    cv::waitKey(0);

    return true;
}
//!
//! \brief Reads the input and mean data, preprocesses, and stores the result in a managed buffer
//!
bool OnnxSSD::processInput(const samplesCommon::BufferManager& buffers, cv::Mat &SsdImg)
{
    const int inputC = mInputDims.d[1];
    const int inputH = mInputDims.d[2];
    const int inputW = mInputDims.d[3];
    const int batchSize = mParams.batchSize;
    
    // cv::cuda::GpuMat dst;
    // dst.upload(img);
    // cv::cuda::resize(dst, dst, cv::Size(300, 300));
    cv::Mat img;
    cv::resize(SsdImg, img,  cv::Size(300, 300));
    // dst.download(img);


    const int  volImg = inputC * inputH * inputW;
    const int  volChl = inputH * inputW;

    float* hostDataBuffer = static_cast<float*>(buffers.getHostBuffer(mParams.inputTensorNames[0]));
    // Host memory for input buffer
    for (int c = 0;c < inputC; ++c)
        {
            cv::Mat_<cv::Vec3b>::iterator it =img.begin<cv::Vec3b>();
            for (unsigned j = 0;j < volChl;++j)
            {
                hostDataBuffer[c*volChl + j] = (float)(*it)[c] / 255.0;
                it++;
            }
        }

    return true;
}

//!
//! \brief Filters output detections and verify result
//!
//! \return whether the detection output matches expectations
//!
std::vector<cv::Rect> OnnxSSD::verifyOutput(const samplesCommon::BufferManager& buffers, cv::Mat &image)
{
    const int inputH = mInputDims.d[2];
    const int inputW = mInputDims.d[3];
    const int batchSize = mParams.batchSize;
    const int keepTopK = mParams.keepTopK;
    const float visualThreshold = mParams.visualThreshold;
    const int outputClsSize = mParams.outputClsSize;

    const float* scores = static_cast<const float*>(buffers.getHostBuffer(mParams.outputTensorNames[0]));

    const float* bboxes = static_cast<const float*>(buffers.getHostBuffer(mParams.outputTensorNames[1]));

    std::vector<std::string> classes(outputClsSize);

    // Gather class labels
    std::ifstream labelFile(locateFile(mParams.labelsFileName, mParams.dataDirs));
    std::string line;
    int id = 0;
    while (getline(labelFile, line))
    {
        classes[id++] = line;
    }
    float scale_w = image.cols / (float)inputW;
    float scale_h = image.rows / (float)inputH;
    bool pass = true;
    
    std::vector<samplesCommon::Bbox> BBox;
    for(int i = 0; i < 3000; i++)
    {
        std::vector<float> conf;
        for(int j = 0; j < outputClsSize; j++)
        {
            conf.emplace_back(scores[i * outputClsSize + j]);
        }
        int max_index = std::max_element(conf.begin(), conf.end()) - conf.begin();
        if (max_index != 0)
        {   
            if(conf[max_index] < 0.5)
                continue;
            samplesCommon::Bbox b;
            int left = bboxes[i * 4] * scale_w * 300;
            int top = bboxes[i * 4 + 1] * scale_h * 300;
            int right = bboxes[ i * 4 + 2] * scale_w * 300;
            int bottom = bboxes[i * 4 + 3] * scale_h * 300;
            b.xmin = std::max(0, left);
            b.ymin = std::max(0, top);
            b.xmax = right;
            b.ymax = bottom;
            b.score = conf[max_index];
            b.cls_idx = max_index;
            BBox.emplace_back(b);
        }
        conf.clear();
    }
    std::sort(BBox.begin(), BBox.end(), comp);
    std::vector<int> keep_index = nms(BBox, 0.2);
    std::vector<cv::Rect> bbox_per_frame;
    for(int i = 0; i < keep_index.size(); i++)
    {
        int left = BBox[keep_index[i]].xmin;
        int top = BBox[keep_index[i]].ymin;
        int right = BBox[keep_index[i]].xmax;
        int bottom = BBox[keep_index[i]].ymax;
        int width = right - left;
        int height = bottom - top;
        int center_x = left + width / 2;
        cv::rectangle(image, cv::Point(left, top), cv::Point(right, bottom), cv::Scalar(0, 0, 255), 3);
        cv::putText(image, classes[BBox[keep_index[i]].cls_idx],  cv::Point(left, top), cv::FONT_HERSHEY_COMPLEX, 2, cv::Scalar(0, 255, 255), 2, 8, 0);
        /****comment for Count_Pig demo*****/
        /*
        if(center_x < 1600 && center_x > 500){
        bbox_per_frame.emplace_back(left, top, width, height);
        }
        */
        bbox_per_frame.emplace_back(left, top, width, height);
    }

    return bbox_per_frame;
}

SampleOnnxSSDParams initializeSampleParams(const samplesCommon::Args& args)
{
    SampleOnnxSSDParams params;
    if (args.dataDirs.empty()) //!< Use default directories if user hasn't provided directory paths
    {
        params.dataDirs.push_back("data/ssd/");
        params.dataDirs.push_back("data/ssd/VOC2007/");
        params.dataDirs.push_back("data/samples/ssd/");
        params.dataDirs.push_back("data/samples/ssd/VOC2007/");
    }
    else //!< Use the data directory provided by the user
    {
        params.dataDirs = args.dataDirs;
    }
    params.onnxFileName = "ssd-mobilenet-coco.onnx";
    params.labelsFileName = "labels.txt";
    params.inputTensorNames.push_back("input_0");
    params.batchSize = 1;
    params.outputTensorNames.push_back("scores");
    params.outputTensorNames.push_back("boxes");
    params.dlaCore = args.useDLACore;
    params.int8 = args.runInInt8;
    params.fp16 = args.runInFp16;

    params.outputClsSize = 21;
    params.calBatchSize = 10;
    params.nbCalBatches = 10;
    params.keepTopK = 100;
    params.visualThreshold = 0.6;

    return params;
}

//!
//! \brief Prints the help information for running this sample
//!
void printHelpInfo()
{
    std::cout
        << "Usage: ./sample_uff_ssd [-h or --help] [-d or --datadir=<path to data directory>] [--useDLACore=<int>]"
        << std::endl;
    std::cout << "--help          Display help information" << std::endl;
    std::cout << "--datadir       Specify path to a data directory, overriding the default. This option can be used "
                 "multiple times to add multiple directories. If no data directories are given, the default is to use "
                 "data/samples/ssd/ and data/ssd/"
              << std::endl;
    std::cout << "--useDLACore=N  Specify a DLA engine for layers that support DLA. Value can range from 0 to n-1, "
                 "where n is the number of DLA engines on the platform."
              << std::endl;
    std::cout << "--fp16          Specify to run in fp16 mode." << std::endl;
    std::cout << "--int8          Specify to run in int8 mode." << std::endl;
}

int main(int argc, char** argv)
{
    samplesCommon::Args args;
    bool argsOK = samplesCommon::parseArgs(args, argc, argv);
    if (!argsOK)
    {
        sample::gLogError << "Invalid arguments" << std::endl;
        printHelpInfo();
        return EXIT_FAILURE;
    }
    if (args.help)
    {
        printHelpInfo();
        return EXIT_SUCCESS;
    }

    auto sampleTest = sample::gLogger.defineTest(gSampleName, argc, argv);

    sample::gLogger.reportTestStart(sampleTest);
    OnnxSSD sample(initializeSampleParams(args));
    sample::gLogInfo << "Building and running a GPU inference engine for SSD" << std::endl;
    
    if (!sample.build())
    {
        return sample::gLogger.reportFail(sampleTest);
    }
    if (!sample.infer())
    {
        return sample::gLogger.reportFail(sampleTest);
    }
    return sample::gLogger.reportPass(sampleTest);
}
