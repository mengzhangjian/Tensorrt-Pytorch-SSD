# Tensorrt-Pytorch-SSD

This repo is TensorRT SSD version for [Pytroch-SSD-repo](https://github.com/qfgaohao/pytorch-ssd). I have tested ssd-mobilenet-v1 and add Sort tracking inferface on Jetson Tx2.

## Buidling
* Jetson Tx2 Jetson Version:
* L4T 32.4.4 [ JetPack UNKNOWN ]
   Ubuntu 18.04.5 LTS
   Kernel Version: 4.9.140-tegra
* CUDA 10.2.89
   CUDA Architecture: 6.2
* OpenCV version: 4.1.1
   OpenCV Cuda: NO
* CUDNN: 8.0.0.180
* TensorRT: 7.1.3.0
* Vision Works: 1.6.0.501
* VPI: 0.4.4
You can get above information for Jetson devices following [this repo](https://github.com/jetsonhacks/jetsonUtilities)
I have provided converted  ssd-mobilnetV1-coco onnx model on /data/ssd/
```
sudo apt-get install libboost-all-dev

git clone https://github.com/mengzhangjian/Tensorrt-Pytorch-SSD.git 
mkdir build && cd build
cmake .. && make
```
## Training from Scratch
You should flow [Pytroch-SSD-repo](https://github.com/qfgaohao/pytorch-ssd) to train your own ssd model. If you work on Jetson devices, you can also flow this well-known [jetson-inference project](https://github.com/dusty-nv/jetson-inference/blob/master/docs/pytorch-ssd.md). This tensorRT version is also suitable for the jetson-inference SSD model. when get trained .pth, 
```
python3 onnx_export.py --model-dir=models/

```
This will generate onnx model, move it to /data/ssd/ and execute
```
./detectnet --fp16
```
