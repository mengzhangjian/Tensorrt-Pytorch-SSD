#include "gpu_nms.hpp"

std::vector<int> nms(std::vector<samplesCommon::Bbox> bboxes, float threshold) {
    if (bboxes.empty()) {
        return std::vector<int>();
    }
    // 1.之前需要按照score排序
    auto *bboxes_1d = new float[bboxes.size() * 5];
    for (int i = 0; i < bboxes.size(); ++i) {
        bboxes_1d[i * 5] = bboxes[i].xmin;
        bboxes_1d[i * 5 + 1] = bboxes[i].ymin;
        bboxes_1d[i * 5 + 2] = bboxes[i].xmax;
        bboxes_1d[i * 5 + 3] = bboxes[i].ymax;
        bboxes_1d[i * 5 + 4] = bboxes[i].score;
    }

    // 2.device malloc cpy
    int *keep_output = new int[bboxes.size()];
    int *num_out = new int;
    _nms(keep_output, num_out, bboxes_1d, bboxes.size(), 5, threshold, 0);
    std::vector<int> keep_idx;
    keep_idx.insert(keep_idx.begin(), keep_output, keep_output + *num_out);
    delete[]bboxes_1d;
    delete[]keep_output;
    delete num_out;

    return keep_idx;
}