#include "common.h"

void _nms(int* keep_out, int* num_out, const float* boxes_host, int boxes_num, int boxes_dim, float nms_overlap_thresh, int device_id);

std::vector<int> nms(std::vector<samplesCommon::Bbox> bboxes, float threshold);