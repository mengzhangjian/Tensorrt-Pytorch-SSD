#pragma once
#include <opencv4/opencv2/opencv.hpp>

constexpr int kNumColors = 32;

constexpr int kMaxCoastCycles = 1;

constexpr int kMinHits = 3;

// Set threshold to 0 to accept all detections
constexpr float kMinConfidence = 0.3;

bool isInside(std::vector<cv::Point> &polygon, cv::Point p);
int InsidePolygon(std::vector<cv::Point> &polygon, cv::Point p);
bool doIntersect(cv::Point p1, cv::Point q1, cv::Point p2, cv::Point q2);
// int orientation(cv::Point p, cv::Point q, cv::Point r);
// bool isInside(std::vector<cv::Point> &polygon, cv::Point p);
