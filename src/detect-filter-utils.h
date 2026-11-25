#ifndef DETECT_FILTER_UTILS_H
#define DETECT_FILTER_UTILS_H

#include <opencv2/core/types.hpp>

void drawDashedLine(cv::Mat &img, cv::Point pt1, cv::Point pt2, cv::Scalar color, int thickness = 1,
		    int lineType = 8, int dashLength = 10);

void drawDashedRectangle(cv::Mat &img, cv::Rect rect, cv::Scalar color, int thickness = 1,
			 int lineType = 8, int dashLength = 10);

#endif // DETECT_FILTER_UTILS_H

#include <vector>
#include <opencv2/core/mat.hpp>

// Fill each rect with colour pulled in smoothly from the surrounding pixels
// (push-pull pyramid fill), add grain matched to the surroundings, and feather the edges.
// frameBGR must be CV_8UC3. expand grows each rect by that many pixels first.
void blendFillRects(cv::Mat &frameBGR, const std::vector<cv::Rect> &rects, int feather,
		    int expand);
