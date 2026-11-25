#include <opencv2/opencv.hpp>
using namespace cv;

void drawDashedLine(Mat &img, Point pt1, Point pt2, Scalar color, int thickness, int lineType,
		    int dashLength)
{
	double lineLength = norm(pt1 - pt2);
	double angle = atan2(pt2.y - pt1.y, pt2.x - pt1.x);

	Point p1 = pt1;
	Point p2;
	bool draw = true;

	for (double d = 0; d < lineLength; d += dashLength) {
		if (draw) {
			p2.x = pt1.x +
			       static_cast<int>(cos(angle) * std::min(d + dashLength, lineLength));
			p2.y = pt1.y +
			       static_cast<int>(sin(angle) * std::min(d + dashLength, lineLength));
			line(img, p1, p2, color, thickness, lineType);
		}
		p1.x = pt1.x + static_cast<int>(cos(angle) * (d + dashLength));
		p1.y = pt1.y + static_cast<int>(sin(angle) * (d + dashLength));
		draw = !draw;
	}
}

void drawDashedRectangle(Mat &img, Rect rect, Scalar color, int thickness, int lineType,
			 int dashLength)
{
	Point pt1(rect.x, rect.y);
	Point pt2(rect.x + rect.width, rect.y);
	Point pt3(rect.x + rect.width, rect.y + rect.height);
	Point pt4(rect.x, rect.y + rect.height);

	drawDashedLine(img, pt1, pt2, color, thickness, lineType, dashLength);
	drawDashedLine(img, pt2, pt3, color, thickness, lineType, dashLength);
	drawDashedLine(img, pt3, pt4, color, thickness, lineType, dashLength);
	drawDashedLine(img, pt4, pt1, color, thickness, lineType, dashLength);
}

#include <opencv2/imgproc.hpp>
#include <algorithm>

void blendFillRects(cv::Mat &frameBGR, const std::vector<cv::Rect> &rects, int feather,
		    int expand)
{
	const cv::Rect frameRect(0, 0, frameBGR.cols, frameBGR.rows);
	feather = std::max(0, feather);
	for (cv::Rect hole : rects) {
		hole.x -= expand;
		hole.y -= expand;
		hole.width += 2 * expand;
		hole.height += 2 * expand;
		hole &= frameRect;
		if (hole.width < 2 || hole.height < 2)
			continue;

		// region of interest: the hole plus a border of known pixels to sample from
		const int mx = (int)(hole.width * 0.35f) + feather * 2 + 4;
		const int my = (int)(hole.height * 0.35f) + feather * 2 + 4;
		cv::Rect roiRect(hole.x - mx, hole.y - my, hole.width + 2 * mx,
				 hole.height + 2 * my);
		roiRect &= frameRect;
		if (roiRect == hole)
			continue; // no surrounding pixels to sample

		cv::Mat roi;
		frameBGR(roiRect).convertTo(roi, CV_32FC3);
		cv::Mat holeMask = cv::Mat::zeros(roi.size(), CV_32F);
		const cv::Rect localHole(hole.x - roiRect.x, hole.y - roiRect.y, hole.width,
					 hole.height);
		holeMask(localHole).setTo(1.0f);
		cv::Mat weight = 1.0f - holeMask;

		// push: build pyramids of weighted colour and weight
		std::vector<cv::Mat> colours, weights;
		cv::Mat weight3;
		cv::merge(std::vector<cv::Mat>{weight, weight, weight}, weight3);
		colours.push_back(roi.mul(weight3));
		weights.push_back(weight);
		while (std::min(colours.back().rows, colours.back().cols) > 2) {
			cv::Mat c, w;
			cv::pyrDown(colours.back(), c);
			cv::pyrDown(weights.back(), w);
			colours.push_back(c);
			weights.push_back(w);
		}

		auto normalise = [](const cv::Mat &c, const cv::Mat &w) {
			cv::Mat wSafe = cv::max(w, 1e-4f), w3, out;
			cv::merge(std::vector<cv::Mat>{wSafe, wSafe, wSafe}, w3);
			cv::divide(c, w3, out);
			return out;
		};

		// pull: fill each level from the coarser estimate where data is missing
		cv::Mat estimate = normalise(colours.back(), weights.back());
		for (int l = (int)colours.size() - 2; l >= 0; --l) {
			cv::Mat up;
			cv::resize(estimate, up, colours[l].size(), 0, 0, cv::INTER_LINEAR);
			cv::Mat alpha = cv::min(cv::max(weights[l] * 2.0f, 0.0f), 1.0f), alpha3;
			cv::merge(std::vector<cv::Mat>{alpha, alpha, alpha}, alpha3);
			cv::Mat known = normalise(colours[l], weights[l]);
			estimate = known.mul(alpha3) + up.mul(cv::Scalar::all(1.0) - alpha3);
		}

		// grain: match the fine texture level of the ring around the hole
		cv::Mat blurred, highpass;
		cv::GaussianBlur(roi, blurred, cv::Size(0, 0), 2.0);
		highpass = roi - blurred;
		cv::Mat holeU8, ring;
		holeMask.convertTo(holeU8, CV_8U, 255);
		cv::dilate(holeU8, ring, cv::Mat::ones(9, 9, CV_8U));
		ring -= holeU8;
		cv::Scalar mean, stddev;
		if (cv::countNonZero(ring) > 0) {
			cv::meanStdDev(highpass, mean, stddev, ring);
		}
		const double sd = (stddev[0] + stddev[1] + stddev[2]) / 3.0;
		if (sd > 0.0) {
			cv::Mat noise(roi.size(), CV_32F), noise3;
			cv::RNG rng(12345); // fixed seed: the grain stays still instead of shimmering
			rng.fill(noise, cv::RNG::NORMAL, 0.0, sd);
			cv::merge(std::vector<cv::Mat>{noise, noise, noise}, noise3);
			estimate += noise3;
		}

		// feathered composite
		cv::Mat soft = holeMask.clone();
		if (feather > 0) {
			cv::dilate(soft, soft, cv::Mat::ones(feather + 1, feather + 1, CV_8U));
			cv::GaussianBlur(soft, soft, cv::Size(feather * 2 + 1, feather * 2 + 1), 0);
		}
		cv::Mat soft3;
		cv::merge(std::vector<cv::Mat>{soft, soft, soft}, soft3);
		cv::Mat result = estimate.mul(soft3) + roi.mul(cv::Scalar::all(1.0) - soft3);
		result.convertTo(frameBGR(roiRect), CV_8UC3);
	}
}
