#pragma once
#include <iostream>
#include "opencv2/imgproc/imgproc.hpp"
#include "opencv2/highgui/highgui.hpp"
#include "opencv2/highgui/highgui_c.h"
#include "opencv2/features2d/features2d.hpp"

class Observator
{
public:
	bool paramTuning = false;

	std::vector<cv::KeyPoint> keypoints;

	int b_min = 0;
	int b_max = 0;
	int r_min = 0;
	int r_max = 0;
	int g_min = 0;
	int g_max = 0;

	int erosion_elem = 0;
	int erosion_size = 0;
	int dilation_elem = 0;
	int dilation_size = 0;
	int erosion_type = 0;
	int dilation_type = 0;

	int minTh = 0;
	int maxTh = 0;
	int minA = 0;
	int maxA = 0;
	int minCi = 0;
	int minCo = 0;
	int minIn = 0;

	std::vector<cv::Mat> dataStream;

	cv::Mat imageThresholding(cv::Mat im);
	cv::Mat erosionDilation(cv::Mat im);
	void blobDetection(cv::Mat EDim, cv::Mat im);

	Observator(bool paramTuning)
	{
		this->paramTuning = paramTuning;
	}
};