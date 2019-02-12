//include "stdafx.h"
#include <iostream>
#include "opencv2/imgproc/imgproc.hpp"
#include "opencv2/highgui/highgui.hpp"
#include "opencv2/highgui/highgui_c.h"
#include "opencv2/features2d/features2d.hpp"
#include "Observator.h"


cv::Mat Observator::imageThresholding(cv::Mat im)
{
	cv::Mat im_out;

	cv::Mat image = im;
	
	if (paramTuning)
	{
		cv::namedWindow("Trackbar", 1);
		cv::createTrackbar("B_min", "Trackbar", &b_min, 255);
		cv::createTrackbar("B_max", "Trackbar", &b_max, 255);
		cv::createTrackbar("R_min", "Trackbar", &r_min, 255);
		cv::createTrackbar("R_max", "Trackbar", &r_max, 255);
		cv::createTrackbar("G_min", "Trackbar", &g_min, 255);
		cv::createTrackbar("G_max", "Trackbar", &g_max, 255);

		cv::inRange(image, cv::Scalar(b_min, g_min, r_min), cv::Scalar(b_max, g_max, r_max), im_out);
		cv::imshow("Image Thresholdig", im_out);
	}
	else
	{
		//cv::inRange(im, cv::Scalar(144, 72, 14), cv::Scalar(255, 136, 93), im_out);
		cv::inRange(im, cv::Scalar(91, 26, 40), cv::Scalar(255, 158, 121), im_out); //BGR
		//cv::imshow("Image Thresholdig", im_out);
	}

	return im_out;
}

cv::Mat Observator::erosionDilation(cv::Mat im)
{
	cv::Mat erosion_dst, im_out;

	if (paramTuning)
	{
		int const max_elem = 2;
		int const max_kernel_size = 21;

		/// Create windows
		cv::namedWindow("Erosion Dilation", CV_WINDOW_AUTOSIZE);
		// Element: 0: Rect, 1: Cross, 2: Ellipse
		cv::createTrackbar("E Element", "Erosion Dilation", &erosion_elem, max_elem);
		// Kernel size: 2n + 1
		cv::createTrackbar("E Size", "Erosion Dilation", &erosion_size, max_kernel_size);
		// Element: 0: Rect, 1: Cross, 2: Ellipse
		cv::createTrackbar("D Element", "Erosion Dilation", &dilation_elem, max_elem);
		// Kernel size: 2n +1
		cv::createTrackbar("D Size", "Erosion Dilation", &dilation_size, max_kernel_size);

		/// Apply the erosion operation
		if (erosion_elem == 0) { erosion_type = cv::MORPH_RECT; }
		else if (erosion_elem == 1) { erosion_type = cv::MORPH_CROSS; }
		else if (erosion_elem == 2) { erosion_type = cv::MORPH_ELLIPSE; }

		cv::Mat elementE = getStructuringElement(erosion_type, cv::Size(2 * erosion_size + 1, 2 * erosion_size + 1), cv::Point(erosion_size, erosion_size));

		erode(im, erosion_dst, elementE);

		/// Apply the dilation operation
		if (dilation_elem == 0) { dilation_type = cv::MORPH_RECT; }
		else if (dilation_elem == 1) { dilation_type = cv::MORPH_CROSS; }
		else if (dilation_elem == 2) { dilation_type = cv::MORPH_ELLIPSE; }

		cv::Mat elementD = getStructuringElement(dilation_type, cv::Size(2 * dilation_size + 1, 2 * dilation_size + 1), cv::Point(dilation_size, dilation_size));

		dilate(erosion_dst, im_out, elementD);
		cv::imshow("Erosion_Dilation image", im_out);
	}
	else
	{
		/// Apply the erosion operation
		erosion_type = cv::MORPH_ELLIPSE;
		erosion_size = 2;

		cv::Mat elementE = getStructuringElement(erosion_type, cv::Size(2 * erosion_size + 1, 2 * erosion_size + 1), cv::Point(erosion_size, erosion_size));

		erode(im, erosion_dst, elementE);

		/// Apply the dilation operation
		dilation_type = cv::MORPH_ELLIPSE;
		dilation_size = 4;

		cv::Mat elementD = getStructuringElement(dilation_type, cv::Size(2 * dilation_size + 1, 2 * dilation_size + 1), cv::Point(dilation_size, dilation_size));

		dilate(erosion_dst, im_out, elementD);
		cv::imshow("Erosion_Dilation image", im_out);
	}

	return im_out;
}

void Observator::blobDetection(cv::Mat EDim, cv::Mat im)
{
	cv::Mat im_out;

	if (paramTuning)
	{
		cv::namedWindow("Trackbar1", 1);
		cv::createTrackbar("minTh", "Trackbar1", &minTh, 1000);
		cv::createTrackbar("maxTh", "Trackbar1", &maxTh, 1000);
		cv::createTrackbar("minA", "Trackbar1", &minA, 2000);
		cv::createTrackbar("maxA", "Trackbar1", &maxA, 80000);
		cv::createTrackbar("minCi", "Trackbar1", &minCi, 100);
		cv::createTrackbar("minCo", "Trackbar1", &minCo, 100);
		cv::createTrackbar("minIn", "Trackbar1", &minIn, 100);

		// Setup SimpleBlobDetector parameters.
		cv::SimpleBlobDetector::Params params;

		// Change thresholds
		params.filterByColor = true;
		params.blobColor = 255;

		params.minThreshold = minTh;
		params.maxThreshold = maxTh;

		// Filter by Area.
		params.filterByArea = true;
		params.minArea = minA;
		params.maxArea = maxA;

		// Filter by Circularity
		params.filterByCircularity = true;
		params.minCircularity = (float)minCi / 100.0;

		// Filter by Convexity
		params.filterByConvexity = true;
		params.minConvexity = (float)minCo / 100.0;

		// Filter by Inertia
		params.filterByInertia = true;
		params.minInertiaRatio = (float)minIn / 100.0;

#if CV_MAJOR_VERSION < 3   // If you are using OpenCV 2

		// Set up detector with params
		SimpleBlobDetector detector(params);

		// Detect blobs
		detector.detect(EDim, keypoints);
#else

		// Set up detector with params
		cv::Ptr<cv::SimpleBlobDetector> detector = cv::SimpleBlobDetector::create(params);

		// Detect blobs
		detector->detect(EDim, keypoints);
#endif

		// Draw detected blobs as red circles.
		// DrawMatchesFlags::DRAW_RICH_KEYPOINTS flag ensures
		// the size of the circle corresponds to the size of blob

		//cv::Mat image = cv::imread("C:/Users/Martijn Cramer/Desktop/Test.png");

		cv::drawKeypoints(EDim, keypoints, im_out, cv::Scalar(0, 0, 255), cv::DrawMatchesFlags::DRAW_RICH_KEYPOINTS);
		cv::imshow("Blob detection", im_out);

	}
	else
	{
		// Setup SimpleBlobDetector parameters.
		cv::SimpleBlobDetector::Params params;

		// Change thresholds
		params.filterByColor = true;
		params.blobColor = 255;

		params.minThreshold = 0;
		params.maxThreshold = 250;

		// Filter by Area.
		params.filterByArea = true;
		params.minArea = 2000;
		params.maxArea = 100000;

		// Filter by Circularity
		params.filterByCircularity = true;
		params.minCircularity = 0;

		// Filter by Convexity
		params.filterByConvexity = true;
		params.minConvexity = 0;

		// Filter by Inertia
		params.filterByInertia = true;
		params.minInertiaRatio = 0;

#if CV_MAJOR_VERSION < 3   // If you are using OpenCV 2

		// Set up detector with params
		SimpleBlobDetector detector(params);

		// Detect blobs
		detector.detect(EDim, keypoints);
#else

		// Set up detector with params
		cv::Ptr<cv::SimpleBlobDetector> detector = cv::SimpleBlobDetector::create(params);

		// Detect blobs
		detector->detect(EDim, keypoints);
#endif

		// Draw detected blobs as red circles.
		// DrawMatchesFlags::DRAW_RICH_KEYPOINTS flag ensures
		// the size of the circle corresponds to the size of blob

		cv::drawKeypoints(im, keypoints, im_out, cv::Scalar(0, 0, 255), cv::DrawMatchesFlags::DRAW_RICH_KEYPOINTS);
		cv::imshow("Blob detection", im_out);
	}
}
