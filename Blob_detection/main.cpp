
#include "opencv2/opencv.hpp"
#include "Observator.h"
using namespace cv;


int main(int argc, char** argv)
{	
    Observator observator(true);

    VideoCapture cap;
    // open the default camera, use something different from 0 otherwise;
    // Check VideoCapture documentation.
    if(!cap.open(1))
        return 0;
    
    for(;;)
    {
          Mat frame;
          cap >> frame;
          if( frame.empty() ) break; // end of video stream
          cv::Mat thresholdedImage = observator.imageThresholding(frame);
	      cv::Mat EDimage = observator.erosionDilation(thresholdedImage);
	      observator.blobDetection(EDimage, frame);
		  cv::imshow("blob_detection_Martijn", frame);
          if( waitKey(10) == 27 ) break; // stop capturing by pressing ESC 
    }
    return 0;

}