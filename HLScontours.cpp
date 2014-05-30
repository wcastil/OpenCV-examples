#include <stdlib.h>
#include <stdio.h>
#include <math.h>
#include <cv.h>
#include <highgui.h>
#include <opencv2/opencv.hpp>
#include "opencv2/imgproc/imgproc.hpp"
#include "opencv2/core/core.hpp"
#include "opencv2/features2d/features2d.hpp"
#include "opencv2/highgui/highgui.hpp"
#include "opencv2/nonfree/nonfree.hpp"
#include "opencv2/calib3d/calib3d.hpp"


using namespace cv;
using namespace std;

void thresh_callback(int, void* );
int thresh = 150;
int max_thresh = 255;
Mat src, src_hls, src_gray;
RNG rng(12345);

Rect findWhiteReference(Mat src); 
int main(int argc, char *argv[])
{
  src= imread(argv[1],1);
  resize(src, src, Size(896,506));

  Rect brec = findWhiteReference(src); 
  Scalar color = Scalar( rng.uniform(0, 0), rng.uniform(0,255), rng.uniform(0,0) );
  rectangle(src, brec, color, 1); 
 

  /*Display */
  string source_window = "Source";
  namedWindow( source_window, CV_WINDOW_NORMAL );
  imshow( source_window, src );
  waitKey(0);

  return 0;
}

Rect findWhiteReference(Mat src){  
  medianBlur(src,src, 5);
  cout << "Blurred Imag\n";
  cvtColor(src, src_hls, CV_BGR2HLS);
  //cvtColor(src, src_gray, CV_BGR2GRAY);
//  blur(src_gray, src_gray, Size(3,3));
  Vec3b COLOR_MIN, COLOR_MAX;
  COLOR_MIN[0]=0;COLOR_MIN[1]=205;COLOR_MIN[2]=0;
  COLOR_MAX[0]=255;COLOR_MAX[1]=255;COLOR_MAX[2]=200;
  //vector<Vec3b> frame_threshold, thresh_dst;
  Mat frame_threshold, thresh_dst;
  cout << "Set Ranges\n";
  inRange(src_hls, COLOR_MIN, COLOR_MAX, frame_threshold);
  cout << "Ranged image\n";
  double ret =threshold(frame_threshold,thresh_dst , 127,255,0);  
  cout << "Thresholded image\n";


  vector<Vec4i> hierarchy;
  vector<vector<Point> > contours; 
  //remove Point?
  findContours( thresh_dst, contours, hierarchy, CV_RETR_TREE, CV_CHAIN_APPROX_SIMPLE, Point(0,0));
  cout << " Found Countours\n"; 

  /* Create Window */
  int max_index = 0;
  double max_area = 0.0;
  double a = 0.0;
  for (int c=0; c<contours.size(); c++){
    a = contourArea(contours[c]);
    if (a > max_area){
      max_area = a;
      max_index = c;
    }
  }
  Rect brec = boundingRect(contours[max_index]);
  return brec;
}
