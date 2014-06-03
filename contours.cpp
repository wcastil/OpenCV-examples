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


using namespace std;
using namespace cv;

void thresh_callback(int, void* );
int thresh = 150;
int max_thresh = 255;
Mat src, src_gray;
vector<cv::Point2f> corners;
RNG rng(12345);
int main(int argc, char *argv[])
{
  src= imread(argv[1],1);
  cout << src.size() << endl;
  resize(src, src, Size(896,506));
  cvtColor(src, src_gray, CV_BGR2GRAY);
  Ptr<LineSegmentDetector> detector = createLineSegmentDetector(LSD_REFINE_ADV);
  vector<Vec4i> lines;
  detector->detect(src_gray, lines);
  drawSegments(src, lines);

//  blur(src_gray, src_gray, Size(3,3));
  cout << "Converted to Gray scale, blurred" << endl;

  cout << src.type()%8 << endl;
  int row = src.rows; int col = src.cols;
  //Create contour
  Mat canny_output;
  vector<vector<Point> > contours; 
  vector<Vec4i> hierarchy;
  //Detect Edges using canny
 
  Canny(src_gray, canny_output, thresh, thresh*2, 3); 
  findContours( canny_output, contours, hierarchy, CV_RETR_TREE, CV_CHAIN_APPROX_SIMPLE, Point(0,0));
  cout << " Found Countours\n"; 

  /* Create Window */
  string source_window = "Source";

  namedWindow( source_window, CV_WINDOW_NORMAL );
  moveWindow(source_window, 100,100);
  imshow( source_window, src );
  namedWindow("contours", CV_WINDOW_NORMAL);
  resizeWindow("contours", 100,100);
  //imshow("contours", drawing);

   createTrackbar( " Canny thresh:", "Source", &thresh, max_thresh, thresh_callback );
    thresh_callback( 0, 0 );

  waitKey(0);


  /*  //    // Create Mask
      Mat_<uchar> mask(row,col);    
      for (int j=0; j<row; j++)
      for (int i=0; i<col; i++)
      {
      if ( pointPolygonTest( contours[0], Point2f(i,j),false) ==0)
      {mask(j,i)=255;}
      else
      {mask(j,i)=0;}
      };
      */
  return 0;
}
void thresh_callback(int, void* )
{
  Mat canny_output;
  vector<vector<Point> > contours;
  vector<Vec4i> hierarchy;

  /// Detect edges using canny
  Canny( src_gray, canny_output, thresh, thresh*2, 3 );
  /// Find contours
  findContours( canny_output, contours, hierarchy, CV_RETR_TREE, CV_CHAIN_APPROX_SIMPLE, Point(0, 0) );

  /// Draw contours
  Mat drawing = Mat::zeros( canny_output.size(), CV_8UC3 );
  int max_idx = 0;
  vector<int> max_indexes;
  int max_area = contourArea(contours[0]);
  for( int i = 0; i< contours.size(); i++ )
  {
    int c_area = contourArea(contours[i]);
    if (c_area > max_area){
      max_idx = i;
      max_area = c_area;
    }
    else if (c_area > max_area/2){
        max_indexes.push_back(i);

    }
  //  Scalar color = Scalar( rng.uniform(0, 255), rng.uniform(0,255), rng.uniform(0,255) );
   // drawContours( drawing, contours, i, color, 2, 8, hierarchy, 0, Point() );
  }

  Scalar color = Scalar( rng.uniform(0, 255), rng.uniform(0,255), rng.uniform(0,255) );
  drawContours( drawing, contours, max_idx, color, 2, 8, hierarchy, 0, Point() );
  for (int k=0; k < max_indexes.size(); k++){
    Scalar color = Scalar( rng.uniform(0, 255), rng.uniform(0,255), rng.uniform(0,255) );
    drawContours( drawing, contours, max_indexes[k], color, 2, 8, hierarchy, 0, Point() );
  }
  /// Show in a window
  namedWindow( "Contours", CV_WINDOW_AUTOSIZE );
  imshow( "Contours", drawing );
}

cv::Point2f computeIntersect(cv::Vec4i a, cv::Vec4i b)
{
  int x1 = a[0], y1 = a[1], x2 = a[2], y2 = a[3];
  int x3 = b[0], y3 = b[1], x4 = b[2], y4 = b[3];

  if (float d = ((float)(x1-x2) * (y3-y4)) - ((y1-y2) * (x3-x4)))
  {
    cv::Point2f pt;
    pt.x = ((x1*y2 - y1*x2) * (x3-x4) - (x1-x2) * (x3*y4 - y3*x4)) / d;
    pt.y = ((x1*y2 - y1*x2) * (y3-y4) - (y1-y2) * (x3*y4 - y3*x4)) / d;
    return pt;
  }
  else
    return cv::Point2f(-1, -1);
}
