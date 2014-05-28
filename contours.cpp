#include <stdlib.h>
#include <stdio.h>
#include <math.h>
#include <cv.h>
#include <highgui.h>
#include <opencv2/opencv.hpp>
#include "opencv2/imgproc/imgproc.hpp"

using namespace std;
using namespace cv;

int thresh = 100;
int max_thresh = 255;
RNG rng(12345);
int main(int argc, char *argv[])
{
  Mat src= imread(argv[1],1);
  Mat src_gray;
  cvtColor(src, src_gray, CV_BGR2GRAY);
  blur(src_gray, src_gray, Size(3,3));
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
  /// Draw contours
  Mat drawing = Mat::zeros( canny_output.size(), CV_8UC3 );
  for( int i = 0; i< contours.size(); i++ )
  {
    Scalar color = Scalar( rng.uniform(0, 255), rng.uniform(0,255), rng.uniform(0,255) );
    drawContours( drawing, contours, i, color, 2, 8, hierarchy, 0, Point() );
  }
  
  /* Create Window */
  string source_window = "Source";
  namedWindow( source_window, CV_WINDOW_AUTOSIZE );
  moveWindow(source_window, 100,100);
  imshow( source_window, src );
  namedWindow("contours", CV_WINDOW_AUTOSIZE );
  resizeWindow("contours", 100,100);
  imshow("contours", drawing);

//  createTrackbar( " Canny thresh:", "Source", &thresh, max_thresh, thresh_callback );
//  thresh_callback( 0, 0 );

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
}
