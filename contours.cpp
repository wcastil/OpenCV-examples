#include <stdlib.h>
#include <stdio.h>
#include <math.h>
#include <cv.h>
#include <highgui.h>
#include <opencv2/opencv.hpp>
#include "opencv2/imgproc/imgproc.hpp"

using namespace std;
using namespace cv;

int main(int argc, char *argv[])
{
  Mat src= imread(argv[1]);
  cout << src.type()%8 << endl;
  src.convertTo(src, CV_8U);
  cout << "Converted Image to 8U" << endl;

  cout << src.type()%8 << endl;
  int row = src.rows; int col = src.cols;
  //Create contour
  vector<vector<Point> > contours; 
  vector<Vec4i> hierarchy;
  Mat src_copy = src.clone();
  findContours( src_copy, contours, hierarchy, RETR_TREE, CHAIN_APPROX_SIMPLE);
  cout << " Found Countours\n"; 

  //    // Create Mask
  Mat_<uchar> mask(row,col);    
  for (int j=0; j<row; j++)
  for (int i=0; i<col; i++)
  {
    if ( pointPolygonTest( contours[0], Point2f(i,j),false) ==0)
    {mask(j,i)=255;}
    else
    {mask(j,i)=0;}
  };
}
