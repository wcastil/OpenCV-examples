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
  detector->drawSegments(src, lines);

  /* Create Window */
  string source_window = "Source";

  namedWindow( source_window, CV_WINDOW_NORMAL );
  moveWindow(source_window, 100,100);
  imshow( source_window, src );


  waitKey(0);


  return 0;
}
