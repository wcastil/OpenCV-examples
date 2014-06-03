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

Mat src, src_gray;
int main(int argc, char *argv[])
{
  src= imread(argv[1],1);
  cout << src.size() << endl;
  resize(src, src, Size(896,506));
  cvtColor(src, src_gray, CV_BGR2GRAY);
  Ptr<LineSegmentDetector> detector = createLineSegmentDetector(LSD_REFINE_ADV);
  Ptr<LineSegmentDetector> detector_std = createLineSegmentDetector(LSD_REFINE_STD);
  Ptr<LineSegmentDetector> detector_none = createLineSegmentDetector(LSD_REFINE_NONE);
  vector<Vec4i> lines_std;
  vector<Vec4i> lines_none;
  vector<Vec4i> lines;
  detector->detect(src_gray, lines);
  detector_std->detect(src_gray, lines_std);
  detector_none->detect(src_gray, lines_none);
  cout << lines[0] << endl;
  vector<Vec4i> big_lines;
  Vec4i line;
  for (int k =0; k < lines.size(); k++){
    line = lines[k];
//    if (abs(line[0]-line[2]) > 10 || abs(line[1]-line[3]) > 10){
      big_lines.push_back(lines[k]);
  //  }

  }

  Mat src_std = src.clone(); 
  Mat src_none = src.clone();
  detector->drawSegments(src, lines);
  detector_std->drawSegments(src_std, lines_std);
  detector_none->drawSegments(src_none, lines_none);
  cout << lines.size() << " STD: " << lines_std.size() << " None: " << lines_none.size() << endl;



  /* Create Window */
  string source_window = "ADV";

  namedWindow( source_window, CV_WINDOW_NORMAL );
  moveWindow(source_window, 100,100);
  imshow( source_window, src );
  imshow( "STD", src_std );
  imshow( "none", src_none );


  waitKey(0);


  return 0;
}
