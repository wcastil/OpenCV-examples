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

int thresh = 150;
int max_thresh = 255;
Mat src, src_hls, src_gray;
RNG rng(12345);

void findWhiteReference(Mat src, Rect& brec, vector<Point> &maxcontour); 
void findRelevantLines(Mat src, vector<Vec4i>& lines);
void getTransformationPoints(vector<vector<Point> >& contours_poly, Rect & boundRect, vector<Point2f>& quad_pts, vector<Point2f>& square_pts);

void drawDimensions(Mat& transformed, vector<Vec4i>& lines, double ratio, Scalar color);
cv::Point2f computeIntersect(cv::Vec4i a, cv::Vec4i b);
vector<Point2f> corners;
int main(int argc, char *argv[])
{
  src= imread(argv[1],1);
  Mat dst(src.rows,src.cols,CV_8UC1,Scalar::all(0)); //create destination image
  /*changing size changes results of relevant lines*/
  resize(src, src, Size(896,506));

  Mat transformed = Mat::zeros(src.rows, src.cols, CV_8UC3);
  Rect brec;
  vector<Point> maxcontour;
/* Find the Reference object */
  findWhiteReference(src, brec, maxcontour); 


/* Transform perspective  */
  vector<vector<Point> > contours_poly(1);
  approxPolyDP( Mat(maxcontour), contours_poly[0],5,true);
  Rect boundRect = boundingRect(maxcontour);
  cout << boundRect.height << " " << boundRect.width<< endl;
  /* Ensure square was found */
  if (contours_poly[0].size() == 4){
      vector<Point2f> quad_pts;
      vector<Point2f> square_pts;
      getTransformationPoints(contours_poly, boundRect, quad_pts, square_pts);

      cout << quad_pts << endl << square_pts << endl;
      //Mat transmtx = getPerspectiveTransform(Mat(quad_pts),Mat(square_pts));
      Mat transmtx = getPerspectiveTransform(quad_pts,square_pts);
//      Mat transmtx = getPerspectiveTransform(square_pts,quad_pts);

         warpPerspective(src, dst, transmtx, src.size());//square_pts.push_back(Point2f(boundRect.x+boundRect.width,boundRect.y+boundRect.height));

  }



/* Draw boundRect */
      Point P1=contours_poly[0][0];
      Point P2=contours_poly[0][1];
      Point P3=contours_poly[0][2];
      Point P4=contours_poly[0][3];
/*      line(src,P1,P2, Scalar(0,0,255),1,CV_AA,0);
      line(src,P2,P3, Scalar(0,0,255),1,CV_AA,0);
      line(src,P3,P4, Scalar(0,0,255),1,CV_AA,0);
      line(src,P4,P1, Scalar(0,0,255),1,CV_AA,0);
      rectangle(src,boundRect,Scalar(0,255,0),1,8,0);
*/

  Scalar color = Scalar( rng.uniform(0, 0), rng.uniform(0,255), rng.uniform(0,0) );
  Scalar color1 = Scalar( rng.uniform(0, 0), rng.uniform(0,0), rng.uniform(0,255) );


/* Find the relevant Lines in image */
  vector<Vec4i> lines; 

  //findWhiteReference(src, brec, maxcontour); 
  findRelevantLines(dst, lines);


//  Scalar color1 = Scalar( rng.uniform(0, 0), rng.uniform(0,0), rng.uniform(0,255) );
//  vector<vector<Point> > contours_poly(1);
  //approxPolyDP( Mat(maxcontour), contours_poly[0],5,true);
 // drawContours(tmp_img, contours_poly, 0, color1, 1,8);
  // Scalar color = Scalar( rng.uniform(0, 0), rng.uniform(0,255), rng.uniform(0,0) );
  //rectangle(src, brec, color, 1); 
//  rectangle(src,boundRect,Scalar(0,255,0),1,8,0);
  
  cout << "Width: " <<  boundRect.width << " Height:  " << boundRect.height<<endl;
  double naive_ratio = max(boundRect.width, boundRect.height)/11.0;
  double mratio = min(boundRect.width, boundRect.height)/8.5;
  cout << "Naive ratio: " << naive_ratio << endl;
  cout << "Naive ratio 2: " << mratio << endl;
  drawDimensions(dst, lines, naive_ratio, color);

 

  /*Display */
  imshow("quadrilateral", dst);
  waitKey(0);
  return 0;
}

void getTransformationPoints(vector<vector<Point> >& contours_poly, Rect & boundRect, vector<Point2f>& quad_pts, vector<Point2f>& square_pts)
{
  quad_pts.push_back(Point2f(contours_poly[0][1].x,contours_poly[0][1].y));
  quad_pts.push_back(Point2f(contours_poly[0][2].x,contours_poly[0][2].y));
  quad_pts.push_back(Point2f(contours_poly[0][3].x,contours_poly[0][3].y));
  quad_pts.push_back(Point2f(contours_poly[0][0].x,contours_poly[0][0].y));
  square_pts.push_back(Point2f(boundRect.x,boundRect.y));
  square_pts.push_back(Point2f(boundRect.x,boundRect.y+boundRect.height));
  square_pts.push_back(Point2f(boundRect.x+boundRect.width,boundRect.y+boundRect.height));
  square_pts.push_back(Point2f(boundRect.x+boundRect.width,boundRect.y));
}

void findWhiteReference(Mat src, Rect& brec, vector<Point> &maxcontour)
{
  medianBlur(src,src, 5);
  cout << "Blurred Imag\n";
  cvtColor(src, src_hls, CV_BGR2HLS);
  //cvtColor(src, src_gray, CV_BGR2GRAY);
//  blur(src_gray, src_gray, Size(3,3));
  Vec3b COLOR_MIN, COLOR_MAX;
  COLOR_MIN[0]=0;COLOR_MIN[1]=190;COLOR_MIN[2]=0;
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
  //  Rect b = boundingRect(contours[c]);

//    rectangle(tmp_img,b,Scalar(0,255,0),1,8,0);
    a = contourArea(contours[c]);
    if (a > max_area){
      max_area = a;
      max_index = c;
    }
  }
  maxcontour = contours[max_index];
  /*
  Scalar color1 = Scalar( rng.uniform(0, 0), rng.uniform(0,0), rng.uniform(0,255) );
  vector<vector<Point> > contours_poly(1);
  approxPolyDP( Mat(maxcontour), contours_poly[0],5,true);
  drawContours(tmp_img, contours_poly, 0, color1, 1,8);
  */
//  brec = boundingRect(contours[max_index]);

  //imshow("ref", tmp_img);
  //waitKey(0);
}

void drawDimensions(Mat& transformed, vector<Vec4i>& lines, double ratio, Scalar color)
{
    double dimx,dimy;
    for (int seg=0; seg<lines.size(); seg++){
      dimx = abs(lines[seg][0]-lines[seg][2])/ratio;
      dimy = abs(lines[seg][1]-lines[seg][3])/ratio;
      double length = norm(Point2f(lines[seg][0],lines[seg][1])-Point2f(lines[seg][2],lines[seg][3]))/ratio;
      if (dimx > 0.75 || dimy > 0.75){
        ostringstream s;
        s << "(" << length << ")";
        Point midpoint;
        double xdist = abs(lines[seg][0]-lines[seg][2]);
        double ydist = abs(lines[seg][1]-lines[seg][3]);
        if (xdist > ydist){
            double xpt = min(lines[seg][0], lines[seg][2])+xdist/2;
            midpoint = Point(xpt, lines[seg][1]);
                }
        else {
            double ypt = min(lines[seg][1],lines[seg][3])+ydist/2;
            midpoint = Point(lines[seg][0], ypt);
        }
        putText(transformed, s.str().c_str(), midpoint, FONT_HERSHEY_SIMPLEX,0.3,color, 1); 
      }
    }
}
void findRelevantLines(Mat src, vector<Vec4i>& lines){
  Mat src_gray;
  cout << src.size() << endl;
  cvtColor(src, src_gray, CV_BGR2GRAY);
//  blur(src_gray, src_gray, Size(3,3));
  cout << LSD_REFINE_NONE << endl;
  LineSegmentDetector* detector = createLineSegmentDetector(LSD_REFINE_NONE);
  //vector<Vec4i> lines;
  detector->detect(src_gray, lines);
  cout << lines[0] << endl;
  vector<Vec4i> big_lines;
  Vec4i line;
  for (int k =0; k < lines.size(); k++){
    line = lines[k];
    if (abs(line[0]-line[2]) > 10 || abs(line[1]-line[3]) > 10){
      big_lines.push_back(lines[k]);

    }

  }
  detector->drawSegments(src, big_lines);
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




/* Notes
  RotatedRect rbrec = minAreaRect(maxcontour);
  cout <<"Created RotatedRect\n";
  Mat bpoints;
  boxPoints(rbrec,bpoints); 
  cout <<"found Bpoints\n";
  cout << bpoints << endl;
  Point2f bl = bpoints.at<Point2f>(0,0);
  Point2f tl = bpoints.at<Point2f>(1,0);
  Point2f tr = bpoints.at<Point2f>(2,0);
  Point2f br = bpoints.at<Point2f>(3,0);
  circle(src, bl,4, color, 1);
  circle(src, tl,4, color, 1);
  circle(src, tr,4, color, 1);
  circle(src, br,4, color, 1);
  corners.push_back(tl);
  corners.push_back(tr);
  corners.push_back(br);
  corners.push_back(bl);
  double ratio = 1.2941176;
  double paperH=sqrt((tr.x-br.x)*(tr.x-br.x)+(tr.y-br.y)*(tr.y-br.y));
  double paperW=ratio*paperH;
  Rect R(bl.x,bl.y,paperW,paperH);
  cv::Mat quad = cv::Mat::zeros(src.size(), CV_8UC3);
  std::vector<cv::Point2f> quad_pts;
  quad_pts.push_back(Point2f(boundRect.x,boundRect.y));
  quad_pts.push_back(cv::Point2f(boundRect.x, boundRect.y+boundRect.height));
  quad_pts.push_back(cv::Point2f(boundRect.x+boundRect.width,boundRect.y));
  quad_pts.push_back(cv::Point2f(boundRect.x+boundRect.width, boundRect.y+boundRect.height));
  //cv::Mat transmtx = cv::getPerspectiveTransform(corners, quad_pts);
  //cv::warpPerspective(src, quad, transmtx, quad.size());
  //
  string source_window = "Source";
 // namedWindow( source_window, CV_WINDOW_NORMAL );
//  imshow( source_window, quad );
      imshow("quadrilateral", transformed);
      //imshow("thr",thr);
  //    imshow("dst",dst);
   //   imshow("src",src);
    //  imwrite("result1.jpg",dst);
     // imwrite("result2.jpg",src);
      //imwrite("result3.jpg",transformed);
*/
//  void circle(Mat& img, Point center, int radius, const Scalar& color, int thickness=1, int lineType=8, int shift=0)

