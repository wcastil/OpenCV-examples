#!/bin/bash
clear
echo "Compiling contours.cpp"
g++ contours.cpp -o contour     -I /usr/local/include/opencv -L /usr/local/lib -lm -lopencv_highgui -lopencv_core -lopencv_imgproc
