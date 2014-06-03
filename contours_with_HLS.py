import numpy as np
import cv2
from matplotlib import pyplot as plt


im = cv2.imread('paper3.jpg')
im = cv2.resize(im, (866, 506))
im = cv2.medianBlur(im,5)    # 5 is a fairly small kernel size
hsv_img = cv2.cvtColor(im, cv2.COLOR_BGR2HLS)
COLOR_MIN = np.array([0, 190, 0],np.uint8)
COLOR_MAX = np.array([255, 255, 200],np.uint8)
frame_threshed = cv2.inRange(hsv_img, COLOR_MIN, COLOR_MAX)
imgray = frame_threshed
ret,thresh = cv2.threshold(frame_threshed,127,255,0)
_, contours, hierarchy = cv2.findContours(thresh,cv2.RETR_TREE,cv2.CHAIN_APPROX_SIMPLE)

# Find the index of the largest contour
areas = [cv2.contourArea(c) for c in contours]
max_index = np.argmax(areas)
#areas[max_index] =0
#max_index = np.argmax(areas)
cnt=contours[max_index]

x,y,w,h = cv2.boundingRect(cnt)
cv2.rectangle(im,(x,y),(x+w,y+h),(0,255,0),2)
print "Size in pixels: "
print  w,h
cv2.imshow("Show",im)
cv2.waitKey()
cv2.destroyAllWindows()
