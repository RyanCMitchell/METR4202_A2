from os.path import isfile, join
from os import listdir
import cv2
import cv2.cv as cv
import numpy as np
import itertools
import sys


#Clear all cv windows
cv2.destroyAllWindows()

# prepare path
pathCups = "TrainingImages/LargeCup/"
testimages = [ f for f in listdir(pathCups) if isfile(join(pathCups,f)) and f[0]<>"."]

img = cv2.imread(str(pathCups+"/"+testimages[4]),0)
img = cv2.medianBlur(img,5)
cimg = cv2.cvtColor(img,cv2.COLOR_GRAY2BGR)

'''
http://stackoverflow.com/questions/10716464/what-are-the-correct-usage-parameter-values-for-houghcircles-in-opencv-for-iris
The key parameter is param2, the so-called accumulator threshold.
Basically, the higher it is the less circles you get. And these
circles have a higher probability of being correct. The best value
is different for every image. I think the best approach is to use
a parameter search on param2
'''
circles = cv2.HoughCircles(img, cv.CV_HOUGH_GRADIENT, 1, 20, param1=50, param2=30, minRadius=0, maxRadius=0)

circles = np.uint16(np.around(circles))
for i in circles[0,:]:
    # draw the outer circle
    cv2.circle(cimg,(i[0],i[1]),i[2],(0,255,0),2)
    # draw the center of the circle
    cv2.circle(cimg,(i[0],i[1]),2,(0,0,255),3)

cv2.imshow('detected circles',cimg)
cv2.waitKey(0)
cv2.destroyAllWindows()
