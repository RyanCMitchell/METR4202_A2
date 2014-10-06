import freenect
import cv2
import cv2.cv as cv
import numpy as np
import frame_convert

"""
cv.NamedWindow('Depth')
cv.NamedWindow('RGB')
keep_running = True


def display_depth(dev, data, timestamp):
    global keep_running
    cv.ShowImage('Depth', frame_convert.pretty_depth_cv(data))
    if cv.WaitKey(10) == 27:
        keep_running = False


def display_rgb(dev, data, timestamp):
    global keep_running
    cv.ShowImage('RGB', frame_convert.video_cv(data))
    if cv.WaitKey(10) == 27:
        keep_running = False


def body(*args):
    if not keep_running:
        raise freenect.Kill


print('Press ESC in window to stop')
freenect.runloop(depth=display_depth,
                 video=display_rgb,
                 body=body)
"""

img = freenect.sync_get_video()[0]
#img =frame_convert.video_cv(img)
#img = cv2.imread('download.jpeg',0)
imd = freenect.sync_get_depth()[0]
img = cv2.medianBlur(img,5)
img = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
cimg = cv2.cvtColor(img,cv2.COLOR_GRAY2BGR)

circles = cv2.HoughCircles(img,cv.CV_HOUGH_GRADIENT,1,20,param1=50,param2=30,minRadius=0,maxRadius=0)

circles = np.uint16(np.around(circles))
for i in circles[0,:]:
    # draw the outer circle
    cv2.circle(cimg,(i[0],i[1]),i[2],(0,255,0),2)
    # draw the center of the circle
    cv2.circle(cimg,(i[0],i[1]),2,(0,0,255),3)

cv2.imshow('detected circles',cimg)
cv2.waitKey(0)
cv2.destroyAllWindows()

"""
img = cv2.imread('download.jpeg',0)
print(type(img))
cv2.imshow('image',img)
cv2.waitKey(0)
cv2.destroyAllWindows()
"""
