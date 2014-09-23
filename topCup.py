import cv2
import numpy as np
import freenect


##
# Converts an RGB image to grayscale, where each pixel
# now represents the intensity of the original image.
##
def rgb_to_gray(img):
    return cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
 
##
# Converts an image into a binary image at the specified threshold.
# All pixels with a value <= threshold become 0, while
# pixels > threshold become 1
def do_threshold(image, threshold = 170):
    (thresh, im_bw) = cv2.threshold(image, threshold, 255, cv2.THRESH_BINARY)
    return (thresh, im_bw)
 
 
##################################################### 
# If you have captured a frame from your camera like in the template program above,
# you can create a bitmap from it as follows:
 
#img_gray = rgb_to_gray(img_orig) # Convert img_orig from video camera from RGB to Grayscale
 
# Converts grayscale image to a binary image with a threshold value of 220. Any pixel with an
# intensity of <= 220 will be black, while any pixel with an intensity > 220 will be white:
#(thresh, img_threshold) = do_threshold(img_gray, 220)
 
#cv2.imshow("Grayscale", img_gray)
#cv2.imshow("Threshold", img_threshold)



#img = cv2.imread('TestImages/Test3.jpg', 0)
img, timestamp = freenect.sync_get_video()
img = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
img = cv2.medianBlur(img, 5)
#ret, thresh = cv2.threshold(img, 169, 255, 0)
thresh = cv2.adaptiveThreshold(img, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 11, 2)
contours, hierarchy = cv2.findContours(thresh, 1, 2)

#print contours
cntLengths = [len(i) for i in contours]
cntMax = cntLengths.index(max(cntLengths))

for i in range(len(contours)):
    if 10 < len(contours[i]) < 1500:
        cnt = contours[i]
        M = cv2.moments(cnt)
        #print M

        cx = int(M['m10']/M['m00'])
        cy = int(M['m01']/M['m00'])
        print "cx: ", cx
        print "cy: ", cy
         
        ellipse = cv2.fitEllipse(cnt)
        cv2.ellipse(img, ellipse, (0, 255, 0), 2)

cv2.imshow('img', img)
cv2.waitKey()
cv2.destroyAllWindows()
