import freenect
import cv2
import cv2.cv as cv
import numpy as np
from matplotlib import pyplot as plt
import frame_convert

#img =frame_convert.video_cv(img)
#img = cv2.imread('download.jpeg',0)
#imd = freenect.sync_get_depth()[0]


img = freenect.sync_get_video()[0]
gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)


# Initiate SIFT detector
sift = cv2.SIFT()
kp = sift.detect(gray, None)
#kp, des = sift.detectAndCompute(img2_gray,None)


img=cv2.drawKeypoints(gray,kp,flags=cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)
cv2.imwrite('cups.jpg', img)
#img2 = cv2.imread('sift_keypoints.jpg', 0)          # queryImage

'''
# Need an image to match to!!! as image1
img1 = cv2.imread('sift_keypoints.jpg', 0)


# find the keypoints and descriptors with SIFT
kp, des = sift.detectAndCompute(img2,None)
kp2, des2 = sift.detectAndCompute(img1,None)


# create BFMatcher object
bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)


# Match descriptors.
matches = bf.match(des1,des2)


# Sort them in the order of their distance.
matches = sorted(matches, key = lambda x:x.distance)


# Draw first 10 matches.
img3 = cv2.drawMatches(img1,kp1,img2,kp2,matches[:10], flags=2)


plt.imshow(img3),plt.show()
'''
