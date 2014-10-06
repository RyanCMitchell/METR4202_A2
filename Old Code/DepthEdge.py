from os.path import isfile, join
from os import listdir
import cv2
import numpy as np
import itertools
import sys
import freenect
from matplotlib import pyplot as plt

img, timestamp = freenect.sync_get_video()
depth1, timestamp = freenect.sync_get_depth(format=freenect.DEPTH_REGISTERED)
cv2.waitKey(1000)
depth2, timestamp = freenect.sync_get_depth(format=freenect.DEPTH_REGISTERED)
cv2.waitKey(1000)
depth3, timestamp = freenect.sync_get_depth(format=freenect.DEPTH_REGISTERED)
depth = depth1

# Average out depth noise
for i in xrange(depth.shape[0]):
    for j in xrange(depth.shape[1]):
        if depth1[i,j] <> 0 or depth2[i,j] <> 0 and depth3[i,j] <> 0:
            d1 = 0
            d2 = 0
            d3 = 0
            if depth1[i,j] <> 0:
                d1 = 1
            if depth2[i,j] <> 0:
                d2 = 1
            if depth3[i,j] <> 0:
                d3 = 1
            depth[i,j] = (depth1[i,j]+depth2[i,j]+depth3[i,j])/(d1+d2+d3)

# Black out 0 depth
for i in xrange(depth.shape[0]):
    for j in xrange(depth.shape[1]):
        if depth[i,j] == 0:
            img[i,j] = [0,0,0]

edges = cv2.Canny(img,100,200)

plt.subplot(121),plt.imshow(img,cmap = 'gray')
plt.title('Original Image'), plt.xticks([]), plt.yticks([])
plt.subplot(122),plt.imshow(edges,cmap = 'gray')
plt.title('Edge Image'), plt.xticks([]), plt.yticks([])

plt.show()



    
