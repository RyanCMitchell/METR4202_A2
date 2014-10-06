import freenect
import numpy as np
import cv2

img, timestamp = freenect.sync_get_video()
tdepth = []
for i in xrange(0,100):
    depth, timestamp = freenect.sync_get_depth(format=freenect.DEPTH_REGISTERED)
    tdepth.append(depth)
aveDepth = np.median(tdepth,0)

# Black out 0 depth
for i in xrange(img.shape[0]):
    for j in xrange(img.shape[1]):
        if aveDepth[i,j] == 0 or aveDepth[i,j]>1000:
            img[i,j] = [0,0,0]

cv2.imshow("Cups Stream", img)
cv2.waitKey(0)
cv2.destroyAllWindows()
