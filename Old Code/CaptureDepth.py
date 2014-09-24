import freenect
import cv2
import cv2.cv as cv
import numpy as np
import frame_convert

cv2.destroyAllWindows()

depth, timestamp = freenect.sync_get_depth()
img, timestamp = freenect.sync_get_video()

for i in xrange(depth.shape[0]):

    
    for j in xrange(depth.shape[1]):
        if depth[i,j] > 700:
            img[i,j] = [0,0,0]

        
cv2.imshow('image',img)
cv2.waitKey(0)
cv2.destroyAllWindows()


    


