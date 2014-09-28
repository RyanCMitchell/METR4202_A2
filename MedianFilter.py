import freenect
import numpy as np
import cv2

def blurDepth(app):
    img, timestamp = freenect.sync_get_video()
    depth, timestamp = freenect.sync_get_depth(format=freenect.DEPTH_REGISTERED)
    blur = cv2.blur(depth,(app,app))
    # Black out 0 depth
    for i in xrange(img.shape[0]):
        for j in xrange(img.shape[1]):
            if blur[i,j] == 0 or blur[i,j]>1000:
                img[i,j] = [0,0,0]
    cv2.imshow("Cups Stream", img)
    cv2.waitKey(0)

blurDepth(5)

