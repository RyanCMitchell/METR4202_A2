import freenect
import cv2
import cv2.cv as cv
import numpy as np
from matplotlib import pyplot as plt
import frame_convert


cup = cv2.imread('cupLarge1.jpg')                        # Template Image
cv2.imshow('image',cup)
cv2.waitKey(0)
cv2.destroyAllWindows()
cgray = cv2.cvtColor(cup, cv2.COLOR_BGR2GRAY)       # Convert to GrayScale
kinect = freenect.sync_get_video()[0]               # Kinect Image
kgray = cv2.cvtColor(kinect, cv2.COLOR_BGR2GRAY)    # Convert to GrayScale


# build feature detector and descriptor extractor
hessian_threshold = 85
detector = cv2.SURF(hessian_threshold)
(ckeypoints, cdescriptors) = detector.detect(cgray, None, useProvidedKeypoints = False)
(kkeypoints, kdescriptors) = detector.detect(kgray, None, useProvidedKeypoints = False)

# extract vectors of size 64 from raw descriptors numpy arrays
rowsize = len(kdescriptors) / len(kkeypoints)
if rowsize > 1:
    krows = numpy.array(kdescriptors, dtype = numpy.float32).reshape((-1, rowsize))
    crows = numpy.array(cdescriptors, dtype = numpy.float32).reshape((-1, rowsize))
    #print hrows.shape, nrows.shape
else:
    krows = numpy.array(kdescriptors, dtype = numpy.float32)
    crows = numpy.array(cdescriptors, dtype = numpy.float32)
    rowsize = len(krows[0])

# kNN training - learn mapping from krow to kkeypoints index
samples = krows
responses = numpy.arange(len(kkeypoints), dtype = numpy.float32)
#print len(samples), len(responses)
knn = cv2.KNearest()
knn.train(samples,responses)

# retrieve index and value through enumeration
for i, descriptor in enumerate(nrows):
    descriptor = numpy.array(descriptor, dtype = numpy.float32).reshape((1, rowsize))
    #print i, descriptor.shape, samples[0].shape
    retval, results, neigh_resp, dists = knn.find_nearest(descriptor, 1)
    res, dist =  int(results[0][0]), dists[0][0]
    #print res, dist

    if dist < 0.1:
        # draw matched keypoints in red color
        color = (0, 0, 255)
    else:
        # draw unmatched in blue color
        color = (255, 0, 0)
    # draw matched key points on haystack image
    x,y = kkeypoints[res].pt
    center = (int(x), int(y))
    cv2.circle(kinect, center, 2, color, -1)
    # draw matched key points on needle image
    x,y = ckeypoints[i].pt
    center = (int(x), int(y))
    cv2.circle(cup, center, 2, color, -1)

cv2.imshow('Kinect', kinect)
cv2.imshow('cup', cup)
cv2.waitKey(0)
cv2.destroyAllWindows()
