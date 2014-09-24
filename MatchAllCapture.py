from os.path import isfile, join
from os import listdir
import cv2
import numpy as np
import itertools
import sys
from MatchingFunctions import findKeyPoints, drawKeyPoints, match, findKeyPointsDist, drawImageMappedPoints
import freenect

#Clear all cv windows
cv2.destroyAllWindows()

#Prepare a list of different training images
pathlarge = "TrainingImages/LargeCup/"
pathmedium = "TrainingImages/MediumCup/"
pathsmall = "TrainingImages/SmallCup/"
pathtest = "TestImages"

largecups = [ f for f in listdir(pathlarge) if isfile(join(pathlarge,f)) and f[0]<>"."]
mediumcups = [ f for f in listdir(pathmedium) if isfile(join(pathmedium,f)) and f[0]<>"."]
smallcups = [ f for f in listdir(pathsmall) if isfile(join(pathsmall,f)) and f[0]<>"."]
testimages = [ f for f in listdir(pathtest) if isfile(join(pathtest,f)) and f[0]<>"."]

depthMatrix = np.float32([[0.98, 0, -13],[0, 1, 7]])


img, timestamp = freenect.sync_get_video()
depth, timestamp = freenect.sync_get_depth(format=freenect.DEPTH_REGISTERED)
#depth = cv2.warpAffine(depth, depthMatrix, (depth.shape[1], depth.shape[0]))
maxdist = 200  # 200 is default

print depth.shape

KeyPointsTotalList = []
DistsTotalList = []

for i in largecups+mediumcups+smallcups:
    if i in largecups:
        temp = cv2.imread(str(pathlarge+"/"+i))
    elif i in mediumcups:
        temp = cv2.imread(str(pathmedium+"/"+i))
    elif i in smallcups:
        temp = cv2.imread(str(pathsmall+"/"+i))
    #print i
    
    KeyPointsOut = findKeyPointsDist(img,temp,maxdist)
    KeyPointsTotalList += KeyPointsOut[0]
    DistsTotalList += KeyPointsOut[1]
    
indices = range(len(DistsTotalList))
indices.sort(key=lambda i: DistsTotalList[i])
DistsTotalList = [DistsTotalList[i] for i in indices]
KeyPointsTotalList = [KeyPointsTotalList[i] for i in indices]

drawImageMappedPoints(img, KeyPointsTotalList)

for i in xrange(depth.shape[0]):
    for j in xrange(depth.shape[1]):
        if depth[i,j] > 0:
            img[i,j] = [0,0,0]

cv2.imshow('image',img)
cv2.waitKey(0)
cv2.destroyAllWindows()
