from os.path import isfile, join
from os import listdir
import cv2
import numpy as np
import itertools
import sys
from MatchingFunctions import findKeyPoints, drawKeyPoints, match, findKeyPointsDist, drawImageMappedPoints

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

img = cv2.imread(str(pathtest+"/"+testimages[7]))
maxdist = 200  # 200 is default

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



