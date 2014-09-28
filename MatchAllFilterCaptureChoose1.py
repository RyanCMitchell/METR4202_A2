from os.path import isfile, join
from os import listdir
import numpy as np
import freenect, itertools, sys, time, cv2
from MatchingFunctions import findKeyPoints, drawKeyPoints, match, findKeyPointsDist, drawImageMappedPoints, saveImageMappedPoints, MatchAllCapture, Cluster, fit_ellipses
from matplotlib import pyplot as plt
from math import sqrt
from convertDepth import convertToWorldCoords


def MatchAllCluster(save, maxdist=200, filtparam=2.0):
    
    PointsList, DisList, img, depth = MatchAllCapture(0,maxdist)
    
    PointsClusterList = []
    for i in xrange(len(PointsList)):
        if depth[PointsList[i].pt[1], PointsList[i].pt[0]] <> 0 and depth[PointsList[i].pt[1], PointsList[i].pt[0]] < 1000:
            PointsClusterList.append([PointsList[i].pt[0], PointsList[i].pt[1]])

    Z = np.array(PointsClusterList)

    if len(Z) < 30:
        cv2.imshow("Cups Stream", img)
        return
        
     
    # convert to np.float32
    Z = np.float32(Z)

    # Determine how many cups there are
    segregated, centers, distFromCenter, distFromCenterAve1 = Cluster(Z, 1)
    segregated, centers, distFromCenter, distFromCenterAve2 = Cluster(Z, 2)
    segregated, centers, distFromCenter, distFromCenterAve3 = Cluster(Z, 3)
    segregated, centers, distFromCenter, distFromCenterAve4 = Cluster(Z, 4)
    segregated, centers, distFromCenter, distFromCenterAve5 = Cluster(Z, 5)

    distFromCenterAveList = [(sum(distFromCenterAve1)/len(distFromCenterAve1))*1.0,
    (sum(distFromCenterAve2)/len(distFromCenterAve2))*2.0,
    (sum(distFromCenterAve3)/len(distFromCenterAve3))*3.0,
    (sum(distFromCenterAve4)/len(distFromCenterAve4))*4.0,
    (sum(distFromCenterAve5)/len(distFromCenterAve5))*5.0]

    groups = distFromCenterAveList.index(min(distFromCenterAveList))+1

    segregated, centers, distFromCenter, distFromCenterAve = Cluster(Z, groups)

    #Create List for reduced points
    segregatedF = []
    for i in xrange(groups):
        segregatedF.append([])

    #Remove points which are not close to centroid
    for j in xrange(groups):
        for i in range(len(segregated[j])):
            if distFromCenter[j][i] < filtparam*distFromCenterAve[j]:
                segregatedF[j].append(segregated[j][i])
    for j in xrange(groups):
        segregatedF[j] = np.array(segregatedF[j])

    #remove clusters that are >= 3 points or all superimposed
    for i in xrange(groups):
        if len(segregatedF[i]) <= 5 or np.isnan(np.std(segregatedF[j])):
            print "HOLY FUCKING SHIT"
            cv2.imshow("Cups Stream", img)
            return

    # Create a centriod depth list
    FinalCenters = []
    for j in xrange(groups):
        FinalCenters.append([centers[j][0],centers[j][1],depth[centers[j][1],centers[j][0]]])

    # Convert to world coordinates
    FinalCentersWC = convertToWorldCoords(FinalCenters)
    
    segregated = segregatedF
    FC = FinalCenters
    colourList=[(0, 255, 0), (255, 0, 0), (0, 0, 255), (0, 255, 255), (255, 255, 0), (255, 0, 255)]

    # Seperate the top of each cup in pixel space
    depthimg = img.copy()

    # Start Cup Classification loop
    for j in xrange(groups):
        # Choose pixel area likley to contain a cup
        w = -0.08811*FC[j][2]+103.0837
        h = -0.13216*FC[j][2]+154.6256
        cup1 = depthimg[(FC[j][1]-h):(FC[j][1]), (FC[j][0]-w):(FC[j][0]+w)]

        # Determine the bouding rectangle of the largest contour in that area
        gray = cv2.cvtColor(cup1,cv2.COLOR_BGR2GRAY)
        blur = cv2.GaussianBlur(gray,(5,5),0)
        thresh = 240
        global contours
        edges = cv2.Canny(blur,thresh,thresh*2)
        contours,hierarchy = cv2.findContours(edges,cv2.RETR_TREE,cv2.CHAIN_APPROX_SIMPLE)
        cnt = max(contours, key=len)
        x,y,w1,h1 = cv2.boundingRect(cnt)

        # Use brounding rectangle size to determine cup type
        s1 = [(FC[j][0]-w)+x,(FC[j][1]-h)+ y,FC[j][2]]
        s2 = [(FC[j][0]-w)+x+w1,(FC[j][1]-h)+ y,FC[j][2]]
        top = [s1,s2]
        topWorld = convertToWorldCoords(top)
        CupTopWidth = topWorld[1][0]-topWorld[0][0]

        if CupTopWidth > 100:
            cupType = "Not a Cup"
        elif CupTopWidth > 84.5:
            cupType = "Large"
        elif CupTopWidth > 71:
            cupType = "Medium"
        elif CupTopWidth > 50:
            cupType = "Small"
        else:
            cupType = "Not a Cup"

        FinalCentersWC[j].append(cupType)

        #Draw the top of the bounding rectangle
        cv2.line(img,(int(round(s1[0],0)),int(round(s1[1],0))),(int(round(s2[0],0)),int(round(s2[1],0))),colourList[j])
##        new_cnt = [[x[0][0],x[0][1]] for x in cnt]
##        print new_cnt
                
        cv2.drawContours(img,[cnt],0,colourList[j],2)
    
    # Draw the groups
    for j in xrange(groups):
        centerst = tuple(np.array(centers[j])+np.array([0,50]))
        cv2.putText(img,str(FinalCentersWC[j]), centerst, cv2.FONT_HERSHEY_SIMPLEX, 0.3, colourList[j])
        cv2.circle(img, centers[j], 10, colourList[j], -1)
        cv2.circle(img, centers[j], 2, (0,0,0), -1)
        for i in range(len(segregated[j])):
            pt_a = (int(segregated[j][i,0]), int(segregated[j][i,1]))
            cv2.circle(img, pt_a, 3, colourList[j])
        rpt1 = tuple(segregated[j].min(axis=0))
        rpt2 = tuple(segregated[j].max(axis=0))
        #cv2.rectangle(img, rpt1, rpt2, colourList[j])
    
    if save == 1:
        cv2.imwrite('ProcessedImages/ProcessedCluster'+str(ImageNo)+'.jpg', img)

    print FinalCenters
    print FinalCentersWC
    
    
    cv2.imshow("Cups Stream", img)

    
    
    """
    depthFindCup1 = img.copy()
    depthFindCup1.fill(0)
    dc = FinalCentersWC[0][2]
    for i in xrange(depth.shape[0]):
        for j in xrange(depth.shape[1]):
            if dc-20<depth[i,j]<dc+70:
                depthFindCup1[i,j] = [255,255,255]
    gray = cv2.cvtColor(depthFindCup1, cv2.COLOR_BGR2GRAY)
    gray = cv2.bilateralFilter(gray, 11, 17, 17)
    edged = cv2.Canny(gray, 30, 200)
    (cnts, _) = cv2.findContours(edged.copy(), cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    cv2.drawContours(img, cnts, -1, (0, 255, 0), 1)
    """
    
    
    """
    # Black out 0 depth
    for i in xrange(depth.shape[0]):
        for j in xrange(depth.shape[1]):
            if depth[i,j] == 0 or depth[i,j]>1000:
                img[i,j] = [0,0,0]

    cv2.imshow("Cups Stream", img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    """


if __name__== '__main__':
    
    
    cv2.destroyAllWindows()
    while 1<2:
        MatchAllCluster(0,80,2)
        cv2.waitKey(10)

    
    
        

    
