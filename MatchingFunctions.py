def findKeyPoints(img, template, maxdist=200):
    import cv2
    import numpy as np
    import itertools
    import sys

    detector = cv2.FeatureDetector_create("FAST")
    descriptor = cv2.DescriptorExtractor_create("SIFT")

    skp = detector.detect(img)
    skp, sd = descriptor.compute(img, skp)

    tkp = detector.detect(template)
    tkp, td = descriptor.compute(template, tkp)

    flann_params = dict(algorithm=1, trees=4)
    
    flann = cv2.flann_Index(sd, flann_params)
    idx, dist = flann.knnSearch(td, 1, params={})
    del flann

    dist = dist[:,0]/2500.0
    dist = dist.reshape(-1,).tolist()
    idx = idx.reshape(-1).tolist()
    indices = range(len(dist))
    indices.sort(key=lambda i: dist[i])
    dist = [dist[i] for i in indices]
    idx = [idx[i] for i in indices]
    skp_final = []
    for i, dis in itertools.izip(idx, dist):
        if dis < maxdist:
            skp_final.append(skp[i])

    flann = cv2.flann_Index(td, flann_params)
    idx, dist = flann.knnSearch(sd, 1, params={})
    del flann

    dist = dist[:,0]/2500.0
    dist = dist.reshape(-1,).tolist()
    idx = idx.reshape(-1).tolist()
    indices = range(len(dist))
    indices.sort(key=lambda i: dist[i])
    dist = [dist[i] for i in indices]
    idx = [idx[i] for i in indices]
    tkp_final = []
    for i, dis in itertools.izip(idx, dist):
        if dis < maxdist:
            tkp_final.append(tkp[i])

    return skp_final, tkp_final

def findKeyPointsDist(img, template, maxdist=200):
    import cv2
    import numpy as np
    import itertools
    import sys

    detector = cv2.FeatureDetector_create("FAST")
    descriptor = cv2.DescriptorExtractor_create("SIFT")

    skp = detector.detect(img)
    skp, sd = descriptor.compute(img, skp)

    tkp = detector.detect(template)
    tkp, td = descriptor.compute(template, tkp)

    flann_params = dict(algorithm=1, trees=4)
    
    flann = cv2.flann_Index(sd, flann_params)
    idx, dist = flann.knnSearch(td, 1, params={})
    del flann

    dist = dist[:,0]/2500.0
    dist = dist.reshape(-1,).tolist()
    idx = idx.reshape(-1).tolist()
    indices = range(len(dist))
    indices.sort(key=lambda i: dist[i])
    dist = [dist[i] for i in indices]
    idx = [idx[i] for i in indices]
    skp_final = []
    skp_final_dist = []
    for i, dis in itertools.izip(idx, dist):
        if dis <= maxdist:
            skp_final.append(skp[i])
            skp_final_dist.append(dis)

    return skp_final, skp_final_dist


def drawKeyPoints(img, template, skp, tkp, num=-1):
    import cv2
    import numpy as np
    import itertools
    import sys

    h1, w1 = img.shape[:2]
    h2, w2 = template.shape[:2]
    nWidth = w1+w2
    nHeight = max(h1, h2)
    hdif = (h1-h2)/2
    newimg = np.zeros((nHeight, nWidth, 3), np.uint8)
    newimg[hdif:hdif+h2, :w2] = template
    newimg[:h1, w2:w1+w2] = img

    maxlen = min(len(skp), len(tkp))
    if num < 0 or num > maxlen:
        num = maxlen
    for i in range(num):
        pt_a = (int(tkp[i].pt[0]), int(tkp[i].pt[1]+hdif))
        pt_b = (int(skp[i].pt[0]+w2), int(skp[i].pt[1]))
        cv2.line(newimg, pt_a, pt_b, (255, 0, 0))
    return newimg

def drawImageMappedPoints(img, skptotal, num=-1):
    import cv2
    import numpy as np
    import itertools
    import sys

    maxlen = len(skptotal)
    if num < 0 or num > maxlen:
        num = maxlen
    for i in range(num):
        pt_b = (int(skptotal[i].pt[0]), int(skptotal[i].pt[1]))
        cv2.circle(img, pt_b, 3, (255, 0, 0))
    cv2.imshow("image", img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

def saveImageMappedPoints(img, skptotal, ImageNo, num=-1):
    import cv2
    import numpy as np
    import itertools
    import sys

    maxlen = len(skptotal)
    if num < 0 or num > maxlen:
        num = maxlen
    for i in range(num):
        pt_b = (int(skptotal[i].pt[0]), int(skptotal[i].pt[1]))
        cv2.circle(img, pt_b, 3, (255, 0, 0))
    cv2.imwrite('ProcessedImages/Processed'+str(ImageNo)+'.jpg', img)


def match(img, temp, dist = 200, num = -1):
    import cv2
    import numpy as np
    import itertools
    import sys

    #img = cv2.imread(img)
    #temp = cv2.imread(template)
    
    skp, tkp = findKeyPoints(img, temp, dist)
    newimg = drawKeyPoints(img, temp, skp, tkp, num)
    cv2.imshow("image", newimg)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


def MatchAll(ImageNo, save, maxdist=200):
    from os.path import isfile, join
    from os import listdir
    import cv2
    import numpy as np
    import itertools
    import sys
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

    img = cv2.imread(str(pathtest+"/"+testimages[ImageNo]))

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
    img1 = img
    if save == 1:
        saveImageMappedPoints(img1, KeyPointsTotalList, ImageNo)
    return KeyPointsTotalList, DistsTotalList, img

def MatchAllCapture(save, maxdist=200):
    from os.path import isfile, join
    import freenect
    from os import listdir
    import cv2
    import numpy as np
    import itertools
    import sys
    #Clear all cv windows
    #cv2.destroyAllWindows()

    #Prepare a list of different training images
    pathlarge = "TrainingImages/LargeCup/"
    pathmedium = "TrainingImages/MediumCup/"
    pathsmall = "TrainingImages/SmallCup/"
    pathtest = "TestImages"

    largecups = [ f for f in listdir(pathlarge) if isfile(join(pathlarge,f)) and f[0]<>"."]
    mediumcups = [ f for f in listdir(pathmedium) if isfile(join(pathmedium,f)) and f[0]<>"."]
    smallcups = [ f for f in listdir(pathsmall) if isfile(join(pathsmall,f)) and f[0]<>"."]
    testimages = [ f for f in listdir(pathtest) if isfile(join(pathtest,f)) and f[0]<>"."]

    img, timestamp = freenect.sync_get_video()
    depth, timestamp = freenect.sync_get_depth(format=freenect.DEPTH_REGISTERED)

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
    img1 = img
    if save == 1:
        saveImageMappedPoints(img1, KeyPointsTotalList, ImageNo)
        
    return KeyPointsTotalList, DistsTotalList, img, depth

    
if __name__== '__main__':
    MatchAllCapture(0)
    

