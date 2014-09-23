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


def homo(img, template):
    MIN_MATCH_COUNT = 10

    # Initiate SIFT detector
    sift = cv2.SIFT()

    # find the keypoints and descriptors with SIFT
    kp1, des1 = sift.detectAndCompute(template,None)
    kp2, des2 = sift.detectAndCompute(img,None)

    FLANN_INDEX_KDTREE = 0
    index_params = dict(algorithm = FLANN_INDEX_KDTREE, trees = 5)
    search_params = dict(checks = 50)

    flann = cv2.FlannBasedMatcher(index_params, search_params)

    matches = flann.knnMatch(des1,des2,k=2)

    # store all the good matches as per Lowe's ratio test.
    good = []
    for m,n in matches:
        if m.distance < 0.7*n.distance:
            good.append(m)

    if len(good)>MIN_MATCH_COUNT:
        src_pts = np.float32([ kp1[m.queryIdx].pt for m in good ]).reshape(-1,1,2)
        dst_pts = np.float32([ kp2[m.trainIdx].pt for m in good ]).reshape(-1,1,2)

        M, mask = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC,5.0)
        matchesMask = mask.ravel().tolist()

        h,w = template.shape
        pts = np.float32([ [0,0],[0,h-1],[w-1,h-1],[w-1,0] ]).reshape(-1,1,2)
        dst = cv2.perspectiveTransform(pts,M)

        # img = cv2.polylines(img,[np.int32(dst)],True,255,3, cv2.LINE_AA)
        # img = cv2.polylines(img,[np.int32(dst)],True,255,3)


    else:
        print "Not enough matches are found - %d/%d" % (len(good),MIN_MATCH_COUNT)
        matchesMask = None

    return M, kp1, kp2, des1, des2 

    
if __name__== '__main__':
    match('Test1.jpg','cupLarge1.jpg')
    

