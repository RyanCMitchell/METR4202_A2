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

    
if __name__== '__main__':
    match('Test1.jpg','cupLarge1.jpg')
    

