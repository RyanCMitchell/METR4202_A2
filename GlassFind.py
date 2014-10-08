import cv2
import freenect
import numpy as np

img, timestamp = freenect.sync_get_video()
depth, timestamp = freenect.sync_get_depth(format=freenect.DEPTH_REGISTERED)
cv2.waitKey(1000)

def GlassFind(img0,depth0):
    import cv2
    import freenect
    import numpy as np
    
    img = np.copy(img0)
    depth = np.copy(depth0)
    
    #Blur and threshold depth
    depth = cv2.blur(depth,(10,10))
    mask = np.zeros(img.shape,dtype=np.uint8)
    for i in xrange(img.shape[0]):
        for j in xrange(img.shape[1]):
            if depth[i,j] == 0 and i>50 and j>30 and j<595:
                mask[i,j] = 255

    #Edge detect and find contours
    thresh1 = 200
    thresh2 = 300
    edges = cv2.Canny(mask,thresh1,thresh2)
    contours,hierarchy = cv2.findContours(edges,cv2.RETR_TREE,cv2.CHAIN_APPROX_SIMPLE)

    #Exclude contours with inappropriate depth
    GlassCenters = []
    GlassContours = []
    GlassBox = []
    for cnt in contours:
        if len(cnt) > 10:
            cnt = cv2.convexHull(cnt)
            area = cv2.contourArea(cnt)
            if 2000<area<10000:
                M = cv2.moments(cnt)
                if M['m00'] <> 0:
                    cx = int(M['m10']/M['m00'])
                    cy = int(M['m01']/M['m00'])
                    x,y,w,h = cv2.boundingRect(cnt)
                    box = [x,y,w,h]
                    if h>w:
                        GlassCenters.append([cx,cy])
                        GlassContours.append(cnt)
                        GlassBox.append(box)
                                        
    return GlassCenters, GlassContours, GlassBox
