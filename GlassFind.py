import cv2
import freenect
while 1<2:
    img, timestamp = freenect.sync_get_video()
    depth, timestamp = freenect.sync_get_depth(format=freenect.DEPTH_REGISTERED)

    #depth = cv2.blur(depth,(4,4))
    depth = cv2.morphologyEx(depth, cv2.MORPH_GRADIENT, (10,10))
    #depth = cv2.morphologyEx(depth, cv2.MORPH_OPEN, (4,4))


    for i in xrange(img.shape[0]):
        for j in xrange(img.shape[1]):
            if depth[i,j] <> 0 or depth[i,j] > 1000:
                img[i,j] = [0,0,0]

    cv2.imshow("Cups Stream", img)
    cv2.waitKey(30)

