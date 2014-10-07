import freenect
import cv2
import cv2.cv as cv
import numpy as np
import frame_convert

cv2.destroyAllWindows()
i = 0
while i < 6:
    img = freenect.sync_get_video()[0]
    cv2.imshow('image',img)
    cv2.imwrite('TrainingImages/AxonCups/Glass/Glass'+str(i)+'.jpg', img)
    cv2.waitKey(0)
    i+=1

cv2.waitKey(0)
cv2.destroyAllWindows()


    


