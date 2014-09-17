import freenect
import cv2
import cv2.cv as cv
import numpy as np
import frame_convert


img = freenect.sync_get_video()[0]
cv2.imshow('image',img)
cv2.imwrite('Test8.jpg', img)
cv2.waitKey(0)
cv2.destroyAllWindows()

