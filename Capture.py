import freenect
import cv2
import cv2.cv as cv
import numpy as np
import frame_convert

'''
A piece of code for connecting to the Kinect and taking images upon a key
press.'''

cv2.destroyAllWindows()
i = 0
amount = 6

while i < amount:
    img = freenect.sync_get_video()[0] #get image from Kinect
    cv2.imshow('image',img) # show image
    # save image in desired folder
    cv2.imwrite('TrainingImages/AxonCups/Glass/Glass'+str(i)+'.jpg', img)
    cv2.waitKey(0) # wait for keypress
    i+=1

cv2.waitKey(0)
cv2.destroyAllWindows()


    


