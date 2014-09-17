from os.path import isfile, join
from os import listdir

#Prepare a list of test images
mypath = "Test/"
images = [ f for f in listdir(mypath) if isfile(join(mypath,f)) ]
