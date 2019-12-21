import cv2
import math
import matplotlib.pyplot as plt
%matplotlib inline
import pandas as pd
from keras.preprocessing import image 
import numpy as np 
from skimage.transform import resize
from keras.utils import np_utils
count = 0
videoFile = "Tom and jerry.mp4"
cap = cv2.VideoCapture(videoFile)   # capturing the video from the given path
frameRate = cap.get(5) #frame rate
x=1
while(cap.isOpened()):
    frameId = cap.get(1) #current frame number
    ret, frame = cap.read()
    if (ret != True):
        break
    if (frameId % math.floor(frameRate) == 0):
        filename ="frame%d.jpg" % count;count+=1
        cv2.imwrite(filename, frame)
cap.release()
print ("Done!")
img = plt.imread('frame0.jpg')   
plt.imshow(img)
data = pd.read_csv('mapping.csv')
data.head()  
X = [ ] 
for img_name in data.Image_ID:
    img = plt.imread('' + img_name)
X.append(img) 
X = np.array(X)
