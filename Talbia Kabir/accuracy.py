
import math   
import matplotlib.pyplot as plt    
matplotlib inline
import pandas as pd
from keras.preprocessing import image  
import numpy as np    
from keras.utils import np_utils
from skimage.transform import resize

count = 0
videoFile = "Tom and jerry.mp4"
cap = cv2.VideoCapture(videoFile)   
frameRate = cap.get(5) 
x=1


while(cap.isOpened()):
    frameId = cap.get(1) 
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

y = data.Class
dummy_y = np_utils.to_categorical(y)    
image = []

for i in range(0,X.shape[0]):
    a = resize(X[i], preserve_range=True, output_shape=(224,224)).astype(int)    
    image.append(a)
    
X = np.array(image)

from keras.applications.vgg16 import preprocess_input
X = preprocess_input(X, mode='tf')      
from sklearn.model_selection import train_test_split
X_train, X_valid, y_train, y_valid = train_test_split(X, dummy_y, test_size=0.3, random_state=42)  

from keras.models import Sequential
from keras.applications.vgg16 import VGG16
from keras.layers import Dense, InputLayer, Dropout

base_model = VGG16(weights='imagenet', include_top=False, input_shape=(224, 224, 3)) 

X_train = base_model.predict(X_train)
X_valid = base_model.predict(X_valid)
X_train.shape, X_valid.shape
X_train = X_train.reshape(208, 7*7*512)     
X_valid = X_valid.reshape(90, 7*7*512)

train = X_train/X_train.max()     
X_valid = X_valid/X_train.max()

model = Sequential()
model.add(InputLayer((7*7*512,)))    
model.add(Dense(units=1024, activation='sigmoid')) 

model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
model.fit(train, y_train, epochs=100, validation_data=(X_valid, y_valid))

from keras.models import Sequential
from keras.applications.vgg16 import VGG16
from keras.layers import Dense, InputLayer, Dropout

base_model = VGG16(weights='imagenet', include_top=False, input_shape=(224, 224, 3))

X_train = base_model.predict(X_train)
X_valid = base_model.predict(X_valid)
test_image = base_model.predict(test_image)
X_train = X_train.reshape(208, 7*7*512)
X_valid = X_valid.reshape(90, 7*7*512)
test_image = test_image.reshape(186, 7*7*512)
train = X_train/X_train.max()
X_valid = X_valid/X_train.max()

test_image = test_image/test_image.max()

model = Sequential()
model.add(InputLayer((7*7*512,)))   
model.add(Dense(units=1024, activation='sigmoid'))   
model.add(Dropout(0.5))      # adding dropout
model.add(Dense(units=512, activation='sigmoid'))   
model.add(Dropout(0.5))      # adding dropout
model.add(Dense(units=256, activation='sigmoid'))   
model.add(Dropout(0.5))      # adding dropout
model.add(Dense(3, activation='softmax'))          
model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

from sklearn.utils.class_weight import compute_class_weight, compute_sample_weight
class_weights = compute_class_weight('balanced',np.unique(data.Class), data.Class)  
from keras.callbacks import ModelCheckpoint
filepath="weights.best.hdf5"
checkpoint = ModelCheckpoint(filepath, monitor='val_loss', verbose=1, save_best_only=True, mode='min')
callbacks_list = [checkpoint]

model.load_weights("weights.best.hdf5")
model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

scores = model.evaluate(test_image, test_y)
print("%s: %.2f%%" % (model.metrics_names[1], scores[1]*100)
