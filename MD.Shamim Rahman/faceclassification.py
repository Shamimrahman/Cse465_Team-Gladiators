import sys
if len(sys.argv) != 2:
	print('usage: ./face_classification.py <# training epochs>')
	sys.exit(1)
import os
import numpy as np
import cv2
import glob
from random import shuffle
import keras
from keras.models import Model
from keras.layers import Input, Dense, Conv2D, MaxPooling2D, Dropout, Flatten 
from keras.optimizers import Adam
from sklearn.metrics import confusion_matrix
from keras.models import model_from_json

def load_imgs(path):

	usr = [item for item in os.listdir(path) if os.path.isdir(os.path.join(path, item))]
	DEFAULT_SIZE = (30,30)
	imgs = []
	for i in usr:
		imgs.append([cv2.resize(cv2.imread(j, cv2.IMREAD_GRAYSCALE), DEFAULT_SIZE, interpolation=cv2.INTER_CUBIC) for j in glob.glob(PATH + i + '/*.png')])

	print('Number of users:', len(imgs))
	for i,j in enumerate(imgs):
		print('\tSamples for user %d: %d' % (i+1,len(j)))

	return imgs, usr

def create_labels(imgs):
	y = []
	for i in range(len(imgs)):
		y.append(i * np.ones(len(imgs[i])))
	y = np.hstack(y)

	return y

def reshape_for_keras(data):
	temp = np.vstack(data)
	
	return temp.reshape(temp.shape + (1,))

def train_test_split(data, test_size):
	rnd_index = np.arange(data.shape[0])
	shuffle(rnd_index)
	last_test_index = round(test_size * len(rnd_index))
	test_index =  rnd_index[:last_test_index]
	train_index = rnd_index[last_test_index:]

	return train_index, test_index
