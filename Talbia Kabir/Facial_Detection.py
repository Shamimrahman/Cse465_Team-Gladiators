import os
import cv2
import numpy as np
import sys
import datetime

if len(sys.argv) != 3:
	print("usage: ./face_detector.py <user_name> <save_face flag>")
	sys.exit(1)

USER = sys.argv[1]
SAVE_FACE = int(sys.argv[2])
FACES_FOLDER = 'Faces' # folder where to store the detected faces

if SAVE_FACE and not os.path.exists(FACES_FOLDER + '/' + USER):
	os.makedirs(FACES_FOLDER + '/' + USER)

print('Detecting faces for user named', USER)

def process_frame(frame, face_classifier):
	gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
	faces = face_classifier.detectMultiScale(gray, 1.6, 8)

	return faces

def display_result(img, faces, save_results):
	for (x,y,w,h) in faces:
		cv2.rectangle(img, (x,y), (x+w,y+h), (255,0,0), 2)
		if save_results and len(faces)==1:
			t_index = datetime.datetime.now()
			t_index = "%s_%s_%s_%s" % (t_index.hour, t_index.minute, t_index.second, str(t_index.microsecond)[:2])
			mask = img[y:y+h,x:x+w,:]
			cv2.imwrite(FACES_FOLDER + '/' + USER + '/' + USER + '_' + t_index + '.png', mask)
	cv2.imshow('Face detector', img)
	cv2.waitKey(1)
