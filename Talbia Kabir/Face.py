
import face_recognition
import argparse
import pickle
import cv2

ap = argparse.ArgumentParser()
ap.add_argument("-e", "--encodings", required=True,
	help="path to serialized db of facial encodings")
ap.add_argument("-i", "--image", required=True,
	help="path to input image")
ap.add_argument("-d", "--detection-method", type=str, default="cnn",
	help="face detection model to use: either `hog` or `cnn`")


image = cv2.imread(args["image"])
rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

boxes = face_recognition.face_locations(rgb,model=args["detection_method"])
encodings = face_recognition.face_encodings(rgb,boxes)
names = []

for encoding in encodings:
	
  matches = face_recognition.compare_faces(data["encodings"],encoding)
	name = "Unknown"

for ((top, right, bottom, left), name) in zip(boxes, names):

  cv2.rectangle(image, (left, top), (right, bottom), (0, 255, 0), 2)
  y = top - 15 if top - 15 > 15 else top + 15
  cv2.putText(image, name, (left, y), cv2.FONT_HERSHEY_SIMPLEX,0.75, (0, 255, 0), 2)

#show the output image
cv2.imshow("Image", image)
cv2.waitKey(0)
