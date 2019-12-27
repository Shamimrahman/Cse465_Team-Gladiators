
import pickle
import cv2
import os


ap = argparse.ArgumentParser()
ap.add_argument("-i", "--dataset", required=True,
help="path to input directory of faces + images")
help="face detection model to use: either `hog` or `cnn`")
args = vars(ap.parse_args())

imagePaths = list(paths.list_images(args["dataset"]))

knownNames = []


for (i, imagePath) in enumerate(imagePaths):

	print("[INFO] processing image {}/{}".format(i + 1,
		len(imagePaths)))
image = cv2.imread(imagePath)
	
	boxes = face_recognition.face_locations(rgb,
		model=args["detection_method"])

	encodings = face_recognition.face_encodings(rgb, boxes)

data = {"encodings": knownEncodings, "names": knownNames}
  f = open(args["encodings"], "wb")
  f.write(pickle.dumps(data))
