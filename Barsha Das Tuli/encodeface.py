from imutils import paths
import face_recognition
import argparse
import pickle
import tensorflow
import os

ap = argparse.ArgumentParser()
ap.add_argument("-i", "--dataset", required=True,
	help="path to input directory of faces + images")

ap.add_argument("-d", "--detection-method", type=str, default="cnn",
	help="face detection model to use: 'cnn'")
args = vars(ap.parse_args())

imagePaths = list(paths.list_images(args["dataset"]))

knownEncodings = []
knownNames = []

for (i, imagePath) in enumerate(imagePaths):

	print("[INFO] processing image {}/{}".format(i + 1,
		len(imagePaths)))
	name = imagePath.split(os.path.sep)[-2]

	
	boxes = face_recognition.face_locations(rgb,
		model=args["detect_method"])

	encodings = face_recognition.face_encodings(rgb, boxes)

	for encoding in encodings:

		knownEncodings.append(encoding)
		knownNames.append(name)

print("[INFO] serializing encodings...")
data = {"encodings": knownEncodings, "names": knownNames}
f = open(args["encodings"], "wb")
f.write(pickle.dumps(data))
f.close()
