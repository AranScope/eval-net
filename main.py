# import the necessary packages
import numpy as np
import argparse
import cv2

CLASSES = ["background", "aeroplane", "bicycle", "bird", "boat",
	"bottle", "bus", "car", "cat", "chair", "cow", "diningtable",
	"dog", "horse", "motorbike", "person", "pottedplant", "sheep",
	"sofa", "train", "tvmonitor"]

net = cv2.dnn.readNetFromCaffe("MobileNetSSD_deploy.prototxt", "MobileNetSSD_deploy.caffemodel")
min_confidence = 0.2

def get_label_confidence(image, label):

	image = cv2.resize(image, (500, 500))
	(h, w) = image.shape[:2]


	blob = cv2.dnn.blobFromImage(image, 0.007843,
		(500, 500), 127.5)

	# pass the blob through the network and obtain the detections and
	# predictions
	print("[INFO] computing object detections...")
	net.setInput(blob)
	detections = net.forward()

	# loop over the detections
	for i in np.arange(0, detections.shape[2]):
		# extract the confidence (i.e., probability) associated with the
		# prediction
		confidence = detections[0, 0, i, 2]

		# filter out weak detections by ensuring the `confidence` is
		# greater than the minimum confidence
		if confidence > min_confidence:
			idx = int(detections[0, 0, i, 1])

			if CLASSES[idx] == label:
				return confidence

	return 0


