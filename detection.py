# utilisation de l'outils de YOLO
# python detection.py --objet objet1/objet2/objet3 --yolo yolo-coco

# import the necessary packages
from google_images_download import google_images_download
import numpy as np
import argparse
from imutils import paths
import time
import cv2 
import os

response = google_images_download.googleimagesdownload()   
#class instantiation

def traitement(paths):
	
	valObjet0 = 0	
	valObjet1 = 0	
	valObjet2 = 0

	image = cv2.imread(paths)
	(H, W) = image.shape[:2]

	ln = net.getLayerNames()
	ln = [ln[i[0] - 1] for i in net.getUnconnectedOutLayers()]

	blob = cv2.dnn.blobFromImage(image, 1 / 255.0, (416, 416),
	swapRB=True, crop=False)
	net.setInput(blob)
	start = time.time()
	layerOutputs = net.forward(ln)
	end = time.time()

	print("[INFO] YOLO took {:.6f} seconds".format(end - start))

	boxes = []
	confidences = []
	classIDs = []

	for output in layerOutputs:
		for detection in output:

			scores = detection[5:]
			classID = np.argmax(scores)
			confidence = scores[classID]

			if confidence > args["confidence"]:
				
				box = detection[0:4] * np.array([W, H, W, H])
				(centerX, centerY, width, height) = 					box.astype("int")
				
				x = int(centerX - (width / 2))
				y = int(centerY - (height / 2))

			
				boxes.append([x, y, int(width), int(height)])
				confidences.append(float(confidence))
				classIDs.append(classID)

	idxs = cv2.dnn.NMSBoxes(boxes, confidences, args["confidence"],
	args["threshold"])

	if len(idxs) > 0:

		for i in idxs.flatten(): 

			(x, y) = (boxes[i][0], boxes[i][1])
			(w, h) = (boxes[i][2], boxes[i][3])
		
		
			color = [int(c) for c in COLORS[classIDs[i]]]
			cv2.rectangle(image, (x, y), (x + w, y + h), color, 2)
			text = "{}: {:.4f}".format(LABELS[classIDs[i]], confidences[i])
			text_objet = text.split(':',1)
			if text_objet[0] == objet0:
				valObjet0 = 1
				img0 = image[boxes[i][1]:boxes[i][1]+boxes[i][3], boxes[i][0]:boxes[i][0]+boxes[i][2]]
				img0 = cv2.resize(img0, (300,300), interpolation = cv2.INTER_AREA)
				cv2.imwrite("search/"+objet0 +".jpg", img0)
			elif text_objet[0] == objet1:
				valObjet1 = 1
				img1 = image[boxes[i][1]:boxes[i][1]+boxes[i][3], boxes[i][0]:boxes[i][0]+boxes[i][2]]
				img1 = cv2.resize(img1, (300,300), interpolation = cv2.INTER_AREA)
				cv2.imwrite("search/"+objet1 +".jpg", img1)
			elif text_objet[0] == objet2:
				valObjet2 = 1
				img2 = image[boxes[i][1]:boxes[i][1]+boxes[i][3], boxes[i][0]:boxes[i][0]+boxes[i][2]]
				img2 = cv2.resize(img2, (300,300), interpolation = cv2.INTER_AREA)
				cv2.imwrite("search/"+objet2 +".jpg", img2)
			cv2.putText(image, text, (x, y - 5), cv2.FONT_HERSHEY_SIMPLEX,0.5, color, 2)
	image = cv2.resize(image, (800,600), interpolation = cv2.INTER_AREA)
	cv2.imshow("Image", image)

	cv2.waitKey(1)
	return (valObjet0 , valObjet1 ,valObjet2)

ap = argparse.ArgumentParser()
ap.add_argument("-o", "--objet", required=True,
	help="path to input objet")
ap.add_argument("-y", "--yolo", required=True,
	help="base path to YOLO directory")
ap.add_argument("-c", "--confidence", type=float, default=0.5,
	help="minimum probability to filter weak detections")
ap.add_argument("-t", "--threshold", type=float, default=0.3,
	help="threshold when applyong non-maxima suppression")
args = vars(ap.parse_args())

labelsPath = os.path.sep.join([args["yolo"], "coco.names"])
LABELS = open(labelsPath).read().strip().split("\n")

np.random.seed(42)
COLORS = np.random.randint(0, 255, size=(len(LABELS), 3),
	dtype="uint8")

weightsPath = os.path.sep.join([args["yolo"], "yolov3.weights"])
configPath = os.path.sep.join([args["yolo"], "yolov3.cfg"])

print("[INFO] loading YOLO from disk...")
net = cv2.dnn.readNetFromDarknet(configPath, weightsPath)


argus = args["objet"].split('/',2)
objet0 = argus[0]
objet1 = argus[1]
objet2 = argus[2]
mot_cle = objet0 + ' '+ objet1 +' ' + objet2
arguments = {"keywords": mot_cle,"limit":1,"print_urls":True}
paths = response.download(arguments) 
paths = paths[0][mot_cle]
paths = ''.join(paths)

(valObjet0,valObjet1,valObjet2) = traitement(paths)
print valObjet0 ,"\n", valObjet1 ,"\n", valObjet2 ,"\n"
if valObjet0 == 0:
	arguments = {"keywords": objet0,"limit":1,"print_urls":True}
	paths = response.download(arguments) 
	paths0 = paths[0][objet0]
	paths0 = ''.join(paths0)
	traitement(paths0)

if valObjet1 == 0:
	arguments = {"keywords": objet1,"limit":1,"print_urls":True}
	paths = response.download(arguments) 
	paths1 = paths[0][objet1]
	paths1 = ''.join(paths1)
	traitement(paths1)

if valObjet2 == 0:
	arguments = {"keywords": objet2,"limit":1,"print_urls":True}
	paths = response.download(arguments) 
	paths2 = paths[0][objet2]
	paths2 = ''.join(paths2)
	traitement(paths2)

cv2.waitKey(0)
