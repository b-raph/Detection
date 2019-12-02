# python combi.py --images search

from imutils import paths
import numpy as np
import argparse
import imutils
import cv2

ap = argparse.ArgumentParser()
ap.add_argument("-i", "--images", type=str, required=True,
	help="path of images")
args = vars(ap.parse_args())

imagePaths = sorted(list(paths.list_images(args["images"])))
images = []

for imagePath in imagePaths:
	image = cv2.imread(imagePath)
	images.append(image)

numpy_vertical = np.hstack((images[0],images[1],images[2]))
	
cv2.imwrite('combi.jpg', numpy_vertical)
