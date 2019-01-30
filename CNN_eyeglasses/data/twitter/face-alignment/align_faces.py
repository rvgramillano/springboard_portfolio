# import the necessary packages
from imutils.face_utils import FaceAligner
from imutils.face_utils import rect_to_bb
import argparse
import imutils
import dlib
import cv2
import os
import matplotlib.pyplot as plt

DIR = '/Users/rvg/Documents/springboard_ds/springboard_portfolio/CNN_eyeglasses/data/twitter/'

# initialize dlib's face detector (HOG-based) and then create
# the facial landmark predictor and the face aligner
detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor(DIR + 'face-alignment/shape_predictor_68_face_landmarks.dat')
aspect_ratio = 178. / 218.
desiredHeight = 300
fa = FaceAligner(predictor, desiredFaceHeight=desiredHeight, desiredFaceWidth=int(aspect_ratio * desiredHeight))

for image_name in os.listdir(DIR + 'profile_pics/'):
	if image_name.endswith('.jpg'):
		print image_name
		# load the input image, resize it, and convert it to grayscale
		image = cv2.imread(DIR + 'profile_pics/' + image_name)
		image = imutils.resize(image, width=800)
		gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
		
		# find face in image with no downsampling
		rects = detector(gray, 0)
		
		# loop over the face detections, align face, and save image
		for rect in rects:
			faceAligned = fa.align(image, gray, rect)

			cv2.imwrite(DIR + 'profile_pics/aligned/' + image_name, faceAligned)