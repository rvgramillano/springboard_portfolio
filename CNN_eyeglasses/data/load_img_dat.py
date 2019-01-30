import PIL.ImageOps
from PIL import Image
from sklearn import preprocessing
import numpy as np
import tensorflow as tf
import random
from imutils.face_utils import FaceAligner
from imutils.face_utils import rect_to_bb
import imutils
import dlib
import cv2
import os
import matplotlib.pyplot as plt

def align_face(img_path, name):
	'''
	This function taken in an input image path of a face and attempts to align the image.
	Returns a tuple of the image and a response designating whether or not the face was successfully found and aligned.
	'''
	img = cv2.imread(img_path)
	img = imutils.resize(img, width=800)
	gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
	rects = detector(gray, 0)
	for rect in rects:
		faceAligned = fa.align(img, gray, rect)
		cv2.imwrite(DIR + '%s.jpg'%name, faceAligned)
	try:
		return (Image.fromarray(faceAligned), 'N')
	except UnboundLocalError:
		return (Image.open(imagesPath),  'Y')


imagespath = '/Users/rvg/Documents/springboard_ds/springboard_portfolio/CNN_eyeglasses/data/CelebA/Img/img_align_celeba/'
filename = '/Users/rvg/Documents/springboard_ds/springboard_portfolio/CNN_eyeglasses/data/CelebA/Anno/list_attr_celeba.txt'
DIR = '/Users/rvg/Documents/springboard_ds/springboard_portfolio/CNN_eyeglasses/data/CelebA/Img/aligned_dataset/'

# initialize dlib's face detector (HOG-based) and then create
# the facial landmark predictor and the face aligner
global detector
global predictor
global fa
n_unaligned = 0
detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor('/Users/rvg/Documents/springboard_ds/springboard_portfolio/CNN_eyeglasses/data/twitter/face-alignment/shape_predictor_68_face_landmarks.dat')
aspect_ratio = 178. / 218.
desiredHeight = 300
fa = FaceAligner(predictor, desiredFaceHeight=desiredHeight, desiredFaceWidth=int(aspect_ratio * desiredHeight))


linenum = 0
eyeglass_labels = []
eyeglass_images = []
non_eyeglass_labels = []
non_eyeglass_images = []
# get all 13193 eyeglasses images from 202599 celeb dataset
for line in open(filename): 
	linenum += 1
	if(linenum==1):
		length = int(line)
	elif(linenum>2):
		lineData = line.split(" ")
		lineData = list(filter(None, lineData))
		lineData[-1] = lineData[-1].strip()
		# the index 16 designates eyeglasses
		if(int(lineData[16])==1):
			eyeglass_labels.append(1)
			eyeglass_images.append(lineData[0])
print(len(eyeglass_labels))
print(len(eyeglass_images))
eyeglass_length = len(eyeglass_images)
noneyeglass_length = 80000 - len(eyeglass_images)
# get the remaining non-eyeglass images such that the sum of eyeglasses + noneyeglasses images = 80000--the size of our data set
linenum = 0
for line in open(filename): 
	linenum = linenum+1
	if(linenum==1):
		length = int(line)
	elif(linenum>2):
		lineData = line.split(" ")
		lineData = list(filter(None, lineData))
		lineData[-1] = lineData[-1].strip()
		# not an eyeglasses image
		if(int(lineData[16])==-1):
			# once we reached the desired length, stop
			if(len(non_eyeglass_labels) == noneyeglass_length):
				break
			non_eyeglass_labels.append(0)
			non_eyeglass_images.append(lineData[0])
print(len(non_eyeglass_labels))
print(len(non_eyeglass_images))
# concatenating all eyeglasses and non-eyeglass images and labels
all_labels = eyeglass_labels + non_eyeglass_labels
all_image_names = eyeglass_images + non_eyeglass_images
print(len(all_labels))
print(len(all_image_names))
# Random shuffling of data
temp = list(zip(all_labels, all_image_names))
random.shuffle(temp)
all_labels, all_image_names = zip(*temp)
# set up structure for storing image data
i=0
# go through dataset, align image, convert image to grayscale, resize and save in separate data structure
imageData = np.zeros((80000,28,28), dtype=np.uint8)
unaligned_inds = []
for images in all_image_names:
	imagesPath = imagespath+images#full image path
	print(i, imagesPath)
	image, response = align_face(imagesPath, i)
	# a 'Y' response indicates that the face could not be found in the image. these images are skipped and their indices recorded to remove
	# from the data structure (imageData) later
	if response=='Y':
		n_unaligned+=1
		unaligned_inds.append(i)
	image = image.convert('L') #convert into grayscale
	image = image.resize((28, 28), Image.BICUBIC) #resize into 28x28 dimension
	img_array = np.asarray(image)
	imageData[i,:,:] = img_array
	i+=1

# remove indices where face could not be found--removed 3.85% of samples, leaving us with a total of 76914 images
imdat = np.delete(imageData, unaligned_inds, axis=0)
labels_all = np.delete(np.asarray(all_labels), unaligned_inds)
all_labels = np.asarray(all_labels)
labels = np.zeros((len(labels_all),1), dtype=np.int)
labels[:,0] = labels_all
all_image_names1 = np.delete(np.asarray(all_image_names), unaligned_inds)
print(all_image_names1.shape)
print(labels_all.shape)
print(imdat.shape)
# save data structure
np.savez("/Users/rvg/Documents/springboard_ds/springboard_portfolio/CNN_eyeglasses/data/CelebA/CelebA_70K_align.npz", imageNames=all_image_names1, labels=labels, imageData=imdat)

