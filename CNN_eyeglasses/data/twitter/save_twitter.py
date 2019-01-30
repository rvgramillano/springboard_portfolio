import numpy as np
import random
import os
import matplotlib.pyplot as plt
import PIL.ImageOps
from PIL import Image


DIR = '/Users/rvg/Documents/springboard_ds/springboard_portfolio/CNN_eyeglasses/data/twitter/'
dim = 28
n_images = len(os.listdir(DIR + 'profile_pics/aligned/'))
# set up structure for storing image data
imageData = np.zeros((n_images,dim,dim), dtype=np.uint8)
# manually input
all_labels = [0, 1, 0, 1, 1, 0, 1, 0, 1, 1, 1, 1, 1, 1, 1, 0, 1]
# set up array for image file names
all_image_names = []

# go through dataset, align image, convert image to grayscale, resize and save in separate data structure
i=0
for image_name in os.listdir(DIR + 'profile_pics/aligned/'):
	print(image_name)
	all_image_names.append(image_name)
	image = Image.open(DIR + 'profile_pics/aligned/' + image_name)
	image = image.convert('L') #convert into grayscale
	image = image.resize((dim, dim), Image.BICUBIC) #resize into 28x28 dimension
	img_array = np.asarray(image)
	imageData[i,:,:] = img_array
	i += 1
	
# save twitter data with labels and images
all_labels = np.asarray(all_labels)
labels = np.zeros((len(all_labels),1), dtype=np.int)
labels[:,0] = all_labels
all_image_names = np.asarray(all_image_names)
print(all_image_names.shape)
print(all_labels.shape)
print(imageData.shape)
np.savez(DIR + "twitter_test_imgs.npz", imageNames=all_image_names, labels=labels, imageData=imageData)
