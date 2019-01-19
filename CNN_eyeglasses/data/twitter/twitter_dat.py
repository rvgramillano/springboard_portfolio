import PIL.ImageOps
from PIL import Image
from sklearn import preprocessing
import numpy as np
import tensorflow as tf
import random
import os

DIR = '/Users/rvg/Documents/springboard_ds/springboard_portfolio/CNN_eyeglasses/'
# set up structure for storing image data
imageData = np.zeros((1,28,28), dtype=np.uint8)
all_labels = [0]
all_image_names = ['00.jpg']

# go through dataset, convert image to grayscale, resize and save in separate data structure
i=0
for image_name in os.listdir(DIR):
	print(image_path)
	if image_name.startswith('.'):
		continue
	image = Image.open(DIR+image_path)
	image = image.convert('L') #convert into grayscale
	image = image.resize((28, 28), Image.BICUBIC) #resize into 28x28 dimension
	img_array = np.asarray(image)
	imageData[i,:,:] = img_array
	i += 1
	
all_labels = np.asarray(all_labels)
labels = np.zeros((len(all_labels),1), dtype=np.int)
labels[:,0] = all_labels
all_image_names = np.asarray(all_image_names)
print(all_image_names.shape)
print(all_labels.shape)
print(imageData.shape)
np.savez(DIR + "data/twitter/twitter_test_imgs.npz", imageNames=all_image_names, labels=labels, imageData=imageData)
return (imageData, labels, all_image_names)


data = np.load(DIR + "data/twitter/twitter_test_imgs.npz")
t_labels = data['labels']
t_celebData = data['imageData']
t_imageNames = data['imageNames']

#flattening the input array and reshaping the labels as per requirement of the tnesorflow algo
twitter_images = trains_images.reshape([train_num,dim**2])
val_images = val_images.reshape([val_num,dim**2])
test_images = test_images.reshape([test_num,dim**2])
train_images_labels = train_images_labels.reshape([train_num,])
val_images_labels = val_images_labels.reshape([val_num,])
test_images_labels = test_images_labels.reshape([test_num,])

#standardizing the image data set with zero mean and unit standard deviation
trains_images = preprocessing.scale(trains_images)
val_images = preprocessing.scale(val_images)
test_images = preprocessing.scale(test_images)


# Evaluate on twitter images
twitter_input_fun = tf.estimator.inputs.numpy_input_fn(
  x={"x": twitter_data},
  y=twitter_labels,
  num_epochs=1,
  shuffle=False)
twitter_results = celeb_classifier.evaluate(input_fn=twitter_input_fn)
print("Twitter accuracy" ,twitter_results)



