import numpy as np
import os
import matplotlib.pyplot as plt
from PIL import Image

DIR = '/Users/rvg/Documents/springboard_ds/springboard_portfolio/CNN_eyeglasses/data/twitter/'
dim = 28
fs = 8
# go through dataset, align image, convert image to grayscale, resize and save in separate data structure
for image_name in os.listdir(DIR + 'profile_pics/aligned/'):
	fig,ax = plt.subplots(1,4)
	print(image_name)
	im_unaligned = Image.open(DIR + 'profile_pics/' + image_name)
	ax[0].imshow(im_unaligned)
	ax[0].axis('off')
	ax[0].set_title('Initial Twitter' + '\n' + 'Profile Picture', fontsize=fs)
	image = Image.open(DIR + 'profile_pics/aligned/' + image_name)
	ax[1].imshow(image)
	ax[1].axis('off')
	ax[1].set_title('Aligned and Resized' + '\n' + '(178 x 218) Image', fontsize=fs)
	image = image.convert('L') #convert into grayscale
	ax[2].imshow(image)
	ax[2].axis('off')
	ax[2].set_title('Grayscale Image', fontsize=fs)
	image = image.resize((dim, dim), Image.BICUBIC) #resize into 28x28 dimension
	ax[3].imshow(image)
	ax[3].axis('off')
	ax[3].set_title('Resized' + '\n' + '(28 x 28) Image', fontsize=fs)
	#plt.tight_layout()
	plt.show()
	plt.savefig('/Users/rvg/Documents/springboard_ds/springboard_portfolio/CNN_eyeglasses/data/twitter/profile_pics/image_process/%s.png'%image_name, dpi=350, bbox_inches='tight')
	raw_input('...')
	plt.clf()
	plt.close()