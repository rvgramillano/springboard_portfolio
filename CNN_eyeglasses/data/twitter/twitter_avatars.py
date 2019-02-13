import requests 
from sklearn import preprocessing
import numpy as np
import tensorflow as tf
import random
import os
import matplotlib.pyplot as plt
import PIL.ImageOps
from PIL import Image

# using avatars.io service to retrieve and save twitter profile pictures
usernames = ['nntaleb', 'SimonDeDeo', 'AndrewYNg', 'EricTopol', 'ericries', 'AOC', 'mims', 'RayDalio', 'gvanrossum', 'alansmurray', 'RandomlyWalking', 'belril', 'geoffreyfowler', 'chrmanning', 'hsu_steve', 'sacca', 'pauldaugh', 'BillGates', 'hugo_larochelle', 'hmason']
#celebA size = 178 x 218
for j,username in enumerate(usernames):
	print username
	photo_url = 'https://avatars.io/twitter/%s' % username
	Picture_request = requests.get(photo_url)
	
	if Picture_request.status_code == 200:
		if j < 10:
			with open("/Users/rvg/Documents/springboard_ds/springboard_portfolio/CNN_eyeglasses/data/twitter/profile_pics/0%d.jpg" % j, 'wb') as f:
				f.write(Picture_request.content)
		else:
			with open("/Users/rvg/Documents/springboard_ds/springboard_portfolio/CNN_eyeglasses/data/twitter/profile_pics/%d.jpg" % j, 'wb') as f:
				f.write(Picture_request.content)