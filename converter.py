import os
from PIL import Image
import random
import numpy as np
import cv2

# dataset dir
DATASET_DIR = os.path.join(os.getcwd(), 'dataset')

# output dataset dir
OUPUT_DIR = os.path.join(os.getcwd(), 'preprocessed_dataset')

# output image size
OUTPUT_SIZE = (224, 224, 3)

# label
YES = [2, 3, 4, 5, 6, 8, 9, 21, 22, 24, 27, 28]
NO = [11, 12, 13, 14, 15, 16, 17, 18, 19]
DISPUTE = [1, 7, 10, 20, 23, 25, 26]

im = [item for item in os.listdir(DATASET_DIR) 
				if item[-3:] == 'jpg' or item[-4:] == 'tiff' or item[-3:] == 'TIF' or item[-3:] == 'tif']

def makedir(name):
	if not os.path.isdir(name):
		os.makedirs(name)

# make directory for preprocessed data
makedir(OUPUT_DIR)
makedir(os.path.join(OUPUT_DIR, 'yes'))
makedir(os.path.join(OUPUT_DIR, 'no'))
makedir(os.path.join(OUPUT_DIR, 'disputed'))



yes = []
no = []
disputed = []

for item in im:
	if not item[1] == '.':
		im_id = int(item[:2])
	else:
		im_id = int(item[0])

	if im_id in YES:
		yes.append(item)
	elif im_id in NO:
		no.append(item)
	else:
		disputed.append(item)

def load_img(dir):
	im = Image.open(os.path.join(DATASET_DIR, dir))
	# (height, width, channel)
	return np.array(im)[:, :, :3]

def crop(img, label, counter):
	assert label == 'yes' or label == 'no' or label == 'disputed'

	max_height = img.shape[0] - OUTPUT_SIZE[0]
	max_width = img.shape[1] - OUTPUT_SIZE[1]
	crop_number = (max_height // OUTPUT_SIZE[0]) * (max_width // OUTPUT_SIZE[1])
	for _ in range(crop_number):
		start_point = (random.randint(0, max_height), random.randint(0, max_width)) # (height, width)
		sub_image = img[start_point[0]:start_point[0] + OUTPUT_SIZE[0], start_point[1]:start_point[1] + OUTPUT_SIZE[1], :]
		
		if label == 'yes':
			counter[0] += 1
			ct = counter[0]
			dir = os.path.join(OUPUT_DIR, 'yes')
		elif label == 'no':
			counter[1] += 1
			ct = counter[1]
			dir = os.path.join(OUPUT_DIR, 'no')
		elif label == 'disputed':
			counter[2] += 1
			ct = counter[2]
			dir = os.path.join(OUPUT_DIR, 'disputed')
		else:
			raise ValueError('Unknown label:', label)

		img_name = str(ct) + '.jpg'
		cv2.imwrite(os.path.join(dir, img_name), sub_image[:, :, (2, 1, 0)])

def get_some_crops_from_one_dispute_painting(img):
	max_height = img.shape[0] - OUTPUT_SIZE[0]
	max_width = img.shape[1] - OUTPUT_SIZE[1]
	crop_number = (max_height // OUTPUT_SIZE[0]) * (max_width // OUTPUT_SIZE[1])
	allsubimages = []
	for _ in range(crop_number):
		start_point = (random.randint(0, max_height), random.randint(0, max_width))  # (height, width)
		sub_image = img[start_point[0]:start_point[0] + OUTPUT_SIZE[0],
					start_point[1]:start_point[1] + OUTPUT_SIZE[1], :]
		allsubimages.append(sub_image)
	return np.array(allsubimages)

# cv2.imwrite('test.jpg', im[:, :, (2, 1, 0)])
# print(im.shape)
counter = [0, 0, 0]
for item in yes:
	im = load_img(item)
	crop(im, 'yes', counter)

for item in no:
	im = load_img(item)
	crop(im, 'no', counter)

disputed_paintings = []
index_to_num = {}
for i in range(len(DISPUTE)):
	index_to_num[DISPUTE[i]] = i

for item in disputed:
	im = load_img(item)
	crop(im, 'disputed', counter)
	crops = get_some_crops_from_one_dispute_painting(im, crop_number=16)
	disputed_paintings.append(crops)

def send_crops_of_this_dispute_painting(number_on_disk):
	if number_on_disk not in DISPUTE:
		raise ValueError('it has to be in 1 7 10 20 23 25 26')
	idx = index_to_num[number_on_disk]
	return disputed_paintings[idx]





