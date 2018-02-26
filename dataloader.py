import cv2
import os
import numpy as np
import random

# dataset dir
DATASET_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'preprocessed_dataset')

# image dir labelled with yes
YES_DIR = os.path.join(DATASET_DIR, 'yes')

# image dir labelled with no
NO_DIR = os.path.join(DATASET_DIR, 'no')

# image used for testing
DISPUTED_DIR = os.path.join(DATASET_DIR, 'disputed')

class DataLoader:
	def __init__(self):

		# get the name list of dataset
		self.yes_image_name = [os.path.join(YES_DIR, item) for item in os.listdir(YES_DIR)]
		self.no_image_name = [os.path.join(NO_DIR, item) for item in os.listdir(NO_DIR)]
		self.test_image_name = [os.path.join(DISPUTED_DIR, item) for item in os.listdir(DISPUTED_DIR)]

	def get_batch(self, batch_size, data_type = 'train', channel = 'bgr'):
		assert data_type == 'train' or data_type == 'test'
		assert channel == 'bgr' or channel == 'rgb'

		if data_type == 'train':
			num_yes_data = batch_size // 2
			num_no_data = batch_size - num_yes_data

			name_list = random.sample(self.yes_image_name, num_yes_data) + random.sample(self.no_image_name, num_no_data)
			label = [1 for _ in range(num_yes_data)] + [0 for _ in range(num_no_data)]
		else:
			name_list = random.sample(self.test_image_name, batch_size)
			label = [2 for _ in range(batch_size)]

		# shuffle the data and label
		c = list(zip(name_list, label))
		random.shuffle(c)
		name_list, label = zip(*c)

		# load image
		images = []
		if channel == 'bgr':
			for img_name in name_list:
				img = cv2.imread(img_name)
				images.append(img)
		else:
			for img_name in name_list:
				img = cv2.imread(img_name)
				images.append(img[:, :, (2, 1, 0)])

		return np.array(images), np.array(label)

if __name__ == '__main__':
	loader = DataLoader()
	batch, label = loader.get_batch(32, data_type = 'test', channel = 'rgb')
	print(batch.shape)
	print(label)



		