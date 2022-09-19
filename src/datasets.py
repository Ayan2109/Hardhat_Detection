import torch
import cv2
import os
import numpy as np
import glob as glob

from xml.etree import ElementTree as et
from config import CLASSES, RESIZE_TO, TRAIN_DIR, VAL_DIR, BATCH_SIZE
from torch.utils.data import Dataset, DataLoader
from utils import collat_fn, get_train_transform, get_val_transform


class CreateDataset(Dataset):
	def __init__(self,dir_path,width,height,classes,transforms = None):
		self.transforms = transforms
		self.dir_path = dir_path
		self.height = height
		self.classes = classes
		self.width = width

		self.image_paths = glob.glob(f"{self.dir_path}/*.png")
		self.all_images = [image_path.split('/')[-1] for image_path in self.image_paths]
		self.all_images = sorted(self.all_images)

	def __getitem__(self,idx):
		image_name = self.all_images[idx]
		image_name = image_name.split('\\')[-1]
		image_path = os.path.join(self.dir_path, image_name)

		image = cv2.imread(image_path)

		image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB).astype(np.float32)
		image_resized = cv2.resize(image, (self.width, self.height))
		image_resized /= 255.0

		annot_fileName = image_name[:-4] + '.xml'
		annot_filePath = os.path.join(self.dir_path, annot_fileName)

		boxes = []
		labels = []
		mytree = et.parse(annot_filePath)
		root = mytree.getroot()

		image_height , image_width ,_ = image.shape

		for member in root.findall('object'):
			if member.find('name').text in self.classes:
				labels.append(self.classes.index(member.find('name').text))

				xmin = int(member.find('bndbox').find('xmin').text)
				ymin = int(member.find('bndbox').find('ymin').text)
				xmax = int(member.find('bndbox').find('xmax').text)
				ymax = int(member.find('bndbox').find('ymax').text)

				xmin_final = int(xmin/image_width)*self.width
				ymin_final = int(ymin/image_height)*self.height
				xmax_final = int(xmax/image_width)*self.width
				ymax_final = int(ymax/image_height)*self.height

	 
				boxes.append([xmin_final, ymin_final, xmax_final, ymax_final])



		boxes = torch.as_tensor(boxes, dtype=torch.float32)
		area = (boxes[:, 3] - boxes[:, 1]) * (boxes[:, 2] - boxes[:, 0])
		iscrowd = torch.zeros((boxes.shape[0],), dtype=torch.int64)
		labels = torch.as_tensor(labels, dtype=torch.int64)


		target = {}
		target["boxes"] = boxes
		target["labels"] = labels
		target["area"] = area
		target["iscrowd"] = iscrowd
		img_id = torch.tensor([idx])
		target["image_id"] = img_id


		if self.transforms:
			sample = self.transforms(image = image_resized,
									bboxes = target['boxes'],
									labels = labels)
			image_resized = sample['image']
			target['boxes'] = torch.tensor(sample['bboxes'])	

		return image_resized, target


	def __len__(self):
		return len(self.all_images)


train_dataset = CreateDataset(TRAIN_DIR,RESIZE_TO,RESIZE_TO,CLASSES,get_train_transform())
val_dataset = CreateDataset(VAL_DIR,RESIZE_TO,RESIZE_TO,CLASSES,get_val_transform())

train_loader = DataLoader(
	train_dataset,
	batch_size = BATCH_SIZE,
	shuffle = True,
	num_workers = 2,
	collate_fn = collat_fn
	)

val_loader = DataLoader(
	val_dataset,
	batch_size = BATCH_SIZE,
	shuffle = False,
	num_workers = 2,
	collate_fn = collat_fn
	)

print(f"Number of training samples : {len(train_dataset)}")
print(f"Number of validation samples : {len(val_dataset)}")


if __name__ == '__main__':
	dataset = CreateDataset(TRAIN_DIR,RESIZE_TO,RESIZE_TO,CLASSES)
	print(f"Number of training samples : {len(train_dataset)}")

	def visualize_img(image,target):
		for i in range(len(target['labels'])):
			box = target['boxes'][i]
			
			label = CLASSES[target['labels'][i]]
			cv2.rectangle(
				image,
				(int(box[0]), int(box[1]), int(box[2]), int(box[3])),
				(0,255,0), 1
				)
			cv2.putText(
				image, label, (int(box[0]), int(box[1])),
				cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,0,255), 2
				)

		cv2.imshow('Image',image)
		cv2.waitKey(0)


	num_img = 5
	for i in range(num_img):
		image, target = dataset[i]
		#print(target['boxes'])
		image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
		visualize_img(image,target)



