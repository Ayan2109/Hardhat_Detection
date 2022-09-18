import os
from pathlib import Path
from tqdm.auto import tqdm
import shutil
from sklearn.model_selection import train_test_split
import xml.etree.ElementTree as et


def split_dataset(annotationsDir,imagesDir,annotations_list,images_list, datasetType):
	dir_path = Path(f"../detections/{datasetType}")
	dir_path.mkdir(parents = True, exist_ok = True)

	prog_bar = tqdm(annotations_list, total = len(annotations_list))

	for id , annotations in enumerate(prog_bar):
		mytree = et.parse(os.path.join(annotationsDir,annotations))
		myroot = mytree.getroot()

		for folder in myroot.iter('folder'):
			folder.text = f"{datasetType}"

		mytree.write(os.path.join(annotationsDir,annotations))
		xml_file = annotations[:-4]

		image_file = xml_file
		xml_src_file = os.path.join(annotationsDir + "/" , xml_file + ".xml")
		image_src_file = os.path.join(imagesDir + "/", image_file + ".png")		
		shutil.copy(xml_src_file, dir_path)
		shutil.copy(image_src_file, dir_path)

		os.remove(xml_src_file)
		os.remove(image_src_file)


if __name__ == '__main__':
	
	annotationsDir = "../detections/annotations/"
	imagesDir = "../detections/images/"

	annotations_train_data, annotations_val_data = train_test_split(os.listdir(annotationsDir), test_size = 0.1)
	images_train_data, images_val_data = train_test_split = train_test_split(os.listdir(imagesDir), test_size = 0.1)

	#creates to different directories for train and val data 
	split_dataset(annotationsDir,imagesDir,annotations_train_data, images_train_data, "train")
	split_dataset(annotationsDir,imagesDir,annotations_val_data, images_val_data, "val")

	os.rmdir(annotationsDir)
	os.rmdir(imagesDir)


