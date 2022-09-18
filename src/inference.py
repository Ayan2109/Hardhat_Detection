import numpy as np 
import cv2
import torch
import glob as glob
import xml.etree.ElementTree as et
from model import create_model
from detectcolors import colors
import argparse
import os


device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')


model = create_model(num_classes=2).to(device)
model.load_state_dict(torch.load(
	'../outputs/model5.pth', map_location = device
	))

model.eval()



DIR_TEST = '../test_data/Images'
DIR_VIDEO = '../test_video'

test_images = glob.glob(f"{DIR_TEST}/*")
print(f"Test_instances: {len(test_images)}")

CLASSES = ['helmet' , 'head']

detection_threshold = 0.7


def generatexmlfile(filename,imgname,image,draw_boxes,pred_classes):
	root = et.Element("annotation")
	s1 = et.Element("folder")
	s1.text = DIR_TEST
	root.append(s1)

	s2 = et.Element("filename")
	s2.text = imgname
	root.append(s2)

	path = et.Element('path')
	path.text = os.path.join(DIR_TEST,imgname)
	root.append(path)

	d1 = et.Element("source")
	s3 = et.SubElement(d1,"database")
	s3.text = 'unknown'
	root.append(d1) 


	image_height , image_width , depth = image.shape

	d2 = et.Element('size')
	s4 = et.SubElement(d2,'width') 
	s4.text = str(image_width)
	s5 = et.SubElement(d2,'height')
	s5.text = str(image_height)
	s6 = et.SubElement(d2,'depth')
	s6.text = str(depth)

	root.append(d2)

	s7 = et.Element("segmented")
	s7.text = str(0)
	root.append(s7)

	for i , boxes in enumerate(draw_boxes):
		d3 = et.Element('object')

		name = et.SubElement(d3,'name')
		name.text = pred_classes[i]

		pose = et.SubElement(d3,'pose')
		name.text = "Unspecified"

		truncated = et.SubElement(d3,'truncated')
		truncated.text = str(0)

		difficult = et.SubElement(d3,'difficult')
		difficult.text = str(0)

		occluded = et.SubElement(d3,'occluded')
		occluded.text = str(0)


		d4 = et.Element('bndbox')

		xmin = et.SubElement(d4,'xmin')
		xmin.text = str(boxes[0])

		ymin = et.SubElement(d4,'ymin')
		ymin.text = str(boxe[1])

		xmax = et.SubElement(d4,'xmax')
		xmax.text = str(boxes[2])

		ymax = et.SubElement(d4,'ymax')
		ymax.text = str(boxes[2])

		d3.append(d4)

	tree = et.ElementTree(root)

	with open (filename, "wb") as files:
		files.write(tree)
  



def predict_images():
	for i in range(len(test_images)):
		img_name = test_images[i].split('/')[-1].split('.')[0]
		img = cv2.imread(test_images[i])

		orig_image = img.copy()

		img = cv2.cvtColor(orig_image, cv2.COLOR_BGR2RGB).astype(np.float32)

		img /= 255.0

		img = np.transpose(img, (2,0,1)).astype(float)
		img = torch.tensor(img , dtype = torch.float).cuda()

		img = torch.unsqueeze(img, 0)
		with torch.no_grad():
			outputs = model(img.to(device))
			 

		outputs = [{k:v.to('cpu') for k, v in t.items()} for t in outputs]

		if len(outputs[0]['boxes']) != 0:
			boxes = outputs[0]['boxes'].data.numpy()
			scores = outputs[0]['scores'].data.numpy()
	
			boxes = boxes[scores >= detection_threshold].astype(np.int32)
			scores = scores[scores >= detection_threshold].astype(np.int32)

			draw_boxes = boxes.copy()
			pred_classes = [CLASSES[i] for i in outputs[0]['labels'].cpu().numpy()]

			for j, box in enumerate(draw_boxes):
				if(pred_classes[j] == 'head'):
					r = 0
					g = 255 
					b = 0
				else:	
					r, g, b = colors(orig_image,box)
					r = int(r)
					g = int(g)
					b = int(b)

				cv2.rectangle(
					orig_image,
					(int(box[0]), int(box[1])),
					(int(box[2]), int(box[3])),
					(r,g,b), 2)

				cv2.putText(
					orig_image, pred_classes[j],
					(int(box[0]), int(box[1] - 5)),
					cv2.FONT_HERSHEY_SIMPLEX, 0.4, (r, g, b),
					1)
				cv2.putText(
					orig_image, str(scores[j] * 100) +"%",
					(int(box[0]+5), int(box[1] - 2)),
					cv2.FONT_HERSHEY_SIMPLEX, 0.4, (r, g, b),
					1)

			cv2.imshow('Prediction', orig_image)
			cv2.waitKey(1)
			cv2.imwrite(f"../test_predictions/images/{img_name}.png", orig_image,)


		#generatexmlfile(img_name+".xml",img_name+".png", orig_image, draw_boxes, pred_classes)
		print(f"Image {i+1} done..")
		print('-'*50)


	print('TEST PREDICTIONS COMPLETE')


def predict_video():
	os.mkdirs(os.path.join("../test_predictions" , 'video'), exist_ok = True)

	cap = cv2.VideoCapture(f"{DIR_VIDEO}/*.mp4")

	if(cap.isOpened() == False):
		print("Error while trying to read vidoe. Check path again")

	frame_width = int(cap.get(3))
	frame_height = int(cap.get(4))

	save_name = "inference_video"
	out = cv2.VideoWriter(f"../test_predictions/video/{save_name}.mp4",
					 	cv2.VideoWriter_fourcc(*'mp4v'), 30, 
                      	(frame_width, frame_height))

	frame_count = 0
	total_fps = 0

	while(cap.isOpened()):
		ret, frame = cap.read()

		if ret:
			img = frame.copy()

			img = cv2.cvtColor(image, cv2.COLOR_BGR2RGB).astype(np.float32)

			img /= 255.0

			img = np.transpose(img, (2,0,1)).astype(float)
			img = torch.tensor(img , dtype = torch.float).cuda()

			img = torch.unsqueeze(img, 0)

			start_time = time.time()
			with torch.no_grad():
				outputs = model(img.to(device))

			end_time = time.time()


			fps = 1/ (end_time - start_time)

			total_fps += fps

			frame_count += 1

			outputs = [{k:v.to('cpu') for k, v in t.items()} for t in outputs]

			if len(outputs[0]['boxes']) != 0:
				boxes = outputs[0]['boxes'].data.numpy()
				scores = outputs[0]['scores'].data.numpy()

				boxes = boxes[scores >= detection_threshold].astype(np.int32)
				scores = scores[scores >= detection_threshold].astype(np.int32)

				draw_boxes = boxes.copy()
				pred_classes = [CLASSES[i] for i in outputs[0]['labels'].cpu().numpy()]

				for j, box in enumerate(draw_boxes):
					r, g, b = colors(frame,box)
					cv2.rectangle(
						frame,
						(int(box[0]), int(box[1])),
						(int(box[2]), int(box[3])),
						(r,g,b), 2)

					cv2.putText(
						frame, pred_classes[j],
						(int(box[0]), int(box[1] - 5)),
						cv2.FONT_HERSHEY_SIMPLEX, 0.6, (r, g, b),
						2)
					cv2.putText(
						frame, str(scores[j] * 100) +"%",
						(int(box[0]+2), int(box[1] - 2)),
						cv2.FONT_HERSHEY_SIMPLEX, 0.6, (r, g, b),
						2)
			cv2.putText(frame, f"{fps:.1f} FPS", 
                    	(15, 25),
                    	cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 
                   		 2, lineType=cv2.LINE_AA
                   		 )
			cv2.imshow('image',frame)
			out.write(frame)

			if cv2.waitKey(1) & 0xFF == ord('q'):
				break

		else:
			break

	cap.release()
	cv2.DestroyAllWindows()

	avg_fps = total_fps / frame_count
	print(f"Average FPS: {avg_fps:.3f}")



if __name__ == '__main__':
	train = input("Predict hardhat on images or video, type video or image\n")

	if(train == 'video'):
		predict_video()

	if(train == 'image'):
		predict_images()






