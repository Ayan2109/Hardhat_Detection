import numpy as np 
import cv2
import torch
import glob as glob
from createxmlfiles import generatexmlfile
from model import create_model
from detectcolors import colors, closest
from config import CLASSES, NUM_CLASSES
import argparse
import os
import time

device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')


model = create_model(NUM_CLASSES).to(device)
model.load_state_dict(torch.load(
	'../outputs/model4.pth', map_location = device
	))

model.eval()



DIR_TEST = '../test_data/Images'
DIR_VIDEO = 'D:\\test_video\\Top 10 Safety Vest For Construction For Men And Women.mp4'

test_images = glob.glob(f"{DIR_TEST}/*")
print(f"Test_instances: {len(test_images)}")


detection_threshold = 0.8



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
			outputs = model(img)
			 

		outputs = [{k:v.to('cpu') for k, v in t.items()} for t in outputs]

		boxes = outputs[0]['boxes'].data.numpy()
		scores = outputs[0]['scores'].data.numpy()
	
		boxes = boxes[scores >= detection_threshold].astype(np.int32)

		draw_boxes = boxes.copy()
		db = draw_boxes
		print(outputs[0]['labels'].cpu().numpy())
		pred_classes = [CLASSES[i] for i in outputs[0]['labels'].cpu().numpy()]

		for j, box in enumerate(draw_boxes):
			if(pred_classes[j] == 'head'):
				r = 0
				g = 255
				b = 0
			else:	
				color = colors(orig_image,box)
				rgb,_ = closest(color)
				r = int(rgb[0])
				g = int(rgb[1])
				b = int(rgb[2])
			cv2.rectangle(
				orig_image,
				(int(box[0]), int(box[1])),
				(int(box[2]), int(box[3])),
				(b,g,r), 2)

			cv2.putText(
				orig_image,f"{pred_classes[j]}: {(scores[j]*100):.0f} %",
				(int(box[0]), int(box[1]-5)),
				cv2.FONT_HERSHEY_SIMPLEX, 0.5,(b,g,r),
				1)
		

			#cv2.imshow('Prediction', orig_image)
			#cv2.waitKey(1)
			cv2.imwrite(f"../test_predictions/Images/{img_name[7:]}.png",orig_image)


		generatexmlfile(img_name[7:]+".xml",img_name[7:]+".png", orig_image, draw_boxes, pred_classes)
		print(f"Image {i+1} done..")
		print('-'*50)


	print('TEST PREDICTIONS COMPLETE')


def predict_video():

	cap = cv2.VideoCapture(DIR_VIDEO)

	if(cap.isOpened() == False):
		print("Error while trying to read vidoe. Check path again")

	frame_width = int(cap.get(3))
	frame_height = int(cap.get(4))
	fourcc = 0x00000021
	save_name = "inference_video"
	out = cv2.VideoWriter(f"../test_predictions/video/{save_name}.mp4",
					 	cv2.VideoWriter_fourcc(*'avc1'), 30, (frame_width, frame_height))

	frame_count = 0
	total_fps = 0

	while(cap.isOpened()):
		ret, frame = cap.read()

		if ret:
			img = frame.copy()

			img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB).astype(np.float32)

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

			boxes = outputs[0]['boxes'].data.numpy()
			scores = outputs[0]['scores'].data.numpy()

			boxes = boxes[scores >= detection_threshold].astype(np.int32)

			draw_boxes = boxes.copy()
			pred_classes = [CLASSES[i] for i in outputs[0]['labels'].cpu().numpy()]

			for j, box in enumerate(draw_boxes):


				if(pred_classes[j] == 'head'):
					r = 0
					g = 255
					b = 0
				elif(pred_classes[j] == 'helmet'):
					r = 255
					g = 255
					b = 0
				#else:	
					#color = colors(orig_image,box)
					#rgb,_ = closest(color)
					#r = int(rgb[0])
					#g = int(rgb[1])
					#b = int(rgb[2])
				

				cv2.rectangle(
					frame,
					(int(box[0]), int(box[1])),
					(int(box[2]), int(box[3])),
					((b),(g),(r),0), 2)

				cv2.putText(
					frame,f"{pred_classes[j]}: {(scores[j]*100):.0f} %" ,
					(int(box[0]), int(box[1]-5)),
					cv2.FONT_HERSHEY_SIMPLEX, 0.5,((b),(g),(r),0),
					1)
					
					
			cv2.putText(frame, f"{fps:.1f} FPS", 
                    	(15, 25),
                    	cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 
                   		 2,)
			out.write(frame)

			if cv2.waitKey(1) & 0xFF == ord('q'):
				break

		else:
			break

	cap.release()

	avg_fps = total_fps / frame_count
	print(f"Average FPS: {avg_fps:.3f}")



if __name__ == '__main__':
	train = input("Predict hardhat on images or video, type video or image\n")

	if(train == 'video'):
		predict_video()

	if(train == 'image'):
		predict_images()






