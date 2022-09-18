from config import DEVICE, NUM_CLASSES, NUM_EPOCHS, OUT_DIR
from config import VISUALIZE_TRANSFORMED_IMAGES
from config import SAVE_PLOTS_EPOCH, SAVE_MODEL_EPOCH
from model import create_model
from utils import Averager
from tqdm.auto import tqdm
from datasets import train_loader, val_loader

import torch
import matplotlib.pyplot as plt 
import time 
from mAP import mean_average_precision

plt.style.use('ggplot')

def train(train_loader, model):
	print("TRAINING")
	global train_itr
	global train_loss_list

	prog_bar = tqdm(train_loader, total = len(train_loader))

	for i, data in enumerate(prog_bar):
		
		images, targets = data
		images = list(image.to(DEVICE) for image in images)
		targets = [{k: v.to(DEVICE) for k, v in t.items()} for t in targets]

		loss_dict = model(images,targets)
	
		losses = sum(loss for loss in loss_dict.values())

		loss_value = losses.item()

		train_loss_list.append(loss_value)
		train_loss_hist.send(loss_value)

		optimizer.zero_grad()
		losses.backward()
		optimizer.step()

		train_itr += 1

		prog_bar.set_description(desc=f"TRAIN_LOSS: {loss_value:.4f}")
		torch.cuda.empty_cache()

	return train_loss_list



def validate(val_loader, model):
	print("VALIDATING")
	global val_itr
	global val_loss_list
	#global val_mAP_list
	prog_bar = tqdm(val_loader, total = len(val_loader))

	for i, data in enumerate(prog_bar):
		images, targets = data 
		images = list(image.to(DEVICE) for image in images)
		targets = [{k: v.to(DEVICE) for k , v in t.items()} for t in targets]

		with torch.no_grad():
			loss_dict = model(images, targets)
		"""
		pred_boxes = []
		label_boxes = []
		for idx , target in range(len(targets)):
			for k in range(len(targets[i]['labels'])):
				pred_boxes.append([
						idx,
						targets[i]['labels'][k],
						targets[i]['boxes'][k][0], #xmin
						targets[i]['boxes'][k][1], #ymin
						targets[i]['boxes'][k][2], #xmax
						targets[i]['boxes'][k][3] #ymax
					])

		loss_dict = model(images,targets)
		# mAP code 
		for idx in range(len(loss_dict)):
			for k in range(len(loss_dict[i]['labels'])):
				pred_boxes.append([
						idx,
						loss_dict[i]['labels'][k],
						loss_dict[i]['scores'][k],
						loss_dict[i]['boxes'][k][0], #xmin
						loss_dict[i]['boxes'][k][1], #ymin
						loss_dict[i]['boxes'][k][2], #xmax
						loss_dict[i]['boxes'][k][3] #ymax
					])


		mAp_value = mean_average_precision(pred_boxes, label_boxes, box_format = "corners", num_classes = NUM_CLASSES)
		"""
		losses = sum(loss for loss in loss_dict.values())
		loss_value = losses.item()

		val_loss_list.append(loss_value)
		val_loss_hist.send(loss_value)
		#val_mAP_list.append(mAp_value)
		#val_mAP_hist.send(mAp_value)

		val_itr += 1 

		prog_bar.set_description(desc=f"VAL_LOSS: {loss_value:.4f}")
		torch.cuda.empty_cache()

	return val_loss_list




if __name__ == '__main__':
	model = create_model(num_classes=NUM_CLASSES)
	model = model.to(DEVICE)

	params =[p for p in model.parameters() if p.requires_grad]
	optimizer = torch.optim.SGD(params, lr = 0.001,  momentum = 0.9, weight_decay = 0.0005)

	train_loss_hist = Averager()
	val_loss_hist = Averager()

	val_mAP_hist = Averager()

	train_itr = 1
	val_itr = 1

	train_loss_list = []
	val_loss_list = []
	#val_mAP_list = []


	MODEL_NAME = 'model'

	if VISUALIZE_TRANSFORMED_IMAGES:
		from utils import show_transform_image
		show_transform_image(train_loader)


	for epoch in range(NUM_EPOCHS):
		print(f"\nEPOCH {epoch+1} of {NUM_EPOCHS}")

		train_loss_hist.reset()
		val_loss_hist.reset()
		val_mAP_hist.reset()
		figure1, train_ax = plt.subplots()
		figure2, valid_ax = plt.subplots()

		start = time.time()

		train_loss = train(train_loader, model)
		val_loss = validate(val_loader, model)

		print(f"EPOCH #{epoch+1} train_loss: {train_loss_hist.value:.3f}")
		print(f"EPOCH #{epoch+1} val_loss: {val_loss_hist.value:.3f}")

		end = time.time()
		print(f"TIME TAKEN: {((end - start)/60):.3f} minutes for epoch {epoch}")
		
		if(epoch+1) % SAVE_MODEL_EPOCH == 0:
			torch.save(model.state_dict(), f"{OUT_DIR}/model{epoch+1}.pth")
			print('SAVING MODEL COMPLETE...\n')

		if(epoch+1) % SAVE_PLOTS_EPOCH == 0:
			train_ax.plot(train_loss, color='blue')
			train_ax.set_xlabel('iterations')
			train_ax.set_ylabel('train loss')
			valid_ax.plot(val_loss, color='red')
			valid_ax.set_xlabel('iterations')
			valid_ax.set_ylabel('train_loss')

			figure1.savefig(f"{OUT_DIR}/train_loss_{epoch+1}.png")
			figure2.savefig(f"{OUT_DIR}/val_loss_{epoch+1}.png")

			print('SAVING PLOTS COMPLETE..')
		
		if(epoch+1) == NUM_EPOCHS:

			train_ax.plot(train_loss, color='blue')
			train_ax.set_xlabel('iterations')
			train_ax.set_ylabel('train loss')
			valid_ax.plot(val_loss, color='red')
			valid_ax.set_xlabel('iterations')
			valid_ax.set_ylabel('train_loss')

			figure1.savefig(f"{OUT_DIR}/train_loss_{epoch+1}.png")
			figure2.savefig(f"{OUT_DIR}/val_loss_{epoch+1}.png")
			torch.save(model.state_dict(), f"{OUT_DIR}/model{epoch+1}.pth")
			print('SAVING MODEL COMPLETE...\n')
			print('SAVING PLOTS COMPLETE..')


		plt.close('all')
		torch.cuda.empty_cache()