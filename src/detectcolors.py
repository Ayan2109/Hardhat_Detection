import torch
import cv2
import numpy as np



def colors(image, bndbox):
	factort = 4
	xmin , ymin , xmax, ymax = bndbox
	_ymax = ymax - (ymax - ymin)
	ROI = image[xmin : xmax, ymin : ymax]

	colors , count = np.unique( 
		ROI.reshape(-1, ROI.shape[-1]), axis = 0, return_counts = True)

	return colors[count.argmax()]





