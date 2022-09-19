import torch
import cv2
import numpy as np



def colors(image, bndbox):
	factor = 4
	xmin , ymin , xmax, ymax = bndbox
	_ymax = ymax - (ymax - ymin) // factor
	ROI = image[xmin : xmax, ymin : _ymax]

	colors , count = np.unique( 
		ROI.reshape(-1, ROI.shape[-1]), axis = 0, return_counts = True)

	return colors[count.argmax()]





def closest(color):
    '''
    Parameter:
    colour: tuple (r, g, b)
    Returns:
    a list, which has [r, g, b] value for closest color from the passed `color` and name of colour
    '''
    color_map = {0: "blue",
                 1: "yellow",
                 2: "orange",
                 3: "white",
                 4: "red"}
    
    colors = np.array([[0, 128, 255], 
                       [255, 255, 0], 
                       [255, 145, 0], 
                       [255, 255, 255],
                       [255, 0 , 0]])
    color = np.array(color)
    distances = np.sqrt(np.sum((colors-color)**2, axis=1))
    index_of_smallest = np.where(distances == np.amin(distances))
    idx = index_of_smallest[0][0]
    return colors[idx], color_map[idx]