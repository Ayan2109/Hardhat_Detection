import torch

def intersection_over_union(boxes_preds, boxes_labels, box_format="corners"):
	"""
	Calculates intersection over uniom

	Parameters: 
		box_preds (tensor): Predictions of bounding boxes (Batch_size = 4)
		boxes_labels (tensor) : Correct labels of bounding boxes (Batch_size = 4)
		box_format (str) : midpoint / corners, if boxes (x,y,w,h) or (xmin,ymin,xmax,ymax)

	returns:
		tensor: intersection over union for all examples
	"""
	box1_x1 = boxes_preds[..., 0:1]
	box1_y1 = boxes_preds[..., 1:2]
	box1_x2 = boxes_preds[..., 2:3]
	box1_y2 = boxes_preds[..., 3:4] # doing this to maintatn the shape of the tenson ( N , 1)

	box2_x1 = boxes_labels[..., 0:1]
	box2_y1 = boxes_labels[..., 1:2]
	box2_x2 = boxes_labels[..., 2:3]
	box2_y2 = boxes_labels[..., 3:4]

	x1 = torch.max(box1_x1, box2_x1)
	y1 = torch.max(box1_y1, box2_y1)
	x2 = torch.min(box1_x2, box2_x2)
	y2 = torch.min(box1_y2, box2_y2)

	# clamp(0) is for the edge case when they do not intersect
	intersection = (x2-x1).clamp(0) * (y2-y1).clamp(0)

	box1_area = abs((box1_x2 - box1_x1) *  (box1_y2 - box1_y1))
	box2_area = abs((box2_x2 - box2_x1) *  (box2_y2 - box2_y1))

	return intersection / box1_area + box2_area - intersection + 1e-6 

