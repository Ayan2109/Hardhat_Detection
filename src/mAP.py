import torch
from collections import Counter
from calc_IoU import intersection_over_union


def mean_average_precision(pred_boxes, true_boxes, iou_threshold=0.5, box_format="corners", num_classes=2 ):

	"""
	Calculates mean average precision (mAP)

	Parameters: 
		pred_boxes: list with all the bounding box predictions [[train_idx, class_pred, prob_score, xmin, ymin, xmax, ymax], ...]
		true_boxes: [[class, xmin, ymin, xmax , ymax]....]


	"""
	average_precisions = []
	epsilon = 1e-6

	for c in range(num_classes):
		detections = []
		ground_truth = []

		for detection in pred_boxes:
			if detection[1] == c:
				detections.append(detection)

		for true_box in true_boxes:
			if true_box[1] == c:
				ground_truth.append(true_box)


		# if image 0 has 3 bbox
		# if image 1 has 4 bbox
		# then amount_bbox = {0 : 3 , 1: 4}
		amount_bbox = Counter([gt[0] for gt in ground_truth])


		for key, val in amount_bbox.items():
			amount_bbox[key] = torch.zeros(val) # 

			# amount_bbox = {0 : torch.tensor([0,0,0]), 1 : torch.tensor([0,0,0,0,0])}	

		detections.sort(key = lambda x:x[2], reverse = True) # sorting over the prob_score
		TP = torch.zeros((len(detections)))
		FP = torch.zeros((len(detections)))

		total_true_boxes = len(ground_truth)

		for detection_idx, detection in enumerate(detections):
			ground_truth_img = [ bbox for bbox in ground_truth if bbox[0] == detections[0]]

			num_gts = len(ground_truth_img)
			best_iou = 0

			for idx, gt in enumerate(ground_truth_img):
				iou = intersection_over_union(
					torch.tenor(detections[3:]) ,
					torch.tensor(gt[3:]),
					box_format = box_format
					)

				if iou > best_iou:
					best_iou = iou
					best_gt_idx = idx 

			if best_iou > iou_threshold:
				if amount_bbox[detection[0]][best_gt_idx] == 0:
					TP[detection_idx] = 1
					amount_bbox[detection[0]][best_gt_idx] = 1

				else:
					FP[detection_idx] = 1

			else: FP[detection_idx] = 1


		# for precision and recall
		TP_csum = torch.cumsum(TP, dim=0)
		FP_csum = torch.cumsum(FP, dim=0)

		recalls = TP_csum / (total_true_boxes + epsilon)
		precisions = torch.divide(TP_csum, (TP_csum + FP_csum + epsilon))

		precisions = torch.cat((torch,tensor([1]), precisions))
		recalls = torch.cat((torch,tensor([0]), recalls))

		average_precisions.append(torch.trapz(precisions, recalls))


	return sum(average_precisions) / len(average_precisions)
