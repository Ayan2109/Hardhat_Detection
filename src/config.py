import torch
import split_train_val as stv 
from sklearn.model_selection import train_test_split
import os 

BATCH_SIZE = 15
RESIZE_TO = 256
NUM_EPOCHS = 1

DEVICE = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

TRAIN_DIR = '../detections/train'
VAL_DIR = '../detections/val'

CLASSES = ['background','helmet' , 'head']

NUM_CLASSES = 3

VISUALIZE_TRANSFORMED_IMAGES = True

OUT_DIR = '../outputs'

SAVE_PLOTS_EPOCH = 2
SAVE_MODEL_EPOCH = 2




