from enum import Enum

RESIZE_DIM = 256
IMAGE_DIM = 224
BATCH_SIZE = 1
RATIO_TRAIN_TEST = 2 / 3
MEAN = [0.485, 0.456, 0.406]
STD = [0.229, 0.224, 0.225]
KMEANS_THRESHOLD = 0.3
LIMIT_TRAIN_EPOCH_LEN = 150
LIMIT_TEST_LEN = 60
N_ENREG_IMG = 140
PATH_IMG_FOLDER = "C:\\Users\\alexi\\PROJET_3A\\projet_3A\\images\\"

class Mode(Enum):
    TARGET = 1
    TARGET_AND_FLEE = 2
    FLEE = 3

class RandomMode(Enum):
    FULL_RANDOM = 1
    TRAIN_KMEANS = 2
    TRAIN_TEST_KMEANS = 3
    
if __name__ == "__main__":
    pass