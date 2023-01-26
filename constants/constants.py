from enum import Enum


PATH_MODEL = 'U:\\PROJET_3A\\projet_BONTEMPS_SCHAMPHELEIRE\\Project Adverserial Patch\\Collision Avoidance\\best_model_extended.pth'
N_CLASSES = 2
PATH_DATASET = 'U:\PROJET_3A\\projet_BONTEMPS_SCHAMPHELEIRE\\Project Adverserial Patch\\Collision Avoidance\\dataset\\'
PATH_CALIBRATION = 'U:\\PROJET_3A\\projet_NOUINOU_MOTET\\calibration\\'
PATH_IMG_FOLDER = "U:\\PROJET_3A\\projet_NOUINOU_MOTET\\imagess\\"

BLUR_KERNEL_SIZE = 9
BLUR_SIGMA_MAX = 1
BRIGHTNESS_BIAS = 0.25
CONTRAST_GAIN = 0.4
RESIZE_DIM = 256
IMAGE_DIM = 224
RATIO_TRAIN_TEST = 2 / 3
MEAN = [0.485, 0.456, 0.406]
STD = [0.229, 0.224, 0.225]
LIMIT_TRAIN_EPOCH_LEN = None
LIMIT_TEST_LEN = None
N_ENREG_IMG = 5


class Mode(Enum):
    TARGET = 1
    TARGET_AND_FLEE = 2
    FLEE = 3

class RandomMode(Enum):
    FULL_RANDOM = 1
    TRAIN_KMEANS = 2
    TRAIN_TEST_KMEANS = 3
