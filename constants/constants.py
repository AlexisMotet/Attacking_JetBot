from enum import Enum


PATH_MODEL = 'U:\\PROJET_3A\\projet_NOUINOU_MOTET\\new_imagenette2-160_model.pth'
PATH_DATASET = 'U:\\PROJET_3A\\imagenette2-160\\train'
PATH_CALIBRATION = 'U:\\PROJET_3A\\projet_NOUINOU_MOTET\\calibration\\'
PATH_DISTORTION = 'U:\\PROJET_3A\\projet_NOUINOU_MOTET\\distortion\\distortion.so'
PATH_PRINTABLE_COLORS = 'U:\\PROJET_3A\\projet_NOUINOU_MOTET\\printability\\printable_colors.txt'
PATH_IMG_FOLDER = "U:\\PROJET_3A\\projet_NOUINOU_MOTET\\images\\"
"""
path_model = 'C:\\Users\\alexi\\PROJET_3A\\projet_3A\\new_imagenette2-160_model.pth'
path_dataset = 'C:\\Users\\alexi\\PROJET_3A\\imagenette2-160\\train'
path_calibration = 'C:\\Users\\alexi\\PROJET_3A\\projet_3A\\calibration\\'
path_distortion = 'C:\\Users\\alexi\\PROJET_3A\\projet_3A\\distortion\\distortion.so'
path_printable_colors = 'C:\\Users\\alexi\\PROJET_3A\\projet_3A\\printability\\printable_colors.txt'
PATH_IMG_FOLDER = "C:\\Users\\alexi\\PROJET_3A\\projet_3A\\images\\"
"""


MIN_BRIGHTNESS = 0.90
MAX_BRIGHTNESS = 1.1
MIN_CONTRAST = 0.90
MAX_CONTRAST = 1.1
MIN_SATURATION = 0.90
MAX_SATURATION = 1.1
MIN_HUE = -0.1
MAX_HUE = 0.1
RESIZE_DIM = 256
IMAGE_DIM = 224
BATCH_SIZE = 1
RATIO_TRAIN_TEST = 2 / 3
MEAN = [0.485, 0.456, 0.406]
STD = [0.229, 0.224, 0.225]
KMEANS_THRESHOLD = 0.3
LIMIT_TRAIN_EPOCH_LEN = 150
LIMIT_TEST_LEN = 60
N_ENREG_IMG = 30


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