from enum import Enum
import numpy as np

consts = {
    "PATH_MODEL" : "",
    "N_CLASSES" : int(),
    "PATH_DATASET" : "",
    "PATH_IMG_FOLDER" : "",
    "PATH_PRINTABLE_COLORS" : "",
    "FX" : float(),
    "FY" : float(),
    "CX" : float(),
    "CY" : float(),
    "K1" : float(),
    "K2" : float(),
    "EXTRINSIC_R" : np.array([]),
    "ANGLES_RANGE" : float(),
    "SCALE_FACTOR_MIN" : float(),
    "NOISE_STD" : np.array([]),
    "BLUR_KERNEL_SIZE" : int(),
    "BLUR_SIGMA_MAX" : float(),
    "BRIGHTNESS_BIAS" : float(),
    "CONTRAST_GAIN" : float(),
    "RESIZE_DIM" : int(),
    "IMAGE_DIM" : int(),
    "RATIO_TRAIN_TEST" : float(),
    "MEAN" : [],
    "STD" : [],
    "LIMIT_TRAIN_EPOCH_LEN" : int(),
    "LIMIT_TEST_LEN" : int(),
    "N_ENREG_IMG" : int(),
}

class Mode(Enum):
    TARGET = 1
    TARGET_AND_FLEE = 2
    FLEE = 3

