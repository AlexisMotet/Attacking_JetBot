from enum import Enum
import numpy as np
import json

path = "C:\\Users\\alexi\\PROJET_3A\\"

consts = {
    "PATH_MODEL" : path + 'Projet_Adversarial_Patch\\Project_Adverserial_Patch\\Collision_Avoidance\\best_model_extended.pth',
    "N_CLASSES" : 2,
    "PATH_DATASET" : path + 'Projet_Adversarial_Patch\\Project_Adverserial_Patch\\Collision_Avoidance\\dataset\\',
    "PATH_IMG_FOLDER" : path +  "projet_3A\\images\\",
    "PATH_PRINTABLE_COLORS" : path +  "projet_3A\\printability\\printable_colors.txt",
    "FX" : 108,
    "FY" : 139,
    "CX" : 107,
    "CY" : 112,
    "K1" : -0.2397,
    "K2" : 0.0341,
    "EXTRINSIC_R" : np.linalg.inv(np.array([[0.997, 0.0077, -0.0243, 0],
                                        [-0.0035, 0.9855, 0.1699, 0],
                                        [0.0252, -0.1697, 0.9852, 0],
                                        [0, 0, 0, 1]])),
    "ANGLES_RANGE" : 0,
    "SCALE_FACTOR_MIN" : 1,
    "NOISE_INTENSITY" : 0.1,
    "BLUR_KERNEL_SIZE" : 5,
    "BLUR_SIGMA_MAX" : 0.5,
    "BRIGHTNESS_BIAS" : 0.25,
    "CONTRAST_GAIN" : 0.4,
    "RESIZE_DIM" : 256,
    "IMAGE_DIM" : 224,
    "RATIO_TRAIN_TEST" : 2 / 3,
    "MEAN" : [0.485, 0.456, 0.406],
    "STD" : [0.229, 0.224, 0.225],
    "LIMIT_TRAIN_EPOCH_LEN" : None,
    "LIMIT_TEST_LEN" : None,
    "N_ENREG_IMG" : 40,
}


class Mode(Enum):
    TARGET = 1
    TARGET_AND_FLEE = 2
    FLEE = 3

