import math
import numpy as np

path = "U:/PROJET_3A/"
path = "C:/Users/alexi/PROJET_3A/"

config = {
    "PATH_MODEL" : path + 'Projet_Adversarial_Patch/Project_Adverserial_Patch/Collision_Avoidance/best_model_extended.pth',
    "N_CLASSES" : 2,
    "PATH_DATASET" : path + 'Projet_Adversarial_Patch/Project_Adverserial_Patch/Collision_Avoidance/dataset/',
    "PATH_IMG_FOLDER" : path +  "projet_MOTET/images/",
    "PATH_PRINTABLE_COLORS" : path +  "projet_MOTET/printability/printable_colors.txt",
    "FX" : 107.75,
    "FY" : 138.46,
    "CX" : 107.06,
    "CY" : 112.53,
    "K1" : -0.2388,
    "K2" : 0.0337,
    "EXTRINSIC_R" : np.linalg.inv(np.array([[0.9997, 0.0078, -0.0247, 0],
                                        [-0.0034, 0.9849, 0.1729, 0],
                                        [0.0257, -0.1728, 0.9846, 0],
                                        [0, 0, 0, 1]])),
    "ANGLES_RANGE" : math.radians(20),
    "SCALE_FACTOR_MIN" : 0.8,
    "NOISE_STD" : 2 * np.array([0.06951714, 0.08960456, 0.14701256]),
    "BLUR_KERNEL_SIZE" : 5,
    "BLUR_SIGMA_MAX" : 0.5,
    "BRIGHTNESS_BIAS" : 0.25,
    "CONTRAST_GAIN" : 0.4,
    "RESIZE_DIM" : 256,
    "IMAGE_DIM" : 224,
    "RATIO_TRAIN_TEST" : 2 / 3,
    "MEAN" : [0.485, 0.456, 0.406],
    "STD" : [0.229, 0.224, 0.225],
    "LIMIT_TRAIN_EPOCH_LEN" : -1,
    "LIMIT_TEST_LEN" : -1,
    "N_ENREG_IMG" : 20,
}

config_aws = config.copy()

config_aws["PATH_MODEL"] = "../model.pth"
config_aws["PATH_DATASET"] = "../dataset/"
config_aws["PATH_IMG_FOLDER"] = "images/"
