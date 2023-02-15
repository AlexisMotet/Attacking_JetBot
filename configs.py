import math
import numpy as np

path = "U:/PROJET_3A/"
path = "C:/Users/alexi/PROJET_3A/"

config = {
    "PATH_MODEL" : path + 'Projet_Adversarial_Patch/Project_Adverserial_Patch/Collision_Avoidance/best_model_extended.pth',
    "N_CLASSES" : 2,
    "PATH_DATASET" : path + 'Projet_Adversarial_Patch/Project_Adverserial_Patch/Collision_Avoidance/dataset/',
    "PATH_IMG_FOLDER" : path +  "projet_MOTET/images/",
    "PATH_PRINTABLE_COLORS" : path + "projet_MOTET/printability/printable_colors.txt",
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
    "ANGLES_RANGE" : math.radians(10),
    "SCALE_FACTOR_MIN" : 0.8,
    "NOISE_STD" : 3*np.array([0.0107681, 0.00957908, 0.01005988]),
    "BLUR_KERNEL_SIZE" : 5,
    "BLUR_SIGMA_MAX" : 0.5,
    "BRIGHTNESS_BIAS" : 0.2,
    "CONTRAST_GAIN" : 0.35,
    "RESIZE_DIM" : 256,
    "IMAGE_DIM" : 224,
    "RATIO_TRAIN_TEST" : 2/3,
    "NORMALIZATION_MEAN" : [0.485, 0.456, 0.406],
    "NORMALIZATION_STD" : [0.229, 0.224, 0.225],
    "LIMIT_TRAIN_EPOCH_LEN" : 150,
    "LIMIT_TEST_LEN" : 70,
    "N_ENREG_IMG" : 1,
}

config_colab = config.copy()

config_colab["PATH_MODEL"] = "/content/PROJET_3A/new_imagenette2-160_model.pth"
config_colab["PATH_DATASET"] = "/content/PROJET_3A/imagenette2-160/train/"
config_colab["N_CLASSES"] = 10
config_colab["PATH_IMG_FOLDER"] = "/content/projet_3A/images/"
config_colab["PATH_PRINTABLE_COLORS"] = "/content/projet_3A/printability/printable_colors.txt"

