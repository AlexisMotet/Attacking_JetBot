import math
import numpy as np

path = "U:/PROJET_3A/"
path = "C:/Users/alexi/PROJET_3A/"

config = {
    "PATH_MODEL" : path + 'Projet_Adversarial_Patch/Project_Adverserial_Patch/Collision_Avoidance/best_model_extended.pth',
    "N_CLASSES" : 2,
    "PATH_DATASET" : path + 'Projet_Adversarial_Patch/Project_Adverserial_Patch/Collision_Avoidance/dataset/',
    "PATH_IMG_FOLDER" : path +  "projet_MOTET/images/",
    "PATH_PRINTABLE_COLORS" : "printability/pantone-colors.json",
    "BATCH_SIZE_TRAIN" : 10,
    "BATCH_SIZE_TEST" : 1,
    "LAMBDA_TV" : 0.003,
    "LAMBDA_PRINT" : 0.002,
    "THRESHOLD" : 0.9,
    "MAX_ITERATIONS" : 10,
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
    "ANGLES_RANGE" : math.radians(5),
    "SCALE_FACTOR_MIN" : 0.7,
    "X_TOP_LEFT" : 44,
    "Y_TOP_LEFT" : 40,
    "X_BOTTOM_RIGHT" : 180,
    "Y_BOTTOM_RIGHT" : 140,
    "NOISE_STD" : np.array([0.0107681, 0.00957908, 0.01005988]),
    "BLUR_KERNEL_SIZE" : 5,
    "BLUR_SIGMA_MAX" : 0.5,
    "BRIGHTNESS_BIAS" : 0.1,
    "CONTRAST_GAIN" : 0.15,
    "RESIZE_DIM" : 256,
    "IMAGE_DIM" : 224,
    "RATIO_TRAIN_TEST" : 4/5,
    "NORMALIZATION_MEAN" : [0.485, 0.456, 0.406],
    "NORMALIZATION_STD" : [0.229, 0.224, 0.225],
    "LIMIT_TRAIN_EPOCH_LEN" : -1,
    "LIMIT_TEST_LEN" : -1,
    "N_ENREG_IMG" : 1,
    "N_COLORS" : 100,
}

config_ = config.copy()
config_["PATH_MODEL"] = "C:/Users/alexi/PROJET_3A/projet_NOUINOU/new_fruit_model.pth"
config_["PATH_DATASET"] = "C:/Users/alexi/PROJET_3A/projet_NOUINOU/new_fruit/train/"
config_["N_CLASSES"] = 5
config_["FX"] = 100.
config_["FY"] = 100.
config_["CX"] = 112.
config_["CY"] = 112.
config_["K1"] = 0.
config_["K2"] = 0.
config_["ANGLES_RANGE"] = 0.
config_["EXTRINSIC_R"] = np.eye(4)
config_["NOISE_STD"] = 0 * config_["NOISE_STD"]
config_["X_TOP_LEFT"] = 0
config_["Y_TOP_LEFT"] = 0
config_["X_BOTTOM_RIGHT"] = 224
config_["Y_BOTTOM_RIGHT"] = 224
"""
config["PATH_MODEL"] = "C:\\Users\\alexi\\PROJET_3A\\projet_MOTET\\new_imagenette2-160_model.pth"
config["PATH_DATASET"] = "C:\\Users\\alexi\\PROJET_3A\\imagenette2-160\\train\\"
config["N_CLASSES"] = 10
"""
config_colab = config.copy()

config_colab["PATH_MODEL"] = "/content/PROJET_3A/best_model_extended.pth"
config_colab["PATH_DATASET"] = "/content/PROJET_3A/dataset/"
config_colab["N_CLASSES"] = 2
config_colab["PATH_IMG_FOLDER"] = "/content/projet_3A/images/"
config_colab["PATH_PRINTABLE_COLORS"] = "/content/projet_3A/printability/pantone-colors.json"

