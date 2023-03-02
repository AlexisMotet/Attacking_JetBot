import math
import numpy as np

path = ""

config = {
    "PATH_MODEL" : "",
    "N_CLASSES" : 2,
    "PATH_DATASET" : "",
    "PATH_IMG_FOLDER" : "images/",
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
    "SCALE_FACTOR_MIN" : 0.9,
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
    "RATIO_TRAIN_TEST" : 3/4,
    "NORMALIZATION_MEAN" : [0.485, 0.456, 0.406],
    "NORMALIZATION_STD" : [0.229, 0.224, 0.225],
    "LIMIT_TRAIN_EPOCH_LEN" : -1,
    "LIMIT_TEST_LEN" : -1,
    "N_ENREG_IMG" : 1,
    "N_COLORS" : 100,
}





