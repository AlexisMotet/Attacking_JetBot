import numpy as np
#https://stackoverflow.com/questions/58127381/opencv-how-to-apply-camera-distortion-to-an-image?noredirect=1&lq=1
#https://stackoverflow.com/questions/70205469/when-will-camera-distortion-coefficients-change

def load_coef(path_calibration):
    #https://stackoverflow.com/questions/39432322/what-does-the-getoptimalnewcameramatrix-do-in-opencv
    cam_mtx = np.loadtxt(path_calibration + 'camera_matrix.txt', dtype=float)
    dist_coef = np.loadtxt(path_calibration + 'distorsion_coefficients.txt', dtype=float)
    return cam_mtx, dist_coef

