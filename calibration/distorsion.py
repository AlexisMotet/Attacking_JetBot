import numpy as np
import torch
#https://stackoverflow.com/questions/58127381/opencv-how-to-apply-camera-distortion-to-an-image?noredirect=1&lq=1
#https://stackoverflow.com/questions/70205469/when-will-camera-distortion-coefficients-change

def load_coef(path_calibration):
    #https://stackoverflow.com/questions/39432322/what-does-the-getoptimalnewcameramatrix-do-in-opencv
    cam_mtx = np.loadtxt(path_calibration + 'camera_matrix.txt', dtype=float)
    dist_coef = np.loadtxt(path_calibration + 'distorsion_coefficients.txt', dtype=float)
    return cam_mtx, dist_coef

def distort_patch(cam_mtx, dist_coef, empty_img_p) :
    cam_mtx = cam_mtx
    fx, fy, cx, cy = cam_mtx[0][0], cam_mtx[1][1], cam_mtx[0][2], cam_mtx[1][2]
    dist_coef = dist_coef
    k1, k2, p1, p2, k3 = dist_coef[0], dist_coef[1], dist_coef[2], dist_coef[3], dist_coef[4]
    
    distorded_patch = torch.zeros_like(empty_img_p)
    
    map_ = {}
    dim_x, dim_y = empty_img_p.shape[3], empty_img_p.shape[2]
    for x in range(dim_x) :
        for y in range(dim_y):
            #normalize the point
            nx = (x - cx)/fx
            ny = (y - cy)/fy
            
            #radial distorsion and tangential distorsion
            r2 = nx**2 + ny**2
            dx = nx + nx * (k1*r2 + k2*r2**2 +k3*r2**3)
            dy = ny + ny * (k1*r2 + k2*r2**2 +k3*r2**3)
            
            dx = int(dx * fx + cx)
            dy = int(dy * fy + cy)
            if 0 <= dx < dim_x and  0 <= dy < dim_y :
                distorded_patch[0, :, dy, dx] = empty_img_p[0, :, y, x]
                map_[(dx, dy)] = (x, y)
    return distorded_patch, map_

def undistort_patch(empty_img_p, distorded_patch, map_):
    zero = torch.zeros(3)
    for dp, p in map_.items() :
            dx, dy = dp
            x, y = p
            if not torch.any(torch.eq(distorded_patch[0, :, dy, dx], zero)) :
                empty_img_p[0, : , y, x] = distorded_patch[0, :, dy, dx]
    return empty_img_p

'''
if __name__=='__main__' :
    path_calibration = 'C:\\Users\\alexi\\PROJET_3A\\projet_CAS\\calibration\\'
    cam_mtx, dist_coef = load_coef(path_calibration)
    distort_patch(cam_mtx, dist_coef, torch.tensor(np.zeros((1, 3, 224, 224))))
'''