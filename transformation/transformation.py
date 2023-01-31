import math
import numpy as np
import constants.constants as c
import torch

class TransformationTool():
    def __init__(self, patch_dim):
        self.patch_dim = patch_dim
        
        cam_mtx = np.loadtxt(c.PATH_CALIBRATION + 'camera_matrix.txt', 
                             dtype=float)
        
        dist_coefs = np.loadtxt(c.PATH_CALIBRATION + 'distortion_coefficients.txt', 
                             dtype=float)
        
        fx, fy = cam_mtx[0][0], cam_mtx[1][1]
        f = (fx + fy)/2
        cx, cy = c.IMAGE_DIM/2, c.IMAGE_DIM/2
        
        self.k1 = dist_coefs[0]
        self.k2 = dist_coefs[1]
        self.k3 = dist_coefs[4]
        
        self.matrix_2dto3d = np.array([[1/f, 0, -cx/f],
                                       [0, 1/f, -cy/f],
                                       [0, 0, 0],
                                       [0, 0, 1]])
        
        self.matrix_3dto2d = np.array([[f, 0, cx, 0],
                                       [0, f, cy, 0],
                                       [0, 0, 1, 0]])
        
    def _get_matrix_transformation(self, x_t, y_t, scale_factor,
                                  alpha, beta, gamma):
        
        rx = np.array([[1, 0, 0, 0],
                       [0, math.cos(alpha), -math.sin(alpha), 0],
                       [0, math.sin(alpha), math.cos(alpha), 0],
                       [0, 0, 0, 1]])

        ry = np.array([[math.cos(beta), 0, -math.sin(beta), 0],
                       [0, 1, 0, 0],
                       [math.sin(beta), 0, math.cos(beta), 0],
                       [0, 0, 0, 1]])

        rz = np.array([[math.cos(gamma), -math.sin(gamma), 0, 0],
                       [math.sin(gamma), math.cos(gamma), 0, 0],
                       [0, 0, 1, 0],
                       [0, 0, 0, 1]])

        r = rx @ ry @ rz

        matrix_translation = np.array([[1, 0, 0, x_t],
                                       [0, 1, 0, y_t],
                                       [0, 0, 1, 1/scale_factor],
                                       [0, 0, 0, 1]])
        
        matrix_transformation = matrix_translation @ r
        
        return matrix_transformation

    def _transform(self, image, mtx_transfo):
        transformed = torch.zeros_like(image)
        map_ = {}
        save = {}
        for i in range(c.IMAGE_DIM//2-self.patch_dim//2, 
                       c.IMAGE_DIM//2+self.patch_dim//2):
            for j in range(c.IMAGE_DIM//2-self.patch_dim//2, 
                           c.IMAGE_DIM//2+self.patch_dim//2):
                if image[0, :, j, i].numpy().any():
                    point_2d = np.array([j, i, 1])
                    point_3d = self.matrix_2dto3d@point_2d
                    point_transformed = mtx_transfo@point_3d
                    tj, ti = point_transformed[:2]
                    
                    r2 = pow(ti, 2) + pow(tj, 2)
                    d = 1/(1 - (self.k1*r2 + self.k2*r2 + self.k3*r2))
                    di = ti * d
                    dj = tj * d
                    point_distorted = point_transformed.copy()
                    point_distorted[:2] = dj, di
                    
                    point_projected = self.matrix_3dto2d@point_distorted
                    nj_float, ni_float = point_projected[:2]/point_projected[2]
                        
                    ni = int(round(ni_float))
                    nj = int(round(nj_float))
                    
                    if 0 <= ni < c.IMAGE_DIM and 0 <= nj < c.IMAGE_DIM :
                        d = math.sqrt((ni - ni_float)**2 + 
                                      (nj - nj_float)**2)
                        weight = (math.sqrt(0.5) - d)/math.sqrt(0.5) + 1/2
                        map_[(j, i)] = (nj, ni)
                        if (nj, ni) not in save:
                            save[(nj, ni)] = {"values" : [], "weights" : []}
                        save[(nj, ni)]["values"].append(weight * image[0, :, j, i])
                        save[(nj, ni)]["weights"].append(weight)
                        transformed[0, :, nj, ni] = sum(save[(nj, ni)]["values"])/ \
                                                    sum(save[(nj, ni)]["weights"])
        return transformed, map_
    
    def random_transform(self, image):
        scale_factor = np.random.uniform(0.8, 1)
        x_c, y_c = c.IMAGE_DIM//2, c.IMAGE_DIM//2
        x_t, y_t = np.random.randint(-c.IMAGE_DIM//2 + self.patch_dim//2, 
                                      c.IMAGE_DIM//2 - self.patch_dim//2, 2)
        point_t = (1/scale_factor) * np.array([x_c + x_t, y_c + y_t, 1])
        nx_t, ny_t = (self.matrix_2dto3d@point_t)[:2]
        angles = np.zeros(3)
        angles[np.random.randint(2)] = np.random.uniform(-0.2, 0.2)
        mtx_transfo = self._get_matrix_transformation(nx_t, ny_t, 
                                                      scale_factor, *angles)
        return self._transform(image, mtx_transfo)
            
    def undo_transform(self, image, transformed, map_):
        for i in range(c.IMAGE_DIM//2-self.patch_dim//2, 
                       c.IMAGE_DIM//2+self.patch_dim//2):
            for j in range(c.IMAGE_DIM//2-self.patch_dim//2, 
                           c.IMAGE_DIM//2+self.patch_dim//2):
                if (j, i) in map_ :
                    nj, ni = map_[(j, i)]
                    if 0 <= ni < c.IMAGE_DIM and 0 <= nj <  c.IMAGE_DIM :
                        image[0, :, j, i] = transformed[0, :, nj, ni]
        return image
