import math
import numpy as np
import constants.constants as c
import torch


class TransformationTool():
    f = c.IMAGE_DIM
    def __init__(self, patch_dim):
        self.patch_dim = patch_dim
        cam_mtx = np.loadtxt(c.PATH_CALIBRATION + 'camera_matrix.txt', 
                             dtype=float)
        dist_coefs = np.loadtxt(c.PATH_CALIBRATION + 'distortion_coefficients.txt', 
                             dtype=float)
        self.fx, self.fy = cam_mtx[0][0], cam_mtx[1][1]
        self.cx, self.cy = cam_mtx[0][2], cam_mtx[1][2]
        self.k1 = dist_coefs[0]
        self.k2 = dist_coefs[1]
        self.k3 = dist_coefs[4]
        
    def _get_matrix_transformation(self, x_translation, y_translation, z_translation,
                                  alpha, beta, gamma):
        
        matrix_2dto3d = np.array([[1, 0, -c.IMAGE_DIM/2],
                                  [0, 1, -c.IMAGE_DIM/2],
                                  [0, 0, 1],
                                  [0, 0, 1]])

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

        matrix_translation = np.array([[1, 0, 0, x_translation],
                                       [0, 1, 0, y_translation],
                                       [0, 0, 1, self.f + z_translation],
                                       [0, 0, 0, 1]])

        matrix_3dto2d = np.array([[self.f, 0, c.IMAGE_DIM/2, 0],
                                  [0, self.f, c.IMAGE_DIM/2, 0],
                                  [0, 0, 1, 0]])

        matrix_transformation = matrix_3dto2d @ (matrix_translation @ 
                                                (r @ matrix_2dto3d))
        
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
                    d = (mtx_transfo[2][0] * i + mtx_transfo[2][1] * j + 
                        mtx_transfo[2][2])
                    
                    new_i_float = (mtx_transfo[0][0] * i + mtx_transfo[0][1] * j + 
                                mtx_transfo[0][2])/d
                    
                    new_j_float = (mtx_transfo[1][0] * i + mtx_transfo[1][1] * j + 
                                mtx_transfo[1][2])/d

                    """ distortion """
                    
                    ni = (new_i_float - self.cx)/self.fx
                    nj = (new_j_float - self.cy)/self.fy

                    r2 = pow(ni, 2) + pow(nj, 2)
                    rk = self.k1*r2 + self.k2*(r2**2) + self.k3*(r2**3)
                    ndi_f = ni + ni * rk
                    ndj_f = nj + nj * rk
                    
                    new_i_float = ndi_f * self.fx + self.cx
                    new_j_float = ndj_f * self.fy + self.cy

                    """ end """
                        
                    new_i = int(round(new_i_float))
                    new_j = int(round(new_j_float))
                    
                    if 0 <= new_i < c.IMAGE_DIM and 0 <= new_j < c.IMAGE_DIM :
                        d = math.sqrt((new_i - new_i_float)**2 + 
                                      (new_j - new_j_float)**2)
                        weight = d/math.sqrt(0.5)
                        map_[(j, i)] = (new_j, new_i)
                        if (new_j, new_i) not in save:
                            save[(new_j, new_i)] = {"values" : [], "weights" : []}
                        save[(new_j, new_i)]["values"].append(weight * image[0, :, j, i])
                        save[(new_j, new_i)]["weights"].append(weight)
                        transformed[0, :, new_j, new_i] = sum(save[(new_j, new_i)]["values"])/sum(save[(new_j, new_i)]["weights"])
        return transformed, map_
    
    def random_transform(self, image):
        x_t, y_t = np.random.randint(-c.IMAGE_DIM//2 + self.patch_dim, 
                                     c.IMAGE_DIM//2 - self.patch_dim, 2)    
        z_t = np.random.randint(-10, 10)
        angles = np.zeros(3)
        angles[np.random.randint(2)] = (2 * np.random.rand() - 1)
        mtx_transfo = self._get_matrix_transformation(x_t, y_t, z_t, *angles)
        return self._transform(image, mtx_transfo)
        
            
    def undo_transform(self, image, transformed, map_):
        for i in range(c.IMAGE_DIM//2-self.patch_dim//2, 
                       c.IMAGE_DIM//2+self.patch_dim//2):
            for j in range(c.IMAGE_DIM//2-self.patch_dim//2, 
                           c.IMAGE_DIM//2+self.patch_dim//2):
                if (j, i) in map_ :
                    new_j, new_i = map_[(j, i)]
                    if 0 <= new_i < c.IMAGE_DIM and 0 <= new_j <  c.IMAGE_DIM :
                        image[0, :, j, i] = transformed[0, :, new_j, new_i]
        return image
