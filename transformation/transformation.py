import math
import numpy as np
import constants.constants as c
import torch
from torchvision.transforms.functional import rgb_to_grayscale
import sklearn.cluster

class TransformationTool():
    def __init__(self, patch_dim):
        self.patch_dim = patch_dim
        self.min_row = c.consts["IMAGE_DIM"]//2 - self.patch_dim//2
        self.max_row = c.consts["IMAGE_DIM"]//2 + self.patch_dim//2 - 1
        self.min_col = c.consts["IMAGE_DIM"]//2 - self.patch_dim//2
        self.max_col = c.consts["IMAGE_DIM"]//2 + self.patch_dim//2 - 1

        self.matrix_2dto3d = np.array([[1/c.consts["FX"], 0, -c.consts["CX"]/c.consts["FX"]],
                                       [0, 1/c.consts["FY"], -c.consts["CY"]/c.consts["FY"]],
                                       [0, 0, 0],
                                       [0, 0, 1]])
        
        self.matrix_3dto2d = np.array([[c.consts["FX"], 0, c.consts["CX"], 0],
                                       [0, c.consts["FY"], c.consts["CY"], 0],
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

        r = c.consts["EXTRINSIC_R"] @ rx @ ry @ rz
        matrix_translation = np.array([[1, 0, 0, x_t],
                                       [0, 1, 0, y_t],
                                       [0, 0, 1, 1/scale_factor],
                                       [0, 0, 0, 1]])
        
        matrix_transformation = matrix_translation @ r
        return matrix_transformation

    def _transform(self, patch, mtx_transfo):
        transformed = torch.zeros_like(patch)
        if torch.cuda.is_available():
            transformed = transformed.to(torch.device("cuda"))
        map_ = {}
        save = {}
        for i in range(self.min_row, self.max_row + 1):
            for j in range(self.min_col, self.max_col + 1):
                point_2d = np.array([i, j, 1])
                point_3d = self.matrix_2dto3d@point_2d
                point_transformed = mtx_transfo@point_3d
                ti, tj = point_transformed[:2]
                
                r2 = pow(ti, 2) + pow(tj, 2)
                d = 1 + c.consts["K1"]*r2 + c.consts["K2"]*r2**2
                di = ti * d
                dj = tj * d
                point_distorted = point_transformed.copy()
                point_distorted[:2] = di, dj
                
                point_projected = self.matrix_3dto2d@point_distorted
                if point_projected[-1] == 0 :
                    continue
                ni_float, nj_float = point_projected[:2]/point_projected[-1]
                    
                ni = int(round(ni_float))
                nj = int(round(nj_float))
                
                if 0 <= ni < c.consts["IMAGE_DIM"] and 0 <= nj < c.consts["IMAGE_DIM"] :
                    d = math.sqrt((ni - ni_float)**2 + 
                                  (nj - nj_float)**2)
                    weight = (2 * math.sqrt(0.5) - d)/(2 * math.sqrt(0.5))
                    map_[(j, i)] = {"ncoords" : (nj, ni), "weight" : weight}
                    if (nj, ni) not in save:
                        save[(nj, ni)] = {"values" : [], 
                                          "weights" : []}
                    save[(nj, ni)]["values"].append(weight * patch[0, :, j, i])
                    save[(nj, ni)]["weights"].append(weight)
                    transformed[0, :, nj, ni] = sum(save[(nj, ni)]["values"])/ \
                                                sum(save[(nj, ni)]["weights"])
        return transformed, map_
    
    def random_transform(self, patch):
        scale_factor = np.random.uniform(c.consts["SCALE_FACTOR_MIN"], 1)

        x_t = np.random.randint(c.consts["X_TOP_LEFT"], c.consts["X_BOTTOM_RIGHT"])
        y_t = np.random.randint(c.consts["Y_TOP_LEFT"], c.consts["Y_BOTTOM_RIGHT"])
        
        point_t = (1/scale_factor) * np.array([x_t, y_t, 1])
        nx_t, ny_t = (self.matrix_2dto3d@point_t)[:2]
        angles = np.zeros(3)
        angles[np.random.randint(2)] = np.random.uniform(-c.consts["ANGLES_RANGE"], 
                                                          c.consts["ANGLES_RANGE"])
        mtx_transfo = self._get_matrix_transformation(nx_t, ny_t, scale_factor, *angles)
        return self._transform(patch, mtx_transfo)
    
            
    def undo_transform(self, patch, transformed, map_):
        new_patch = patch.clone()
        for i in range(self.min_row, self.max_row + 1):
            for j in range(self.min_col, self.max_col + 1):
                if (j, i) in map_ :
                    nj, ni = map_[(j, i)]["ncoords"]
                    w = map_[(j, i)]["weight"]
                    if 0 <= ni < c.consts["IMAGE_DIM"] and 0 <= nj <  c.consts["IMAGE_DIM"] :
                        new_patch[0, :, j, i] = ((1 - w) * new_patch[0, :, j, i] + 
                                                 transformed[0, :, nj, ni])/(2 - w)
        return new_patch
