import ctypes
import numpy as np
import torch


class CamMtx(ctypes.Structure):
    _fields_ = [
        ("fx", ctypes.c_float),
        ("fy", ctypes.c_float),
        ("cx", ctypes.c_float),
        ("cy", ctypes.c_float),
    ]

class DistCoefs(ctypes.Structure):
    _fields_ = [
        ("k1", ctypes.c_float),
        ("k2", ctypes.c_float),
        ("k3", ctypes.c_float),
    ]

class DistortionTool():
    def __init__(self, path_calibration, path_distortion):
        lib = ctypes.cdll.LoadLibrary(path_distortion)
        self.c_distort = lib.cdistort
        self.c_distort.restype = None
        self.c_distort.argtypes = [
            np.ctypeslib.ndpointer(ctypes.c_float, flags="C_CONTIGUOUS"),
            ctypes.c_size_t,
            ctypes.c_size_t,
            ctypes.POINTER(CamMtx),
            ctypes.POINTER(DistCoefs),
            np.ctypeslib.ndpointer(ctypes.c_uint32, flags="C_CONTIGUOUS"),
            np.ctypeslib.ndpointer(ctypes.c_float, flags="C_CONTIGUOUS")
        ]

        self.c_distort_with_map = lib.cdistort_with_map
        self.c_distort_with_map.restype = None
        self.c_distort_with_map.argtypes = [
            np.ctypeslib.ndpointer(ctypes.c_float, flags="C_CONTIGUOUS"),
            ctypes.c_size_t,
            ctypes.c_size_t,
            np.ctypeslib.ndpointer(ctypes.c_uint32, flags="C_CONTIGUOUS"),
            np.ctypeslib.ndpointer(ctypes.c_float, flags="C_CONTIGUOUS")
            ]

        self.c_undistort = lib.cundistort
        self.c_undistort.restype = None
        self.c_undistort.argtypes = [
            np.ctypeslib.ndpointer(ctypes.c_float, flags="C_CONTIGUOUS"),
            ctypes.c_size_t,
            ctypes.c_size_t,
            np.ctypeslib.ndpointer(ctypes.c_uint32, flags="C_CONTIGUOUS"),
            np.ctypeslib.ndpointer(ctypes.c_float, flags="C_CONTIGUOUS")
            ]

        cam_mtx = np.loadtxt(path_calibration + 'camera_matrix.txt', 
                             dtype=float)
        dist_coefs = np.loadtxt(path_calibration + 'distortion_coefficients.txt', 
                                dtype=float)

        self.mtx = CamMtx()
        self.mtx.fx, self.mtx.fy = cam_mtx[0][0], cam_mtx[1][1]
        self.mtx.cx, self.mtx.cy = cam_mtx[0][2], cam_mtx[1][2]
        self.coefs = DistCoefs()
        self.coefs.k1 = dist_coefs[0]
        self.coefs.k2 = dist_coefs[1]
        self.coefs.k3 = dist_coefs[4]
        
    def distort(self, image):
        image_array = image.numpy()
        image_distorted = np.zeros_like(image_array)
        row, col = image_array.shape[2:]
        map_ = np.zeros((row, col), dtype=np.uint32)
        self.c_distort(image_array, row, col, self.mtx, self.coefs, map_,
                       image_distorted)
        return torch.from_numpy(image_distorted), map_

    def distort_with_map(self, image, map_):
        image_array = image.numpy()
        image_distorted = np.zeros_like(image_array)
        row, col = image_array.shape[2:]
        self.c_distort_with_map(image_array, row, col, map_, image_distorted)
        return torch.from_numpy(image_distorted)

    def undistort(self, image_distorted, map_, image):
        row, col = image.shape[2:]
        self.c_undistort(image_distorted.numpy(), row, col, map_, 
                         image.numpy())
        return image