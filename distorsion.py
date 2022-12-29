import ctypes
import numpy as np
import calibration.distorsion
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

class DistorsionTool():
    def __init__(self, path_calibration) :
        lib = ctypes.cdll.LoadLibrary("./distorsion.so")
        self.cdistort = lib.cdistort
        self.cdistort.restype = None
        self.cdistort.argtypes = [np.ctypeslib.ndpointer(ctypes.c_float, flags="C_CONTIGUOUS"),
                        ctypes.c_size_t,
                        ctypes.c_size_t,
                        ctypes.POINTER(CamMtx),
                        ctypes.POINTER(DistCoefs),
                        np.ctypeslib.ndpointer(ctypes.c_uint32, flags="C_CONTIGUOUS"),
                        np.ctypeslib.ndpointer(ctypes.c_float, flags="C_CONTIGUOUS")]

        self.cdistort_with_map = lib.cdistort_with_map
        self.cdistort_with_map.restype = None
        self.cdistort_with_map.argtypes = [np.ctypeslib.ndpointer(ctypes.c_float, flags="C_CONTIGUOUS"),
                        ctypes.c_size_t,
                        ctypes.c_size_t,
                        np.ctypeslib.ndpointer(ctypes.c_uint32, flags="C_CONTIGUOUS"),
                        np.ctypeslib.ndpointer(ctypes.c_float, flags="C_CONTIGUOUS")]

        self.cundistort = lib.cundistort
        self.cundistort.restype = None
        self.cundistort.argtypes = [np.ctypeslib.ndpointer(ctypes.c_float, flags="C_CONTIGUOUS"),
                        ctypes.c_size_t,
                        ctypes.c_size_t,
                        np.ctypeslib.ndpointer(ctypes.c_uint32, flags="C_CONTIGUOUS"),
                        np.ctypeslib.ndpointer(ctypes.c_float, flags="C_CONTIGUOUS")]

        cam_mtx, dist_coefs = calibration.distorsion.load_coef(path_calibration)
    
        self.mtx = CamMtx()
        self.mtx.fx, self.mtx.fy, self.mtx.cx, self.mtx.cy = cam_mtx[0][0], cam_mtx[1][1], \
                                                             cam_mtx[0][2], cam_mtx[1][2]
        self.coefs = DistCoefs()
        self.coefs.k1, self.coefs.k2, self.coefs.k3 = dist_coefs[0], dist_coefs[1], dist_coefs[4]
    
    def distort(self, image):
        image = image.numpy()
        image_distorded = np.zeros_like(image, dtype=np.single)
        row, col = image.shape[2],  image.shape[3]
        map = np.zeros((row, col), dtype=np.uint32)
        self.cdistort(image, row, col, self.mtx, self.coefs, map, 
                      image_distorded)
        return torch.tensor(image_distorded), map

    def distort_with_map(self, image, map):
        image_distorded = np.zeros_like(image)
        row, col = image.shape[2],  image.shape[3]
        self.cdistort_with_map(image.numpy(), row, col, map, image_distorded)
        return torch.tensor(image_distorded)

    def undistort(self, image_distorded, map, image):
        row, col = image.shape[2],  image.shape[3]
        self.cundistort(image_distorded.numpy(), row, col, map, image.numpy())
        return image

"""
if __name__=="__main__" :
    import time
    import torch
    import new_patch
    import matplotlib.pyplot as plt
    import calibration.distorsion
    def tensor_to_numpy_array(tensor):
        tensor = torch.squeeze(tensor)
        array = tensor.detach().cpu().numpy()
        return np.transpose(array, (1, 2, 0))
    path_model = 'U:\\PROJET_3A\\projet_NOUINOU_MOTET\\imagenette2-160_model.pth'
    path_dataset = 'U:\\PROJET_3A\\imagenette2-160\\train'
    path_calibration = 'U:\\PROJET_3A\\projet_NOUINOU_MOTET\\\calibration\\'

    patch_trainer = new_patch.PatchTrainer(path_model, path_dataset, path_calibration)
    _, _, empty_with_patch = patch_trainer.random_transform()

    cdistort, cundistort, mtx, coefs = load_everything(path_calibration)
    _, (ax1, ax2, ax3, ax4, ax5) = plt.subplots(1, 5)

    t1 = time.time()

    ax1.imshow(tensor_to_numpy_array(empty_with_patch), interpolation='nearest')
    empty_with_patch_distorded, map = distort_patch(cdistort, mtx, coefs, empty_with_patch)
    ax2.imshow(tensor_to_numpy_array(empty_with_patch_distorded), interpolation='nearest')

    empty_with_patch = undistort_patch(cundistort, empty_with_patch_distorded, map, empty_with_patch)
    ax3.imshow(tensor_to_numpy_array(empty_with_patch), interpolation='nearest')
    
    t2 = time.time()
    print(t2 - t1)

    cam_mtx, dist_coefs = calibration.distorsion.load_coef(path_calibration)
    print(empty_with_patch.shape)
    t1 = time.time()
    empty_with_patch_distorded, map_ = calibration.distorsion.distort_patch(cam_mtx, dist_coefs, empty_with_patch)
    ax4.imshow(tensor_to_numpy_array(empty_with_patch_distorded), interpolation='nearest')

    empty_with_patch = calibration.distorsion.undistort_patch(empty_with_patch, empty_with_patch_distorded, map_)
    ax5.imshow(tensor_to_numpy_array(empty_with_patch), interpolation='nearest')
    t2 = time.time()
    print(t2 - t1)
    plt.show()
"""