import torch
from torchvision.transforms import Normalize
from torchvision.transforms.functional import gaussian_blur
import constants.constants as c
import random
import numpy as np
import utils.utils as u

class PatchProcessingModule(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.jitter()
        
    def jitter(self):
        self.sigma_blur = float(torch.rand(1) * c.consts["BLUR_SIGMA_MAX"] + 1e-5)
        self.noise = u.array_to_tensor(c.consts["NOISE_STD"] * np.random.standard_normal((
                                                                       c.consts["IMAGE_DIM"], 
                                                                       c.consts["IMAGE_DIM"], 3)))
        if torch.cuda.is_available():
            self.noise = self.noise.to(torch.device("cuda"))
    def forward(self, patch):
        blurred = gaussian_blur(patch, kernel_size=c.consts["BLUR_KERNEL_SIZE"], 
                                        sigma=self.sigma_blur)
        noisy = blurred + self.noise
        return noisy


class ImageProcessingModule(torch.nn.Module):
    def __init__(self, normalize=True):
        super().__init__()
        self.jitter()
        if normalize :
            self.normalize = Normalize(c.consts["MEAN"], c.consts["STD"])
        else :
            self.normalize = lambda x: x
        
    def jitter(self):
        self.brightness = float(torch.empty(1).uniform_(-c.consts["BRIGHTNESS_BIAS"], 
                                                         c.consts["BRIGHTNESS_BIAS"]))
        self.contrast = float(torch.empty(1).uniform_(1-c.consts["CONTRAST_GAIN"], 
                                                      1+c.consts["CONTRAST_GAIN"]))
        self.order = random.getrandbits(1)
        
    def forward(self, image):
        if self.order : 
            brightness = image + self.brightness
            modified = brightness * self.contrast
        else :
            contrast = image * self.contrast
            modified = contrast + self.brightness
        normalized = self.normalize(modified)
        return normalized