import torch
from torchvision.transforms import Normalize
from torchvision.transforms.functional import gaussian_blur
import constants.constants as c
import random

class PatchProcessingModule(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.jitter()
        
    def jitter(self):
        self.sigma_blur = float(torch.rand(1) * c.consts["BLUR_SIGMA_MAX"] + 1e-5)
        self.noise = c.consts["NOISE_INTENSITY"] * torch.randn(1, 3, c.consts["IMAGE_DIM"], 
                                                                     c.consts["IMAGE_DIM"])
        self.order = random.getrandbits(1)
        
    def forward(self, patch):
        if self.order : 
            blurred = gaussian_blur(patch, kernel_size=c.consts["BLUR_KERNEL_SIZE"], 
                                    sigma=self.sigma_blur)
            modified = blurred + self.noise
        else :
            noisy = patch + self.noise
            modified = gaussian_blur(noisy, kernel_size=c.consts["BLUR_KERNEL_SIZE"], 
                                    sigma=self.sigma_blur)
        return modified


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