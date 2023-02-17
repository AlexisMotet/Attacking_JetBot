import torch
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
        self.brightness = float(torch.empty(1).uniform_(-c.consts["BRIGHTNESS_BIAS"], 
                                                         c.consts["BRIGHTNESS_BIAS"]))
        
        self.contrast = float(torch.empty(1).uniform_(1-c.consts["CONTRAST_GAIN"], 
                                                      1+c.consts["CONTRAST_GAIN"]))
        
        self.sigma_blur = float(torch.rand(1) * c.consts["BLUR_SIGMA_MAX"] + 1e-5)
        
        self.noise = u.array_to_tensor(c.consts["NOISE_STD"] * np.random.standard_normal((
                                                                c.consts["IMAGE_DIM"], 
                                                                c.consts["IMAGE_DIM"], 3)))
        if torch.cuda.is_available():
            self.noise = self.noise.to(torch.device("cuda"))
        
        self.order = random.getrandbits(1)
        
    def brightness_contrast(self, patch):
        if self.order : 
            brightness = patch + self.brightness
            modified = brightness * self.contrast
        else :
            contrast = patch * self.contrast
            modified = contrast + self.brightness
        return modified

    def blurring(self, patch):
        blurred = gaussian_blur(patch, kernel_size=c.consts["BLUR_KERNEL_SIZE"], 
                                sigma=self.sigma_blur)
        return blurred
    
    def noising(self, patch):
        noisy = patch + self.noise
        return noisy
        
    def forward(self, patch):
        modified = self.brightness_contrast(patch)
        blurred = self.blurring(modified)
        noisy = self.noising(blurred)
        return noisy