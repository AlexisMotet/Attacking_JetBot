import torch
from torchvision.transforms import Normalize
from torchvision.transforms.functional import gaussian_blur
import constants.constants as c
import random

class IntrisicModule(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.jitter()
        
    def jitter(self):
        self.sigma_blur = float(torch.rand(1) * c.BLUR_SIGMA_MAX + 1e-5)
        self.noise = c.NOISE_INTENSITY * torch.randn(1, 3, c.IMAGE_DIM, c.IMAGE_DIM)
        self.order = random.getrandbits(1)
        
    def forward(self, image):
        if self.order : 
            blurred = gaussian_blur(image, kernel_size=c.BLUR_KERNEL_SIZE, 
                                    sigma=self.sigma_blur)
            modified = blurred + self.noise
        else :
            noisy = image + self.noise
            modified = gaussian_blur(noisy, kernel_size=c.BLUR_KERNEL_SIZE, 
                                    sigma=self.sigma_blur)
        return modified


class ExtrinsicModule(torch.nn.Module):
    def __init__(self, normalize=True):
        super().__init__()
        self.jitter()
        if normalize :
            self.normalize = Normalize(c.MEAN, c.STD)
        else :
            self.normalize = lambda x: x
        
    def jitter(self):
        self.brightness = float(torch.empty(1).uniform_(-c.BRIGHTNESS_BIAS, 
                                                         c.BRIGHTNESS_BIAS))
        self.contrast = float(torch.empty(1).uniform_(1-c.CONTRAST_GAIN, 
                                                      1+c.CONTRAST_GAIN))
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