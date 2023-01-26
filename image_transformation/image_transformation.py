import torch
import torchvision
import constants.constants as c
import random


class ImageTransformationModule(torch.nn.Module):
    def __init__(self):
        super(ImageTransformationModule, self).__init__()
        self.jitter()
        self.normalize = torchvision.transforms.Normalize(c.MEAN, c.STD)
        
    def jitter(self):
        self.sigma_blur = float(torch.rand(c.BLUR_SIGMA_MAX) + 1e-5)
        self.noise = torch.empty(1, 3, c.IMAGE_DIM, c.IMAGE_DIM)

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
        blurred = torchvision.transforms.functional.gaussian_blur(modified, kernel_size=c.BLUR_KERNEL_SIZE, sigma=self.sigma_blur)
        noisy = blurred + self.noise
        return self.normalize(noisy)