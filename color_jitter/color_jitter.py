import torch
import constants.constants as c
import random

class ColorJitterModule(torch.nn.Module):
    def __init__(self):
        super(ColorJitterModule, self).__init__()
        self.jitter()
        
    def jitter(self):
        self.brightness = float(torch.empty(1).uniform_(-c.BRIGHTNESS_BIAS, 
                                                        c.BRIGHTNESS_BIAS))
        self.contrast = float(torch.empty(1).uniform_(1-c.CONTRAST_GAIN, 
                                                      1+c.CONTRAST_GAIN))
        self.order = random.getrandbits(1)
        
    def forward(self, image):
        if self.order : 
            brightness = image + self.brightness
            output = brightness * self.contrast
        else :
            contrast = image * self.contrast
            output = contrast + self.brightness
        return output