import torch
import torchvision.transforms.functional as F
import constants.constants as c

class ColorJitterModule(torch.nn.Module):
    def __init__(self):
        super(ColorJitterModule, self).__init__()
        self.jitter()
        
    def jitter(self):
        self.brightness = float(torch.empty(1).uniform_(c.MIN_BRIGHTNESS, 
                                                        c.MAX_BRIGHTNESS))
        self.contrast = float(torch.empty(1).uniform_(c.MIN_CONTRAST, 
                                                      c.MAX_CONTRAST))
        self.saturation = float(torch.empty(1).uniform_(c.MIN_SATURATION, 
                                                        c.MAX_SATURATION))
        self.hue = float(torch.empty(1).uniform_(c.MIN_HUE, 
                                                 c.MAX_HUE))
        self.permutation = torch.randperm(4)
        
    def forward(self, image):
        for index in self.permutation :
            if index == 0: image = F.adjust_brightness(image, self.brightness)
            elif index == 1: image = F.adjust_contrast(image, self.contrast)
            elif index == 2: image = F.adjust_saturation(image, self.saturation)
            elif index == 3: image = F.adjust_hue(image, self.hue)
        return image