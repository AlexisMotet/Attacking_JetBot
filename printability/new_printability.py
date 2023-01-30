import torch
import numpy as np
import constants.constants as c

class PrintabilityModule(torch.nn.Module):
    def __init__(self):
        super(PrintabilityModule, self).__init__()
        _colors = np.loadtxt(c.PATH_PRINTABLE_COLORS, delimiter=";")
        _colors = _colors/255
        ones = np.ones((1, 3, c.IMAGE_DIM, c.IMAGE_DIM))
        self.colors = torch.from_numpy((ones.T * _colors.T).T)
        
    def forward(self, image):
        delta = image - self.colors[:, np.newaxis, :, :, :]
        delta = torch.squeeze(delta)
        abs = torch.abs(delta)
        min, _ = torch.min(abs, dim=0)
        score = torch.sum(min)
        return score
    
    