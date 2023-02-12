import torch
import numpy as np
import constants.constants as c

class PrintabilityModule(torch.nn.Module):
    def __init__(self, patch_dim):
        super().__init__()
        _colors = np.loadtxt("printable_colors.txt", delimiter=";")
        _colors = _colors/255
        ones = np.ones((1, 3, patch_dim, patch_dim))
        self.colors = torch.from_numpy((ones.T * _colors.T).T)
        
    def forward(self, patch):
        delta = patch - self.colors[:, np.newaxis, :, :, :]
        delta = torch.squeeze(delta)
        abs = torch.abs(delta)
        min, _ = torch.min(abs, dim=0)
        score = torch.sum(min)
        return score
    
    