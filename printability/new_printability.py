import torch
import numpy as np
import constants.constants as c

class PrintabilityModule(torch.nn.Module):
    def __init__(self, patch_dim):
        super().__init__()
        _colors = np.loadtxt(c.consts["PATH_PRINTABLE_COLORS"], delimiter=";")
        _colors = _colors/255
        ones = np.ones((1, 3, patch_dim, patch_dim))
        self.colors = torch.from_numpy((ones.T * _colors.T).T)
        if torch.cuda.is_available():
            self.colors = self.colors.to(torch.device("cuda"))
        
    def forward(self, patch):
        delta = patch - self.colors[:, np.newaxis, :, :, :]
        delta = torch.squeeze(delta)
        abs = torch.abs(delta)
        min, _ = torch.min(abs, dim=0)
        score = torch.sum(min)
        return score
    
    
