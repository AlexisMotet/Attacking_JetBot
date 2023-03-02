import torch
import numpy as np
import constants.constants as c
import json
from PIL import ImageColor
class PrintabilityModule(torch.nn.Module):
    def __init__(self, patch_dim):
        super().__init__()
        with open(c.consts["PATH_PRINTABLE_COLORS"], "r") as js:
            _colors = np.array([ImageColor.getcolor(c, "RGB") for c in 
                                json.loads(js.read())["values"]])/255
        _colors = _colors[np.random.choice(range(len(_colors)), size=c.consts["N_COLORS"])] 
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
    
    
