import torch
import numpy as np

class PrintabilityModule(torch.nn.Module):
    def __init__(self, path_printable_colors, image_dim):
        super(PrintabilityModule, self).__init__()
        self.colors = self.get_colors(path_printable_colors, image_dim)
    
    def forward(self, image):
        delta = image - self.colors[:, np.newaxis, :, :, :]
        delta = torch.squeeze(delta)
        abs = torch.abs(delta)
        min, _ = torch.min(abs, dim=0)
        score = torch.sum(min)
        return score
    
    def get_colors(self, path_printable_colors, image_dim):
        colors = np.loadtxt(path_printable_colors, delimiter=";")
        colors = colors/255
        ones = np.ones((1, 3, image_dim, image_dim))
        return torch.from_numpy((ones.T * colors.T).T)