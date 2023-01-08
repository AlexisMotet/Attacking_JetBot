import torch
import numpy as np
torch.autograd.set_detect_anomaly(True)

class TotalVariationModule(torch.nn.Module):
    def __init__(self):
        super(TotalVariationModule, self).__init__()

    def forward(self, input):
        tv_row = input[:, :, 1:, :] - input[:, :, :-1, :]
        tv_col = input[:, :, :, 1:] - input[:, :, :, :-1]

        sqrt = torch.sqrt(tv_row[:, :, :, :-1] ** 2 + tv_col[:, :, :-1, :] ** 2 + 1e-5)

        tv_loss = torch.sum(sqrt)
        return tv_loss
    