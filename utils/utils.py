import torch
import numpy as np


def tensor_to_array(tensor):
    tensor = torch.squeeze(tensor)
    array = tensor.detach().cpu().numpy()
    if len(array.shape) == 3:
        return np.transpose(array, (1, 2, 0))
    return array


def normalize_tensor(tensor):
    return tensor / torch.abs(torch.max(tensor))
