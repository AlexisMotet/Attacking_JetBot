import numpy as np
import torch

def indexing_with_argmin(arr, argmin, axis):
    # https://stackoverflow.com/questions/46103044/index-n-dimensional-array-with-n-1-d-array
    """indexing_with_argmin(arr, arr.argmin(axis), axis) == arr.min(axis)"""
    new_shape = list(arr.shape)
    del new_shape[axis]

    grid = np.ogrid[tuple(map(slice, new_shape))]
    grid.insert(axis, argmin)

    return arr[tuple(grid)]

class PrintabilityTool():
    def __init__(self, path_printable_vals, image_dim):
        vals = np.loadtxt(path_printable_vals, delimiter=";")
        self.colors = np.ones((len(vals), image_dim, image_dim, 3))
        for i, val in enumerate(vals) :
            self.colors[i] *= val/255
        self.colors = np.transpose(self.colors, (0, 3, 1, 2))

    def score(self, image) :
        image = image.numpy()
        delta = image - self.colors
        abs = np.abs(delta)
        res = indexing_with_argmin(delta, np.argmin(abs, axis=0), axis=0)
        grad = np.sign(res)
        score = np.sum(np.min(abs, axis=0), axis=None)
        return score, torch.tensor(grad[np.newaxis, :])


if __name__=="__main__" :
    printability_tool = PrintabilityTool("printable_vals.dat", 50)
    im = torch.rand((1, 3, 50, 50))
    print(printability_tool.score(im))
