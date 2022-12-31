import torch

# https://github.com/mahmoods01/accessorize-to-a-crime

def total_variation(image):
    r_row = torch.roll(image, -1, dims=2)
    r_row[0, :, -1, :] = r_row[0, :, -2, :]
    r_col = torch.roll(image, -1, dims=3)
    r_col[0, :, :, -1] = r_col[0, :, :, -2]

    d_row = r_row - image
    d_col = r_col - image
    sqrt = torch.sqrt((d_row**2 + d_col**2))
    tv_loss = torch.sum(sqrt)

    reciprocal = torch.reciprocal(torch.fmax(sqrt, 1e-5 * torch.ones_like(sqrt)))
    d_row *= reciprocal
    d_col *= reciprocal
    
    dr_row = torch.roll(d_row, 1, dims=2)
    dr_row[0, :, 0, :] = dr_row[0, :, 1, :]
    dr_col = torch.roll(d_col, 1, dims=3)
    dr_col[0, :, 0, :] = dr_col[0, :, 1, :]

    fdr_row = dr_row - d_row
    fdr_col = dr_col - d_col

    fdr_row[0, :, 0, :] = -d_row[0, :, 0, :]
    fdr_col[0, :, :, 0] = -d_col[0, :, :, 0]
    
    grad = fdr_row + fdr_col

    return tv_loss, grad

"""
if __name__ == "__main__" :
    x = np.array([[1, 2, 1, 5, 6], [3, 4, 1, 5, 6], [5, 6, 1, 5, 6]], dtype=np.float64)
    y = torch.zeros((1, 3, *x.shape))
    y[0, :, :, :] = torch.tensor(x)
    _, grad = total_variation(y)
"""
