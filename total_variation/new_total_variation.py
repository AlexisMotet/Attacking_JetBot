import torch

class TotalVariationModule(torch.nn.Module):
    def forward(self, patch):
        tv_row = patch[:, :, 1:, :] - patch[:, :, :-1, :]
        tv_col = patch[:, :, :, 1:] - patch[:, :, :, :-1]

        sqrt = torch.sqrt(tv_row[:, :, :, :-1] ** 2 + tv_col[:, :, :-1, :] ** 2 + 1e-5)

        tv_loss = torch.sum(sqrt)
        return tv_loss
    