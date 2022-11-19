import torch

def unnormalize(normalized_img) :
    #https://sparrow.dev/pytorch-normalize/#:~:text=The%20Normalize()%20transform&text=In%20PyTorch%2C%20you%20can%20normalize,by%20the%20channel%20standard%20deviation.
    mean = torch.tensor([0.485, 0.456, 0.406])
    std = torch.tensor([0.229, 0.224, 0.225])
    # None = np.newaxis
    return normalized_img * std[:, None, None] + mean[:, None, None]