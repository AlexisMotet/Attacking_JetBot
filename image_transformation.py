import torch
import torchvision

class ImageTransformer():
    def __init__(self):
        self.mean = torch.tensor([0.485, 0.456, 0.406])
        self.std = torch.tensor([0.229, 0.224, 0.225])
        self.normalization_transform = torchvision.transforms.Normalize(self.mean, self.std)
        self.color_jitter_transform = torchvision.transforms.ColorJitter(0.1, 0.1, 0.1, 0.1)
        
    def normalize(self, img):
        return self.normalization_transform(self.color_jitter_transform(img))

    def unnormalize(self, normalized_img) :
        #https://sparrow.dev/pytorch-normalize/#:~:text=The%20Normalize()%20transform&text=In%20PyTorch%2C%20you%20can%20normalize,by%20the%20channel%20standard%20deviation.
        # None = np.newaxis
        return normalized_img * self.std[:, None, None] + self.mean[:, None, None]