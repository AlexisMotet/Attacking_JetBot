import torch
import numpy as np
import torchvision
import constants.constants as consts

def tensor_to_array(tensor):
    tensor = torch.squeeze(tensor)
    array = tensor.detach().cpu().numpy()
    if len(array.shape) == 3:
        return np.transpose(array, (1, 2, 0))
    return array


def normalize_tensor(tensor):
    return tensor / torch.abs(torch.max(tensor))

def load_dataset(path_dataset):
    dataset = torchvision.datasets.ImageFolder(
        path_dataset,
        torchvision.transforms.Compose([
            # torchvision.transforms.ColorJitter(0.1, 0.1, 0.1, 0.1),
            torchvision.transforms.RandomHorizontalFlip(),
            torchvision.transforms.Resize(consts.RESIZE_DIM),
            torchvision.transforms.CenterCrop(consts.IMAGE_DIM),
            torchvision.transforms.ToTensor(),
        ])
    )

    n_train = int(consts.RATIO_TRAIN_TEST * len(dataset))
    n_test = len(dataset) - n_train
    train_dataset, test_dataset = torch.utils.data.random_split(dataset, [n_train, n_test])

    train_loader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=consts.BATCH_SIZE,
        shuffle=True,
    )

    test_loader = torch.utils.data.DataLoader(
        test_dataset,
        batch_size=consts.BATCH_SIZE,
        shuffle=True,
    )
    return train_loader, test_loader


def load_model(path_model, n_classes):
    model = torchvision.models.alexnet(pretrained=False)
    model.classifier[6] = torch.nn.Linear(model.classifier[6].in_features, n_classes)
    model.load_state_dict(torch.load(path_model, map_location=torch.device('cpu')))
    return model

if __name__ == "__main__":
    pass