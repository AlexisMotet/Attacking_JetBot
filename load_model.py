import torch
import torchvision

def load_model(path_dataset, path_model):
    
    dataset = torchvision.datasets.ImageFolder(
        path_dataset,
        torchvision.transforms.Compose([
            torchvision.transforms.Resize((224, 224)),
            torchvision.transforms.ToTensor(),
        ])
    )
    """
    dataset = torchvision.datasets.ImageFolder(
        path_dataset,
        torchvision.transforms.Compose([
            torchvision.transforms.ColorJitter(0.1, 0.1, 0.1, 0.1),
            torchvision.transforms.Resize((224, 224)),
            torchvision.transforms.ToTensor(),
            torchvision.transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ])
    )
    """
    ratio = 2/3
    n_train = int(ratio * len(dataset))
    n_test = len(dataset) - n_train

    train_dataset, test_dataset = torch.utils.data.random_split(dataset, [n_train, n_test])
    
    ratio = 9/10
    n_train = int(ratio * len(train_dataset))
    n_validation = len(train_dataset) - n_train
    
    train_dataset, validation_dataset = torch.utils.data.random_split(train_dataset, [n_train, n_validation])

    train_loader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=1,
        shuffle=True,
        num_workers=0
    )

    test_loader = torch.utils.data.DataLoader(
        test_dataset,
        batch_size=1,
        shuffle=True,
        num_workers=0
    )
    
    validation_loader = torch.utils.data.DataLoader(
        validation_dataset,
        batch_size=1,
        shuffle=True,
        num_workers=0
    )

    model = torchvision.models.alexnet(pretrained=False) # pretrained false ?
    model.classifier[6] = torch.nn.Linear(model.classifier[6].in_features, 2)
    model.load_state_dict(torch.load(path_model, map_location=torch.device('cpu')))
    
    return model, train_loader, validation_loader, test_loader


if __name__=='__main__' :
    path_dataset = 'U:\\PROJET_3A\\projet_BONTEMPS_SCHAMPHELEIRE\\Project Adverserial Patch\\Collision Avoidance\\dataset'
    path_model = 'U:\\PROJET_3A\\projet_BONTEMPS_SCHAMPHELEIRE\\Project Adverserial Patch\\Collision Avoidance\\best_model_extended.pth'
    load_model(path_dataset, path_model)
