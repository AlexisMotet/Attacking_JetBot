import sys
import torch
import numpy as np
import torchvision
import constants.constants as consts
from IPython.display import clear_output

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
            torchvision.transforms.RandomHorizontalFlip(),
            torchvision.transforms.Resize(consts.RESIZE_DIM),
            torchvision.transforms.CenterCrop(consts.IMAGE_DIM),
            torchvision.transforms.ToTensor(),
        ])
    )

    n_train = int(consts.RATIO_TRAIN_TEST * len(dataset))
    n_test = len(dataset) - n_train
    train_dataset, test_dataset = torch.utils.data.random_split(dataset, 
                                                                [n_train, n_test])

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

class Attribute():
    def __init__(self, name):
        self.name = name
    
    def get_attribute(self, patch_trainer) :
        return getattr(patch_trainer, self.name)
    
    def get_tuple(self, patch_trainer):
        return (self.name, self.get_attribute(patch_trainer))

class PrettyPrinter():
    def __init__(self, trainer, notebook=True):
        self.last_len = None
        self.saved = None
        self.trainer = trainer
        self.attributes = (Attribute("date"),
                            Attribute("path_model"),
                            Attribute("path_dataset"),
                            Attribute("limit_train_epoch_len"),
                            Attribute("limit_test_len"),
                            Attribute("target_class"),
                            Attribute("patch_relative_size"),
                            Attribute("jitter"),
                            Attribute("distort"),
                            Attribute("n_epochs"),
                            Attribute("mode"),
                            Attribute("random_mode"),
                            Attribute("lambda_tv"),
                            Attribute("lambda_print"),
                            Attribute("threshold"),
                            Attribute("max_iterations"))
        self.notebook = notebook

    def training(self):
        print("================ TRAINING ================")
        for i, attribute in enumerate(self.attributes) :
            name, val = attribute.get_tuple(self.trainer)
            if i%2 == 0 : txt = "%s=%s" % (name, val)
            else :
                txt += " || " + "%s=%s" % (name, val)
                print(txt)
        if len(self.attributes) % 2 != 0 : print("%s=%s" % (name, val))
        print("==========================================")
        
    def update_test(self, epoch, total, success_rate):
        txt = "[TEST] Epoch %2d - Image %3d - Success rate %1.3f%%" % (epoch, total, success_rate)
        if len(txt) != self.last_len : self.clear()
        print(txt, end="\r")
        self.last_len = len(txt)

    def update_image(self, epoch, total, success_rate):
        self.saved = "[TRAINING] Epoch %2d - Image %3d - Success rate of training on epoch %1.3f%%" % (epoch, total, success_rate)

    def update_iteration(self, i, target_proba):
        assert self.saved
        txt = "%s - [ATTACK] Gradient descent iteration %2d - Target probability %1.3f" % (self.saved, i, target_proba)
        if len(txt) != self.last_len : self.clear()
        print(txt, end="\r")
        self.last_len = len(txt)

    def clear(self):
        if self.last_len:
            spaces = " " * self.last_len
            print(spaces, end="\r")
    """
if __name__ == "__main__":
    import time
    pretty_printer = PrettyPrinter(None)
    pretty_printer.training()
    for i in range(10):
        pretty_printer.update_image(i,2)
        for j in range(10):
            pretty_printer.update_iteration(j)
            time.sleep(0.4)
    pretty_printer.test()
"""