import torch
import torchvision
import numpy as np
import datetime
import distortion.distortion as distortion
import total_variation.total_variation as total_variation
import printability.printability as printability
import pickle
import sklearn.cluster

resize_dim = 256
image_dim = 224
batch_size = 1
ratio_train_test = 2/3
mean = [0.485, 0.456, 0.406]
std = [0.229, 0.224, 0.225]
limit_train_len = 150
limit_test_len = 80
n_enreg_img = 10
path_image_folder = "C:\\Users\\alexi\\PROJET_3A\\projet_3A\\images\\"


def load_dataset(path_dataset):
    dataset = torchvision.datasets.ImageFolder(
        path_dataset,
        torchvision.transforms.Compose([
            torchvision.transforms.RandomHorizontalFlip(),
            torchvision.transforms.Resize(resize_dim),
            torchvision.transforms.CenterCrop(image_dim),
            torchvision.transforms.ToTensor(),
        ])
    )

    n_train = int(ratio_train_test * len(dataset))
    n_test = len(dataset) - n_train
    train_dataset, test_dataset = torch.utils.data.random_split(dataset, [n_train, n_test])

    train_loader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
    )
    
    test_loader = torch.utils.data.DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=True,
    )
    return train_loader, test_loader

def load_model(path_model, n_classes):
    model = torchvision.models.alexnet(pretrained=False)
    model.classifier[6] = torch.nn.Linear(model.classifier[6].in_features, n_classes)
    model.load_state_dict(torch.load(path_model, map_location=torch.device('cpu')))
    return model


class PatchTrainer():
    def __init__(self, path_model, path_dataset, path_calibration, path_distortion,
                 path_printable_vals, n_classes=2, target_class=1, patch_relative_size=0.05, 
                 distort=True, n_epochs=1, lambda_tv=0.0005, lambda_print=0.005, 
                 threshold=0.9, max_iterations=10):

        self.date = datetime.datetime.now().strftime("%d/%m/%Y %H:%M:%S")

        self.path_model = path_model
        self.path_dataset = path_dataset
        self.path_calibration = path_calibration
        self.path_distortion = path_distortion
        self.path_printable_vals = path_printable_vals
        self.n_classes = n_classes
        self.target_class = target_class
        self.patch_relative_size = patch_relative_size
        self.distort = distort
        self.n_epochs = n_epochs
        self.lambda_tv = lambda_tv
        self.lambda_print = lambda_print
        self.threshold = threshold
        self.max_iterations = max_iterations
        
        self.model = load_model(self.path_model, n_classes=self.n_classes)
        self.train_loader, self.test_loader = load_dataset(self.path_dataset)
        
        image_size = image_dim**2
        patch_size = image_size * self.patch_relative_size
        self.patch_dim = int(patch_size**(0.5))

        self.normalize = torchvision.transforms.Normalize(mean=mean, std=std)
        self.grad_normalization = torchvision.transforms.Normalize([0, 0, 0], std)
        
        if (self.distort) :
            self.distortion_tool = distortion.DistortionTool(self.path_calibration, self.path_distortion)

        self.printability_tool = printability.PrintabilityTool(self.path_printable_vals, 
                                                               self.patch_dim)
        
        self.patch = self.random_patch_init()
        
        self.kMeans = sklearn.cluster.KMeans()
        self.weights = torch.ones_like(self.patch)

        self.target_proba_train = {}
        self.target_proba_test = []
        
    def random_patch_init(self):
        patch = torch.empty(1, 3, self.patch_dim, self.patch_dim)
        for i, (m, s) in enumerate(zip(mean, std)) :
            patch[0, i, :, :].normal_(mean=m, std=s)
        return patch

    def test_model(self) :
        self.model.eval()
        success, total = 0, 0
        for loader in [self.train_loader ,self.test_loader] : 
            for image, label in loader: 
                output = self.model(self.normalize(image))
                success += (label == output.argmax(1)).sum()
                total += len(label)
                print('sucess/total : %d/%d accuracy : %f' % (success, total, (100 * success/float(total))))
    
    def image_attack(self, image, row0, col0):
        self.model.eval()
        c = 0

        empty_with_patch = torch.zeros(1, 3, image_dim, image_dim)
        empty_with_patch[0, :, row0:row0 + self.patch_dim, col0:col0 + self.patch_dim] = self.patch
        mask = self.get_mask(empty_with_patch)
        while True :
            adversarial_image = torch.mul((1-mask), image) + torch.mul(mask, empty_with_patch)
            adversarial_image =  self.normalize(adversarial_image)
            var_adversarial_image = torch.autograd.Variable(adversarial_image, requires_grad=True)
            vector_scores = self.model(var_adversarial_image)
            target_proba = torch.nn.functional.softmax(vector_scores, dim=1)[0, self.target_class].item()
            if (c == 0):
                first_target_proba = target_proba
            print('iteration : %d target proba : %f' % (c, target_proba))
            c += 1
            if target_proba >= self.threshold or c > self.max_iterations :
                break
            loss_target = -torch.nn.functional.log_softmax(vector_scores, dim=1)[0, self.target_class]
            loss_target.backward()
            grad = var_adversarial_image.grad.clone()
            # var_adversarial_image.grad.data.zero_()

            grad = self.grad_normalization(grad)
            empty_with_patch -= grad
            
            _, tv_grad = total_variation.total_variation(self.patch)
            _, print_grad = self.printability_tool.score(self.patch)

            self.patch = torch.Tensor.contiguous(empty_with_patch[:, :, row0:row0 + self.patch_dim, col0:col0 + self.patch_dim])

            self.patch -= self.lambda_tv * tv_grad + self.lambda_print * print_grad

            empty_with_patch = torch.zeros(1, 3, image_dim, image_dim)
            empty_with_patch[0, :, row0:row0 + self.patch_dim, col0:col0 + self.patch_dim] = self.patch 
        return first_target_proba, adversarial_image, empty_with_patch

    def image_attack_with_distortion(self, image, row0, col0):
        self.model.eval()
        c = 0

        empty_with_patch = torch.zeros(1, 3, image_dim, image_dim)
        empty_with_patch[0, :, row0:row0 + self.patch_dim, col0:col0 + self.patch_dim] = self.patch

        # empty_with_patch, matrix = rotation_tool.random_rotation(empty_with_patch)
        empty_with_patch_distorded, map = self.distortion_tool.distort(empty_with_patch)
        mask = self.get_mask(empty_with_patch_distorded)

        while True :
            adversarial_image = torch.mul((1-mask), image) + torch.mul(mask, empty_with_patch_distorded)
            adversarial_image =  self.normalize(adversarial_image)
            var_adversarial_image = torch.autograd.Variable(adversarial_image, requires_grad=True)
            vector_scores = self.model(var_adversarial_image)
            target_proba = torch.nn.functional.softmax(vector_scores, dim=1)[0, self.target_class].item()
            if (c == 0):
                first_target_proba = target_proba
                print('first target proba : %f' % first_target_proba)
            else :
                print('iteration : %d target proba : %f' % (c, target_proba))
            c += 1
            if target_proba >= self.threshold or c > self.max_iterations :
                break
            
            loss = -torch.nn.functional.log_softmax(vector_scores, dim=1)
            loss_target = loss[0, self.target_class]
            loss_target.backward()
            grad = var_adversarial_image.grad.clone()

            grad = self.grad_normalization(grad)
            empty_with_patch_distorded -= grad
            
            _, tv_grad = total_variation.total_variation(self.patch)
            _, print_grad = self.printability_tool.score(self.patch)
        
            empty_with_patch = self.distortion_tool.undistort(empty_with_patch_distorded, map, 
                                                              empty_with_patch)

            # empty_with_patch = rotation_tool.undo_rotation(empty_with_patch, matrix)
            
            self.patch = torch.Tensor.contiguous(empty_with_patch[:, :, row0:row0 + self.patch_dim, col0:col0 + self.patch_dim])

            self.patch -= self.lambda_tv * tv_grad + self.lambda_print * print_grad

            empty_with_patch = torch.zeros(1, 3, image_dim, image_dim)
            empty_with_patch[0, :, row0:row0 + self.patch_dim, col0:col0 + self.patch_dim] = self.patch 
            empty_with_patch_distorded = self.distortion_tool.distort_with_map(empty_with_patch, map)
        return first_target_proba, adversarial_image, empty_with_patch

    def random_transform(self):
        row0, col0 = np.random.choice(image_dim - self.patch_dim, size=2)
        empty_with_patch = torch.zeros(1, 3, image_dim, image_dim)
        empty_with_patch[0, :, row0:row0 + self.patch_dim, col0:col0 + self.patch_dim] = self.patch
        return empty_with_patch, row0, col0
    
    def get_mask(self, image) :
        mask = torch.zeros_like(image)
        mask[image != 0] = 1
        return mask
    
    def find_patch_position(self, grad):
        conv = torch.nn.functional.conv2d(torch.abs(grad), self.weights)
        conv = torch.squeeze(conv).numpy()
        conv = conv/np.max(conv)
        binary = np.where(conv < 0.5, 0, 1)
        X  = np.transpose(binary.nonzero())
        self.kMeans.fit(X)
        row, col = self.kMeans.cluster_centers_[np.random.randint(
            len(self.kMeans.cluster_centers_))]
        row0 = int(max(row - self.patch_dim, 0))
        col0 = int(max(col - self.patch_dim/2, 0))
        return row0, col0
        
    def test(self):
        total, success = 0, 0
        for image, true_label in self.test_loader :
            var_image = torch.autograd.Variable(image, requires_grad=True)
            vector_scores = self.model(var_image)
            model_label = torch.argmax(vector_scores.data).item()
            if model_label is not true_label.item() or model_label is self.target_class :
                continue
            total += 1
            
            loss_model = -torch.nn.functional.log_softmax(vector_scores, 
                                                          dim=1)[0, model_label]
            loss_model.backward()
            row0, col0 = self.find_patch_position(var_image.grad.clone())
            
            empty_with_patch = torch.zeros(1, 3, image_dim, image_dim)
            empty_with_patch[0, :, row0:row0 + self.patch_dim, 
                             col0:col0 + self.patch_dim] = self.patch
            
            if (self.distort) :
                empty_with_patch_distorded, _ = self.distortion_tool.distort(empty_with_patch)
                mask = self.get_mask(empty_with_patch_distorded)
                adversarial_image = torch.mul((1-mask), image) + torch.mul(mask, empty_with_patch_distorded)
            else :
                mask = self.get_mask(empty_with_patch)
                adversarial_image = torch.mul((1-mask), image) + torch.mul(mask, empty_with_patch)

            vector_scores = self.model(self.normalize(adversarial_image))
            target_proba = torch.nn.functional.softmax(vector_scores, dim=1)[0, self.target_class].item()
            adversarial_label = torch.argmax(vector_scores.data).item()
            if adversarial_label == self.target_class:
                success += 1
            
            if (total % n_enreg_img == 0):
                torchvision.utils.save_image(adversarial_image.data, path_image_folder 
                                            + 'test_target_proba_%f_label_%d.png' 
                                            % (target_proba, adversarial_label))

            self.target_proba_test.append(target_proba)
            print('sucess/total : %d/%d accuracy : %f' % (success, total, (100 * success/float(total))))
            if limit_train_len is not None and total >= limit_test_len :
                break
        self.test_success_rate = 100 * (success/float(total))
        
            
    def train(self):
        self.model.eval()
        for epoch in range(self.n_epochs) :
            total = 0
            self.target_proba_train[epoch] = []
            for image, true_label in self.train_loader :
                var_image = torch.autograd.Variable(image, requires_grad=True)
                vector_scores = self.model(var_image)
                model_label = torch.argmax(vector_scores.data).item()
                if model_label is not true_label.item() or model_label is self.target_class  :
                    continue
                total += 1
                
                loss_model = -torch.nn.functional.log_softmax(vector_scores, dim=1)[0, model_label]
                loss_model.backward()
                
                row0, col0 = self.find_patch_position(var_image.grad.clone())
                
                if (self.distort) :
                    first_target_proba, adversarial_image, empty_with_patch = \
                        self.image_attack_with_distortion(image, row0, col0)
                else :
                    first_target_proba, adversarial_image, empty_with_patch = \
                        self.image_attack(image, row0, col0)

                if (total % n_enreg_img == 0):
                    torchvision.utils.save_image(image.data, path_image_folder 
                                                + 'epoch%d_image%d_label%d_original.png' 
                                                % (epoch, total, true_label.item()))
                    
                    torchvision.utils.save_image(empty_with_patch.data, path_image_folder 
                                                + 'epoch%d_image%d_empty_with_patch.png' 
                                                % (epoch, total))
                    
                    torchvision.utils.save_image(adversarial_image.data, path_image_folder 
                                                + 'epoch%d_image%d_adversarial.png' 
                                                % (epoch, total))
                    
                    torchvision.utils.save_image(self.patch.data, path_image_folder
                                                + 'epoch%d_image%d_patch.png' 
                                                % (epoch, total))
                    
                self.target_proba_train[epoch].append(first_target_proba)
                if limit_train_len is not None and total >= limit_train_len :
                    break

    def save_patch(self, path):
        self.distortion_tool = None
        pickle.dump(self, open(path, "wb"))


if __name__=="__main__":
    path_model = 'U:\PROJET_3A\projet_NOUINOU_MOTET\objet_model.pth'
    path_dataset = 'U:\\PROJET_3A\\dataset_objets'
    path_calibration = 'U:\\PROJET_3A\\projet_NOUINOU_MOTET\\calibration\\'
    path_printable_vals = 'U:\\PROJET_3A\\projet_NOUINOU_MOTET\\printable_vals.dat'
    patch_trainer = PatchTrainer(path_model, path_dataset, path_calibration, path_printable_vals, 
                                       n_classes=2, target_class=1, distort=False, patch_relative_size=0.01, n_epochs=2)
    patch_trainer.test_model()
