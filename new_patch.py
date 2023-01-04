import torch
import torchvision
import numpy as np
import datetime
import distortion.distortion as distortion
import total_variation.total_variation as total_variation
import printability.printability as printability
import pickle
import sklearn.cluster
import matplotlib.pyplot as plt
import utils.utils as utils

RESIZE_DIM = 256
IMAGE_DIM = 224
BATCH_SIZE = 1
RATIO_TRAIN_TEST = 2/3
MEAN = [0.485, 0.456, 0.406]
STD = [0.229, 0.224, 0.225]

limit_train_len = None
limit_test_len = None
n_enreg_img = 1
path_image_folder = "C:\\Users\\alexi\\PROJET_3A\\projet_3A\\images\\"

def load_dataset(path_dataset):
    dataset = torchvision.datasets.ImageFolder(
        path_dataset,
        torchvision.transforms.Compose([
            torchvision.transforms.ColorJitter(0.1, 0.1, 0.1, 0.1),
            torchvision.transforms.RandomHorizontalFlip(),
            torchvision.transforms.Resize(RESIZE_DIM),
            torchvision.transforms.CenterCrop(IMAGE_DIM),
            torchvision.transforms.ToTensor(),
        ])
    )

    n_train = int(RATIO_TRAIN_TEST * len(dataset))
    n_test = len(dataset) - n_train
    train_dataset, test_dataset = torch.utils.data.random_split(dataset, [n_train, n_test])

    train_loader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=BATCH_SIZE,
        shuffle=True,
    )
    
    test_loader = torch.utils.data.DataLoader(
        test_dataset,
        batch_size=BATCH_SIZE,
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
                 path_printable_vals, mode=0, random_mode=2, lambda_target = 0.9, 
                 lambda_flee = 0.1, validation=False, n_classes=2, target_class=1, 
                 class_to_flee=0, patch_relative_size=0.05, distort=True, n_epochs=1, 
                 lambda_tv=0.0005, lambda_print=0.005, threshold=0.9, max_iterations=10):

        self.date = datetime.datetime.now().strftime("%d/%m/%Y %H:%M:%S")

        self.path_model = path_model
        self.path_dataset = path_dataset
        self.path_calibration = path_calibration
        self.path_distortion = path_distortion
        self.path_printable_vals = path_printable_vals
        self.mode = mode
        self.random_mode = random_mode
        self.lambda_target = lambda_target
        self.lambda_flee = lambda_flee
        self.validation = validation
        self.n_classes = n_classes
        self.target_class = target_class
        self.class_to_flee = class_to_flee
        self.patch_relative_size = patch_relative_size
        self.distort = distort
        self.n_epochs = n_epochs
        self.lambda_tv = lambda_tv
        self.lambda_print = lambda_print
        self.threshold = threshold
        self.max_iterations = max_iterations
        
        self.model = load_model(self.path_model, n_classes=self.n_classes)
        self.train_loader, self.test_loader = load_dataset(self.path_dataset)
        
        image_size = IMAGE_DIM**2
        patch_size = image_size * self.patch_relative_size
        self.patch_dim = int(patch_size**(0.5))

        self.normalize = torchvision.transforms.Normalize(mean=MEAN,
                                                          std=STD)
        self.grad_normalization = torchvision.transforms.Normalize([0, 0, 0], 
                                                                   STD)
        
        if (self.distort) :
            self.distortion_tool = distortion.DistortionTool(self.path_calibration, self.path_distortion)

        self.printability_tool = printability.PrintabilityTool(self.path_printable_vals, 
                                                               self.patch_dim)
        
        self.patch = self.random_patch_init()
        
        if self.random_mode > 0 :
            self.kMeans = sklearn.cluster.KMeans(n_clusters=self.n_epochs)
            self.weights = torch.ones_like(self.patch)

        self.target_proba_test = {}
        self.success_rate_test = {}
        
    def random_patch_init(self):
        patch = torch.empty(1, 3, self.patch_dim, self.patch_dim)
        for i, (m, s) in enumerate(zip(MEAN, STD)) :
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
                print('sucess/total : %d/%d accuracy : %.2f' % (success, total, (100 * success/float(total))))
    
    def image_attack(self, image, row0, col0):
        c = 0

        empty_with_patch = torch.zeros(1, 3, IMAGE_DIM, IMAGE_DIM)
        empty_with_patch[0, :, row0:row0 + self.patch_dim, col0:col0 + self.patch_dim] = self.patch
        mask = self.get_mask(empty_with_patch)
        while True :
            adversarial_image = torch.mul((1-mask), image) + torch.mul(mask, empty_with_patch)
            adversarial_image = self.normalize(adversarial_image)
            adversarial_image.requires_grad = True
            vector_scores = self.model(adversarial_image)
            target_proba = torch.nn.functional.softmax(vector_scores, dim=1)[0, self.target_class].item()
            if c > 0 :
                print('iteration : %d target proba : %.2f' % (c, target_proba))
            c += 1
            if target_proba >= self.threshold or c > self.max_iterations :
                break
            
            loss = -torch.nn.functional.log_softmax(vector_scores, dim=1)
            if self.mode == 0 :
                loss_target = loss[0, self.target_class]
                loss_target.backward()
                adversarial_image.requires_grad = False
                target_grad = self.grad_normalization(adversarial_image.grad)
                target_grad = utils.normalize_tensor(target_grad)
                empty_with_patch -= self.lambda_target * target_grad
                del loss_target
            elif self.mode == 1 :
                loss_class_to_flee = loss[0, self.class_to_flee]
                loss_class_to_flee.backward()
                adversarial_image.requires_grad = False
                flee_grad = -self.grad_normalization(adversarial_image.grad)
                flee_grad = utils.normalize_tensor(flee_grad)
                empty_with_patch -= self.lambda_flee * flee_grad
                del loss_class_to_flee
            elif self.mode == 2 :
                loss_target = loss[0, self.target_class]
                loss_target.backward(retain_graph=True)
                with torch.no_grad() :
                    target_grad = self.grad_normalization(adversarial_image.grad)
                    target_grad = utils.normalize_tensor(target_grad)
                adversarial_image.grad.data.zero_()  
                loss_class_to_flee = loss[0, self.class_to_flee]
                loss_class_to_flee.backward()
                adversarial_image.requires_grad = False
                flee_grad = -self.grad_normalization(adversarial_image.grad)
                flee_grad = utils.normalize_tensor(flee_grad)
                empty_with_patch -= self.lambda_flee * flee_grad \
                    + self.lambda_target * target_grad
                del loss_target, loss_class_to_flee
            else :
                assert False
            del loss 
            
            _, tv_grad = total_variation.total_variation(self.patch)
            _, print_grad = self.printability_tool.score(self.patch)

            self.patch = empty_with_patch[:, :, row0:row0 + self.patch_dim, col0:col0 + self.patch_dim]

            self.patch -= self.lambda_tv * tv_grad + self.lambda_print * print_grad

            empty_with_patch = torch.zeros(1, 3, IMAGE_DIM, IMAGE_DIM)
            empty_with_patch[0, :, row0:row0 + self.patch_dim, col0:col0 + self.patch_dim] = self.patch 
        return adversarial_image, empty_with_patch

    def image_attack_with_distortion(self, image, row0, col0):
        c = 0

        empty_with_patch = torch.zeros(1, 3, IMAGE_DIM, IMAGE_DIM)
        empty_with_patch[0, :, row0:row0 + self.patch_dim, col0:col0 + self.patch_dim] = self.patch

        empty_with_patch_distorded, map = self.distortion_tool.distort(empty_with_patch)
        mask = self.get_mask(empty_with_patch_distorded)

        while True :
            adversarial_image = torch.mul((1-mask), image) + torch.mul(mask, empty_with_patch_distorded)
            adversarial_image = self.normalize(adversarial_image)
            adversarial_image.requires_grad = True
            vector_scores = self.model(adversarial_image)
            target_proba = torch.nn.functional.softmax(vector_scores, dim=1)[0, self.target_class].item()
            if c > 0 :
                print('iteration : %d target proba : %.2f' % (c, target_proba))
            c += 1
            if target_proba >= self.threshold or c > self.max_iterations :
                break
            
            loss = -torch.nn.functional.log_softmax(vector_scores, dim=1)
            if self.mode == 0 :
                loss_target = loss[0, self.target_class]
                loss_target.backward()
                adversarial_image.requires_grad = False
                target_grad = self.grad_normalization(adversarial_image.grad)
                target_grad = utils.normalize_tensor(target_grad)
                empty_with_patch_distorded -= self.lambda_target * target_grad
                del loss_target
            elif self.mode == 1 :
                loss_class_to_flee = loss[0, self.class_to_flee]
                loss_class_to_flee.backward()
                adversarial_image.requires_grad = False
                flee_grad = - self.grad_normalization(adversarial_image.grad)
                flee_grad = utils.normalize_tensor(flee_grad)
                empty_with_patch_distorded -= self.lambda_flee * flee_grad
                del loss_class_to_flee
            elif self.mode == 2 :
                loss_target = loss[0, self.target_class]
                loss_target.backward(retain_graph=True)
                with torch.no_grad() :
                    target_grad = self.grad_normalization(adversarial_image.grad)
                    target_grad = utils.normalize_tensor(target_grad)
                adversarial_image.grad.data.zero_()  
                loss_class_to_flee = loss[0, self.class_to_flee]
                loss_class_to_flee.backward()
                adversarial_image.requires_grad = False
                flee_grad = - self.grad_normalization(adversarial_image.grad)
                flee_grad = utils.normalize_tensor(flee_grad)
                empty_with_patch_distorded -= self.lambda_flee * flee_grad \
                + self.lambda_target * target_grad
                del loss_target, loss_class_to_flee
            else :
                assert False
            del loss 
            
            _, tv_grad = total_variation.total_variation(self.patch)
            _, print_grad = self.printability_tool.score(self.patch)
            
            tv_grad = utils.normalize_tensor(tv_grad)
            print_grad = utils.normalize_tensor(tv_grad)
        
            empty_with_patch = self.distortion_tool.undistort(empty_with_patch_distorded, map, 
                                                            empty_with_patch)

            self.patch = empty_with_patch[:, :, row0:row0 + self.patch_dim, col0:col0 + self.patch_dim]

            self.patch -= self.lambda_tv * tv_grad + self.lambda_print * print_grad

            empty_with_patch = torch.zeros(1, 3, IMAGE_DIM, IMAGE_DIM)
            empty_with_patch[0, :, row0:row0 + self.patch_dim, col0:col0 + self.patch_dim] = self.patch 
            empty_with_patch_distorded = self.distortion_tool.distort_with_map(empty_with_patch, map)
        return adversarial_image, empty_with_patch
    
    def get_mask(self, image) :
        mask = torch.zeros_like(image)
        mask[image != 0] = 1
        return mask
    
    def find_patch_position(self, grad):
        conv = torch.nn.functional.conv2d(torch.abs(grad), self.weights)
        conv = torch.squeeze(conv).numpy()
        conv = conv/np.max(conv)
        binary = np.where(conv < 0.3, 0, 1)
        X  = np.transpose(binary.nonzero())
        self.kMeans.fit(X)
        row, col = self.kMeans.cluster_centers_[np.random.randint(
            len(self.kMeans.cluster_centers_))]
        row0 = int(max(row - self.patch_dim/2, 0))
        col0 = int(max(col - self.patch_dim/2, 0))
        return row0, col0
    
    def random_transform(self):
        row0, col0 = np.random.choice(IMAGE_DIM - self.patch_dim, size=2)
        return row0, col0
        
    def test(self, epoch, random=True):
        self.target_proba_test[epoch] = []
        total, success = 0, 0
        for image, true_label in self.test_loader :
            image.requires_grad = True
            vector_scores = self.model(image)
            model_label = torch.argmax(vector_scores.data).item()
            if model_label is not true_label.item() or model_label is self.target_class :
                continue
            total += 1
            
            if not random :
                loss_model = -torch.nn.functional.log_softmax(vector_scores, 
                                                            dim=1)[0, model_label]
                loss_model.backward()
                del loss_model
                image.requires_grad = False
                row0, col0 = self.find_patch_position(image.grad)
            else :
                row0, col0 = self.random_transform()
                
            empty_with_patch = torch.zeros(1, 3, IMAGE_DIM, IMAGE_DIM)
            empty_with_patch[0, :, row0:row0 + self.patch_dim, 
                             col0:col0 + self.patch_dim] = self.patch
                
            if (self.distort) :
                empty_with_patch_distorded, _ = self.distortion_tool.distort(empty_with_patch)
                mask = self.get_mask(empty_with_patch_distorded)
                adversarial_image = torch.mul((1-mask), image) + torch.mul(mask, empty_with_patch_distorded)
            else :
                mask = self.get_mask(empty_with_patch)
                adversarial_image = torch.mul((1-mask), image) + torch.mul(mask, empty_with_patch)

            adversarial_image = self.normalize(adversarial_image)
            vector_scores = self.model(adversarial_image)
            target_proba = torch.nn.functional.softmax(vector_scores, dim=1)[0, self.target_class].item()
            adversarial_label = torch.argmax(vector_scores.data).item()
            if adversarial_label == self.target_class:
                success += 1
            
            if (total % n_enreg_img == 0):
                torchvision.utils.save_image(adversarial_image.data, path_image_folder 
                                            + 'test_epoch%d_target_proba%.2f_label%d.png' 
                                            % (epoch, target_proba, adversarial_label))
                plt.imshow(utils.tensor_to_array(image))
                if not random :
                    r = plt.scatter(self.kMeans.cluster_centers_[:, 1],
                                    self.kMeans.cluster_centers_[:, 0],
                                    s=100, c="orange")
                    plt.savefig(path_image_folder + 'test_epoch%d_clusters.png' % epoch, 
                                bbox_inches='tight')
                    r.remove()
                
            self.target_proba_test[epoch].append(target_proba)
            print('sucess/total : %d/%d accuracy : %.2f' % (success, total, (100 * success/float(total))))
            if limit_train_len is not None and total >= limit_test_len :
                break
        self.success_rate_test[epoch] = 100 * (success/float(total))
            
    def train_and_test(self, random=True):
        self.model.eval()
        for epoch in range(self.n_epochs) :
            n = 0
            for image, true_label in self.train_loader :
                image.requires_grad = True
                vector_scores = self.model(image)
                model_label = torch.argmax(vector_scores.data).item()
                if model_label is not true_label.item() or model_label is self.target_class  :
                    continue
                n += 1
                
                loss_model = -torch.nn.functional.log_softmax(vector_scores, dim=1)[0, model_label]
                loss_model.backward()
                
                del loss_model
                image.requires_grad = False
                
                if self.random_mode == 0 :
                    row0, col0 = self.random_transform()
                else :
                    row0, col0 = self.find_patch_position(image.grad)
                
                if (self.distort) :
                    adversarial_image, empty_with_patch = self.image_attack_with_distortion(image, row0, col0)
                else :
                    adversarial_image, empty_with_patch = self.image_attack(image, row0, col0)

                if (n % n_enreg_img == 0):
                    torchvision.utils.save_image(image.data, path_image_folder 
                                                + 'epoch%d_image%d_label%d_original.png' 
                                                % (epoch, n, true_label.item()))
                    
                    torchvision.utils.save_image(empty_with_patch.data, path_image_folder 
                                                + 'epoch%d_image%d_empty_with_patch.png' 
                                                % (epoch, n))
                    
                    torchvision.utils.save_image(adversarial_image.data, path_image_folder 
                                                + 'epoch%d_image%d_adversarial.png' 
                                                % (epoch, n))
                    
                    torchvision.utils.save_image(self.patch.data, path_image_folder
                                                + 'epoch%d_image%d_patch.png' 
                                                % (epoch, n))
                    
                    plt.imshow(utils.tensor_to_array(image))
                    r = plt.scatter(self.kMeans.cluster_centers_[:, 1],
                                    self.kMeans.cluster_centers_[:, 0],
                                    s=100, c="orange")
                    plt.savefig(path_image_folder + 'epoch%d_image%d_clusters.png' % (epoch, n), 
                                bbox_inches='tight')
                    r.remove()
                    
                if limit_train_len is not None and n >= limit_train_len :
                    break
                
            if self.validation :
                self.test(epoch, random=self.random_mode==1)
        if not self.validation :
            self.test(-1, random=self.random_mode==1)  
        plt.clf()
        return None       
        
    def save_patch(self, path):
        self.model = None
        self.train_loader = None
        self.test_loader = None
        self.distortion_tool = None
        self.printability_tool = None
        self.kMeans = None
        pickle.dump(self, open(path, "wb"))
"""
if __name__=="__main__":
    path_model = 'C:\\Users\\alexi\\PROJET_3A\\Projet Adversarial Patch\\Project Adverserial Patch\\Collision Avoidance\\best_model_extended.pth'
    path_dataset = 'C:\\Users\\alexi\\PROJET_3A\\Projet Adversarial Patch\\Project Adverserial Patch\\Collision Avoidance\\dataset'
    path_calibration = 'C:\\Users\\alexi\\PROJET_3A\\projet_3A\\calibration\\'
    path_distortion = 'C:\\Users\\alexi\\PROJET_3A\\projet_3A\\distortion\\distortion.so'
    path_printable_vals = 'C:\\Users\\alexi\\PROJET_3A\\projet_3A\\printability\\printable_vals.dat'
    
    patch_trainer = PatchTrainer(path_model, path_dataset, path_calibration, 
                                       path_distortion, path_printable_vals, 
                                       validation=True, lambda_tv=0.005, 
                                       lambda_print=0.005, n_classes=2, 
                                       target_class=1, distort=True, 
                                       patch_relative_size=0.05, n_epochs=2)
    
    patch_trainer.save_patch("coucou.patch")
"""