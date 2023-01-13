import torch
import torchvision
import numpy as np
import datetime
import distortion.distortion as distortion
import total_variation.new_total_variation as total_variation
import printability.new_printability as printability
import pickle
import sklearn.cluster
import matplotlib.pyplot as plt
import utils.utils as u
import constants.constants as consts

class PatchTrainer():
    def __init__(self, path_model, path_dataset, path_calibration, path_distortion,
                 path_printable_colors, mode=consts.Mode.TARGET, 
                 random_mode=consts.RandomMode.FULL_RANDOM, validation=True, n_classes=10, 
                 target_class=1, class_to_flee=None, patch_relative_size=0.05, 
                 distort=False, n_epochs=2, lambda_tv=0, lambda_print=0, 
                 threshold=0.9, max_iterations=10):

        self.date = datetime.datetime.now().strftime("%d/%m/%Y %H:%M:%S")
        self.path_model = path_model
        self.path_dataset = path_dataset
        self.path_calibration = path_calibration
        self.path_distortion = path_distortion
        self.path_printable_colors = path_printable_colors
        self.mode = mode
        self.random_mode = random_mode
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

        self.model = u.load_model(self.path_model, n_classes=self.n_classes)
        self.train_loader, self.test_loader = u.load_dataset(self.path_dataset)

        image_size = consts.IMAGE_DIM ** 2
        patch_size = image_size * self.patch_relative_size
        self.patch_dim = int(patch_size ** (0.5))

        self.normalize = torchvision.transforms.Normalize(mean=consts.MEAN, std=consts.STD)

        if self.distort:
            self.distortion_tool = distortion.DistortionTool(self.path_calibration, 
                                                             self.path_distortion)

        self.tv_module = total_variation.TotalVariationModule()

        self.printability_module = printability.PrintabilityModule(self.path_printable_colors,
                                                                   self.patch_dim)

        self.patch = self.random_patch_init()

        if self.random_mode == (consts.RandomMode.TRAIN_KMEANS or consts.RandomMode.TRAIN_TEST_KMEANS) :
            self.kMeans = sklearn.cluster.KMeans(n_clusters=self.n_epochs)
            self.weights = torch.ones_like(self.patch)

        self.target_proba_test = {}
        self.success_rate_test = {}

    def random_patch_init(self):
        patch = torch.empty(1, 3, self.patch_dim, self.patch_dim)
        for i, (m, s) in enumerate(zip(consts.MEAN, consts.STD)):
            patch[:, i, :, :].normal_(mean=m, std=s)
        return patch

    def test_model(self):
        success, total = 0, 0
        for loader in [self.train_loader, self.test_loader]:
            for image, label in loader:
                output = self.model(self.normalize(image))
                success += (label == output.argmax(1)).sum()
                total += len(label)
                print('sucess/total : %d/%d accuracy : %.2f' 
                    % (success, total, (100 * success / float(total))))
        
    @staticmethod
    def get_mask(image):
        mask = torch.zeros_like(image)
        mask[image != 0] = 1
        return mask

    def find_patch_position(self, grad):
        assert self.random_mode > 0
        conv = torch.nn.functional.conv2d(torch.abs(grad), self.weights)
        conv_array = torch.squeeze(conv).numpy()
        conv_normalized = conv_array / np.max(conv_array)
        binary = np.where(conv_normalized < consts.KMEANS_THRESHOLD, 0, 1)
        X = np.transpose(binary.nonzero())
        self.kMeans.fit(X)
        row, col = self.kMeans.cluster_centers_[np.random.randint(
                                                len(self.kMeans.cluster_centers_))]
        row0 = int(max(row - self.patch_dim / 2, 0))
        col0 = int(max(col - self.patch_dim / 2, 0))
        return row0, col0

    def random_transform(self):
        row0, col0 = np.random.choice(consts.IMAGE_DIM - self.patch_dim, size=2)
        return row0, col0

    def attack(self, image, row0, col0):
        c = 0

        empty_with_patch = torch.zeros(1, 3, consts.IMAGE_DIM, consts.IMAGE_DIM)
        empty_with_patch[:, :, row0:row0 + self.patch_dim, 
                               col0:col0 + self.patch_dim] = self.patch
        mask = self.get_mask(empty_with_patch)
        for c in range(self.max_iterations + 1) :
            adversarial_image = torch.mul(1 - mask, image) + torch.mul(mask, empty_with_patch)
            adversarial_image.requires_grad = True
            normalized = self.normalize(adversarial_image)
            vector_scores = self.model(normalized)
            vector_proba = torch.nn.functional.softmax(vector_scores, dim=1)
            target_proba = vector_proba[0, self.target_class].item()
            if c > 0:
                print('iteration : %d target proba : %f' % (c, target_proba))
             
            if target_proba >= self.threshold:
                break
                    
            loss = -torch.nn.functional.log_softmax(vector_scores, dim=1)
            
            if self.mode == consts.Mode.TARGET :
                loss[0, self.target_class].backward()
                empty_with_patch -= adversarial_image.grad
            elif self.mode == consts.Mode.FLEE:
                loss[0, self.class_to_flee].backward()
                empty_with_patch -= adversarial_image.grad
            else :
                loss[0, self.target_class].backward(retain_graph=True)
                target_grad = adversarial_image.grad.clone()
                adversarial_image.grad.zero_()
                loss[0, self.class_to_flee].backward()
                empty_with_patch -= target_grad - adversarial_image.grad

            if self.lambda_tv > 0 or self.lambda_print > 0 :
                self.patch.requires_grad = True
                if self.lambda_tv > 0 :
                    loss = self.tv_module(self.patch)
                    loss.backward()
                    tv_grad = self.patch.grad.clone()

                    self.patch.grad.zero_()
                if self.lambda_print > 0 :
                    loss = self.printability_module(self.patch)
                    loss.backward()
                    print_grad = self.patch.grad.clone()

                    self.patch.grad.zero_()

                self.patch.requires_grad = False

                self.patch = empty_with_patch[:, :, row0:row0 + self.patch_dim, 
                                                    col0:col0 + self.patch_dim]

                self.patch -= self.lambda_tv * tv_grad + self.lambda_print * print_grad
                empty_with_patch = torch.zeros(1, 3, consts.IMAGE_DIM, consts.IMAGE_DIM)
                empty_with_patch[0, :, row0:row0 + self.patch_dim, 
                                       col0:col0 + self.patch_dim] = self.patch

            del loss
        if self.lambda_tv == 0 and self.lambda_print ==  0 :
            self.patch = empty_with_patch[:, :, row0:row0 + self.patch_dim, 
                                                col0:col0 + self.patch_dim]
        return normalized, empty_with_patch

    def attack_with_distortion(self, image, row0, col0):
        c = 0

        empty_with_patch = torch.zeros(1, 3, consts.IMAGE_DIM, consts.IMAGE_DIM)
        empty_with_patch[:, :, row0:row0 + self.patch_dim,
                               col0:col0 + self.patch_dim] = self.patch

        empty_with_patch_distorted, map_ = self.distortion_tool.distort(empty_with_patch)
        mask = self.get_mask(empty_with_patch_distorted)

        for c in range(self.max_iterations + 1) :
            adversarial_image = torch.mul(1 - mask, image) + torch.mul(mask, 
                                                                       empty_with_patch_distorted)
            adversarial_image.requires_grad = True
            normalized = self.normalize(adversarial_image)
            vector_scores = self.model(normalized)
            vector_proba = torch.nn.functional.softmax(vector_scores, dim=1)
            target_proba = vector_proba[0, self.target_class].item()
            if c > 0:
                print('iteration : %d target proba : %f' % (c, target_proba))
            if target_proba >= self.threshold:
                break

            loss = -torch.nn.functional.log_softmax(vector_scores, dim=1)
            if self.mode == consts.Mode.TARGET :
                loss[0, self.target_class].backward()
                empty_with_patch_distorted -= adversarial_image.grad
            elif self.mode == consts.Mode.FLEE:
                loss[0, self.class_to_flee].backward()
                empty_with_patch_distorted -= adversarial_image.grad
            else :
                loss[0, self.target_class].backward(retain_graph=True)
                target_grad = adversarial_image.grad.clone()
                adversarial_image.grad.zero_()
                loss[0, self.class_to_flee].backward()
                empty_with_patch_distorted -= target_grad - adversarial_image.grad

            
            if self.lambda_tv > 0 or self.lambda_print > 0 :
                self.patch.requires_grad = True

                if self.lambda_tv > 0 :
                    loss = self.tv_module(self.patch)
                    loss.backward()
                    tv_grad = self.patch.grad.clone()

                    self.patch.grad.zero_()
                
                if self.lambda_print > 0 :
                    loss = self.printability_module(self.patch)
                    loss.backward()
                    print_grad = self.patch.grad.clone()

                    self.patch.grad.zero_()

                self.patch.requires_grad = False

                empty_with_patch = self.distortion_tool.undistort(empty_with_patch_distorted, map_,
                                                                  empty_with_patch)
                self.patch = empty_with_patch[:, :, row0:row0 + self.patch_dim, 
                                                    col0:col0 + self.patch_dim]
                self.patch -= self.lambda_tv * tv_grad + self.lambda_print * print_grad

                empty_with_patch = torch.zeros(1, 3, consts.IMAGE_DIM, consts.IMAGE_DIM)
                empty_with_patch[:, :, row0:row0 + self.patch_dim, 
                                       col0:col0 + self.patch_dim] = self.patch
                empty_with_patch_distorted = self.distortion_tool.distort_with_map(empty_with_patch, 
                                                                                   map_)

            del loss
        
        if self.lambda_tv == 0 and self.lambda_print ==  0 :
            empty_with_patch = self.distortion_tool.undistort(empty_with_patch_distorted, map_,
                                                              empty_with_patch)
            self.patch = empty_with_patch[:, :, row0:row0 + self.patch_dim, 
                                                col0:col0 + self.patch_dim]
        return normalized, empty_with_patch

    def test(self, epoch=-1):
        self.target_proba_test[epoch] = []
        total, success = 0, 0
        for image, true_label in self.test_loader:
            if self.random_mode == consts.RandomMode.TRAIN_TEST_KMEANS :
                image.requires_grad = True
            vector_scores = self.model(image)
            model_label = torch.argmax(vector_scores).item()
            if model_label is not true_label.item() or model_label is self.target_class:
                continue
            total += 1

            if self.random_mode == consts.RandomMode.TRAIN_TEST_KMEANS :
                loss = -torch.nn.functional.log_softmax(vector_scores, dim=1)
                loss[0, model_label].backward()
                row0, col0 = self.find_patch_position(image.grad)
                image.requires_grad = False
                del loss
            else:
                row0, col0 = self.random_transform()

            empty_with_patch = torch.zeros(1, 3, consts.IMAGE_DIM, consts.IMAGE_DIM)
            empty_with_patch[:, :, row0:row0 + self.patch_dim, 
                                   col0:col0 + self.patch_dim] = self.patch

            if self.distort:
                empty_with_patch_distorted, _ = self.distortion_tool.distort(empty_with_patch)
                mask = self.get_mask(empty_with_patch_distorted)
                adversarial_image = torch.mul(1 - mask, image) + \
                                    torch.mul(mask, empty_with_patch_distorted)
            else:
                mask = self.get_mask(empty_with_patch)
                adversarial_image = torch.mul(1 - mask, image) + torch.mul(mask, empty_with_patch)

            normalized = self.normalize(adversarial_image)
            vector_scores = self.model(normalized)
            adversarial_label = torch.argmax(vector_scores).item()
            vector_proba = torch.nn.functional.softmax(vector_scores, dim=1)
            target_proba = vector_proba[0, self.target_class].item()
            if adversarial_label == self.target_class:
                success += 1

            if total % consts.N_ENREG_IMG == 0:
                torchvision.utils.save_image(normalized, consts.PATH_IMG_FOLDER
                                             + 'test_epoch%d_target_proba%.2f_label%d.png'
                                             % (epoch, target_proba, adversarial_label))
                plt.imshow(u.tensor_to_array(image))
                if self.random_mode == consts.RandomMode.TRAIN_TEST_KMEANS :
                    r = plt.scatter(self.kMeans.cluster_centers_[:, 1],
                                    self.kMeans.cluster_centers_[:, 0],
                                    s=100, c="orange")
                    plt.savefig(consts.PATH_IMG_FOLDER + 'test_epoch%d_clusters.png' % epoch,
                                bbox_inches='tight')
                    r.remove()

            self.target_proba_test[epoch].append(target_proba)
            print('sucess/total : %d/%d accuracy : %.2f' % 
                 (success, total, (100 * success / float(total))))
            if consts.LIMIT_TEST_LEN is not None and total >= consts.LIMIT_TEST_LEN:
                break
        self.success_rate_test[epoch] = 100 * (success / float(total))

    def train(self):
        self.model.eval()
        for epoch in range(self.n_epochs):
            n = 0
            for image, true_label in self.train_loader:
                if self.random_mode == (consts.RandomMode.TRAIN_KMEANS or 
                                        consts.RandomMode.TRAIN_TEST_KMEANS):
                    image.requires_grad = True
                vector_scores = self.model(image)
                model_label = torch.argmax(vector_scores).item()
                if model_label is not true_label.item() or model_label is self.target_class:
                    continue
                n += 1

                if self.random_mode == (consts.RandomMode.TRAIN_KMEANS or 
                                        consts.RandomMode.TRAIN_TEST_KMEANS):
                    loss = -torch.nn.functional.log_softmax(vector_scores, dim=1)
                    loss[0, model_label].backward()
                    row0, col0 = self.find_patch_position(image.grad)
                    image.requires_grad = False
                    del loss
                else:
                    row0, col0 = self.random_transform()

                if self.distort:
                    adversarial_image, empty_with_patch = self.attack_with_distortion(image, row0, col0)
                else:
                    adversarial_image, empty_with_patch = self.attack(image, row0, col0)

                if n % consts.N_ENREG_IMG == 0:
                    torchvision.utils.save_image(image, consts.PATH_IMG_FOLDER
                                                 + 'epoch%d_image%d_label%d_original.png'
                                                 % (epoch, n, true_label.item()))

                    torchvision.utils.save_image(empty_with_patch, consts.PATH_IMG_FOLDER
                                                 + 'epoch%d_image%d_empty_with_patch.png'
                                                 % (epoch, n))

                    torchvision.utils.save_image(adversarial_image, consts.PATH_IMG_FOLDER
                                                 + 'epoch%d_image%d_adversarial.png'
                                                 % (epoch, n))

                    torchvision.utils.save_image(self.patch, consts.PATH_IMG_FOLDER
                                                 + 'epoch%d_image%d_patch.png'
                                                 % (epoch, n))

                    plt.imshow(u.tensor_to_array(image))

                    if self.random_mode == (consts.RandomMode.TRAIN_KMEANS or 
                                            consts.RandomMode.TRAIN_TEST_KMEANS):
                        r = plt.scatter(self.kMeans.cluster_centers_[:, 1],
                                        self.kMeans.cluster_centers_[:, 0],
                                        s=100, c="orange")
                        plt.savefig(consts.PATH_IMG_FOLDER + 'epoch%d_image%d_clusters.png' % (epoch, n),
                                    bbox_inches='tight')
                        r.remove()

                if consts.LIMIT_TRAIN_EPOCH_LEN is not None and n >= consts.LIMIT_TRAIN_EPOCH_LEN :
                    break

            if self.validation:
                self.test(epoch)
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

