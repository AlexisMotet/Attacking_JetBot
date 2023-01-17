import torch
import torchvision
import numpy as np
import datetime
import distortion.distortion as distortion
import total_variation.new_total_variation as total_variation
import printability.new_printability as printability
import color_jitter.color_jitter as color_jitter
import pickle
import sklearn.cluster
import matplotlib.pyplot as plt
import utils.utils as u
import constants.constants as consts

class PatchTrainer():
    def __init__(self, mode=consts.Mode.TARGET, random_mode=consts.RandomMode.FULL_RANDOM, 
                 validation=True, n_classes=10,  target_class=2, patch_relative_size=0.05, 
                 jitter=False, distort=False, n_epochs=2, lambda_tv=0, lambda_print=0, 
                 threshold=0.9, max_iterations=10):

        self.pretty_printer = u.PrettyPrinter(self, False)
        self.date = datetime.datetime.now().strftime("%d/%m/%Y %H:%M:%S")
        self.path_model = consts.PATH_MODEL
        self.path_dataset = consts.PATH_DATASET
        self.limit_train_epoch_len = consts.LIMIT_TRAIN_EPOCH_LEN
        self.limit_test_len = consts.LIMIT_TEST_LEN
        self.mode = mode
        self.random_mode = random_mode
        self.validation = validation
        self.n_classes = n_classes
        self.target_class = target_class
        self.patch_relative_size = patch_relative_size
        self.jitter = jitter
        self.distort = distort
        self.n_epochs = n_epochs
        self.lambda_tv = lambda_tv
        self.lambda_print = lambda_print
        self.threshold = threshold
        self.max_iterations = max_iterations

        self.model = u.load_model(self.path_model, n_classes=self.n_classes)
        self.model.eval()
        self.train_loader, self.test_loader = u.load_dataset(self.path_dataset)

        image_size = consts.IMAGE_DIM ** 2
        patch_size = image_size * self.patch_relative_size
        self.patch_dim = int(patch_size ** (0.5))
        
        if not self.jitter : 
            self.normalize = torchvision.transforms.Normalize(mean=consts.MEAN, 
                                                              std=consts.STD)
        else :
            self.color_jitter_module = color_jitter.ColorJitterModule()
            self.normalize = torchvision.transforms.Compose([
                            self.color_jitter_module,
                            torchvision.transforms.Normalize(mean=consts.MEAN, 
                                                            std=consts.STD)
                            ])
 
        if self.distort:
            self.dist_tool = distortion.DistortionTool(consts.PATH_CALIBRATION, 
                                                       consts.PATH_DISTORTION)

        self.tv_module = total_variation.TotalVariationModule()

        self.print_module = printability.PrintabilityModule(consts.PATH_PRINTABLE_COLORS,
                                                            self.patch_dim)

        self.patch = self.random_patch_init()

        if self.random_mode in [consts.RandomMode.TRAIN_KMEANS,
                                consts.RandomMode.TRAIN_TEST_KMEANS] :
            self.kMeans = sklearn.cluster.KMeans(n_clusters=self.n_epochs)
            self.weights = torch.ones_like(self.patch)

        self.target_proba_train = {}
        self.success_rate_test = {}

    def random_patch_init(self):
        patch = torch.empty(3, self.patch_dim, self.patch_dim)
        for i, (m, s) in enumerate(zip(consts.MEAN, consts.STD)):
            patch[i, :, :].normal_(mean=m, std=s)
        patch = patch[None, :]
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
        assert self.random_mode in [consts.RandomMode.TRAIN_KMEANS,
                                    consts.RandomMode.TRAIN_TEST_KMEANS]
        conv = torch.nn.functional.conv2d(torch.abs(grad), self.weights)
        conv_array = torch.squeeze(conv).numpy()
        conv_normalized = conv_array / np.max(conv_array)
        binary = np.where(conv_normalized < consts.KMEANS_THRESHOLD, 0, 1)
        x = np.transpose(binary.nonzero())
        self.kMeans.fit(x)
        row, col = self.kMeans.cluster_centers_[np.random.randint(
                                                len(self.kMeans.cluster_centers_))]
        row0 = int(max(row - self.patch_dim / 2, 0))
        col0 = int(max(col - self.patch_dim / 2, 0))
        return row0, col0

    def random_position(self):
        row0, col0 = np.random.choice(consts.IMAGE_DIM - self.patch_dim, size=2)
        return row0, col0
    
    def create_empty_with_patch(self, row0, col0):
        empty_with_patch = torch.zeros(1, 3, consts.IMAGE_DIM, consts.IMAGE_DIM)
        empty_with_patch[0, :, row0:row0 + self.patch_dim, 
                               col0:col0 + self.patch_dim] = self.patch
        return empty_with_patch
    
    def attack(self, image, row0, col0):
        empty_with_patch = self.create_empty_with_patch(row0, col0)
        if not self.distort : 
            mask = self.get_mask(empty_with_patch)
        else :
            distorted, map_ = self.dist_tool.distort(empty_with_patch)
            mask = self.get_mask(distorted)
        for i in range(self.max_iterations + 1) :
            if not self.distort :
                attacked = torch.mul(1 - mask, image) + \
                           torch.mul(mask, empty_with_patch)
            else :
                attacked = torch.mul(1 - mask, image) + \
                           torch.mul(mask, distorted)
            attacked.requires_grad = True
            normalized = self.normalize(attacked)
            vector_scores = self.model(normalized)
            vector_proba = torch.nn.functional.softmax(vector_scores, dim=1)
            target_proba = vector_proba[0, self.target_class].item()
            
            if i == 0: first_target_proba = target_proba
            else: self.pretty_printer.update_iteration(i, target_proba)

            if self.mode in [consts.Mode.TARGET, consts.Mode.TARGET_AND_FLEE]:
                if target_proba >= self.threshold : break    
            elif self.mode == consts.Mode.FLEE :
                if target_proba <= self.threshold : break
            else : assert False
                
            loss = -torch.nn.functional.log_softmax(vector_scores, dim=1)
            if self.mode == consts.Mode.TARGET :
                loss[0, self.target_class].backward()
                if not self.distort : empty_with_patch -= attacked.grad
                else : distorted -= attacked.grad
            elif self.mode == consts.Mode.FLEE:
                loss[0, self.target_class].backward()
                if not self.distort : empty_with_patch += attacked.grad
                else : distorted += attacked.grad
            elif self.mode == consts.Mode.TARGET_AND_FLEE:
                loss[0, self.target_class].backward(retain_graph=True)
                if not self.distort : empty_with_patch -= attacked.grad
                else : distorted -= attacked.grad
                model_label = torch.argmax(vector_scores).item()
                if model_label != self.target_class :
                    attacked.grad.zero_()
                    loss[0, model_label].backward()
                    if not self.distort : empty_with_patch += attacked.grad
                    else : distorted += attacked.grad

            self.patch.requires_grad = True
            
            loss = self.tv_module(self.patch)
            loss.backward()
            tv_grad = self.patch.grad.clone()
            self.patch.grad.zero_()
            
            loss = self.print_module(self.patch)
            loss.backward()
            print_grad = self.patch.grad.clone()
            self.patch.grad.zero_()
            
            self.patch.requires_grad = False

            if self.distort : 
                empty_with_patch = self.dist_tool.undistort(distorted, map_,
                                                            empty_with_patch)
            
            self.patch = empty_with_patch[:, :, row0:row0 + self.patch_dim, 
                                                col0:col0 + self.patch_dim]
            self.patch -= self.lambda_tv * tv_grad + self.lambda_print * print_grad
            empty_with_patch = self.create_empty_with_patch(row0, col0)
            
            if self.distort : 
                distorted = self.dist_tool.distort_with_map(empty_with_patch, 
                                                            map_)
            
        if not self.distort : return first_target_proba, normalized, empty_with_patch
        else : return first_target_proba, normalized, distorted

    def train(self):
        self.pretty_printer.training()
        for epoch in range(self.n_epochs):
            total, success, success_rate = 0, 0, 0
            self.target_proba_train[epoch] = []
            for image, true_label in self.train_loader:
                if self.random_mode in [consts.RandomMode.TRAIN_KMEANS,
                                        consts.RandomMode.TRAIN_TEST_KMEANS]:
                    image.requires_grad = True
                if self.jitter : 
                    self.color_jitter_module.jitter()
                vector_scores = self.model(self.normalize(image))
                model_label = torch.argmax(vector_scores).item()
                if model_label != true_label.item() :
                    continue
                elif self.mode in [consts.Mode.TARGET, consts.Mode.TARGET_AND_FLEE] and \
                        model_label == self.target_class:
                    continue
                elif self.mode == consts.Mode.FLEE and \
                        model_label != self.target_class:
                    continue
                self.pretty_printer.update_image(epoch, total, success_rate)
                total += 1
                
                if self.random_mode == consts.RandomMode.FULL_RANDOM :
                    row0, col0 = self.random_position()
                else :
                    loss = -torch.nn.functional.log_softmax(vector_scores, dim=1)
                    loss[0, model_label].backward()
                    row0, col0 = self.find_patch_position(image.grad)
                    image.requires_grad = False

                ret = self.attack(image, row0, col0)
                first_target_proba, attacked, empty_with_patch = ret
                self.target_proba_train[epoch].append(first_target_proba)

                if self.mode in [consts.Mode.TARGET, consts.Mode.TARGET_AND_FLEE] :
                    if first_target_proba >= self.threshold : success += 1
                elif self.mode == consts.Mode.FLEE :
                    if first_target_proba <= self.threshold : success += 1

                success_rate = success/total
                
                if total % consts.N_ENREG_IMG == 0:
                    torchvision.utils.save_image(image, 
                                                 consts.PATH_IMG_FOLDER + 
                                                 'epoch%d_image%d_label%d_original.png'
                                                 % (epoch, total, true_label.item()))

                    torchvision.utils.save_image(empty_with_patch, 
                                                 consts.PATH_IMG_FOLDER + 
                                                 'epoch%d_image%d_empty_with_patch.png'
                                                 % (epoch, total))

                    torchvision.utils.save_image(attacked, 
                                                 consts.PATH_IMG_FOLDER + 
                                                 'epoch%d_image%d_attacked.png'
                                                 % (epoch, total))

                    torchvision.utils.save_image(self.patch, 
                                                 consts.PATH_IMG_FOLDER + 
                                                 'epoch%d_image%d_patch.png'
                                                 % (epoch, total))

                    plt.imshow(u.tensor_to_array(image))

                    if self.random_mode in [consts.RandomMode.TRAIN_KMEANS,
                                            consts.RandomMode.TRAIN_TEST_KMEANS]:
                        r = plt.scatter(self.kMeans.cluster_centers_[:, 1],
                                        self.kMeans.cluster_centers_[:, 0],
                                        s=100, c="orange")
                        plt.savefig(consts.PATH_IMG_FOLDER 
                                    + 'epoch%d_image%d_clusters.png' % (epoch, total),
                                    bbox_inches='tight')
                        r.remove()
                if self.limit_train_epoch_len is not None and \
                        total >= self.limit_train_epoch_len :
                    break
            if self.validation:
                self.test(epoch)
        plt.clf()
        self.pretty_printer.clear()
        return None

    def test(self, epoch=-1):
        total, success = 0, 0
        for image, true_label in self.test_loader:
            if self.random_mode == consts.RandomMode.TRAIN_TEST_KMEANS :
                image.requires_grad = True
            if self.jitter : 
                self.color_jitter_module.jitter()
            vector_scores = self.model(self.normalize(image))
            model_label = torch.argmax(vector_scores).item()
            if model_label != true_label.item() :
                continue
            elif self.mode in [consts.Mode.TARGET, consts.Mode.TARGET_AND_FLEE] \
                    and model_label == self.target_class:
                continue
            elif self.mode == consts.Mode.FLEE and model_label != self.target_class:
                continue
            
            total += 1
            
            if self.random_mode != consts.RandomMode.TRAIN_TEST_KMEANS :
                row0, col0 = self.random_position()
            else :
                loss = -torch.nn.functional.log_softmax(vector_scores, dim=1)
                loss[0, model_label].backward()
                row0, col0 = self.find_patch_position(image.grad)
                image.requires_grad = False

            empty_with_patch = self.create_empty_with_patch(row0, col0)

            if not self.distort:
                mask = self.get_mask(empty_with_patch)
                attacked = torch.mul(1 - mask, image) + \
                           torch.mul(mask, empty_with_patch)
            else:
                distorted, _ = self.dist_tool.distort(empty_with_patch)
                mask = self.get_mask(distorted)
                attacked = torch.mul(1 - mask, image) + \
                           torch.mul(mask, distorted)

            normalized = self.normalize(attacked)
            vector_scores = self.model(normalized)
            attacked_label = torch.argmax(vector_scores).item()
            vector_proba = torch.nn.functional.softmax(vector_scores, dim=1)
            target_proba = vector_proba[0, self.target_class].item()
            
            if self.mode in [consts.Mode.TARGET, consts.Mode.TARGET_AND_FLEE] :
                if attacked_label == self.target_class: success += 1
            elif self.mode == consts.Mode.FLEE :
                if attacked_label != self.target_class: success += 1

            if total % consts.N_ENREG_IMG == 0:
                torchvision.utils.save_image(normalized, consts.PATH_IMG_FOLDER + 
                                             'test_epoch%d_target_proba%.2f_label%d.png'
                                             % (epoch, target_proba, attacked_label))
                plt.imshow(u.tensor_to_array(image))
                if self.random_mode == consts.RandomMode.TRAIN_TEST_KMEANS :
                    r = plt.scatter(self.kMeans.cluster_centers_[:, 1],
                                    self.kMeans.cluster_centers_[:, 0],
                                    s=100, c="orange")
                    plt.savefig(consts.PATH_IMG_FOLDER + 
                                'test_epoch%d_clusters.png' % epoch,
                                bbox_inches='tight')
                    r.remove()
            self.pretty_printer.update_test(epoch, total, (100 * success / float(total)))
            #print('success/total : %d/%d accuracy : %.2f' % 
            #     (success, total, (100 * success / float(total))))
            if self.limit_test_len is not None and total >= self.limit_test_len :
                break
        self.success_rate_test[epoch] = 100 * (success / float(total))

    def save_patch(self, path):
        self.model = None
        self.train_loader = None
        self.test_loader = None
        self.dist_tool = None
        self.printability_tool = None
        self.kMeans = None
        pickle.dump(self, open(path, "wb"))

