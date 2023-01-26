import torch
import torchvision
import datetime
import total_variation.new_total_variation as total_variation
import printability.new_printability as printability
import color_jitter.color_jitter as color_jitter
import pickle
import matplotlib.pyplot as plt
import utils.utils as u
import constants.constants as consts
import transformation

class PatchTrainer():
    def __init__(self, mode=consts.Mode.TARGET, validation=True, 
                 target_class=2, patch_relative_size=0.05, 
                 jitter=False, distort=False, n_epochs=2, 
                 lambda_tv=0, lambda_print=0, 
                 threshold=0.9, max_iterations=10):

        self.pretty_printer = u.PrettyPrinter(self)
        self.date = datetime.datetime.now().strftime("%d/%m/%Y %H:%M:%S")
        self.path_model = consts.PATH_MODEL
        self.path_dataset = consts.PATH_DATASET
        self.limit_train_epoch_len = consts.LIMIT_TRAIN_EPOCH_LEN
        self.limit_test_len = consts.LIMIT_TEST_LEN
        self.mode = mode
        self.validation = validation
        self.target_class = target_class
        self.patch_relative_size = patch_relative_size
        self.jitter = jitter
        self.distort = distort
        self.n_epochs = n_epochs
        self.lambda_tv = lambda_tv
        self.lambda_print = lambda_print
        self.threshold = threshold
        self.max_iterations = max_iterations

        self.model = u.load_model(self.path_model, n_classes=consts.N_CLASSES)
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
        
        self.transformation_tool = transformation.TransformationTool(self.patch_dim,
                                                                     distort)

        self.tv_module = total_variation.TotalVariationModule()

        self.print_module = printability.PrintabilityModule(self.patch_dim)

        self.patch = self._random_patch_init()

        self.target_proba_train = {}
        self.success_rate_test = {}

    def _random_patch_init(self):
        patch = torch.empty(1, 3, consts.IMAGE_DIM, consts.IMAGE_DIM)
        row0, col0 = consts.IMAGE_DIM//2 - self.patch_dim//2, \
                     consts.IMAGE_DIM//2 - self.patch_dim//2
        patch[0, :, row0:row0 + self.patch_dim, 
                    col0:col0 + self.patch_dim] = torch.rand(3, self.patch_dim, 
                                                                self.patch_dim)
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
    def _get_mask(image):
        mask = torch.zeros_like(image)
        mask[image != 0] = 1
        return mask
    
    def _get_patch(self):
        row0, col0 = consts.IMAGE_DIM//2 - self.patch_dim//2, \
                     consts.IMAGE_DIM//2 - self.patch_dim//2
        return self.patch[:, :, row0:row0 + self.patch_dim, 
                                col0:col0 + self.patch_dim]
    
    def _get_grad(self, module):
        _patch = self._get_patch()
        _patch.requires_grad = True
        loss = module(_patch)
        loss.backward()
        grad = torch.zeros(1, 3, consts.IMAGE_DIM, consts.IMAGE_DIM)
        row0, col0 = consts.IMAGE_DIM//2 - self.patch_dim//2, \
                     consts.IMAGE_DIM//2 - self.patch_dim//2
        grad[0, :, row0:row0 + self.patch_dim, 
                   col0:col0 + self.patch_dim] = _patch.grad.clone().detach()
        return grad

    def attack(self, image):
        transformed, map_ = self.transformation_tool.random_transfom(self.patch)
        mask = self._get_mask(transformed)
        for i in range(self.max_iterations + 1) :
            attacked = torch.mul(1 - mask, image) + \
                       torch.mul(mask, transformed)
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
                transformed -= attacked.grad
            elif self.mode == consts.Mode.FLEE:
                loss[0, self.target_class].backward()
                transformed += attacked.grad
            elif self.mode == consts.Mode.TARGET_AND_FLEE:
                loss[0, self.target_class].backward(retain_graph=True)
                transformed -= attacked.grad 
                model_label = torch.argmax(vector_scores).item()
                if model_label != self.target_class :
                    attacked.grad.zero_()
                    loss[0, model_label].backward()
                    transformed += attacked.grad

        self.patch = self.transformation_tool.undo_transform(self.patch, 
                                                             transformed, map_)
        print_grad = self._get_grad(self.print_module)
        tv_grad = self._get_grad(self.tv_module)
        
        self.patch -= self.lambda_tv * tv_grad + self.lambda_print * print_grad
        
        return first_target_proba, normalized

    def train(self):
        self.pretty_printer.training()
        for epoch in range(self.n_epochs):
            total, success = 0, 0
            self.target_proba_train[epoch] = []
            for image, true_label in self.train_loader:
                if self.jitter : 
                    self.color_jitter_module.jitter()
                vector_scores = self.model(self.normalize(image))
                model_label = torch.argmax(vector_scores).item()
                if model_label != true_label.item() :
                    continue
                elif self.mode in [consts.Mode.TARGET, 
                                   consts.Mode.TARGET_AND_FLEE] and \
                     model_label == self.target_class:
                    continue
                elif self.mode == consts.Mode.FLEE and \
                        model_label != self.target_class:
                    continue
                if total == 0 :
                    success_rate = None
                else :
                    success_rate = 100 * (success / float(total))
                    
                self.pretty_printer.update_image(epoch, success_rate, total)
        
                total += 1
                
                ret = self.attack(image)
                first_target_proba, attacked = ret
                self.target_proba_train[epoch].append(first_target_proba)

                if self.mode in [consts.Mode.TARGET, consts.Mode.TARGET_AND_FLEE] :
                    if first_target_proba >= self.threshold : success += 1
                elif self.mode == consts.Mode.FLEE :
                    if first_target_proba <= self.threshold : success += 1

                
                if total % consts.N_ENREG_IMG == 0:
                    torchvision.utils.save_image(image, 
                                                 consts.PATH_IMG_FOLDER + 
                                                 'epoch%d_image%d_label%d_original.png'
                                                 % (epoch, total, true_label.item()))

                    torchvision.utils.save_image(attacked, 
                                                 consts.PATH_IMG_FOLDER + 
                                                 'epoch%d_image%d_attacked.png'
                                                 % (epoch, total))

                    torchvision.utils.save_image(self._get_patch(), 
                                                 consts.PATH_IMG_FOLDER + 
                                                 'epoch%d_image%d_patch.png'
                                                 % (epoch, total))
                    
                if self.limit_train_epoch_len is not None and \
                        total >= self.limit_train_epoch_len :
                    break
            if self.validation:
                self.test(epoch)
        return None

    def test(self, epoch=-1):
        total, success = 0, 0
        for image, true_label in self.test_loader:
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
            
            transformed, _ = self.transformation_tool.random_transfom(self.patch)
            mask = self._get_mask(transformed)
            attacked = torch.mul(1 - mask, image) + torch.mul(mask, transformed)
            normalized = self.normalize(attacked)
            vector_scores = self.model(normalized)
            attacked_label = int(torch.argmax(vector_scores))
            vector_proba = torch.nn.functional.softmax(vector_scores, dim=1)
            target_proba = float(vector_proba[0, self.target_class])
            
            if self.mode in [consts.Mode.TARGET, consts.Mode.TARGET_AND_FLEE] :
                if attacked_label == self.target_class: 
                    success += 1
            elif self.mode == consts.Mode.FLEE :
                if attacked_label != self.target_class: 
                    success += 1

            if total % consts.N_ENREG_IMG == 0:
                torchvision.utils.save_image(normalized, consts.PATH_IMG_FOLDER + 
                                             'test_epoch%d_target_proba%.2f_label%d.png'
                                             % (epoch, target_proba, attacked_label))
            self.pretty_printer.update_test(epoch, 100 * success / float(total), 
                                            total)
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

