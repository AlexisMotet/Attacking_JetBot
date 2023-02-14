import torch
import torchvision
import datetime
import image_processing.image_processing as i
import utils.utils as u
import constants.constants as c
import transformation.transformation as t
import printability.new_printability as pt
import total_variation.new_total_variation as tv
import pickle
from configs import config


class PatchTrainer():
    def __init__(self, config=config, mode=c.Mode.TARGET, validation=True, 
                 target_class=1, patch_relative_size=0.05, n_epochs=2, 
                 lambda_tv=0, lambda_print=0,
                 threshold=0.9, max_iterations=10):
        self.pretty_printer = u.PrettyPrinter(self)
        self.date = datetime.datetime.now().strftime("%d/%m/%Y %H:%M:%S")
        if config :
            u.setup_config(config)
        self.pretty_printer.print_config(config)
        self.mode = mode
        self.validation = validation
        self.target_class = target_class
        self.patch_relative_size = patch_relative_size
        self.lambda_tv = lambda_tv
        self.lambda_print = lambda_print
        self.n_epochs = n_epochs
        self.threshold = threshold
        self.max_iterations = max_iterations

        self.model = u.load_model()
        self.model.eval()
        self.train_loader, self.test_loader = u.load_dataset()

        image_size = c.consts["IMAGE_DIM"] ** 2
        patch_size = image_size * self.patch_relative_size
        self.patch_dim = int(patch_size ** (0.5))
        if self.patch_dim % 2 != 0 :
            self.patch_dim -= 1
            
        self.r0, self.c0 = c.consts["IMAGE_DIM"]//2 - self.patch_dim//2, \
                           c.consts["IMAGE_DIM"]//2 - self.patch_dim//2

        self.image_processing_module = i.ImageProcessingModule(normalize=True)
        self.patch_processing_module = i.PatchProcessingModule()
        
        self.transformation_tool = t.TransformationTool(self.patch_dim)
        
        self.print_module = pt.PrintabilityModule(self.patch_dim)
        self.tv_module = tv.TotalVariationModule()
        
        self.patch = self._random_patch_init()

        self.target_proba_train = {}
        self.success_rate_test = {}

    def _random_patch_init(self):
        patch = torch.zeros(1, 3, c.consts["IMAGE_DIM"], c.consts["IMAGE_DIM"])
        if torch.cuda.is_available():
            patch = patch.to(torch.device("cuda"))
        rand = torch.rand(3, self.patch_dim, self.patch_dim) + 1e-5
        patch[:, :, self.r0:self.r0 + self.patch_dim, 
                    self.c0:self.c0 + self.patch_dim] = rand
        return patch
    
    def test_model(self):
        success, total = 0, 0
        for loader in [self.train_loader, self.test_loader]:
            for image, label in loader:
                if torch.cuda.is_available():
                    image = image.to(torch.device("cuda"))
                output = self.model(self.image_processing_module(image))
                success += (label == output.argmax(1)).sum()
                total += len(label)
                print('sucess/total : %d/%d accuracy : %.2f' 
                        % (success, total, (100 * success / float(total))))
        
    @staticmethod
    def _get_mask(image):
        mask = torch.zeros_like(image)
        if torch.cuda.is_available():
            mask = mask.to(torch.device("cuda"))
        mask[image != 0] = 1
        return mask
    
    def _get_patch(self):
        return self.patch[:, :, self.r0:self.r0 + self.patch_dim, 
                                self.c0:self.c0 + self.patch_dim]
    
    def _apply_specific_grads(self):
        patch_ = self._get_patch()
        patch_.requires_grad = True    
        loss = self.print_module(patch_)
        loss.backward()
        print_grad = patch_.grad.clone()
        patch_.grad.zero_()
        loss = self.tv_module(patch_)
        loss.backward()
        tv_grad = patch_.grad.clone()
        patch_.requires_grad = False
        patch_ -= self.lambda_tv * tv_grad + self.lambda_print * print_grad 
        self.patch[:, :, self.r0:self.r0 + self.patch_dim, 
                         self.c0:self.c0 + self.patch_dim] = patch_
    
    def attack(self, image):
        transformed, map_ = self.transformation_tool.random_transform(self.patch)
        mask = self._get_mask(transformed)
        transformed.requires_grad = True
        for i in range(self.max_iterations + 1) :
            modified = self.patch_processing_module(transformed)
            attacked = torch.mul(1 - mask, image) + \
                       torch.mul(mask, modified)
            normalized = self.image_processing_module(attacked)
            vector_scores = self.model(normalized)
            vector_proba = torch.nn.functional.softmax(vector_scores, dim=1)
            target_proba = float(vector_proba[0, self.target_class])
            
            if i == 0: first_target_proba = target_proba
            else: self.pretty_printer.update_iteration(i, target_proba)

            if self.mode in [c.Mode.TARGET, c.Mode.TARGET_AND_FLEE]:
                if target_proba >= self.threshold : break    
            elif self.mode == c.Mode.FLEE :
                if target_proba <= self.threshold : break
            else : 
                assert False
                
            loss = -torch.nn.functional.log_softmax(vector_scores, dim=1)
            if self.mode == c.Mode.TARGET :
                loss[0, self.target_class].backward()
                with torch.no_grad():
                    transformed -= transformed.grad
            elif self.mode == c.Mode.FLEE:
                loss[0, self.target_class].backward()
                with torch.no_grad():
                    transformed += transformed.grad
            elif self.mode == c.Mode.TARGET_AND_FLEE:
                loss[0, self.target_class].backward(retain_graph=True)
                target_grad = transformed.grad.clone()
                model_label = int(torch.argmax(vector_scores))
                transformed.grad.zero_()
                if model_label != self.target_class :
                    loss[0, model_label].backward()
                with torch.no_grad():
                    transformed -= target_grad - transformed.grad
            transformed.grad.zero_()
            transformed.clamp_(1e-5, 1)    
        self.patch = self.transformation_tool.undo_transform(self.patch, 
                                                             transformed.detach(),
                                                             map_)
        self._apply_specific_grads()
        return first_target_proba, normalized

    def train(self):
        self.pretty_printer.training()
        for epoch in range(self.n_epochs):
            total, success = 0, 0
            self.target_proba_train[epoch] = []
            for image, true_label in self.train_loader:
                if torch.cuda.is_available():
                    image = image.to(torch.device("cuda"))
                vector_scores = self.model(self.image_processing_module(image))
                model_label = int(torch.argmax(vector_scores))
                if model_label != int(true_label) :
                    continue
                elif self.mode in [c.Mode.TARGET, c.Mode.TARGET_AND_FLEE] and \
                        model_label == self.target_class:
                    continue
                elif self.mode == c.Mode.FLEE and \
                        model_label != self.target_class:
                    continue

                if total == 0 :
                    success_rate = None
                else :
                    success_rate = 100 * (success / float(total))
                    
                self.pretty_printer.update_image(epoch, success_rate, total)
        
                total += 1
                
                self.image_processing_module.jitter()
                self.patch_processing_module.jitter()
                
                first_target_proba, attacked = self.attack(image)
                self.target_proba_train[epoch].append(first_target_proba)

                if self.mode in [c.Mode.TARGET, c.Mode.TARGET_AND_FLEE] :
                    if first_target_proba >= self.threshold : 
                        success += 1
                elif self.mode == c.Mode.FLEE :
                    if first_target_proba <= self.threshold : 
                        success += 1

                if total % c.consts["N_ENREG_IMG"] == 0:
                    torchvision.utils.save_image(image, 
                                                 c.consts["PATH_IMG_FOLDER"] + 
                                                 'epoch%d_image%d_label%d_original.png'
                                                 % (epoch, total, int(true_label)))

                    torchvision.utils.save_image(attacked, 
                                                 c.consts["PATH_IMG_FOLDER"] + 
                                                 'epoch%d_image%d_attacked.png'
                                                 % (epoch, total))

                    torchvision.utils.save_image(self.patch, 
                                                 c.consts["PATH_IMG_FOLDER"] + 
                                                 'epoch%d_image%d_patch.png'
                                                 % (epoch, total))
                    
                if c.consts["LIMIT_TRAIN_EPOCH_LEN"] != -1 and \
                        total >= c.consts["LIMIT_TRAIN_EPOCH_LEN"] :
                    break
            if self.validation:
                self.test(epoch)

    def test(self, epoch=-1):
        total, success = 0, 0
        for image, true_label in self.test_loader:
            if torch.cuda.is_available():
                    image = image.to(torch.device("cuda"))
            vector_scores = self.model(self.image_processing_module(image))
            model_label = int(torch.argmax(vector_scores))
            if model_label != int(true_label) :
                continue
            elif self.mode in [c.Mode.TARGET, c.Mode.TARGET_AND_FLEE] \
                    and model_label == self.target_class:
                continue
            elif self.mode == c.Mode.FLEE and model_label != self.target_class:
                continue
            
            total += 1
            
            self.image_processing_module.jitter()
            self.patch_processing_module.jitter()
            
            transformed, _ = self.transformation_tool.random_transform(self.patch)
            mask = self._get_mask(transformed)
            modified = self.patch_processing_module(transformed)
            attacked = torch.mul(1 - mask, image) + torch.mul(mask, modified)
            normalized = self.image_processing_module(attacked)
            vector_scores = self.model(normalized)
            attacked_label = int(torch.argmax(vector_scores))
            vector_proba = torch.nn.functional.softmax(vector_scores, dim=1)
            target_proba = float(vector_proba[0, self.target_class])
            
            if self.mode in [c.Mode.TARGET, c.Mode.TARGET_AND_FLEE] :
                if attacked_label == self.target_class: 
                    success += 1
            elif self.mode == c.Mode.FLEE :
                if attacked_label != self.target_class: 
                    success += 1

            if total % c.consts["N_ENREG_IMG"] == 0:
                torchvision.utils.save_image(normalized, c.consts["PATH_IMG_FOLDER"] + 
                                             'test_epoch%d_target_proba%.2f_label%d.png'
                                             % (epoch, target_proba, attacked_label))
            self.pretty_printer.update_test(epoch, 100 * success / float(total), 
                                            total)
            if c.consts["LIMIT_TEST_LEN"] != -1 and total >= c.consts["LIMIT_TEST_LEN"] :
                break
        self.success_rate_test[epoch] = 100 * (success / float(total))

    def save_patch(self, path):
        self.consts = c.consts
        self.model = None
        self.train_loader = None
        self.test_loader = None
        self.patch = self._get_patch()
        pickle.dump(self, open(path, "wb"))
        