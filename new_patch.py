import torch
import torchvision
from torchvision.transforms import Normalize
import datetime
import image_processing.image_processing as i
import utils.utils as u
import constants.constants as c
import transformation.transformation as t
import printability.new_printability as pt
import total_variation.new_total_variation as tv
import pickle
from configs import config
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt

class PatchTrainer():
    def __init__(self, config=config, 
                 path_image_init=None, 
                 target_class=1, 
                 flee_class=0,
                 patch_relative_size=0.05, 
                 n_epochs=2):
        
        self.pretty_printer = u.PrettyPrinter(self)
        self.date = datetime.datetime.now().strftime("%d/%m/%Y %H:%M:%S")
        u.setup_config(config)
        self.pretty_printer.print_config(config)
        self.path_image_init = path_image_init
        self.target_class = target_class
        self.flee_class = flee_class
        self.patch_relative_size = patch_relative_size
        self.n_epochs = n_epochs

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

        self.normalize = Normalize(c.consts["NORMALIZATION_MEAN"], 
                                   c.consts["NORMALIZATION_STD"])
        
        self.patch_processing_module = i.PatchProcessingModule()
        
        self.transfo_tool = t.TransformationTool(self.patch_dim)
        
        self.print_module = pt.PrintabilityModule(self.patch_dim)
        
        self.tv_module = tv.TotalVariationModule()
        
        self.patch = self._random_patch_init()

        self.target_proba_train = {}
        self.success_rate_test = {}
        self.patches = {}

    def _random_patch_init(self):
        patch = torch.zeros(1, 3, c.consts["IMAGE_DIM"], c.consts["IMAGE_DIM"])
        if torch.cuda.is_available():
            patch = patch.to(torch.device("cuda"))
        if self.path_image_init is not None :
            image_init = u.array_to_tensor(np.asarray(Image.open(self.path_image_init))/255)
            resize = torchvision.transforms.Resize((self.patch_dim, self.patch_dim))
            patch[:, :, self.r0:self.r0 + self.patch_dim, 
                        self.c0:self.c0 + self.patch_dim] = resize(image_init)
        else :
            rand = torch.rand(3, self.patch_dim, self.patch_dim) + 1e-5
            patch[:, :, self.r0:self.r0 + self.patch_dim, 
                        self.c0:self.c0 + self.patch_dim] = rand
        return patch
    
    def test_model(self):
        success, total = 0, 0
        for loader in [self.train_loader, self.test_loader]:
            for batch, labels in loader:
                if torch.cuda.is_available():
                    batch = batch.to(torch.device("cuda"))
                output = self.model(self.normalize(batch))
                success += (labels == output.argmax(1)).sum()
                total += len(labels)
                print('sucess/total : %d/%d accuracy : %.2f' 
                        % (success, total, (100 * success / float(total))))
        
    @staticmethod
    def _get_mask(patch):
        mask = torch.zeros_like(patch)
        if torch.cuda.is_available():
            mask = mask.to(torch.device("cuda"))
        mask[patch != 0] = 1
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
        patch_ -= c.consts["LAMBDA_TV"] * tv_grad + c.consts["LAMBDA_PRINT"] * print_grad 
        self.patch[:, :, self.r0:self.r0 + self.patch_dim, 
                         self.c0:self.c0 + self.patch_dim] = patch_
    
    def attack(self, batch):
        transformed, map_ = self.transfo_tool.random_transform(self.patch)
        mask = self._get_mask(transformed)
        transformed.requires_grad = True
        for i in range(c.consts["MAX_ITERATIONS"] + 1) :
            modified = self.patch_processing_module(transformed)
            attacked = torch.mul(1 - mask, batch) + \
                       torch.mul(mask, modified)
            normalized = self.normalize(attacked)
            vector_scores = self.model(normalized)
            vector_proba = torch.nn.functional.softmax(vector_scores, dim=1)
            target_proba = float(torch.mean(vector_proba[:, self.target_class]))

            if i == 0: 
                first_target_proba = target_proba
                successes = len(batch[vector_proba[:, self.target_class] >= c.consts["THRESHOLD"]])
            else: 
                self.pretty_printer.update_iteration(i, target_proba)

            if target_proba >= c.consts["THRESHOLD"] : 
                break
                
            if self.flee_class is not None :
                loss = -torch.nn.functional.log_softmax(vector_scores, dim=1)
                torch.mean(loss[:, self.target_class]).backward(retain_graph=True)
                target_grad = transformed.grad.clone()
                transformed.grad.zero_()
                torch.mean(loss[:, self.flee_class]).backward()
                with torch.no_grad():
                    transformed -= target_grad - transformed.grad
                    transformed.clamp_(0, 1)
            else :
                loss = -torch.nn.functional.log_softmax(vector_scores, dim=1)
                torch.mean(loss[:, self.target_class]).backward()
                with torch.no_grad():
                    transformed -= transformed.grad
                    transformed.clamp_(0, 1)
            transformed.grad.zero_()  
        
        self.patch = self.transfo_tool.undo_transform(self.patch, transformed.detach(),
                                                      map_)
        self._apply_specific_grads()
        self.patch.clamp_(0, 1)
        return first_target_proba, successes, normalized

    def train(self):
        self.pretty_printer.training()
        for epoch in range(self.n_epochs):
            total, successes = 0, 0
            self.target_proba_train[epoch] = []
            i = 0
            for batch, true_labels in self.train_loader:
                if torch.cuda.is_available():
                    batch = batch.to(torch.device("cuda"))
                    true_labels = true_labels.to(torch.device("cuda"))
                    
                vector_scores = self.model(self.normalize(batch))
                model_labels = torch.argmax(vector_scores, axis=1)
                
                logical = torch.logical_and(model_labels == true_labels, 
                                            model_labels != self.target_class)
                batch = batch[logical]
                true_labels = true_labels[logical]
                
                if len(batch) == 0:
                    continue    
                    
                if total == 0 :
                    success_rate = None
                else :
                    success_rate = 100 * (successes / float(total))
                    
                self.pretty_printer.update_batch(epoch, success_rate, i, len(batch))
        
                total += len(batch)
                
                self.patch_processing_module.jitter()
                first_target_proba, s, attacked = self.attack(batch)
                
                successes += s
                self.target_proba_train[epoch].append(first_target_proba)

                if i % c.consts["N_ENREG_IMG"] == 0:
                    torchvision.utils.save_image(batch[0], 
                                                 c.consts["PATH_IMG_FOLDER"] + 
                                                 'epoch%d_batch%d_label%d_original.png'
                                                 % (epoch, i, true_labels[0]))

                    torchvision.utils.save_image(attacked[0], 
                                                 c.consts["PATH_IMG_FOLDER"] + 
                                                 'epoch%d_batch%d_attacked.png'
                                                 % (epoch, i))

                    torchvision.utils.save_image(self.patch, 
                                                 c.consts["PATH_IMG_FOLDER"] + 
                                                 'epoch%d_batch%d_patch.png'
                                                 % (epoch, i))
                    
                i += 1
                if c.consts["LIMIT_TRAIN_EPOCH_LEN"] != -1 and \
                        i >= c.consts["LIMIT_TRAIN_EPOCH_LEN"] :
                    break
            self.test(epoch)

    def test(self, epoch):
        total, successes = 0, 0
        i = 0
        for batch, true_labels in self.test_loader:
            if torch.cuda.is_available():
                    batch = batch.to(torch.device("cuda"))
                    true_labels = true_labels.to(torch.device("cuda"))
            vector_scores = self.model(self.normalize(batch))
            model_labels = torch.argmax(vector_scores, axis=1)
            logical = torch.logical_and(model_labels == true_labels, 
                                        model_labels != self.target_class)
            batch = batch[logical]
            
            if len(batch) == 0:
                continue   
                    
            total += len(batch)
            
            self.patch_processing_module.jitter()
            
            transformed, _ = self.transfo_tool.random_transform(self.patch)
            mask = self._get_mask(transformed)
            modified = self.patch_processing_module(transformed)
            attacked = torch.mul(1 - mask, batch) + torch.mul(mask, modified)
            normalized = self.normalize(attacked)
            vector_scores = self.model(normalized)
            attacked_label = torch.argmax(vector_scores, axis=1)
            vector_proba = torch.nn.functional.softmax(vector_scores, dim=1)
            target_proba = vector_proba[:, self.target_class]

            successes += len(batch[attacked_label == self.target_class])

            if i % c.consts["N_ENREG_IMG"] == 0:
                torchvision.utils.save_image(normalized[0], c.consts["PATH_IMG_FOLDER"] + 
                                             'test_epoch%d_batch_%dtarget_proba%.2f_label%d.png'
                                             % (epoch, i, target_proba[0], attacked_label[0]))
            self.pretty_printer.update_test(epoch, 100*successes/float(total), i, len(batch))
            i += 1
            if c.consts["LIMIT_TEST_LEN"] != -1 and i >= c.consts["LIMIT_TEST_LEN"] :
                break
        sucess_rate = 100 * (successes / float(total))
        self.success_rate_test[epoch] = sucess_rate
        self.patches[epoch] = (self._get_patch(), sucess_rate)

    def save_patch(self, path):
        self.consts = c.consts
        self.model = None
        self.train_loader = None
        self.test_loader = None
        best_patch, best_success_rate = None, 0
        for patch, success_rate in self.patches.values() :
            if success_rate >= best_success_rate :
                best_patch = patch
                best_success_rate = success_rate
        self.patches = None
        self.print_loss = float(self.print_module(best_patch))
        self.tv_loss = float(self.tv_module(best_patch))
        self.best_patch = best_patch
        self.max = float(torch.max(best_patch))
        self.min = float(torch.min(best_patch))
        pickle.dump(self, open(path, "wb"))
        