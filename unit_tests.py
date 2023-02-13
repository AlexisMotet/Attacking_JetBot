import unittest
import new_patch
import constants.constants as c
import time
import torch
import matplotlib.pyplot as plt
import utils.utils as u
import torchvision
import random
import image_processing.image_processing as i
from configs import config
import numpy as np
from PIL import Image

def tensor_to_array(tensor):
    tensor = torch.clamp(tensor, 0, 1)
    return u.tensor_to_array(tensor)

class ImageTransformation(unittest.TestCase):
    def setUp(self):
        u.setup_config(config)
        self.train_loader, _ = u.load_dataset()
        self.normalize = torchvision.transforms.Normalize(mean=c.consts["NORMALIZATION_MEAN"], 
                                                          std=c.consts["NORMALIZATION_STD"])
        img_path = "C:\\Users\\alexi\\PROJET_3A\\imagenette2-160\\train\\n02102040\\ILSVRC2012_val_00032959.JPEG"
        x = Image.open(img_path)
        t = torchvision.transforms.Resize((80, 80))
        self.image = t(u.array_to_tensor(np.asarray(x)/255))

    def test_normalization(self):
        _, (ax1, ax2) = plt.subplots(1, 2)
        plt.suptitle("TEST NORMALIZATION")
        for img, _ in self.train_loader:
            ax1.imshow(tensor_to_array(img), interpolation='nearest')
            ax1.set_title('original image')

            t0 = time.time()
            normalized_img = self.normalize(img)
            t1 = time.time()
            ax2.imshow(tensor_to_array(normalized_img), interpolation='nearest')
            ax2.set_title('normalized image\ndeltat=%.2fms' % ((t1 - t0)*1e3))

            plt.pause(1)
        plt.show()
        
    def test_brightness_contrast(self):
        _, (ax1, ax2) = plt.subplots(1, 2)
        plt.suptitle("TEST BRIGHTNESS")
        for img, _ in self.train_loader:
            brightness = float(torch.empty(1).uniform_(-c.consts["BRIGHTNESS_BIAS"], 
                                                        c.consts["BRIGHTNESS_BIAS"]))
            contrast = float(torch.empty(1).uniform_(1-c.consts["CONTRAST_GAIN"], 
                                                     1+c.consts["CONTRAST_GAIN"]))
            ax1.imshow(tensor_to_array(img), interpolation='nearest')
            ax1.set_title('original image')
            if random.getrandbits(1):
                modified = contrast*img + brightness
            else :
                modified = (img + brightness) * contrast
            ax2.imshow(tensor_to_array(modified), interpolation='nearest')
            ax2.set_title('brightness and contrast changed')

            plt.pause(1)
        plt.show()
        
    def test_blurring(self):
        _, (ax1, ax2) = plt.subplots(1, 2)
        img = torch.zeros(1, 3, 224, 224)
        plt.suptitle("TEST BLURRING")
        img[0, :, 112-40:112+40, 112-40:112+40] = self.image
        sigma_blur = c.consts["BLUR_SIGMA_MAX"]
        ax1.imshow(tensor_to_array(img), interpolation='nearest')
        ax1.set_title('original patch')
        blurred = torchvision.transforms.functional.gaussian_blur(img, 
            kernel_size=c.consts["BLUR_KERNEL_SIZE"], sigma=sigma_blur)
        ax2.imshow(tensor_to_array(blurred), interpolation='nearest')
        ax2.set_title('blurred patch')

        plt.show()
        
    def test_noise(self):
        _, (ax1, ax2) = plt.subplots(1, 2)
        img = torch.zeros(1, 3, 224, 224)
        img[0, :, 112-40:112+40, 112-40:112+40] = self.image
        plt.suptitle("TEST NOISE")
        for _ in range(100):
            ax1.imshow(tensor_to_array(img), interpolation='nearest')
            ax1.set_title('original patch')
            noise = torch.randn_like(img)
            noise = u.array_to_tensor(c.consts["NOISE_STD"] * np.random.standard_normal((
                                                                c.consts["IMAGE_DIM"], 
                                                                c.consts["IMAGE_DIM"], 3)))
            ax2.imshow(tensor_to_array(img + noise), interpolation='nearest')
            ax2.set_title('noisy patch')

            plt.pause(1)
        plt.show()
        
    def test_patch_processing_module(self):
        patch_processing_module = i.PatchProcessingModule()
        _, (ax1, ax2) = plt.subplots(1, 2)
        img = torch.zeros(1, 3, 224, 224)
        img[0, :, 112-40:112+40, 112-40:112+40] = self.image
        plt.suptitle("TEST patch_processing_module")
        for _ in range(100):
            patch_processing_module.jitter()
            ax1.imshow(tensor_to_array(img), interpolation='nearest')
            ax1.set_title('original patch')
            ax2.imshow(tensor_to_array(patch_processing_module(img)), 
                       interpolation='nearest')
            ax2.set_title('patch processing module modif')

            plt.pause(1)
        plt.show()
        
    def test_image_processing_module(self):
        image_processing_module = i.ImageProcessingModule(normalize=False)
        _, (ax1, ax2) = plt.subplots(1, 2)
        img = torch.zeros(1, 3, 224, 224)
        img[0, :, 112-40:112+40, 112-40:112+40] = self.image
        plt.suptitle("TEST IMAGE PROCESSING MODULE WITHOUT N")
        for _ in range(100):
            image_processing_module.jitter()
            ax1.imshow(tensor_to_array(img), interpolation='nearest')
            ax1.set_title('original patch')
            ax2.imshow(tensor_to_array(image_processing_module(img)), 
                       interpolation='nearest')
            ax2.set_title('image processing module modif')

            plt.pause(1)
        plt.show()
        
    def test_global(self):
        patch_processing_module = i.PatchProcessingModule()
        image_processing_module = i.ImageProcessingModule(normalize=False)
        _, (ax1, ax2) = plt.subplots(1, 2)
        img = torch.zeros(1, 3, 224, 224)
        img[0, :, 112-40:112+40, 112-40:112+40] = self.image
        plt.suptitle("TEST GLOBAL")
        for _ in range(100):
            patch_processing_module.jitter()
            image_processing_module.jitter()
            ax1.imshow(tensor_to_array(img), interpolation='nearest')
            ax1.set_title('original patch')
            ax2.imshow(tensor_to_array(image_processing_module(patch_processing_module(img))), 
                       interpolation='nearest')
            ax2.set_title('modif patch')

            plt.pause(1)
        plt.show()

class Trainer(unittest.TestCase):
    def setUp(self):
        self.patch_trainer = new_patch.PatchTrainer(patch_relative_size=0.05, lambda_tv=0.001, lambda_print=0.025)
        img_path = "C:\\Users\\alexi\\PROJET_3A\\imagenette2-160\\train\\n02102040\\ILSVRC2012_val_00032959.JPEG"
        x = Image.open(img_path)
        t = torchvision.transforms.Resize((80, 80))
        self.image = t(u.array_to_tensor(np.asarray(x)/255))
                                                    
    def test_transformation(self):
        _, (ax1, ax2, ax3) = plt.subplots(1, 3)
        plt.suptitle("TEST TRANSFORMATION")
        trainer = self.patch_trainer
        img = torch.zeros(1, 3, 224, 224)
        img[0, :, 112-40:112+40, 112-40:112+40] = self.image
        for _ in range(100):
            ax1.imshow(u.tensor_to_array(img))
            transformed, map_ = trainer.transformation_tool.random_transform(img)
            ax2.imshow(u.tensor_to_array(transformed))
            new_img = trainer.transformation_tool.undo_transform(img, transformed, map_)
            ax3.imshow(u.tensor_to_array(new_img))
            dist = (img - new_img).pow(2).sum().sqrt()
            ax3.set_title("euclidian dist : %f" % dist)
            plt.pause(0.5)
            
    def test_attack(self):
        _, (ax1, ax2, ax3, ax4, ax5) = plt.subplots(1, 5)
        plt.suptitle("TEST ATTACK")
        trainer = self.patch_trainer
        for img, true_label in trainer.train_loader:
            vector_scores = trainer.model(trainer.image_processing_module(img))
            model_label = int(torch.argmax(vector_scores))
            if model_label != int(true_label) :
                continue
            elif model_label == trainer.target_class:
                continue
            trainer.image_processing_module.jitter()
            trainer.patch_processing_module.jitter()
            transformed, map_ = trainer.transformation_tool.random_transform(trainer.patch)
            mask =  trainer._get_mask(transformed)
            transformed.requires_grad = True
            for _ in range(trainer.max_iterations + 1) :
                torch.clamp(transformed, 0, 1)
                modified = trainer.patch_processing_module(transformed)
                attacked = torch.mul(1 - mask, img) + \
                        torch.mul(mask, modified)
                normalized = trainer.image_processing_module(attacked)
                vector_scores = trainer.model(normalized)
                vector_proba = torch.nn.functional.softmax(vector_scores, dim=1)
                loss = -torch.nn.functional.log_softmax(vector_scores, dim=1)
                loss[0, trainer.target_class].backward()
                with torch.no_grad():
                    transformed -= transformed.grad
                transformed.grad.zero_()
            ax1.imshow(u.tensor_to_array(img))
            ax2.imshow(u.tensor_to_array(transformed))
            ax3.imshow(u.tensor_to_array(modified))
            ax4.imshow(u.tensor_to_array(attacked))
            trainer.patch = trainer.transformation_tool.undo_transform(trainer.patch, 
                                                       transformed.detach(),
                                                       map_)
            trainer._apply_specific_grads()
            ax5.imshow(u.tensor_to_array(trainer.patch))
            plt.pause(0.1)
        
    



if __name__ == '__main__':
    unittest.main()
