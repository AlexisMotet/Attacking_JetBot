import unittest
import new_patch
import constants.constants as c
import time
import numpy as np
import torch
import matplotlib.pyplot as plt
import printability.new_printability as p
import utils.utils as u
import sklearn.cluster
import PIL
import torchvision
import random
import image_transformation.image_transformation as img_transfo


def tensor_to_array(tensor):
    tensor = torch.clamp(tensor, 0, 1)
    return u.tensor_to_array(tensor)

class ImageTransformation(unittest.TestCase):
    def setUp(self):
        self.train_loader, _ = u.load_dataset(c.PATH_DATASET)
        self.normalize = torchvision.transforms.Normalize(mean=c.MEAN, std=c.STD)

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
            brightness = float(torch.empty(1).uniform_(-c.BRIGHTNESS_BIAS, 
                                                        c.BRIGHTNESS_BIAS))
            contrast = float(torch.empty(1).uniform_(1-c.CONTRAST_GAIN, 
                                                     1+c.CONTRAST_GAIN))
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
        img[0, :, 112-40:112+40, 112-40:112+40] = torch.rand(3, 80, 80)
        plt.suptitle("TEST BLURRING")
        for _ in range(100):
            sigma_blur = c.BLUR_SIGMA_MAX
            ax1.imshow(tensor_to_array(img), interpolation='nearest')
            ax1.set_title('original patch')
            blurred = torchvision.transforms.functional.gaussian_blur(img, 
                kernel_size=c.BLUR_KERNEL_SIZE, sigma=sigma_blur)
            ax2.imshow(tensor_to_array(blurred), interpolation='nearest')
            ax2.set_title('blurred patch')

            plt.pause(1)
        plt.show()
        
    def test_noise(self):
        _, (ax1, ax2) = plt.subplots(1, 2)
        img = torch.zeros(1, 3, 224, 224)
        img[0, :, 112-40:112+40, 112-40:112+40] = torch.rand(3, 80, 80)
        plt.suptitle("TEST NOISE")
        for _ in range(100):
            ax1.imshow(tensor_to_array(img), interpolation='nearest')
            ax1.set_title('original patch')
            noise = torch.randn_like(img)
            ax2.imshow(tensor_to_array(img + c.NOISE_INTENSITY * noise), interpolation='nearest')
            ax2.set_title('blurred patch')

            plt.pause(1)
        plt.show()
        
    def test_intrinsic(self):
        intrinsic_module = img_transfo.IntrisicModule()
        _, (ax1, ax2) = plt.subplots(1, 2)
        img = torch.zeros(1, 3, 224, 224)
        img[0, :, 112-40:112+40, 112-40:112+40] = torch.rand(3, 80, 80)
        plt.suptitle("TEST INTRINSIC")
        for _ in range(100):
            intrinsic_module.jitter()
            ax1.imshow(tensor_to_array(img), interpolation='nearest')
            ax1.set_title('original patch')
            ax2.imshow(tensor_to_array(intrinsic_module(img)), 
                       interpolation='nearest')
            ax2.set_title('intrinsic modif patch')

            plt.pause(1)
        plt.show()
        
    def test_extrinsic(self):
        extrinsic_module = img_transfo.ExtrinsicModule(normalize=False)
        _, (ax1, ax2) = plt.subplots(1, 2)
        img = torch.zeros(1, 3, 224, 224)
        img[0, :, 112-40:112+40, 112-40:112+40] = torch.rand(3, 80, 80)
        plt.suptitle("TEST EXTRINSIC")
        for _ in range(100):
            extrinsic_module.jitter()
            ax1.imshow(tensor_to_array(img), interpolation='nearest')
            ax1.set_title('original patch')
            ax2.imshow(tensor_to_array(extrinsic_module(img)), 
                       interpolation='nearest')
            ax2.set_title('extrinsic modif patch')

            plt.pause(1)
        plt.show()
        
    def test_global(self):
        intrinsic_module = img_transfo.IntrisicModule()
        extrinsic_module = img_transfo.ExtrinsicModule(normalize=False)
        _, (ax1, ax2) = plt.subplots(1, 2)
        img = torch.zeros(1, 3, 224, 224)
        img[0, :, 112-40:112+40, 112-40:112+40] = torch.rand(3, 80, 80)
        plt.suptitle("TEST GLOBAL")
        for _ in range(100):
            intrinsic_module.jitter()
            extrinsic_module.jitter()
            ax1.imshow(tensor_to_array(img), interpolation='nearest')
            ax1.set_title('original patch')
            ax2.imshow(tensor_to_array(extrinsic_module(intrinsic_module(img))), 
                       interpolation='nearest')
            ax2.set_title('extrinsic modif patch')

            plt.pause(1)
        plt.show()

        
class Trainer(unittest.TestCase):
    def setUp(self):
        self.patch_trainer = new_patch.PatchTrainer(patch_relative_size=0.1)
                                                    
        self.patch_trainer_flee = new_patch.PatchTrainer(mode=c.Mode.FLEE,
                                                         threshold=0,
                                                         patch_relative_size=0.05)
    def test_transformation(self):
        _, (ax1, ax2) = plt.subplots(1, 2)
        plt.suptitle("TEST TRANSFORMATION")
        trainer = self.patch_trainer
        for _ in range(100):
            ax1.imshow(u.tensor_to_array(trainer.patch))
            transformed, _ = trainer.transformation_tool.random_transform(trainer.patch)
            ax2.imshow(u.tensor_to_array(transformed))
            plt.pause(0.5)
            
    
        
    



if __name__ == '__main__':
    unittest.main()
