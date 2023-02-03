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
        
    def test_patch_processing_module(self):
        patch_processing_module = img_transfo.IntrisicModule()
        _, (ax1, ax2) = plt.subplots(1, 2)
        img = torch.zeros(1, 3, 224, 224)
        img[0, :, 112-40:112+40, 112-40:112+40] = torch.rand(3, 80, 80)
        plt.suptitle("TEST patch_processing_module")
        for _ in range(100):
            patch_processing_module.jitter()
            ax1.imshow(tensor_to_array(img), interpolation='nearest')
            ax1.set_title('original patch')
            ax2.imshow(tensor_to_array(patch_processing_module(img)), 
                       interpolation='nearest')
            ax2.set_title('patch_processing_module modif patch')

            plt.pause(1)
        plt.show()
        
    def test_extrinsic(self):
        image_processing_module = img_transfo.ExtrinsicModule(normalize=False)
        _, (ax1, ax2) = plt.subplots(1, 2)
        img = torch.zeros(1, 3, 224, 224)
        img[0, :, 112-40:112+40, 112-40:112+40] = torch.rand(3, 80, 80)
        plt.suptitle("TEST EXTRINSIC")
        for _ in range(100):
            image_processing_module.jitter()
            ax1.imshow(tensor_to_array(img), interpolation='nearest')
            ax1.set_title('original patch')
            ax2.imshow(tensor_to_array(image_processing_module(img)), 
                       interpolation='nearest')
            ax2.set_title('extrinsic modif patch')

            plt.pause(1)
        plt.show()
        
    def test_global(self):
        patch_processing_module = img_transfo.IntrisicModule()
        image_processing_module = img_transfo.ExtrinsicModule(normalize=False)
        _, (ax1, ax2) = plt.subplots(1, 2)
        img = torch.zeros(1, 3, 224, 224)
        img[0, :, 112-40:112+40, 112-40:112+40] = torch.rand(3, 80, 80)
        plt.suptitle("TEST GLOBAL")
        for _ in range(100):
            patch_processing_module.jitter()
            image_processing_module.jitter()
            ax1.imshow(tensor_to_array(img), interpolation='nearest')
            ax1.set_title('original patch')
            ax2.imshow(tensor_to_array(image_processing_module(patch_processing_module(img))), 
                       interpolation='nearest')
            ax2.set_title('extrinsic modif patch')

            plt.pause(1)
        plt.show()

class Trainer(unittest.TestCase):
    def setUp(self):
        self.patch_trainer = new_patch.PatchTrainer(patch_relative_size=0.05)
                                                    
    def test_transformation(self):
        _, (ax1, ax2) = plt.subplots(1, 2)
        plt.suptitle("TEST TRANSFORMATION")
        trainer = self.patch_trainer
        for _ in range(100):
            ax1.imshow(u.tensor_to_array(trainer.patch))
            transformed, _ = trainer.transformation_tool.random_transform(trainer.patch)
            ax2.imshow(u.tensor_to_array(transformed))
            plt.pause(0.5)
            
    def test_attack(self):
        _, (ax1, ax2, ax3, ax4) = plt.subplots(1, 4)
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
            for i in range(trainer.max_iterations + 1) :
                torch.clamp(transformed, 0, 1)
                modified = trainer.patch_processing_module(transformed)
                attacked = torch.mul(1 - mask, img) + \
                        torch.mul(mask, modified)
                normalized = trainer.image_processing_module(attacked)
                vector_scores = trainer.model(normalized)
                vector_proba = torch.nn.functional.softmax(vector_scores, dim=1)
                target_proba = float(vector_proba[0, trainer.target_class])
                loss = -torch.nn.functional.log_softmax(vector_scores, dim=1)
                loss[0, trainer.target_class].backward()
                with torch.no_grad():
                    transformed -= transformed.grad
                transformed.grad.zero_()
                print(target_proba)
            ax1.imshow(u.tensor_to_array(img))
            ax2.imshow(u.tensor_to_array(transformed))
            ax3.imshow(u.tensor_to_array(modified))
            ax4.imshow(u.tensor_to_array(attacked))
            self.patch = trainer.transformation_tool.undo_transform(trainer.patch, 
                                                            transformed.detach(),
                                                            map_)
            plt.pause(0.1)
        
    



if __name__ == '__main__':
    unittest.main()
