import unittest
import new_patch
import constants.constants as c
import time
import torch
import matplotlib.pyplot as plt
import utils.utils as u

def tensor_to_array(tensor):
    max = torch.max(tensor)
    if max > 1 :
        print("Tensor is out of image domain with max value %f" % max)
    min = torch.min(tensor)
    if min < 0 :
        print("Tensor is out of image domain with min value %f" % min)
    return u.tensor_to_array(tensor)

class Trainer(unittest.TestCase):
    def setUp(self):
        self.trainer = new_patch.PatchTrainer(path_image_init="dog.JPEG",
                                              patch_relative_size=0.2)

    def test_normalization(self):
        _, (ax1, ax2) = plt.subplots(1, 2)
        plt.suptitle("TEST NORMALIZATION")
        ax1.imshow(tensor_to_array(self.trainer.patch), interpolation='nearest')
        ax1.set_title('original image')

        t0 = time.time()
        normalized = self.trainer.normalize(self.trainer.patch)
        t1 = time.time()
        ax2.imshow(tensor_to_array(normalized), interpolation='nearest')
        ax2.set_title('normalized\ndeltat=%.2fms' % ((t1 - t0)*1e3))

        plt.show()
        
    def test_brightness_contrast(self):
        _, (ax1, ax2) = plt.subplots(1, 2)
        plt.suptitle("TEST BRIGHTNESS")
        for _ in range(100):
            ax1.imshow(tensor_to_array(self.trainer.patch), interpolation='nearest')
            ax1.set_title('original image')
            t0 = time.time()
            modified = self.trainer.patch_processing_module.brightness_contrast(self.trainer.patch)
            t1 = time.time()
            ax2.imshow(tensor_to_array(modified), interpolation='nearest')
            ax2.set_title('brightness and contrast changed\ndeltat=%.2fms' % ((t1 - t0)*1e3))
            self.trainer.patch_processing_module.jitter()
            plt.pause(1)
        plt.show()
        
    def test_blurring(self):
        _, (ax1, ax2) = plt.subplots(1, 2)
        plt.suptitle("TEST BLURRING")
        for _ in range(100):
            ax1.imshow(tensor_to_array(self.trainer.patch), interpolation='nearest')
            ax1.set_title('original patch')
            t0 = time.time()
            blurred = self.trainer.patch_processing_module.blurring(self.trainer.patch)
            t1 = time.time()
            ax2.imshow(tensor_to_array(blurred), interpolation='nearest')
            ax2.set_title('blurred patch\ndeltat=%.2fms' % ((t1 - t0)*1e3))
            self.trainer.patch_processing_module.jitter()
            plt.pause(1)
        plt.show()
        
    def test_noise(self):
        _, (ax1, ax2) = plt.subplots(1, 2)
        plt.suptitle("TEST NOISE")
        for _ in range(100):
            ax1.imshow(tensor_to_array(self.trainer.patch), interpolation='nearest')
            ax1.set_title('original patch')
            t0 = time.time()
            noisy = self.trainer.patch_processing_module.noising(self.trainer.patch)
            t1 = time.time()
            ax2.imshow(tensor_to_array(noisy), interpolation='nearest')
            ax2.set_title('noisy patch\ndeltat=%.2fms' % ((t1 - t0)*1e3))
            self.trainer.patch_processing_module.jitter()
            plt.pause(1)
        plt.show()
        
    def test_patch_processing_module(self):
        _, (ax1, ax2) = plt.subplots(1, 2)
        plt.suptitle("TEST patch_processing_module")
        for _ in range(100):
            ax1.imshow(tensor_to_array(self.trainer.patch), interpolation='nearest')
            ax1.set_title('original patch')
            t0 = time.time()
            res = self.trainer.patch_processing_module(self.trainer.patch)
            t1 = time.time()
            ax2.imshow(tensor_to_array(res), 
                       interpolation='nearest')
            ax2.set_title('patch processing module modif\ndeltat=%.2fms' % ((t1 - t0)*1e3))
            self.trainer.patch_processing_module.jitter()
            plt.pause(1)
        plt.show()
    
    def test_zone(self):
        _, (ax1) = plt.subplots(1, 1)
        plt.suptitle("TEST RANDOM ATTACK ZONE")
        x0, x1 = c.consts["X_TOP_LEFT"], c.consts["X_BOTTOM_RIGHT"]
        y0, y1 = c.consts["Y_TOP_LEFT"], c.consts["Y_BOTTOM_RIGHT"]
        zeros = torch.zeros(1, 3, 224, 224)
        zeros[0, :, y0:y1, x0:x1] = torch.ones(3, y1 - y0, x1 - x0)
        ax1.imshow(tensor_to_array(zeros))
        plt.show()
                                                      
    def test_transformation(self):
        _, (ax1, ax2, ax3) = plt.subplots(1, 3)
        plt.suptitle("TEST TRANSFORMATION")
        for _ in range(100):
            ax1.imshow(tensor_to_array(self.trainer.patch))
            t0 = time.time()
            transformed, map_ = self.trainer.transfo_tool.random_transform(self.trainer.patch)
            t1 = time.time()
            ax2.imshow(tensor_to_array(transformed))
            ax2.set_title('transformation\ndeltat=%.2fms' % ((t1 - t0)*1e3))
            new_img = self.trainer.transfo_tool.undo_transform(self.trainer.patch, transformed, map_)
            ax3.imshow(tensor_to_array(new_img))
            dist = (self.trainer.patch - new_img).pow(2).sum().sqrt()
            ax3.set_title("euclidian dist : %f" % dist)
            plt.pause(1)
        plt.show()
        
    def test_transformation_3d(self):
        ax = plt.figure().add_subplot(projection='3d')

        
        
        
        
        
        
    def test_total_variation(self):
        _, (ax1, ax2) = plt.subplots(1, 2)
        plt.suptitle("TEST TV LAMBDA_TV=%f" % c.consts["LAMBDA_TV"])
        original = self.trainer._get_patch()
        x = original.clone()
        x.requires_grad = True
        for _ in range(100):
            tv_loss = self.trainer.tv_module(x)
            tv_loss.backward()
            with torch.no_grad():
                x -= c.consts["LAMBDA_TV"] * x.grad
            x.grad.zero_()
            ax1.imshow(tensor_to_array(original))
            ax2.imshow(tensor_to_array(x))
            plt.pause(1)
        plt.show() 
        
    def test_printability(self):
        _, (ax1, ax2) = plt.subplots(1, 2)
        plt.suptitle("TEST PRINT LAMBDA_PRINT=%f" % c.consts["LAMBDA_PRINT"])
        original = self.trainer._get_patch()
        x = original.clone()
        x.requires_grad = True
        for _ in range(100):
            print_loss = self.trainer.print_module(x)
            print_loss.backward()
            with torch.no_grad():
                x -= c.consts["LAMBDA_PRINT"] * x.grad
            x.grad.zero_()
            ax1.imshow(tensor_to_array(original))
            ax2.imshow(tensor_to_array(x))
            plt.pause(1)
        plt.show()
if __name__ == '__main__':
    unittest.main()
