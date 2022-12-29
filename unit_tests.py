import unittest
import new_patch
import numpy as np
import torch
import matplotlib.pyplot as plt
import distorsion
import total_variation
import printability

path_model = 'U:\\PROJET_3A\\projet_NOUINOU_MOTET\\imagenette2-160_model.pth'
path_dataset = 'U:\\PROJET_3A\\imagenette2-160\\train'
path_calibration = 'U:\\PROJET_3A\\projet_NOUINOU_MOTET\\\calibration\\'
path_printable_vals = 'U:\\PROJET_3A\\projet_NOUINOU_MOTET\\printable_vals.dat'

def tensor_to_numpy_array(tensor):
    tensor = torch.squeeze(tensor)
    array = tensor.detach().cpu().numpy()
    return np.transpose(array, (1, 2, 0))

class ImageTransformationTestCase(unittest.TestCase):
    def setUp(self):
        self.patch_trainer = new_patch.PatchTrainer(path_model, path_dataset, path_calibration, distort=True)
        
    def test_normalization(self):
        _, (ax1, ax2) = plt.subplots(1, 2)
        for img, _ in self.patch_trainer.train_loader :
            ax1.imshow(tensor_to_numpy_array(img), interpolation='nearest')
            ax1.set_title('original image')
            
            normalized_img = self.patch_trainer.normalize(img)
            ax2.imshow(tensor_to_numpy_array(normalized_img), interpolation='nearest')
            ax2.set_title('normalized image')
            
            plt.show()
            plt.close()
            break

    def test_distorsion(self):
        _, (ax1, ax2, ax3, ax4) = plt.subplots(1, 4)
        for row0 in range(0, new_patch.image_dim - self.patch_trainer.patch_dim, 10):
            for col0 in range(0, new_patch.image_dim - self.patch_trainer.patch_dim, 10):
                empty_with_patch = torch.zeros(1, 3, new_patch.image_dim, new_patch.image_dim)
                empty_with_patch[0, :, row0:row0 + self.patch_trainer.patch_dim, 
                                 col0:col0 + self.patch_trainer.patch_dim] = self.patch_trainer.patch

                ax1.imshow(tensor_to_numpy_array(empty_with_patch), interpolation='nearest')
                ax1.set_title('empty image patch')

                empty_with_patch_distorded, map = self.patch_trainer.distorsion_tool.distort(empty_with_patch)
                ax2.imshow(tensor_to_numpy_array(empty_with_patch_distorded), interpolation='nearest')
                ax2.set_title('after distorsion')
                
                empty_with_patch = self.patch_trainer.distorsion_tool.undistort(empty_with_patch_distorded, map, empty_with_patch)
                ax3.imshow(tensor_to_numpy_array(empty_with_patch), interpolation='nearest')
                ax3.set_title('after undistorsion')

                empty_with_patch_distorded = self.patch_trainer.distorsion_tool.distort_with_map(empty_with_patch, map)
                ax4.imshow(tensor_to_numpy_array(empty_with_patch_distorded), interpolation='nearest')
                ax4.set_title('with map')
                
                plt.pause(0.05)
        plt.show()
        plt.close()

class PatchTestCase(unittest.TestCase):
    def setUp(self):
        self.patch_trainer = new_patch.PatchTrainer(path_model, path_dataset, path_calibration, distort=True)
        
    
    def test_initialized_patch(self):
        plt.imshow(tensor_to_numpy_array(self.patch_trainer.patch), interpolation='nearest')
        plt.show()
        plt.close()
        
    def test_translation(self):
        empty_with_patch, row0, col0 = self.patch_trainer.random_transform()
        empty_with_patch[0, :, row0:row0+5, col0:col0+5] = torch.ones((1, 3, 5, 5))
        
        plt.imshow(tensor_to_numpy_array(empty_with_patch))
        plt.title('after translation')
        
        plt.show()
        plt.close()
        
    
    def test_mask(self):
        _, (ax1, ax2) = plt.subplots(1, 2)
        empty_with_patch, row0, col0 = self.patch_trainer.random_transform()
        empty_with_patch_distorded, map = distorsion.distort_patch(self.patch_trainer.cdistort, self.patch_trainer.cam_mtx, 
                                                                   self.patch_trainer.dist_coefs, empty_with_patch)
        mask = self.patch_trainer.get_mask(empty_with_patch_distorded)
        
        ax1.imshow(tensor_to_numpy_array(empty_with_patch_distorded), interpolation='nearest')
        ax1.set_title('empty image patch')
        
        ax2.imshow(tensor_to_numpy_array(mask), interpolation='nearest')
        ax2.set_title('mask')
        
        plt.show()
        plt.close()

class VariousTestCase(unittest.TestCase):
    def setUp(self):
        self.patch_trainer = new_patch.PatchTrainer(path_model, path_dataset, path_calibration, 
                                                    path_printable_vals, distort=True)

    def test_total_variation(self):
        _, (ax1, ax2) = plt.subplots(1, 2)
        for _ in range(100):
            tv_loss, patch_tv_grad = total_variation.total_variation(self.patch_trainer.patch)
            print(patch_tv_grad.shape)
            ax1.imshow(tensor_to_numpy_array(self.patch_trainer.patch), interpolation='nearest')
            ax1.set_title('patch tv loss : %f' % tv_loss)
            
            ax2.imshow(tensor_to_numpy_array(patch_tv_grad), interpolation='nearest')
            ax2.set_title('total variation grad')

            self.patch_trainer.patch -= 0.5 * patch_tv_grad

            plt.pause(0.5)

        plt.show()
        plt.close()

    def test_printability(self):
        _, (ax1, ax2) = plt.subplots(1, 2)
        for _ in range(100):
            loss, grad = self.patch_trainer.printability_tool.score(self.patch_trainer.patch)
            
            ax1.imshow(tensor_to_numpy_array(self.patch_trainer.patch), interpolation='nearest')
            ax1.set_title('patch printability loss : %f' % loss)
            
            ax2.imshow(tensor_to_numpy_array(grad), interpolation='nearest')
            ax2.set_title('grad')

            self.patch_trainer.patch -= 0.05 * grad

            plt.pause(1)

        plt.show()
        plt.close()
    
    def test_printability_1(self):
        _, (ax1, ax2, ax3) = plt.subplots(1, 3)
        im = torch.rand(1, 3, 3, 3)

        print_tool = printability.PrintabilityTool(path_printable_vals, 3)
        colors = print_tool.colors[:, :, 0, 0]
        colors = colors.reshape(5, 6, 3)

        e = 0.001

        for i in range(500):
            loss, grad = print_tool.score(im)
            im -= e * grad
            print('iteration %d : loss %f' % (i, loss))
        
        ax1.imshow(tensor_to_numpy_array(im), interpolation='nearest')
        ax1.set_title('patch printability loss : %f' % loss)

        ax2.imshow(colors, interpolation='nearest')
        ax2.set_title('color set')

        ax3.imshow(tensor_to_numpy_array(grad), interpolation='nearest')
        ax3.set_title('grad')

        plt.show()
        plt.close()

class GradientCheckTestCase(unittest.TestCase):
    def setUp(self):
        self.patch_trainer = new_patch.PatchTrainer(path_model, path_dataset, path_calibration, path_printable_vals, distort=True)
    def test_normalization(self):
        _, (ax1, ax2) = plt.subplots(1, 2)
        e = 10e-5
        ide = np.ones_like(self.patch_trainer.patch) * e
        f1 = self.patch_trainer.normalize(self.patch_trainer.patch + ide)
        f2 = self.patch_trainer.normalize(self.patch_trainer.patch - ide)
        deltaf = (1/(2*e)) * (f1 - f2)
        ax1.imshow(tensor_to_numpy_array(deltaf), interpolation='nearest')
        ones = torch.ones_like(self.patch_trainer.patch)
        grad = self.patch_trainer.grad_normalization(ones)
        ax2.imshow(tensor_to_numpy_array(grad), interpolation='nearest')
        plt.show()
        plt.close()

    def test_total_variation(self):
        _, (ax1, ax2, ax3) = plt.subplots(1, 3)
        e = 10e-5
        num_grad = torch.zeros_like(self.patch_trainer.patch)
        el = np.zeros_like(self.patch_trainer.patch)
        for i in range(self.patch_trainer.patch_dim) :
            for j in range(self.patch_trainer.patch_dim) :
                for c in range(3):
                    el_cpy = el.copy()
                    el_cpy[0, c, i, j] = e * np.ones((1, 1))
                    f1, _ = total_variation.total_variation(self.patch_trainer.ctotal_variation, 
                                                            self.patch_trainer.patch + el_cpy)
                    f2, _ = total_variation.total_variation(self.patch_trainer.ctotal_variation, 
                                                            self.patch_trainer.patch - el_cpy)
                    deltaf = (1/(2*e)) * (f1 - f2)
                    num_grad[0, c, i, j] = deltaf
        ax1.imshow(tensor_to_numpy_array(num_grad), interpolation='nearest')
        _, grad = total_variation.total_variation(self.patch_trainer.ctotal_variation, 
                                                     self.patch_trainer.patch)
        ax2.imshow(tensor_to_numpy_array(grad), interpolation='nearest')
        ax3.imshow(tensor_to_numpy_array(grad - num_grad), interpolation='nearest')
        plt.show()
        plt.close()

if __name__ == '__main__':
    unittest.main()