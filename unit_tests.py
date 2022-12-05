import unittest
import patch
import numpy as np
import torch
import torchvision
import matplotlib.pyplot as plt
import load_model
import image_transformation
import calibration.distorsion

def tensor_to_numpy_array(tensor):
    tensor = torch.squeeze(tensor)
    array = tensor.detach().cpu().numpy()
    return np.transpose(array, (1, 2, 0))

class ImageTransformationTestCase(unittest.TestCase):
    def setUp(self):
        path_dataset = 'U:\\PROJET_3A\\projet_BONTEMPS_SCHAMPHELEIRE\\Project Adverserial Patch\\Collision Avoidance\\dataset'
        path_model = 'U:\\PROJET_3A\\projet_BONTEMPS_SCHAMPHELEIRE\\Project Adverserial Patch\\Collision Avoidance\\best_model_extended.pth'
        path_calibration = 'U:\\PROJET_3A\\projet_NOUINOU_MOTET\calibration\\'
        
        self.image_dim = 224
        self.patch_relative_size = 5/100
        self.cam_mtx, self.dist_coef = calibration.distorsion.load_coef(path_calibration) 
        self.patch_desc = patch.PatchDesc(self.image_dim, self.patch_relative_size, self.cam_mtx, self.dist_coef)
        self.model, self.train_loader, _, _ = load_model.load_model(path_dataset, path_model)
        self.image_transformer = image_transformation.ImageTransformer()
        
    def test_normalization(self):
        _, (ax1, ax2, ax3) = plt.subplots(1, 3)
        for _, (img, _) in enumerate(self.train_loader):
            ax1.imshow(tensor_to_numpy_array(img), interpolation='nearest')
            ax1.set_title('original image')
            
            normalized_img = self.image_transformer.normalize(img)
            print(img.shape)
            ax2.imshow(tensor_to_numpy_array(normalized_img), interpolation='nearest')
            ax2.set_title('normalized image')
            
            unnormalized_img = self.image_transformer.unnormalize(normalized_img)
            ax3.imshow(tensor_to_numpy_array(unnormalized_img), interpolation='nearest')
            ax3.set_title('unnormalized image')
            plt.show()
            plt.close()
            break
    
    def test_jitter(self):
        _, (ax1, ax2) = plt.subplots(1, 2)
        for _, (img, _) in enumerate(self.train_loader):
            ax1.imshow(tensor_to_numpy_array(img), interpolation='nearest')
            ax1.set_title('original image')
            
            jitter_img = self.image_transformer.color_jitter_transform(img)    
            ax2.imshow(tensor_to_numpy_array(jitter_img), interpolation='nearest')
            ax2.set_title('after jitter')

            plt.show()
            plt.close()
            break
        
    def test_distorsion(self):
        _, (ax1, ax2, ax3) = plt.subplots(1, 3)
        
        for row0 in range(0, self.image_dim - self.patch_desc.patch_dim, 10):
            for col0 in range(0, self.image_dim - self.patch_desc.patch_dim, 10):
                empty_image_patch = self.patch_desc.get_empty_image_patch(row0, col0)

                ax1.imshow(tensor_to_numpy_array(empty_image_patch), interpolation='nearest')
                ax1.set_title('empty image patch')
                
                distorded_patch, map_ = calibration.distorsion.distort_patch(self.cam_mtx, self.dist_coef, empty_image_patch)
                ax2.imshow(tensor_to_numpy_array(distorded_patch), interpolation='nearest')
                ax2.set_title('after distorsion')
                
                undistorded_patch = calibration.distorsion.undistort_patch(empty_image_patch, distorded_patch, map_)
                ax3.imshow(tensor_to_numpy_array(undistorded_patch), interpolation='nearest')
                ax3.set_title('after undistorsion')
                plt.pause(0.05)
        plt.show()
        plt.close()

class PatchTestCase(unittest.TestCase):
    def setUp(self):
        path_dataset = 'U:\\PROJET_3A\\projet_BONTEMPS_SCHAMPHELEIRE\\Project Adverserial Patch\\Collision Avoidance\\dataset'
        path_model = 'U:\\PROJET_3A\\projet_BONTEMPS_SCHAMPHELEIRE\\Project Adverserial Patch\\Collision Avoidance\\best_model_extended.pth'
        path_calibration = 'U:\\PROJET_3A\\projet_NOUINOU_MOTET\calibration\\'
        
        self.show = True
        self.image_dim = 224
        self.patch_relative_size = 5/100
        self.cam_mtx, self.dist_coef = calibration.distorsion.load_coef(path_calibration)
        self.patch_desc = patch.PatchDesc(self.image_dim, self.patch_relative_size, self.cam_mtx, self.dist_coef)
        self.model, self.train_loader, _, _ = load_model.load_model(path_dataset, path_model)
    
    def test_initialized_patch(self):
        img_size = self.image_dim**2    
        patch_size = img_size * self.patch_relative_size    
        patch_dim = int(patch_size**0.5)
        
        assert self.patch_desc.patch.shape == (1, 3, patch_dim, patch_dim)
        
        assert torch.all(self.patch_desc.patch >= 0) and torch.all(self.patch_desc.patch < 1)
        plt.imshow(tensor_to_numpy_array(self.patch_desc.patch), interpolation='nearest')
        plt.show()
        plt.close()
        
    def test_rotation(self):
        _, (ax1, ax2) = plt.subplots(1, 2)
        patch_dim = self.patch_desc.patch_dim
        
        p = self.patch_desc.patch
        p[0, :, :patch_dim//5, :patch_dim//5] = torch.ones((1, 3, patch_dim//5, patch_dim//5))
        
        ax1.imshow(tensor_to_numpy_array(p), interpolation='nearest')
        ax1.set_title('before rotation')
        
        self.patch_desc.random_rotation()
        
        ax2.imshow(tensor_to_numpy_array(p), interpolation='nearest')
        ax2.set_title('after rotation')
        
        if (self.show):
            plt.show()
        plt.close()
        
        
    def test_translation(self):
        row0, col0 = self.patch_desc.random_translation()
        
        assert row0 < self.image_dim - self.patch_desc.patch_dim
        assert col0 < self.image_dim - self.patch_desc.patch_dim
        
        empty_img_patch = self.patch_desc.get_empty_image_patch(row0, col0)
        
        empty_img_patch[0, :, row0:row0+5, col0:col0+5] = torch.ones((1, 3, 5, 5))
        
        plt.imshow(tensor_to_numpy_array(empty_img_patch))
        plt.title('after translation')
        if (self.show):
            plt.show()
        plt.close()
        
    
    def test_mask(self):
        _, (ax1, ax2) = plt.subplots(1, 2)
        row0, col0 = self.patch_desc.random_translation()
        empty_img_p = self.patch_desc.get_empty_image_patch(row0, col0)
        distorded_patch, _ = calibration.distorsion.distort_patch(self.cam_mtx, self.dist_coef, empty_img_p)
        
        ax1.imshow(tensor_to_numpy_array(distorded_patch), interpolation='nearest')
        ax1.set_title('empty image patch')
        
        mask = self.patch_desc.get_mask(distorded_patch)
        
        ax2.imshow(tensor_to_numpy_array(mask), interpolation='nearest')
        ax2.set_title('mask')
        if (self.show):
            plt.show()
        plt.close()
        

    def test_transform(self):
        _, (ax1, ax2, ax3, ax4, ax5) = plt.subplots(1, 5)
        patch_dim = self.patch_desc.patch_dim
        
        p = self.patch_desc.patch
        p[0, :, :patch_dim//10, :patch_dim//10] = torch.ones((1, 3, patch_dim//10, patch_dim//10))
        
        ax1.imshow(tensor_to_numpy_array(p), interpolation='nearest')
        ax1.set_title('before transform')
        
        empty_img_p, distorded_patch, map_, mask, _, _ = self.patch_desc.random_transform()
        
        ax2.imshow(tensor_to_numpy_array(empty_img_p), interpolation='nearest')
        ax2.set_title('after translation')
        
        ax3.imshow(tensor_to_numpy_array(distorded_patch), interpolation='nearest')
        ax3.set_title('after distorsion')
        
        ax4.imshow(tensor_to_numpy_array(mask), interpolation='nearest')
        ax4.set_title('mask')
        
        undistorded_patch = calibration.distorsion.undistort_patch(empty_img_p, distorded_patch, map_)
        
        ax5.imshow(tensor_to_numpy_array(undistorded_patch), interpolation='nearest')
        ax5.set_title('after undistorsion')
        
        if (self.show):
            plt.show()
        plt.close()

    def test_image_attack(self):
        self.model.eval()
        for _, (img, true_label) in enumerate(self.train_loader):
            scores = self.model(self.patch_desc.image_transformer.normalize(img))
            model_label = torch.argmax(scores.data, dim=1).item()
            if model_label is not true_label.item() :
                continue
            if model_label is self.patch_desc.target_class :
                continue

            patch_clone = self.patch_desc.patch.clone()
            empty_img_p, distorded_patch, map_, mask, row0, col0 = self.patch_desc.random_transform()
            adv_img, _ = self.patch_desc.image_attack(self.model, img, distorded_patch, mask)

            distorded_patch = torch.mul(mask, empty_img_p)
            
            empty_img_p = calibration.distorsion.undistort_patch(empty_img_p, distorded_patch, map_)

            self.patch_desc.patch = empty_img_p[0, :, row0:row0+self.patch_desc.patch_dim, col0:col0+self.patch_desc.patch_dim]
                    
            _, (ax1, ax2) = plt.subplots(2, 2)
            ax1[0].imshow(tensor_to_numpy_array(patch_clone), interpolation='nearest')
            ax1[0].set_title('avant entrainement')
            ax1[1].imshow(tensor_to_numpy_array(self.patch_desc.patch), interpolation='nearest')
            ax1[1].set_title('apres entrainement')
            
            ax2[0].imshow(tensor_to_numpy_array(img), interpolation='nearest')
            ax2[0].set_title('img')
            ax2[1].imshow(tensor_to_numpy_array(adv_img), interpolation='nearest')
            ax2[1].set_title('adv img')
            if (self.show):
                plt.show()
                
                
if __name__ == '__main__':
    unittest.main()