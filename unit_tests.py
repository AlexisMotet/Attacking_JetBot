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
        path_dataset = 'C:\\Users\\alexi\\PROJET_3A\\Projet Adversarial Patch\\Project Adverserial Patch\\Collision Avoidance\\dataset'
        path_model = 'C:\\Users\\alexi\\PROJET_3A\\Projet Adversarial Patch\\Project Adverserial Patch\\Collision Avoidance\\best_model_extended.pth'
        
        self.image_dim = 224
        self.patch_relative_size = 70/100
        self.patch_desc = patch.PatchDesc(self.image_dim, self.patch_relative_size)
        self.model, self.train_loader, self.test_loader = load_model.load_model(path_dataset, path_model)
        self.normalization_transform = torchvision.transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        self.unnormalization_transform = torchvision.transforms.Compose([
            torchvision.transforms.Normalize(mean = [0., 0., 0.], std = [1/0.229, 1/0.224, 1/0.225]),
            torchvision.transforms.Normalize(mean = [-0.485, -0.456, -0.406], std = [1., 1., 1.]),
            ])
        
        self.color_jitter_transform = torchvision.transforms.ColorJitter(0.1, 0.1, 0.1, 0.1)
    
    def test_normalization(self):
        _, (ax1, ax2, ax3) = plt.subplots(1, 3)
        for _, (img, _) in enumerate(self.train_loader):
            ax1.imshow(tensor_to_numpy_array(img), interpolation='nearest')
            ax1.set_title('original image')
            
            normalized_img = self.normalization_transform(img)    
            ax2.imshow(tensor_to_numpy_array(normalized_img), interpolation='nearest')
            ax2.set_title('normalized image')
            
            unnormalized_img = image_transformation.unnormalize(normalized_img)
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
            
            jitter_img = self.color_jitter_transform(img)    
            ax2.imshow(tensor_to_numpy_array(jitter_img), interpolation='nearest')
            ax2.set_title('after jitter')

            plt.show()
            plt.close()
            break
        
    def test_distorsion(self):
        
        _, (ax1, ax2) = plt.subplots(1, 2)
        
        transform = self.patch_desc.random_transform()
    
        empty_img_patch = transform['empty_image_patch']
        ax1.imshow(tensor_to_numpy_array(empty_img_patch), interpolation='nearest')
        ax1.set_title('after transform')
        
        path_calibration = 'C:\\Users\\alexi\\PROJET_3A\\projet_CAS\\calibration\\'
        cam_mtx, dist_coef = calibration.distorsion.load_coef(path_calibration)
        distorded_patch = calibration.distorsion.distort_patch(cam_mtx, dist_coef, empty_img_patch)
        ax2.imshow(tensor_to_numpy_array(distorded_patch), interpolation='nearest')
        ax2.set_title('after distorsion')
        plt.show()
        plt.close()

        

class PatchTestCase(unittest.TestCase):
    def setUp(self):
        self.show = True
        self.image_dim = 224
        self.patch_relative_size = 5/100
        self.patch_desc = patch.PatchDesc(self.image_dim, self.patch_relative_size)
        
        path_dataset = 'C:\\Users\\alexi\\PROJET_3A\\Projet Adversarial Patch\\Project Adverserial Patch\\Collision Avoidance\\dataset'
        path_model = 'C:\\Users\\alexi\\PROJET_3A\\Projet Adversarial Patch\\Project Adverserial Patch\\Collision Avoidance\\best_model_extended.pth'
        self.model, self.train_loader, self.test_loader = load_model.load_model(path_dataset, path_model)
    
    def test_initialized_patch(self):
        img_size = self.image_dim**2    
        patch_size = img_size * self.patch_relative_size    
        patch_dim = int(patch_size**0.5)
        
        assert self.patch_desc.patch.shape == (patch_dim, patch_dim, 3)
        
        assert torch.all(self.patch_desc.patch >= 0) and torch.all(self.patch_desc.patch < 1)
    
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
        p_pos, empty_img_patch = self.patch_desc.random_translation_in_empty_image()
        
        assert p_pos['x'] < self.image_dim - self.patch_desc.patch_dim
        assert p_pos['y'] < self.image_dim - self.patch_desc.patch_dim
        
        x = p_pos['x']
        y = p_pos['y']
        empty_img_patch[0, :, x:x+5, y:y+5] = torch.ones((1, 3, 5, 5))
        
        plt.imshow(tensor_to_numpy_array(empty_img_patch))
        plt.title('after translation')
        if (self.show):
            plt.show()
        plt.close()
        
    
    def test_mask(self):
        _, (ax1, ax2) = plt.subplots(1, 2)
        _, empty_img_patch = self.patch_desc.random_translation_in_empty_image()
        
        ax1.imshow(tensor_to_numpy_array(empty_img_patch), interpolation='nearest')
        ax1.set_title('empty image patch')
        
        mask = self.patch_desc.get_mask(empty_img_patch)
        
        ax2.imshow(tensor_to_numpy_array(mask), interpolation='nearest')
        ax2.set_title('mask')
        if (self.show):
            plt.show()
        plt.close()
        

    def test_transform(self):
        _, (ax1, ax2, ax3) = plt.subplots(1, 3)
        patch_dim = self.patch_desc.patch_dim
        
        p = self.patch_desc.patch
        p[0, :, :patch_dim//10, :patch_dim//10] = torch.ones((1, 3, patch_dim//10, patch_dim//10))
        
        ax1.imshow(tensor_to_numpy_array(p), interpolation='nearest')
        ax1.set_title('before transform')
        
        transform = self.patch_desc.random_transform()
        
        empty_img_patch = transform['empty_image_patch']
        
        ax2.imshow(tensor_to_numpy_array(empty_img_patch), interpolation='nearest')
        ax2.set_title('after transform')
        
        mask = transform['mask']
        
        ax3.imshow(tensor_to_numpy_array(mask), interpolation='nearest')
        ax3.set_title('mask')
        if (self.show):
            plt.show()
        plt.close()

    def test_batch_attack(self):
        
        self.model.eval()
        c = 0
        for _, (img, true_label) in enumerate(self.train_loader):
           
            vector_scores = self.model(img)
            model_label = torch.argmax(vector_scores.data).item()
            if model_label is not true_label.item() :
                continue
            if model_label is self.patch_desc.target_class :
                continue
                
            transform = self.patch_desc.random_transform()
            empty_img_p = transform['empty_image_patch'].clone()
            mask = transform['mask']

            adv_img, vector_proba = self.patch_desc.batch_attack(self.model, img, empty_img_p, mask)
            
            x, y = transform['patch_pos']['x'], transform['patch_pos']['y']
            self.patch = adv_img[:, :, x:x + self.patch_desc.patch_dim, y:y + self.patch_desc.patch_dim]
            
            assert torch.all(vector_proba >= 0) and torch.all(vector_proba <= 1)
            
            _, (ax1, ax2) = plt.subplots(2, 2)
            ax1[0].imshow(tensor_to_numpy_array(transform['empty_image_patch']), interpolation='nearest')
            ax1[0].set_title('avant entrainement')
            ax1[1].imshow(tensor_to_numpy_array(empty_img_p), interpolation='nearest')
            ax1[1].set_title('apres entrainement')
            
            ax2[0].imshow(tensor_to_numpy_array(img), interpolation='nearest')
            ax2[0].set_title('img')
            ax2[1].imshow(tensor_to_numpy_array(adv_img), interpolation='nearest')
            ax2[1].set_title('adv img')
            if (self.show):
                plt.show()
                
                
if __name__ == '__main__':
    unittest.main()