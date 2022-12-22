import unittest
import patch
import numpy as np
import torch
import torchvision
import matplotlib.pyplot as plt
import load
import image_transformation
import calibration.distorsion

def tensor_to_numpy_array(tensor):
    tensor = torch.squeeze(tensor)
    array = tensor.detach().cpu().numpy()
    return np.transpose(array, (1, 2, 0))

class ImageTransformationTestCase(unittest.TestCase):
    def setUp(self):
        self.patch_desc = patch.PatchDesc()
        self.patch_desc.load_everything()
        
    def test_normalization(self):
        _, (ax1, ax2, ax3) = plt.subplots(1, 3)
        for img, _ in self.patch_desc.train_loader :
            ax1.imshow(tensor_to_numpy_array(img), interpolation='nearest')
            ax1.set_title('original image')
            
            normalized_img = self.patch_desc.image_transformer.normalize(img)
            ax2.imshow(tensor_to_numpy_array(normalized_img), interpolation='nearest')
            ax2.set_title('normalized image')
            
            unnormalized_img = self.patch_desc.image_transformer.unnormalize(normalized_img)
            ax3.imshow(tensor_to_numpy_array(unnormalized_img), interpolation='nearest')
            ax3.set_title('unnormalized image')
            plt.show()
            plt.close()
            break
    
    def test_jitter(self):
        _, (ax1, ax2) = plt.subplots(1, 2)
        for img, _ in self.patch_desc.train_loader :
            ax1.imshow(tensor_to_numpy_array(img), interpolation='nearest')
            ax1.set_title('original image')
            
            jitter_img = self.patch_desc.image_transformer.color_jitter_transform(img)    
            ax2.imshow(tensor_to_numpy_array(jitter_img), interpolation='nearest')
            ax2.set_title('after jitter')

            plt.show()
            plt.close()
            break
        
    def test_distorsion(self):
        _, (ax1, ax2, ax3) = plt.subplots(1, 3)
        for row0 in range(0, self.patch_desc.image_dim - self.patch_desc.patch_dim, 10):
            for col0 in range(0, self.patch_desc.image_dim - self.patch_desc.patch_dim, 10):
                empty_image_patch = self.patch_desc.get_empty_image_patch(row0, col0)

                ax1.imshow(tensor_to_numpy_array(empty_image_patch), interpolation='nearest')
                ax1.set_title('empty image patch')
                
                distorded_patch, map_ = calibration.distorsion.distort_patch(self.patch_desc.cam_mtx, 
                                                                             self.patch_desc.dist_coef, 
                                                                             empty_image_patch)
                ax2.imshow(tensor_to_numpy_array(distorded_patch), interpolation='nearest')
                ax2.set_title('after distorsion')
                
                undistorded_patch = calibration.distorsion.undistort_patch(empty_image_patch, 
                                                                           distorded_patch, map_)
                ax3.imshow(tensor_to_numpy_array(undistorded_patch), interpolation='nearest')
                ax3.set_title('after undistorsion')
                plt.pause(0.05)
        plt.show()
        plt.close()

class PatchTestCase(unittest.TestCase):
    def setUp(self):
        self.patch_desc = patch.PatchDesc()
        self.patch_desc.load_everything()
    
    def test_initialized_patch(self):
        assert self.patch_desc.patch.shape == (1, 3, self.patch_desc.patch_dim, self.patch_desc.patch_dim)
        
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
        
        plt.show()
        plt.close()
        
        
    def test_translation(self):
        row0, col0 = self.patch_desc.random_translation()
        
        assert row0 < self.patch_desc.image_dim - self.patch_desc.patch_dim
        assert col0 < self.patch_desc.image_dim - self.patch_desc.patch_dim
        
        empty_img_patch = self.patch_desc.get_empty_image_patch(row0, col0)
        
        empty_img_patch[0, :, row0:row0+5, col0:col0+5] = torch.ones((1, 3, 5, 5))
        
        plt.imshow(tensor_to_numpy_array(empty_img_patch))
        plt.title('after translation')
        
        plt.show()
        plt.close()
        
    
    def test_mask(self):
        _, (ax1, ax2) = plt.subplots(1, 2)
        row0, col0 = self.patch_desc.random_translation()
        empty_img_p = self.patch_desc.get_empty_image_patch(row0, col0)
        distorded_patch, _ = calibration.distorsion.distort_patch(self.patch_desc.cam_mtx, 
                                                                  self.patch_desc.dist_coef, 
                                                                  empty_img_p)
        
        ax1.imshow(tensor_to_numpy_array(distorded_patch), interpolation='nearest')
        ax1.set_title('empty image patch')
        
        mask = self.patch_desc.get_mask(distorded_patch)
        
        ax2.imshow(tensor_to_numpy_array(mask), interpolation='nearest')
        ax2.set_title('mask')
        
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
        
        plt.show()
        plt.close()

    def test_image_attack(self):
        self.patch_desc.model.eval()
        for img, true_label in self.patch_desc.train_loader :
            scores = self.patch_desc.model(self.patch_desc.image_transformer.normalize(img))
            model_label = torch.argmax(scores.data, dim=1).item()
            if model_label is not true_label.item() :
                continue
            if model_label is self.patch_desc.target_class :
                continue

            patch_clone = self.patch_desc.patch.clone()
            empty_img_p, distorded_patch, map_, mask, row0, col0 = self.patch_desc.random_transform()
            adv_img, _ = self.patch_desc.image_attack(img, distorded_patch, mask)

            distorded_patch = torch.mul(mask, empty_img_p)
            
            empty_img_p = calibration.distorsion.undistort_patch(empty_img_p, distorded_patch, map_)

            self.patch_desc.patch = empty_img_p[0, :, row0:row0+self.patch_desc.patch_dim, 
                                                col0:col0+self.patch_desc.patch_dim]
                    
            _, (ax1, ax2) = plt.subplots(2, 2)
            ax1[0].imshow(tensor_to_numpy_array(patch_clone), interpolation='nearest')
            ax1[0].set_title('avant entrainement')
            ax1[1].imshow(tensor_to_numpy_array(self.patch_desc.patch), interpolation='nearest')
            ax1[1].set_title('apres entrainement')
            
            ax2[0].imshow(tensor_to_numpy_array(img), interpolation='nearest')
            ax2[0].set_title('img')
            ax2[1].imshow(tensor_to_numpy_array(adv_img), interpolation='nearest')
            ax2[1].set_title('adv img')
            
            plt.show()
            break
                    
    class Distort(torch.nn.Module):
        def __init__(self, patch_desc) :
            super(PatchTestCase.Distort, self).__init__()
            self.patch_desc = patch_desc
            
        def forward(self, input1, input2):
            row0, col0 = self.patch_desc.random_translation()
            self.patch_desc.row0 = row0
            self.patch_desc.col0 = col0
            empty_img_p = torch.zeros(1, 3, self.patch_desc.image_dim, self.patch_desc.image_dim)
            empty_img_p[0, :, row0:row0 + self.patch_desc.patch_dim, 
                        col0:col0 + self.patch_desc.patch_dim] = input2
            distorded_patch, map_ = calibration.distorsion.distort_patch(self.patch_desc.cam_mtx, 
                                                                         self.patch_desc.dist_coef, 
                                                                         empty_img_p)
            self.patch_desc.map_ = map_
            mask = torch.zeros_like(distorded_patch)
            mask[distorded_patch != 0] = 1
            adv_img = torch.mul((1-mask), input1) + torch.mul(mask, distorded_patch)
            adv_img = self.patch_desc.image_transformer.normalize(adv_img)
            self.patch_desc.adv_img = adv_img
            return self.patch_desc.model(adv_img)
            
    def test_specific_model(self):
        _, (ax1, ax2, ax3, ax4) = plt.subplots(1, 4)
        #model = torch.nn.Sequential(
        #    PatchTestCase.Distort(self.patch_desc.cam_mtx, self.patch_desc.dist_coef),
        #    self.patch_desc.model,
        #)
        coef = 1e9
        self.patch_desc.distorded_patch = None
        model = PatchTestCase.Distort(self.patch_desc)
        model.eval()
        var_patch = torch.autograd.Variable(self.patch_desc.patch, requires_grad=True)
        for img, _ in self.patch_desc.train_loader :
            """
            vector_scores = self.patch_desc.model(self.patch_desc.image_transformer.normalize(img))
            model_label = torch.argmax(vector_scores.data).item()
            if model_label is not true_label.item() :
                continue
            if model_label is self.patch_desc.target_class :
                continue
            """
            var_img = torch.autograd.Variable(img, requires_grad=True)
            vector_scores = model(var_img, var_patch)
            loss_target_spec = -torch.nn.functional.log_softmax(vector_scores, dim=1)[0, self.patch_desc.target_class]
            loss_target_spec.backward()
            
            var_img.grad.data.zero_()
            vector_scores = self.patch_desc.model(self.patch_desc.image_transformer.normalize(var_img))
            loss_target = -torch.nn.functional.log_softmax(vector_scores, dim=1)[0, self.patch_desc.target_class]
            loss_target.backward()
            
            empty = torch.zeros(1, 3, self.patch_desc.image_dim, self.patch_desc.image_dim)
            grad = calibration.distorsion.undistort_patch(empty, var_img.grad, self.patch_desc.map_)
            row0, col0 = self.patch_desc.row0, self.patch_desc.col0
            patch_grad = grad[0, :, row0:row0+self.patch_desc.patch_dim, 
                              col0:col0+self.patch_desc.patch_dim]
            
            ax1.imshow(tensor_to_numpy_array(var_patch), interpolation='nearest')
            ax2.imshow(tensor_to_numpy_array(coef * var_patch.grad), interpolation='nearest')
            ax3.imshow(tensor_to_numpy_array(self.patch_desc.adv_img), interpolation='nearest')
            ax4.imshow(tensor_to_numpy_array(coef * patch_grad), interpolation='nearest')
            plt.show()
            break
        
             
                
class VariousTestCase(unittest.TestCase):
    #https://stackoverflow.com/questions/62475627/pytorch-add-input-normalization-to-model-division-layer
    
    class Normalize(torch.nn.Module):
        def __init__(self) :
            super().__init__()
            self.image_transformer = image_transformation.ImageTransformer()
            
        def forward(self, input):
            return self.image_transformer.normalize(input)

    class Distorsion(torch.nn.Module):
        def __init__(self) :
            super().__init__()
            path_calibration = 'C:\\Users\\alexi\\PROJET_3A\\projet_3A\\calibration\\'
            self.cam_mtx, self.dist_coef = calibration.distorsion.load_coef(path_calibration)
            
        def forward(self, input):
            return calibration.distorsion.distort_patch(self.cam_mtx, self.dist_coef, input)[0]
        
    def setUp(self):
        path_dataset = "C:\\Users\\alexi\\PROJET_3A\\Projet Adversarial Patch\\Project Adverserial Patch\\Collision Avoidance\\dataset"
        self.train_loader, self.validation_loader, self.test_loader = load.load_dataset(path_dataset)
        self.model = torch.nn.Sequential(
            #VariousTestCase.Normalize(),
            VariousTestCase.Distorsion(),
            #torch.nn.MaxPool2d(kernel_size=(224,224))
        )
        
    def test_gradient(self):
        self.model.eval()
        for img, _ in self.train_loader :
            #img = torch.squeeze(img)
            var_img = torch.autograd.Variable(img.data, requires_grad=True)
            scores = self.model(var_img)
            print(scores.shape)
            #score = torch.nn.functional.threshold(, -999, 0)[0]
            #print(score)
            #score.backward()
            break
            
if __name__ == '__main__':
    unittest.main()