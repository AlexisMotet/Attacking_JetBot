import torch
import torchvision
import numpy as np
import calibration.distorsion
import load


class PatchTrainer():
    def __init__(self, path_model, path_dataset, path_calibration, path_image_folder,
                 n_epochs, threshold=0.9, max_iterations=10, target_class=1, 
                 patch_relative_size = 0.05, image_dim=224):
        super().__init__()
        self.model = load.load_model(path_model)
        self.train_loader, self.valid_loader, self.test_loader = load.load_dataset(path_dataset)
        self.cam_mtx, self.dist_coef = calibration.distorsion.load_coef(path_calibration)
        self.path_image_folder = path_image_folder
        
        self.image_dim = image_dim
        image_size = self.image_dim**2
        patch_size = image_size * patch_relative_size
        self.patch_dim = int(patch_size**(0.5))
        self.n_epochs = n_epochs
        self.threshold = threshold
        self.max_iterations = max_iterations
        self.target_class = target_class
        
        self.normalize = torchvision.transforms.Normalize(mean=[0.485, 0.456, 0.406], 
                                                          std=[0.229, 0.224, 0.225])
        #self.model_preprocessing = ModelWithPreprocessing(self.model, self.cam_mtx,self.dist_coef,
        #                                                  self.normalize, patch_relative_size)
        self.patch = self.random_patch_init()
        
    def random_patch_init(self):
        patch = torch.rand(1, 3, self.patch_dim, self.patch_dim)
        patch = (patch + torch.flip(patch, [3]))/2
        #return torch.autograd.Variable(patch, requires_grad=True)
        return patch
        
    def image_attack(self, image, empty_with_patch_distorded, mask):
        c = 0
        while True :
            adversarial_image = torch.mul((1-mask), image) + torch.mul(mask, empty_with_patch_distorded)
            var_image = torch.autograd.Variable(adversarial_image, requires_grad=True)
            vector_scores = self.model(self.normalize(var_image))
            vector_proba = torch.nn.functional.softmax(vector_scores, dim=1)
            target_proba = vector_proba[0, self.target_class]
            c += 1
            print('iteration %d target proba %f' % (c, target_proba))
            if target_proba >= self.threshold or c >= self.max_iterations :
                break
            loss_target = -torch.nn.functional.log_softmax(vector_scores, dim=1)[0, self.target_class]
            loss_target.backward()
            empty_with_patch_distorded -= 10 * var_image.grad
            empty_with_patch_distorded = torch.clamp(empty_with_patch_distorded, 0, 1)
        return target_proba, adversarial_image, empty_with_patch_distorded
            
    def train(self):
        self.model.eval()
        success, total = 0, 0
        for epoch in range(self.n_epochs) :
            for image, true_label in self.train_loader:
                vector_scores = self.model(self.normalize(image))
                model_label = torch.argmax(vector_scores.data).item()
                if model_label is not true_label.item() or model_label is self.target_class  :
                    continue
                total += 1
                
                row0, col0 = np.random.choice(self.image_dim - self.patch_dim, size=2)
                empty_with_patch = torch.zeros(1, 3, self.image_dim, self.image_dim)
                empty_with_patch[0, :, row0:row0 + self.patch_dim, col0:col0 + self.patch_dim] = self.patch
                empty_with_patch_distorded, map = calibration.distorsion.distort_patch(self.cam_mtx, self.dist_coef, 
                                                                      empty_with_patch)
                mask = torch.zeros_like(empty_with_patch_distorded)
                mask[empty_with_patch_distorded != 0] = 1
                target_proba, adversarial_image, empty_with_patch_distorded = self.image_attack(image, empty_with_patch_distorded, mask)
                if target_proba >= self.threshold :
                    success += 1
                
                empty_with_patch = calibration.distorsion.undistort_patch(empty_with_patch, 
                                                                          empty_with_patch_distorded, map)    
                self.patch = empty_with_patch[0, :, row0:row0 + self.patch_dim, col0:col0 + self.patch_dim]
                #self.patch += 1e-9
                
                if (total % 2 == 0):
                    torchvision.utils.save_image(image.data, self.path_image_folder 
                                                + 'epoch%d_batch%d_label%d_original.png' 
                                                % (epoch, total, true_label.item()), normalize=True)
                    
                    torchvision.utils.save_image(empty_with_patch_distorded.data, self.path_image_folder 
                                                + 'epoch%d_batch%d_distorded.png' 
                                                % (epoch, total), normalize=True)
                    
                    torchvision.utils.save_image(adversarial_image.data, self.path_image_folder 
                                                + 'epoch%d_batch%d_adversarial.png' 
                                                % (epoch, total), normalize=True)
                    
                    torchvision.utils.save_image(self.patch.data, self.path_image_folder
                                                + 'epoch%d_batch%d_patch.png' 
                                                % (epoch, total), normalize=True)
                    
                    torchvision.utils.save_image(mask.data, self.path_image_folder
                                                + 'epoch%d_batch%d_mask.png' 
                                                % (epoch, total), normalize=True)
                    
                    
                print(success/total)

"""
class ModelWithPreprocessing(torch.nn.Module):
        def __init__(self, model, cam_mtx, dist_coef, normalize, patch_relative_size=0.05, image_dim=224):
            super().__init__()
            self.model = model
            self.cam_mtx = cam_mtx
            self.dist_coef = dist_coef
            self.normalize = normalize
            image_size = image_dim**2
            patch_size = image_size * patch_relative_size
            self.patch_dim = int(patch_size**(0.5))
            self.image_dim = image_dim
            
        
        def forward(self, image, patch):
            #row0, col0 = np.random.choice(self.image_dim - self.patch_dim, size=2)
            row0, col0 = 0, 0
            
            empty_with_patch = torch.zeros(1, 3, self.image_dim, self.image_dim)
            empty_with_patch[0, :, row0:row0 + self.patch_dim, col0:col0 + self.patch_dim] = patch
            print("coucou")
            distorded, _ = calibration.distorsion.distort_patch(self.cam_mtx, self.dist_coef, 
                                                                empty_with_patch)
            
            # mask = torch.zeros_like(distorded)
            # mask[distorded != 0] = 1
            mask = torch.zeros_like(empty_with_patch)
            self.mask = mask
            #self.adversarial_image = torch.mul((1-mask), image) + torch.mul(mask, distorded)
            #image = self.normalize(image)
            self.adversarial_image = torch.mul((1-mask), image) + torch.mul(mask, empty_with_patch)
            return self.model(self.adversarial_image)        
"""

if __name__=="__main__":
    path_model = 'C:\\Users\\alexi\\PROJET_3A\\Projet Adversarial Patch\\Project Adverserial Patch\\Collision Avoidance\\best_model_extended.pth'
    # path_dataset = 'C:\\Users\\alexi\\PROJET_3A\\Projet Adversarial Patch\\Project Adverserial Patch\\Collision Avoidance\\dataset'
    path_dataset = 'C:\\Users\\alexi\\PROJET_3A\\imagenette2-160\\train'
    path_image_folder = 'C:\\Users\\alexi\\PROJET_3A\\projet_3A\\img\\'
    path_calibration = 'C:\\Users\\alexi\\PROJET_3A\\projet_3A\\calibration\\'
    patch_trainer = PatchTrainer(path_model, path_dataset, path_calibration, path_image_folder, 1,
                                 threshold=0.9, max_iterations=10, target_class=1)
    patch_trainer.train()