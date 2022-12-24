
import numpy as np
import torch
import torchvision
import datetime
import uuid
import image_transformation
import calibration.distorsion
import load

class PatchDesc():
    def __init__(self):
        self.date = datetime.datetime.now().strftime("%d/%m/%Y %H:%M:%S")
        
        self.path_model = 'C:\\Users\\alexi\\PROJET_3A\\Projet Adversarial Patch\\Project Adverserial Patch\\Collision Avoidance\\best_model_extended.pth'
        self.path_dataset = 'C:\\Users\\alexi\\PROJET_3A\\Projet Adversarial Patch\\Project Adverserial Patch\\Collision Avoidance\\dataset'
        
        self.model = None
        
        self.train_loader, self.valid_loader, self.test_loader = None, None, None
        
        self.path_image_folder = 'C:\\Users\\alexi\\PROJET_3A\\projet_3A\\img\\'
        
        self.path_calibration = 'C:\\Users\\alexi\\PROJET_3A\\projet_3A\\calibration\\'
        self.cam_mtx, self.dist_coef = None, None

        self.id = str(uuid.uuid4())
        self.image_dim = 224
        self.patch_relative_size = 10/100
        image_size = self.image_dim**2
        patch_size = image_size * self.patch_relative_size
        self.patch_dim = int(patch_size**(0.5))
        
        self.n_epochs = 1
        self.target_class = 1
        self.threshold = 0.9
        self.max_iterations = 100
        
        self.image_transformer = image_transformation.ImageTransformer()
        
        self.train_success_rate = {}
        self.valid_success_rate = {}
        self.test_success_rate = []
        
        self.random_init()
        
    def load_everything(self):
        self.model = load.load_model(self.path_model)
        self.train_loader, self.valid_loader, self.test_loader = load.load_dataset(self.path_dataset)
        self.cam_mtx, self.dist_coef = calibration.distorsion.load_coef(self.path_calibration)
            
    def random_init(self):
        self.patch = torch.rand(1, 3, self.patch_dim, self.patch_dim)
        self.patch = (self.patch + torch.flip(self.patch, [3]))/2

    def random_rotation(self):
        k = np.random.randint(0, 3)
        for i in range(3):
            self.patch[0, i, :, :] = torch.rot90(self.patch[0, i, :, :], k)

    def random_translation(self):
        return np.random.choice(self.image_dim - self.patch_dim, size=2)
    
    def get_mask(self, empty_img_p):
        mask = torch.zeros_like(empty_img_p)
        mask[empty_img_p != 0] = 1 
        return mask

    def get_empty_image_patch(self, row0, col0):
        empty_img_p = torch.zeros(1, 3, self.image_dim, self.image_dim)
        empty_img_p[0, :, row0:row0 + self.patch_dim, col0:col0 + self.patch_dim] = self.patch
        return empty_img_p
    
    def random_transform(self):
        assert self.cam_mtx is not None
        assert self.dist_coef is not None
        row0, col0 = self.random_translation()
        empty_img_p = self.get_empty_image_patch(row0, col0)
        distorded_patch, map_ = calibration.distorsion.distort_patch(self.cam_mtx, self.dist_coef, empty_img_p)
        mask = self.get_mask(distorded_patch)
        return empty_img_p, distorded_patch, map_, mask, row0, col0
    
    def image_attack(self, img, empty_img_p, mask):
        assert self.model is not None
        
        self.model.eval()
        
        c = 0
        while True :
            adv_img = torch.mul((1-mask), img) + torch.mul(mask, empty_img_p)
            var_adv_img = torch.autograd.Variable(adv_img.data, requires_grad=True)
            vector_scores = self.model(var_adv_img)
            vector_proba = torch.nn.functional.softmax(vector_scores, dim=1)
            target_proba = vector_proba[0, self.target_class]
            print('iteration %d target proba mean %f' % (c, target_proba))
            c += 1
            if target_proba >= self.threshold or c >= self.max_iterations :
                break
            loss_target = -torch.nn.functional.log_softmax(vector_scores, dim=1)[0, self.target_class]
            loss_target.backward()
            grad = var_adv_img.grad.clone()
            var_adv_img.grad.data.zero_()
            empty_img_p -= grad
            empty_img_p = torch.clamp(empty_img_p, 0, 1)
        return adv_img, target_proba
    
    def check(self, img, true_label):
        vector_scores = self.model(img)
        model_label = torch.argmax(vector_scores.data).item()
        if model_label is not true_label.item() :
            return None
        if model_label is self.target_class :
            return None

        _, distorded_patch, _, mask, _, _ = self.random_transform()
        
        adv_img = torch.mul((1-mask), img) + torch.mul(mask, distorded_patch)
        adv_img = torch.clamp(adv_img, 0, 1)
        vector_scores = self.model(adv_img)
        adv_label = torch.argmax(vector_scores.data).item()
        
        if adv_label == self.target_class:
            return True
        return False
        
    def validation(self):
        assert self.valid_loader is not None
        total, success = 0, 0
        for img, true_label in self.valid_loader :
            res = self.check(img, true_label)
            if res is None :
                continue
            if res :
                success += 1
                total += 1
            else :
                total += 1
        assert total != 0
        return success/total
            
    def train(self, callback):
        assert self.train_loader is not None
        assert self.model is not None
        
        self.model.eval()
        
        for epoch in range(self.n_epochs) :
            success, total = 0, 0
            self.train_success_rate[epoch] = []
            self.valid_success_rate[epoch] = []

            for img, true_label in self.train_loader:
                vector_scores = self.model(img)
                model_label = torch.argmax(vector_scores.data).item()
                if model_label is not true_label.item() :
                    continue
                if model_label is self.target_class :
                    continue
                
                total += 1
                
                empty_img_p, distorded_patch, map_, mask, row0, col0 = self.random_transform()
                adv_img, target_proba = self.image_attack(img, distorded_patch, mask)
                
                if target_proba >= self.threshold :
                    success += 1
                
                distorded_patch = torch.mul(mask, adv_img)
                empty_img_p = calibration.distorsion.undistort_patch(empty_img_p, distorded_patch, map_)
                
                self.patch = empty_img_p[0, :, row0:row0+self.patch_dim, col0:col0+self.patch_dim]
                
                if (total % 2 == 0):
                    
                    torchvision.utils.save_image(img.data, self.path_image_folder 
                                                + 'epoch%d_batch%d_label%d_original.png' 
                                                % (epoch, total, true_label.item()))
                    
                    torchvision.utils.save_image(adv_img.data, self.path_image_folder 
                                                + 'epoch%d_batch%d_adversarial.png' 
                                                % (epoch, total))
                    
                    torchvision.utils.save_image(self.patch.data, self.path_image_folder
                                                + 'epoch%d_batch%d_patch.png' 
                                                % (epoch, total))
                    
                    torchvision.utils.save_image(distorded_patch.data, self.path_image_folder
                                                + 'epoch%d_batch%d_distorded_patch.png' 
                                                % (epoch, total))
                    
                #valid_rate = self.validation()
                valid_rate = 0
                print('img %d success rate %f val rate %f' % (total - 1, success/total, valid_rate))
                
                self.train_success_rate[epoch].append(success/total)
                self.valid_success_rate[epoch].append(valid_rate)
                callback()
                

    def test(self, callback):
        assert self.valid_loader is not None
        assert self.model is not None
        
        self.model.eval()
        success, total = 0, 0

        for img, true_label in self.test_loader :
            res = self.check(img, true_label)
            if res is None :
                continue
            if res :
                success += 1
                total += 1
            else :
                total += 1
                
            print('img %d success rate %f' % (total - 1, success/total))
            self.test_success_rate.append(success/total)
            callback()
