
import numpy as np
import torch
import torchvision
import datetime
import uuid
import image_transformation
import calibration.distorsion
import math

class PatchDesc():
    def __init__(self, image_dim, patch_relative_size, cam_mtx, dist_coef):
        assert patch_relative_size >  0 and patch_relative_size < 1
        self.target_class = 1   #0: 'blocked', 1 : 'free'
        self.image_dim = image_dim
        self.patch_dim = None
        self.patch = None
        self.threshold = 0.9
        self.max_iterations = 10
        self.image_transformer = image_transformation.ImageTransformer()
        self.cam_mtx = cam_mtx
        self.dist_coef = dist_coef
        
        self.id = uuid.uuid4()
        self.random_init(patch_relative_size)

    def random_init(self, patch_relative_size):
        image_size = self.image_dim**2
        patch_size = image_size * patch_relative_size
        self.patch_dim = int(patch_size**(0.5))
        self.patch = torch.rand(1, 3, self.patch_dim, self.patch_dim)
        #impose symetry
        self.patch = (self.patch + torch.flip(self.patch, [3]))/2

    def random_rotation(self):
        assert self.patch is not None
        k = np.random.randint(0, 3)
        for i in range(3):
            self.patch[0, i, :, :] = torch.rot90(self.patch[0, i, :, :], k)

    def random_translation(self):
        assert self.patch_dim is not None
        return np.random.choice(self.image_dim - self.patch_dim, size=2)
    
    def get_mask(self, empty_img_p):
        mask = torch.zeros_like(empty_img_p)
        mask[empty_img_p != 0] = 1 
        return mask

    def get_empty_image_patch(self, row0, col0):
        assert self.patch is not None
        empty_img_p = torch.zeros(1, 3, self.image_dim, self.image_dim)
        empty_img_p[0, :, row0:row0 + self.patch_dim, col0:col0 + self.patch_dim] = self.patch
        return empty_img_p
    
    def random_transform(self):
        #self.random_rotation()
        row0, col0 = self.random_translation()
        empty_img_p = self.get_empty_image_patch(row0, col0)
        distorded_patch, map_ = calibration.distorsion.distort_patch(self.cam_mtx, self.dist_coef, empty_img_p)
        mask = self.get_mask(distorded_patch)
        return empty_img_p, distorded_patch, map_, mask, row0, col0

    def image_attack(self, model, img, empty_img_p, mask):
        model.eval() # https://stackoverflow.com/questions/60018578/what-does-model-eval-do-in-pytorch
        c = 0
        while True :
            adv_img = torch.mul((1-mask), img) + torch.mul(mask, empty_img_p)
            var_adv_img = torch.autograd.Variable(adv_img.data, requires_grad=True)
            vector_scores = model(self.image_transformer.normalize(var_adv_img))
            vector_proba = torch.nn.functional.softmax(vector_scores, dim=1)
            target_proba = vector_proba[0, self.target_class]
            print('iteration %d target proba mean %f' % (c, target_proba))
            c += 1
            if target_proba >= self.threshold or c >= self.max_iterations :
                break
            loss_target = -torch.nn.functional.log_softmax(vector_scores, dim=1)[0, self.target_class]
            loss_target.backward()
            e = 1
            empty_img_p -= e * var_adv_img.grad
            empty_img_p = torch.clamp(empty_img_p, 0, 1)
        return adv_img, target_proba
    
    def check(self, model, img, true_label):
        vector_scores = model(self.image_transformer.normalize(img))
        model_label = torch.argmax(vector_scores.data).item()
        if model_label is not true_label.item() :
            return None
        if model_label is self.target_class :
            return None
        
        _, distorded_patch, _, mask, _, _ = self.random_transform()
        
        adv_img = torch.mul((1-mask), img) + torch.mul(mask, distorded_patch)
        adv_img = torch.clamp(adv_img, 0, 1)
        vector_scores = model(self.image_transformer.normalize(adv_img))
        adv_label = torch.argmax(vector_scores.data).item()
        
        vector_proba = torch.nn.functional.softmax(vector_scores, dim=1)
        print('target proba %f' % vector_proba.data[0][self.target_class])
        if adv_label == self.target_class:
            return True
        return False
        

    def validation(self, model, valid_loader):
        total, success = 0, 0
        for _, (img, true_label) in enumerate(valid_loader):
            res = self.check(model, img, true_label)
            if (res == None):
                continue
            elif (res == False):
                total += 1
            elif (res == True) :
                success += 1
                total += 1
        assert total != 0
        return success/total
            
    def train(self, n_epochs, model, train_loader, valid_loader, path_img_folder, path_training_results):
        assert self.patch is not None
        assert self.patch_dim is not None
        
        with open(path_training_results, 'a') as f :
            f.write('id %s\n' % self.id)
            f.write(str(datetime.datetime.now().strftime('%d/%m/%Y %H:%M:%S')) + '\n')
        
            model.eval()
            
            for epoch in range(n_epochs) :
                success, total = 0, 0
                f.write('epoch %d\n' % epoch)
                for _, (img, true_label) in enumerate(train_loader):
                    scores = model(self.image_transformer.normalize(img))
                    model_label = torch.argmax(scores.data, dim=1).item()
                    if model_label is not true_label.item() :
                        continue
                    if model_label is self.target_class :
                        continue
                    total += 1
                    
                    empty_img_p, distorded_patch, map_, mask, row0, col0 = self.random_transform()
                    adv_img, target_proba = self.image_attack(model, img, distorded_patch, mask)
                    
                    if target_proba >= self.threshold :
                        success += 1
                    
                    distorded_patch = torch.mul(mask, adv_img)
                    empty_img_p = calibration.distorsion.undistort_patch(empty_img_p, distorded_patch, map_)
                    
                    self.patch = empty_img_p[0, :, row0:row0+self.patch_dim, col0:col0+self.patch_dim]
                    
                    if (total % 5 == 0):
                        
                        torchvision.utils.save_image(img.data, path_img_folder 
                                                    + 'epoch%d_batch%d_label%d_original.png' 
                                                    % (epoch, total, true_label.item()), normalize=True)
                        
                        torchvision.utils.save_image(adv_img.data, path_img_folder 
                                                    + 'epoch%d_batch%d_adversarial.png' 
                                                    % (epoch, total), normalize=True)
                        
                        torchvision.utils.save_image(self.patch.data, path_img_folder
                                                    + 'epoch%d_batch%d_patch.png' 
                                                    % (epoch, total), normalize=True)
                        
                        torchvision.utils.save_image(distorded_patch.data, path_img_folder
                                                    + 'epoch%d_batch%d_distorded_patch.png' 
                                                    % (epoch, total), normalize=True)
                    
                        
                    #validation_rate = self.validation(model, valid_loader)
                    validation_rate = 0
                    print('img %d success rate %f val rate %f' % (total - 1, success/total, validation_rate))
                    f.write('img %d success rate %f val rate %f\n' % (total - 1, success/total, validation_rate))
                

    def test(self, model, test_loader, path_test_results):
        model.eval()
        success, total = 0, 0
        
        with open(path_test_results, 'a') as f :
            f.write('id %s\n' % self.id)
            f.write(str(datetime.datetime.now().strftime('%d//%m//%Y %H:%M:%S')) + '\n')
            
            for _, (img, true_label) in enumerate(test_loader):
                res = self.check(model, img, true_label)
                if (res == None):
                    continue
                elif (res == False):
                    total += 1
                elif (res == True) :
                    success += 1
                    total += 1
                    
                print('img %d success rate %f' % (total - 1, success/total))
                f.write('img %d success rate %f\n' % (total - 1, success/total))
