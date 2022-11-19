
import numpy as np
import matplotlib.pyplot as plt
import torch
import torchvision
import datetime
import cv2
import uuid
import image_transformation

class PatchDesc():
    def __init__(self, image_dim, patch_relative_size):
        assert patch_relative_size >=  0 and patch_relative_size < 1
        self.target_class = 1   #0: 'blocked', 1 : 'free'
        self.image_dim = image_dim
        self.patch_dim = None
        self.patch = None
        self.threshold = 0.9
        self.max_iterations = 10
        self.random_init(patch_relative_size)
        self.id = uuid.uuid4()
        self.transform = torchvision.transforms.Compose([
            torchvision.transforms.ColorJitter(0.1, 0.1, 0.1, 0.1),
            torchvision.transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
            ])
        

    def random_init(self, patch_relative_size):
        image_size = self.image_dim**2
        patch_size = image_size * patch_relative_size
        self.patch_dim = int(patch_size**(0.5))
        self.patch = torch.rand(1, 3, self.patch_dim, self.patch_dim)

    def random_rotation(self):
        assert self.patch is not None
        k = np.random.randint(0, 3)
        for i in range(3):
            self.patch[0, i, :, :] = torch.rot90(self.patch[0, i, :, :], k)

    def random_translation_in_empty_image(self):
        assert self.patch is not None
        assert self.patch_dim is not None
        empty_image_patch = torch.zeros((1, 3, self.image_dim, self.image_dim))
        new_patch_pos = np.random.choice(self.image_dim - self.patch_dim, size=2)
        x, y = new_patch_pos
        empty_image_patch[0, :, x:x + self.patch_dim, y:y + self.patch_dim] = self.patch
        return {'x' : x, 'y' : y}, empty_image_patch
    
    def get_mask(self, empty_image_patch):
        mask = torch.zeros_like(empty_image_patch)
        mask[empty_image_patch != 0] = 1
        return mask
    
    def random_transform(self):
        assert self.patch is not None
        #self.random_rotation()
        p_pos, empty_image_patch = self.random_translation_in_empty_image()
        mask = self.get_mask(empty_image_patch)
        return {'empty_image_patch': empty_image_patch, 'mask' : mask, 'patch_pos': p_pos}

    def batch_attack(self, model, img, empty_img_p, mask):
        model.eval() # https://stackoverflow.com/questions/60018578/what-does-model-eval-do-in-pytorch
        c = 0

        while True :
            adv_img = torch.mul((1-mask), img) + torch.mul(mask, empty_img_p)
            adv_img = torch.clamp(adv_img, 0, 1)
            var_adv_img = torch.autograd.Variable(adv_img.data, requires_grad=True)
            transformed_adv_img = self.transform(var_adv_img)
            vector_scores = model(transformed_adv_img)
            vector_proba = torch.nn.functional.softmax(vector_scores, dim=1)
            target_proba = vector_proba.data[0][self.target_class]
            '''
            print('iteration %d target probability %f threshold %f' % (c, target_proba, self.threshold))
            '''  
            c+=1 
            
            if target_proba >= self.threshold or c >= self.max_iterations :
                break
            
            loss = -torch.nn.functional.log_softmax(vector_scores, dim=1)[0][self.target_class]
            loss.backward()
            
            transformed_adv_img -= var_adv_img.grad
            empty_img_p = torch.mul(mask, transformed_adv_img)
            """
            empty_img_p -= var_adv_img.grad
            """
        return adv_img, vector_proba

    def train(self, model, train_loader, path_img_folder, path_training_results):
        assert self.patch is not None
        assert self.patch_dim is not None
        
        with open(path_training_results, 'a') as f :
            f.write('id %s\n' % self.id)
            f.write(str(datetime.datetime.now().strftime('%d/%m/%Y %H:%M:%S')) + '\n')
            
        
            model.eval()
            success, total = 0, 0
            
            for _, (img, true_label) in enumerate(train_loader):
                transformed_img = self.transform(img)
                vector_scores = model(transformed_img)
                model_label = torch.argmax(vector_scores.data).item()
                if model_label is not true_label.item() :
                    continue
                if model_label is self.target_class :
                    continue
                
                total +=1
                
                transform = self.random_transform()
                empty_img_p = transform['empty_image_patch']
                mask = transform['mask']
                            
                adv_img, vector_proba = self.batch_attack(model, img, empty_img_p, mask)
                
                adv_label = torch.argmax(vector_proba.data).item()
                if adv_label is self.target_class :
                    success += 1
                            
                x, y = transform['patch_pos']['x'], transform['patch_pos']['y']
                self.patch = adv_img[:, :, x:x + self.patch_dim, y:y + self.patch_dim]

                torchvision.utils.save_image(transformed_img.data, path_img_folder 
                                            + 'batch%d_label%d_original.png' 
                                            % (total, true_label.item()), normalize=True)
                
                torchvision.utils.save_image(self.transform(adv_img).data, path_img_folder 
                                            + 'batch%d_label%d_adversarial.png' 
                                            % (total, adv_label), normalize=True)
                
                torchvision.utils.save_image(self.transform(empty_img_p).data, path_img_folder
                                            + 'batch%d_patch.png' 
                                            % (total), normalize=True)
                '''
                def tensor_to_numpy_array(tensor):
                    tensor = torch.squeeze(tensor)
                    array = tensor.cpu().numpy()
                    return np.transpose(array, (1, 2, 0))

                _, (ax1, ax2) = plt.subplots(2, 2)
                ax1[0].imshow(tensor_to_numpy_array(transform['empty_image_patch']), interpolation='nearest')
                ax1[0].set_title('avant entrainement')
                ax1[1].imshow(tensor_to_numpy_array(empty_img_p), interpolation='nearest')
                ax1[1].set_title('apres entrainement')
                
                ax2[0].imshow(tensor_to_numpy_array(img), interpolation='nearest')
                ax2[0].set_title('img')
                ax2[1].imshow(tensor_to_numpy_array(adv_img), interpolation='nearest')
                ax2[1].set_title('adv img')
                plt.show()
                '''
                
                print('batch %d success rate %f' % (total - 1, success/total))
                f.write('batch %d success rate %f\n' % (total - 1, success/total))

    def test(self, model, test_loader, path_test_results):
        model.eval()
        success, total = 0, 0
        
        with open(path_test_results, 'a') as f :
            f.write('id %s\n' % self.id)
            f.write(str(datetime.datetime.now().strftime('%d//%m//%Y %H:%M:%S')) + '\n')
            
            for _, (img, true_label) in enumerate(test_loader):
                vector_scores = model(self.transform(img))
                model_label = torch.argmax(vector_scores.data).item()
                if model_label is not true_label.item() :
                    continue
                if model_label is self.target_class :
                    continue
                
                total += 1
                
                transform = self.random_transform()
                empty_img_p = transform['empty_image_patch']
                mask = transform['mask']
                
                adv_img = torch.mul((1-mask), img) + torch.mul(mask, empty_img_p)
                adv_img = torch.clamp(adv_img, 0, 1)
                adv_img = self.transform(adv_img)
                vector_scores = model(self.transform(adv_img))
                adv_label = torch.argmax(vector_scores.data).item()
                
                if adv_label == self.target_class:
                    success += 1
                    
                print('batch %d success rate %f' % (total - 1, success/total))
                f.write('batch %d success rate %f\n' % (total - 1, success/total))
