import unittest
import new_patch
import constants.constants as c
import time
import numpy as np
import torch
import matplotlib.pyplot as plt
# import distortion.distortion as d
import total_variation.new_total_variation as tv
import printability.new_printability as p
import color_jitter.color_jitter as color_jitter
import utils.utils as u
import sklearn.cluster
import PIL
import torchvision


def tensor_to_array(tensor):
    tensor = torch.clamp(tensor, 0, 1)
    return u.tensor_to_array(tensor)

class ImageTransformation(unittest.TestCase):
    def setUp(self):
        self.train_loader, _ = u.load_dataset(c.PATH_DATASET)
        self.normalize = torchvision.transforms.Normalize(mean=c.MEAN, std=c.STD)
        self.color_jitter_module = color_jitter.ColorJitterModule()

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

    def test_jitter(self):
        _, (ax1, ax2) = plt.subplots(1, 2)
        plt.suptitle("TEST JITTER")
        for img, _ in self.train_loader:
            self.color_jitter_module.jitter()
            ax1.imshow(tensor_to_array(img), interpolation='nearest')
            ax1.set_title('original image')

            t0 = time.time()
            jittered_img = self.color_jitter_module(img)
            print(torch.equal(img, jittered_img))
            t1 = time.time()
            ax2.imshow(tensor_to_array(jittered_img), interpolation='nearest')
            ax2.set_title('jitter image\ndeltat=%.2fms' % ((t1 - t0)*1e3))

            plt.pause(1)
        plt.show()
    
    def test_skew(self):
        _, (ax1, ax2) = plt.subplots(1, 2)
        plt.suptitle("TEST SKEW")
        import math
        matrix_skew = np.eye(3)
        matrix_skew[0, 1] = math.tan(math.radians(5))
        print(matrix_skew)   
        mask = np.zeros((200, 200))
        mask[50:100, 50:100] = np.ones((50, 50))
        ax1.imshow(mask)
        skew = matrix_skew@mask
        ax2.imshow(skew)
        plt.show()
        
        
        
class Trainer(unittest.TestCase):
    def setUp(self):
        self.patch_trainer = new_patch.PatchTrainer(distort=True,
                                                    patch_relative_size=0.05)
                                                    
        self.patch_trainer_flee = new_patch.PatchTrainer(distort=True,
                                                         mode=c.Mode.FLEE,
                                                         threshold=0,
                                                         patch_relative_size=0.05)
    def test_initialization(self):
        patch = self.patch_trainer.patch
        plt.imshow(tensor_to_array(patch), interpolation='nearest')
        mean = [int(x*1e3)/1e3 for x in torch.mean(patch, dim=[2, 3]).tolist()[0]]
        std = [int(x*1e3)/1e3 for x in torch.std(patch, dim=[2, 3]).tolist()[0]]
        title = "mean=%s std=%s\n true_mean=%s true_std=%s" % \
                (str(mean), (std), str(c.MEAN), str(c.STD))
        plt.title(title)
        plt.show()
        plt.close()

    def test_mask(self):
        trainer = self.patch_trainer
        _, (ax1, ax2) = plt.subplots(1, 2)
        plt.suptitle("TEST MASK")
        for _ in range(100) :
            row0, col0 = self.patch_trainer.random_position()
            empty_with_patch = torch.zeros(1, 3, c.IMAGE_DIM, c.IMAGE_DIM)
            empty_with_patch[0, :, row0:row0 + trainer.patch_dim, 
                                col0:col0 + trainer.patch_dim] = trainer.patch
            
            distorted, _ = trainer.dist_tool.distort(empty_with_patch)
            mask = self.patch_trainer.get_mask(distorted)

            ax1.imshow(tensor_to_array(distorted), interpolation='nearest')
            ax1.set_title('distorted')

            ax2.imshow(tensor_to_array(mask), interpolation='nearest')
            ax2.set_title('mask')
            plt.pause(1)
        plt.show()

    def test_distortion(self):
        _, (ax1, ax2, ax3, ax4) = plt.subplots(1, 4)
        plt.suptitle("TEST DISTORTION")
        trainer = self.patch_trainer
        patch_dim = trainer.patch_dim
        for x in range(0, c.IMAGE_DIM - patch_dim, 10):
            row0, col0 = x, x
            empty_with_patch = torch.zeros(1, 3, c.IMAGE_DIM, 
                                                 c.IMAGE_DIM)
            empty_with_patch[0, :, row0:row0 + patch_dim,
                                    col0:col0 + patch_dim] = trainer.patch
            
            ax1.imshow(tensor_to_array(empty_with_patch), interpolation='nearest')
            ax1.set_title('empty image patch')

            t0 = time.time()
            distorted, map_ = trainer.dist_tool.distort(empty_with_patch)
            t1 = time.time()
            ax2.imshow(tensor_to_array(distorted), interpolation='nearest')
            ax2.set_title('after distortion\ndeltat=%.2fms' % ((t1 - t0)*1e3))

            t0 = time.time()
            empty_with_patch = trainer.dist_tool.undistort(distorted, map_,
                                                           empty_with_patch)
            t1 = time.time()
            ax3.imshow(tensor_to_array(empty_with_patch), interpolation='nearest')
            ax3.set_title('after undistortion\ndeltat=%.2fms' % ((t1 - t0)*1e3))

            t0 = time.time()
            distorted = trainer.dist_tool.distort_with_map(empty_with_patch, map_)
            t1 = time.time()
            ax4.imshow(tensor_to_array(distorted), 
                       interpolation='nearest')
            ax4.set_title('with map\ndeltat=%.2fms' % ((t1 - t0)*1e3))

            plt.pause(1)

    def test_attack(self):
        trainer = self.patch_trainer
        _, (ax1, ax2, ax3, ax4, ax5) = plt.subplots(1, 5)
        plt.suptitle("TEST ATTACK")
        for image, true_label in trainer.train_loader:
            ax1.imshow(tensor_to_array(image), interpolation='nearest')
            ax1.set_title('image')
            vector_scores = trainer.model(trainer.normalize(image))
            model_label = torch.argmax(vector_scores).item()
            if model_label is not true_label.item() or \
                    model_label is trainer.target_class:
                continue

            row0, col0 = trainer.random_position()
            empty_with_patch = torch.zeros(1, 3, c.IMAGE_DIM, 
                                                 c.IMAGE_DIM)
            empty_with_patch[0, :, row0:row0 + trainer.patch_dim, 
                                   col0:col0 + trainer.patch_dim] = trainer.patch
            mask = trainer.get_mask(empty_with_patch)
            i = 0

            while True:
                t0 = time.time()
                attacked = torch.mul(1 - mask, image) + torch.mul(mask, empty_with_patch)
                attacked.requires_grad = True
                
                normalized = trainer.normalize(attacked)
                
                vector_scores = trainer.model(normalized)
                vector_proba = torch.nn.functional.softmax(vector_scores, dim=1)
                target_proba = vector_proba[0, trainer.target_class]
                
                ax2.imshow(tensor_to_array(normalized), interpolation='nearest')
                ax2.set_title('attacked \nproba : %.2f' % target_proba)
                if i > 0:
                    print('iteration %d target proba %.2f' % (i, target_proba))
                if target_proba >= trainer.threshold or \
                        i >= trainer.max_iterations :
                    break
                i += 1
                loss = -torch.nn.functional.log_softmax(vector_scores, dim=1)
                loss[0, trainer.target_class].backward()
                empty_with_patch -= attacked.grad
                

                t1 = time.time()
                normalized_grad = u.normalize_tensor(attacked.grad)
                ax3.imshow(tensor_to_array(normalized_grad), 
                           interpolation='nearest')
                ax3.set_title('normalized grad\ndeltat=%.2fms' % ((t1 - t0)*1e3))
                
                ax4.imshow(tensor_to_array(empty_with_patch), interpolation='nearest')
                ax4.set_title('empty with patch')

                
                ax5.imshow(tensor_to_array(trainer.patch), interpolation='nearest')
                ax5.set_title('patch')
                plt.pause(0.1)
            trainer.patch = empty_with_patch[0, :, row0:row0 + trainer.patch_dim, 
                                                   col0:col0 + trainer.patch_dim]
        plt.show()
        

    def test_attack2(self):
        trainer = self.patch_trainer
        _, (ax1, ax2, ax3, ax4, ax5, ax6) = plt.subplots(1, 6)
        jitter_module = color_jitter.ColorJitterModule()
        my_transform = torchvision.transforms.Compose([
                            jitter_module,
                            torchvision.transforms.Normalize(mean=c.MEAN, 
                                                            std=c.STD)
                            ])
        plt.suptitle("TEST ATTACK")
        for image, true_label in trainer.train_loader:
            ax1.imshow(tensor_to_array(image), interpolation='nearest')
            ax1.set_title('image')
            vector_scores = trainer.model(trainer.normalize(image))
            model_label = int(torch.argmax(vector_scores))
            if model_label is not int(true_label) or \
                    model_label is trainer.target_class:
                continue

            row0, col0 = trainer.random_position()
            empty_with_patch = torch.zeros(1, 3, c.IMAGE_DIM, 
                                                 c.IMAGE_DIM)
            empty_with_patch[0, :, row0:row0 + trainer.patch_dim, 
                                   col0:col0 + trainer.patch_dim] = trainer.patch
            mask = trainer.get_mask(empty_with_patch)
            i = 0

            while True:
                attacked = torch.mul(1 - mask, image) + torch.mul(mask, empty_with_patch)
                attacked.requires_grad = True
                
                normalized = trainer.normalize(attacked)
                
                vector_scores = trainer.model(normalized)
                vector_proba = torch.nn.functional.softmax(vector_scores, dim=1)
                target_proba = vector_proba[0, trainer.target_class]
                
                ax2.imshow(tensor_to_array(normalized), interpolation='nearest')
                ax2.set_title('attacked \nproba : %.2f' % target_proba)
                if i > 0:
                    print('iteration %d target proba %.2f' % (i, target_proba))
                if target_proba >= trainer.threshold or \
                        i >= trainer.max_iterations :
                    break
                i += 1
                loss = -torch.nn.functional.log_softmax(vector_scores, dim=1)
                loss[0, trainer.target_class].backward()
                sauv_grad1 = attacked.grad.clone()
                sauv_attacked1 = attacked.clone().detach()
                empty_with_patch -= attacked.grad
                
                normalized_grad = u.normalize_tensor(attacked.grad)
                ax3.imshow(tensor_to_array(normalized_grad), 
                           interpolation='nearest')

                ax4.imshow(tensor_to_array(empty_with_patch), interpolation='nearest')
                ax4.set_title('empty with patch')

                attacked.grad.zero_()

                jittered = torchvision.transforms.functional.adjust_brightness(attacked, 1)
                normalized2 = my_transform(jittered)
                
                vector_scores = trainer.model(normalized2)
                vector_proba = torch.nn.functional.softmax(vector_scores, dim=1)
                target_proba = vector_proba[0, trainer.target_class]

                ax5.imshow(tensor_to_array(normalized2), interpolation='nearest')
                ax5.set_title('attacked \nproba : %.2f' % target_proba)

                loss = -torch.nn.functional.log_softmax(vector_scores, dim=1)
                loss[0, trainer.target_class].backward()
                sauv_grad2 = attacked.grad.clone()
                sauv_attacked2 = attacked.clone().detach()

                print(torch.equal(sauv_grad1, sauv_grad2))
                print(torch.equal(sauv_attacked1, sauv_attacked2))
                copie = attacked.clone().detach()
                print("%f" % torch.nn.functional.mse_loss(copie, torchvision.transforms.functional.adjust_brightness(copie, 1.0)))
                print(torch.equal(copie, torchvision.transforms.functional.adjust_brightness(copie, 1.0)))

                normalized_grad = u.normalize_tensor(attacked.grad)
                ax6.imshow(tensor_to_array(normalized_grad), 
                           interpolation='nearest')
                ax6.set_title('normalized grad')
                

                plt.pause(0.1)
            trainer.patch = empty_with_patch[0, :, row0:row0 + trainer.patch_dim, 
                                                   col0:col0 + trainer.patch_dim]
        plt.show()


    def test_attack3(self):
        trainer = self.patch_trainer
        _, (ax1, ax2, ax3, ax4, ax5) = plt.subplots(1, 5)
        plt.suptitle("TEST ATTACK")
        patch = torch.zeros(1, 3, 224, 224)
        patch[0, :, 112-20:112+20, 112-20:112+20] = torch.rand(3, 40, 40)
        import transformation
        transformation_tool = transformation.TransformationTool(40)
        for image, true_label in trainer.train_loader:
            ax1.imshow(tensor_to_array(image), interpolation='nearest')
            ax1.set_title('image')
            vector_scores = trainer.model(trainer.normalize(image))
            model_label = torch.argmax(vector_scores).item()
            if model_label is not true_label.item() or \
                    model_label is trainer.target_class:
                continue

            transformed, map_ = transformation_tool.random_transfom(patch)
            mask = trainer._get_mask(transformed)
            i = 0

            while True:
                t0 = time.time()
                attacked = torch.mul(1 - mask, image) + torch.mul(mask, transformed)
                attacked.requires_grad = True
                
                normalized = trainer.normalize(attacked)
                
                vector_scores = trainer.model(normalized)
                vector_proba = torch.nn.functional.softmax(vector_scores, dim=1)
                target_proba = vector_proba[0, 1]
                
                ax2.imshow(tensor_to_array(normalized), interpolation='nearest')
                ax2.set_title('attacked \nproba : %.2f' % target_proba)
                if i > 0:
                    print('iteration %d target proba %.2f' % (i, target_proba))
                if target_proba >= trainer.threshold or \
                        i >= trainer.max_iterations :
                    break
                i += 1
                loss = -torch.nn.functional.log_softmax(vector_scores, dim=1)
                loss[0, 1].backward()
                with torch.no_grad() :
                    transformed -= attacked.grad
                
                t1 = time.time()
                normalized_grad = u.normalize_tensor(attacked.grad)
                ax3.imshow(tensor_to_array(normalized_grad), 
                           interpolation='nearest')
                ax3.set_title('normalized grad\ndeltat=%.2fms' % ((t1 - t0)*1e3))
                
                ax4.imshow(tensor_to_array(attacked), interpolation='nearest')
                ax4.set_title('empty with patch')

                
                ax5.imshow(tensor_to_array(patch), interpolation='nearest')
                ax5.set_title('patch')
                plt.pause(1)
            patch = transformation_tool.undo_transform(patch, transformed, map_)
        plt.show()

    def test_attack_target(self):
        trainer = self.patch_trainer
        _, (ax1, ax2, ax3, ax4) = plt.subplots(1, 4)
        plt.suptitle("TEST ATTACK TARGET")
        n = 0
        for image, true_label in trainer.train_loader:
            normalized = trainer.normalize(image)
            vector_scores = trainer.model(normalized)
            model_label = torch.argmax(vector_scores).item()
            if model_label is not true_label.item() or \
                    model_label is trainer.target_class:
                continue
            n += 1
            
            row0, col0 = trainer.random_position()
            ret = trainer.attack(image, row0, col0)
            first_target_proba, attacked, empty_with_patch = ret
            ax1.imshow(tensor_to_array(normalized), interpolation='nearest')
            ax1.set_title("image %d" % n)
            ax2.imshow(tensor_to_array(attacked), interpolation='nearest') 
            ax2.set_title("attacked\nfirst target proba=%.2f" % first_target_proba)
            ax3.imshow(tensor_to_array(empty_with_patch), interpolation='nearest')
            ax3.set_title("empty with patch")        
            ax4.imshow(tensor_to_array(trainer.patch), interpolation='nearest')
            ax4.set_title("patch")
            plt.pause(1)
        plt.show()
    
    def test_attack_flee(self):
        trainer = self.patch_trainer_flee
        _, (ax1, ax2, ax3, ax4) = plt.subplots(1, 4)
        plt.suptitle("TEST ATTACK FLEE")
        n = 0
        for image, true_label in trainer.train_loader:
            normalized = trainer.normalize(image)
            vector_scores = trainer.model(normalized)
            model_label = torch.argmax(vector_scores).item()
            if model_label is not true_label.item() or \
                    model_label != trainer.target_class:
                continue
            n += 1
            
            row0, col0 = trainer.random_position()
            ret = trainer.attack(image, row0, col0)
            first_target_proba, attacked, empty_with_patch = ret
            ax1.imshow(tensor_to_array(normalized), interpolation='nearest')
            ax1.set_title("image %d" % n)
            ax2.imshow(tensor_to_array(attacked), interpolation='nearest') 
            ax2.set_title("attacked\nfirst target proba=%.2f" % first_target_proba)
            ax3.imshow(tensor_to_array(empty_with_patch), interpolation='nearest')
            ax3.set_title("empty with patch")        
            ax4.imshow(tensor_to_array(trainer.patch), interpolation='nearest')
            ax4.set_title("patch")
            plt.pause(1)
        plt.show() 
        
    def test_attack_target_and_flee(self):
        trainer = self.patch_trainer
        trainer.mode = c.Mode.TARGET_AND_FLEE
        _, (ax1, ax2, ax3, ax4) = plt.subplots(1, 4)
        plt.suptitle("TEST ATTACK TARGET AND FLEE")
        n = 0
        for image, true_label in trainer.train_loader:
            normalized = trainer.normalize(image)
            vector_scores = trainer.model(normalized)
            model_label = torch.argmax(vector_scores).item()
            if model_label is not true_label.item() or \
                    model_label == trainer.target_class:
                continue
            n += 1
            
            row0, col0 = trainer.random_position()
            ret = trainer.attack(image, row0, col0)
            first_target_proba, attacked, empty_with_patch = ret
            ax1.imshow(tensor_to_array(normalized), interpolation='nearest')
            ax1.set_title("image %d" % n)
            ax2.imshow(tensor_to_array(attacked), interpolation='nearest') 
            ax2.set_title("attacked\nfirst target proba=%.2f" % first_target_proba)
            ax3.imshow(tensor_to_array(empty_with_patch), interpolation='nearest')
            ax3.set_title("empty with patch")        
            ax4.imshow(tensor_to_array(trainer.patch), interpolation='nearest')
            ax4.set_title("patch")
            plt.pause(0.1)
        plt.show()  

    def test_comparaison(self):
        trainer = self.patch_trainer
        trainer_flee = self.patch_trainer_flee

        image_target, image_flee = None, None
        max_iterations = 10
        _, (ax1, ax2) = plt.subplots(2, 4)
        for image, true_label in trainer.train_loader:
            normalized = trainer.normalize(image)
            vector_scores = trainer.model(normalized)
            model_label = torch.argmax(vector_scores).item()
            if model_label is not true_label.item() : continue
            if model_label == trainer.target_class: image_flee = image
            else : image_target = image
            if torch.is_tensor(image_target) and torch.is_tensor(image_flee) : 
                c = 0
                row0, col0 = trainer.random_position()
                empty_with_patch_target = trainer.create_empty_with_patch(row0, col0)
                empty_with_patch_flee = trainer_flee.create_empty_with_patch(row0, col0)
                mask = trainer.get_mask(empty_with_patch_target)
                while True :
                    # FLEE
                    attacked = torch.mul(1 - mask, image_flee) + \
                                    torch.mul(mask, empty_with_patch_flee)
                    attacked.requires_grad = True
                    normalized = trainer.normalize(attacked)
                    vector_scores = trainer.model(normalized)
                    vector_proba = torch.nn.functional.softmax(vector_scores, dim=1)
                    target_proba = vector_proba[0, trainer.target_class].item()
                    if c > 0 : print('iteration : %d target proba flee : %f' % (c, target_proba))

                    if target_proba > 0.10 :
                        loss = -torch.nn.functional.log_softmax(vector_scores, dim=1)
                        loss[0, trainer.target_class].backward()
                        empty_with_patch_flee += attacked.grad
                        trainer_flee.patch = empty_with_patch_flee[:, :, row0:row0 + trainer.patch_dim, 
                                                                        col0:col0 + trainer.patch_dim]
                    ax1[0].imshow(tensor_to_array(image_flee), interpolation='nearest')
                    ax1[0].set_title("image")
                    ax1[1].imshow(tensor_to_array(normalized), interpolation='nearest') 
                    ax1[1].set_title("attacked (flee)\ntarget proba=%.2f" % target_proba)
                    ax1[2].imshow(tensor_to_array(empty_with_patch_flee), interpolation='nearest')
                    ax1[2].set_title("empty with patch")        
                    ax1[3].imshow(tensor_to_array(trainer_flee.patch), interpolation='nearest')
                    ax1[3].set_title("patch")
                    
                    # TARGET
                    attacked = torch.mul(1 - mask, image_target) + \
                               torch.mul(mask, empty_with_patch_target)
                    attacked.requires_grad = True
                    normalized = trainer.normalize(attacked)
                    vector_scores = trainer.model(normalized)
                    vector_proba = torch.nn.functional.softmax(vector_scores, dim=1)
                    target_proba = vector_proba[0, trainer.target_class].item()
                    if c > 0 : print('iteration : %d target proba : %f' % (c, target_proba))
                    if  c >= max_iterations :
                        image_target = None
                        image_flee = None
                        break
                    c += 1
                    if target_proba < 0.90 :
                        loss = -torch.nn.functional.log_softmax(vector_scores, dim=1)
                        loss[0, trainer.target_class].backward()
                        empty_with_patch_target -= attacked.grad
                        trainer.patch = empty_with_patch_target[:, :, row0:row0 + trainer.patch_dim, 
                                                                      col0:col0 + trainer.patch_dim]
                    ax2[0].imshow(tensor_to_array(image_target), interpolation='nearest')
                    ax2[0].set_title("image")
                    ax2[1].imshow(tensor_to_array(normalized), interpolation='nearest') 
                    ax2[1].set_title("attacked\ntarget proba=%.2f" % target_proba)
                    ax2[2].imshow(tensor_to_array(empty_with_patch_target), interpolation='nearest')
                    ax2[2].set_title("empty with patch")        
                    ax2[3].imshow(tensor_to_array(trainer.patch), interpolation='nearest')
                    ax2[3].set_title("patch")
                    plt.suptitle("iteration %d" % c)
                    plt.pause(0.1)
                    
class Tools(unittest.TestCase):
    def setUp(self):
        self.patch_trainer = new_patch.PatchTrainer(distort=True)

        transform = torchvision.transforms.Compose([
            torchvision.transforms.Resize(256),
            torchvision.transforms.CenterCrop(224),
            torchvision.transforms.ToTensor()
            ])
        
        self.batch = torch.empty(2, 3, 224, 224)
        for i, path in enumerate(["\\n01440764\\n01440764_17174.JPEG", 
                                  "\\n03394916\\ILSVRC2012_val_00007536.JPEG"]):
            
            image = PIL.Image.open(c.PATH_DATASET + path)
            image = transform(image)
            self.batch[i] = image
    
    def test_total_variation(self):
        assert len(self.batch) >= 1
        e = 0.0001
        tv_module = tv.TotalVariationModule()
        image = self.batch[0]
        image = image[None, :]
        image.requires_grad = True
        for i in range(100):
            tv_loss = tv_module(image)
            tv_loss.backward()
            with torch.no_grad() :
                self.batch -= e * image.grad
            image.grad.zero_()

            plt.imshow(tensor_to_array(image), interpolation='nearest')
            plt.title('%d\ne=%f\npatch_tv_loss=%f' % (i, e, tv_loss))

            plt.pause(0.5)
        plt.show()
        
    def test_printability(self):
        assert len(self.batch) >= 1
        e = 0.005
        print_module = p.PrintabilityModule(c.PATH_PRINTABLE_COLORS, c.IMAGE_DIM)
        image = self.batch[0]
        image = image[None, :]
        image.requires_grad = True
        for i in range(100):
            print_loss = print_module(image)
            print_loss.backward()
            with torch.no_grad() :
                self.batch -= e * image.grad
            image.grad.zero_()

            plt.imshow(tensor_to_array(image), interpolation='nearest')
            plt.title('%d\ne=%f\nprintability_loss=%f' % (i, e, print_loss))

            plt.pause(0.5)
        plt.show()

    def test_printability2(self):
        e = 0.005
        _, (ax1, ax2) = plt.subplots(1, 2)
        image = torch.rand(1, 3, 3, 3)
        image.requires_grad = True
        print_module = p.PrintabilityModule(c.PATH_PRINTABLE_COLORS, 3)
        colors = print_module.colors[:, :, 0, 0]
        assert len(colors) == 30
        colors = colors.reshape(5, 6, 3)
        for i in range(5):
            for j in range(6):
                ax2.text(j, i, str((i, j)))
        _,_, h, w = image.size()
        texts = []
        for i in range(h):
            line = []
            for j in range(w):
                line.append(ax1.text(j, i, ""))
            texts.append(line)

        for n in range(20):
            print_loss = print_module(image)
            print_loss.backward()
            
            with torch.no_grad() :
                image -= e * image.grad
            image.grad.zero_()
            print_loss = 0
            ax1.imshow(tensor_to_array(image), interpolation='nearest')
            ax1.set_title('%d\ne=%f\nprintability_loss=%f' % (n, e, print_loss))

            ax2.imshow(colors, interpolation='nearest')
            ax2.set_title('color set')
            if n%5 == 0:
                for i in range(h):
                    for j in range(w):
                        c = image[0, :, i, j]
                        diff = colors.reshape(30, 3) - c
                        pow = diff**2
                        norm = torch.sqrt(torch.sum(pow, 1))
                        argmin = torch.argmin(norm).item()
                        row = argmin // 6
                        col = argmin % 6
                        texts[i][j].set_text(str((row, col)))
            plt.pause(0.5)
        plt.show()
        
    def test_kMeans(self):
        trainer = self.patch_trainer
        weights = torch.ones(1, 3, 40, 40)
        kmeans = sklearn.cluster.KMeans(n_clusters=5)
        _, (ax1, ax2, ax3) = plt.subplots(1, 3)

        for image, true_label in self.patch_trainer.train_loader:
            image.requires_grad = True
            vector_scores = self.patch_trainer.model(trainer.normalize(image))
            model_label = torch.argmax(vector_scores).item()
            if model_label != true_label.item() or \
                model_label == trainer.target_class:
                continue
            loss = -torch.nn.functional.log_softmax(vector_scores,
                                                           dim=1)
            loss[0, model_label].backward()
            
            conv = torch.nn.functional.conv2d(torch.abs(image.grad), weights)
            conv_array = torch.squeeze(conv).numpy()
            abs = np.abs(conv_array)
            normalized = abs / np.max(abs)
            binary = np.where(normalized < 0.3, 0, 1)
            x = np.transpose(binary.nonzero())
            kmeans.fit(x)
            
            ax1.imshow(tensor_to_array(image))
            ax2.imshow(tensor_to_array(image.grad))
            ax3.imshow(binary)
            r = ax3.scatter(kmeans.cluster_centers_[:, 1],
                            kmeans.cluster_centers_[:, 0])
            plt.pause(1)
            r.remove()
        plt.show()


if __name__ == '__main__':
    unittest.main()
