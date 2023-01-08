import unittest
import new_patch
import numpy as np
import torch
import matplotlib.pyplot as plt
import distortion.distortion as d
import total_variation.new_total_variation as tv
import printability.new_printability as p
import utils.utils as u
import sklearn.cluster
import PIL
import torchvision

path_model = 'U:\\PROJET_3A\\projet_BONTEMPS_SCHAMPHELEIRE\\Project Adverserial Patch\\Collision Avoidance\\best_model_extended.pth'
path_dataset = 'U:\\PROJET_3A\\projet_BONTEMPS_SCHAMPHELEIRE\\Project Adverserial Patch\\Collision Avoidance\\dataset'
path_model = 'U:\\PROJET_3A\\projet_NOUINOU_MOTET\\imagenette2-160_model.pth'
path_dataset = 'U:\\PROJET_3A\\imagenette2-160\\train'
path_calibration = 'U:\\PROJET_3A\\projet_NOUINOU_MOTET\\calibration\\'
path_distortion = 'U:\\PROJET_3A\\projet_NOUINOU_MOTET\\distortion\\distortion.so'
path_printable_colors = 'U:\\PROJET_3A\\projet_NOUINOU_MOTET\\printability\\printable_colors.dat'
"""
path_model = 'C:\\Users\\alexi\\PROJET_3A\\Projet Adversarial Patch\\Project Adverserial Patch\\Collision Avoidance\\best_model_extended.pth'
path_dataset = 'C:\\Users\\alexi\\PROJET_3A\\Projet Adversarial Patch\\Project Adverserial Patch\\Collision Avoidance\\dataset'
path_calibration = 'C:\\Users\\alexi\\PROJET_3A\\projet_3A\\calibration\\'
path_distortion = 'C:\\Users\\alexi\\PROJET_3A\\projet_3A\\distortion\\distortion.so'
path_printable_colors = 'C:\\Users\\alexi\\PROJET_3A\\projet_3A\\printability\\printable_colors.dat'
"""


class ImageTransformationTestCase(unittest.TestCase):
    def setUp(self):
        self.patch_trainer = new_patch.PatchTrainer(path_model,
                                                    path_dataset,
                                                    path_calibration,
                                                    path_distortion,
                                                    path_printable_colors,
                                                    mode=1,
                                                    distort=True)

    def test_normalization(self):
        _, (ax1, ax2) = plt.subplots(1, 2)
        for img, _ in self.patch_trainer.train_loader:
            ax1.imshow(u.tensor_to_array(img), interpolation='nearest')
            ax1.set_title('original image')

            normalized_img = self.patch_trainer.normalize(img)
            ax2.imshow(u.tensor_to_array(normalized_img), interpolation='nearest')
            ax2.set_title('normalized image')

            plt.show()
            plt.close()
            break

    def test_distortion(self):
        _, (ax1, ax2, ax3, ax4) = plt.subplots(1, 4)
        for row0 in range(0, new_patch.IMAGE_DIM - self.patch_trainer.patch_dim, 10):
            for col0 in range(0, new_patch.IMAGE_DIM - self.patch_trainer.patch_dim, 10):
                empty_with_patch = torch.zeros(1, 3, new_patch.IMAGE_DIM, new_patch.IMAGE_DIM)
                empty_with_patch[0, :, row0:row0 + self.patch_trainer.patch_dim,
                col0:col0 + self.patch_trainer.patch_dim] = self.patch_trainer.patch

                ax1.imshow(u.tensor_to_array(empty_with_patch), interpolation='nearest')
                ax1.set_title('empty image patch')

                empty_with_patch_distorted, map_ = self.patch_trainer.distortion_tool.distort(empty_with_patch)
                ax2.imshow(u.tensor_to_array(empty_with_patch_distorted), interpolation='nearest')
                ax2.set_title('after distortion')

                empty_with_patch = self.patch_trainer.distortion_tool.undistort(empty_with_patch_distorted, 
                                                                                map_,
                                                                                empty_with_patch)
                ax3.imshow(u.tensor_to_array(empty_with_patch), interpolation='nearest')
                ax3.set_title('after undistortion')

                empty_with_patch_distorted = self.patch_trainer.distortion_tool.distort_with_map(empty_with_patch,
                                                                                                 map_)
                ax4.imshow(u.tensor_to_array(empty_with_patch_distorted), interpolation='nearest')
                ax4.set_title('with map')

                plt.pause(0.5)
        plt.show()
        plt.close()


class PatchTestCase(unittest.TestCase):
    def setUp(self):
        self.patch_trainer = new_patch.PatchTrainer(path_model,
                                                    path_dataset,
                                                    path_calibration,
                                                    path_distortion,
                                                    path_printable_colors,
                                                    distort=True)
        self.model = self.patch_trainer.model

    def test_initialized_patch(self):
        plt.imshow(u.tensor_to_array(self.patch_trainer.patch), interpolation='nearest')
        plt.show()
        plt.close()

    def test_translation(self):
        empty_with_patch, row0, col0 = self.patch_trainer.random_transform()
        empty_with_patch[0, :, row0:row0 + 5, col0:col0 + 5] = torch.ones((1, 3, 5, 5))

        plt.imshow(u.tensor_to_array(empty_with_patch))
        plt.title('after translation')

        plt.show()
        plt.close()

    def test_mask(self):
        _, (ax1, ax2) = plt.subplots(1, 2)
        empty_with_patch, _, _= self.patch_trainer.random_transform()
        empty_with_patch_distorted, _ = d.distort_patch(self.patch_trainer.c_distort, self.patch_trainer.cam_mtx,
                                                          self.patch_trainer.dist_coefs, empty_with_patch)
        mask = self.patch_trainer.get_mask(empty_with_patch_distorted)

        ax1.imshow(u.tensor_to_array(empty_with_patch_distorted), interpolation='nearest')
        ax1.set_title('empty image patch')

        ax2.imshow(u.tensor_to_array(mask), interpolation='nearest')
        ax2.set_title('mask')

        plt.show()
        plt.close()

    def test_image_attack(self):
        _, (ax1, ax2, ax3, ax4, ax5) = plt.subplots(1, 5)
        for image, true_label in self.patch_trainer.train_loader:
            vector_scores = self.patch_trainer.model(image)
            model_label = torch.argmax(vector_scores.data).item()
            if model_label is not true_label.item() or model_label is self.patch_trainer.target_class:
                continue

            row0, col0 = self.patch_trainer.random_transform()

            empty_with_patch = torch.zeros(1, 3, new_patch.IMAGE_DIM, new_patch.IMAGE_DIM)
            empty_with_patch[0, :, row0:row0 + self.patch_trainer.patch_dim, 
                col0:col0 + self.patch_trainer.patch_dim] = self.patch_trainer.patch

            empty_with_patch.requires_grad = True
            mask = self.patch_trainer.get_mask(empty_with_patch)
            c = 0
            while True:
                adversarial_image = torch.mul((1-mask), image) + torch.mul(mask, empty_with_patch)
                vector_scores = self.patch_trainer.model(adversarial_image)
                vector_proba = torch.nn.functional.softmax(vector_scores, dim=1)
                target_proba = vector_proba[0, self.patch_trainer.target_class]
                c += 1
                print('iteration %d target proba %.2f' % (c, target_proba))
                if target_proba >= self.patch_trainer.threshold or \
                        c >= self.patch_trainer.max_iterations :
                    break
                loss_target = -torch.nn.functional.log_softmax(vector_scores, dim=1)[0, 
                                                        self.patch_trainer.target_class]
                loss_target.backward()
                with torch.no_grad() :
                    empty_with_patch -= u.normalize_tensor(empty_with_patch.grad)
                    ax4.imshow(u.tensor_to_array(empty_with_patch.grad), interpolation='nearest')
                    ax4.set_title('grad')
                empty_with_patch.requires_grad = False
                self.patch_trainer.model.zero_grad()
            
                self.patch_trainer.patch = empty_with_patch[:, :, 
                    row0:row0 + self.patch_trainer.patch_dim, col0:col0 + self.patch_trainer.patch_dim]
                ax1.imshow(u.tensor_to_array(image), interpolation='nearest')
                ax1.set_title('image')

                ax2.imshow(u.tensor_to_array(adversarial_image), interpolation='nearest')
                ax2.set_title('aversarial_image \nproba : %.2f' % target_proba)

                ax3.imshow(u.tensor_to_array(self.patch_trainer.patch), interpolation='nearest')
                ax3.set_title('patch')

                ax5.imshow(u.tensor_to_array(empty_with_patch), interpolation='nearest')
                ax5.set_title('grad')
                plt.pause(0.1)
        plt.show()
        plt.close()

    def test_grad(self):
        _, (ax1, ax2, ax3, ax4) = plt.subplots(1, 4)
        image = PIL.Image.open("U:\\PROJET_3A\\imagenette2-160\\train\\n01440764\\n01440764_17174.JPEG")
        transform = torchvision.transforms.Compose([torchvision.transforms.Resize(256),
                                                    torchvision.transforms.CenterCrop(224),
                                                    torchvision.transforms.ToTensor(),])
        image = transform(image)
        image = image[None, :]

        mean = [0.485, 0.456, 0.406]
        std = [0.229, 0.224, 0.225]

        normalize = torchvision.transforms.Normalize(mean=mean, std=std)
        grad_of_normalization = torchvision.transforms.Normalize(mean=0, std=std)
        target_class = 1

        test1 = image.clone().detach()
        test1.requires_grad = True
        vector_scores = self.model(normalize(test1))  
        loss_target = -torch.nn.functional.log_softmax(vector_scores, dim=1)[0, target_class]
        loss_target.backward()
        grad_test1 = test1.grad.clone().detach()

        self.model.zero_grad()

        test2 = torch.autograd.Variable(normalize(image.clone().detach()), requires_grad=True)
        vector_scores = self.model(test2)  
        loss_target = -torch.nn.functional.log_softmax(vector_scores, dim=1)[0, target_class]
        loss_target.backward()
        grad_test2 = test2.grad.clone().detach()
        grad_test2 = grad_of_normalization(grad_test2)

        self.model.zero_grad()

        test3 = normalize(image.clone().detach())
        test3.requires_grad = True
        vector_scores = self.model(test3)  
        loss_target = -torch.nn.functional.log_softmax(vector_scores, dim=1)[0, target_class]
        loss_target.backward()
        grad_test3 = test3.grad.clone().detach()
        grad_test3 = grad_of_normalization(grad_test3)

        print("mse grad 1 grad 2 %f" % torch.nn.functional.mse_loss(grad_test1, grad_test2))
        print("mse grad 1 grad 3 %f" % torch.nn.functional.mse_loss(grad_test1, grad_test3))
        print("mse grad 2 grad 3 %f" % torch.nn.functional.mse_loss(grad_test2, grad_test3))


        ax2.imshow(u.tensor_to_array(grad_test1), interpolation='nearest')
        ax2.set_title('grad_test1')

        ax3.imshow(u.tensor_to_array(grad_test2), interpolation='nearest')
        ax3.set_title('grad_test2')

        ax4.imshow(u.tensor_to_array(grad_test3), interpolation='nearest')
        ax4.set_title('grad_test3')

        ax1.imshow(u.tensor_to_array(image), interpolation='nearest')
        ax1.set_title('image')
        plt.show()
        plt.close()

    def test_gradcheck(self):
        _, (ax1, ax2, ax3, ax4, ax5) = plt.subplots(1, 5)
        image = PIL.Image.open("U:\\PROJET_3A\\imagenette2-160\\train\\n01440764\\n01440764_17174.JPEG")
        transform = torchvision.transforms.Compose([torchvision.transforms.Resize(256),
                                                    torchvision.transforms.CenterCrop(224),
                                                    torchvision.transforms.ToTensor(),])
        image = transform(image)
        image = image[None, :]

        mean = [0.485, 0.456, 0.406]
        std = [0.229, 0.224, 0.225]

        normalize = torchvision.transforms.Normalize(mean=mean, std=std)
        grad_of_normalization = torchvision.transforms.Normalize(mean=0, std=std)
        test = torch.autograd.gradcheck(grad_of_normalization, image)        



class VariousTestCase(unittest.TestCase):
    def setUp(self):
        self.patch_trainer = new_patch.PatchTrainer(path_model,
                                                    path_dataset,
                                                    path_calibration,
                                                    path_distortion,
                                                    path_printable_colors,
                                                    mode=1,
                                                    distort=True)
        transform = torchvision.transforms.Compose([torchvision.transforms.Resize(256),
                                                    torchvision.transforms.CenterCrop(224),
                                                    torchvision.transforms.ToTensor()])
        self.batch = torch.empty(2, 3, 224, 224)
        image = PIL.Image.open("U:\\PROJET_3A\\imagenette2-160\\train\\n01440764\\n01440764_17174.JPEG")
        image = transform(image)
        self.batch[0] = image
        image = PIL.Image.open("U:\\PROJET_3A\\imagenette2-160\\train\\n03394916\\ILSVRC2012_val_00007536.JPEG")
        image = transform(image)
        self.batch[1] = image
        
    
    def test_total_variation(self):
        _, (ax1, ax2, ax3, ax4) = plt.subplots(1, 4)
        for _ in range(100):
            self.batch.requires_grad = True
            tv_loss = self.patch_trainer.tv_module(self.batch)
            tv_loss.backward()
            self.batch.requires_grad = False
            self.batch -= 0.005 * self.batch.grad
            ax2.imshow(u.tensor_to_array(self.batch.grad[0]), interpolation='nearest')
            ax2.set_title('total variation grad')

            ax4.imshow(u.tensor_to_array(self.batch.grad[1]), interpolation='nearest')
            ax4.set_title('total variation grad')
            self.batch.grad.data.zero_()

            ax1.imshow(u.tensor_to_array(self.batch[0]), interpolation='nearest')
            ax1.set_title('patch tv loss : %f' % tv_loss)

            ax3.imshow(u.tensor_to_array(self.batch[1]), interpolation='nearest')
            ax3.set_title('patch tv loss : %f' % tv_loss)
            plt.pause(0.5)

        plt.show()
        plt.close()

    def test_printability(self):
        _, (ax1, ax2, ax3, ax4) = plt.subplots(1, 4)
        p_mod = p.PrintabilityModule(path_printable_colors, 224)

        for _ in range(100):
            self.batch.requires_grad = True
            print_loss = p_mod(self.batch)
            print_loss.backward()
            self.batch.requires_grad = False
            self.batch -= 0.005 * self.batch.grad
            ax2.imshow(u.tensor_to_array(self.batch.grad[0]), interpolation='nearest')
            ax2.set_title('grad')
            ax4.imshow(u.tensor_to_array(self.batch.grad[1]), interpolation='nearest')
            ax4.set_title('grad')
            self.batch.grad.data.zero_()

            ax1.imshow(u.tensor_to_array(self.batch[0]), interpolation='nearest')
            ax1.set_title('patch printability loss : %f' % print_loss)

            ax3.imshow(u.tensor_to_array(self.batch[1]), interpolation='nearest')
            ax3.set_title('patch printability loss : %f' % print_loss)
            plt.pause(0.5)
        plt.show()
        plt.close()

    def test_printability2(self):
        _, (ax1, ax2) = plt.subplots(1, 2)
        image = torch.rand(1, 3, 3, 3)

        n = 100

        p_mod = p.PrintabilityModule(path_printable_colors, 3)

        colors = p_mod.colors[:, :, 0, 0]
        colors = colors.reshape(5, 6, 3)

        for i in range(n):
            image.requires_grad = True
            print_loss = p_mod(image)
            print_loss.backward()
            image.requires_grad = False
            image -= 0.0005 * image.grad
            print('iteration %d : loss %f' % (i, print_loss))

        ax1.imshow(u.tensor_to_array(image), interpolation='nearest')
        ax1.set_title('im1 printability loss : %f' % print_loss)

        ax2.imshow(colors, interpolation='nearest')
        ax2.set_title('color set')

        plt.show()
        plt.close()

    def test_kMeans(self):
        weights = torch.ones(1, 3, 40, 40)
        kmeans = sklearn.cluster.KMeans(n_clusters=5)
        _, (ax1, ax2, ax3) = plt.subplots(1, 3)

        self.patch_trainer.model.eval()
        for image, true_label in self.patch_trainer.train_loader:
            image.requires_grad = True
            vector_scores = self.patch_trainer.model(image)
            model_label = torch.argmax(vector_scores.data).item()
            if model_label is not true_label.item() or model_label is \
                    self.patch_trainer.target_class:
                continue
            loss_target = -torch.nn.functional.log_softmax(vector_scores,
                                                           dim=1)[0, model_label]
            loss_target.backward()
            image.requires_grad = False
            ax1.imshow(u.tensor_to_array(image))
            output = torch.nn.functional.conv2d(torch.abs(image.grad), weights)
            output = torch.squeeze(output).numpy()
            ax2.imshow(output)
            output = np.abs(output)
            output = output / np.max(output)
            output = np.where(output < 0.3, 0, 1)
            ax3.imshow(output)
            x = np.transpose(output.nonzero())
            kmeans.fit(x)
            r = ax3.scatter(kmeans.cluster_centers_[:, 1],
                            kmeans.cluster_centers_[:, 0])
            plt.pause(5)
            r.remove()
        plt.show()
        plt.close()



if __name__ == '__main__':
    unittest.main()
