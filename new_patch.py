import torch
import torchvision
import datetime
import image_transformation.image_transformation as i
import pickle
import utils.utils as u
import constants.constants as c
import transformation.transformation as t

class PatchTrainer():
    def __init__(self, mode=c.Mode.TARGET, validation=True, 
                 target_class=1, patch_relative_size=0.08, n_epochs=2, 
                 threshold=0.9, max_iterations=10):

        self.pretty_printer = u.PrettyPrinter(self)
        self.date = datetime.datetime.now().strftime("%d/%m/%Y %H:%M:%S")
        self.path_model = c.PATH_MODEL
        self.path_dataset = c.PATH_DATASET
        self.limit_train_epoch_len = c.LIMIT_TRAIN_EPOCH_LEN
        self.limit_test_len = c.LIMIT_TEST_LEN
        self.mode = mode
        self.validation = validation
        self.target_class = target_class
        self.patch_relative_size = patch_relative_size
        self.n_epochs = n_epochs
        self.threshold = threshold
        self.max_iterations = max_iterations

        self.model = u.load_model(self.path_model, n_classes=c.N_CLASSES)
        self.model.eval()
        self.train_loader, self.test_loader = u.load_dataset(self.path_dataset)

        image_size = c.IMAGE_DIM ** 2
        patch_size = image_size * self.patch_relative_size
        self.patch_dim = int(patch_size ** (0.5))

        self.extrinsic_module = i.ExtrinsicModule()
        
        self.intrinsic_module = i.IntrinsicModule()
        
        self.transformation_tool = t.TransformationTool(self.patch_dim)
        
        self.patch = self._random_patch_init()

        self.target_proba_train = {}
        self.success_rate_test = {}

    def _random_patch_init(self):
        patch = torch.zeros(1, 3, c.IMAGE_DIM, c.IMAGE_DIM)
        row0, col0 = c.IMAGE_DIM//2 - self.patch_dim//2, \
                     c.IMAGE_DIM//2 - self.patch_dim//2
        patch[0, :, row0:row0 + self.patch_dim, 
                    col0:col0 + self.patch_dim] = torch.rand(3, self.patch_dim, 
                                                                self.patch_dim)
        return patch
    
    def test_model(self):
        success, total = 0, 0
        for loader in [self.train_loader, self.test_loader]:
            for image, label in loader:
                output = self.model(self.extrinsic_module(image))
                success += (label == output.argmax(1)).sum()
                total += len(label)
                print('sucess/total : %d/%d accuracy : %.2f' 
                        % (success, total, (100 * success / float(total))))
        
    @staticmethod
    def _get_mask(image):
        mask = torch.zeros_like(image)
        mask[image != 0] = 1
        return mask
    
    def _get_patch(self):
        row0, col0 = c.IMAGE_DIM//2 - self.patch_dim//2, \
                     c.IMAGE_DIM//2 - self.patch_dim//2
        return self.patch[:, :, row0:row0 + self.patch_dim, 
                                col0:col0 + self.patch_dim]

    def attack(self, image):
        transformed, map_ = self.transformation_tool.random_transform(self.patch)
        mask = self._get_mask(transformed)
        transformed.requires_grad = True
        for i in range(self.max_iterations + 1) :
            torch.clamp(transformed, 0, 1)
            modified = self.intrinsic_module(transformed)
            attacked = torch.mul(1 - mask, image) + \
                       torch.mul(mask, modified)
            normalized = self.extrinsic_module(attacked)
            vector_scores = self.model(normalized)
            vector_proba = torch.nn.functional.softmax(vector_scores, dim=1)
            target_proba = float(vector_proba[0, self.target_class])
            
            if i == 0: first_target_proba = target_proba
            else: self.pretty_printer.update_iteration(i, target_proba)

            if self.mode in [c.Mode.TARGET, c.Mode.TARGET_AND_FLEE]:
                if target_proba >= self.threshold : break    
            elif self.mode == c.Mode.FLEE :
                if target_proba <= self.threshold : break
            else : assert False
                
            loss = -torch.nn.functional.log_softmax(vector_scores, dim=1)
            if self.mode == c.Mode.TARGET :
                loss[0, self.target_class].backward()
                with torch.no_grad():
                    transformed -= transformed.grad
            elif self.mode == c.Mode.FLEE:
                loss[0, self.target_class].backward()
                with torch.no_grad():
                    transformed += transformed.grad
            elif self.mode == c.Mode.TARGET_AND_FLEE:
                loss[0, self.target_class].backward(retain_graph=True)
                with torch.no_grad():
                    transformed -= transformed.grad 
                model_label = int(torch.argmax(vector_scores))
                if model_label != self.target_class :
                    transformed.grad.zero_()
                    loss[0, model_label].backward()
                    with torch.no_grad():    
                        transformed += transformed.grad
            transformed.grad.zero_()
        self.patch = self.transformation_tool.undo_transform(self.patch, 
                                                             transformed.detach(),
                                                             map_)
        return first_target_proba, attacked

    def train(self):
        self.pretty_printer.training()
        for epoch in range(self.n_epochs):
            total, success = 0, 0
            self.target_proba_train[epoch] = []
            for image, true_label in self.train_loader:
                vector_scores = self.model(self.extrinsic_module(image))
                model_label = int(torch.argmax(vector_scores))
                if model_label != int(true_label) :
                    continue
                elif self.mode in [c.Mode.TARGET, c.Mode.TARGET_AND_FLEE] and \
                        model_label == self.target_class:
                    continue
                elif self.mode == c.Mode.FLEE and \
                        model_label != self.target_class:
                    continue

                if total == 0 :
                    success_rate = None
                else :
                    success_rate = 100 * (success / float(total))
                    
                self.pretty_printer.update_image(epoch, success_rate, total)
        
                total += 1
                
                self.extrinsic_module.jitter()
                self.intrinsic_module.jitter()
                
                first_target_proba, attacked = self.attack(image)
                self.target_proba_train[epoch].append(first_target_proba)

                if self.mode in [c.Mode.TARGET, c.Mode.TARGET_AND_FLEE] :
                    if first_target_proba >= self.threshold : 
                        success += 1
                elif self.mode == c.Mode.FLEE :
                    if first_target_proba <= self.threshold : 
                        success += 1

                if total % c.N_ENREG_IMG == 0:
                    torchvision.utils.save_image(image, 
                                                 c.PATH_IMG_FOLDER + 
                                                 'epoch%d_image%d_label%d_original.png'
                                                 % (epoch, total, int(true_label)))

                    torchvision.utils.save_image(attacked, 
                                                 c.PATH_IMG_FOLDER + 
                                                 'epoch%d_image%d_attacked.png'
                                                 % (epoch, total))

                    torchvision.utils.save_image(self.patch, 
                                                 c.PATH_IMG_FOLDER + 
                                                 'epoch%d_image%d_patch.png'
                                                 % (epoch, total))
                    
                if self.limit_train_epoch_len is not None and \
                        total >= self.limit_train_epoch_len :
                    break
            if self.validation:
                self.test(epoch)

    def test(self, epoch=-1):
        total, success = 0, 0
        for image, true_label in self.test_loader:
            self.extrinsic_module.jitter()
            self.intrinsic_module.jitter()
            vector_scores = self.model(self.extrinsic_module(image))
            model_label = int(torch.argmax(vector_scores))
            if model_label != int(true_label) :
                continue
            elif self.mode in [c.Mode.TARGET, c.Mode.TARGET_AND_FLEE] \
                    and model_label == self.target_class:
                continue
            elif self.mode == c.Mode.FLEE and model_label != self.target_class:
                continue
            
            total += 1
            
            transformed, _ = self.transformation_tool.random_transform(self.patch)
            mask = self._get_mask(transformed)
            modified = self.intrinsic_module(transformed)
            attacked = torch.mul(1 - mask, image) + torch.mul(mask, modified)
            normalized = self.extrinsic_module(attacked)
            vector_scores = self.model(normalized)
            attacked_label = int(torch.argmax(vector_scores))
            vector_proba = torch.nn.functional.softmax(vector_scores, dim=1)
            target_proba = float(vector_proba[0, self.target_class])
            
            if self.mode in [c.Mode.TARGET, c.Mode.TARGET_AND_FLEE] :
                if attacked_label == self.target_class: 
                    success += 1
            elif self.mode == c.Mode.FLEE :
                if attacked_label != self.target_class: 
                    success += 1

            if total % c.N_ENREG_IMG == 0:
                torchvision.utils.save_image(normalized, c.PATH_IMG_FOLDER + 
                                             'test_epoch%d_target_proba%.2f_label%d.png'
                                             % (epoch, target_proba, attacked_label))
            self.pretty_printer.update_test(epoch, 100 * success / float(total), 
                                            total)
            if self.limit_test_len is not None and total >= self.limit_test_len :
                break
        self.success_rate_test[epoch] = 100 * (success / float(total))

    def save_patch(self, path):
        self.model = None
        self.train_loader = None
        self.test_loader = None
        self.patch = self._get_patch()
        pickle.dump(self, open(path, "wb"))

