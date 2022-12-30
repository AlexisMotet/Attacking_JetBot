import new_patch
import torch
import numpy as np
import matplotlib.pyplot as plt
import cv2
import sklearn.cluster
import time

def tensor_to_numpy_array(tensor):
    tensor = torch.squeeze(tensor)
    array = tensor.detach().cpu().numpy()
    if (len(array.shape)==3):
        return np.transpose(array, (1, 2, 0))
    return array

path_model = 'C:\\Users\\alexi\\PROJET_3A\\Projet Adversarial Patch\\Project Adverserial Patch\\Collision Avoidance\\best_model_extended.pth'
path_dataset = 'C:\\Users\\alexi\\PROJET_3A\\Projet Adversarial Patch\\Project Adverserial Patch\\Collision Avoidance\\dataset'
path_calibration = 'C:\\Users\\alexi\\PROJET_3A\\projet_3A\\calibration\\'
path_printable_vals = 'C:\\Users\\alexi\\PROJET_3A\\projet_3A\\printable_vals.dat'

if __name__=="__main__":
    weights = torch.ones(1, 3, 40, 40)
    kmeans = sklearn.cluster.KMeans()
    _, (ax1, ax2, ax3) = plt.subplots(1, 3)
    patch_trainer = new_patch.PatchTrainer(path_model,
                                            path_dataset,
                                            path_calibration,
                                            path_printable_vals)
    patch_trainer.model.eval()
    for image, true_label in patch_trainer.train_loader :
        var_image = torch.autograd.Variable(image, requires_grad=True)
        vector_scores = patch_trainer.model(var_image)
        model_label = torch.argmax(vector_scores.data).item()
        if model_label is not true_label.item() or model_label is patch_trainer.target_class  :
            continue
        loss_target = -torch.nn.functional.log_softmax(vector_scores, dim=1)[0, 0]
        loss_target.backward()
        grad = var_image.grad.clone()
        ax1.imshow(tensor_to_numpy_array(image))
        with torch.no_grad() :
            output = torch.nn.functional.conv2d(torch.abs(grad), weights)
            output = torch.squeeze(output)
            output = output.numpy()
            ax2.imshow(output)
            output = np.abs(output)
            output = output/np.max(output)
            output = np.where(output < 0.5, 0, 1)
            ax3.imshow(output)
            X  = np.transpose(output.nonzero())
            kmeans.fit(X)
            r = ax3.scatter(kmeans.cluster_centers_[:, 1],
                        kmeans.cluster_centers_[:, 0])
            plt.pause(5)
            r.remove()
            
    plt.show()
    plt.close() 