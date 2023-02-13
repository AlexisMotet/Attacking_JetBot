import cv2
import matplotlib.pyplot as plt
import numpy as np

show = False
width = 24
if __name__ == "__main__": 
    patches = np.empty((100, width, width, 3))
    means = []
    for i in range(100):
        img = cv2.imread("noise\\noise%d.jpeg" % i)
        patch = img[112-width//2:112+width//2, 112-width//2:112+width//2, :]
        patch = cv2.cvtColor(patch, cv2.COLOR_BGR2RGB)/255
        if show :
            plt.imshow(patch)
            plt.pause(0.5)
            plt.title(i)
        means.append(np.mean(patch, axis=(0, 1)))
        patches[i, :] = patch
    
    mean_ = np.mean(patches, axis=0)
    variances = []
    for patch in patches : 
        var = np.mean((patch - mean_) ** 2, axis=(0,1))
        variances.append(var)
    std_ = np.sqrt(np.mean(variances, axis=0))
    print("std=%s" % std_)


