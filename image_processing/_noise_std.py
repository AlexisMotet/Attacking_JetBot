import cv2
import matplotlib.pyplot as plt
import numpy as np

width = 24
if __name__ == "__main__": 
    patches = []
    means = []
    for i in range(100):
        img = cv2.imread("noise\\noise%d.jpeg" % i)
        patch = img[112-width//2:112+width//2, 112-width//2:112+width//2, :]
        patch = cv2.cvtColor(patch, cv2.COLOR_BGR2RGB)/255
        if i == 0 :
            plt.imshow(patch)
            plt.show()
        means.append(np.mean(patch, axis=(0, 1)))
        patches.append(patch)
        
    mean_ = np.mean(means, axis=0)
    variances = []
    for patch in patches : 
        var = np.mean((patch - mean_) ** 2, axis=(0,1))
        variances.append(var)
    std_ = np.sqrt(np.mean(variances, axis=0))
    print(std_)



