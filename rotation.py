import numpy as np
import cv2
import math
import torch

def tensor_to_array(tensor):
    tensor = torch.squeeze(tensor)
    array = tensor.numpy()
    return np.transpose(array, (1, 2, 0))

def array_to_tensor(array):
    array = np.transpose(array, (2, 0, 1))
    array = array[np.newaxis, :]
    return torch.tensor(array)

class RotationTool():
    def __init__(self, f=200):
        self.f = f
        self.edge_filter = [[-1,-1,-1],
                            [-1, 8,-1],
                            [-1,-1,-1]]
        
    def rotate(self, image, alpha, beta, gamma):
        image = tensor_to_array(image)
        w, h = image.shape[:2]
        matrix_2dto3d = np.array([[1, 0, -w/2],
                                  [0, 1, -h/2],
                                  [0, 0, 1],
                                  [0, 0, 1]])

        RX = np.array([[1, 0, 0, 0],
                       [0, math.cos(alpha), -math.sin(alpha), 0],
                       [0, math.sin(alpha), math.cos(alpha), 0],
                       [0, 0, 0, 1]])
        
        RY = np.array([[math.cos(beta), 0, -math.sin(beta), 0],
                       [0, 1, 0, 0],
                       [math.sin(beta), 0, math.cos(beta), 0],
                       [0, 0, 0, 1]])
        
        RZ = np.array([[math.cos(gamma), -math.sin(gamma), 0, 0],
                       [math.sin(gamma), math.cos(gamma), 0, 0],
                       [0, 0, 1, 0],
                       [0, 0, 0, 1]])
        
        R = RX @ RY @ RZ

        matrix_translation = np.array([[1, 0, 0, 0],
                                       [0, 1, 0, 0],
                                       [0, 0, 1, self.f],
                                       [0, 0, 0, 1]])

        matrix_3dto2d = np.array([[self.f, 0, w/2, 0],
                                  [0, self.f, h/2, 0],
                                  [0, 0, 1, 0]])
        
        matrix_transformation = matrix_3dto2d @ (matrix_translation @ 
                                (R @ matrix_2dto3d))

        output = cv2.warpPerspective(image, matrix_transformation, 
                                    image.shape[:2],
                                    cv2.INTER_CUBIC | cv2.WARP_INVERSE_MAP)
        
        return array_to_tensor(output), matrix_transformation
    
    def get_angle(self, mask):
        mask = tensor_to_array(mask)[:, :, 0]
        mask = mask.astype(np.uint8)
        
        blur = cv2.blur(mask, (3,3))
        canny = cv2.Canny(blur, 0, 0)

        contours, _ = cv2.findContours(canny, 
                                       cv2.RETR_TREE, 
                                       cv2.CHAIN_APPROX_SIMPLE)
        for c in sorted(contours, key=cv2.contourArea) :
            if len(c) > 5 :
                ellipse = cv2.fitEllipse(c)
                _, _, angle = ellipse
                return angle
        return 90
            
    def undo_rotate(self, image, matrix_transformation):
        image = tensor_to_array(image)
        inverse_transformation = np.linalg.pinv(matrix_transformation)
        output = cv2.warpPerspective(image, inverse_transformation, 
                                    image.shape[:2],
                                    cv2.INTER_CUBIC | cv2.WARP_INVERSE_MAP)

        return array_to_tensor(output)

if __name__=="__main__":
    import matplotlib.pyplot as plt
    _, (ax1, ax2, ax3) = plt.subplots(1, 3)
    
    path = "C:\\Users\\alexi\\PROJET_3A\\projet_3A\\img\\epoch1_batch120_label0_original.png"
    zeros, grid = test2.grid(path)
    row0, col0 = np.random.choice(224 - 40, size=2)
    mask = torch.zeros(1, 3, 224, 224)
    mask[0, :, row0:row0+40, col0:col0+40] = torch.ones(40, 40)
    
    print(row0, col0)
    angle = grid[row0, col0]
    min = 180
    last_angle_2d, rad_save = 0, 0
    rotation_tool = RotationTool()
    for i in range(-20, 20):
        rad = math.radians(5 * i)
        # ax1.imshow(tensor_to_array(mask), interpolation='nearest')
        # ax1.set_title('before rotation')
        rot, mat = rotation_tool.rotate(mask, 0, rad, 0)
        
        angle_2d = rotation_tool.get_angle(rot)
        if (abs(angle_2d - angle) < min):
            min = abs(angle_2d - angle)
            rad_save = rad
            angle_2d_save = angle
        
        # ax2.imshow(tensor_to_array(rot), interpolation='nearest')
        # ax2.set_title('rotation angle %d 2d %d' % (angle, angle_2d))
        
        # plt.pause(0.5)
    ax1.imshow(zeros, interpolation='nearest') 
    image = cv2.imread(path)
    ax2.imshow(image, interpolation='nearest')
    ax2.set_title("angle %d" % angle)
    rot, mat = rotation_tool.rotate(mask, 0, rad_save, 0)
    ax3.imshow(tensor_to_array(rot), interpolation='nearest')
    ax3.set_title("angle_2d %d" % angle_2d_save)
    plt.show()
