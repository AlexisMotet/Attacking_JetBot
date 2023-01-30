import numpy as np
import matplotlib.pyplot as plt
import math
import constants.constants as c

fx = 100
fy = 100
k1 = -0.32
cx = 112
cy = 112

ax = plt.figure().add_subplot(projection='3d')

def get_xx_yy_zz(image):
    xx = [point[0] for point in image]
    yy = [point[1] for point in image]
    zz = [point[2] for point in image]
    return xx, yy, zz

image = []
for x in range(224):
    for y in range(224):
        image.append(np.array([x, y, 1]))

projection_2d_to_3d = np.array([[1/fx, 0, -cx/fx],
                                [0, 1/fx, -cy/fx],
                                [0, 0, 0],
                                [0, 0, 1]])

image_3d = []
for point in image :
    projected = projection_2d_to_3d@point
    image_3d.append(projected)

beta = 1
ry = np.array([[math.cos(beta), 0, -math.sin(beta), 0],
               [0, 1, 0, 0],
               [math.sin(beta), 0, math.cos(beta), 0],
               [0, 0, 0, 1]])

image_rotated = []
for point_3d in image_3d :
    rotated = ry@(point_3d)
    image_rotated.append(rotated)
scale_factor = 1/2
translation = np.array([[1, 0, 0, 0],
                       [0, 1, 0, 0],
                       [0, 0, 1, 3],
                       [0, 0, 0, 1]])

image_translated = []
for point_3d in image_rotated :
    translated = translation@point_3d
    print(point_3d, translated)
    image_translated.append(translated)

image_distorted = []
for point_3d in image_translated :
    j, i = point_3d[:2]
    r2 = pow(j, 2) + pow(i, 2)
    d = 1/(1 - k1*r2)
    ni = i*d
    nj = j*d
    new_point_3d = point_3d.copy()
    new_point_3d[0] = nj
    new_point_3d[1] = ni
    image_distorted.append(new_point_3d)


projection_3d_to_2d = np.array([[fx, 0, cx, 0],
                                [0, fy, cy, 0],
                                [0, 0, 1, 0]])

final_image = []
for point_3d in image_distorted :
    projected = projection_3d_to_2d@point_3d
    projected[:2] = projected[:2]/projected[2]
    projected[2] = 2
    final_image.append(projected)
    
""" 
xx, yy, zz = get_xx_yy_zz(image_3d)
ax.scatter(xx, yy , zz)

xx, yy, zz = get_xx_yy_zz(image_rotated)
ax.scatter(xx, yy , zz)
 
xx, yy, zz = get_xx_yy_zz(image_translated)
ax.scatter(xx, yy , zz)

xx, yy, zz = get_xx_yy_zz(image_distorted)
ax.scatter(xx, yy , zz)
"""
xx, yy, zz = get_xx_yy_zz(final_image)
ax.scatter(xx, yy , zz)

ax.legend()
ax.set_xlabel('X')
ax.set_ylabel('Y')
ax.set_zlabel('Z')
ax.set_xlim(-200, 200)
ax.set_ylim(-200, 200)
ax.set_zlim(-200, 200)

ax.view_init(elev=30, azim=10, roll=90)

if __name__ == "__main__":
    plt.show()


