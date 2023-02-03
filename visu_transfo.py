import numpy as np
import matplotlib.pyplot as plt
import math

fx = 108
fy = 139
k1 = -0.32
cx = 10
cy = 10

ax = plt.figure().add_subplot(projection='3d')

def _set_axes_radius(ax, origin, radius):
    x, y, z = origin
    ax.set_xlim3d([x - radius, x + radius])
    ax.set_ylim3d([y - radius, y + radius])
    ax.set_zlim3d([z - radius, z + radius])

def set_axes_equal(ax):
    limits = np.array([
        ax.get_xlim3d(),
        ax.get_ylim3d(),
        ax.get_zlim3d(),
    ])
    origin = np.mean(limits, axis=1)
    radius = 0.5 * np.max(np.abs(limits[:, 1] - limits[:, 0]))
    _set_axes_radius(ax, origin, radius)

def get_xx_yy_zz(image, normalized=True):
    xx = [point[0] for point in image]
    yy = [point[1] for point in image]
    zz = [point[2] for point in image]
    return xx, yy, zz

image = []
for x in range(20):
    for y in range(20):
        image.append(np.array([x, y, 1]))

projection_2d_to_3d = np.array([[1/fx, 0, -cx/fx],
                                [0, 1/fy, -cy/fy],
                                [0, 0, 0],
                                [0, 0, 1]])

image_3d = []
for point in image :
    projected = projection_2d_to_3d@point
    image_3d.append(projected)

r = np.array([[1, 0, -0.024, 0],
               [0, 1, 0.17, 0],
               [0.025, -0.17, 1, 0],
               [0, 0, 0, 1]])

image_rotated = []
for point_3d in image_3d :
    rotated = r@point_3d
    image_rotated.append(rotated)

scale_factor = 1/3
translation = np.array([[1, 0, 0, 0],
                       [0, 1, 0, 0],
                       [0, 0, 1, 1/scale_factor],
                       [0, 0, 0, 1]])

image_translated = []
for point_3d in image_rotated :
    translated = translation@point_3d
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

projection_3d_to_2d = np.array([[fx, 0, 0, 0],
                                [0, fy, 0, 0],
                                [0, 0, 1, 0]])

final_image = []
for point_3d in image_distorted :
    projected = projection_3d_to_2d@point_3d
    projected[:2] = projected[:2]/projected[2]
    projected[2] = 0
    final_image.append(projected)
    
ax.scatter([0], [0], [0], label="zero")

"""
xx, yy, zz = get_xx_yy_zz(image_3d)
scatter = ax.scatter(xx, yy , zz, label="image 3d")

xx, yy, zz = get_xx_yy_zz(image_rotated)
ax.scatter(xx, yy , zz, label="rotated")
"""

xx, yy, zz = get_xx_yy_zz(image_translated)
ax.scatter(xx, yy , zz, label="translated")


xx, yy, zz = get_xx_yy_zz(image_distorted)
ax.scatter(xx, yy , zz, label="distorted")

xx, yy, zz = get_xx_yy_zz(final_image)
ax.scatter(xx, yy , zz, label="result")
ax.legend()

ax.set_xlabel('X')
ax.set_ylabel('Y')
ax.set_zlabel('Z')
ax.set_xlim(-20, 20)
ax.set_ylim(-20, 20)
ax.set_zlim(-20, 20)

ax.view_init(elev=30, azim=10, roll=90)

ax.set_box_aspect([1,1,1])
set_axes_equal(ax)

if __name__ == "__main__":
    plt.show()


