import scipy.io
import numpy as np
import h5py

f = h5py.File("matlab.mat", "r")
data = f.get("RadialDistortion")
print(data)