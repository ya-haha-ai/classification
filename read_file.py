from glob import glob
import numpy as np
from scipy.io import loadmat
import rasterio as rio

S_sentinel_bands = glob("E:/3d-data/sundarbans_data-main/*B?*.tiff")
S_sentinel_bands.sort()

l = []
for i in S_sentinel_bands:
  with rio.open(i, 'r') as f:
    l.append(f.read(1))

# Data
arr_st = np.stack(l)

# Ground Truth
y_data = loadmat('E:/3d-data/sundarbans_data-main/Sundarbands_gt.mat')['gt']
print(y_data)