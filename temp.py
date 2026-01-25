import numpy as np
import rasterio
import torch
from PIL import Image
from torchvision.transforms import v2

img = Image.open('./datasets/Potsdam/train/Images/top_potsdam_2_10_2.tif')
labelSrc = rasterio.open('./datasets/Potsdam/train/Images/top_potsdam_2_10_2.tif'.replace('Images', 'Labels').replace('rgb', 'label'), mode='r')
label = labelSrc.read(list(range(1, 3+1))).astype(np.uint8)
imgTransformed = v2.ToDtype(torch.float32, scale=True)(v2.ToImage()(img))
print(imgTransformed)