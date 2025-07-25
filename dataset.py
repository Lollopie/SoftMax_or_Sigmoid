import os

from PIL import Image
import torch
import torch.utils.data
from torch.utils.data import Dataset, DataLoader
from torchvision import models, datasets, tv_tensors
from torchvision.transforms import v2
from torch import nn
import numpy as np
from pathlib import Path
pathlist = Path('./2_Ortho_RGB/2_Ortho_RGB/').rglob('*.tif')
labelPath = Path('./5_Labels_all/')
trainPath = Path('./trainImages')
testPath = Path('./testImages')
valPath = Path('./validationImages')
paths = []
for path in pathlist:
    path_in_str = path
    paths.append(path_in_str)
trainNumber = np.ceil(len(paths) * 0.6)
testNumber = np.ceil(len(paths) * 0.2)
valNumber = len(paths) - trainNumber - testNumber
N = 100
targetHeight = 512
targetWidth = 512
toImageTransform = v2.Compose(
    [
        v2.ToImage(),
    ]
)
datasetTransformation = v2.Compose(
    [
        v2.RandomCrop((targetHeight, targetWidth)),
        v2.RandomHorizontalFlip(0.5),
    ]
)
inputTransformation = v2.Compose(
    [
        v2.ToDtype(torch.float32, scale=True),
        # v2.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229,0.224,0.225]),
    ]
)
for index, path in enumerate(paths):
    img = Image.open(path)
    label = Image.open(labelPath / (path.stem[:-3] + 'label.tif'))
    img = toImageTransform(img)
    label = toImageTransform(label)
    savePath = trainPath if index < trainNumber else (testPath if index < trainNumber + testNumber else valPath)
    for i in range(N):
        sampleImg, sampleLabel = datasetTransformation(img, label)
        sampleImg = inputTransformation(sampleImg)
        v2.ToPILImage()(sampleImg).save(savePath / (path.stem[:-3] + str(i) + '.tif'))
        v2.ToPILImage()(sampleLabel).save(savePath / (path.stem[:-3] + str(i) + '_label.tif'))