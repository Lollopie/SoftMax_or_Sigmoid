import json

from PIL import Image
import torch
from torchvision.transforms import v2
import numpy as np
from pathlib import Path

dataset = Path('./datasets/2025 Data Fusion RGB')
with open(dataset / 'dataset.json') as f:
    datasetDetails = json.load(f)
pathlist = (dataset / datasetDetails['InputDirectory']).rglob('*.tif')
labelPath = dataset / datasetDetails['LabelDirectory']
trainPath = dataset / 'train'
testPath = dataset / 'test'
valPath = dataset / 'val'
for directory in [trainPath, testPath, valPath]:
    (directory / 'Images').mkdir(parents=True, exist_ok=True)
    (directory / 'Labels').mkdir(parents=True, exist_ok=True)
paths = []
for path in pathlist:
    path_in_str = path
    paths.append(path_in_str)
trainNumber = np.ceil(len(paths) * 0.6)
testNumber = np.ceil(len(paths) * 0.2)
valNumber = len(paths) - trainNumber - testNumber
N = datasetDetails['NumberOfImagesPerInputImage']
targetHeight = datasetDetails['TargetImageSize']
targetWidth = datasetDetails['TargetImageSize']
toImageTransform = v2.Compose(
    [
        v2.ToImage(),
    ]
)
datasetTransformation = v2.Compose(
    [
        v2.RandomCrop((targetHeight, targetWidth)),
        v2.RandomHorizontalFlip(0.5),
        v2.RandomVerticalFlip(0.5),
    ]
)
inputTransformation = v2.Compose(
    [
        v2.ToDtype(torch.float32, scale=True),
    ]
)
for index, path in enumerate(paths):
    img = Image.open(path)
    label = Image.open(labelPath / path.name)
    img = toImageTransform(img)
    label = toImageTransform(label)
    savePath = trainPath if index < trainNumber else (testPath if index < trainNumber + testNumber else valPath)
    for i in range(N):
        sampleImg, sampleLabel = datasetTransformation(img, label)
        sampleImg = inputTransformation(sampleImg)
        v2.ToPILImage()(sampleImg).save(savePath / 'Images' / (path.stem + '_' + str(i) + '.tif'))
        v2.ToPILImage()(sampleLabel).save(savePath / 'Labels' / (path.stem + '_' + str(i) + '.tif'))