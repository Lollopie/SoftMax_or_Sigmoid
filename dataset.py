import json

from PIL import Image
import torch
from torchvision.transforms import v2
import numpy as np
from pathlib import Path

dataset = Path('./datasets/Potsdam')
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
#Train, Test, Val split of 60%, 20% and remainder
trainNumber = np.ceil(len(paths) * 0.6)
testNumber = np.ceil(len(paths) * 0.2)
valNumber = len(paths) - trainNumber - testNumber
N = datasetDetails['NumberOfImagesPerInputImage']
targetHeightTrain = datasetDetails['TargetImageSizeTrain']
targetHeightVal = datasetDetails['TargetImageSizeVal']
targetHeightTest = datasetDetails['TargetImageSizeTest']
targetWidthTrain = datasetDetails['TargetImageSizeTrain']
targetWidthVal = datasetDetails['TargetImageSizeVal']
targetWidthTest = datasetDetails['TargetImageSizeTest']
toImageTransform = v2.Compose(
    [
        v2.ToImage(),
    ]
)
datasetTransformationTrain = v2.Compose(
    [
        v2.RandomCrop((targetHeightTrain, targetWidthTrain)),
        v2.RandomHorizontalFlip(0.5),
        v2.RandomVerticalFlip(0.5),
    ]
)
datasetTransformationVal = v2.Compose(
    [
        v2.RandomCrop((targetHeightVal, targetWidthVal)),
        v2.RandomHorizontalFlip(0.5),
        v2.RandomVerticalFlip(0.5),
    ]
)
datasetTransformationTest = v2.Compose(
    [
        v2.RandomCrop((targetHeightTest, targetWidthTest)),
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
    datasetTransformation = datasetTransformationTrain if index < trainNumber else (datasetTransformationTest if index < trainNumber + testNumber else datasetTransformationVal)
    for i in range(N):
        sampleImg, sampleLabel = datasetTransformation(img, label)
        sampleImg = inputTransformation(sampleImg)
        v2.ToPILImage()(sampleImg).save(savePath / 'Images' / (path.stem + '_' + str(i) + '.tif'))
        v2.ToPILImage()(sampleLabel).save(savePath / 'Labels' / (path.stem + '_' + str(i) + '.tif'))