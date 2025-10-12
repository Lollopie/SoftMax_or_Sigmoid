import os

import rasterio
from ignite.engine import Engine
from ignite.handlers import ProgressBar, Checkpoint
from ignite.metrics import ConfusionMatrix
from ignite.metrics import mIoU
from ignite.metrics import Metric
from ignite.metrics import Accuracy
from ignite.exceptions import NotComputableError
from ignite.metrics.metric import sync_all_reduce, reinit__is_reduced
from ignite.engine import Events

from PIL import Image
import torch
import torch.utils.data
from torch.utils.data import Dataset, DataLoader
from torchvision.transforms import v2
from torchvision import tv_tensors
from torch import nn
import numpy as np
import json
from pathlib import Path

# p = Path('/content/drive/MyDrive/') if Path('/content/drive/MyDrive/').exists() else Path('.')

"""# 1. Image Preprocessing"""

transformPILImageToTvTensorImage = v2.Compose(
    [
        v2.ToImage(),
    ]
)

"""#TODO: Properly set ignore index

# 2. Dataset Setup
"""
dataset = Path('./datasets/Potsdam')
with open(dataset / 'dataset.json') as f:
    datasetDetails = json.load(f)
colorDict = {}
for i, color in enumerate(datasetDetails['LabelMapping']):
    colorDict[torch.tensor([color['R'], color['G'], color['B']])] = i
indexToIgnore = None
targetHeight = datasetDetails['TargetImageSize']
targetWidth = datasetDetails['TargetImageSize']

def transformLabel(label: torch.Tensor, colorDictionary: dict[torch.Tensor, int], numberOfLabelChannels: int):
    label = label.to(torch.int64)
    label2D = label.reshape(numberOfLabelChannels, -1).transpose(0, 1)
    indexMap = label2D
    if numberOfLabelChannels > 1:
        indexMap = torch.zeros(label2D.shape[0])
        for color, index in colorDictionary.items():
            indexMap += torch.where(torch.all(label2D == color, dim=1), index, 0)
        indexMap = indexMap.to(torch.int64)
    labelOHE = nn.functional.one_hot(indexMap, num_classes=len(colorDictionary.keys())).transpose(0, 1).reshape(
        len(colorDictionary.keys()), targetHeight, targetWidth).to(torch.float16)
    return labelOHE, indexMap.reshape(targetHeight, targetWidth)

transformInputImage = v2.Compose(
    [
        v2.ToDtype(torch.float32, scale=True),
    ]
)

class EarthObservationImageryDataset(Dataset):
    def __init__(self, listOfPaths, numberOfLabelChannels):
        self.listOfPaths = listOfPaths
        self.numberOfLabelChannels = numberOfLabelChannels

    def __getitem__(self, idx):
        img = Image.open(self.listOfPaths[idx])
        labelSrc = rasterio.open(str(self.listOfPaths[idx]).replace('Images', 'Labels').replace('rgb', 'label'), mode='r')
        label = labelSrc.read(list(range(1, self.numberOfLabelChannels+1))).astype(np.uint8)
        imgTransformed = transformInputImage(transformPILImageToTvTensorImage(img))
        labelTransformed = transformLabel(torch.tensor(label), colorDict, self.numberOfLabelChannels)
        return {'img': imgTransformed, 'label': labelTransformed}

    def __len__(self):
        return len(self.listOfPaths)

def loadDataset(basePath: Path):
    dataset = []
    pathGenerator = (basePath / 'Images').rglob('*.tif')
    for path in pathGenerator:
        dataset.append(path)
    return dataset

trainDataset = loadDataset(Path(dataset / 'train_old'))
valDataset = loadDataset(Path(dataset / 'val_old'))
testDataset = loadDataset(Path(dataset / 'test_old'))
trainDataset = EarthObservationImageryDataset(trainDataset, datasetDetails['NumberOfLabelChannels'])
valDataset = EarthObservationImageryDataset(valDataset, datasetDetails['NumberOfLabelChannels'])
testDataset = EarthObservationImageryDataset(testDataset, datasetDetails['NumberOfLabelChannels'])
inputChannels = datasetDetails['NumberOfInputChannels']

TRAIN_BATCH_SIZE = 1
VAL_BATCH_SIZE = 1
TEST_BATCH_SIZE = 1
LEARNING_RATE = 1e-04
EPOCHS = 50
device = 'cuda' if torch.cuda.is_available() else 'cpu'

trainLoader = DataLoader(trainDataset, batch_size=TRAIN_BATCH_SIZE, shuffle=True, num_workers=0, drop_last=True)
valLoader = DataLoader(valDataset, batch_size=VAL_BATCH_SIZE, shuffle=False, num_workers=0, drop_last=True)
testLoader = DataLoader(testDataset, batch_size=TEST_BATCH_SIZE, shuffle=False, num_workers=0, drop_last=True)
"""# 3. Model Setup"""
class UNetEncoding(nn.Sequential):
    def __init__(self, channels):
        super().__init__(nn.MaxPool2d(kernel_size=(2, 2), stride=2),
                         nn.Conv2d(channels // 2, channels, kernel_size=(3, 3), padding='same'), nn.ReLU(),
                         nn.Conv2d(channels, channels, kernel_size=(3, 3), padding='same'), nn.ReLU())
class UNetUpConv(nn.Sequential):
    def __init__(self, channels):
        super().__init__(nn.Upsample(scale_factor=2.0, mode='bilinear'),
                         nn.Conv2d(channels * 2, channels, kernel_size=(2, 2), padding='same'))
class UNetDecoding(nn.Sequential):
    def __init__(self, channels):
        super().__init__(nn.Conv2d(channels * 2, channels, kernel_size=(3, 3), padding='same'), nn.ReLU(),
                       nn.Conv2d(channels, channels, kernel_size=(3, 3), padding='same'), nn.ReLU())

class UNet(nn.Module):
    def __init__(self, inputChannels, outLength):
        super().__init__()
        self.contracting = nn.ModuleList()
        currentChannels = 64
        self.input = nn.Sequential(nn.Conv2d(inputChannels, currentChannels, kernel_size=(3, 3), padding='same'), nn.ReLU(),
                         nn.Conv2d(currentChannels, currentChannels, kernel_size=(3, 3), padding='same'), nn.ReLU())
        self.encoding1 = UNetEncoding(currentChannels := currentChannels * 2)
        self.encoding2 = UNetEncoding(currentChannels := currentChannels * 2)
        self.encoding3 = UNetEncoding(currentChannels := currentChannels * 2)
        self.encoding4 = UNetEncoding(currentChannels := currentChannels * 2)
        self.upConv1 = UNetUpConv(currentChannels := currentChannels // 2)
        self.decoding1 = UNetDecoding(currentChannels)
        self.upConv2 = UNetUpConv(currentChannels := currentChannels // 2)
        self.decoding2 = UNetDecoding(currentChannels)
        self.upConv3 = UNetUpConv(currentChannels := currentChannels // 2)
        self.decoding3 = UNetDecoding(currentChannels)
        self.upConv4 = UNetUpConv(currentChannels := currentChannels // 2)
        self.decoding4 = UNetDecoding(currentChannels)
        self.output = nn.Sequential(nn.Conv2d(currentChannels, outLength, kernel_size=(1, 1)))

    def forward(self, x):
        x1 = self.input(x)
        x2 = self.encoding1(x1)
        x3 = self.encoding2(x2)
        x4 = self.encoding3(x3)
        x = self.encoding4(x4)
        x = self.upConv1(x)
        x = torch.cat((x4, x), dim=1)
        x = self.decoding1(x)
        x = self.upConv2(x)
        x = torch.cat((x3, x), dim=1)
        x = self.decoding2(x)
        x = self.upConv3(x)
        x = torch.cat((x2, x), dim=1)
        x = self.decoding3(x)
        x = self.upConv4(x)
        x = torch.cat((x1, x), dim=1)
        x = self.decoding4(x)
        x = self.output(x)
        return x

class DeepResidualUNet(nn.Module):
    def __init__(self, input_channels, out_length, base_channels = 64):
        # Add Convolutional Layers with Kernel 1 Stride 1 Padding same to increase channels of residual input
        super().__init__()
        self.encoding1 = nn.Sequential(nn.Conv2d(input_channels, base_channels, kernel_size=(3, 3), padding='same'),
                                       nn.BatchNorm2d(base_channels), nn.ReLU(),
                                       nn.Conv2d(base_channels, base_channels, kernel_size=(3, 3), padding='same'))
        self.upproject1 = nn.Sequential(
            nn.Conv2d(input_channels, base_channels, kernel_size=(1, 1), stride=1, padding='same'),
            nn.BatchNorm2d(base_channels))
        current_channels = base_channels * 2
        self.encoding2 = nn.Sequential(nn.BatchNorm2d(base_channels), nn.ReLU(),
                                       nn.Conv2d(base_channels, current_channels, kernel_size=(3, 3), stride=2,
                                                 padding=1), nn.BatchNorm2d(current_channels), nn.ReLU(),
                                       nn.Conv2d(current_channels, current_channels, kernel_size=(3, 3),
                                                 padding='same'))
        self.upproject2 = nn.Sequential(
            nn.Conv2d(current_channels // 2, current_channels, kernel_size=(1, 1), stride=2),
            nn.BatchNorm2d(current_channels))
        current_channels *= 2
        self.encoding3 = nn.Sequential(nn.BatchNorm2d(current_channels // 2), nn.ReLU(),
                                       nn.Conv2d(current_channels // 2, current_channels, kernel_size=(3, 3), stride=2,
                                                 padding=1), nn.BatchNorm2d(current_channels), nn.ReLU(),
                                       nn.Conv2d(current_channels, current_channels, kernel_size=(3, 3),
                                                 padding='same'))
        self.upproject3 = nn.Sequential(
            nn.Conv2d(current_channels // 2, current_channels, kernel_size=(1, 1), stride=2),
            nn.BatchNorm2d(current_channels))
        current_channels *= 2
        self.bridge = nn.Sequential(nn.BatchNorm2d(current_channels // 2), nn.ReLU(),
                                    nn.Conv2d(current_channels // 2, current_channels, kernel_size=(3, 3), stride=2,
                                              padding=1), nn.BatchNorm2d(current_channels), nn.ReLU(),
                                    nn.Conv2d(current_channels, current_channels, kernel_size=(3, 3), padding='same'))
        self.upproject4 = nn.Sequential(
            nn.Conv2d(current_channels // 2, current_channels, kernel_size=(1, 1), stride=2),
            nn.BatchNorm2d(current_channels))
        current_channels //= 2
        self.decoding1 = nn.Sequential(nn.BatchNorm2d(current_channels * 2 + current_channels), nn.ReLU(),
                                       nn.Conv2d(current_channels * 2 + current_channels, current_channels,
                                                 kernel_size=(3, 3), padding='same'), nn.BatchNorm2d(current_channels),
                                       nn.ReLU(), nn.Conv2d(current_channels, current_channels, kernel_size=(3, 3),
                                                            padding='same'))
        self.downproject1 = nn.Sequential(
            nn.Conv2d(current_channels * 2 + current_channels, current_channels, kernel_size=(1, 1)),
            nn.BatchNorm2d(current_channels))
        current_channels //= 2
        self.decoding2 = nn.Sequential(nn.BatchNorm2d(current_channels * 2 + current_channels), nn.ReLU(),
                                       nn.Conv2d(current_channels * 2 + current_channels, current_channels,
                                                 kernel_size=(3, 3), padding='same'), nn.BatchNorm2d(current_channels),
                                       nn.ReLU(), nn.Conv2d(current_channels, current_channels, kernel_size=(3, 3),
                                                            padding='same'))
        self.downproject2 = nn.Sequential(
            nn.Conv2d(current_channels * 2 + current_channels, current_channels, kernel_size=(1, 1)),
            nn.BatchNorm2d(current_channels))
        current_channels //= 2
        self.decoding3 = nn.Sequential(nn.BatchNorm2d(current_channels * 2 + current_channels), nn.ReLU(),
                                       nn.Conv2d(current_channels * 2 + current_channels, current_channels,
                                                 kernel_size=(3, 3), padding='same'), nn.BatchNorm2d(current_channels),
                                       nn.ReLU(), nn.Conv2d(current_channels, current_channels, kernel_size=(3, 3),
                                                            padding='same'))
        self.downproject3 = nn.Sequential(
            nn.Conv2d(current_channels * 2 + current_channels, current_channels, kernel_size=(1, 1)),
            nn.BatchNorm2d(current_channels))
        self.output = nn.Conv2d(base_channels, out_length, kernel_size=(1, 1))

    def forward(self, x):
        x1 = self.encoding1(x) + self.upproject1(x)
        x2 = self.encoding2(x1) + self.upproject2(x1)
        x3 = self.encoding3(x2) + self.upproject3(x2)
        x = self.bridge(x3) + self.upproject4(x3)
        # Possibly Replace Upsample (and downproject) with ConvTranspose with Kernel and Stride 2
        x = nn.Upsample(scale_factor=2, mode='bilinear')(x)
        x = torch.cat((x3, x), dim=1)
        x = self.decoding1(x) + self.downproject1(x)
        x = nn.Upsample(scale_factor=2, mode='bilinear')(x)
        x = torch.cat((x2, x), dim=1)
        x = self.decoding2(x) + self.downproject2(x)
        x = nn.Upsample(scale_factor=2, mode='bilinear')(x)
        x = torch.cat((x1, x), dim=1)
        x = self.decoding3(x) + self.downproject3(x)
        x = self.output(x)
        return x
class AtrousDenseUDeconvNet(nn.Module):
    def __init__(self, inChannels, outChannels, dropoutChance=0.1):
        super().__init__()
        #TODO: test padding modes
        self.l2 = nn.BatchNorm2d(inChannels)
        self.l3 = nn.Sequential(nn.Conv2d(inChannels, 16, kernel_size=(3,3), padding='same'), nn.ReLU())
        self.l4 = nn.Sequential(nn.Conv2d(16, 8, kernel_size=(3,3), padding='same', dilation=2), nn.ReLU())
        self.l5 = nn.Sequential(nn.Conv2d(8, 16, kernel_size=(3,3), padding='same', dilation=5), nn.ReLU())
        self.l6 = nn.Sequential(nn.Conv2d(16, 32, kernel_size=(3,3), padding='same', dilation=8), nn.ReLU())
        self.l8To11 = nn.Sequential(nn.Conv2d(72, 64, kernel_size=(3,3), padding='same'), nn.ReLU(),
                                    nn.Dropout2d(dropoutChance),
                                    nn.AvgPool2d(kernel_size=(2,2)),
                                    nn.BatchNorm2d(64, track_running_stats=False))
        self.l12 = nn.Sequential(nn.Conv2d(64, 32, kernel_size=(3,3), padding='same', dilation=2), nn.ReLU())
        self.l13 = nn.Sequential(nn.Conv2d(32, 64, kernel_size=(3,3), padding='same', dilation=5), nn.ReLU())
        self.l14 = nn.Sequential(nn.Conv2d(64, 128, kernel_size=(3,3), padding='same', dilation=8), nn.ReLU())
        self.l16To21 = nn.Sequential(nn.Conv2d(288, 256, kernel_size=(3,3), padding='same'), nn.ReLU(),
                                 nn.Dropout2d(dropoutChance),
                                 nn.AvgPool2d(kernel_size=(2,2)),
                                 nn.BatchNorm2d(256, track_running_stats=False),
                                 nn.Conv2d(256, 128, kernel_size=(3,3), padding='same'), nn.ReLU(),
                                 nn.Dropout2d(dropoutChance))
        self.l22To24 = nn.Sequential(nn.MaxPool2d(kernel_size=(2,2)),
                                 nn.Conv2d(128, 256, kernel_size=(3,3), padding='same'), nn.ReLU(),
                                 nn.Dropout2d(dropoutChance))
        self.l25To27 = nn.Sequential(nn.MaxPool2d(kernel_size=(2,2)),
                                 nn.Conv2d(256, 512, kernel_size=(3,3), padding='same'), nn.ReLU(),
                                 nn.Dropout2d(dropoutChance))
        self.l28To32 = nn.Sequential(nn.MaxPool2d(kernel_size=(2,2)),
                                 nn.Conv2d(512, 1024, kernel_size=(3,3), padding='same'), nn.ReLU(),
                                 nn.Dropout2d(dropoutChance),
                                 nn.Upsample(scale_factor=2.0, mode='bilinear'),
                                 nn.Conv2d(1024, 512, kernel_size=(3,3), padding='same'), nn.ReLU())
        self.l34To36 = nn.Sequential(nn.Conv2d(1024, 512, kernel_size=(3,3), padding='same'), nn.ReLU(),
                                 nn.Upsample(scale_factor=2.0, mode='bilinear'),
                                 nn.Conv2d(512, 256, kernel_size=(3,3), padding='same'), nn.ReLU())
        self.l38To40 = nn.Sequential(nn.Conv2d(512, 256, kernel_size=(3,3), padding='same'), nn.ReLU(),
                                 nn.Upsample(scale_factor=2.0, mode='bilinear'),
                                 nn.Conv2d(256, 128, kernel_size=(3,3), padding='same'), nn.ReLU())
        self.l42To46 = nn.Sequential(nn.Conv2d(256, 128, kernel_size=(3,3), padding='same'), nn.ReLU(),
                                     nn.ConvTranspose2d(128, 64, kernel_size=(3,3), stride=2, padding=1, output_padding=1), nn.ReLU(),
                                     nn.ConvTranspose2d(64, 64, kernel_size=(3,3), padding=1), nn.ReLU(),
                                     nn.ConvTranspose2d(64, 18, kernel_size=(3,3), stride=2, padding=1, output_padding=1), nn.ReLU(),
                                     nn.ConvTranspose2d(18, outChannels, kernel_size=(3,3), padding=1))
    def forward(self, x):
        x = self.l2(x)
        x1 = self.l3(x)
        x2 = self.l4(x1)
        x3 = self.l5(x2)
        x4 = self.l6(x3)
        x = torch.cat((x1, x2, x3, x4), dim=1)
        x1 = self.l8To11(x)
        x2 = self.l12(x1)
        x3 = self.l13(x2)
        x4 = self.l14(x3)
        x = torch.cat((x1, x2, x3, x4), dim=1)
        x1 = self.l16To21(x)
        x2 = self.l22To24(x1)
        x3 = self.l25To27(x2)
        x4 = self.l28To32(x3)
        x = torch.cat((x3, x4), dim=1)
        x = self.l34To36(x)
        x = torch.cat((x2, x), dim=1)
        x = self.l38To40(x)
        x = torch.cat((x1, x), dim=1)
        x = self.l42To46(x)
        return x
model = DeepResidualUNet(inputChannels, len(colorDict.keys()))
startEpoch = 50
baseName = 'DeepResidualUNet_Sigmoid_FullData_'
checkpoint_dir = Path(f"./checkpoints/{baseName}/")

if startEpoch != 0 and (fileList := os.listdir(checkpoint_dir)):
    checkpoint = None
    fileList.sort()
    stateDict = torch.load(checkpoint_dir / fileList[-1], weights_only=True)
    removedKeys = ["l8To11.4.running_mean", "l8To11.4.running_var", "l8To11.4.num_batches_tracked", "l16To21.4.running_mean", "l16To21.4.running_var", "l16To21.4.num_batches_tracked"]
    for key in removedKeys:
        if key in stateDict:
            del stateDict[key]
    model.load_state_dict(stateDict)
else:
    to_save = {'model': model}

    checkpoint = Checkpoint(
        to_save,
        checkpoint_dir,
        n_saved=5,
        filename_prefix=baseName,
        score_name='miou',
        greater_or_equal=True,
    )
model = model.to(device)

"""# 4. Model Training"""

lossFunction = nn.BCEWithLogitsLoss()
optimizer = torch.optim.AdamW(params=model.parameters(), lr=LEARNING_RATE)


def train_step(engine, batch):
    img = batch['img']
    labelOHE, label = batch['label']
    img = img.to(device)
    labelOHE = labelOHE.to(device)
    label = label.to(device).to(torch.int64)
    optimizer.zero_grad()
    y_pred = model(img).squeeze(1)
    loss = lossFunction(y_pred, labelOHE)
    loss.backward()
    optimizer.step()
    return {'loss': loss.item(), 'y_pred': y_pred, 'label': label}

def evaluation_step(engine, batch):
    img = batch['img']
    labelOHE, label = batch['label']
    img = img.to(device)
    labelOHE = labelOHE.to(device)
    label = label.to(device).to(torch.int64)
    y_pred = model(img).squeeze(1)
    loss = lossFunction(y_pred, labelOHE)
    return {'loss': loss.item(), 'y_pred': y_pred, 'label': label}


class ExpectedCalibrationError(Metric):
    def __init__(self, ignoredClass, numBins: torch.Tensor, outputTransform=lambda x: x, device="cpu"):
        self.ignoredClass = ignoredClass
        self.numBins = numBins
        self._num_correct = torch.zeros(numBins)
        self._num_examples = torch.zeros(numBins)
        self._accumulated_confidence = torch.zeros(numBins)
        super(ExpectedCalibrationError, self).__init__(output_transform=outputTransform, device=device)

    @reinit__is_reduced
    def reset(self):
        self._num_correct = torch.zeros(self.numBins)
        self._num_examples = torch.zeros(self.numBins)
        self._accumulated_confidence = torch.zeros(self.numBins)
        super(ExpectedCalibrationError, self).reset()

    @reinit__is_reduced
    def update(self, output):
        y_pred, y = output[0].detach().reshape(output[0].shape[1], -1), output[1].detach().reshape(-1, 1)

        confidences = torch.max(torch.sigmoid(y_pred), dim=0)
        indices = confidences.indices.reshape(-1, 1)
        mask = (y != self.ignoredClass)
        mask &= (indices != self.ignoredClass)
        y_filtered = y[mask]
        indices = indices[mask]
        correct = torch.eq(indices, y_filtered).reshape(-1)
        confidences = confidences.values.reshape(-1, 1)[mask].reshape(-1)
        for bin in range(self.numBins):
            mask = torch.logical_and((bin / self.numBins) < confidences, confidences <= ((bin + 1) / self.numBins))

            self._num_correct[bin] += torch.sum(correct[mask]).to(self._device)
            self._num_examples[bin] += torch.sum(mask).to(self._device)
            self._accumulated_confidence[bin] += torch.sum(confidences[mask]).to(self._device)

    @sync_all_reduce("_num_examples:SUM", "_num_correct:SUM", "_accumulated_confidence:SUM")
    def compute(self):
        if torch.sum(self._num_examples) == 0:
            raise NotComputableError(
                "ExpectedCalibrationError must have at least one example before it can be computed")
        ECE = 0
        for bin in range(self.numBins):
            if self._num_examples[bin] == 0:
                continue
            ECE += (self._num_examples[bin] / torch.sum(self._num_examples)) * torch.abs(
                (self._num_correct[bin] / self._num_examples[bin]) - (
                            self._accumulated_confidence[bin] / self._num_examples[bin]))
        return ECE


class StaticCalibrationError(Metric):
    def __init__(self, ignoredClass: int | None, numBins: int, numClasses: int, outputTransform=lambda x: x, device:str="cpu"):
        self.ignoredClass = ignoredClass
        self.numBins = numBins
        self.numClasses = numClasses
        self._num_correct = torch.zeros(self.numBins, self.numClasses)
        self._num_examples = torch.zeros(self.numBins, self.numClasses)
        self._accumulated_confidence = torch.zeros(self.numBins, self.numClasses)
        super(StaticCalibrationError, self).__init__(output_transform=outputTransform, device=device)

    @reinit__is_reduced
    def reset(self):
        self._num_correct = torch.zeros(self.numBins, self.numClasses)
        self._num_examples = torch.zeros(self.numBins, self.numClasses)
        self._accumulated_confidence = torch.zeros(self.numBins, self.numClasses)
        super(StaticCalibrationError, self).reset()

    @reinit__is_reduced
    def update(self, output):
        y_pred, y = output[0].detach().reshape(self.numClasses, -1), output[1].detach().reshape(-1, 1)

        indices = torch.argmax(y_pred, dim=0).reshape(-1, 1)

        mask = (y != self.ignoredClass)
        mask &= (indices != self.ignoredClass)
        y = y[mask]
        y_pred = y_pred[mask.repeat(1, self.numClasses)].reshape(-1, self.numClasses)

        for classIndex in range(self.numClasses):
            if classIndex == self.ignoredClass:
                continue
            correct = torch.eq(y, classIndex).reshape(-1)
            for bin in range(self.numBins):
                mask = torch.logical_and((bin / self.numBins) < y_pred[:, classIndex],
                                         y_pred[:, classIndex] <= ((bin + 1) / self.numBins))
                self._num_correct[bin, classIndex] += torch.sum(correct[mask]).to(self._device)
                self._num_examples[bin, classIndex] += torch.sum(mask).to(self._device)
                self._accumulated_confidence[bin, classIndex] += torch.sum(y_pred[:, classIndex][mask]).to(self._device)

    @sync_all_reduce("_num_examples:SUM", "_num_correct:SUM", "_accumulated_confidence:SUM")
    def compute(self):
        if torch.sum(self._num_examples) == 0:
            raise NotComputableError("StaticCalibrationError must have at least one example before it can be computed")

        SCE = 0

        for classIndex in range(self.numClasses):
            if classIndex == self.ignoredClass:
                continue
            for bin in range(self.numBins):
                if self._num_examples[bin, classIndex] == 0:
                    continue
                SCE += (self._num_examples[bin, classIndex] / torch.sum(self._num_examples[:, classIndex])) * torch.abs(
                    (self._num_correct[bin, classIndex] / self._num_examples[bin, classIndex]) - (
                                self._accumulated_confidence[bin, classIndex] / self._num_examples[bin, classIndex]))

        if self.ignoredClass is not None and isinstance(self.ignoredClass, int) and 0 <= self.ignoredClass < self.numClasses:
            SCE = (1 / (self.numClasses - 1)) * SCE
        else:
            SCE = (1 / self.numClasses) * SCE
        return SCE


class AdaptiveExpectedCalibrationError(Metric):
    def __init__(self, numBins:int, outputTransform=lambda x: x, device="cpu"):
        self.numBins = numBins
        self._data = torch.empty((0, 2))
        super(AdaptiveExpectedCalibrationError, self).__init__(output_transform=outputTransform, device=device)

    @reinit__is_reduced
    def reset(self):
        self._data = torch.empty((0, 2))
        super(AdaptiveExpectedCalibrationError, self).reset()

    @reinit__is_reduced
    def update(self, output):
        #TODO: Check if y_pred shape is correct and torch max dim is correct
        y_pred, y = output[0].detach(), output[1].detach()

        confidences = torch.max(y_pred, dim=1)
        indices = confidences.indices
        confidences = confidences.values.reshape(-1).to(self._device)
        correct = torch.eq(indices, y).reshape(-1).to(self._device)
        self._data = torch.cat((torch.stack((confidences, correct), dim=1), self._data), dim=0)

    def compute(self):
        if self._data.shape[0] == 0:
            raise NotComputableError(
                "AdaptiveExpectedCalibrationError must have at least one example before it can be computed")
        sortedData = self._data[:, 0].clone().sort().values
        boundaries = torch.quantile(sortedData, torch.arange(self.numBins + 1) / self.numBins, interpolation='midpoint')
        AECE = 0
        for bin in range(self.numBins):
            mask = torch.logical_and(boundaries[bin] <= self._data[:, 0], self._data[:, 0] <= boundaries[bin+1])
            AECE += torch.abs((torch.sum(self._data[:, 1][mask]) / torch.sum(mask)) - (
                        torch.sum(self._data[:, 0][mask]) / torch.sum(mask)))
        AECE *= 1 / self.numBins
        return AECE


class AdaptiveStaticCalibrationError(Metric):
    def __init__(self, numBins:int, numClasses:int, outputTransform=lambda x: x, device:str="cpu"):
        self.numBins = numBins
        self.numClasses = numClasses
        self._data = torch.empty((0, self.numClasses + 1))
        super(AdaptiveStaticCalibrationError, self).__init__(output_transform=outputTransform, device=device)

    @reinit__is_reduced
    def reset(self):
        self._data = torch.empty((0, self.numClasses + 1))
        super(AdaptiveStaticCalibrationError, self).reset()

    @reinit__is_reduced
    def update(self, output):
        # TODO: Check if y_pred shape is correct and torch argmax dim is correct
        y_pred, y = output[0].detach(), output[1].detach()

        indices = torch.argmax(y_pred, dim=1)
        y_pred = y_pred.reshape(-1, self.numClasses).to(self._device)
        correct = torch.eq(indices, y).reshape(-1, 1).to(self._device)
        self._data = torch.cat((torch.cat((y_pred, correct), dim=1), self._data), dim=0)

    def compute(self):
        if self._data.numel == 0:
            raise NotComputableError(
                "AdaptiveStaticCalibrationError must have at least one example before it can be computed")
        ASCE = 0
        for classIndex in range(self.numClasses):
            sortedData = self._data[:, classIndex].clone().sort().values
            boundaries = torch.quantile(sortedData, torch.arange(self.numBins + 1) / self.numBins, interpolation='midpoint')
            for bin in range(self.numBins):
                mask = torch.logical_and(boundaries[bin] <= self._data[:, classIndex], self._data[:, classIndex] <= boundaries[bin+1])
                ASCE += torch.abs((torch.sum(self._data[:, -1][mask]) / torch.sum(mask)) - (
                            torch.sum(self._data[:, classIndex][mask]) / torch.sum(mask)))
        ASCE *= 1 / (self.numClasses * self.numBins)
        return ASCE


def output_transform(output):
    y_pred = output['y_pred']
    y = output['label']
    return y_pred, y


NUM_BINS = torch.tensor(5)
trainer = Engine(train_step)
evaluator = Engine(evaluation_step)
cm = ConfusionMatrix(num_classes=len(colorDict.keys()), output_transform=output_transform)
metric = mIoU(cm, ignore_index=indexToIgnore)
metric.attach(evaluator, 'miou')
ECE = ExpectedCalibrationError(indexToIgnore, NUM_BINS, output_transform)
ECE.attach(evaluator, "ECE")
ACC = Accuracy(output_transform)
ACC.attach(evaluator, 'acc')


# SCE = StaticCalibrationError(indexToIgnore, NUM_BINS, torch.tensor(len(colorDict.keys())), output_transform)
# SCE.attach(evaluator, "SCE")
# AECE = AdaptiveExpectedCalibrationError(NUM_BINS, output_transform)
# AECE.attach(evaluator, "AECE")
# ASCE = AdaptiveStaticCalibrationError(NUM_BINS, torch.tensor(len(colorDict.keys())), output_transform)
# ASCE.attach(evaluator, "ASCE")
@trainer.on(Events.EPOCH_COMPLETED)
def trainer_on_epoch_completed(engine):
    epoch = engine.state.epoch
    loss = engine.state.output['loss']
    print(f'Epoch {epoch}: train_loss = {loss}', end=" ")
    model.eval()
    with torch.no_grad():
        evaluator.run(valLoader, max_epochs=1)



@evaluator.on(Events.EPOCH_COMPLETED)
def evaluator_on_epoch_completed(engine):
    loss = engine.state.output['loss']
    miou = engine.state.metrics['miou']
    ece = engine.state.metrics['ECE']
    # sce = engine.state.metrics['SCE']
    # aece = engine.state.metrics['AECE']
    # asce = engine.state.metrics['ASCE']
    acc = engine.state.metrics['acc']
    print(f'Validation: val_loss = {loss}, mIoU = {miou}, ECE = {ece}, ACC = {acc}'
          # f', SCE = {sce}, AECE = {aece}, ASCE = {asce}'
          )

# if checkpoint is not None:
#     evaluator.add_event_handler(Events.EPOCH_COMPLETED, checkpoint)
# ProgressBar().attach(trainer)
# trainer.run(trainLoader, max_epochs=EPOCHS - startEpoch)
ProgressBar().attach(evaluator)
model.eval()
evaluator.run(valLoader, max_epochs=1)

# from calibrate import calibrate
# from calibrationLoss import *
# temperature = calibrate(model, valLoader, indexToIgnore, NUM_BINS, 6, ExpectedCalibrationErrorLoss(indexToIgnore, NUM_BINS, device), device, epochs=10)
# temperature.requires_grad_(False)
#
# from plot import plotReliabilityDiagram
# plotReliabilityDiagram(model, testLoader, indexToIgnore, NUM_BINS, 6, baseName, device, temperature=temperature)
# from plot import plotUncertainty
# plotUncertainty(model, testLoader, Path('./classOutput/'), device)