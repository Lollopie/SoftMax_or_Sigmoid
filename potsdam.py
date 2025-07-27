from ignite.engine import Engine
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
from torch import nn
import numpy as np

from pathlib import Path

p = Path('/content/drive/MyDrive/') if Path('/content/drive/MyDrive/').exists() else Path('.')

"""# 1. Image Preprocessing"""

labelPath = p / "top_potsdam_7_9_label.tif"
imgPath = p / "top_potsdam_7_9_RGB.tif"
valPath = p / "top_potsdam_7_8_RGB.tif"
valLabelPath = p / "top_potsdam_7_8_label.tif"
img = Image.open(imgPath)
label = Image.open(labelPath)
valImg = Image.open(imgPath)
valLabel = Image.open(labelPath)

transforms = v2.Compose(
    [
        v2.ToImage(),
    ]
)

imgTransformed, labelTransformed = transforms((img, label))
del img
del label
valImgTransformed, valLabelTransformed = transforms((valImg, valLabel))
del valImg
del valLabel

colorDict = {}
i = 0
label2D = labelTransformed.reshape(3, -1).transpose(0, 1)
for color in label2D.unique(dim=0):
    colorDict[color] = i
    i += 1

"""#TODO: Properly set ignore index

# 2. Dataset Setup
"""


def labelTransformation(label, colorDictionary):
    label = label.to(torch.int64)
    label2D = label.reshape(3, -1).transpose(0, 1)
    indexMap = torch.zeros(label2D.shape[0])
    for color, index in colorDictionary.items():
        indexMap += torch.where(torch.all(label2D == color, dim=1), index, 0)
    indexMap = indexMap.to(torch.int64)
    labelOHE = nn.functional.one_hot(indexMap, num_classes=len(colorDictionary.keys())).transpose(0, 1).reshape(
        len(colorDictionary.keys()), targetHeight, targetWidth).to(torch.float16)
    return labelOHE, indexMap.reshape(targetHeight, targetWidth)


N = 200
valN = 200
targetHeight = 512
targetWidth = 512
datasetTransformation = v2.Compose(
    [
        v2.RandomCrop((targetHeight, targetWidth)),
        v2.RandomHorizontalFlip(0.5),
    ]
)
inputTransformation = v2.Compose(
    [
        v2.ToDtype(torch.float32, scale=True),
        v2.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ]
)


def generateDataset(N, img, label):
    dataset = []
    for i in range(N):
        sampleImg, sampleLabel = datasetTransformation(img, label)
        sampleImg = inputTransformation(sampleImg)
        dataObject = {}
        dataObject['img'] = sampleImg
        dataObject['label'] = labelTransformation(sampleLabel, colorDict)
        dataset.append(dataObject)
    return dataset


trainDataset = generateDataset(N, imgTransformed, labelTransformed)
valDataset = generateDataset(valN, valImgTransformed, valLabelTransformed)
inputChannels = imgTransformed.shape[0]
del imgTransformed
del labelTransformed
del valImgTransformed
del valLabelTransformed


class ImageDataset(Dataset):
    def __init__(self, dataset):
        self.data = dataset
        self.len = len(dataset)

    def __getitem__(self, idx):
        return self.data[idx]

    def __len__(self):
        return self.len


train_dataset = ImageDataset(trainDataset)
val_dataset = ImageDataset(valDataset)

TRAIN_BATCH_SIZE = 1
VAL_BATCH_SIZE = 1
TEST_BATCH_SIZE = 1
LEARNING_RATE = 1e-04
EPOCHS = 10
device = 'cuda' if torch.cuda.is_available() else 'cpu'

train_loader = DataLoader(train_dataset, batch_size=TRAIN_BATCH_SIZE, shuffle=True, num_workers=0, drop_last=True)
val_loader = DataLoader(valDataset, batch_size=VAL_BATCH_SIZE, shuffle=True, num_workers=0, drop_last=True)

"""# 3. Model Setup"""


class UNet(nn.Module):
    def __init__(self, image_size, min_size, input_channels, base_channels, out_length):
        super().__init__()
        self.contracting = nn.ModuleList()
        self.contracting.append(nn.ModuleList(
            nn.Sequential(nn.Conv2d(input_channels, base_channels, kernel_size=(3, 3), padding='same'), nn.ReLU(),
                          nn.Conv2d(base_channels, base_channels, kernel_size=(3, 3), padding='same'), nn.ReLU())))
        current_size = (image_size) / 2
        current_channels = base_channels
        layer_count = 1
        while current_size != min_size:
            current_channels *= 2
            self.contracting.append(nn.ModuleList(nn.Sequential(nn.MaxPool2d(kernel_size=(2, 2), stride=2),
                                                                nn.Conv2d(current_channels // 2, current_channels,
                                                                          kernel_size=(3, 3), padding='same'),
                                                                nn.ReLU(), nn.Conv2d(current_channels, current_channels,
                                                                                     kernel_size=(3, 3),
                                                                                     padding='same'), nn.ReLU())))
            current_size = (current_size) / 2
            layer_count += 1
        current_channels *= 2
        self.contracting.append(nn.ModuleList(nn.Sequential(nn.MaxPool2d(kernel_size=(2, 2), stride=2),
                                                            nn.Conv2d(current_channels // 2, current_channels,
                                                                      kernel_size=(3, 3), padding='same'), nn.ReLU(),
                                                            nn.Conv2d(current_channels, current_channels,
                                                                      kernel_size=(3, 3), padding='same'), nn.ReLU())))
        self.expanding = nn.ModuleList()
        self.expanding.append(nn.ModuleList(
            nn.Sequential(nn.Conv2d(current_channels, current_channels, kernel_size=(3, 3), padding='same'), nn.ReLU(),
                          nn.Conv2d(current_channels, current_channels, kernel_size=(3, 3), padding='same'),
                          nn.ReLU())))
        while layer_count > 0:
            current_channels //= 2
            self.expanding.append(nn.ModuleList(
                nn.Sequential(nn.Conv2d(current_channels * 2, current_channels, kernel_size=(2, 2), padding='same'),
                              nn.Conv2d(current_channels * 2, current_channels, kernel_size=(3, 3), padding='same'),
                              nn.ReLU(),
                              nn.Conv2d(current_channels, current_channels, kernel_size=(3, 3), padding='same'),
                              nn.ReLU())))
            layer_count -= 1
        # TODO: determine if output channels == number of classes
        self.expanding.append(nn.ModuleList(nn.Sequential(nn.Conv2d(current_channels, out_length, kernel_size=(1, 1)))))
        self.skip_list = []

    def forward(self, x):
        for model_list in self.contracting:
            for block in model_list:
                x = block(x)
            self.skip_list.append(x)
        self.skip_list.pop()
        for i, model_list in enumerate(self.expanding):
            if len(model_list) > 1 and i > 0:
                x = nn.Upsample(scale_factor=2, mode='bilinear')(x)
            x = model_list[0](x)
            if len(model_list) > 1 and i > 0:
                # crop_size = (self.skip_list[-i].shape[-1] - x.shape[-1]) // 2
                # TODO: check crop size for off by 1 errors
                x = torch.cat((self.skip_list[-i], x), dim=1)
            for block in model_list[1:]:
                x = block(x)
        self.skip_list = []
        return x


class DeepResidualUNet(nn.Module):
    def __init__(self, image_size, min_size, input_channels, base_channels, out_length):
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


model = DeepResidualUNet(512, 32, inputChannels, 64, len(colorDict.keys()))
model = model.to(device)

"""# 4. Model Training"""

lossFunction = nn.CrossEntropyLoss()
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


"""def train(model, epochs, lossFunction, learningRate, verbose = True, scheduler = None, optimizer = None):
    if optimizer is None:
        optimizer = torch.optim.AdamW(params = model.parameters(), lr = learningRate)
    losses = []
    for epoch in range(epochs):
        model.train()
        aggregatedLoss = 0
        stepsPerPrint = 5
        epochsPerValidation = 1
        for batch, data in enumerate(train_loader):
            img = data[:, 0]
            label = data[:, 1].to(torch.int64)
            img = img.to(device)
            label2DOHE = label2DOHE.to(device)
            output = model(img).squeeze(1)
            loss = lossFunction(output, label2DOHE)
            aggregatedLoss += loss.item()
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            if batch % stepsPerPrint == 0 and not batch == 0:
                if verbose:
                    print(f"Epoch: {epoch}, Batch: {batch // TRAIN_BATCH_SIZE}, Loss: {aggregatedLoss / (stepsPerPrint)}")
                aggregatedLoss = 0
        if epoch % epochsPerValidation == 0:
            model.eval()
            with torch.inference_mode():
                valAggregatedLoss = 0
                valAggregatedMeanIoU = 0
                steps = 0
                iouSteps = 0
                for batch, data in enumerate(val_loader):
                    img = data[:, 0]
                    label = data[:, 1].to(torch.int64)
                    label2D = label.reshape(3, -1).transpose(0,1)
                    indexMap = torch.zeros(label2D.shape[0])
                    for color, index in colorDict.items():
                        indexMap += torch.where(torch.all(label2D == color, dim=1), index, 0)
                    label2DOHE = nn.functional.one_hot(indexMap.to(torch.int64), num_classes=len(colorDict.keys())).transpose(0, 1).reshape(VAL_BATCH_SIZE, len(colorDict.keys()), targetHeight, targetWidth).to(torch.float16)
                    img = img.to(device)
                    label2DOHE = label2DOHE.to(device)
                    output = model(img).squeeze(1)
                    loss = lossFunction(output, label2DOHE)
                    valAggregatedLoss += loss.item()
                    for color, index in colorDict.items():
                        groundTruthMap = torch.all(label2D == color, dim=1).to(device)
                        predClassMap = torch.argmax(output, dim = 1).reshape(1, -1) == index
                        sumMap = groundTruthMap.to(torch.int) + predClassMap.to(torch.int)
                        andMap = predClassMap & groundTruthMap
                        truePositive = torch.sum(andMap)
                        falsePositive = torch.sum(sumMap - groundTruthMap.to(torch.int))
                        falseNegative = torch.sum(sumMap - predClassMap.to(torch.int))
                        if truePositive:
                            valAggregatedMeanIoU += truePositive / (truePositive + falsePositive + falseNegative)
                            iouSteps += 1
                    steps += 1
                print(f"Average Dev Loss: {valAggregatedLoss / steps}")
                print(f"Average IoU Loss: {valAggregatedMeanIoU / iouSteps}")
    return losses
    train(model, EPOCHS, lossFunction, LEARNING_RATE, optimizer = optimizer)

# 5. Evaluation Metrics
"""


class ExpectedCalibrationError(Metric):
    def __init__(self, ignoredClass, numBins, outputTransform=lambda x: x, device="cpu"):
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
        y_pred, y = output[0].detach(), output[1].detach()

        confidences = torch.max(y_pred, dim=1)
        indices = confidences.indices
        mask = (y != self.ignoredClass)
        mask &= (indices != self.ignoredClass)
        y_filtered = y[mask]
        indices = indices[mask]
        correct = torch.eq(indices, y_filtered).reshape(-1)
        confidences = confidences.values[mask]
        for bin in range(self.numBins):
            mask = torch.logical_and((bin / self.numBins) < confidences, confidences < ((bin + 1) / self.numBins))

            self._num_correct[bin] += torch.sum(correct[mask]).to(self._device)
            self._num_examples[bin] += torch.sum(mask).to(self._device)
            self._accumulated_confidence[bin] += torch.sum(confidences[mask]).to(self._device)

    @sync_all_reduce("_num_examples:SUM", "_num_correct:SUM", "_accumulated_confidence:SUM")
    def compute(self):
        if torch.any(self._num_examples == 0):
            raise NotComputableError(
                "ExpectedCalibrationError must have at least one example per bin before it can be computed")
        # Possibly set Error to 0 if no items in bin, and just require at least one sample
        ECE = 0
        for bin in range(self.numBins):
            ECE += (self._num_examples[bin] / torch.sum(self._num_examples)) * torch.abs(
                (self._num_correct[bin] / self._num_examples[bin]) - (
                            self._accumulated_confidence[bin] / self._num_examples[bin]))
        return ECE


class StaticCalibrationError(Metric):
    def __init__(self, ignoredClass, numBins, numClasses, outputTransform=lambda x: x, device="cpu"):
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
        y_pred, y = output[0].detach(), output[1].detach()

        indices = torch.argmax(y_pred, dim=1)

        mask = (y != self.ignoredClass)
        mask &= (indices != self.ignoredClass)
        y = y[mask]
        indices = indices[mask]
        y_pred = y_pred[mask.repeat(self.numClasses, 1, 1).unsqueeze(0)].reshape(self.numClasses, -1)

        for classIndex in range(self.numClasses):
            if classIndex == self.ignoredClass:
                continue
            correct = torch.eq(indices, classIndex).reshape(-1)
            for bin in range(self.numBins):
                mask = torch.logical_and((bin / self.numBins) < y_pred[classIndex, :],
                                         y_pred[classIndex, :] < ((bin + 1) / self.numBins))
                self._num_correct[bin, classIndex] += torch.sum(correct[mask]).to(self._device)
                self._num_examples[bin, classIndex] += torch.sum(mask).to(self._device)
                self._accumulated_confidence[bin, classIndex] += torch.sum(y_pred[classIndex, :][mask]).to(self._device)

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
                SCE += (self._num_examples[bin, classIndex] / torch.sum(self._num_examples)) * torch.abs(
                    (self._num_correct[bin, classIndex] / self._num_examples[bin, classIndex]) - (
                                self._accumulated_confidence[bin, classIndex] / self._num_examples[bin, classIndex]))

        if self.ignoredClass != -1:
            SCE = (1 / (self.numClasses - 1)) * SCE
        else:
            SCE = (1 / self.numClasses) * SCE
        return SCE


class AdaptiveExpectedCalibrationError(Metric):
    def __init__(self, numBins, outputTransform=lambda x: x, device="cpu"):
        self.numBins = numBins
        self._data = torch.empty((0, 2))
        super(AdaptiveExpectedCalibrationError, self).__init__(output_transform=outputTransform, device=device)

    @reinit__is_reduced
    def reset(self):
        self._data = torch.empty((0, 2))
        super(AdaptiveExpectedCalibrationError, self).reset()

    @reinit__is_reduced
    def update(self, output):
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
        boundaries = torch.zeros(self.numBins)
        for bin in range(self.numBins):
            boundaries[bin] = sortedData[torch.ceil(((bin + 1) / self.numBins) * sortedData.shape[0]).to(torch.int) - 1]
        indexTensor = torch.bucketize(sortedData, boundaries)
        AECE = 0
        for bin in range(self.numBins):
            mask = indexTensor == bin
            AECE += torch.abs((torch.sum(self._data[:, 1][mask]) / torch.sum(mask)) - (
                        torch.sum(self._data[:, 0][mask]) / torch.sum(mask)))
        AECE *= 1 / self.numBins
        return AECE


class AdaptiveStaticCalibrationError(Metric):
    def __init__(self, numBins, numClasses, outputTransform=lambda x: x, device="cpu"):
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
            boundaries = torch.zeros(self.numBins)
            for bin in range(self.numBins):
                boundaries[bin] = sortedData[
                    torch.ceil(((bin + 1) / self.numBins) * sortedData.shape[0]).to(torch.int) - 1]
            indexTensor = torch.bucketize(sortedData, boundaries)
            for bin in range(self.numBins):
                mask = indexTensor == bin
                ASCE += torch.abs((torch.sum(self._data[:, -1][mask]) / torch.sum(mask)) - (
                            torch.sum(self._data[:, classIndex][mask]) / torch.sum(mask)))
        ASCE *= 1 / (self.numClasses * self.numBins)
        return ASCE


def output_transform(output):
    y_pred = output['y_pred']
    y = output['label']
    return y_pred, y


NUM_BINS = torch.tensor(5)
evaluator = Engine(evaluation_step)
trainer = Engine(train_step)
cm = ConfusionMatrix(num_classes=len(colorDict.keys()), output_transform=output_transform)
indexToIgnore = -1
for color, index in colorDict.items():
    if torch.allclose(color, torch.tensor([255, 255, 255]).to(torch.uint8)):
        indexToIgnore = index
assert indexToIgnore != -1
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
    evaluator.run(val_loader, max_epochs=1)


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


trainer.run(train_loader, max_epochs=EPOCHS)

"""{'miou': 0.06197101828009659, 'ECE': 0.41888636350631714}"""


def calibrate(model: nn.Module, learningRate=1e-4, epochs=1):
    global optimizer
    model.eval()
    temperature = torch.tensor(1, requires_grad=True)
    temp = optimizer
    localOptimizer = torch.optim.AdamW(temperature, lr=learningRate)
    optimizer = localOptimizer
    engine = Engine(train_step)
    model.requires_grad_(False)
    engine.run(val_loader, max_epochs=epochs)

    # Reset Previous State
    model.requires_grad_(True)
    optimizer = temp
    return temperature
