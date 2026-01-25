import os

import rasterio
from ignite.engine import Engine
from ignite.handlers import ProgressBar, Checkpoint
from ignite.metrics import ConfusionMatrix
from ignite.metrics import mIoU
from ignite.metrics import Accuracy
from ignite.engine import Events
from PIL import Image
import torch
import torch.utils.data
from torch.utils.data import Dataset, DataLoader
from torchvision.transforms import v2
from torch import nn
import numpy as np
import json
from pathlib import Path
from metrics import *
from models import *
# 2. Dataset Setup
def loadDatasetDetails(datasetIndex):
    datasetsDirectoryPath = Path('./datasets')
    datasets = [f.name for f in os.scandir(datasetsDirectoryPath) if f.is_dir()]
    dataset = datasetsDirectoryPath / (datasets[datasetIndex])
    with open(dataset / 'dataset.json') as f:
        datasetDetails = json.load(f)
    colorDict = {}
    for i, color in enumerate(datasetDetails['LabelMapping']):
        colorDict[torch.tensor([color['R'], color['G'], color['B']])] = i
    return dataset, datasetDetails, colorDict
def transformLabel(label: torch.Tensor, colorDictionary: dict[torch.Tensor, int], numberOfLabelChannels: int, targetHeight: int, targetWidth: int):
    label = label.to(torch.int64)
    label2D = label.reshape(numberOfLabelChannels, -1).transpose(0, 1)
    indexMap = label2D.reshape(-1)
    if numberOfLabelChannels > 1:
        indexMap = torch.zeros(label2D.shape[0])
        for color, index in colorDictionary.items():
            indexMap += torch.where(torch.all(label2D == color, dim=1), index, 0)
        indexMap = indexMap.to(torch.int64)
    labelOHE = nn.functional.one_hot(indexMap, num_classes=len(colorDictionary.keys())).transpose(0, 1).reshape(
        len(colorDictionary.keys()), targetHeight, targetWidth).to(torch.float16)
    return labelOHE, indexMap.reshape(targetHeight, targetWidth)


class EarthObservationImageryDataset(Dataset):
    def __init__(self, listOfPaths, numberOfLabelChannels, targetHeight, targetWidth, colorDict):
        self.listOfPaths = listOfPaths
        self.numberOfLabelChannels = numberOfLabelChannels
        self.targetHeight = targetHeight
        self.targetWidth = targetWidth
        self.colorDict = colorDict
    def __getitem__(self, idx):
        img = Image.open(self.listOfPaths[idx])
        labelSrc = rasterio.open(str(self.listOfPaths[idx]).replace('Images', 'Labels').replace('rgb', 'label'), mode='r')
        label = labelSrc.read(list(range(1, self.numberOfLabelChannels+1))).astype(np.uint8)
        imgTransformed = v2.ToDtype(torch.float32, scale=True)(v2.ToImage()(img))
        labelTransformed = transformLabel(torch.tensor(label), self.colorDict, self.numberOfLabelChannels, self.targetHeight, self.targetWidth)
        return {'img': imgTransformed, 'label': labelTransformed}

    def __len__(self):
        return len(self.listOfPaths)

def loadSubDataset(basePath: Path):
    dataset = []
    pathGenerator = (basePath / 'Images').rglob('*.tif')
    for path in pathGenerator:
        dataset.append(path)
    return dataset

def loadDataLoaders(dataset, datasetDetails, colorDict, trainBatchSize, valBatchSize, testBatchSize):
    trainDataset = loadSubDataset(Path(dataset / 'train'))
    valDataset = loadSubDataset(Path(dataset / 'val'))
    testDataset = loadSubDataset(Path(dataset / 'test'))
    trainDataset = EarthObservationImageryDataset(trainDataset, datasetDetails['NumberOfLabelChannels'], datasetDetails['targetHeightTrain'], datasetDetails['targetWidthTrain'], colorDict)
    valDataset = EarthObservationImageryDataset(valDataset, datasetDetails['NumberOfLabelChannels'], datasetDetails['targetHeightVal'], datasetDetails['targetWidthVal'], colorDict)
    testDataset = EarthObservationImageryDataset(testDataset, datasetDetails['NumberOfLabelChannels'], datasetDetails['targetHeightTest'], datasetDetails['targetWidthTest'], colorDict)
    # inputChannels = datasetDetails['NumberOfInputChannels']
    trainLoader = DataLoader(trainDataset, batch_size=trainBatchSize, shuffle=True, num_workers=0, drop_last=True)
    valLoader = DataLoader(valDataset, batch_size=valBatchSize, shuffle=False, num_workers=0, drop_last=True)
    testLoader = DataLoader(testDataset, batch_size=testBatchSize, shuffle=False, num_workers=0, drop_last=True)
    return trainLoader, valLoader, testLoader


"""# 3. Model Setup"""
def loadModel(model, softMax, checkpoint_dir, epochs, device, trainer):
    softMax = bool(softMax)
    fileList = list(checkpoint_dir.rglob('*.pt'))
    startEpoch = len(fileList)
    train = startEpoch != epochs
    checkpoint = None
    if fileList:
        fileList.sort()
        stateDict = torch.load(fileList[-1], weights_only=True)
        model.load_state_dict(stateDict)
    if train:
        checkpoint_dir.mkdir(exist_ok=True, parents=True)
        subDirectories = [f.name for f in os.scandir(checkpoint_dir) if f.is_dir()]
        subDirectories.sort()
        subDirIndex = '0'
        if subDirectories:
            subDirIndex = str(int(subDirectories[-1]) + 1)
        to_save = {'model': model}
        globalStepTransform = lambda *_: trainer.state.epoch + startEpoch
        checkpoint = Checkpoint(
            to_save,
            checkpoint_dir / subDirIndex,
            n_saved=epochs,
            global_step_transform=globalStepTransform
        )
    model = model.to(device)
    return model, checkpoint, train
"""# 4. Model Training"""

def modelOutputFilter(y_pred, label, indexToIgnore):
    mask = label != indexToIgnore
    batchSize = y_pred.shape[0]
    numClasses = y_pred.shape[1]
    height = y_pred.shape[2]
    width = y_pred.shape[3]
    label = label[mask].reshape(batchSize, -1)
    y_pred = y_pred[mask.repeat(numClasses, 1, 1).reshape(1, numClasses, height, width)].reshape(
        batchSize, numClasses, -1)
    return y_pred, label

def output_transform(output):
    y_pred = output['y_pred']
    y = output['label']
    return y_pred, y

def run(softMax, modelIndex, datasetIndex):
    dataset, datasetDetails, colorDict = loadDatasetDetails(datasetIndex)
    NUM_CLASSES = len(colorDict.keys())
    TRAIN_BATCH_SIZE = 16
    VAL_BATCH_SIZE = 1
    TEST_BATCH_SIZE = 1
    LEARNING_RATE = 1e-04
    EPOCHS = 20
    NUM_BINS = torch.tensor(15)
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    trainDataLoader, valDataLoader, testDataLoader = loadDataLoaders(dataset, datasetDetails, colorDict, TRAIN_BATCH_SIZE, VAL_BATCH_SIZE, TEST_BATCH_SIZE)
    modelDict = {0: UNet,
                 1: DeepResidualUNet,
                 2: AtrousDenseUDeconvNet}
    def train_step(engine, batch):
        img = batch['img']
        labelOHE, label = batch['label']
        img = img.to(device)
        labelOHE = labelOHE.to(device)
        label = label.to(device).to(torch.int64)
        optimizer.zero_grad()
        y_pred = model(img).squeeze(1)
        if softMax:
            loss = lossFunction(y_pred, label)
        else:
            loss = lossFunction(y_pred, labelOHE)
        loss.backward()
        optimizer.step()
        y_pred, label = modelOutputFilter(y_pred, label, datasetDetails['IndexToIgnore'])
        return {'loss': loss.item(), 'y_pred': y_pred, 'label': label}

    def evaluation_step(engine, batch):
        img = batch['img']
        labelOHE, label = batch['label']
        img = img.to(device)
        labelOHE = labelOHE.to(device)
        label = label.to(device).to(torch.int64)
        y_pred = model(img).squeeze(1)
        if softMax:
            loss = lossFunction(y_pred, label)
        else:
            loss = lossFunction(y_pred, labelOHE)
        y_pred, label = modelOutputFilter(y_pred, label, datasetDetails['IndexToIgnore'])
        return {'loss': loss.item(), 'y_pred': y_pred, 'label': label}
    trainer = Engine(train_step)
    if modelIndex in modelDict:
        model = modelDict[modelIndex](datasetDetails['NumberOfInputChannels'], NUM_CLASSES)
    else:
        exit()
    baseName = model.__class__.__name__ + '_' + ('Softmax' if softMax else 'Sigmoid') + dataset.name
    checkpoint_dir = Path(f"./checkpoints_wo_bg_batched/{baseName}/")
    model, checkpoint, TRAIN = loadModel(model, softMax, checkpoint_dir, EPOCHS, device, trainer)
    lossFunction = nn.CrossEntropyLoss(ignore_index=datasetDetails['IndexToIgnore']) if softMax else nn.BCEWithLogitsLoss()
    optimizer = torch.optim.AdamW(params=model.parameters(), lr=LEARNING_RATE)
    trainer = Engine(train_step)
    evaluator = Engine(evaluation_step)
    cm = ConfusionMatrix(num_classes=NUM_CLASSES, output_transform=output_transform)
    cm.attach(evaluator, 'cm')
    metric = mIoU(cm, ignore_index=datasetDetails['IndexToIgnore'])
    metric.attach(evaluator, 'miou')
    ECE = ExpectedCalibrationError(datasetDetails['IndexToIgnore'], NUM_BINS, softMax, output_transform)
    ECE.attach(evaluator, "ECE")
    ACC = Accuracy(output_transform)
    ACC.attach(evaluator, 'acc')
    CACC = ClassWiseAccuracy(len(colorDict.keys()), device, output_transform)
    CACC.attach(evaluator, 'cacc')
    # SCE = StaticCalibrationError(datasetDetails['IndexToIgnore'], NUM_BINS, torch.tensor(NUM_CLASSES), output_transform)
    # SCE.attach(evaluator, "SCE")
    # AECE = AdaptiveExpectedCalibrationError(NUM_BINS, output_transform)
    # AECE.attach(evaluator, "AECE")
    # ASCE = AdaptiveStaticCalibrationError(NUM_BINS, torch.tensor(NUM_CLASSES), output_transform)
    # ASCE.attach(evaluator, "ASCE")
    @trainer.on(Events.EPOCH_COMPLETED)
    def trainer_on_epoch_completed(engine):
        epoch = engine.state.epoch
        loss = engine.state.output['loss']
        print(f'Epoch {epoch}: train_loss = {loss}', end=" ")
        model.eval()
        with torch.no_grad():
            evaluator.run(valDataLoader, max_epochs=1)

    @evaluator.on(Events.EPOCH_COMPLETED)
    def evaluator_on_epoch_completed(engine):
        loss = engine.state.output['loss']
        miou = engine.state.metrics['miou']
        ece = engine.state.metrics['ECE']
        cm = engine.state.metrics['cm']
        # sce = engine.state.metrics['SCE']
        # aece = engine.state.metrics['AECE']
        # asce = engine.state.metrics['ASCE']
        acc = engine.state.metrics['acc']
        cacc = engine.state.metrics['cacc']
        targetFileName = 'results.txt' if TRAIN else 'metrics.txt'
        # print(f'Validation: val_loss = {loss}, mIoU = {miou}, ECE = {ece}, ACC = {acc}\n')
        with open(checkpoint_dir / targetFileName, 'a') as results:
            results.write(
                f'Validation: val_loss = {loss}, mIoU = {miou}, ECE = {ece}, ACC = {acc}\n CACC = {cacc}\n cm={cm}'
                # f', SCE = {sce}, AECE = {aece}, ASCE = {asce}, \'
                )

    if TRAIN and checkpoint is not None:
        evaluator.add_event_handler(Events.EPOCH_COMPLETED, checkpoint)
    # ProgressBar().attach(trainer)
    # trainer.run(trainLoader, max_epochs=EPOCHS - startEpoch)
    ProgressBar().attach(evaluator)
    model.eval()
    evaluator.run(testDataLoader, max_epochs=1)
    #
    # from calibrate import calibrate
    # from calibrationLoss import *

    # temperature = calibrate(model, valLoader, indexToIgnore, NUM_BINS, NUM_CLASSES, ExpectedCalibrationErrorLoss(indexToIgnore, NUM_BINS, device, softMax), device, softMax, args.pos_datasetIndex, args.pos_calibrationMode, epochs=50)
    # print(temperature)
    # temperature.requires_grad_(False)
    # if softMax:
    #     temperature = torch.tensor(1.0, device=device)
    # else:
    #     temperature = torch.ones(NUM_CLASSES, device=device)
    # from plot import plotReliabilityDiagram
    # plotReliabilityDiagram(model, testLoader, indexToIgnore, NUM_BINS, NUM_CLASSES, checkpoint_dir, device, softMax, temperature = temperature)
    # from plot import plotRandomImageClassOutput
    # plotRandomImageClassOutput(model, testLoader, checkpoint_dir, device, colorDict, targetHeightTest, targetWidthTest, NUM_CLASSES)
    # from plot import plotUncertainty
    # plotUncertainty(model, testLoader, checkpoint_dir, device, softMax=softMax)
    # from plot import plotLearningGraph
    # plotLearningGraph(checkpoint_dir, "ACC")
    # plotLearningGraph(checkpoint_dir, "mIoU")