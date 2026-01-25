import torch
import torchvision.transforms
from ignite.engine import Engine, Events
from matplotlib import pyplot as plt
import numpy as np
from mpl_toolkits.axes_grid1 import make_axes_locatable
from torch import nn
from pathlib import Path
import datetime
from tqdm import tqdm
class AccuracyUncertaintyPlot(nn.Module):
    def __init__(self, indexToIgnore, numBins, numClasses, savePath, temperatureString, device, softMax):
        #maybe add option for device
        super().__init__()
        self.indexToIgnore = indexToIgnore
        self.numBins = numBins
        self.numClasses = numClasses
        self.softMax = softMax
        self._device = device
        self._num_correct = torch.zeros(self.numBins).to(self._device)
        self._num_examples = torch.zeros(self.numBins).to(self._device)
        self._accumulated_confidence = torch.zeros(self.numBins).to(self._device)
        self.temperatureString = temperatureString
        self.savePath = savePath
        self.iteration = 0
    def forward(self, y_pred, y):
        y_pred, y = y_pred.reshape(self.numClasses, -1), y.reshape(-1, 1)
        if self.softMax:
            confidences = torch.max(torch.softmax(y_pred, dim=0), dim=0)
        else:
            confidences = torch.max(torch.sigmoid(y_pred), dim=0)
        indices = confidences.indices.reshape(-1, 1)
        mask = (y != self.indexToIgnore)
        y_filtered = y[mask]
        indices = indices[mask]

        correct = torch.eq(indices, y_filtered).reshape(-1)
        confidences = confidences.values.reshape(-1, 1)[mask].reshape(-1)
        for bin in range(self.numBins):
            mask = torch.logical_and((bin / self.numBins) < confidences, confidences <= ((bin + 1) / self.numBins))

            self._num_correct[bin] += torch.sum(correct[mask]).to(self._device)
            self._num_examples[bin] += torch.sum(mask).to(self._device)
            self._accumulated_confidence[bin] += torch.sum(confidences[mask]).to(self._device)
    def compute(self):
        confidence = (self._accumulated_confidence / self._num_examples).to('cpu')
        accuracy = (self._num_correct / self._num_examples).to('cpu')
        boundaries = (torch.arange(self.numBins) / self.numBins + 1 / (2 * self.numBins)).to('cpu')
        frequency = (self._num_examples).to('cpu').numpy()

        fig, ax = plt.subplots(2, sharex=True)
        ax[0].set_xlim(0, 1)
        ax[0].set_ylim(0, 1)
        ax[0].set_ylabel('accuracy')
        ax[0].bar(boundaries.numpy(), accuracy, width=0.9 / self.numBins, label='accuracy')
        diag = np.linspace(0, 1, 100)
        #If confidence is lower than accuracy draw confidence is front
        ax[0].bar(boundaries.numpy(), torch.where(confidence < accuracy, confidence, 0), width=0.9 / self.numBins, color='xkcd:light pink', zorder=5, label='mean confidence')
        # If confidence is higher than accuracy draw confidence is behind
        ax[0].bar(boundaries.numpy(), torch.where(confidence > accuracy, confidence, 0), width=0.9 / self.numBins, color='xkcd:light pink', zorder=-5)
        ax[0].plot(diag, diag, 'r', zorder=10)
        ax[0].plot(boundaries.numpy(), accuracy, 'bo', zorder=9)
        ax[0].legend()
        ax[1].bar(boundaries.numpy(), frequency, width=0.9 / self.numBins)
        ax[1].set_xlabel('confidence')
        ax[1].set_ylabel('number of samples')
        ax[1].set_yscale('log')
        with open(self.savePath / 'calibration.txt', 'a') as f:
            f.write(self.temperatureString.__str__())
        plt.savefig(self.savePath / '{date:%Y-%m-%d_%H-%M-%S}.png'.format(date=datetime.datetime.now()))
        self._num_correct = torch.zeros(self.numBins)
        self._num_examples = torch.zeros(self.numBins)
        self._accumulated_confidence = torch.zeros(self.numBins)
        self.iteration += 1


def plotReliabilityDiagram(model: nn.Module, testLoader, indexToIgnore, numBins, numClasses, savePath, device, softMax, temperature=torch.tensor(1.0)):
    accuracyUncertaintyPlot = AccuracyUncertaintyPlot(indexToIgnore, numBins, numClasses, savePath, f"Temperature: {str(temperature).replace('.',',')}", device, softMax)
    globalTemperature = temperature

    def plotStep(engine, batch):
        img = batch['img']
        labelOHE, label = batch['label']
        img = img.to(device)
        label = label.to(device).to(torch.int64)
        y_pred = model(img).squeeze(1)
        y_pred = y_pred / globalTemperature.reshape(-1).unsqueeze(1).unsqueeze(1).repeat(1, y_pred.shape[2],
                                                                                         y_pred.shape[3])
        accuracyUncertaintyPlot(y_pred, label)
        return {'y_pred': y_pred, 'label': label}
    engine = Engine(plotStep)
    model.requires_grad_(False)
    @engine.on(Events.EPOCH_COMPLETED)
    def evaluatorOnEpochCompleted(engine):
        accuracyUncertaintyPlot.compute()
    with torch.no_grad():
        engine.run(testLoader, max_epochs=1)
    # Reset Previous State
    model.requires_grad_(True)

def plotModelPrediction(model:nn.Module, testLoader, savePath: Path, device, colorDict, targetHeight: int, targetWidth: int, numClasses: int):
    colorChannels = list(colorDict.keys())[0].shape[0]
    colorTensor = torch.zeros((colorChannels, len(colorDict.keys())))
    for i, color in enumerate(colorDict.keys()):
        colorTensor[:, i] = color
    colorTensor = colorTensor.to(device)
    savePath = (savePath / '{date:%Y-%m-%d_%H-%M-%S}'.format( date=datetime.datetime.now()))
    savePath.mkdir(parents=True, exist_ok=False)
    for i, batch in enumerate(tqdm(testLoader)):
        image = batch['img'].to(device)
        labelOHE = batch['label'][0].to(device)
        y_pred = model(image)
        #Compute Predicted Class
        classMap = torch.argmax(y_pred, dim=1).squeeze(0)
        classMap = classMap.reshape(-1)
        classMapOHE = nn.functional.one_hot(classMap, numClasses).T.to(torch.float)
        #Convert Class Map from class index to RGB
        result = (colorTensor @ classMapOHE).reshape(colorChannels, targetHeight, targetWidth)
        labelOHE = labelOHE.reshape(numClasses, -1).to(torch.float)
        #Convert ground truth label from class index to RGB for comparison
        original = (colorTensor @ labelOHE).reshape(colorChannels, targetHeight, targetWidth)
        inputImage = image
        (savePath / str(i)).mkdir(parents=True, exist_ok=False)
        torchvision.utils.save_image(inputImage, savePath / str(i) / 'input.png')
        torchvision.utils.save_image(result / 255.0, savePath / str(i) / 'result.png')
        torchvision.utils.save_image(original / 255.0, savePath / str(i) / 'original.png')

def plotUncertaintyImage(model:nn.Module, testLoader, savePath, device, uncertaintyFunction = lambda x : 1 - x, softMax = True):
    savePath = (savePath / '{date:%Y-%m-%d_%H-%M-%S}'.format(date=datetime.datetime.now()))
    savePath.mkdir(parents=True, exist_ok=False)
    for i, batch in enumerate(tqdm(testLoader)):
        image = batch['img'].to(device)
        y_pred = model(image)
        if softMax:
            y_pred = torch.softmax(y_pred, dim=1)
        else:
            y_pred = torch.sigmoid(y_pred)
        confidenceMap = torch.max(y_pred, dim=1).values
        uncertaintyMap = uncertaintyFunction(confidenceMap)
        fig, ax = plt.subplots()
        divider = make_axes_locatable(ax)
        cax = divider.append_axes('right', size='5%', pad=0.05)
        im = ax.imshow(uncertaintyMap.squeeze(0).cpu().detach().numpy(), cmap='seismic', vmin=0, vmax=1)
        fig.colorbar(im, cax=cax, orientation='vertical')
        plt.savefig(savePath / f'{i}.png')
        plt.close(fig)

def plotLearningGraph(basePath: Path, propertyName: str):
    x = []
    with open(basePath / 'results.txt', 'r') as f:
        for line in f:
            x.append(float(line.split(propertyName + " = ")[1].split('\n')[0].split(',')[0]))
    plt.scatter(list(range(len(x))), x)
    plt.ylabel(propertyName)
    plt.xlabel('Epoch')
    plt.xticks(list(range(len(x))))
    plt.savefig(basePath / f'{propertyName}.png')
    plt.close()