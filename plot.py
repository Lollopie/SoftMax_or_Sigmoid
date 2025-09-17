import torch
from ignite.engine import Engine, Events
from matplotlib import pyplot as plt
import numpy as np
from torch import nn
class AccuracyUncertaintyPlot(nn.Module):
    def __init__(self, indexToIgnore, numBins, numClasses, namingScheme, device):
        #maybe add option for device
        super().__init__()
        self.indexToIgnore = indexToIgnore
        self.numBins = numBins
        self.numClasses = numClasses
        self._device = device
        self._num_correct = torch.zeros(self.numBins).to(self._device)
        self._num_examples = torch.zeros(self.numBins).to(self._device)
        self._accumulated_confidence = torch.zeros(self.numBins).to(self._device)
        self.namingScheme = namingScheme

        self.iteration = 0
    def forward(self, y_pred, y):
        y_pred, y = y_pred.reshape(self.numClasses, -1), y.reshape(-1, 1)

        confidences = torch.max(torch.softmax(y_pred, dim=0), dim=0)
        indices = confidences.indices.reshape(-1, 1)
        mask = (y != self.indexToIgnore)
        mask &= (indices != self.indexToIgnore)
        y_filtered = y[mask]
        indices = indices[mask]

        #Accuracy after Guo et al.
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
        edges = (torch.arange(self.numBins) / self.numBins + 1 / (2 * self.numBins)).to('cpu')
        fig, ax = plt.subplots()
        ax.set_xlim(0, 1)
        ax.set_xlabel('confidence')
        ax.set_ylim(0, 1)
        ax.set_ylabel('relative frequency')
        plt.bar(edges.numpy(), accuracy, width=0.9 / self.numBins, label='relative frequency')
        diag = np.linspace(0, 1, 100)
        plt.bar(edges.numpy(), torch.where(confidence < accuracy, confidence, 0), width=0.9 / self.numBins, color='xkcd:light pink', zorder=5, label='mean confidence')
        plt.bar(edges.numpy(), torch.where(confidence > accuracy, confidence, 0), width=0.9 / self.numBins, color='xkcd:light pink', zorder=-5)
        plt.plot(diag, diag, 'r', zorder=10)
        plt.plot(edges.numpy(), accuracy, 'bo', zorder=9)
        ax.legend()
        plt.savefig(self.namingScheme + str(self.iteration))
        self._num_correct = torch.zeros(self.numBins)
        self._num_examples = torch.zeros(self.numBins)
        self._accumulated_confidence = torch.zeros(self.numBins)
        self.iteration += 1

globalTemperature = None
globalDevice = None
globalModel = None
accuracyUncertaintyPlot = None

def plotStep(engine, batch):
    img = batch['img']
    labelOHE, label = batch['label']
    img = img.to(globalDevice)
    label = label.to(globalDevice).to(torch.int64)
    y_pred = globalModel(img).squeeze(1)
    y_pred = y_pred / globalTemperature
    accuracyUncertaintyPlot(y_pred, label)
    return {'y_pred': y_pred, 'label': label}

def plot(model: nn.Module, testLoader, indexToIgnore, numBins, numClasses, namingScheme, device, epochs=1, temperature=1):
    global globalTemperature
    global accuracyUncertaintyPlot
    global globalDevice
    global globalModel
    globalDevice = device
    globalModel = model
    accuracyUncertaintyPlot = AccuracyUncertaintyPlot(indexToIgnore, numBins, numClasses, namingScheme, device)
    globalTemperature = temperature
    engine = Engine(plotStep)
    model.requires_grad_(False)
    @engine.on(Events.EPOCH_COMPLETED)
    def evaluatorOnEpochCompleted(engine):
        accuracyUncertaintyPlot.compute()
    engine.run(testLoader, max_epochs=epochs)
    # Reset Previous State
    model.requires_grad_(True)