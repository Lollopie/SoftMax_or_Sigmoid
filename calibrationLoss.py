import torch
from torch import nn
from ignite.exceptions import NotComputableError
class ExpectedCalibrationErrorLoss(nn.Module):
    def __init__(self, indexToIgnore, numBins, device):
        super().__init__()
        self.indexToIgnore = indexToIgnore
        self.numBins = numBins
        self.device = device
        self._num_correct = torch.zeros(self.numBins, device=self.device)
        self._num_examples = torch.zeros(self.numBins, device=self.device)
        self._accumulated_confidence = torch.zeros(self.numBins, device=self.device)
    def forward(self, y_pred, y):
        y_pred, y = y_pred.reshape(y_pred.shape[1], -1), y.reshape(-1, 1)
        confidences = torch.max(torch.sigmoid(y_pred), dim=0)
        indices = confidences.indices.reshape(-1, 1)
        mask = (y != self.indexToIgnore)
        mask &= (indices != self.indexToIgnore)
        y_filtered = y[mask]
        indices = indices[mask]
        correct = torch.eq(indices, y_filtered).reshape(-1)
        confidences = confidences.values.reshape(-1, 1)[mask].reshape(-1)
        for bin in range(self.numBins):
            mask = torch.logical_and((bin / self.numBins) < confidences, confidences <= ((bin + 1) / self.numBins))
            self._num_correct[bin] += torch.sum(correct[mask]).to(self.device)
            self._num_examples[bin] += torch.sum(mask).to(self.device)
            self._accumulated_confidence[bin] += torch.sum(confidences[mask]).to(self.device)

    def compute(self):
        if torch.sum(self._num_examples) == 0:
            raise NotComputableError(
                "ExpectedCalibrationErrorLoss must have at least one example before it can be computed")
        # Possibly set Error to 0 if no items in bin, and just require at least one sample
        ECE = torch.tensor(0.0, device=self.device)
        for bin in range(self.numBins):
            if self._num_examples[bin] == 0:
                continue
            ECE += (self._num_examples[bin] / torch.sum(self._num_examples)) * torch.abs(
                (self._num_correct[bin] / self._num_examples[bin]) - (
                            self._accumulated_confidence[bin] / self._num_examples[bin]))
        self._num_correct = torch.zeros(self.numBins, device=self.device)
        self._num_examples = torch.zeros(self.numBins, device=self.device)
        self._accumulated_confidence = torch.zeros(self.numBins, device=self.device)
        return ECE
class StaticCalibrationErrorLoss(nn.Module):
    def __init__(self, indexToIgnore, numBins, numClasses, device):
        super().__init__()
        self.indexToIgnore = indexToIgnore
        self.numBins = numBins
        self.device = device
        self.numClasses = numClasses
        self._num_correct = torch.zeros(self.numBins, self.numClasses, device=self.device)
        self._num_examples = torch.zeros(self.numBins, self.numClasses, device=self.device)
        self._accumulated_confidence = torch.zeros(self.numBins, self.numClasses, device=self.device)
    def forward(self, y_pred, y):
        y_pred, y = y_pred.reshape(self.numClasses, -1), y.reshape(-1, 1)
        confidences = torch.sigmoid(y_pred)
        indices = torch.argmax(confidences, dim=0).reshape(-1, 1)
        mask = (y != self.indexToIgnore)
        mask &= (indices != self.indexToIgnore)
        y_filtered = y[mask]
        confidences = confidences[mask.repeat(1, self.numClasses)].reshape(-1, self.numClasses)
        for classIndex in range(self.numClasses):
            if classIndex == self.indexToIgnore:
                continue
            correct = torch.eq(y_filtered, classIndex).reshape(-1, 1)
            for bin in range(self.numBins):
                mask = torch.logical_and((bin / self.numBins) < confidences[:, classIndex],
                                         confidences[:, classIndex] <= ((bin + 1) / self.numBins))
                self._num_correct[bin, classIndex] += torch.sum(correct[mask]).to(self.device)
                self._num_examples[bin, classIndex] += torch.sum(mask).to(self.device)
                self._accumulated_confidence[bin, classIndex] += torch.sum(confidences[:, classIndex][mask]).to(self.device)

    def compute(self):
        if torch.sum(self._num_examples) == 0:
            raise NotComputableError(
                "StaticCalibrationErrorLoss must have at least one example before it can be computed")
        SCE = torch.tensor(0.0).to(self.device)
        for classIndex in range(self.numClasses):
            if classIndex == self.indexToIgnore:
                continue
            for bin in range(self.numBins):
                if self._num_examples[bin, classIndex] == 0:
                    continue
                SCE += (self._num_examples[bin, classIndex] / torch.sum(self._num_examples[:, classIndex])) * torch.abs(
                    (self._num_correct[bin, classIndex] / self._num_examples[bin, classIndex]) - (
                                self._accumulated_confidence[bin, classIndex] / self._num_examples[bin, classIndex]))
        self._num_correct = torch.zeros(self.numBins, self.numClasses, device=self.device)
        self._num_examples = torch.zeros(self.numBins, self.numClasses, device=self.device)
        self._accumulated_confidence = torch.zeros(self.numBins, self.numClasses, device=self.device)
        return SCE

class AdaptiveExpectedCalibrationErrorLoss(nn.Module):
    def __init__(self, numBins, device):
        super().__init__()
        self.numBins = numBins
        self.device = device
        self._data = torch.empty((0, 2)).to(self.device)
    def forward(self, y_pred, y):
        y_pred, y = y_pred.reshape(y_pred.shape[1], -1), y.reshape(-1, 1)
        confidences = torch.max(torch.sigmoid(y_pred), dim=0)
        indices = confidences.indices.reshape(-1, 1)
        correct = torch.eq(indices, y).reshape(-1)
        confidences = confidences.values.reshape(-1)
        self._data = torch.cat((torch.stack((confidences, correct), dim=1), self._data), dim=0)

    def compute(self):
        if torch.sum(self._num_examples) == 0:
            raise NotComputableError(
                "AdaptiveExpectedCalibrationErrorLoss must have at least one example before it can be computed")
        sortedData = self._data[:, 0].clone().sort().values
        boundaries = torch.quantile(sortedData, torch.arange(self.numBins + 1) / self.numBins, interpolation='midpoint')
        AECE = torch.tensor(0.0, device=self.device)
        for bin in range(self.numBins):
            mask = torch.logical_and(boundaries[bin] < self._data[:, 0], self._data[:, 0] <= boundaries[bin+1])
            if torch.sum(mask) == 0:
                continue
            AECE += (torch.sum(mask) / self._data.shape[0]) * torch.abs(
                (torch.sum(self._data[:, 1][mask]) / torch.sum(mask)) -
                (torch.sum(self._data[:, 0][mask]) / torch.sum(mask)))
        self._data = torch.empty((0, 2), device=self.device)
        return AECE
class AdaptiveStaticCalibrationErrorLoss(nn.Module):
    def __init__(self, numBins, numClasses, device):
        super().__init__()
        self.numBins = numBins
        self.numClasses = numClasses
        self.device = device
        self._data = torch.empty((0, self.numClasses + 1), device=self.device)
    def forward(self, y_pred, y):
        y_pred, y = y_pred.reshape(y_pred.shape[1], -1), y.reshape(-1, 1)
        confidences = torch.sigmoid(y_pred).to(self.device)
        indices = torch.argmax(confidences, dim=0).reshape(-1, 1)
        correct = torch.eq(indices, y).reshape(-1, 1).to(self.device)
        self._data = torch.cat((torch.cat((y_pred, correct), dim=1), self._data), dim=0)

    def compute(self):
        if torch.sum(self._num_examples) == 0:
            raise NotComputableError(
                "AdaptiveStaticCalibrationErrorLoss must have at least one example before it can be computed")
        ASCE = torch.tensor(0.0, device=self.device)
        for classIndex in range(self.numClasses):
            sortedData = self._data[:, classIndex].clone().sort().values
            boundaries = torch.quantile(sortedData, torch.arange(self.numBins + 1) / self.numBins, interpolation='midpoint')
            for bin in range(self.numBins):
                mask = torch.logical_and(boundaries[bin] < self._data[:, classIndex], self._data[:, classIndex] <= boundaries[bin+1])
                ASCE += ((torch.sum(mask) / sortedData.shape[0]) *
                         torch.abs((torch.sum(self._data[:, -1][mask]) / torch.sum(mask)) -
                                   torch.sum(self._data[:, classIndex][mask]) / torch.sum(mask)))
        self._data = torch.empty((0, self.numClasses + 1), device=self.device)
        return ASCE