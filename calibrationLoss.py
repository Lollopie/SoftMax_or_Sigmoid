import torch
from torch import nn
from ignite.exceptions import NotComputableError
class CalibrationLoss(nn.Module):
    def __init__(self):
        super().__init__()
    def compute(self):
        raise NotImplementedError
class ExpectedCalibrationErrorLoss(CalibrationLoss):
    def __init__(self, indexToIgnore, numBins, device, softMax):
        super().__init__()
        self.indexToIgnore = indexToIgnore
        self.numBins = numBins
        self.device = device
        self.softMax = softMax
        self._num_correct = torch.zeros(self.numBins, device=self.device)
        self._num_examples = torch.zeros(self.numBins, device=self.device)
        self._accumulated_confidence = torch.zeros(self.numBins, device=self.device)
    def forward(self, y_pred, y):
        y_pred, y = y_pred.reshape(y_pred.shape[1], -1), y.reshape(-1, 1)
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
            self._num_correct[bin] += torch.sum(correct[mask]).to(self.device)
            self._num_examples[bin] += torch.sum(mask).to(self.device)
            self._accumulated_confidence[bin] += torch.sum(confidences[mask]).to(self.device)

    def compute(self):
        if torch.sum(self._num_examples) == 0:
            raise NotComputableError(
                "ExpectedCalibrationErrorLoss must have at least one example before it can be computed")
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
class StaticCalibrationErrorLoss(CalibrationLoss):
    def __init__(self, indexToIgnore, numBins, numClasses, device, softMax):
        super().__init__()
        self.indexToIgnore = indexToIgnore
        self.numBins = numBins
        self.device = device
        self.numClasses = numClasses
        self.softMax = softMax
        self._num_correct = torch.zeros(self.numBins, self.numClasses, device=self.device)
        self._num_examples = torch.zeros(self.numBins, self.numClasses, device=self.device)
        self._accumulated_confidence = torch.zeros(self.numBins, self.numClasses, device=self.device)
    def forward(self, y_pred, y):
        y_pred, y = y_pred.reshape(self.numClasses, -1), y.reshape(-1, 1)
        if self.softMax:
            confidences = torch.softmax(y_pred, dim=0)
        else:
            confidences = torch.sigmoid(y_pred)
        indices = torch.argmax(confidences, dim=0).reshape(-1, 1)
        mask = (y != self.indexToIgnore)
        y_filtered = y[mask]
        confidences = confidences[mask.repeat(1, self.numClasses).T].reshape(-1, self.numClasses)
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

class AdaptiveExpectedCalibrationErrorLoss(CalibrationLoss):
    def __init__(self, numBins, device, softMax):
        super().__init__()
        self.numBins = numBins
        self.device = device
        self.softMax = softMax
        self._data = torch.empty((0, 2)).to(self.device)
    def forward(self, y_pred, y):
        y_pred, y = y_pred.reshape(y_pred.shape[1], -1), y.reshape(-1, 1)
        if self.softMax:
            confidences = torch.max(torch.softmax(y_pred, dim=0), dim=0)
        else:
            confidences = torch.max(torch.sigmoid(y_pred), dim=0)
        indices = confidences.indices.reshape(-1, 1)
        correct = torch.eq(indices, y).reshape(-1)
        confidences = confidences.values.reshape(-1)
        self._data = torch.cat((torch.stack((confidences, correct), dim=1), self._data), dim=0)

    def compute(self):
        if self._data.shape[0] == 0:
            raise NotComputableError(
                "AdaptiveExpectedCalibrationErrorLoss must have at least one example before it can be computed")
        sortedData = self._data[self._data[:, 0].argsort()]
        boundaries = torch.floor((torch.ones(self.numBins + 1, device=self.device) * sortedData.shape[0]) * (torch.arange(self.numBins + 1, device=self.device) / self.numBins))
        AECE = torch.tensor(0.0, device=self.device)
        for bin in range(self.numBins):
            lowerBound = boundaries[bin].to(torch.int)
            upperBound = boundaries[bin + 1].to(torch.int)
            AECE += ((upperBound - lowerBound) / self._data.shape[0]) * torch.abs(
                (torch.sum(sortedData[lowerBound:upperBound, 1]) / (upperBound - lowerBound)) -
                (torch.sum(sortedData[lowerBound:upperBound, 0]) / (upperBound - lowerBound)))
        self._data = torch.empty((0, 2), device=self.device)
        return AECE
class AdaptiveStaticCalibrationErrorLoss(CalibrationLoss):
    def __init__(self, numBins, numClasses, device, softMax):
        super().__init__()
        self.numBins = numBins
        self.numClasses = numClasses
        self.device = device
        self.softMax = softMax
        self._data = torch.empty((0, self.numClasses + 1), device=self.device)
    def forward(self, y_pred, y):
        y_pred, y = y_pred.reshape(y_pred.shape[1], -1), y.reshape(-1, 1)
        if self.softMax:
            confidences = torch.softmax(y_pred, dim=0).to(self.device)
        else:
            confidences = torch.sigmoid(y_pred).to(self.device)
        indices = torch.argmax(confidences, dim=0).reshape(-1, 1)
        correct = torch.eq(indices, y).reshape(-1, 1).to(self.device)
        self._data = torch.cat((torch.cat((y_pred.T, correct), dim=1), self._data), dim=0)

    def compute(self):
        if self._data.shape[0] == 0:
            raise NotComputableError(
                "AdaptiveStaticCalibrationErrorLoss must have at least one example before it can be computed")
        ASCE = torch.tensor(0.0, device=self.device)
        for classIndex in range(self.numClasses):
            sortedConfidences = self._data[:, classIndex].sort().values
            sortedCorrectness = self._data[self._data[:, classIndex].argsort()][:, -1]
            boundaries = torch.floor((torch.ones(self.numBins + 1, device=self.device) * sortedConfidences.shape[0]) * (torch.arange(self.numBins + 1, device=self.device) / self.numBins))
            for bin in range(self.numBins):
                lowerBound = boundaries[bin].to(torch.int)
                upperBound = boundaries[bin + 1].to(torch.int)
                ASCE += (((upperBound - lowerBound) / sortedConfidences.shape[0]) *
                         torch.square((torch.sum(sortedCorrectness[lowerBound:upperBound]) / (upperBound - lowerBound)) -
                                   torch.sum(sortedConfidences[lowerBound:upperBound]) / (upperBound - lowerBound)))
        self._data = torch.empty((0, self.numClasses + 1), device=self.device)
        ASCE *= 1 / (self.numBins * self.numClasses)
        return ASCE