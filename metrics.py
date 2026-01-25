import torch
from ignite.metrics import Metric
from ignite.exceptions import NotComputableError
from ignite.metrics.metric import sync_all_reduce, reinit__is_reduced

class ExpectedCalibrationError(Metric):
    def __init__(self, ignoredClass, numBins: torch.Tensor, softMax: bool, outputTransform=lambda x: x, device="cpu"):
        self.ignoredClass = ignoredClass
        self.numBins = numBins
        self._num_correct = torch.zeros(numBins)
        self._num_examples = torch.zeros(numBins)
        self._accumulated_confidence = torch.zeros(numBins)
        self.softMax = softMax
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
        if self.softMax:
            confidences = torch.max(torch.softmax(y_pred, dim=0), dim=0)
        else:
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
    def __init__(self, ignoredClass: int | None, numBins: torch.Tensor, numClasses: torch.Tensor, outputTransform=lambda x: x, device:str="cpu"):
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
    def __init__(self, numBins:torch.Tensor, outputTransform=lambda x: x, device="cpu"):
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
    def __init__(self, numBins:torch.Tensor, numClasses:torch.Tensor, outputTransform=lambda x: x, device:str="cpu"):
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
class ClassWiseAccuracy(Metric):
    def __init__(self, numClasses: int, device: str, outputTransform=lambda x: x):
        self.numClasses = numClasses
        self.numCorrect = torch.zeros(numClasses, device=device)
        self.numExamples = torch.zeros(numClasses, device=device)
        super(ClassWiseAccuracy, self).__init__(output_transform=outputTransform, device=device)
    def reset(self) -> None:
        self.numCorrect = torch.zeros(self.numClasses, device=self._device)
        self.numExamples = torch.zeros(self.numClasses, device=self._device)
        super(ClassWiseAccuracy, self).reset()

    def update(self, output) -> None:
        y_pred, y = output[0].detach().reshape(self.numClasses, -1), output[1].detach().reshape(-1)

        indices = torch.argmax(y_pred, dim=0)
        correct = torch.eq(indices, y).view(-1)
        correctClasses = indices[correct]
        counts = torch.unique(correctClasses, return_counts=True)
        for classIndex, count in zip(counts[0], counts[1]):
            self.numCorrect[classIndex] += count
        for classIndex in range(self.numClasses):
            self.numExamples[classIndex] += torch.sum(torch.eq(y, classIndex))

    def compute(self):
        # if torch.any(torch.eq(self.numExamples, 0)):
        #     raise NotComputableError('ClassWiseAccuracy must have at least one example per class before it can be computed.')
        # print(self.numExamples / torch.sum(self.numExamples))
        return self.numCorrect / self.numExamples