import torch
from torch import nn
from ignite.engine import Engine, Events
from ignite.exceptions import NotComputableError
class ExpectedCalibrationErrorLoss(nn.Module):
    def __init__(self, indexToIgnore, numBins, device):
        self.indexToIgnore = indexToIgnore
        self.numBins = numBins
        self.device = device
        self._num_correct = torch.zeros(self.numBins).to(self.device)
        self._num_examples = torch.zeros(self.numBins).to(self.device)
        self._accumulated_confidence = torch.zeros(self.numBins).to(self.device)
        super().__init__()
    def forward(self, y_pred, y):
        confidences = torch.max(torch.softmax(y_pred, dim=1), dim=1)
        indices = confidences.indices
        mask = (y != self.indexToIgnore)
        mask &= (indices != self.indexToIgnore)
        y_filtered = y[mask]
        indices = indices[mask]
        correct = torch.eq(indices, y_filtered).reshape(-1)
        confidences = confidences.values[mask].reshape(-1)
        for bin in range(self.numBins):
            mask = torch.logical_and((bin / self.numBins) < confidences, confidences <= ((bin + 1) / self.numBins))
            self._num_correct[bin] += torch.sum(correct[mask]).to(self.device)
            self._num_examples[bin] += torch.sum(mask).to(self.device)
            self._accumulated_confidence[bin] += torch.sum(confidences[mask]).to(self.device)

    def compute(self):
        if torch.sum(self._num_examples) == 0:
            raise NotComputableError(
                "ExpectedCalibrationError must have at least one example per bin before it can be computed")
        # Possibly set Error to 0 if no items in bin, and just require at least one sample
        ECE = torch.tensor(0.0).to(self.device)
        for bin in range(self.numBins):
            if self._num_examples[bin] == 0:
                continue
            ECE += (self._num_examples[bin] / torch.sum(self._num_examples)) * torch.abs(
                (self._num_correct[bin] / self._num_examples[bin]) - (
                            self._accumulated_confidence[bin] / self._num_examples[bin]))
        self._num_correct = torch.zeros(self.numBins).to(self.device)
        self._num_examples = torch.zeros(self.numBins).to(self.device)
        self._accumulated_confidence = torch.zeros(self.numBins).to(self.device)
        return ECE
def output_transform(output):
    y_pred = output['y_pred']
    y = output['label']
    return y_pred, y
globalDevice = None
globalModel = None
expectedCalibrationErrorLoss = None
optimizer = None
temperature = None
def calibrationStep(engine, batch):
    global temperature
    global globalDevice
    global globalModel
    img = batch['img']
    labelOHE, label = batch['label']
    img = img.to(globalDevice)
    # labelOHE = labelOHE.to(device)
    label = label.to(globalDevice).to(torch.int64)
    y_pred = globalModel(img).squeeze(1)
    y_pred = y_pred / temperature
    expectedCalibrationErrorLoss(y_pred, label)
    return {'y_pred': y_pred, 'label': label}
def calibrate(model: nn.Module, valLoader, indexToIgnore, numBins, device, learningRate=5e-3, epochs=1):
    global optimizer
    global temperature
    global expectedCalibrationErrorLoss
    global globalDevice
    global globalModel
    globalDevice = device
    globalModel = model
    expectedCalibrationErrorLoss = ExpectedCalibrationErrorLoss(indexToIgnore, numBins, device)
    model.eval()
    temperature = torch.tensor(1.0, requires_grad=True)
    localOptimizer = torch.optim.SGD([temperature], lr=learningRate)
    engine = Engine(calibrationStep)
    @engine.on(Events.EPOCH_COMPLETED)
    def evaluator_on_epoch_completed(engine):
        global temperature
        loss = expectedCalibrationErrorLoss.compute()
        loss.backward()
        # localOptimizer.step()
        localOptimizer.zero_grad()
        print(f'Calibration: ECE = {loss.item()}, {temperature}')
    model.requires_grad_(False)
    engine.run(valLoader, max_epochs=epochs)
    # Reset Previous State
    model.requires_grad_(True)
    return temperature