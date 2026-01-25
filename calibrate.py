import torch
from ignite.handlers import ProgressBar
from ignite.metrics import Accuracy
from torch import nn
from ignite.engine import Engine, Events
from calibrationLoss import CalibrationLoss
def output_transform(output):
    y_pred = output['y_pred']
    y = output['label']
    return y_pred, y

def calibrate(model: nn.Module, valLoader, numClasses, loss: CalibrationLoss, device, softMax, datasetIndex, calibrationMode, learningRate=9e-2, epochs=1):
    if calibrationMode:
        temperature = torch.tensor(1.0, dtype=torch.float, requires_grad=True, device=device)
    else:
        temperature = torch.ones(numClasses, dtype=torch.float, requires_grad=True, device=device)
    optimizer = torch.optim.SGD([temperature], lr=learningRate)

    def calibrationStep(engine, batch):
        img = batch['img']
        labelOHE, label = batch['label']
        img = img.to(device)
        label = label.to(device).to(torch.int64)
        y_pred = model(img).squeeze(1)
        #Scale Temperature up to y_pred shape (necessary for class-wise temperature)
        y_pred = y_pred / temperature.reshape(-1).unsqueeze(1).unsqueeze(1).repeat(1, y_pred.shape[2], y_pred.shape[3])
        loss(y_pred, label)
        return {'y_pred': y_pred, 'label': label}
    def gradientDescentStep(engine):
        localLoss = loss.compute()
        localLoss.backward()
        optimizer.step()
        optimizer.zero_grad()
        print(f'Calibration: ECE = {loss.item()}')
    engine = Engine(calibrationStep)
    ACC = Accuracy(output_transform)
    ACC.attach(engine, 'acc')
    ProgressBar().attach(engine)
    def doEveryThousandIterations(engine, event):
        if event % 1000 == 0:
            return True
        return False

    def printStep(engine):
        print(f'Calibration: {temperature}, {engine.state.metrics["acc"]}')
    if datasetIndex == 2:
        engine.add_event_handler(Events.EPOCH_COMPLETED, gradientDescentStep)
    else:
        #2025 data fusion dataset is too large to train for the entire epoch on 16GB VRAM
        engine.add_event_handler(Events.ITERATION_COMPLETED(event_filter=doEveryThousandIterations), gradientDescentStep)
    engine.add_event_handler(Events.EPOCH_COMPLETED, printStep)
    model.requires_grad_(False)
    model.eval()
    engine.run(valLoader, max_epochs=epochs)
    # Reset Previous State
    model.requires_grad_(True)
    return temperature