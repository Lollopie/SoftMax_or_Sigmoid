import torch
from ignite.handlers import ProgressBar
from ignite.metrics import Accuracy
from torch import nn
from ignite.engine import Engine, Events

def output_transform(output):
    y_pred = output['y_pred']
    y = output['label']
    return y_pred, y
globalDevice = None
globalModel = None
globalLoss = None
globalSoftMax = None
optimizer = None
temperature = None
def calibrationStep(engine, batch):
    global temperature
    global globalDevice
    global globalModel
    global globalLoss
    img = batch['img']
    labelOHE, label = batch['label']
    img = img.to(globalDevice)
    # labelOHE = labelOHE.to(device)
    label = label.to(globalDevice).to(torch.int64)
    y_pred = globalModel(img).squeeze(1)
    if globalSoftMax:
        y_pred = y_pred / temperature
    else:
        y_pred = y_pred / temperature.unsqueeze(1).unsqueeze(1).repeat(1, y_pred.shape[2], y_pred.shape[3])
    globalLoss(y_pred, label)
    return {'y_pred': y_pred, 'label': label}
def calibrate(model: nn.Module, valLoader, indexToIgnore, numBins, numClasses, loss, device, softMax, learningRate=9e-2, epochs=1):
    global optimizer
    global temperature
    global globalDevice
    global globalModel
    global globalLoss
    global globalSoftMax
    globalDevice = device
    globalModel = model
    globalLoss = loss
    globalSoftMax = softMax
    # model.eval()
    #Calibration: ECE = 0.045048683881759644, 1.493001937866211
    if softMax:
        temperature = torch.tensor(1.0, dtype=torch.float, requires_grad=True, device=device)
    else:
        temperature = torch.ones(numClasses, dtype=torch.float, requires_grad=True, device=device)
    localOptimizer = torch.optim.SGD([temperature], lr=learningRate)
    engine = Engine(calibrationStep)
    ACC = Accuracy(output_transform)
    ACC.attach(engine, 'acc')
    ProgressBar().attach(engine)
    def gradientDescentStep(engine, epochComplete):
        global temperature
        global globalLoss
        loss = globalLoss.compute()
        loss.backward()
        localOptimizer.step()
        localOptimizer.zero_grad()
        if epochComplete:
            print(f'Calibration: ECE = {loss.item()}, {temperature}, {engine.state.metrics["acc"]}')
    engine.add_event_handler(Events.EPOCH_COMPLETED, gradientDescentStep, True)
    engine.add_event_handler(Events.ITERATION_COMPLETED(every=1000), gradientDescentStep, False)
    model.requires_grad_(False)
    model.eval()
    engine.run(valLoader, max_epochs=epochs)
    # Reset Previous State
    model.requires_grad_(True)
    return temperature