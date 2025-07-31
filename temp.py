import torch
from ignite.exceptions import NotComputableError

NUM_BINS = 1
y_pred = torch.tensor([[1.0,0.0]])
y = torch.tensor([1])
indexToIgnore = -1
device = 'cpu'
num_correct = torch.zeros(NUM_BINS)
num_examples = torch.zeros(NUM_BINS)
accumulated_confidence = torch.zeros(NUM_BINS)
y_pred, y = y_pred.detach(), y.detach()

confidences = torch.max(y_pred, dim=1)
indices = confidences.indices
mask = (y != indexToIgnore)
mask &= (indices != indexToIgnore)
y_filtered = y[mask]
indices = indices[mask]
correct = torch.eq(indices, y_filtered).reshape(-1)
confidences = confidences.values[mask]
for bin in range(NUM_BINS):
    mask = torch.logical_and((bin / NUM_BINS) < confidences, confidences <= ((bin + 1) / NUM_BINS))
    num_correct[bin] += torch.sum(correct[mask]).to(device)
    num_examples[bin] += torch.sum(mask).to(device)
    accumulated_confidence[bin] += torch.sum(confidences[mask]).to(device)
if torch.sum(num_examples) == 0:
    raise NotComputableError(
        "ExpectedCalibrationError must have at least one example per bin before it can be computed")
# Possibly set Error to 0 if no items in bin, and just require at least one sample
ECE = 0
for bin in range(NUM_BINS):
    if num_examples[bin] == 0:
        continue
    ECE += (num_examples[bin] / torch.sum(num_examples)) * torch.abs(
        (num_correct[bin] / num_examples[bin]) - (
                accumulated_confidence[bin] / num_examples[bin]))
print(ECE)