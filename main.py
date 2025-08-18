import torch
from matplotlib import pyplot as plt
import numpy as np
y_pred = torch.tensor([[0.78, 0.22],
                    [0.36, 0.64],
                    [0.08, 0.92],
                    [0.58, 0.42],
                    [0.49, 0.51],
                    [0.85, 0.15],
                    [0.30, 0.70],
                    [0.63, 0.37],
                    [0.17, 0.83]])

y = torch.tensor([0,1,0,0,0,0,1,1,1]).reshape(-1, 1)
numBins = 5
numClasses = 2
_num_correct = torch.zeros(numBins)
_num_examples = torch.zeros(numBins)
_accumulated_confidence = torch.zeros(numBins)
confidences = y_pred.T.flatten().detach().clone()
#Accuracy after Guo et al.
correct = torch.eq(torch.argmax(y_pred, dim=1).reshape(-1, 1), y).reshape(-1).repeat(numClasses)
for bin in range(numBins):
    mask = torch.logical_and((bin / numBins) < confidences, confidences <= ((bin + 1) / numBins))

    _num_correct[bin] += torch.sum(correct[mask])
    _num_examples[bin] += torch.sum(mask)
    _accumulated_confidence[bin] += torch.sum(confidences[mask])

confidence = _accumulated_confidence / _num_examples
accuracy = _num_correct / _num_examples
edges = torch.arange(numBins) / numBins + 1 / (2 * numBins)
fig, ax = plt.subplots()
ax.set_xlim(0, 1)
ax.set_xlabel('confidence')
ax.set_ylim(0, 1)
ax.set_ylabel('relative frequency')
plt.bar(edges.numpy(), accuracy, width=0.9 / numBins, label='relative frequency')
diag = np.linspace(0, 1, 100)
plt.bar(edges.numpy(), torch.where(confidence < accuracy, confidence, 0), width=0.9 / numBins, color='xkcd:light pink', zorder=5, label='mean confidence')
plt.bar(edges.numpy(), torch.where(confidence > accuracy, confidence, 0), width=0.9 / numBins, color='xkcd:light pink', zorder=-5)
plt.plot(diag, diag, 'r', zorder=10)
plt.plot(edges.numpy(), accuracy, 'bo', zorder=9)
ax.legend()
plt.show()