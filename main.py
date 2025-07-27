import torch
from tqdm import tqdm
from PIL import Image
from pathlib import Path
from torchvision.transforms import v2
from time import time
transforms = v2.Compose(
    [
        v2.ToImage(),
    ]
)
rootdir = Path('./5_Labels_all')
pathlist = rootdir.rglob('*.tif')
paths = []
labels = []
for path in pathlist:
    img = Image.open(path)
    imgTransformed = transforms(img)
    labels.append(imgTransformed)
    paths.append(path)
# rgbs = []
# labels = []
# for path in paths:
#     img = Image.open(rootdir / path)
#     label = Image.open(rootdir / path.replace('rgb', 'label'))
#     imgTransformed = transforms(img)
#     labelTransformed = transforms(img)
#     rgbs.append(imgTransformed)
#     labels.append(labelTransformed)
colorDict = {}
i = 0
start = time()
for i, label in enumerate(labels):
    label2D = label.reshape(3, -1).transpose(0, 1)
    text = label2D.unique(dim=0).__str__()
    with open('results.txt', 'a') as file:
        file.write(paths[i].name + "\n" + text + "\n\n")
    # for color in label2D.unique(dim=0):
    #     if color not in colorDict:
    #         colorDict[color] = i
    #         i += 1