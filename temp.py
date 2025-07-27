import os
from pathlib import Path

pathlist = Path('.').rglob('**/*_rgb')
for path in pathlist:
    if 'label' in path.stem:
        continue
    if 'rgb_rgb' not in path.stem:
        continue
    newName = None
    if "RGB" in path.stem:
        newName = path.stem.replace('_rgb', '')
    else:
        newName = path.stem.replace('_rgb_rgb', '_rgb')
    os.rename(path, path.parent / newName)
