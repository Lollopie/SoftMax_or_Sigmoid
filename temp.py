# string = """[[   47427,        0,      591,     5224,      230,      326,     5690,                0,      562],
#         [       1,   309795,  1625129,  1839768,    45425,    14140,    27819,           763233,   110154],
#         [       3,   106763, 51584582,  8577820,   391594,  5130905,   406873,          1232901,   262098],
#         [     197,   121102,  3839764, 51431167,  3684769,   377274,   287639,           429567,  4603850],
#         [       0,     5463,   629571,  8504673, 10664720,   182972,   164863,            39319,   926017],
#         [       0,    14874,  5025700,   596728,    48330, 34484614,   140619,            23743,    89510],
#         [     475,    15893,  1473003,  2539746,   247351,   110038, 28136262,           209858,   291379],
#         [       0,   788036,  4793846,  2736053,   127062,   183671,   743378,         10381151,     4714],
#         [      33,     3908,   311440,  7711242,   447240,   132459,    60738,              233, 23843253]]"""
# print(string.replace(' ', '').replace('[','').replace(']','').replace(',','\t').replace('\t\n','\n'))


import os
from pathlib import Path
from plot import plotLearningGraph

checkpoint_dir = Path(f"./checkpoints_learning_graph/")
subDirectories = [f.name for f in os.scandir(checkpoint_dir) if f.is_dir()]
for directory in subDirectories:
    if (checkpoint_dir / directory / 'ACC.png').is_file():
        os.remove(checkpoint_dir / directory / 'ACC.png')
        os.remove(checkpoint_dir / directory / 'mIoU.png')
    plotLearningGraph(checkpoint_dir / directory, "ACC")
    plotLearningGraph(checkpoint_dir / directory, "mIoU")