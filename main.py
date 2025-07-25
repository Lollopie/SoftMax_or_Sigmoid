import rasterio
import matplotlib.pyplot as plt
src_path = "top_potsdam_7_9_RGB.tif"
img = rasterio.open(src_path)
plt.imshow(img.read([1,2,3]).transpose(1, 2, 0))