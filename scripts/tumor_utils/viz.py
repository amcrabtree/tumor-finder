import numpy as np
from matplotlib import pyplot as plt

def ViewNpyImg(npy_file:str):
    """ Display .npy tile image as pop-up. """
    tile_np = np.load(npy_file)
    plt.imshow(tile_np, interpolation='nearest')
    plt.show()
