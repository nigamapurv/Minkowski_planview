import numpy as np

def bit_get(val, idx):
  return (val >> idx) & 1

def create_generic_colormap(n):
    colormap = np.zeros((n, 3), dtype=int)
    ind = np.arange(n, dtype=int)

    for shift in reversed(range(8)):
        for channel in range(3):
          colormap[:, channel] |= bit_get(ind, channel) << shift
        ind >>= 3
    return colormap
