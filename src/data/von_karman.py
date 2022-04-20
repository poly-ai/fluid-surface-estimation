import netCDF4
import numpy as np

def load_von_karman_dataset():
  dataset = netCDF4.Dataset('src/data/cylinder2d.nc')
  # print(dataset)
  # print(dataset['nu'][0])
  # print(dataset['radius'][0])
  # print(dataset['Re'][0])

  u = np.array(dataset['u']).swapaxes(1, 2) # (T, X, Y)
  v = np.array(dataset['v']).swapaxes(1, 2) # (T, X, Y)

  h = (u ** 2) + (v ** 2)

  return h
