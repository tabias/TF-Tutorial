import numpy as np

xy = np.loadtxt('train.txt', unpack=True, dtype = 'float32')
x = xy[:,2]
y = xy[2:3]

print(x)