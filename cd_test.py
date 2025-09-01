import numpy as np
from numpy import *

cd = zeros(8)
f = ones(24)

for i in range(1, 16):
    f[i] += f[i-1]+i
f.shape = 3,8
f = f.transpose()
f[:,2] -= 1
l = list(f[:,1])
l.reverse()
f[:,1]= l
print(f)

f = f[f[:,0].argsort()[::-1]]
print(f)

f[1:-1,2] += (f[2:,0] - f[0:-2,0])/(f[-1,0]-f[0,0])
f[0,2] = np.inf
f[-1,2]= np.inf
print(f)

f = f[f[:,1].argsort()[::-1]]
print(f)

f[1:-1,2] += (f[2:,1] - f[0:-2,1])/(f[-1,1]-f[0,1])
f[0,2] = np.inf
f[-1,2]= np.inf
print(f)