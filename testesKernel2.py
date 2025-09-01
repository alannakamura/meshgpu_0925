import pycuda.autoinit
import pycuda.driver as cuda
from pycuda.compiler import SourceModule
from pycuda.curandom import *
import cupy as cp
import numpy as np

f = open('mesh.cu')
code = f.read()
f.close()
mod = SourceModule(code, no_extern_c=True)

position_dim = 10
fitness_dim = 3
population_size = 10
problem = 1
alpha = 1.0

def vetor1(valor, dtype=np.float64):
    temp = np.array([valor], dtype=dtype)
    temp_g = (cuda.mem_alloc(temp.nbytes))
    cuda.memcpy_htod(temp_g, temp)
    return temp_g, temp

def vetorn_zeros(shape, dtype=np.float64):
    temp = np.zeros(shape, dtype= dtype)
    temp_g = cuda.mem_alloc(temp.nbytes)
    cuda.memcpy_htod(temp_g, temp)
    return temp_g, temp

def vetorn_random(shape, dtype=np.float64):
    temp = np.random.random(shape)
    temp_g = cuda.mem_alloc(temp.nbytes)
    cuda.memcpy_htod(temp_g, temp)
    return temp_g, temp

func_n_g, func_n = vetor1(problem, np.int32)
position_dim_g, position_dim = vetor1(position_dim, np.int32)
alpha_g, alpha = vetor1(alpha)
fitness_g, fitness = vetorn_zeros((population_size, fitness_dim))
position_g, position = vetorn_random((population_size, position_dim[0]))


cuda.memcpy_dtoh(fitness, fitness_g)
print(fitness)

function = mod.get_function("function")
function(func_n_g, position_g, position_dim_g,
         fitness_g, alpha_g, block=(population_size, 1, 1), grid=(1, 1, 1))

cuda.Context.synchronize()

cuda.memcpy_dtoh(fitness, fitness_g)
print(fitness)