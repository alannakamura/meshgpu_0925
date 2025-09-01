import pycuda.autoinit
import pycuda.driver as cuda
from pycuda.compiler import SourceModule
from pycuda.curandom import *
import cupy as cp

f = open('mesh.cu')
code = f.read()
f.close()
mod = SourceModule(code, no_extern_c=True)

func_n_g = cp.zeros(1, dtype=cp.int32)
func_n_g[0] = 1
func_n_g_ptr = func_n_g.data.ptr

position_g = cp.random.random((128,10), dtype=cp.float64)
position_g_ptr = position_g.data.ptr

fitness_g = cp.zeros((128,3), dtype=cp.float64)
fitness_g_ptr = fitness_g.data.ptr

position_dim_g = cp.zeros(1, dtype=cp.int32)
position_dim_g[0] = 10
position_dim_g_ptr = position_dim_g.data.ptr

alpha_g = cp.zeros(1, dtype=cp.float64)
alpha_g[0] = 1.0
alpha_g_ptr = alpha_g.data.ptr

population_size = 128

function = mod.get_function("function")
function(func_n_g_ptr, position_g_ptr, position_dim_g_ptr,
         fitness_g_ptr, alpha_g_ptr, block=(128, 1, 1), grid=(1, 1, 1))

cuda.Context.synchronize()