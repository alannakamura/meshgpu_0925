from pymoo.problems import get_problem
import numpy as np
import pycuda.autoinit
from pycuda.compiler import SourceModule
import pycuda.driver as cuda
import matplotlib.pyplot as plt
from pygmo import fast_non_dominated_sorting
import pickle

f = open('wfg1_forca_bruta.pkl','wb')

nt = 1024
nb = 1
n = nb*nt

t = int(1e3)
out2 = np.array([[1e9, 1e9]])

for i in range(t):
    v = np.random.rand(n,12)

    for j in range(12):
        v[:,j] *= (j+1)*2

    v.shape = (n,12)
    # print(v)

    out = {}

    problem = get_problem('wfg1', n_var=12, n_obj=2)
    # print(problem)

    problem._evaluate(v, out)
    # print(out['F'])

    out2 = np.concatenate((out2, out['F']), axis=0)

    a, b, c, d = fast_non_dominated_sorting(points=out2)
    out2 = out2[a[0]]
    print((i+1)/t)

pickle.dump(out2, f)
f.close()
plt.plot(out2[:,0], out2[:,1],'ro')
plt.show()

# f = open('mesh.cu')
# code = f.read()
# f.close()
# mod = SourceModule(code, no_extern_c=True)
#
# func_n_g = (cuda.mem_alloc(np.array([1], dtype=np.int32).nbytes))
# cuda.memcpy_htod(func_n_g, np.array([21], dtype=np.int32))
#
# position_g = (cuda.mem_alloc(np.array(v, dtype=np.float64).nbytes))
# cuda.memcpy_htod(position_g, np.array(v.flatten(), dtype=np.float64))
#
# position_dim_g = (cuda.mem_alloc(np.array([1], dtype=np.int32).nbytes))
# cuda.memcpy_htod(position_dim_g, np.array([12], dtype=np.int32))
#
# fitness_g = (cuda.mem_alloc(np.array([0.0]*n*2, dtype=np.float64).nbytes))
# cuda.memcpy_htod(fitness_g, np.zeros(n*2, dtype=np.float64))
#
# alpha_g = (cuda.mem_alloc(np.array([1.0], dtype=np.float64).nbytes))
# cuda.memcpy_htod(alpha_g, np.array([1.0], dtype=np.float64))
#
# population_size_g = (cuda.mem_alloc(np.array([1], dtype=np.int32).nbytes))
# cuda.memcpy_htod(population_size_g, np.array([n], dtype=np.int32))
#
# function = mod.get_function("function1")
# function(func_n_g, position_g, position_dim_g, fitness_g, alpha_g, population_size_g,
#          block=(nt, 1, 1), grid=(nb, 1, 1))
# cuda.Context.synchronize()
#
# saida = np.zeros(n*2, dtype=np.float64)
# cuda.memcpy_dtoh(saida, fitness_g)
# print(saida)
#
# erro = out['F'].flatten()-saida
# print(min(erro), max(erro))
# print(erro)
# plt.plot(erro)
# plt.hist(erro)
# plt.show()
