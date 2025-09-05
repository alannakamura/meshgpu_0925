import pickle
import matplotlib.pyplot as plt
from pymoo.problems import get_problem
from pygmo import fast_non_dominated_sorting, hypervolume
import numpy as np
import pandas as pd
import datetime as dt
from optimisationMap import *
import pickle

# name_file = 'results.pkl'
name_file = 'results_311_100sim_100iter_128pop_3posdim_5070_Ti.pkl'

f = open(name_file, 'rb')
results = pickle.load(f)
f.close()

name_problem = optimisationMap[int(results['problem'])].lower()
pos_dim = int(results['pos_dim'])
sim = int(name_file.split('_')[2][:-3])
tam_pop = int(name_file.split('_')[4][:-3])
print('sim', sim, 'tam_pop', tam_pop, 'pos_dim', pos_dim, 'name_problem', name_problem)

if name_problem == 'wfg1':
    problem = get_problem(name_problem, n_var=pos_dim, n_obj=2)
else:
    problem = get_problem(name_problem, n_var = pos_dim)

if name_problem == 'wfg1':
    f = open('wfg1.pkl','rb')
    pf_b = pickle.load(f)
    pf_a = problem.pareto_front()
    f.close()
else:
    pf_a = problem.pareto_front()
# pf_a = problem.pareto_front(use_cache=False)

fit = []
lenMem = []

figsize = (10, 10)
for i in range(sim):
    result = results[i]
    lenMem.append(result[2][0])
    fim = tam_pop*2 + result[2][0]
    n_obj = result[-1][-1]
    fit.extend(result[1][tam_pop*2:fim][:, 0:n_obj])

fit = np.array(fit)

# plt.figure()
# plt.title('all memory points ')
# plt.plot(pf_a[:, 0], pf_a[:, 1], 'ro', fit[:, 0], fit[:, 1], 'bo')
# plt.legend(['paretto', 'GPU'])
# plt.show()

a, b, c, d = fast_non_dominated_sorting(points=fit)
fit = fit[a[0]]

x = np.max(fit[:, 0]) + 0.1
y = np.max(fit[:, 1]) + 0.1
x2 = np.max(pf_a[:, 0]) + 0.1
y2 = np.max(pf_a[:, 1]) + 0.1
x = float(max(x, x2))
y = float(max(y, y2))
ref = [x, y]
# ref = [2]*2

print('ref', ref)

if name_problem != 'wfg1':
    plt.figure()
    plt.plot(pf_a[:, 0], pf_a[:, 1], 'ro', fit[:, 0], fit[:, 1], 'bo')
    plt.title('after fast non dominating sorting')
    plt.legend(['pareto', 'GPU'])
    plt.show()
else:
    plt.figure()
    plt.plot(pf_a[:, 0], pf_a[:, 1], 'ro', pf_b[:, 0], pf_b[:, 1], 'mo', fit[:, 0], fit[:, 1], 'bo')
    plt.title('after fast non dominating sorting')
    plt.legend(['pareto pymoo', 'pareto teste', 'GPU'])
    plt.show()

plt.figure()
plt.plot(pf_a[:, 0], pf_a[:, 1], 'ro')
plt.title(name_problem.upper()+ ' PARETO')
plt.show()

plt.figure()
plt.plot(fit[:, 0], fit[:, 1], 'ro')
plt.title(name_problem.upper()+ ' MESH GPU')
plt.show()

hv = hypervolume(pf_a)
print('hypervolume_paretto', hv.compute(ref))

hv2 = hypervolume(fit)
print('hypervolume_gpu', hv2.compute(ref))

gpu_pareto = abs(hv.compute(ref) - hv2.compute(ref))
print('hypervolume_gpu_pareto', gpu_pareto)

# gpu = results['gpu']
# plt.figure()
# plt.title(name_problem.upper()+ ' GPU times')
# df = pd.DataFrame(gpu)
# df.boxplot()
# print('GPU\n', df.describe())
# plt.show()

gpu = results['gpu2']
plt.figure()
plt.title(name_problem.upper()+ ' GPU times')
df = pd.DataFrame(gpu)
df.boxplot()
print('GPU\n', df.describe())
plt.show()

print(problem)
print('total time in GPU: ', sum(gpu))
print('total time in GPU: ', str(dt.timedelta(seconds=sum(gpu)))[:-7])