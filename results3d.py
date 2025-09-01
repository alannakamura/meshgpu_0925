import pickle
import matplotlib.pyplot as plt
from pymoo.problems import get_problem
from pygmo import fast_non_dominated_sorting, hypervolume
from pymoo.util.ref_dirs import get_reference_directions
from pymoo.visualization.scatter import Scatter
import numpy as np
import pandas as pd
from optimisationMap import *

# name_file = 'results.pkl'if(diff == tam[0])
# name_file = 'results_1_100sim_30iter_128pop_3posdim_1.0alpha_3060_1.pkl'
# name_file = 'testes/220725/versao1/results_2_100sim_30iter_128pop_3posdim_1.0alpha_3060.pkl'
# name_file = 'testes/220725/versao2/results_2_100sim_30iter_128pop_3posdim_1.0alpha_3060.pkl'
# name_file = 'results_1_10sim_50iter_256pop_3posdim_1.0alpha_3060.pkl'
name_file = 'results_4_100sim_100iter_128pop_3posdim_1.0alpha_5070_Ti.pkl'

f = open(name_file, 'rb')
results = pickle.load(f)
f.close()

name_problem = optimisationMap[int(results['problem'])].lower()
pos_dim = int(results['pos_dim'])
sim = int(name_file.split('_')[2][:-3])
# sim = 28
tam_pop = int(name_file.split('_')[4][:-3])
print('sim', sim, 'tam_pop', tam_pop, 'pos_dim', pos_dim, 'name_problem', name_problem)

problem = get_problem(name_problem)

if name_problem in ['dtlz1', 'dtlz2', 'dtlz3', 'dtlz4']:
    ref_dirs = get_reference_directions("das-dennis", 3, n_partitions=100)
    pf_a = problem.pareto_front(ref_dirs)
else:
    pf_a = problem.pareto_front()


fit = []
fit2 = []
lenMem = []

figsize = (10, 10)
for i in range(sim):
    result = results[i]
    lenMem.append(result[2][0])
    fim = tam_pop*2 + result[2][0]
    fit.append(np.array(result[1][tam_pop * 2:fim]))

fit2 = fit[0]


for i in fit[1:]:
    fit2 = np.concatenate((fit2, i), axis=0)
    a, b, c, d = fast_non_dominated_sorting(points=fit2)
    fit2 = fit2[a[0]]
fit = fit2

x = np.max(fit[:, 0]) + 0.1
y = np.max(fit[:, 1]) + 0.1
z = np.max(fit[:, 2]) + 0.1
x2 = np.max(pf_a[:, 0]) + 0.1
y2 = np.max(pf_a[:, 1]) + 0.1
z2 = np.max(pf_a[:, 2]) + 0.1
# x3 = np.max(res2[:, 0]) + 0.1
# y3 = np.max(res2[:, 1]) + 0.1
# z3 = np.max(res2[:, 2]) + 0.1
# x = max(x, x2, x3)
# y = max(y, y2, y3)
# z = max(z, z2, z3)
x = max(x, x2)
y = max(y, y2)
z = max(z, z2)
ref = [x, y, z]
# ref = [2]*3
# ref[2] = 7
# ref = [60] * 3 #dtlz1
# ref = [91] * 3 #dtlz3

print('ref', ref)

hv = hypervolume(pf_a)
print('hypervolume_paretto', hv.compute(ref))

hv2 = hypervolume(fit)
print('hypervolume_mesh', hv2.compute(ref))

# hv3 = hypervolume(res2)
# print('hypervolume_nsga2', hv3.compute(ref))

# print('hypervolume_ratio_mp', hv2.compute(ref) / hv.compute(ref))
# print('hypervolume_ratio_np', hv3.compute(ref) / hv.compute(ref))
# print('hypervolume_ratio_mn', hv2.compute(ref) / hv3.compute(ref),'\n')
hypervolume_distance_mesh_pareto = abs(hv2.compute(ref) - hv.compute(ref))
print('hypervolume_distance_mesh_pareto', hypervolume_distance_mesh_pareto)
# hypervolume_distance_nsga2_pareto = abs(hv3.compute(ref) - hv.compute(ref))
# print('hypervolume_distance_nsga2_pareto', hypervolume_distance_nsga2_pareto)
# if hypervolume_distance_mesh_pareto < hypervolume_distance_nsga2_pareto:
#     print('melhor e MESH')
# elif hypervolume_distance_mesh_pareto > hypervolume_distance_nsga2_pareto:
#     print('melhor e NSGA2')
# else:
#     print('iguais')
# print('hypervolume_distance_mesh_nsga2', abs(hv2.compute(ref) - hv3.compute(ref)), '\n')

plt.figure()
plt.plot(pf_a[:, 0], pf_a[:, 1], 'ro', fit[:, 0], fit[:, 1], 'bo')
plt.title('after fast non dominating sorting MESH')
plt.xlabel('x')
plt.ylabel('y')
plt.legend(['paretto', 'GPU'])
# plt.axis([0, 1.5, 0, 1.5])
plt.show()

# if test_nsga2:
#     plt.figure()
#     plt.plot(pf_a[:, 0], pf_a[:, 1], 'ro', res2[:, 0], res2[:, 1], 'bo')
#     plt.title('after fast non dominating sorting NSGA2')
#     plt.xlabel('x')
#     plt.ylabel('y')
#     plt.legend(['paretto', 'GPU'])
#     plt.show()

plt.figure()
plt.plot(pf_a[:, 0], pf_a[:, 2], 'ro', fit[:, 0], fit[:, 2], 'bo')
plt.title('after fast non dominating sorting MESH')
plt.xlabel('x')
plt.ylabel('z')
plt.legend(['paretto', 'GPU'])
# plt.axis([0, 1.5, 0, 1.5])
plt.show()

# if test_nsga2:
#     plt.figure()
#     plt.plot(pf_a[:, 0], pf_a[:, 2], 'ro', res2[:, 0], res2[:, 2], 'bo')
#     plt.title('after fast non dominating sorting NSGA2')
#     plt.xlabel('x')
#     plt.ylabel('z')
#     plt.legend(['paretto', 'GPU'])
#     plt.show()

plt.figure()
plt.plot(pf_a[:, 1], pf_a[:, 2], 'ro', fit[:, 1], fit[:, 2], 'bo')
plt.title('after fast non dominating sorting MESH')
plt.xlabel('y')
plt.ylabel('z')
plt.legend(['paretto', 'GPU'])
# plt.axis([0, 1.5, 0, 1.5])
plt.show()

# if test_nsga2:
#     plt.figure()
#     plt.plot(pf_a[:, 1], pf_a[:, 2], 'ro', res2[:, 1], res2[:, 2], 'bo')
#     plt.title('after fast non dominating sorting NSGA2')
#     plt.xlabel('y')
#     plt.ylabel('z')
#     plt.legend(['paretto', 'GPU'])
#     plt.show()

s = Scatter(angle=(20, 20), title = 'MESH 3D')
s.add(pf_a, color='red')
# s.add(fit[a[0]], color = 'blue')
s.add(fit, color='blue')
s.show()

s = Scatter(angle=(20, 20), title = name_problem.upper()+' PARETO')
s.add(pf_a, color='red')
s.show()

s = Scatter(angle=(20, 20), title = name_problem.upper()+' MESH GPU')
s.add(fit, color='red')
s.show()

gpu = results['gpu']
plt.figure()
plt.title(name_problem.upper()+' GPU TIMES')
df = pd.DataFrame(gpu)
df.boxplot()
print('GPU\n', df.describe())
plt.show()

gpu2 = results['gpu2']
plt.figure()
plt.title(name_problem.upper()+' GPU2 TIMES')
df = pd.DataFrame(gpu2)
df.boxplot()
print('GPU\n', df.describe())
plt.show()

print(problem)