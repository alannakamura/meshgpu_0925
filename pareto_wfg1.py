from pymoo.algorithms.moo.nsga2 import NSGA2
from pymoo.algorithms.moo.nsga3 import NSGA3
from pymoo.algorithms.moo.moead import MOEAD
from pymoo.algorithms.moo.rvea import RVEA
from pymoo.algorithms.moo.spea2 import SPEA2
from pymoo.util.ref_dirs import get_reference_directions
from pymoo.problems.many.wfg import WFG1
from pymoo.termination import get_termination
from pymoo.optimize import minimize
from pymoo.visualization.scatter import Scatter
import numpy as np
import pickle
import pygmo as pg
from matplotlib.pyplot import plot
from pygmo import hypervolume

l = []
dim = 2
n_var = 5
c = np.empty((0, 2))
f = open('wfg1.pkl', 'wb')

# Problema
problem = WFG1(n_var=n_var, n_obj=dim, k=4)
# problem.xl = np.zeros(n_var)
# problem.xu = np.array([2*(i+1) for i in range(n_var)])

pymoo_par = problem.pareto_front()
ref = 5,5
hv1 = hypervolume(pymoo_par)
res1 = hv1.compute(ref)
print(res1)

# Parada
termination = get_termination("n_gen", 30)

X = np.random.rand(128, n_var)  # None funciona pois só queremos amostrar
# Escala para [0, 2*(i+1)]
for j in range(n_var):
    X[:, j] = X[:, j] * 2 * (j+1)

s = Scatter(title="NSGA-II on WFG1")

# NSGA-2
for i in range(30):
    # Algoritmo
    algorithm = NSGA2(
        pop_size=128,
    )

    # Otimização
    res = minimize(
        problem,
        algorithm,
        termination,
        seed=i,
        verbose=True,
        X = X
    )

    l.append((res.F, res.X))
    s.add(l[-1][0])
    c = np.concatenate((c, res.F), axis=0)

# # MOEAD
# for i in range(30):
#     # Gerar os vetores de referência com método das divisões uniformes
#     ref_dirs = get_reference_directions("das-dennis", 2, n_partitions=99)
#     a = len(ref_dirs)
#
#     algorithm = MOEAD(
#         ref_dirs=ref_dirs,
#     )
#
#     res = minimize(
#         problem,
#         algorithm,
#         termination,
#         seed=i+100,
#         verbose=True
#     )
#
#     l.append((res.F, res.X))
#     s.add(l[-1][0])
#     c = np.concatenate((c, res.F), axis=0)
#     pass
#
# for i in range(30):
#     # Gerar os vetores de referência com método das divisões uniformes
#     ref_dirs = get_reference_directions("das-dennis", 2, n_partitions=99)
#
#     # Criar algoritmo NSGA-III
#     algorithm = NSGA3(
#         ref_dirs=ref_dirs,
#         pop_size=128,
#     )
#
#     res = minimize(
#         problem,
#         algorithm,
#         termination,
#         seed=i+200,
#         verbose=True
#     )
#
#     l.append((res.F, res.X))
#     s.add(l[-1][0])
#     c = np.concatenate((c, res.F), axis=0)
#     pass
#
# for i in range(30):
#
#     # Gerar 100 vetores de referência usando 99 partições
#     ref_dirs = get_reference_directions("das-dennis", 2, n_partitions=99)
#
#     # Configurar o algoritmo RVEA
#     algorithm = RVEA(
#         ref_dirs=ref_dirs,
#     )
#
#     # Executar otimização
#     res = minimize(
#         problem,
#         algorithm,
#         termination,
#         seed=i+300,
#         verbose=True
#     )
#
#     l.append((res.F, res.X))
#     s.add(l[-1][0])
#     c = np.concatenate((c, res.F), axis=0)
#     pass
#
# # SPEA-2
# for i in range(30):
#     # Algoritmo
#     algorithm = SPEA2(
#         pop_size=128,
#     )
#
#     # Otimização
#     res = minimize(
#         problem,
#         algorithm,
#         termination,
#         seed=i+400,
#         verbose=True
#     )
#
#     l.append((res.F, res.X))
#     s.add(l[-1][0])
#     c = np.concatenate((c, res.F), axis=0)
#     pass

fronts = pg.fast_non_dominated_sorting(points=c)[0]
c = c[fronts[0]]
s.add(c, marker='s',color='red')
s.add(pymoo_par, marker='s',color='blue')
pickle.dump(c, f)
f.close()

hv2 = hypervolume(c)
res2 = hv2.compute(ref)
print(res1)
print(res2)
print(abs(res2-res1))

# Plot

# s = Scatter(title="NSGA-II on WFG1")
# s.add(l[0][0], color='red', marker = 'o')
# s.add(l[1][0], color = 'blue', marker='x')
# s.add(l[2][0], color = 'black',marker='s')
# s.add(l[3][0], color='green', marker = 'o')
# s.add(l[4][0], color = 'magenta', marker='x')
# s.add(l[5][0], color = 'pink',marker='s')
# s.show()

# s.add(l[0][0], color='red', marker = 'o')
# s.add(l[1][0], color = 'blue', marker='x')
# s.add(l[2][0], color = 'black',marker='s')
# s.add(l[3][0], color='green', marker = 'o')
# s.add(l[4][0], color = 'magenta', marker='x')
# s.add(l[5][0], color = 'pink',marker='s')
# s.show()
