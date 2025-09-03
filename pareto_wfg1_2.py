from pymoo.algorithms.moo.nsga2 import NSGA2
from pymoo.problems.many.wfg import WFG1
from pymoo.termination import get_termination
from pymoo.optimize import minimize
from pymoo.visualization.scatter import Scatter
import numpy as np
import pickle
import pygmo as pg
from matplotlib.pyplot import plot, show, legend
from pygmo import hypervolume

l = []
dim = 2
n_var = 5
c = np.empty((0, 2))
f = open('wfg1.pkl', 'wb')

problem = WFG1(n_var=n_var, n_obj=dim, k=4)

pymoo_par = problem.pareto_front()
ref = 5,5
hv1 = hypervolume(pymoo_par)
res1 = hv1.compute(ref)
print(res1)

# Parada
termination = get_termination("n_gen", 300)

X = np.random.rand(128, n_var)  # None funciona pois só queremos amostrar
for j in range(n_var):
    X[:, j] = X[:, j] * 2 * (j+1)

s = Scatter(title="NSGA-II on WFG1")

# NSGA-2
for i in range(30):

    algorithm = NSGA2(
        pop_size=128,
        # sampling= X
    )

    res = minimize(
        problem,
        algorithm,
        termination,
        seed=i,
        verbose=True
        # X = X
    )

    l.append((res.F, res.X))
    s.add(l[-1][0])
    c = np.concatenate((c, res.F), axis=0)

fronts = pg.fast_non_dominated_sorting(points=c)[0]
c = c[fronts[0]]
pickle.dump(c, f)
f.close()

hv2 = hypervolume(c)
res2 = hv2.compute(ref)
print(res1)
print(res2)
print(abs(res2-res1))

plot(pymoo_par[:,0], pymoo_par[:, 1], 'ro', c[:,0], c[:,1], 'bo')
legend(['pymoo','manual'])
show()
