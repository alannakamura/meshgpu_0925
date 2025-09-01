import pickle
import numpy as np
import matplotlib.pyplot as plt
from pymoo.problems import get_problem

f = get_problem("zdt1")
pf_a = f.pareto_front(use_cache=False)

f = open('testesSimulacao/results_teste250624_rtx3060_mem128_rand_60sim.pkl', 'rb')
num = 10

for i in range(num):
    plt.subplot(1, num, i+1)
    l = pickle.load(f)
    fit = l[1]
    # plt.plot(pf_a[:, 0], pf_a[:, 1], 'ro', fit[256:, 0], fit[256:, 1], 'bo')
    plt.plot(fit[256:, 0], fit[256:, 1], 'bo')
    p = l[0]
plt.show()
f.close()



