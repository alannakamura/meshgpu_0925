import pickle
import matplotlib.pyplot as plt
from pymoo.problems import get_problem
from pygmo import fast_non_dominated_sorting, hypervolume
import numpy as np
import pandas as pd
import datetime as dt
from pymoo.algorithms.moo.nsga2 import NSGA2
from pymoo.optimize import minimize


def main(numL, numC):
    # f = get_problem("zdt1")
    # f = get_problem("zdt2")
    # f = get_problem("zdt3")
    # f = get_problem("zdt4")
    # f = get_problem("zdt5", normalize=False)
    # f = get_problem("zdt6")
    problem = get_problem("mw7", n_var=10, n_obj=2)
    pf_a = problem.pareto_front()
    # pf_a = problem.pareto_front(use_cache=False)

    # dir = 'testes/mw/301024/'
    dire = ''
    # name = 'results'
    name = 'results_37_100sim_100iter_128pop_-1.0alpha_4060'

    if name == 'results':
        sim = 30
        itera = 40
        pop = 128
    else:
        sim = int(name.split('_')[2][:-3])
        itera = int(name.split('_')[3][:-4])
        pop = int(name.split('_')[4][:-3])
    f = open(dire+name+'.pkl', 'rb')
    results = pickle.load(f)
    f.close()

    pos = []
    fit = []
    lenMem = []

    figsize = (10, 10)
    if numL > 0:
        for i in range(numL):
            for j in range(numC):
                numT = i*numC+j
                result = results[numT]
                lenMem.append(result[2][0])

                fim = 256 + result[2][0]
                fit.extend(result[1][256:fim])

                if numL * numC <= lim:
                    plt.subplot(numL, numC, numT + 1)
                    plt.plot(pf_a[:, 0], pf_a[:, 1], 'ro', result[1][256:fim, 0],
                         result[1][256:fim, 1], 'bo')
        plt.show()
    else:
        for i in range(numC):
            result = results[i]
            lenMem.append(result[2][0])
            fit.extend(result[1][256:])
            fim = 256 + result[2][0]
            a, b, c, d = fast_non_dominated_sorting(points=result[1][256:])

    f.close()
    fit = np.array(fit)

    # teste nsga2
    # sim = 30
    res2 = []
    print(problem)
    for i in range(sim):
        print(i)
        algorithm = NSGA2(pop_size=pop)
        res = minimize(problem,
                       algorithm,
                       ('n_gen', itera),
                       seed=i,
                       verbose=False)
        try:
            res2.extend(res.F.tolist())
        except:
            print(res.F)


    res2 = np.array(res2)

    plt.figure()
    plt.title('all memory points ')
    plt.plot(pf_a[:, 0], pf_a[:, 1], 'ro', fit[:, 0], fit[:, 1], 'bo', res2[:, 0], res2[:, 1], 'go')
    plt.legend(['paretto', 'GPU'])
    plt.show()

    a, b, c, d = fast_non_dominated_sorting(points=fit)

    x = np.max(fit[a[0], 0]) + 0.1
    y = np.max(fit[a[0], 1]) + 0.1
    x2 = np.max(pf_a[:, 0]) + 0.1
    y2 = np.max(pf_a[:, 1]) + 0.1
    x3 = np.max(res2[:, 0]) + 0.1
    y3 = np.max(res2[:, 1]) + 0.1
    x = max(x, x2, x3)
    y = max(y, y2, y3)
    ref = [x, y]

    print('ref', ref)

    plt.figure()
    plt.plot(pf_a[:, 0], pf_a[:, 1], 'ro', fit[a[0], 0], fit[a[0], 1], 'bo', res2[:, 0], res2[:, 1], 'go')
    # plt.plot(pf_a[:, 0], pf_a[:, 1], 'ro', fit[a[0], 0], fit[a[0], 1], 'bo')
    plt.title('after fast non dominating sorting')

    hv = hypervolume(pf_a)
    print('hypervolume_paretto', hv.compute(ref))

    hv2 = hypervolume(fit[a[0]])
    print('hypervolume_gpu', hv2.compute(ref))

    hv3 = hypervolume(res2)
    print('hypervolume_nsga2', hv3.compute(ref))

    gpu_pareto = abs(hv.compute(ref) - hv2.compute(ref))
    print('hypervolume_gpu_pareto', gpu_pareto)
    nsga2_pareto = abs(hv.compute(ref) - hv3.compute(ref))
    print('hypervolume_nsga2_pareto', nsga2_pareto)
    if gpu_pareto < nsga2_pareto:
        print('mesh e melhor')
    elif gpu_pareto > nsga2_pareto:
        print('nsga2 e melhor')

    print('hypervolume_gpu_nsga2', abs(hv2.compute(ref) - hv3.compute(ref)))

    # plt.axis([0, 20, -0.1, 1.1])
    plt.legend(['paretto', 'GPU'])
    plt.show()

    # gpu = results['gpu']
    # plt.figure()
    # plt.title('GPU')
    # df = pd.DataFrame(gpu)
    # df.boxplot()
    # print('GPU\n', df.describe())
    # plt.show()
    #
    # print('total time in GPU: ', sum(gpu))
    # print('total time in GPU: ', str(dt.timedelta(seconds=sum(gpu)))[:-7])


if __name__ == '__main__':
    numL = 30
    numC = 1
    lim = 2
    main(numL, numC)