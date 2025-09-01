import pickle
import matplotlib.pyplot as plt
from pymoo.problems import get_problem
from pygmo import fast_non_dominated_sorting, hypervolume
import numpy as np
import pandas as pd

def main(numL, numC):
    # f = get_problem("zdt1")
    # f = get_problem("zdt2")
    f = get_problem("zdt3")
    # f = get_problem("zdt4")
    pf_a = f.pareto_front(use_cache=False)

    # f = open('E2V2D1C2_ZDT1_2obj.pkl', 'rb')
    # f = open('E2V2D1C2_ZDT2_2obj.pkl', 'rb')
    # f = open('E2V2D1C2_ZDT3_2obj.pkl', 'rb')

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
                plt.subplot(numL, numC, numT+1)
                result = results[numT+1]
                # lenMem.append(result[2][0])

                fit.extend(result['F'])
                # fim = 256+result[2][0]
                plt.plot(pf_a[:, 0], pf_a[:, 1], 'bo', result['F'][:, 0], result['F'][:, 1], 'ro')
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

    plt.figure()
    plt.title('all memory points ')
    plt.plot(pf_a[:, 0], pf_a[:, 1]+1e-2, 'bo', fit[:, 0], fit[:, 1], 'ro')
    plt.show()

    a, b, c, d = fast_non_dominated_sorting(points=fit)

    ref = [2, 2]
    hv = hypervolume(fit[a[0]])
    print('hypervolume', hv.compute(ref))

    plt.figure()
    plt.plot(pf_a[:, 0], pf_a[:, 1]+1e-2, 'bo', fit[a[0], 0], fit[a[0], 1], 'ro')
    plt.title('after fast non dominating sorting')
    plt.show()

    # plt.figure()
    # plt.plot(fit[a[0], 0], fit[a[0], 1], 'ro')
    # plt.title('only gpu results')
    # plt.show()

    gpu = results['times']

    plt.title('GPU')
    df = pd.DataFrame(gpu)
    df.boxplot()
    print('GPU\n', df.describe())
    plt.show()

    print('total GPU', sum(gpu))

if __name__ == '__main__':
    numL = 5
    numC = 6
    main(numL, numC)