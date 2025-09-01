import pickle
import matplotlib.pyplot as plt
from pymoo.problems import get_problem
from pygmo import fast_non_dominated_sorting, hypervolume
from pymoo.util.ref_dirs import get_reference_directions
from pymoo.visualization.scatter import Scatter
import numpy as np
import pandas as pd

def main(numL, numC):
    ref_dirs = get_reference_directions("das-dennis", 3, n_partitions=100)
    pf_a = get_problem("dtlz1").pareto_front(ref_dirs)

    f = open('E2V2D1C2_DTLZ1_3obj.pkl', 'rb')

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

    a, b, c, d = fast_non_dominated_sorting(points=fit)

    x = np.max(fit[a[0]][:, 0]) + 0.1
    y = np.max(fit[a[0]][:, 1]) + 0.1
    z = np.max(fit[a[0]][:, 2]) + 0.1
    ref = [x, y, z]
    hv = hypervolume(fit[a[0]])
    print('hypervolume', hv.compute(ref))
    print('ref', ref)

    plt.figure()
    plt.plot(pf_a[:, 0], pf_a[:, 1], 'ro', fit[a[0], 0], fit[a[0], 1], 'bo')
    plt.title('after fast non dominating sorting')
    plt.xlabel('x')
    plt.ylabel('y')
    plt.legend(['paretto', 'GPU'])
    plt.show()

    plt.figure()
    plt.plot(pf_a[:, 0], pf_a[:, 2], 'ro', fit[a[0], 0], fit[a[0], 2], 'bo')
    plt.title('after fast non dominating sorting')
    plt.xlabel('x')
    plt.ylabel('z')
    plt.legend(['paretto', 'GPU'])
    plt.show()

    plt.figure()
    plt.plot(pf_a[:, 1], pf_a[:, 2], 'ro', fit[a[0], 1], fit[a[0], 2], 'bo')
    plt.title('after fast non dominating sorting')
    plt.xlabel('y')
    plt.ylabel('z')
    plt.legend(['paretto', 'GPU'])
    plt.show()

    # gpu = results['times']
    #
    # gpu = results['gpu']
    # plt.figure()
    # plt.title('GPU')
    # df = pd.DataFrame(gpu)
    # df.boxplot()
    # print('GPU\n', df.describe())
    # plt.show()
    # print('total time in GPU: ', sum(gpu))
    # print('total time in GPU: ', str(dt.timedelta(seconds=sum(gpu)))[:-7])

    s = Scatter(angle=(20, 20), title='3D')
    s.add(pf_a, color='red')
    s.add(fit[a[0]], color='blue')
    s.show()

if __name__ == '__main__':
    numL = 5
    numC = 6
    main(numL, numC)