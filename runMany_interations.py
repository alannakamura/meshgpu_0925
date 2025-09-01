import pickle
from tqdm import tqdm
import os
import pycuda.driver as drv
import numpy as np

problem = 31
inicio = 30
fim = 100
interval = 10
iterations = np.arange(inicio, fim+1, interval)
alpha = [1.0]*len(iterations)
for j in range(len(iterations)):
    print('problem', problem)
    # GPU = '3060'

    # GPU = '1050'
    GPU = '3060'
    # GPU = '4060'

    num = 100
    # iterations = 40
    population = 128

    f = open('results.pkl', 'wb')
    results = {'count': -1, 'cpu': [], 'gpu': []}
    pickle.dump(results, f)
    f.close()

    for i in tqdm(range(num)):
        print('simulation ', i+1, (i + 1) / num * 100, '%')
        os.system("python run2.py "+str(problem) + ' ' + str(iterations[j])+ ' ' + str(population) + ' '+ str(alpha[j]))

    alpha2 = str(alpha[j]).split('.')
    os.rename('results.pkl', 'results_' + str(problem) + '_'
              + str(num) +'sim_'
              + str(iterations[j]) +'iter_'
              + str(population) +'pop_'
              + alpha2[0] + '.' + alpha2[1] +'alpha_'
              + GPU +'.pkl')

# main(0, num)


