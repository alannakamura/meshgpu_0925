import pickle
from tqdm import tqdm
import os
import pycuda.driver as drv
import numpy as np

problem = [33]
for alpha in np.linspace(0, 1, 21):
    for j in range(len(problem)):
    # for problem in [31, 32, 33, 34, 35, 36, 37]:
        print('problem', problem[j])
        GPU = '3060'
        # GPU = '1050'
        # GPU = '4060'

        num = 100
        iterations = 100
        population = 128

        f = open('results2.pkl', 'wb')
        results = {'count': -1, 'cpu': [], 'gpu': []}
        pickle.dump(results, f)
        f.close()

        for i in tqdm(range(num)):
            print('simulation ', i+1, (i + 1) / num * 100, '%')
            os.system("python run2.py "+str(problem[j]) + ' ' + str(iterations)+ ' ' + str(population) + ' '+ str(alpha))

        alpha2 = str(alpha).split('.')
        os.rename('results2.pkl', 'results_' + str(problem[j]) + '_'
                  + str(num) +'sim_'
                  + str(iterations) +'iter_'
                  + str(population) +'pop_'
                  + alpha2[0] + '.' + alpha2[1][:3] +'alpha_'
                  + GPU +'.pkl')

# main(0, num)


