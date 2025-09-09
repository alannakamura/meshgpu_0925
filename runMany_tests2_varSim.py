import pickle
from tqdm import tqdm
import os
import pycuda.driver as cuda

problem = 21
num = list(range(10,101,10))
for j in range(len(num)):
    print('problem', problem)

    cuda.init()
    GPU = cuda.Device(0).name().split()
    GPU = '_'.join(GPU[3:])

    iterations = 100
    population = 128
    pos_dim = 5
    f = open('results.pkl', 'wb')
    results = {'count': -1, 'cpu': [], 'gpu': [], 'problem': problem,
               'pos_dim': pos_dim, 'gpu2':[]}
    pickle.dump(results, f)
    f.close()

    for i in tqdm(range(num[j])):
        print('simulation ', i+1, (i + 1) / num[j] * 100, '%')
        os.system("python run2.py "+str(problem) + ' ' +
                  str(iterations)+ ' ' + str(population) + ' '+ str(0.0)+ ' '+
                  str(pos_dim))

    alpha2 = str(0.0).split('.')
    os.rename('results.pkl', 'results_' + str(problem) + '_'
              + str(num[j]) +'sim_'
              + str(iterations) +'iter_'
              + str(population) +'pop_'
              + str(pos_dim) +'posdim_'
              # + alpha2[0] + '.' + alpha2[1] +'alpha_'
              + GPU +'.pkl')


