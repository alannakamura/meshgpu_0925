import pickle
from tqdm import tqdm
import os
import pycuda.driver as cuda

problem = 21
iter = list(range(100,1000+1,100))
for j in range(len(iter)):
    print('problem', problem)

    cuda.init()
    GPU = cuda.Device(0).name().split()
    GPU = '_'.join(GPU[3:])

    num = 1
    iterations = iter[j]
    population = 128
    pos_dim = 5
    f = open('results.pkl', 'wb')
    results = {'count': -1, 'cpu': [], 'gpu': [], 'problem': problem,
               'pos_dim': pos_dim, 'gpu2':[]}
    pickle.dump(results, f)
    f.close()

    for i in tqdm(range(num)):
        print('simulation ', i+1, (i + 1) / num * 100, '%')
        os.system("python run2.py "+str(problem) + ' ' +
                  str(iterations)+ ' ' + str(population) + ' '+ str(0.0)+ ' '+
                  str(pos_dim))

    os.rename('results.pkl', 'results_' + str(problem) + '_'
              + str(num) +'sim_'
              + str(iterations) +'iter_'
              + str(population) +'pop_'
              + str(pos_dim) +'posdim_'
              + GPU +'.pkl')


