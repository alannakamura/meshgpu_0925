import pickle
from tqdm import tqdm
import os
import pycuda.driver as cuda

problem = [21]
# problem = [31,32,33,35,36,37,39,310,311]
# problem = [11,12,13,14,16]
# problem = [4,7,1,2,3,5,6]
#problem = [31, 32, 33, 35, 36, 37]
alpha = [2.0]*len(problem)
for j in range(len(problem)):
    print('problem', problem[j])

    cuda.init()
    GPU = cuda.Device(0).name().split()
    GPU = '_'.join(GPU[3:])

    num = 100
    iterations = 300
    population = 128
    pos_dim = 3
    f = open('results.pkl', 'wb')
    results = {'count': -1, 'cpu': [], 'gpu': [], 'problem': problem[j],
               'pos_dim': pos_dim, 'gpu2':[]}
    pickle.dump(results, f)
    f.close()

    for i in tqdm(range(num)):
        print('simulation ', i+1, (i + 1) / num * 100, '%')
        os.system("python run2.py "+str(problem[j]) + ' ' +
                  str(iterations)+ ' ' + str(population) + ' '+ str(alpha[j])+ ' '+
                  str(pos_dim))

    alpha2 = str(alpha[j]).split('.')
    os.rename('results.pkl', 'results_' + str(problem[j]) + '_'
              + str(num) +'sim_'
              + str(iterations) +'iter_'
              + str(population) +'pop_'
              + str(pos_dim) +'posdim_'
              # + alpha2[0] + '.' + alpha2[1] +'alpha_'
              + GPU +'.pkl')


