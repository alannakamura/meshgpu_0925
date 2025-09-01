import pickle
from tqdm import tqdm
import os
import pycuda.driver as drv

# problem = 6
for problem in [1,2,3,4,5,6,7,11,12,13,14,15,16]:
    print('problem', problem)
    GPU = '3060'
    # GPU = '1050'
    # GPU = '4060'
    if problem == 1:
        num = 500
        iterations = 100
        population = 128
    elif problem == 2:
        num = 500
        iterations = 100
        population = 128
    elif problem == 3:
        num = 500
        iterations = 100
        population = 128
    elif problem == 4:
        num = 500
        iterations = 100
        population = 128
    elif problem == 5:
        num = 100
        iterations = 40
        population = 128
    elif problem == 6:
        num = 100
        iterations = 40
        population = 128
    elif problem == 7:
        num = 500
        iterations = 40
        population = 128
    elif problem == 11:
        num = 30
        iterations = 40
        population = 128
    elif problem == 12:
        num = 100
        iterations = 40
        population = 128
    elif problem == 13:
        num = 30
        iterations = 40
        population = 128
    elif problem == 14: # necessario para pegar os pontos proximos de f2=1
        num = 500
        iterations = 40
        population = 128
    elif problem == 15:
        num = 500
        iterations = 40
        population = 128
    elif problem == 16:
        num = 30
        iterations = 40
        population = 128
    else:
        num = -1
        iterations = -1

    f = open('results2.pkl', 'wb')
    results = {'count': -1, 'cpu': [], 'gpu': []}
    pickle.dump(results, f)
    f.close()

    for i in tqdm(range(num)):
        print('simulation ', i+1, (i + 1) / num * 100, '%')
        os.system("python run2.py "+str(problem) + ' ' + str(iterations)+ ' ' + str(population))
        # os.system("python run2.py " + str(problem))
    os.rename('results2.pkl', 'results_' + str(problem) + '_'
              + str(num) + 'sim_'
              + str(iterations) + 'iter_'
              + str(population) + 'pop_'
              + GPU + '.pkl')

# main(0, num)


