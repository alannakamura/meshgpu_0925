import pickle
from tqdm import tqdm
import os
from results2d import main

# for problem in [1, 2, 3, 4, 5, 6, 7, 11, 12, 13, 14, 15, 16]:
for problem in [1, 2]:
    num = 1

    f = open('results2.pkl', 'wb')
    results = {'count': -1, 'cpu': [], 'gpu': []}
    pickle.dump(results, f)
    f.close()

    for i in tqdm(range(num)):
        print('simulation ', i+1, (i + 1) / num * 100, '%')
        os.system("python run2.py "+str(problem))
    os.rename('results2.pkl', 'results_' + str(problem) + '.pkl')

# main(0, num)


