import pickle
from tqdm import tqdm
import os
from results2d import main

problem = 14
num = 100
iterations = 100

f = open('results2.pkl', 'wb')
results = {'count': -1, 'cpu': [], 'gpu': []}
pickle.dump(results, f)
f.close()

for i in tqdm(range(num)):
    print('simulation ', i+1, (i + 1) / num * 100, '%')
    os.system("python run2.py "+str(problem) + ' ' + str(iterations))
    # os.system("python run2.py " + str(problem))

# main(0, num)


