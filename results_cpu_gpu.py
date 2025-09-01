import pickle
import matplotlib.pyplot as plt
from pymoo.problems import get_problem
from pygmo import fast_non_dominated_sorting, hypervolume
import numpy as np
import pandas as pd

# f = open('results_zdt1.pkl', 'rb')
f = open('testes/130824/results_zdt1_3060.pkl', 'rb')
# f = open('results_zdt2.pkl', 'rb')
# f = open('results_zdt2_3060.pkl', 'rb')
# f = open('results_zdt3.pkl', 'rb')
# f = open('results_zdt3_3060.pkl', 'rb')
# f = open('results_dtlz1_3060_500sim.pkl', 'rb')
results_gpu = pickle.load(f)
f.close()

f = open('E2V2D1C2_ZDT1_2obj.pkl', 'rb')
# f = open('E2V2D1C2_ZDT2_2obj.pkl', 'rb')
# f = open('E2V2D1C2_ZDT3_2obj.pkl', 'rb')
results_cpu = pickle.load(f)
f.close()

figsize = (10, 10)
gpu = results_gpu['gpu']
plt.subplot(1,2,1)
plt.title('GPU')
df = pd.DataFrame(gpu)
df.boxplot()
print('GPU\n', df.describe())
print('total GPU', sum(gpu))

cpu = results_cpu['times']
plt.subplot(1,2,2)
plt.title('CPU')
df = pd.DataFrame(cpu)
df.boxplot()
print('CPU\n', df.describe())
print('total CPU', sum(cpu))
plt.show()

# plt.subplot(1,3,3)
# plt.title('ratio GPU/CPU')
# df = pd.DataFrame(np.array(gpu)/np.array(cpu))
# df.boxplot()
# plt.show()



