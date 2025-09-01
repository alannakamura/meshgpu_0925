import pickle
import matplotlib.pyplot as plt
import pandas as pd
import datetime as dt
import numpy as np

dir_cpu = '/home/alan/PycharmProjects/mesh_cpu/testesDesempenho040924_10sim_20iter/'
# dir_cpu = 'result/'
dir_gpu = '//home//alan//PycharmProjects//MESH_FN//testes//testeDesempenho040924//'

f = open(dir_cpu+'E2V2D1C2_DTLZ1_3obj.pkl', 'rb')
res_cpu = pickle.load(f)
median_cpu = np.median(np.array(res_cpu['times']))
f.close()

f = open(dir_gpu+'results_1_10sim_20iter_1050.pkl', 'rb')
res_gpu = pickle.load(f)
median_gpu = np.median(np.array(res_gpu['gpu']))
f.close()

plt.figure()
plt.title('GPU/CPU: '+str(median_gpu/median_cpu*100)[:4]+'%')
# plt.subplot(1, 2, 1)
plt.title('CPU')
# df = pd.DataFrame(res_cpu['times'])
# df.boxplot()
plt.boxplot(res_cpu['times'])
# print('GPU\n', df.describe())
# plt.show()
print('total time in CPU: ', sum(res_cpu['times']))
print('total time in CPU: ', str(dt.timedelta(seconds=sum(res_cpu['times'])))[:-7])

plt.figure()
plt.title('GPU')
plt.boxplot(res_gpu['gpu'])

plt.figure()
plt.title('GPU/CPU')
plt.boxplot(np.array(res_gpu['gpu'])/np.array(res_cpu['times'])*100)

plt.figure()
plt.title('CPU/GPU')
plt.boxplot(np.array(res_cpu['times'])/np.array(res_gpu['gpu'])*100)
plt.show()

print('total time in GPU: ', sum(res_gpu['gpu']))
print('total time in GPU: ', str(dt.timedelta(seconds=sum(res_gpu['gpu'])))[:-7])

# f2 = open(dir_cpu+'E2V2D1C2_DTLZ1_3obj.pkl')

