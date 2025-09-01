import pickle
from matplotlib.pyplot import *
from pymoo.problems.many.wfg import WFG1
from pygmo import hypervolume
from optimisationMap import *
from pygmo import fast_non_dominated_sorting, hypervolume

# manual
f = open('wfg1.pkl', 'rb')
res_manual = pickle.load(f)
f.close()

f = open('wfg1_forca_bruta.pkl', 'rb')
res_fb = pickle.load(f)
f.close()

# pareto
dim = 2
problem = WFG1(n_var=12, n_obj=dim, k=4)
pymoo_par = problem.pareto_front()

# meu programa
name_file = 'testes/250825/results_21_100sim_100iter_128pop_12posdim_1.0alpha_4060.pkl'
f = open(name_file, 'rb')
results = pickle.load(f)
f.close()

name_problem = optimisationMap[int(results['problem'])].lower()
pos_dim = int(results['pos_dim'])
sim = int(name_file.split('_')[2][:-3])
tam_pop = int(name_file.split('_')[4][:-3])
print('sim', sim, 'tam_pop', tam_pop, 'pos_dim', pos_dim, 'name_problem', name_problem)

fit = []
lenMem = []

figsize = (10, 10)
for i in range(sim):
    result = results[i]
    lenMem.append(result[2][0])
    fim = tam_pop*2 + result[2][0]
    fit.extend(result[1][tam_pop*2:fim])

fit = np.array(fit)

a, b, c, d = fast_non_dominated_sorting(points=fit)
fit = fit[a[0]]

# figure()
# plot(pf_a[:, 0], pf_a[:, 1], 'ro', fit[:, 0], fit[:, 1], 'bo')
# title('after fast non dominating sorting')
# legend(['pareto', 'GPU'])
# show()




plot(res_manual[:,0], res_manual[:,1],'ro', pymoo_par[:,0], pymoo_par[:,1], 'bo',
     fit[:,0], fit[:,1],'go', res_fb[:, 0], res_fb[:, 1],'mo')
legend(['manual', 'pymoo', 'MESH GPU','forca_bruta'])

# ref = 5,5
# hv1 = hypervolume(pymoo_par)
# res1 = hv1.compute(ref)
# print('hypervolume pymoo', res1)
#
# hv2 = hypervolume(a)
# res2 = hv2.compute(ref)
# print('hypervolume manual', res2)
#
# print('difference', abs(res2-res1))

show()