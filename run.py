###########################################################################
# Lucas Braga, MS.c. (email: lucas.braga.deo@gmail.com )
# Gabriel Matos Leite, PhD candidate (email: gmatos@cos.ufrj.br)
# Carolina Marcelino, PhD (email: carolimarc@ic.ufrj.br)
# June 16, 2021
###########################################################################
# Copyright (c) 2021, Lucas Braga, Gabriel Matos Leite, Carolina Marcelino
# All rights reserved.
#
# Redistribution and use in source and binary forms, with or without
# modification, are permitted provided that the following conditions are
# met:
#
#    * Redistributions of source code must retain the above copyright
#      notice, this list of conditions and the following disclaimer.
#    * Redistributions in binary form must reproduce the above copyright
#      notice, this list of conditions and the following disclaimer in
#      the documentation and/or other materials provided with the
#      distribution
#
# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS USING 
# THE CREATIVE COMMONS LICENSE: CC BY-NC-ND "AS IS"
# AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE 
# IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE
# ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT OWNER OR CONTRIBUTORS BE
# LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR
# CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF
# SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS
# INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN
# CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE)
# ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE
# POSSIBILITY OF SUCH DAMAGE.

import datetime
from objectives import *
# from anteriores.MESH import *
from MESH import *
import pygmo as pg
import pickle
from tqdm import tqdm
from pathlib import Path
import pandas as pd
import matplotlib.pyplot as plt

optimizationMap = {
    1: 'DTLZ1',
    2: 'DTLZ2',
    3: 'DTLZ3',
    4: 'DTLZ4',
    5: 'DTLZ5',
    6: 'DTLZ6',
    7: 'DTLZ7',
    11: 'ZDT1',
    12: 'ZDT2',
    13: 'ZDT3',
    14: 'ZDT4',
    15: 'ZDT5',
    16: 'ZDT6',
    21: 'WFG1',
    22: 'WFG2',
    23: 'WFG3',
    24: 'WFG4',
    25: 'WFG5',
    26: 'WFG6',
    27: 'WFG7',
    28: 'WFG8',
    29: 'WFG9',
    30: 'f'
}


def main(f):
    Path("result").mkdir(parents=False, exist_ok=True)
    
    # for func_n in [1, 2, 3, 5, 6, 7]:
    for func_n in [11]:
        # num_runs = 30
        num_runs = 30
        # objectives_dim = 3
        objectives_dim = 2
        func = get_function(func_n, objectives_dim)

        otimizations_type = [False] * objectives_dim
        # otimizations_type[1] = True
        max_iterations = 0
        # max_fitness_eval = 15000
        max_fitness_eval = 15000
        # position_dim = 10
        position_dim = 10
        position_max_value = [1] * position_dim
        position_min_value = [0] * position_dim
        # population_size = 100
        population_size = 128
        num_final_solutions = 128  # aparece no final
        memory_size = 5
        memory_update_type = 0

        communication_probability = 0.7 # 0.5
        mutation_rate = 0.9
        personal_guide_array_size = 3

        global_best_attribution_type = 1     # 0 -> E1 | 1 -> E2 | 2 -> E3 | 3 -> E4
        Xr_pool_type = 1                  # 0 ->  V1 | 1 -> V2 | 2 -> V3
        DE_mutation_type = 0        # 0 -> DE\rand\1\Bin | 1 -> DE\rand\2\Bin | 2 -> DE/Best/1/Bin | 3 -> DE/Current-to-best/1/Bin | 4 -> DE/Current-to-rand/1/Bin
        crowd_distance_type = 1     # 0 -> Crowding Distance Tradicional | 1 -> Crowding Distance Suganthan


        config = f"E{global_best_attribution_type + 1}V{Xr_pool_type + 1}D{DE_mutation_type + 1}C{crowd_distance_type+1}"
        
        print(f"Running E{global_best_attribution_type+1}V{Xr_pool_type+1}D{DE_mutation_type+1}C{crowd_distance_type+1} on {optimizationMap[func_n]}")

        result = {}
        combined = None
        cpu_l = []
        gpu_l = []
        cpu2 = []
        gpu2 = []
        cpu_t = 0
        gpu_t = 0
        for i in tqdm(range(num_runs)):
            # params = MESH_Params(objectives_dim, otimizations_type, max_iterations,
            #                      max_fitness_eval, position_dim, position_max_value,
            #                      position_min_value, population_size, memory_size,
            #                      memory_update_type, global_best_attribution_type,
            #                      DE_mutation_type, Xr_pool_type, crowd_distance_type,
            #                      communication_probability, mutation_rate, personal_guide_array_size)
            params = MESH_Params(objectives_dim, otimizations_type, max_iterations,
                                 max_fitness_eval, position_dim, position_max_value,
                                 position_min_value, population_size, memory_size,
                                 memory_update_type, global_best_attribution_type,
                                 DE_mutation_type, Xr_pool_type, crowd_distance_type,
                                 communication_probability, mutation_rate, personal_guide_array_size,
                                 gpu=True)
            # MCDEEPSO = MESH(params, func)
            MCDEEPSO = MESH(params, func, gpu=True)
            MCDEEPSO.log_memory = f"result/{config}C1_{i}-{optimizationMap[func_n]}-{objectives_dim}obj-"
            # np.random.seed(0)
            # MCDEEPSO.run()
            print(str((i+1)/num_runs*100)+'%', i)
            cpu, gpu = MCDEEPSO.run()
            cpu_l.append((sum(cpu), cpu))
            gpu_l.append((sum(gpu), gpu))
            cpu2.append(sum(cpu))
            gpu2.append(sum(gpu))
            cpu_t += sum(cpu)
            gpu_t += sum(gpu)

            F = open(MCDEEPSO.log_memory+"fit.txt", 'r').read().split("\n")[-2]
            F = np.array([v.split() for v in F.split(",")], dtype=np.float64)

            P = open(MCDEEPSO.log_memory+"pos.txt", 'r').read().split("\n")[-2]
            P = np.array([v.split() for v in P.split(",")], dtype=np.float64)

            result[i+1] = {"F": F, "P": P}
            if combined is None:
                combined = F
            else:
                combined = np.vstack((combined, F))
            time = datetime.datetime.now() - start
            print(func_n, i, time)
            f.write('func_n= '+str(func_n)+' i= '+ str(i)+' time= '+str(time)+' \n')

        print('total cpu', cpu_t, 'total_gpu', gpu_t)
        # print(cpu2, gpu2)

        graficos = True
        if graficos:
            plt.subplot(1, 3, 1)
            plt.title('CPU')
            df = pd.DataFrame(cpu2)
            df.boxplot()
            print('CPU\n', df.describe())
            # print('cpu')

            plt.subplot(1, 3, 2)
            plt.title('GPU')
            df = pd.DataFrame(gpu2)
            df.boxplot()
            print('GPU\n', df.describe())
            # plt.show()

            plt.subplot(1, 3, 3)
            plt.title('GPU/CPU ratio')
            df = pd.DataFrame(np.array(gpu2)/np.array(cpu2)*100)
            df.boxplot()
            print('GPU\n', df.describe())
            plt.show()

        ndf, dl, dc, ndr = pg.fast_non_dominated_sorting(points=combined)
        n = num_final_solutions
        if len(ndf[0]) < num_final_solutions:
            n = len(ndf[0])
        best_idx = pg.sort_population_mo(points=combined)[:n]

        result['combined'] = (best_idx, combined[best_idx])

        with open(f'result/{config}_{optimizationMap[func_n]}_{objectives_dim}obj.pkl', 'wb') as f:
            pickle.dump(result, f)


if __name__ == '__main__':
    f = open('times.txt', 'w')
    start = datetime.datetime.now()
    main(f)
    time = datetime.datetime.now() - start
    f.write('final '+str(time)+' \n')
    f.close()
