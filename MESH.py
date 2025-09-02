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
import os
# import numpy as np
import sys
import copy

import numpy as np
from scipy.stats import truncnorm
from Particle import *
from tqdm import tqdm
import matplotlib.pyplot as plt
from pymoo.problems import get_problem
import pycuda.autoinit
import pycuda.driver as cuda
from pycuda.compiler import SourceModule
from pycuda.curandom import *
# import cupy as cp
import struct
from datetime import datetime as dt
import sys
import pickle
from pycuda.tools import clear_context_caches

# a = np.zeros(9)

class MESH_Params:
    def __init__(self,
                 objectives_dim,
                 otimizations_type,
                 max_iterations,
                 max_fitness_eval,
                 position_dim,
                 position_max_value,
                 position_min_value,
                 population_size,
                 memory_size,
                 memory_update_type,
                 global_best_attribution_type,
                 DE_mutation_type,
                 Xr_pool_type,
                 crowd_distance_type,
                 communication_probability,
                 mutation_rate,
                 personal_guide_array_size,
                 secondary_params=False,
                 initial_state=False,
                 restrictions_dim = 0,
                 func_n = 11):

        self.objectives_dim = np.array([objectives_dim+restrictions_dim, restrictions_dim, objectives_dim],
                                       dtype=np.int32)
        # self.restrictions_dim = restrictions_dim
        self.otimizations_type = otimizations_type

        self.max_iterations = max_iterations
        self.max_fitness_eval = max_fitness_eval

        self.position_dim = position_dim
        self.position_max_value = position_max_value
        self.position_min_value = position_min_value

        self.velocity_min_value = list()
        self.velocity_max_value = list()
        for i in range(position_dim):
            self.velocity_min_value.append(-1 * self.position_max_value[i] + self.position_min_value[i])
            self.velocity_max_value.append(-1 * self.velocity_min_value[i])

        self.population_size = population_size

        # self.memory_size = memory_size
        self.memory_size = np.array(memory_size, dtype=np.int32)
        self.memory_update_type = memory_update_type

        self.global_best_attribution_type = global_best_attribution_type

        self.DE_mutation_type = DE_mutation_type
        self.Xr_pool_type = Xr_pool_type
        self.crowd_distance_type = crowd_distance_type

        self.communication_probability = communication_probability
        self.mutation_rate = mutation_rate

        self.personal_guide_array_size = personal_guide_array_size

        self.secondary_params = secondary_params
        self.initial_state = initial_state

        self.func_n = func_n

        self.copy = np.array([0], dtype=np.int32)

        # alocacao dos parametros na gpu
        self.objectives_dim_g = (
            cuda.mem_alloc(self.objectives_dim.nbytes))
        cuda.memcpy_htod(self.objectives_dim_g,self.objectives_dim)

        # self.restrictions_dim_g = (
        #     cuda.mem_alloc(np.array(restrictions_dim, dtype=np.int32).nbytes))
        # cuda.memcpy_htod(self.restrictions_dim_g,
        #                  np.array(self.restrictions_dim, dtype=np.int32))

        self.otimizations_type_g = (
            cuda.mem_alloc(np.array(otimizations_type, dtype=np.int32).nbytes))
        cuda.memcpy_htod(self.otimizations_type_g,
                         np.array(self.otimizations_type, dtype=np.int32))

        self.max_iterations_g = (
            cuda.mem_alloc(np.array(max_iterations, dtype=np.int32).nbytes))
        cuda.memcpy_htod(self.max_iterations_g,
                         np.array(self.max_iterations, dtype=np.int32))
        self.max_fitness_eval_g = (
            cuda.mem_alloc(np.array(max_fitness_eval, dtype=np.int32).nbytes))
        cuda.memcpy_htod(self.max_fitness_eval_g,
                         np.array(self.max_fitness_eval, dtype=np.int32))

        self.position_dim_g = (
            cuda.mem_alloc(np.array(position_dim, dtype=np.int32).nbytes))
        cuda.memcpy_htod(self.position_dim_g,
                         np.array(self.position_dim, dtype=np.int32))

        self.position_max_value_g = (
            cuda.mem_alloc(np.array(position_max_value, dtype=np.float64).nbytes))
        cuda.memcpy_htod(self.position_max_value_g,
                         np.array(self.position_max_value, dtype=np.float64))

        self.position_min_value_g = (
            cuda.mem_alloc(np.array(position_min_value, dtype=np.float64).nbytes))
        cuda.memcpy_htod(self.position_min_value_g,
                         np.array(self.position_min_value, dtype=np.float64))

        self.velocity_min_value_g = (
            cuda.mem_alloc(np.array(self.velocity_min_value, dtype=np.float64).nbytes))
        cuda.memcpy_htod(self.velocity_min_value_g,
                         np.array(self.velocity_min_value, dtype=np.float64))

        self.velocity_max_value_g = (
            cuda.mem_alloc(np.array(self.velocity_max_value, dtype=np.float64).nbytes))
        cuda.memcpy_htod(self.velocity_max_value_g,
                         np.array(self.velocity_max_value, dtype=np.float64))

        self.population_size_g = (
            cuda.mem_alloc(np.array(self.population_size, dtype=np.int32).nbytes))
        cuda.memcpy_htod(self.population_size_g,
                         np.array(self.population_size, dtype=np.int32))

        self.memory_size_g = (
            cuda.mem_alloc(np.array(self.memory_size, dtype=np.int32).nbytes))
        # cuda.memcpy_htod(self.memory_size_g,
        #                  np.array(self.memory_size, dtype=np.int32))
        cuda.memcpy_htod(self.memory_size_g, self.memory_size)

        self.current_memory_size = 0

        self.current_memory_size_g = (
            cuda.mem_alloc(np.array(self.memory_size, dtype=np.int32).nbytes))
        cuda.memcpy_htod(self.current_memory_size_g,
                         np.array(-1, dtype=np.int32))

        self.memory_update_type_g = (
            cuda.mem_alloc(np.array(self.memory_update_type, dtype=np.int32).nbytes))
        cuda.memcpy_htod(self.memory_update_type_g,
                         np.array(self.memory_update_type, dtype=np.int32))

        self.global_best_attribution_type_g = (
            cuda.mem_alloc(np.array(self.global_best_attribution_type, dtype=np.int32).nbytes))
        cuda.memcpy_htod(self.global_best_attribution_type_g,
                         np.array(self.global_best_attribution_type, dtype=np.int32))

        self.DE_mutation_type_g = (
            cuda.mem_alloc(np.array(self.DE_mutation_type, dtype=np.int32).nbytes))
        cuda.memcpy_htod(self.DE_mutation_type_g,
                         np.array(self.DE_mutation_type, dtype=np.int32))

        self.Xr_pool_type_g = (
            cuda.mem_alloc(np.array(self.Xr_pool_type, dtype=np.int32).nbytes))
        cuda.memcpy_htod(self.Xr_pool_type_g,
                         np.array(self.Xr_pool_type, dtype=np.int32))

        self.crowd_distance_type_g = (
            cuda.mem_alloc(np.array(self.crowd_distance_type, dtype=np.int32).nbytes))
        cuda.memcpy_htod(self.crowd_distance_type_g,
                         np.array(self.crowd_distance_type, dtype=np.int32))

        self.communication_probability_g = (
            cuda.mem_alloc(np.array(self.communication_probability, dtype=np.float64).nbytes))
        cuda.memcpy_htod(self.communication_probability_g,
                         np.array(self.communication_probability, dtype=np.float64))

        self.mutation_rate_g = (
            cuda.mem_alloc(np.array(self.mutation_rate, dtype=np.float64).nbytes))
        cuda.memcpy_htod(self.mutation_rate_g,
                         np.array(self.mutation_rate, dtype=np.float64))

        self.personal_guide_array_size_g = (
            cuda.mem_alloc(np.array(self.personal_guide_array_size, dtype=np.int32).nbytes))
        cuda.memcpy_htod(self.personal_guide_array_size_g,
                         np.array(self.personal_guide_array_size, dtype=np.int32))

        self.secondary_params_g = (
            cuda.mem_alloc(np.array(self.secondary_params, dtype=np.int32).nbytes))
        cuda.memcpy_htod(self.secondary_params_g,
                         np.array(self.secondary_params, dtype=np.int32))

        self.initial_state_g = (
            cuda.mem_alloc(np.array(self.initial_state, dtype=np.int32).nbytes))
        cuda.memcpy_htod(self.initial_state_g,
                         np.array(self.initial_state, dtype=np.int32))

        self.func_n_g = (
            cuda.mem_alloc(np.array([0], dtype=np.int32).nbytes))
        cuda.memcpy_htod(self.func_n_g,np.array([self.func_n], dtype=np.int32))

        self.copy_g = cuda.mem_alloc(self.copy.nbytes)
        cuda.memcpy_htod(self.copy_g, self.copy)

class MESH:
    def __init__(self, params, fitness_function, max_num_iters=40, alpha=-1, gpu=True):
        self.params = params
        self.stopping_criteria_reached = False
        self.generation_count = 0

        self.population = []
        self.population_copy = []

        self.memory = []
        self.fronts = []

        self.fitness_function = fitness_function
        self.fitness_eval_count = 0

        self.max_num_iters = max_num_iters
        self.alpha = alpha

        # np.random.seed(0)
        w1 = np.random.uniform(0.0, 1.0, [4, self.params.population_size])
        w2 = np.random.uniform(0.0, 0.5, [1, self.params.population_size])
        w3 = np.random.uniform(0.0, 2.0, [1, self.params.population_size])

        self.weights = np.concatenate((w1, w2, w3), axis=0)
        self.weights_copy = []

        self.update_from_differential_mutation = False
        self.log_memory = False
        self.copy_pop = True
        self.gpu = gpu

        if gpu:
            total = self.params.population_size * 2 + self.params.memory_size

            # self.position = (
            #     cuda.mem_alloc(np.zeros(total * self.params.position_dim,
            #                             dtype=np.float64).nbytes))
            #
            # self.fitness = (
            #     cuda.mem_alloc(np.zeros(total * self.params.objectives_dim,
            #                             dtype=np.float64).nbytes))

            self.alpha_g = (
                cuda.mem_alloc(np.array(self.alpha, dtype=np.float64).nbytes))
            cuda.memcpy_htod(self.alpha_g,
                             np.array(self.alpha, dtype=np.float64))

            self.stopping_criteria_reached_g = (
                cuda.mem_alloc(np.array(self.stopping_criteria_reached, dtype=np.int32).nbytes))
            cuda.memcpy_htod(self.stopping_criteria_reached_g,
                             np.array(self.stopping_criteria_reached, dtype=np.int32))

            self.generation_count_g = (
                cuda.mem_alloc(np.array(self.generation_count, dtype=np.int32).nbytes))
            cuda.memcpy_htod(self.generation_count_g,
                             np.array(self.generation_count, dtype=np.int32))

            self.fitness_eval_count_g = (
                cuda.mem_alloc(np.array(self.fitness_eval_count, dtype=np.int32).nbytes))
            cuda.memcpy_htod(self.fitness_eval_count_g,
                             np.array(self.fitness_eval_count, dtype=np.int32))

            self.weights_g = (
                cuda.mem_alloc(np.array(self.weights, dtype=np.float64).nbytes))
            cuda.memcpy_htod(self.weights_g,
                             np.array(self.weights, dtype=np.float64))

            self.weights_copy_g = (
                cuda.mem_alloc(np.array(self.weights, dtype=np.float64).nbytes))

            self.update_from_differential_mutation2 = np.zeros(self.params.population_size * 2 +
                                                               self.params.memory_size, dtype=np.int32)
            self.update_from_differential_mutation_g = (
                cuda.mem_alloc(self.update_from_differential_mutation2.nbytes))
            cuda.memcpy_htod(self.update_from_differential_mutation_g,
                             self.update_from_differential_mutation2)

            self.copy_pop_g = (
                cuda.mem_alloc(np.array(self.copy_pop, dtype=np.int32).nbytes))
            cuda.memcpy_htod(self.copy_pop_g,
                             np.array(self.copy_pop, dtype=np.int32))

            self.domination_counter_g = (
                cuda.mem_alloc(np.zeros((total+1) * (total+1), dtype=np.int32).nbytes))
            cuda.memcpy_htod(self.domination_counter_g,
                             np.zeros((total+1) * (total+1), dtype=np.int32))

            self.fronts_g = (
                cuda.mem_alloc(np.zeros(2 * self.params.population_size, dtype=np.int32).nbytes))
            cuda.memcpy_htod(self.fronts_g,
                             np.zeros(2 * self.params.population_size, dtype=np.int32))

            self.front0_mem_g = (
                cuda.mem_alloc(np.zeros(2*self.params.population_size + self.params.memory_size,
                                        dtype=np.int32).nbytes))
            cuda.memcpy_htod(self.front0_mem_g,
                             np.zeros(2*self.params.population_size + self.params.memory_size,
                                      dtype=np.int32))

            self.tam_front0_mem_g = (
                cuda.mem_alloc(np.zeros(1, dtype=np.int32).nbytes))
            cuda.memcpy_htod(self.tam_front0_mem_g, np.zeros(1, dtype=np.int32))

            self.tams_fronts_g = (
                cuda.mem_alloc(np.zeros(2 * self.params.population_size, dtype=np.int32).nbytes))
            cuda.memcpy_htod(self.tams_fronts_g,
                             np.zeros(2 * self.params.population_size, dtype=np.int32))

            self.crowding_distance_g = (
                cuda.mem_alloc(np.zeros(total, dtype=np.float64).nbytes))
            cuda.memcpy_htod(self.crowding_distance_g,
                             np.zeros(total, dtype=np.float64))

            self.whatPersonal = -1 * np.ones(self.params.population_size * 2 + self.params.memory_size,
                                             dtype=np.int32)
            self.whatPersonal_g = cuda.mem_alloc(self.whatPersonal.nbytes)
            cuda.memcpy_htod(self.whatPersonal_g, self.whatPersonal)

            self.xr_pool = np.zeros((self.params.population_size * 2 + self.params.memory_size) *
                                    self.params.population_size,
                                    dtype=np.int32)
            self.xr_pool_g = cuda.mem_alloc(self.xr_pool.nbytes)
            cuda.memcpy_htod(self.xr_pool_g, self.xr_pool)

            self.xr_list = np.zeros((self.params.population_size * 2 + self.params.memory_size) * 5,
                                    dtype=np.int32)
            self.xr_list_g = cuda.mem_alloc(self.xr_list.nbytes)
            cuda.memcpy_htod(self.xr_list_g, self.xr_list)

            self.xst = np.zeros((self.params.population_size * 2 + self.params.memory_size) *
                                self.params.position_dim, dtype=np.float64)
            self.xst_g = cuda.mem_alloc(self.xst.nbytes)
            cuda.memcpy_htod(self.xst_g, self.xst)

            self.xst_fitness = np.zeros((self.params.population_size * 2 + self.params.memory_size) *
                                        self.params.objectives_dim[0], dtype=np.float64)
            self.xst_fitness_g = cuda.mem_alloc(self.xst_fitness.nbytes)
            cuda.memcpy_htod(self.xst_fitness_g, self.xst_fitness)

            self.mutation_index = np.zeros((self.params.population_size * 2 + self.params.memory_size)
                                           , dtype=np.int32)
            self.mutation_index_g = cuda.mem_alloc(self.mutation_index.nbytes)
            cuda.memcpy_htod(self.mutation_index_g, self.mutation_index)

            self.mutation_chance = np.zeros((self.params.population_size * 2 + self.params.memory_size) *
                                            self.params.position_dim, dtype=np.float64)
            self.mutation_chance_g = cuda.mem_alloc(self.mutation_chance.nbytes)
            cuda.memcpy_htod(self.mutation_chance_g, self.mutation_chance)

            self.xst_dominate = np.zeros((self.params.population_size * 2 + self.params.memory_size) *
                                         1, dtype=np.int32)
            self.xst_dominate_g = cuda.mem_alloc(self.xst_dominate.nbytes)
            cuda.memcpy_htod(self.xst_dominate_g, self.xst_dominate)

            self.personal_best_tam = np.ones((self.params.population_size * 2 + self.params.memory_size)
                                             , dtype=np.int32)
            self.personal_best_tam_g = cuda.mem_alloc(self.personal_best_tam.nbytes)
            cuda.memcpy_htod(self.personal_best_tam_g, self.personal_best_tam)

            self.front0 = np.zeros(total+1, dtype=np.float64)
            self.front0_g = cuda.mem_alloc(self.front0.nbytes)
            cuda.memcpy_htod(self.front0_g, self.front0)

            self.tam_front0 = np.zeros(1, dtype=np.int32)
            self.tam_front0_g = cuda.mem_alloc(self.tam_front0.nbytes)
            cuda.memcpy_htod(self.tam_front0_g, self.tam_front0)

            self.communication = -1 * np.ones(total * self.params.position_dim, dtype=np.float64)
            self.communication_g = cuda.mem_alloc(self.communication.nbytes)
            cuda.memcpy_htod(self.communication_g, self.communication)

            self.cooperation_rand = -1 * np.ones(total, dtype=np.float64)
            self.cooperation_rand_g = cuda.mem_alloc(self.cooperation_rand.nbytes)
            cuda.memcpy_htod(self.cooperation_rand_g, self.cooperation_rand)

            self.population_index = np.zeros(self.params.population_size * 2, dtype=np.int32)
            self.population_index_g = cuda.mem_alloc(self.population_index.nbytes)

            self.rank_g = (
                cuda.mem_alloc(np.zeros(total, dtype=np.int32).nbytes))
            cuda.memcpy_htod(self.rank_g,
                             -1 * np.ones(total, dtype=np.int32))

            self.fitness_g = (
                cuda.mem_alloc(np.zeros(total * self.params.objectives_dim[0], dtype=np.float64).nbytes))
            cuda.memcpy_htod(self.fitness_g,
                             np.zeros(total * self.params.objectives_dim[0], dtype=np.float64))

            # if self.params.restrictions_dim > 0:
            #     self.restrictions_g = (
            #         cuda.mem_alloc(np.zeros(total * self.params.restrictions_dim, dtype=np.float64).nbytes))
            #     cuda.memcpy_htod(self.restrictions_g,
            #                      np.zeros(total * self.params.restrictions_dim, dtype=np.float64))
            # else:
            #     self.restrictions_g = (
            #         cuda.mem_alloc(np.zeros(1, dtype=np.float64).nbytes))
            self.restrictions_g = (cuda.mem_alloc(np.zeros(1, dtype=np.float64).nbytes))

            self.seed_g = cuda.mem_alloc(np.zeros(1, dtype=np.float64).nbytes)
            cuda.memcpy_htod(self.seed_g, np.random.randint(0, int(1e9), dtype=np.int32))

            self.position_g = (
                cuda.mem_alloc(np.zeros(total * self.params.position_dim, dtype=np.float64).nbytes))

            self.velocity_g = (
                cuda.mem_alloc(np.zeros(total * self.params.position_dim, dtype=np.float64).nbytes))

            self.dominated_set_g = (
                cuda.mem_alloc(np.zeros(total * total, dtype=np.int32).nbytes))
            cuda.memcpy_htod(self.dominated_set_g,
                             np.zeros(self.params.population_size, dtype=np.int32))

            self.sigma_g = (
                cuda.mem_alloc(np.zeros(total * 3, dtype=np.float64).nbytes))
            cuda.memcpy_htod(self.sigma_g,
                             np.zeros(total * 3, dtype=np.float64))

            self.global_best_g = (
                cuda.mem_alloc(np.zeros(total, dtype=np.int32).nbytes))
            cuda.memcpy_htod(self.global_best_g,
                             np.zeros(total, dtype=np.int32))

            self.personal_best_position_g = (
                cuda.mem_alloc(np.zeros(total * self.params.position_dim * self.params.personal_guide_array_size,
                                        dtype=np.float64).nbytes))
            cuda.memcpy_htod(self.personal_best_position_g,
                             1e10 + np.zeros(total * self.params.position_dim * self.params.personal_guide_array_size,
                                             dtype=np.float64))

            self.personal_best_velocity_g = (
                cuda.mem_alloc(np.zeros(total * self.params.position_dim * self.params.personal_guide_array_size,
                                        dtype=np.float64).nbytes))
            cuda.memcpy_htod(self.personal_best_velocity_g,
                             np.zeros(total * self.params.position_dim * self.params.personal_guide_array_size,
                                      dtype=np.float64))

            self.personal_best_fitness_g = (
                cuda.mem_alloc(np.zeros(total * self.params.objectives_dim[0] * self.params.personal_guide_array_size,
                                        dtype=np.float64).nbytes))
            cuda.memcpy_htod(self.personal_best_fitness_g,
                             np.zeros(total * self.params.objectives_dim[0] * self.params.personal_guide_array_size,
                                      dtype=np.float64))

            self.population_copy2 = copy.deepcopy(self.population)

            self.aux = np.zeros((2 * self.params.population_size + self.params.memory_size) *
                                self.params.position_dim, dtype=np.float64)
            self.aux_g = cuda.mem_alloc(self.aux.nbytes)

            self.aux2 = np.zeros((2 * self.params.population_size + self.params.memory_size) *
                                 self.params.position_dim, dtype=np.float64)
            self.aux2_g = cuda.mem_alloc(self.aux2.nbytes)

            self.aux3 = np.zeros((2 * self.params.population_size + self.params.memory_size) *
                                 self.params.position_dim * self.params.personal_guide_array_size, dtype=np.float64)
            self.aux3_g = cuda.mem_alloc(self.aux3.nbytes)

            self.aux4 = np.zeros(2*self.params.population_size*self.params.objectives_dim[0],
                                 dtype=np.int32)
            self.aux4_g = cuda.mem_alloc(self.aux4.nbytes)

            self.initial_memory = np.zeros(self.params.population_size, dtype=np.int32)
            self.initial_memory_g = cuda.mem_alloc(self.initial_memory.nbytes)

            f = open('mesh.cu')
            code = f.read()
            f.close()
            self.mod = SourceModule(code, no_extern_c=True)

    def init_population(self):

        div1 = int(np.ceil(self.params.population_size/128))
        div2 = int(np.ceil(self.params.position_dim/8))

        cuda.memcpy_htod(self.seed_g, np.random.randint(0, int(1e9), dtype=np.int32))

        init_population = self.mod.get_function("init_population")
        init_population(self.position_g, self.params.position_dim_g,self.seed_g,
                        self.params.position_min_value_g, self.params.position_max_value_g,
                        self.params.population_size_g,
                   block=(int(self.params.population_size/div1),
                          int(self.params.position_dim/div2),
                          1), grid=(div1, div2, 1))
        cuda.Context.synchronize()

        init_population = self.mod.get_function("init_population")
        init_population(self.velocity_g, self.params.position_dim_g, self.seed_g,
                        self.params.velocity_min_value_g, self.params.velocity_max_value_g,
                        self.params.population_size_g,
                        block=(int(self.params.population_size / div1),
                               int(self.params.position_dim / div2),
                               1), grid=(div1, div2, 1))
        cuda.Context.synchronize()

    def particle_copy(self, particle):
        copy = Particle(self.params.position_min_value, self.params.position_max_value, self.params.position_dim,
                        self.params.velocity_min_value, self.params.velocity_max_value,
                        self.params.objectives_dim, self.params.otimizations_type,
                        self.params.secondary_params)
        copy.position = particle.position
        copy.fitness = particle.fitness
        copy.velocity = particle.velocity
        if particle.personal_best is not None:
            copy_personal_best_list = []
            for pb in particle.personal_best:
                copy_pb = Particle(self.params.position_min_value, self.params.position_max_value,
                                   self.params.position_dim,
                                   self.params.velocity_min_value, self.params.velocity_max_value,
                                   self.params.objectives_dim, self.params.otimizations_type,
                                   self.params.secondary_params)
                copy_pb.position = pb.position
                copy_pb.fitness = pb.fitness
                copy_personal_best_list.append(copy_pb)
            copy.personal_best = copy_personal_best_list
        return copy

    def fitness_evaluation(self, function, *args):
        self.fitness_eval_count = self.fitness_eval_count + 1
        if self.params.initial_state:
            args_and_initial_state = []
            args_and_initial_state.append(args[0])
            args_and_initial_state.extend(self.params.initial_state)
            return function(args_and_initial_state)
        else:
            return function(*args)

    def fast_nondominated_sort_gpu(self, first_front_only=False, use_copy_population=False,
                                   specific_population=None):
        population = []
        population2 = []
        fronts = [[]]

        for p in self.population:
            population.append(p)
            population2.append(p.position)
        if use_copy_population:
            for o in self.population_copy:
                population.append(o)
                population2.append(o.position)
        population2 = np.array(population2, dtype=np.float64)
        domination = np.zeros((len(population), len(population)), dtype=np.int32)
        # counter = np.zeros((population.shape[0]), dtype=np.float64)

        population_g = cuda.mem_alloc(population2.nbytes)
        # counter_g = cuda.mem_alloc(counter.nbytes)
        dim_g = cuda.mem_alloc(np.array(population2.shape, dtype=np.int32).nbytes)
        domination_g = cuda.mem_alloc(domination.nbytes)

        cuda.memcpy_htod(population_g, population2)
        # cuda.memcpy_htod(counter_g, counter)
        cuda.memcpy_htod(dim_g, np.array(population2.shape, dtype=np.int32))
        cuda.memcpy_htod(domination_g, domination)

        f = open('testes/ordenacao.cu')
        code = f.read()
        f.close()
        mod = SourceModule(code)

        sort2 = mod.get_function("sort2")

        block_x = 32
        block_y = 32
        grid_x = int(len(population) / 32)
        grid_y = int(len(population) / 32)
        sort2(population_g, dim_g, domination_g, block=(block_x, block_y, 1), grid=(grid_x, grid_y, 1))
        cuda.Context.synchronize()

        # cuda.memcpy_dtoh(population, population_g)
        cuda.memcpy_dtoh(domination, domination_g)

        for i in range(len(population)):
            self.population[i].domination_counter = np.sum(domination[:, i])
            self.population[i].dominated_set = np.array(self.population)[domination[i, :] == 1]
            if self.population[i].domination_counter == 0:
                fronts[0].append(self.population[i])
                self.population[i].rank = 0

        i = 0
        if not first_front_only:
            while len(fronts[i]) != 0:
                new_front = []
                for p in fronts[i]:
                    for s in p.dominated_set:
                        s.domination_counter = s.domination_counter - 1
                        if s.domination_counter == 0:
                            new_front.append(s)
                            s.rank = i + 1
                i += 1
                fronts.append(list(new_front))
            fronts.pop()
            for p in population:
                p.dominated_set = []
            return fronts
        else:
            for p in population:
                p.dominated_set = []
            return fronts[0]

    def fast_nondominated_sort(self, first_front_only=False, use_copy_population=False,
                               specific_population=None):
        population = []
        fronts = []
        fronts.append([])

        if specific_population != None:
            for s in specific_population:
                population.append(s)
        else:
            for p in self.population:
                population.append(p)
            if use_copy_population:
                for o in self.population_copy:
                    population.append(o)

        for p in population:
            p.domination_counter = 0
            for q in population:
                if p == q:
                    continue
                if p >> q:
                    p.dominated_set.append(q)
                elif p << q:
                    p.domination_counter = p.domination_counter + 1
            if p.domination_counter == 0:
                fronts[0].append(p)
                p.rank = 0

        i = 0
        if not first_front_only:
            while len(fronts[i]) != 0:
                new_front = []
                for p in fronts[i]:
                    for s in p.dominated_set:
                        s.domination_counter = s.domination_counter - 1
                        if s.domination_counter == 0:
                            new_front.append(s)
                            s.rank = i + 1
                i += 1
                fronts.append(list(new_front))
            fronts.pop()
            for p in population:
                p.dominated_set = []
            return fronts
        else:
            for p in population:
                p.dominated_set = []
            return fronts[0]

    def crowding_distance(self, front):
        for j in front:
            j.crowd_distance = 0
        for objective_index in range(self.params.objectives_dim):
            front.sort(key=lambda x: x.fitness[objective_index])
            front[0].crowd_distance = sys.maxsize
            front[-1].crowd_distance = sys.maxsize
            for p in range(1, len(front) - 1):
                if front[p].crowd_distance == sys.maxsize:
                    continue
                if front[-1].fitness[objective_index] - front[0].fitness[objective_index] == 0:
                    continue
                front[p].crowd_distance += (front[p + 1].fitness[objective_index] - front[p - 1].fitness[
                    objective_index]) / (front[-1].fitness[objective_index] - front[0].fitness[objective_index])

            # apenas para validacao, apagar depois. Foi necessario para manter a ordem identica
            # dos fronts
            # if len(front) == 256:
            #     for i in range(len(front)):
            #         if len(np.where(np.array(self.population_copy2) == front[i])[0]) != 0:
            #             self.aux4[objective_index*256+i] = (
            #                 np.where(np.array(self.population_copy2) == front[i])[0][0])
            #         else:
            #             self.aux4[objective_index * 256 + i] = ((
            #                 np.where(np.array(self.population_copy3) == front[i])[0][0])
            #                                                     + self.params.population_size)
            # pass



    def crowd_distance_selection(self, particle_A, particle_B):
        if particle_A.rank < particle_B.rank:
            return particle_A
        elif particle_B.rank < particle_A.rank:
            return particle_B
        elif particle_A.rank == particle_B.rank:
            if particle_A.crowd_distance > particle_B.crowd_distance:
                return particle_A
            elif particle_B.crowd_distance >= particle_A.crowd_distance:
                return particle_B

    def check_position_limits(self, position_input):
        position = position_input[:]
        for i in range(self.params.position_dim):
            if position[i] < self.params.position_min_value[i]:
                position[i] = self.params.position_min_value[i]
            if position[i] > self.params.position_max_value[i]:
                position[i] = self.params.position_max_value[i]
        return position

    def check_velocity_limits(self, velocity_input, position_input=None):
        velocity = velocity_input[:]
        if position_input is not None:
            position = position_input[:]
            for i in range(self.params.position_dim):
                if position[i] == self.params.position_min_value[i] and velocity[i] < 0:
                    velocity[i] = -1 * velocity[i]
                elif position[i] == self.params.position_max_value[i] and velocity[i] > 0:
                    velocity[i] = -1 * velocity[i]
        else:
            for i in range(self.params.position_dim):
                if velocity[i] < self.params.velocity_min_value[i]:
                    velocity[i] = self.params.velocity_min_value[i]
                if velocity[i] > self.params.velocity_max_value[i]:
                    velocity[i] = self.params.velocity_max_value[i]
        return velocity

    def euclidian_distance(self, a, b):
        a = np.asarray(a)
        b = np.asarray(b)
        return np.linalg.norm(a - b)

    def sigma_eval(self, particle):
        squared_power = np.power(particle.fitness, 2)
        denominator = np.sum(squared_power)
        numerator = []
        # testeTempo250624 para self.params.objectives_dim > 3?
        if self.params.objectives_dim == 2:
            numerator = squared_power[0] - squared_power[1]
        else:
            for i in range(self.params.objectives_dim):
                if i != self.params.objectives_dim - 1:
                    numerator.append(squared_power[i] - squared_power[i + 1])
                else:
                    numerator.append(squared_power[i] - squared_power[0])
        sigma = np.divide(numerator, denominator)
        particle.sigma_value = sigma
        pass
        # particle.sigma_value = sigma.astype(np.float64)
        pass

    def sigma_nearest(self, particle, search_pool):
        sigma_distance = sys.maxsize
        nearest_particle = None
        for p in search_pool:
            if particle != p:
                new_distance = self.euclidian_distance(particle.sigma_value, p.sigma_value)
                if sigma_distance > new_distance:
                    sigma_distance = new_distance
                    nearest_particle = p
        if nearest_particle is None:  # se a distancia inicial testeTempo250624 maxima, entao
            # sempre vai ter um melhor pq
            # 1 distancia nao sera menor que a maxima?
            nearest_particle = particle
        nearest_particle = copy.deepcopy(nearest_particle)
        # entao global best nao testeTempo250624 a melhor posicao entre todas as particulas?
        particle.global_best = nearest_particle

    def move_particle(self, particle, particle_index, is_copy):
        if is_copy:
            weights = self.weights_copy
        else:
            weights = self.weights

        # original
        # personal_best_pos = particle.personal_best[np.random.choice(len(particle.personal_best))].position

        # teste
        personal_best_pos = particle.personal_best[np.random.choice(len(particle.personal_best))]
        if is_copy:
            self.whatPersonal[particle_index + self.params.population_size] = (
                np.where(np.array(particle.personal_best) == personal_best_pos))[0][0]
        else:
            self.whatPersonal[particle_index] = (
                np.where(np.array(particle.personal_best) == personal_best_pos))[0][0]
        personal_best_pos = personal_best_pos.position

        inertia_term = np.asarray(particle.velocity) * weights[0][particle_index]
        memory_term = weights[1][particle_index] * (np.asarray(personal_best_pos) - np.asarray(particle.position))

        # original
        # communication = (np.random.uniform(0.0, 1.0, self.params.position_dim) < self.params.communication_probability) * 1

        # teste
        communication = (np.random.uniform(0.0, 1.0, self.params.position_dim))
        if is_copy:
            for i in range(self.params.position_dim):
                self.communication[(particle_index + self.params.population_size) * self.params.position_dim + i] \
                    = communication[i]
        else:
            for i in range(self.params.position_dim):
                self.communication[particle_index * self.params.position_dim + i] \
                    = communication[i]
        communication = communication < self.params.communication_probability * 1

        # original
        # nao entendi o por que de multiplicar o global best por 1+(entre 0 testeTempo250624 1)*w3
        # cooperation_term = weights[2][particle_index] * (np.asarray(particle.global_best.position)
        # * (1 + (weights[3][particle_index] * np.random.normal(0, 1))) - np.asarray(particle.position))
        # cooperation_term = cooperation_term * communication

        # teste
        if is_copy:
            self.cooperation_rand[particle_index + self.params.population_size] = np.random.normal(0, 1)
            rand = self.cooperation_rand[particle_index + self.params.population_size]
        else:
            self.cooperation_rand[particle_index] = np.random.normal(0, 1)
            rand = self.cooperation_rand[particle_index]
        cooperation_term = weights[2][particle_index] * (np.asarray(particle.global_best.position)
                                                         * (1 + (
                        weights[3][particle_index] * rand)) - np.asarray(particle.position))
        cooperation_term = cooperation_term * communication

        # original
        new_velocity = inertia_term + memory_term + cooperation_term
        new_velocity = self.check_velocity_limits(new_velocity)

        new_position = np.asarray(particle.position) + new_velocity
        new_position = self.check_position_limits(new_position)
        new_velocity = self.check_velocity_limits(new_velocity, new_position)

        particle.velocity = new_velocity
        particle.position = new_position

        if self.params.secondary_params:
            fit_eval = self.fitness_evaluation(self.fitness_function, particle.position)
            particle.fitness = fit_eval[0]
            particle.secondary_params = fit_eval[1:]
        else:
            particle.fitness = self.fitness_evaluation(self.fitness_function, particle.position)

    def mutate_weights(self):
        for i in range(len(self.weights)):
            for j in range(len(self.weights[i])):
                # weight[i] = weight[i] + np.random.normal(0,1)*self.params.mutation_rate
                if i < 4:
                    self.weights[i][j] = truncnorm.rvs(0, 1) * self.params.mutation_rate
                    if self.weights[i][j] > 1:
                        self.weights[i][j] = 1
                    elif self.weights[i][j] < 0:
                        self.weights[i][j] = 0
                if i == 4:
                    self.weights[i][j] = truncnorm.rvs(0, 0.5) * self.params.mutation_rate
                    if self.weights[i][j] > 0.5:
                        self.weights[i][j] = 0.5
                    elif self.weights[i][j] < 0:
                        self.weights[i][j] = 0
                if i == 5:
                    self.weights[i][j] = truncnorm.rvs(0, 2) * self.params.mutation_rate
                    if self.weights[i][j] > 2:
                        self.weights[i][j] = 2
                    elif self.weights[i][j] < 0:
                        self.weights[i][j] = 0
        if self.copy_pop:
            for i in range(len(self.weights_copy)):
                for j in range(len(self.weights_copy[i])):
                    # weight[i] = weight[i] + np.random.normal(0,1)*self.params.mutation_rate
                    if i < 4:
                        self.weights_copy[i][j] = truncnorm.rvs(0, 1) * self.params.mutation_rate
                        if self.weights_copy[i][j] > 1:
                            self.weights_copy[i][j] = 1
                        elif self.weights_copy[i][j] < 0:
                            self.weights_copy[i][j] = 0
                    if i == 4:
                        self.weights_copy[i][j] = truncnorm.rvs(0, 0.5) * self.params.mutation_rate
                        if self.weights_copy[i][j] > 0.5:
                            self.weights_copy[i][j] = 0.5
                        elif self.weights_copy[i][j] < 0:
                            self.weights_copy[i][j] = 0
                    if i == 5:
                        self.weights_copy[i][j] = truncnorm.rvs(0, 2) * self.params.mutation_rate
                        if self.weights_copy[i][j] > 2:
                            self.weights_copy[i][j] = 2
                        elif self.weights_copy[i][j] < 0:
                            self.weights_copy[i][j] = 0

    def differential_mutation(self, particle, particle_index):
        Xr_pool = []

        # np.random.seed(0)
        personal_best = particle.personal_best[np.random.choice(len(particle.personal_best))]

        # teste8 = np.zeros((self.params.population_size * 2 + self.params.memory_size) *
        #                   self.params.position_dim,
        #                   dtype=np.float64)
        # cuda.memcpy_dtoh(teste8, self.position_g)
        # teste8.shape = 261, 10
        # l = []
        # for i in range(self.params.memory_size):
        #     l.append(abs(np.array(self.memory[i].position) - teste8[256 + i]))
        #     l[-1] = max(l[-1])
        # self.l61.append(l)

        self.whatPersonal[particle_index] = (
            np.where(np.array(particle.personal_best) == personal_best))[0][0]

        # teste8 = np.zeros((self.params.population_size * 2 + self.params.memory_size) *
        #                   self.params.position_dim,
        #                   dtype=np.float64)
        # cuda.memcpy_dtoh(teste8, self.position_g)
        # teste8.shape = 261, 10
        # l = []
        # for i in range(self.params.memory_size):
        #     l.append(abs(np.array(self.memory[i].position) - teste8[256 + i]))
        #     l[-1] = max(l[-1])
        # self.l62.append(l)

        if self.params.Xr_pool_type == 0:  # Apenas Populacao
            for p in self.population:
                if not personal_best == p or not particle == p:
                    if not particle >> p:
                        Xr_pool.append(p)
        elif self.params.Xr_pool_type == 1:  # Apenas Memoria
            for m in self.memory:
                if not personal_best == m or not particle == m:
                    if not particle >> m:
                        Xr_pool.append(m)
        elif self.params.Xr_pool_type == 2:  # Combinacao Memoria testeTempo250624 Populacao
            for m in self.memory:
                if not personal_best == m and not particle == m:
                    if not particle >> m:
                        Xr_pool.append(m)
            for p in self.population:
                if not personal_best == p and not particle == p and p not in Xr_pool and p.rank > particle.rank:
                    if not particle >> p:
                        Xr_pool.append(p)

        if self.params.DE_mutation_type == 0 and len(Xr_pool) >= 3:  # DE\rand\1\Bin
            # np.random.seed(0)
            Xr_list = np.random.choice(Xr_pool, 3, replace=False)

            if self.params.Xr_pool_type == 1:
                self.xr_list[particle_index * 5 + 0] = np.where(np.array(self.memory) == Xr_list[0])[0][0]
                self.xr_list[particle_index * 5 + 1] = np.where(np.array(self.memory) == Xr_list[1])[0][0]
                self.xr_list[particle_index * 5 + 2] = np.where(np.array(self.memory) == Xr_list[2])[0][0]
                self.xr_list[particle_index * 5 + 3] = -1
                self.xr_list[particle_index * 5 + 4] = -1

            # teste8 = np.zeros((self.params.population_size * 2 + self.params.memory_size) *
            #                   self.params.position_dim,
            #                   dtype=np.float64)
            # cuda.memcpy_dtoh(teste8, self.position_g)
            # teste8.shape = 261, 10
            # l = []
            # for i in range(self.params.memory_size):
            #     l.append(abs(np.array(self.memory[i].position) - teste8[256 + i]))
            #     l[-1] = max(l[-1])
            # self.l63.append(l)

            Xr1 = np.asarray(Xr_list[0].position)
            Xr2 = np.asarray(Xr_list[1].position)
            Xr3 = np.asarray(Xr_list[2].position)

            Xst = Xr1 + self.weights[5][particle_index] * (Xr2 - Xr3)
            # if particle_index == 0:
            #     print(Xst)
            # if particle_index == 127:
            #     print(Xst)
            Xst = Xst.tolist()
            # if particle_index == 0:
            #     print(Xst)
            # if particle_index == 127:
            #     print(Xst)
            Xst = self.check_position_limits(Xst)
            # if particle_index == 0:
            #     print(Xst)
            # if particle_index == 127:
            #     print(Xst)

            # np.random.seed(0)
            mutation_index = np.random.choice(self.params.position_dim)

            # teste8 = np.zeros((self.params.population_size * 2 + self.params.memory_size) *
            #                   self.params.position_dim,
            #                   dtype=np.float64)
            # cuda.memcpy_dtoh(teste8, self.position_g)
            # teste8.shape = 261, 10
            # l = []
            # for i in range(self.params.memory_size):
            #     l.append(abs(np.array(self.memory[i].position) - teste8[256 + i]))
            #     l[-1] = max(l[-1])
            # self.l64.append(l)

            self.mutation_index[particle_index] = mutation_index

            # teste8 = np.zeros((self.params.population_size * 2 + self.params.memory_size) *
            #                   self.params.position_dim,
            #                   dtype=np.float64)
            # cuda.memcpy_dtoh(teste8, self.position_g)
            # teste8.shape = 261, 10
            # l = []
            # for i in range(self.params.memory_size):
            #     l.append(abs(np.array(self.memory[i].position) - teste8[256 + i]))
            #     l[-1] = max(l[-1])
            # self.l65.append(l)

            # if particle_index < 5 or particle_index >= 123:
            #     print(mutation_index)

            # np.random.seed(0)
            mutation_chance = np.random.uniform(0.0, 1.0, self.params.position_dim)

            # teste8 = np.zeros((self.params.population_size * 2 + self.params.memory_size) *
            #                   self.params.position_dim,
            #                   dtype=np.float64)
            # cuda.memcpy_dtoh(teste8, self.position_g)
            # teste8.shape = 261, 10
            # l = []
            # for i in range(self.params.memory_size):
            #     l.append(abs(np.array(self.memory[i].position) - teste8[256 + i]))
            #     l[-1] = max(l[-1])
            # self.l66.append(l)

            mutation_chance = mutation_chance.astype(np.float64)

            # teste8 = np.zeros((self.params.population_size * 2 + self.params.memory_size) *
            #                   self.params.position_dim,
            #                   dtype=np.float64)
            # cuda.memcpy_dtoh(teste8, self.position_g)
            # teste8.shape = 261, 10
            # l = []
            # for i in range(self.params.memory_size):
            #     l.append(abs(np.array(self.memory[i].position) - teste8[256 + i]))
            #     l[-1] = max(l[-1])
            # self.l67.append(l)

            self.mutation_chance[particle_index * self.params.position_dim:(particle_index + 1)
                                                                           * self.params.position_dim] = mutation_chance.copy()

            # teste8 = np.zeros((self.params.population_size * 2 + self.params.memory_size) *
            #                   self.params.position_dim,
            #                   dtype=np.float64)
            # cuda.memcpy_dtoh(teste8, self.position_g)
            # teste8.shape = 261, 10
            # l = []
            # for i in range(self.params.memory_size):
            #     l.append(abs(np.array(self.memory[i].position) - teste8[256 + i]))
            #     l[-1] = max(l[-1])
            # self.l68.append(l)

            # if self.gpu:
            #     self.xst[particle_index*self.params.position_dim:(particle_index+1)*self.params.position_dim] = Xst

            # teste8 = np.zeros((self.params.population_size * 2 + self.params.memory_size) *
            #                   self.params.position_dim,
            #                   dtype=np.float64)
            # cuda.memcpy_dtoh(teste8, self.position_g)
            # teste8.shape = 261, 10
            # l = []
            # for i in range(self.params.memory_size):
            #     l.append(abs(np.array(self.memory[i].position) - teste8[256 + i]))
            #     l[-1] = max(l[-1])
            # self.l69.append(l)

            # self.teste8 = np.zeros((self.params.population_size * 2 + self.params.memory_size) *
            #                        self.params.objectives_dim,
            #                        dtype=np.float64)
            # self.teste8 = self.xst_fitness.copy()
            # self.teste8.shape = 261, 2

            for i in range(self.params.position_dim):
                if mutation_chance[i] < self.weights[4][particle_index] or i == mutation_index:
                    Xst[i] = personal_best.position[i]
            if self.gpu:
                self.xst[
                particle_index * self.params.position_dim:(particle_index + 1) * self.params.position_dim] = Xst

            # teste8 = np.zeros((self.params.population_size * 2 + self.params.memory_size) *
            #                   self.params.position_dim,
            #                   dtype=np.float64)
            # cuda.memcpy_dtoh(teste8, self.position_g)
            # teste8.shape = 261, 10
            # l = []
            # for i in range(self.params.memory_size):
            #     l.append(abs(np.array(self.memory[i].position) - teste8[256 + i]))
            #     l[-1] = max(l[-1])
            # self.l610.append(l)

        elif self.params.DE_mutation_type == 1 and len(Xr_pool) >= 5:  # DE\rand\2\Bin
            Xr_list = np.random.choice(Xr_pool, 5, replace=False)

            Xr1 = np.asarray(Xr_list[0].position)
            Xr2 = np.asarray(Xr_list[1].position)
            Xr3 = np.asarray(Xr_list[2].position)
            Xr4 = np.asarray(Xr_list[3].position)
            Xr5 = np.asarray(Xr_list[4].position)

            Xst = Xr1 + self.weights[5][particle_index] * ((Xr2 - Xr3) + (Xr4 - Xr5))
            Xst = Xst.tolist()
            Xst = self.check_position_limits(Xst)

            mutation_index = np.random.choice(self.params.position_dim)
            mutation_chance = np.random.uniform(0.0, 1.0, self.params.position_dim)

            for i in range(self.params.position_dim):
                if (mutation_chance[i] < self.weights[4][particle_index] or i == mutation_index):
                    Xst[i] = personal_best.position[i]

        elif self.params.DE_mutation_type == 2 and len(Xr_pool) >= 2:  # DE/Best/1/Bin
            Xr_list = np.random.choice(Xr_pool, 2, replace=False)

            Xr1 = np.asarray(Xr_list[0].position)
            Xr2 = np.asarray(Xr_list[1].position)

            Xst = particle.global_best.position + self.weights[5][particle_index] * (Xr1 - Xr2)
            Xst = Xst.tolist()
            Xst = self.check_position_limits(Xst)

            mutation_index = np.random.choice(self.params.position_dim)
            mutation_chance = np.random.uniform(0.0, 1.0, self.params.position_dim)

            for i in range(self.params.position_dim):
                if not (mutation_chance[i] < self.weights[4][particle_index] or i == mutation_index):
                    Xst[i] = particle.global_best.position[i]

        elif self.params.DE_mutation_type == 3 and len(Xr_pool) >= 2:  # DE/Current-to-best/1/Bin
            Xr_list = np.random.choice(Xr_pool, 2, replace=False)

            Xr1 = np.asarray(Xr_list[0].position)
            Xr2 = np.asarray(Xr_list[1].position)

            Xst = np.asarray(personal_best.position) + self.weights[5][particle_index] * (
                        (Xr1 - Xr2) + (np.asarray(particle.global_best.position) - np.asarray(personal_best.position)))
            Xst = Xst.tolist()
            Xst = self.check_position_limits(Xst)

            mutation_index = np.random.choice(self.params.position_dim)
            mutation_chance = np.random.uniform(0.0, 1.0, self.params.position_dim)

            for i in range(self.params.position_dim):
                if not (mutation_chance[i] < self.weights[4][particle_index] or i == mutation_index):
                    Xst[i] = particle.global_best.position[i]

        elif self.params.DE_mutation_type == 4 and len(Xr_pool) >= 3:  # DE/Current-to-rand/1/Bin
            Xr_list = np.random.choice(Xr_pool, 3, replace=False)

            Xr1 = np.asarray(Xr_list[0].position)
            Xr2 = np.asarray(Xr_list[1].position)
            Xr3 = np.asarray(Xr_list[2].position)

            Xst = np.asarray(personal_best.position) + self.weights[5][particle_index] * (
                        (Xr1 - Xr2) + (Xr3 - np.asarray(personal_best.position)))
            Xst = Xst.tolist()
            Xst = self.check_position_limits(Xst)

            mutation_index = np.random.choice(self.params.position_dim)
            mutation_chance = np.random.uniform(0.0, 1.0, self.params.position_dim)

            for i in range(self.params.position_dim):
                if not (mutation_chance[i] < self.weights[4][particle_index] or i == mutation_index):
                    Xst[i] = particle.global_best.position[i]

        else:
            return

        if self.params.secondary_params:
            fit_eval = self.fitness_evaluation(self.fitness_function, Xst)
            Xst_fit = fit_eval[0]
        else:
            Xst_fit = self.fitness_evaluation(self.fitness_function, Xst)
            if self.gpu:
                # self.xst_fitness[particle_index*self.params.objectives_dim:(particle_index+1)*self.params.objectives_dim]\
                #     = Xst_fit.copy()
                self.xst_fitness[
                particle_index * self.params.objectives_dim:(particle_index + 1) * self.params.objectives_dim] \
                    = Xst_fit
        Xst_particle = Particle(self.params.position_min_value, self.params.position_max_value,
                                self.params.position_dim,
                                self.params.velocity_min_value, self.params.velocity_max_value,
                                self.params.objectives_dim, self.params.otimizations_type,
                                self.params.secondary_params)
        Xst_particle.fitness = Xst_fit
        Xst_particle.position = Xst
        if self.params.secondary_params:
            Xst_particle.secondary_params = fit_eval[1:]

        # self.teste = np.zeros((self.params.population_size * 2 + self.params.memory_size) *
        #                  self.params.position_dim,
        #                  dtype=np.float64)
        # cuda.memcpy_dtoh(self.teste, self.xst_g)
        # self.teste.shape = 261, 10
        #
        # self.teste2 = np.zeros((self.params.population_size * 2 + self.params.memory_size) *
        #                   self.params.objectives_dim,
        #                   dtype=np.float64)
        # cuda.memcpy_dtoh(self.teste2, self.xst_fitness_g)
        # self.teste2.shape = 261, 2
        #
        # l = []
        # for i in range(len(self.teste2)):
        #     l.append(abs(np.array(self.xst_fitness[i] - self.teste2[i])))
        #     l[-1] = max(l[-1])
        # temp = np.where(np.array(l) > 1e-5)[0]
        # print(temp)
        # l2 = []
        # for i in range(len(self.teste)):
        #     l2.append(abs(np.array(self.xst[i] - self.teste[i])))
        #     l2[-1] = max(l2[-1])
        # temp2 = np.where(np.array(l2) > 1e-5)[0]
        # print(temp2)
        # pass

        # self.teste10.append([])
        # self.teste10[-1].append(self.xst)
        # self.teste10[-1].append(self.xst_fitness)
        # self.teste10[-1].append(self.xst_dominate)
        # self.teste10[-1].append(self.xst)

        # if particle_index == 80:
        #     pass

        # pq apenas atualiza a posicao testeTempo250624 a avaliação?
        if Xst_particle >> particle:
            # if Xst_particle >> particle and particle_index < 84:
            self.xst_dominate[particle_index] = 1
            # self.teste10[-1].append(self.xst)
            # self.teste10[-1].append(self.xst_fitness)
            # self.teste10[-1].append(self.xst_dominate)

            particle.fitness = Xst_fit
            particle.position = Xst

            # self.teste10[-1].append(self.xst)
            # self.teste10[-1].append(self.xst_fitness)
            # self.teste10[-1].append(self.xst_dominate)

            self.update_from_differential_mutation = True
            self.update_personal_best(particle)
            pass



        # else:
        #     self.teste10.append([])

    def memory_update(self):
        new_memory_candidates = []
        # l = []
        for f in self.fronts[0]:
            new_memory_candidates.append(f)
            # l.append(int(np.where(np.array(self.population) == f)[0]))
        for m in self.memory:
            if m not in new_memory_candidates:
                new_memory_candidates.append(m)
                # l.append(int(np.where(np.array(self.memory) == m)[0])+256)
        new_memory_front = self.fast_nondominated_sort(True, False, new_memory_candidates)

        # teste
        # l2 = []
        # for f in new_memory_front:
        #     if len(np.where(np.array(self.population) == f)[0]) > 0:
        #         l2.append(int(np.where(np.array(self.population) == f)[0]))
        #     else:
        #         l2.append(int(np.where(np.array(self.memory) == f)[0]) + 256)
        # print(l2)

        new_memory = []
        if len(new_memory_front) <= self.params.memory_size:
            for f in new_memory_front:
                new_memory.append(f)
        else:
            self.crowding_distance(new_memory_front)
            new_memory_front.sort(key=lambda x: x.crowd_distance)
            i = len(new_memory_front) - 1
            while len(new_memory) < self.params.memory_size:
                new_memory.append(new_memory_front[i])
                i = i - 1

        # self.memory2 = self.memory
        # self.memory3 = l
        self.memory = copy.deepcopy(new_memory)

    # def update_personal_best_gpu(self, particle):
    def update_personal_best_gpu(self):
        # i = len(particle.personal_best)

        # teste = np.zeros(self.params.population_size*2+self.params.memory_size, dtype= np.int32)
        # cuda.memcpy_dtoh(teste, self.personal_best_g)
        # print(teste)

        update_personal_best = self.mod.get_function("update_personal_best")
        update_personal_best(self.personal_best_position_g, self.personal_best_velocity_g, self.personal_best_fitness_g,
                             self.params.objectives_dim_g, self.params.position_dim_g, self.position_g, self.fitness_g,
                             self.params.personal_guide_array_size_g,
                             block=(self.params.population_size, 1, 1), grid=(1, 1, 1))
        cuda.Context.synchronize()

        # teste = np.zeros((self.params.population_size*2+self.params.memory_size)*self.params.position_dim*
        #                  self.params.personal_guide_array_size, dtype=np.float64)
        # cuda.memcpy_dtoh(teste, self.personal_best_position_g)
        # teste2 = np.zeros((self.params.population_size * 2 + self.params.memory_size) * self.params.objectives_dim *
        #                  self.params.personal_guide_array_size, dtype=np.float64)
        # cuda.memcpy_dtoh(teste2, self.personal_best_fitness_g)
        # pass

        # if particle.personal_best[0] is None:
        #     pass
            # new_personal_best = Particle(self.params.position_min_value,self.params.position_max_value, self.params.position_dim,
            #                     self.params.velocity_min_value, self.params.velocity_max_value,
            #                     self.params.objectives_dim,self.params.otimizations_type,
            #                     self.params.secondary_params)
            # new_personal_best.position = particle.position
            # new_personal_best.fitness = particle.fitness
            # particle.personal_best = []
            # particle.personal_best.append(new_personal_best)

        # else:
        #     pass
            # remove_list = []
            # include_flag = False
            # for s in particle.personal_best:
            #     if particle == s:
            #         break
            #     if particle >> s:
            #         include_flag = True
            #         if s not in remove_list:
            #             remove_list.append(s)
            #     if not particle << s:
            #         i = i - 1
            # if len(remove_list) > 0:
            #     for r in remove_list:
            #         particle.personal_best.remove(r)
            # # tem algo estranho aqui. Se i = 1 testeTempo250624 include_flag = true, significa que a particula
            # # é dominada por 1 particula do conjunto testeTempo250624 domina pelo menos 1 outra
            # # tentar obter um exemplo. talvez um ou exclusivo?
            # if i == 0 or include_flag:
            #     new_personal_best = Particle(self.params.position_min_value,self.params.position_max_value, self.params.position_dim,
            #                     self.params.velocity_min_value, self.params.velocity_max_value,
            #                     self.params.objectives_dim,self.params.otimizations_type,
            #                     self.params.secondary_params)
            #     new_personal_best.position = particle.position
            #     new_personal_best.fitness = particle.fitness
            #
            #     if self.params.personal_guide_array_size > 0 and len(particle.personal_best) == self.params.personal_guide_array_size:
            #         particle.personal_best.pop(0)
            #     particle.personal_best.append(new_personal_best)

    def update_personal_best(self, particle):
        i = len(particle.personal_best)
        if particle.personal_best[0] is None:
            new_personal_best = Particle(self.params.position_min_value, self.params.position_max_value,
                                         self.params.position_dim,
                                         self.params.velocity_min_value, self.params.velocity_max_value,
                                         self.params.objectives_dim, self.params.otimizations_type,
                                         self.params.secondary_params)

            # esse trecho faz a position testeTempo250624 fitness como o emsmo objeto da particula no eprsonal best
            # testeTempo250624 na particula atual. Quando a particula troca sua position testeTempo250624 fitness, seus valores sao
            # tb alterados. Acho que deveria guardar uam copai deles, ja q eles deveriam guardar aos melhores
            # nesse momento testeTempo250624 nao ao longo dos passos comoe stava ate agora

            # new_personal_best.position = particle.position
            # new_personal_best.fitness = particle.fitness
            new_personal_best.position = particle.position.copy()
            new_personal_best.fitness = particle.fitness.copy()

            particle.personal_best = []
            particle.personal_best.append(new_personal_best)
        else:
            remove_list = []
            include_flag = False
            for s in particle.personal_best:
                if particle == s:
                    break
                if particle >> s:
                    include_flag = True
                    if s not in remove_list:
                        remove_list.append(s)
                if not particle << s:
                    i = i - 1
            if len(remove_list) > 0:
                for r in remove_list:
                    particle.personal_best.remove(r)
            # tem algo estranho aqui. Se i = 1 testeTempo250624 include_flag = true, significa que a particula
            # é dominada por 1 particula do conjunto testeTempo250624 domina pelo menos 1 outra
            # tentar obter um exemplo. talvez um ou exclusivo?
            if i == 0 or include_flag:
                new_personal_best = Particle(self.params.position_min_value, self.params.position_max_value,
                                             self.params.position_dim,
                                             self.params.velocity_min_value, self.params.velocity_max_value,
                                             self.params.objectives_dim, self.params.otimizations_type,
                                             self.params.secondary_params)

                # esse trecho faz a position testeTempo250624 fitness como o emsmo objeto da particula no eprsonal best
                # testeTempo250624 na particula atual. Quando a particula troca sua position testeTempo250624 fitness, seus valores sao
                # tb alterados. Acho que deveria guardar uam copai deles, ja q eles deveriam guardar aos melhores
                # nesse momento testeTempo250624 nao ao longo dos passos comoe stava ate agora

                # new_personal_best.position = particle.position
                # new_personal_best.fitness = particle.fitness
                new_personal_best.position = particle.position.copy()
                new_personal_best.fitness = particle.fitness.copy()

                if self.params.personal_guide_array_size > 0 and len(
                        particle.personal_best) == self.params.personal_guide_array_size:
                    particle.personal_best.pop(0)
                particle.personal_best.append(new_personal_best)

    def global_best_attribution_gpu(self, use_copy_population=False):
        fitness = []

        for m in self.memory:
            fitness.append(m.fitness)
        for p in self.population:
            fitness.append(p.fitness)
        if use_copy_population:
            for c in self.population_copy:
                fitness.append(c.fitness)

        fitness = np.array(fitness, dtype=np.float64)
        sigma = np.zeros((len(fitness), 3), dtype=np.float64)
        # fitness = np.array(fitness, dtype=np.float64)
        # sigma = np.zeros((len(fitness), 3), dtype=np.float64)

        fitness_g = cuda.mem_alloc(fitness.nbytes)
        dim_g = cuda.mem_alloc(np.array(fitness.shape, dtype=np.int32).nbytes)
        sigma_g = cuda.mem_alloc(sigma.nbytes)

        cuda.memcpy_htod(fitness_g, fitness)
        cuda.memcpy_htod(dim_g, np.array(fitness.shape, dtype=np.int32))
        cuda.memcpy_htod(sigma_g, sigma)

        f = open('testes/ordenacao.cu')
        code = f.read()
        f.close()
        mod = SourceModule(code)

        sigma_eval = mod.get_function("sigma_eval")

        block_x = len(fitness)
        sigma_eval(fitness_g, dim_g, sigma_g, block=(block_x, 1, 1), grid=(1, 1, 1))
        cuda.Context.synchronize()

        cuda.memcpy_dtoh(sigma, sigma_g)
        i = 0
        for m in self.memory:
            m.sigma_value = sigma[i, :]
            i += 1
        for p in self.population:
            p.sigma_value = sigma[i, :]
            i += 1
        if use_copy_population:
            for c in self.population_copy:
                c.sigma_value = sigma[i, :]
                i += 1

        fronts = []
        tam = []
        pos_mem = []

        if use_copy_population:
            pool = np.zeros(2 * len(self.population), dtype=np.int32)
            nearest = np.zeros(2 * len(self.population), dtype=np.int32)
            pos_mem = np.zeros(2 * len(self.population), dtype=np.int32)
        else:
            pool = np.zeros(len(self.population), dtype=np.int32)
            nearest = np.zeros(len(self.population), dtype=np.int32)
            pos_mem = np.zeros(len(self.population), dtype=np.int32)

        for i in range(len(self.fronts)):
            for j in range(len(self.fronts[i])):
                fronts.append(np.where(np.array(self.population) == self.fronts[i][j])[0][0])
            tam.append(len(self.fronts[i]))
        fronts = np.array(fronts, dtype=np.int32)
        tam = np.array(tam, dtype=np.int32)

        for i in range(len(self.population)):
            if self.population[i].rank == 0:
                pool[i] = -1
                if len(np.where(np.array(self.memory) == self.population[i])[0]) == 1:
                    pos_mem[i] = np.where(np.array(self.memory) == self.population[i])[0][0]
                else:
                    pos_mem[i] = -1

            else:
                pool[i] = self.population[i].rank - 1
                pos_mem[i] = -2
        if use_copy_population:
            for i in range(len(self.population_copy)):
                if self.population_copy[i].rank == 0:
                    pool[i + len(self.population)] = -1
                    if len(np.where(np.array(self.memory) == self.population_copy[i])[0]) == 1:
                        pos_mem[i + len(self.population)] = \
                        np.where(np.array(self.memory) == self.population_copy[i])[0][0]
                    else:
                        pos_mem[i + len(self.population)] = -1
                else:
                    pool[i + len(self.population)] = self.population[i].rank - 1
                    pos_mem[i + len(self.population)] = -2

        fronts_g = cuda.mem_alloc(fronts.nbytes)
        tam_g = cuda.mem_alloc(tam.nbytes)
        tam_memoria_g = cuda.mem_alloc(np.array([1], dtype=np.int32).nbytes)
        pool_g = cuda.mem_alloc(pool.nbytes)
        nearest_g = cuda.mem_alloc(nearest.nbytes)
        pos_mem_g = cuda.mem_alloc(pos_mem.nbytes)

        cuda.memcpy_htod(fronts_g, fronts)
        cuda.memcpy_htod(tam_g, tam)
        cuda.memcpy_htod(tam_memoria_g, np.array([len(self.memory)], np.int32))
        cuda.memcpy_htod(pool_g, pool)
        cuda.memcpy_htod(nearest_g, nearest)
        cuda.memcpy_htod(pos_mem_g, pos_mem)

        f = open('testes/ordenacao.cu')
        code = f.read()
        f.close()
        mod = SourceModule(code)

        sigma_nearest = mod.get_function("sigma_nearest")
        block_x = len(self.population) + len(self.population_copy)
        sigma_nearest(fronts_g, tam_g, tam_memoria_g, pool_g, sigma_g, nearest_g, pos_mem_g,
                      block=(block_x, 1, 1), grid=(1, 1, 1))
        cuda.Context.synchronize()

        cuda.memcpy_dtoh(nearest, nearest_g)

        for i in range(len(self.population)):
            if pool[i] != -1:
                self.population[i].global_best = self.population[nearest[i]]
                # self.population[i].global_best2 = self.population[nearest[i]]
            else:
                self.population[i].global_best = self.memory[nearest[i]]
                # self.population[i].global_best2 = self.memory[nearest[i]]
        if use_copy_population:
            for i in range(len(self.population_copy)):
                if pool[i] != -1:
                    self.population_copy[i].global_best = (
                        self.population)[nearest[i + len(self.population)]]
                    # self.population_copy[i].global_best2 = (
                    #     self.population)[nearest[i + len(self.population)]]
                else:
                    self.population_copy[i].global_best = (
                        self.memory)[nearest[i + len(self.population)]]
                    # self.population_copy[i].global_best2 = (
                    #     self.memory)[nearest[i + len(self.population)]]

        # validacao global_best
        # l = []
        # for i in self.population:
        #     l.append(i.global_best == i.global_best2)
        # print('soma', sum(l), l)
        # l2 = []
        # for i in self.population_copy:
        #     l2.append(i.global_best == i.global_best2)
        # print('soma', sum(l2), l2)
        # a = 1

    def global_best_attribution(self, use_copy_population=False):
        if self.params.global_best_attribution_type == 0 or self.params.global_best_attribution_type == 1:
            for m in self.memory:
                self.sigma_eval(m)
            # Sigma com memoria apenas.
            if self.params.global_best_attribution_type == 0:
                for p in self.population:
                    self.sigma_eval(p)
                    self.sigma_nearest(p, self.memory)
                if use_copy_population:
                    for c in self.population_copy:
                        self.sigma_eval(c)
                        self.sigma_nearest(c, self.memory)
            # Sigma por fronteiras.
            if self.params.global_best_attribution_type == 1:
                for p in self.population:
                    self.sigma_eval(p)
                if use_copy_population:
                    for c in self.population_copy:
                        self.sigma_eval(c)
                for p in self.population:
                    if p.rank == 0:
                        self.sigma_nearest(p, self.memory)
                    else:
                        self.sigma_nearest(p, self.fronts[p.rank - 1])
                for c in self.population_copy:
                    if c.rank == 0:
                        self.sigma_nearest(c, self.memory)
                    else:
                        self.sigma_nearest(c, self.fronts[c.rank - 1])
            # Random na memoria - mas apenas entra no if se for 0 ou 1
            if self.params.global_best_attribution_type == 2:
                for p in self.population:
                    p.global_best = self.memory[np.random.choice(len(self.memory))]
                if use_copy_population:
                    for c in self.population_copy:
                        c.global_best = self.memory[np.random.choice(len(self.memory))]
            # Random por fronteiras
            if self.params.global_best_attribution_type == 3:
                for p in self.population:
                    if p.rank == 0:
                        p.global_best = self.memory[np.random.choice(len(self.memory))]
                    else:
                        p.global_best = self.fronts[p.rank - 1][np.random.choice(len(self.fronts[p.rank - 1]))]
                if use_copy_population:
                    for c in self.population:
                        if c.rank == 0:
                            c.global_best = self.memory[np.random.choice(len(self.memory))]
                        else:
                            c.global_best = self.fronts[c.rank - 1][np.random.choice(len(self.fronts[c.rank - 1]))]
            a = 1

    def check_stopping_criteria(self):
        # if self.params.max_fitness_eval != 0 and self.fitness_eval_count >= self.params.max_fitness_eval:
        #     self.stopping_criteria_reached = True
        # if self.params.max_iterations != 0 and self.generation_count == self.params.max_iterations:
        #     self.stopping_criteria_reached = True
        if self.generation_count == self.max_num_iters:
            self.stopping_criteria_reached = True

    def grafico_espaco_objetivo_2D(self):
        f1 = []
        f2 = []
        plt.clf()
        for i in self.population:
            f1.append(i.fitness[0])
            f2.append(i.fitness[1])

        p = get_problem('zdt1', ndim=2).pareto_front()
        plt.plot(f1, f2, 'bo', p[:, 0], p[:, 1], 'ro')
        # fig.clear()
        # axis.plot(f1, f2)
        plt.savefig('testesSimulacao/' + str(self.generation_count) + '.png')

    def run(self, func='ZDT1', teste_grafico=False, teste_tempo = False):

        # cp.cuda.Device(0).use()
        # _ = cp.zeros(1, dtype=cp.float32)
        #
        # cuda.init()
        # dev = cuda.Device(0)
        # ctx = dev.retain_primary_context()  # pega o primary context (mesmo do CuPy)
        # ctx.push()

        #teste de tempo
        cpu = 10*[0]
        gpu = 10*[0]
        teste = {}

        with tqdm(total=self.params.max_fitness_eval, leave=False) as pbar:

            self.init_population()
            prev_fitness_eval = 0

            if self.gpu:
                div = int(np.ceil(self.params.population_size / 1024))

                function = self.mod.get_function("function1")
                function(self.params.func_n_g, self.position_g, self.params.position_dim_g,
                         self.fitness_g, self.restrictions_g, self.alpha_g, self.params.population_size_g,
                    block=(int(np.ceil(self.params.population_size/div)), 1, 1),
                    grid=(div, 1, 1))
                cuda.Context.synchronize()
                self.fitness_eval_count += self.params.population_size
                self.update_personal_best_gpu()

            if self.gpu:
                # div = int(self.params.population_size/16)
                div = int(np.ceil(self.params.population_size / 32))

                fast_nondominated_sort = self.mod.get_function("fast_nondominated_sort")
                fast_nondominated_sort(self.fitness_g, self.params.objectives_dim_g,
                                       self.domination_counter_g, self.params.population_size_g,
                                       self.params.otimizations_type_g,
                                       # block=(16, 32, 1),
                                       # grid=(div, int(div/2), 1))
                                       block=(int(np.ceil(self.params.population_size / div)),
                                              int(np.ceil(self.params.population_size / div)), 1),
                                       grid=(div, div, 1))
                cuda.Context.synchronize()

                div = int(np.ceil(self.params.population_size / 1024))

                fast_nondominated_sort2 = self.mod.get_function("fast_nondominated_sort2")
                fast_nondominated_sort2(self.domination_counter_g, self.params.population_size_g,
                                        # block=(self.params.population_size, 1, 1), grid=(1, 1, 1))
                                        block=(int(np.ceil(self.params.population_size/div)), 1, 1),
                                        grid=(div, 1, 1))
                cuda.Context.synchronize()

                fast_nondominated_sort3 = self.mod.get_function("fast_nondominated_sort3")
                # fast_nondominated_sort3(self.domination_counter_g, self.params.population_size_g,
                #                         self.params.population_size_g, self.fronts_g,
                #                         self.tams_fronts_g,
                #                         self.rank_g,
                fast_nondominated_sort3(self.domination_counter_g,
                                        self.params.population_size_g,
                                        self.fronts_g,
                                        self.tams_fronts_g,
                                        self.rank_g,
                                        block=(1, 1, 1), grid=(1, 1, 1))
                cuda.Context.synchronize()

            if self.gpu:
                # o front0 ser usado para prencher a memoria
                # se o front 0 for menor que o tamanho da memoria copia ele para a memoria
                # caso contrario tem que selecionar os melhores com crowding distance
                tam_fronts = np.zeros(2*self.params.population_size, dtype=np.int32)
                cuda.memcpy_dtoh(tam_fronts, self.tams_fronts_g)

                # atualiza memoria pela GPU
                # extedner depois, já que etria que ter um front0 inicial maior que 1024 pra testar
                if tam_fronts[0] <= self.params.memory_size:

                    # o tamanho da memoria atual sera o tamanho do front 0
                    cuda.memcpy_htod(self.params.current_memory_size_g, tam_fronts[0])

                    div = int(np.ceil(tam_fronts[0]/1024))

                    # copia front 0 para as primeiras posicoes da memoria
                    memory_inicialization1 = self.mod.get_function("memory_inicialization1")
                    memory_inicialization1(self.position_g, self.fitness_g, self.fronts_g,
                                           self.params.position_dim_g, self.params.objectives_dim_g,
                                           self.params.population_size_g, self.tams_fronts_g,
                                           block=(int(np.ceil(tam_fronts[0]/div)), 1, 1),
                                           grid=(div, 1, 1))
                    cuda.Context.synchronize()
                # else:
                #     # esse trecho nunca e usado se o tamanho da memoria e igual ao da populacao
                #
                #     # o tamanho da memoria inicial sera o tamanho maximo
                #     cuda.memcpy_htod(self.params.current_memory_size_g, self.params.memory_size)
                #
                #     # zera todas as posicoes do vetor crowding distance
                #
                #     div = int(np.ceil(self.params.population_size/1024))
                #
                #     crowding_distance_inicialization = self.mod.get_function("crowding_distance_inicialization")
                #     crowding_distance_inicialization(self.crowding_distance_g,
                #                                      block=(int(np.ceil(self.params.population_size/div)), 1, 1),
                #                                      grid=(1, 1, 1))
                #     cuda.Context.synchronize()
                #
                #
                #     teste2 = np.zeros(2 * self.params.population_size, dtype=np.int32)
                #
                #     # i_g sera uma variavel em gpu que indica qual dimensao se esta calculando
                #     i_g = cuda.mem_alloc(np.array([1], np.int32).nbytes)
                #     for i in range(self.params.objectives_dim):
                #         # ordena o front 0 de acordo com o valor de fitness da dimensao i
                #         cuda.memcpy_htod(i_g, np.array([i], dtype=np.int32))
                #         front_sort = self.mod.get_function("front_sort")
                #         front_sort(self.fronts_g, self.tams_fronts_g, self.fitness_g,
                #                    self.params.objectives_dim_g, i_g, self.crowding_distance_g,
                #                    block=(1, 1, 1), grid=(1, 1, 1))
                #         cuda.Context.synchronize()
                #
                #
                #         crowding_distance = self.mod.get_function("crowding_distance")
                #         crowding_distance(self.fronts_g, self.tams_fronts_g, self.fitness_g,
                #                           self.params.objectives_dim_g, self.tams_fronts_g, i_g,
                #                           self.crowding_distance_g,
                #                           block=(int(teste[0] - 2), 1, 1), grid=(1, 1, 1))
                #     front_sort_crowding_distance = self.mod.get_function("front_sort_crowding_distance")
                #     front_sort_crowding_distance(self.fronts_g, self.tams_fronts_g,
                #                                  self.crowding_distance_g,
                #                                  block=(1, 1, 1), grid=(1, 1, 1))
                #     memory_inicialization2 = self.mod.get_function("memory_inicialization2")
                #     memory_inicialization2(self.position_g, self.fitness_g, self.fronts_g,
                #                            self.params.position_dim_g, self.params.objectives_dim_g,
                #                            self.params.population_size_g,
                #                            block=(self.params.memory_size, 1, 1), grid=(1, 1, 1))
                #     cuda.Context.synchronize()

            while not self.stopping_criteria_reached:
                # encontra os melhores globais de cada particula
                if 2 <= self.params.DE_mutation_type <= 4:
                    self.global_best_attribution()

                if self.gpu:
                    div = int(np.ceil(self.params.population_size / 1024))

                    cuda.memcpy_htod(self.seed_g, np.random.randint(0, int(1e9), dtype=np.int32))
                    differential_mutation = self.mod.get_function("differential_mutation")
                    differential_mutation(self.params.func_n_g, self.params.Xr_pool_type_g,
                                          self.params.population_size_g, self.params.memory_size_g,
                                          self.position_g, self.params.position_dim_g,
                                          self.personal_best_position_g,
                                          self.params.personal_guide_array_size_g,
                                          self.fitness_g, self.restrictions_g, self.params.objectives_dim_g,
                                          self.params.otimizations_type_g, self.xr_pool_g,
                                          self.params.DE_mutation_type_g, self.xr_list_g,
                                          self.weights_g, self.xst_g,
                                          self.params.position_min_value_g, self.params.position_max_value_g,
                                          self.params.secondary_params_g,
                                          self.xst_fitness_g, self.xst_dominate_g, self.personal_best_fitness_g,
                                          self.personal_best_velocity_g, self.personal_best_tam_g,
                                          self.update_from_differential_mutation_g, self.seed_g, self.alpha_g,
                                          block=(int(np.ceil(self.params.population_size / div)), 1, 1),
                                          grid=(div, 1, 1))
                    cuda.Context.synchronize()

                if self.gpu:

                    start = dt.now()

                    # atualizar fronts
                    # div = int(self.params.population_size/16)
                    div = int(np.ceil(self.params.population_size / 32))

                    fast_nondominated_sort = self.mod.get_function("fast_nondominated_sort")
                    fast_nondominated_sort(self.fitness_g, self.params.objectives_dim_g,
                                           self.domination_counter_g, self.params.population_size_g,
                                           self.params.otimizations_type_g,
                                           # block=(16, 32, 1),
                                           # grid=(div, int(div/2), 1))
                                           block=(int(np.ceil(self.params.population_size / div)),
                                                  int(np.ceil(self.params.population_size / div)), 1),
                                           grid=(div, div, 1))
                    cuda.Context.synchronize()

                    div = int(np.ceil(self.params.population_size / 1024))
                    fast_nondominated_sort2 = self.mod.get_function("fast_nondominated_sort2")
                    fast_nondominated_sort2(self.domination_counter_g, self.params.population_size_g,
                                            # block=(self.params.population_size, 1, 1), grid=(1, 1, 1))
                                            block=(int(np.ceil(self.params.population_size / div)), 1, 1),
                                            grid=(div, 1, 1))
                    cuda.Context.synchronize()

                    fast_nondominated_sort3 = self.mod.get_function("fast_nondominated_sort3")
                    # fast_nondominated_sort3(self.domination_counter_g, self.params.population_size_g,
                    #                         self.params.population_size_g, self.fronts_g,
                    #                         self.tams_fronts_g,
                    #                         self.rank_g,
                    fast_nondominated_sort3(self.domination_counter_g,
                                            self.params.population_size_g,
                                            self.fronts_g,
                                            self.tams_fronts_g,
                                            self.rank_g,
                                            block=(1, 1, 1), grid=(1, 1, 1))
                    cuda.Context.synchronize()

                    # atualiza memoria pela GPU
                    # inicializa o vetor front0+memoria
                    inicialize_front0_mem = self.mod.get_function("inicialize_front0_mem")
                    inicialize_front0_mem(self.fronts_g, self.front0_mem_g, self.tams_fronts_g,
                                          self.tam_front0_mem_g, self.position_g, self.params.memory_size_g,
                                          self.params.population_size_g,
                                          self.params.position_dim_g, self.params.current_memory_size_g,
                                          block=(1, 1, 1), grid=(1, 1, 1))
                    cuda.Context.synchronize()

                    # ordena o conjunto front0 mais memoria
                    tam_front = np.zeros(1, dtype=np.int32)
                    cuda.memcpy_dtoh(tam_front, self.tam_front0_mem_g)
                    div = int(np.ceil(tam_front[0] / 32))

                    # if tam_front > 32:
                    #     block_x = 32
                    #     grid_x = int(np.ceil(tam_front[0]/32))
                    # else:
                    #     block_x = int(tam_front[0])
                    #     grid_x = 1

                    # print(block_x, grid_x, tam_front)

                    # ordena o vetor front0+memoria
                    fast_nondominated_sort4 = self.mod.get_function("fast_nondominated_sort4")
                    fast_nondominated_sort4(self.fitness_g, self.params.objectives_dim_g,
                                            self.domination_counter_g, self.params.population_size_g,
                                            self.params.otimizations_type_g,
                                            self.params.objectives_dim_g,
                                            self.front0_mem_g, self.tam_front0_mem_g,
                                            # block=(int(tam_front[0]), int(tam_front[0]), 1),
                                            # grid=(1, 1, 1))
                                            # block=(block_x, block_x, 1),
                                            # grid=(grid_x, grid_x, 1))
                                            block=(32, 32, 1),
                                            grid=(div, div, 1))
                    cuda.Context.synchronize()

                    div = int(np.ceil(tam_front[0] / 1024))
                    fast_nondominated_sort5 = self.mod.get_function("fast_nondominated_sort5")
                    fast_nondominated_sort5(self.domination_counter_g,
                                            self.tam_front0_mem_g,
                                            # block=(int(tam_front[0]), 1, 1),
                                            # grid=(1, 1, 1))
                                            block=(int(np.ceil(tam_front[0] / div)),
                                                   1, 1), grid=(div, 1, 1))
                    cuda.Context.synchronize()

                    fast_nondominated_sort6 = self.mod.get_function("fast_nondominated_sort6")
                    fast_nondominated_sort6(self.domination_counter_g, self.tam_front0_mem_g,
                                            self.front0_mem_g, self.tam_front0_g, self.front0_g,
                                            block=(1, 1, 1), grid=(1, 1, 1))
                    cuda.Context.synchronize()

                    i_g = cuda.mem_alloc(np.array([1], np.int32).nbytes)
                    #melhorar depois, colocar como atributo
                    tam_front0 = np.zeros(1, dtype=np.int32)
                    cuda.memcpy_dtoh(tam_front0, self.tam_front0_g)

                    if tam_front0[0] <= self.params.memory_size:
                        cuda.memcpy_htod(self.params.current_memory_size_g, tam_front0)

                        div = int(np.ceil(tam_front0[0] / 1024))
                        # copia posicoes e fitness de front 0, que e o front 0 do conjunto front0_mem
                        # para vetores auxiliares
                        memory_inicialization2_1 = self.mod.get_function("memory_inicialization2_1")
                        memory_inicialization2_1(self.position_g, self.fitness_g, self.front0_g,
                                                 self.params.position_dim_g, self.params.objectives_dim_g,
                                                 self.params.population_size_g, self.aux_g, self.aux2_g,
                                                 self.tam_front0_g,
                                                 block=(int(np.ceil(tam_front0[0]/div)), 1, 1),
                                                 grid=(div, 1, 1))
                        cuda.Context.synchronize()

                        # move para as primeiras posicoes da memoria os valores dos vetores auxiliares
                        memory_inicialization2_2 = self.mod.get_function("memory_inicialization2_2")
                        memory_inicialization2_2(self.position_g, self.fitness_g, self.front0_g,
                                                 self.params.position_dim_g, self.params.objectives_dim_g,
                                                 self.params.population_size_g, self.aux_g, self.aux2_g,
                                                 self.tam_front0_g,
                                                 block=(int(np.ceil(tam_front0[0]/div)), 1, 1),
                                                 grid=(div, 1, 1))
                        cuda.Context.synchronize()
                    else:


                        # zerar vetor crowding distance
                        div = int(np.ceil(self.params.population_size / 1024))
                        crowding_distance_inicialization =self.mod.get_function("crowding_distance_inicialization")
                        crowding_distance_inicialization(self.crowding_distance_g,
                                                         self.params.population_size_g,
                                                         block=(int(np.ceil(self.params.population_size/div)),
                                                                1, 1),
                                                         grid=(div, 1, 1))
                        cuda.Context.synchronize()

                        div = int(np.ceil(tam_front0[0] / 1024))
                        for i in range(self.params.objectives_dim):
                            # ordena os fronts em ordem crescente de cada coordenada fitness
                            cuda.memcpy_htod(i_g, np.array([i], dtype=np.int32))
                            front_sort = self.mod.get_function("front_sort")
                            front_sort(self.front0_g, self.tam_front0_g, self.fitness_g,
                                       self.params.objectives_dim_g, i_g, self.crowding_distance_g,
                                       block=(1, 1, 1), grid=(1, 1, 1))
                            cuda.Context.synchronize()

                            # calcula crowding_distance relativo a dimensao i
                            crowding_distance = self.mod.get_function("crowding_distance")
                            crowding_distance(self.front0_g, self.tam_front0_g, self.fitness_g,
                                              self.params.objectives_dim_g, self.tams_fronts_g, i_g,
                                              self.crowding_distance_g,
                                              block=(int(np.ceil(int(tam_front0[0]/div))) - 2, 1, 1),
                                              grid=(div, 1, 1))

                        # ordena o front de acordo com o crowding distance
                        front_sort_crowding_distance = self.mod.get_function("front_sort_crowding_distance")
                        front_sort_crowding_distance(self.front0_g, self.tam_front0_g,
                                                     self.crowding_distance_g,
                                                     block=(1, 1, 1), grid=(1, 1, 1))

                        div = int(np.ceil(self.params.memory_size / 1024))

                        # copia a posicao e fitness dos melhores para os vetores auxiliares
                        memory_inicialization2_1 = self.mod.get_function("memory_inicialization2_1")
                        memory_inicialization2_1(self.position_g, self.fitness_g, self.front0_g,
                                                 self.params.position_dim_g, self.params.objectives_dim_g,
                                                 self.params.population_size_g, self.aux_g, self.aux2_g,
                                                 self.params.memory_size_g,
                                                 block=(int(np.ceil(self.params.memory_size/div)), 1, 1),
                                                 grid=(div, 1, 1))
                        cuda.Context.synchronize()

                        # copia para a memoria os valores dos vetores auxiliares
                        memory_inicialization2_2 = self.mod.get_function("memory_inicialization2_2")
                        memory_inicialization2_2(self.position_g, self.fitness_g, self.front0_g,
                                                 self.params.position_dim_g, self.params.objectives_dim_g,
                                                 self.params.population_size_g, self.aux_g, self.aux2_g,
                                                 self.params.memory_size_g,
                                                 block=(int(np.ceil(self.params.memory_size/div)), 1, 1),
                                                 grid=(div, 1, 1))
                        cuda.Context.synchronize()

                        cuda.memcpy_htod(self.params.current_memory_size_g, self.params.memory_size)
                        # try:
                        #     cuda.memcpy_htod(self.params.current_memory_size_g, self.params.memory_size)
                        # except Exception:
                        #     print(self.params.memory_size, type(self.params.memory_size))
                        #     exit(0)

                    gpu[1] += (dt.now() - start).total_seconds()

                start = dt.now()
                if self.gpu:

                    # versao 220725
                    # copia de rank
                    copy = self.mod.get_function("copy_1d")
                    div = int(np.ceil(self.params.population_size / 1024))
                    copy(self.rank_g, self.params.population_size_g,
                         block=(int(np.ceil(self.params.population_size / div)), 1, 1),
                         grid=(div, 1, 1))
                    cuda.Context.synchronize()

                    # copia de position
                    div = int(np.ceil(self.params.population_size*self.params.position_dim/1024))
                    copy = self.mod.get_function("copy_2d")
                    copy(self.position_g, self.params.population_size_g, self.params.position_dim_g,
                          block=(int(np.ceil(self.params.population_size*self.params.position_dim / div)),
                          1, 1), grid=(div, 1, 1))
                    cuda.Context.synchronize()

                    # copia de fitness
                    div = int(np.ceil(self.params.population_size * self.params.objectives_dim[0] / 1024))
                    copy(self.fitness_g, self.params.population_size_g, self.params.objectives_dim_g,
                         block=(int(np.ceil(self.params.population_size * self.params.objectives_dim[0] / div)),
                                1, 1), grid=(div, 1, 1))
                    cuda.Context.synchronize()

                    # copia de velocity
                    div = int(np.ceil(self.params.population_size * self.params.position_dim / 1024))
                    copy(self.velocity_g, self.params.population_size_g, self.params.position_dim_g,
                         block=(int(np.ceil(self.params.population_size*self.params.position_dim / div)),
                         1, 1), grid=(div, 1, 1))
                    cuda.Context.synchronize()

                    # copia de personal_best
                    copy = self.mod.get_function("copy_3d")
                    div = int(np.ceil(self.params.population_size*self.params.personal_guide_array_size*
                              self.params.position_dim/1024))
                    copy(self.personal_best_position_g, self.params.position_dim_g,
                          self.params.personal_guide_array_size_g, self.params.population_size_g,
                          block=(int(np.ceil(self.params.population_size*self.params.position_dim *
                                     self.params.personal_guide_array_size / div)),
                          1, 1), grid=(div, 1, 1))
                    cuda.Context.synchronize()

                    div = int(np.ceil(self.params.population_size *
                                      self.params.personal_guide_array_size *
                                      self.params.objectives_dim[0] / 1024))
                    copy(self.personal_best_fitness_g, self.params.objectives_dim_g,
                          self.params.personal_guide_array_size_g, self.params.population_size_g,
                         block=(int(np.ceil(self.params.population_size * self.params.objectives_dim[0] *
                                            self.params.personal_guide_array_size / div)),
                                1, 1), grid=(div, 1, 1))
                    cuda.Context.synchronize()

                    # copia de weights
                    copy = self.mod.get_function("copy_2d_vv")

                    div = int(np.ceil(self.params.population_size * 6 / 1024))
                    copy(self.weights_g, self.weights_copy_g, self.params.population_size_g,
                         block=(int(np.ceil(self.params.population_size * 6 / div)), 1, 1),
                         grid=(div, 1, 1))
                    cuda.Context.synchronize()

                # parei aqui em 220725
                gpu[2] += (dt.now() - start).total_seconds()

                start = dt.now()
                cpu[3] += (dt.now() - start).total_seconds()

                # para manter a igualdade das implementacoes, simplesmente copiei os novos vetores
                # mais tarde elembrar de  impleementar essa mutação via gpu
                if self.gpu:
                    start = dt.now()

                    # zera os pesos na gpu
                    weights = np.zeros(6 * self.params.population_size, dtype=np.float64)
                    cuda.memcpy_htod(self.weights_g, weights)
                    cuda.memcpy_htod(self.seed_g, np.random.randint(0, int(1e9), dtype=np.int32))

                    div = int(np.ceil(self.params.population_size * 6 / 1024))
                    mutate_weights = self.mod.get_function("mutate_weights")
                    mutate_weights(self.weights_g, self.seed_g, self.params.population_size_g,
                                   self.params.mutation_rate_g,
                                   block=(int(np.ceil(self.params.population_size*6 / div)), 1, 1),
                                   grid=(div, 1, 1))
                    cuda.Context.synchronize()

                    div = int(np.ceil(self.params.population_size/ 256))
                    # os pesos de 0 a 3, sao limitados entre 0 e 1
                    mutate_weights2 = self.mod.get_function("mutate_weights2")
                    mutate_weights2(self.weights_g, self.params.population_size_g,
                                   block=(4, int(np.ceil(self.params.population_size/div)), 1),
                                   grid=(1, div, 1))
                    cuda.Context.synchronize()

                    div = int(np.ceil(self.params.population_size / 1024))
                    # aplica limites aos pesos 4 e 5
                    mutate_weights3 = self.mod.get_function("mutate_weights3")
                    mutate_weights3(self.weights_g, self.params.population_size_g,
                                    block=(int(np.ceil(self.params.population_size/div)), 1, 1),
                                    grid=(div, 1, 1))
                    cuda.Context.synchronize()

                    #weights copy

                    div = int(np.ceil(self.params.population_size * 6 / 1024))
                    cuda.memcpy_htod(self.seed_g, np.random.randint(0, int(1e9), dtype=np.int32))
                    mutate_weights(self.weights_copy_g, self.seed_g, self.params.population_size_g,
                                   self.params.mutation_rate_g,
                                   block=(int(np.ceil(self.params.population_size*6 / div)), 1, 1),
                                   grid=(div, 1, 1))
                    cuda.Context.synchronize()

                    div = int(np.ceil(self.params.population_size / 256))
                    mutate_weights2(self.weights_copy_g, self.params.population_size_g,
                                    block=(4, int(np.ceil(self.params.population_size/div)), 1),
                                    grid=(1, div, 1))
                    cuda.Context.synchronize()

                    div = int(np.ceil(self.params.population_size / 1024))
                    mutate_weights3(self.weights_copy_g, self.params.population_size_g,
                                    block=(int(np.ceil(self.params.population_size)), 1, 1),
                                    grid=(div, 1, 1))
                    cuda.Context.synchronize()

                    gpu[3] += (dt.now() - start).total_seconds()

                start = dt.now()
                cpu[4] += (dt.now() - start).total_seconds()

                start = dt.now()

                if self.gpu:
                    # calculo do global_best
                    # calculo do sigma da particula i
                    total = self.params.population_size * 2 + self.params.memory_size
                    div = int(np.ceil(total/1024))

                    sigma_eval = self.mod.get_function("sigma_eval")
                    sigma_eval(self.sigma_g, self.fitness_g, self.params.objectives_dim_g,
                               block=(int(np.ceil(total/div)),1, 1),
                               grid=(div, 1, 1))
                    cuda.Context.synchronize()

                    total = self.params.population_size * 2
                    div = int(np.ceil(total / 1024))
                    # sigma_nearest = self.mod.get_function("sigma_nearest")
                    sigma_nearest = self.mod.get_function("sigma_nearest")
                    sigma_nearest(self.sigma_g, self.fronts_g, self.tams_fronts_g, self.rank_g,
                                  self.params.population_size_g, self.params.memory_size_g,
                                  self.params.objectives_dim_g, self.global_best_g, self.fitness_g,
                                  block=(int(np.ceil(total/div)), 1, 1),
                                  grid=(div, 1, 1))
                    cuda.Context.synchronize()

                gpu[4] += (dt.now() - start).total_seconds()

                start = dt.now()
                cpu[5] += (dt.now() - start).total_seconds()

                start = dt.now()

                if self.gpu:

                    # div = int(self.params.population_size/64)
                    div  = int(np.ceil(self.params.population_size / 64))
                    div2 = int(np.ceil(self.params.position_dim / 16))

                    cuda.memcpy_htod(self.seed_g, np.random.randint(0, int(1e9), dtype=np.int32))

                    #aqui uso so a populacao pq no emsmo kernel ele calcula a copia tb
                    move_particle = self.mod.get_function("move_particle")
                    move_particle(self.weights_g, self.weights_copy_g, self.personal_best_position_g,
                                  self.position_g, self.velocity_g,
                                  self.params.personal_guide_array_size_g,
                                  self.params.communication_probability_g,
                                  self.global_best_g, self.params.velocity_max_value_g,
                                  self.params.velocity_min_value_g, self.seed_g,
                                  self.params.population_size_g, self.params.position_dim_g,
                                  block=(int(np.ceil(self.params.population_size / div)),
                                         int(np.ceil(self.params.position_dim/ div2)), 1),
                                  grid=(div, div2, 1))
                    cuda.Context.synchronize()

                    # define limites apos o movimento das posicoes e velocidades
                    # div *= 2
                    # move_particle2 = self.mod.get_function("move_particle2")
                    # move_particle2(self.position_g, self.velocity_g, self.params.position_min_value_g,
                    #                self.params.position_max_value_g,
                    #                block=(int(2 * self.params.population_size / div), self.params.position_dim, 1),
                    #                grid=(div, 1, 1))
                    # cuda.Context.synchronize()

                    div *= 2
                    move_particle2 = self.mod.get_function("move_particle2")
                    move_particle2(self.position_g, self.velocity_g, self.params.position_min_value_g,
                                   self.params.position_max_value_g,
                                   self.params.population_size_g, self.params.position_dim_g,
                                   block=(int(np.ceil(2 * self.params.population_size / div)),
                                          int(np.ceil(self.params.position_dim/div2)), 1),
                                   grid=(div, 1, 1))
                    cuda.Context.synchronize()

                    div = int(np.ceil(2*self.params.population_size / 1024))
                    function = self.mod.get_function("function2")
                    function(self.params.func_n_g, self.position_g, self.params.position_dim_g,
                             self.fitness_g, self.restrictions_g, self.alpha_g,
                             self.params.population_size_g,
                             block=(int(np.ceil(2*self.params.population_size / div)), 1, 1),
                             grid=(div, 1, 1))
                    cuda.Context.synchronize()

                    # update_personal_best3 = self.mod.get_function("update_personal_best3")
                    div = int(np.ceil(2*self.params.population_size / 1024))

                    update_personal_best3 = self.mod.get_function("update_personal_best3_validation")
                    update_personal_best3(self.personal_best_position_g, self.personal_best_velocity_g,
                                          self.personal_best_fitness_g,
                                          self.params.objectives_dim_g, self.params.position_dim_g, self.position_g,
                                          self.fitness_g,
                                          self.params.personal_guide_array_size_g,
                                          self.params.otimizations_type_g,
                                          block=(int(np.ceil(2 * self.params.population_size/div)), 1, 1), grid=(div, 1, 1))
                    cuda.Context.synchronize()

                gpu[5] += (dt.now() - start).total_seconds()

                start = dt.now()
                cpu[6] += (dt.now() - start).total_seconds()

                start = dt.now()

                if self.gpu:
                    # atualizar fronts
                    div = int(np.ceil(2*self.params.population_size/32))

                    fast_nondominated_sort = self.mod.get_function("fast_nondominated_sort_copy")
                    fast_nondominated_sort(self.fitness_g, self.params.objectives_dim_g,
                                           self.domination_counter_g, self.params.population_size_g,
                                           self.params.otimizations_type_g,
                                           block=(32, 32, 1),
                                           grid=(div, div, 1))
                    cuda.Context.synchronize()

                    div = int(np.ceil(2 * self.params.population_size / 1024))
                    fast_nondominated_sort2 = self.mod.get_function("fast_nondominated_sort2_copy")
                    fast_nondominated_sort2(self.domination_counter_g, self.params.population_size_g,
                                            block=(int(np.ceil(2 * self.params.population_size/div)),
                                                   1, 1), grid=(div, 1, 1))
                    cuda.Context.synchronize()

                    fast_nondominated_sort3 = self.mod.get_function("fast_nondominated_sort3_copy")
                    fast_nondominated_sort3(self.domination_counter_g, self.params.population_size_g,
                                            self.fronts_g, self.tams_fronts_g,
                                            self.rank_g,
                                            block=(1, 1, 1), grid=(1, 1, 1))
                    cuda.Context.synchronize()

                gpu[6] += (dt.now() - start).total_seconds()

                # se nao tiver copia, nao ha seleção pra proxima geracao?
                fronts3 = []
                fronts4 = []
                broken = False

                start = dt.now()
                cpu[7] += (dt.now() - start).total_seconds()

                start = dt.now()

                if self.gpu:
                    # guarda em tam_front[tam_pop-2] o front que sera quebrado
                    # guarda em tam_front[tam_pop-1] o numero de particulas do front a ser quebrado
                    # para completar o nuemro da populacao. Se for zero significa que nao precisa quebrar o
                    # front indicado em tam_front[tam_pop-2], ja que os fronts anteriores inteiros somaram o tamanho da
                    # populacao
                    nextgen1 = self.mod.get_function("nextgen1")
                    nextgen1(self.fronts_g, self.tams_fronts_g, self.params.population_size_g,
                             block=(1, 1, 1), grid=(1, 1, 1))
                    cuda.Context.synchronize()

                    # parei aqui em 23/07/25
                    #verifica se sera preciso usar o crowding distance para quebrar o front
                    teste_tam = np.zeros(self.params.population_size, dtype=np.int32)
                    cuda.memcpy_dtoh(teste_tam, self.tams_fronts_g)
                    if teste_tam[-1] > 0:
                        # zerar vetor crowding distance

                        # parei aqui em 240725
                        # div = int(np.ceil(2*self.params.population_size/1024))
                        div = int(np.ceil(teste_tam[-1] / 1024))

                        crowding_distance_inicialization2 = self.mod.get_function(
                            "crowding_distance_inicialization2")
                        crowding_distance_inicialization2(self.crowding_distance_g,
                                                         block=(int(np.ceil(teste_tam[-1]/div)),
                                                                1, 1),
                                                         grid=(div, 1, 1))

                        # crowding_distance_inicialization2(self.crowding_distance_g,
                        #                                   block=(int(np.ceil(2 * self.params.population_size / div)),
                        #                                          1, 1),
                        #                                   grid=(div, 1, 1))

                        cuda.Context.synchronize()

                        # inicializa o vetor population_index_g,que guarda inicialmente a posicao de cada
                        # particula na populacao
                        # isso e necessario pq ao reordenar varias vezes a populacao, o indice se altera
                        # e a referencia para os fronts sao os indices atuais da sparticulas
                        # a ideia seria ordenar pro crowding distance e no final voltar a ordem inicial para evitar ter
                        # que recalcular os fronts

                        population_index_inicialization = self.mod.get_function(
                            "population_index_inicialization")
                        population_index_inicialization(self.population_index_g, self.params.population_size_g,
                                                        block=(int(np.ceil(2*self.params.population_size/div)),
                                                               1, 1),
                                                        grid=(div, 1, 1))
                        cuda.Context.synchronize()

                        # ordena o vetor de crowding distance testeTempo250624 population_index para auxiliar
                        # no calculo ddo crowding distance.

                        div = int(np.ceil(self.params.population_size/1024))
                        div2 = int(np.ceil(2*self.params.population_size / 1024))

                        # por enquanto populacao tem que ser potencia de 2.
                        # tem que etstar se o crowding funciona para mais de 1 bloco, ou seja populacao maior que 512
                        for i in range(self.params.objectives_dim[0]):
                            cuda.memcpy_htod(i_g, np.array([i], dtype=np.int32))

                            # para validacao, essa parte sera comentada, descomentar depois
                            # ordena os vetores de indice e os crowding distance de acordo com
                            # o fitness da dimensao
                            for j in range(2*self.params.population_size):
                                if j % 2 == 0:
                                    front_sort5_par = self.mod.get_function("front_sort5_par")
                                    front_sort5_par(self.fitness_g,
                                                    self.params.objectives_dim_g, i_g, self.crowding_distance_g,
                                                    self.params.population_size_g, self.population_index_g,
                                                    block=(int(np.ceil(self.params.population_size/div)), 1, 1),
                                                    grid=(div, 1, 1))
                                    cuda.Context.synchronize()
                                else:
                                    front_sort5_impar = self.mod.get_function("front_sort5_impar")
                                    front_sort5_impar(self.fitness_g,
                                                      self.params.objectives_dim_g, i_g, self.crowding_distance_g,
                                                      self.params.population_size_g, self.population_index_g,
                                                      block=(int(np.ceil(self.params.population_size)/div) - 1, 1, 1),
                                                      grid=(div, 1, 1))
                                    cuda.Context.synchronize()

                            crowding_distance4 = self.mod.get_function("crowding_distance4")
                            crowding_distance4(self.fitness_g,
                                               self.params.objectives_dim_g, self.tams_fronts_g, i_g,
                                               self.crowding_distance_g, self.params.population_size_g,
                                               self.population_index_g,
                                               block=(int(np.ceil(2*self.params.population_size/div2)) - 2, 1, 1),
                                               grid=(div2, 1, 1))

                        for j in range(self.params.population_size):
                            index_sort_par = self.mod.get_function("index_sort_par")
                            index_sort_par(self.crowding_distance_g, self.population_index_g,
                                           block=(int(self.params.population_size / 2), 1, 1), grid=(1, 1, 1))
                            cuda.Context.synchronize()

                            index_sort_impar = self.mod.get_function("index_sort_impar")
                            index_sort_impar(self.crowding_distance_g, self.population_index_g,
                                             block=(int(self.params.population_size / 2) - 1, 1, 1), grid=(1, 1, 1))
                            cuda.Context.synchronize()

                        # ordenada o front a ser quebrado em ordem decrescente de cd.
                        # apos isso, os tam_pop primeiros indices de fronts erao os selecionados
                        # para a proxima geracao
                        front_sort_crowding_distance4 = self.mod.get_function("front_sort_crowding_distance4")
                        front_sort_crowding_distance4(self.fronts_g, self.tams_fronts_g,
                                                      self.crowding_distance_g, self.params.population_size_g,
                                                      block=(1, 1, 1), grid=(1, 1, 1))
                        cuda.Context.synchronize()


                # div = 2
                div = int(np.ceil(self.params.population_size/256))
                div2 = int(np.ceil(self.params.position_dim/4))

                # copiar os selecionados para uma area auxiliar
                create_next_gen1_pop = self.mod.get_function("create_next_gen1_pop")
                create_next_gen1_pop(self.position_g, self.aux_g, self.fronts_g,
                                 self.params.population_size_g, self.params.position_dim_g,
                                 block=(int(np.ceil(self.params.population_size / div)),
                                        int(np.ceil(self.params.position_dim/ div2)), 1),
                                 grid=(div, div2, 1))
                cuda.Context.synchronize()

                # copiar os selecionados para as posicoes da populacao da proxima geracao
                create_next_gen2_pop = self.mod.get_function("create_next_gen2_pop")
                create_next_gen2_pop(self.position_g, self.aux_g,
                                 self.params.population_size_g, self.params.position_dim_g,
                                 block=(int(np.ceil(self.params.population_size/div)),
                                        int(np.ceil(self.params.position_dim/div2)), 1),
                                 grid=(div, div2, 1))
                cuda.Context.synchronize()

                # o mesmo para a velocidade
                create_next_gen1_pop = self.mod.get_function("create_next_gen1_pop")
                create_next_gen1_pop(self.velocity_g, self.aux_g, self.fronts_g,
                                 self.params.population_size_g, self.params.position_dim_g,
                                 block=(int(np.ceil(self.params.population_size / div)),
                                        int(np.ceil(self.params.position_dim/div2)), 1),
                                 grid=(div, div2, 1))
                cuda.Context.synchronize()

                create_next_gen2_pop = self.mod.get_function("create_next_gen2_pop")
                create_next_gen2_pop(self.velocity_g, self.aux_g,
                                 self.params.population_size_g, self.params.position_dim_g,
                                 block=(int(np.ceil(self.params.population_size / div)),
                                        int(np.ceil(self.params.position_dim/div2)), 1),
                                 grid=(div, div2, 1))
                cuda.Context.synchronize()

                # div = 1
                # div = 8
                div = int(np.ceil(self.params.population_size/256))
                div2 = int(np.ceil(self.params.position_dim * self.params.personal_guide_array_size / 4))

                create_next_gen1_pb = self.mod.get_function("create_next_gen1_pb")
                create_next_gen1_pb(self.personal_best_position_g, self.aux3_g, self.fronts_g,
                                 self.params.population_size_g, self.params.position_dim_g,
                                 self.params.personal_guide_array_size_g,
                                 block=(int(np.ceil(self.params.population_size / div)),
                                        int(np.ceil(self.params.position_dim *
                                                    self.params.personal_guide_array_size/div2)),
                                                    1),
                                 grid=(div, div2, 1))
                cuda.Context.synchronize()

                create_next_gen2_pb = self.mod.get_function("create_next_gen2_pb")
                create_next_gen2_pb(self.personal_best_position_g, self.aux3_g,
                                 self.params.population_size_g, self.params.position_dim_g,
                                 self.params.personal_guide_array_size_g,
                                 block=(int(np.ceil(self.params.population_size / div)),
                                        int(np.ceil(self.params.position_dim
                                                    * self.params.personal_guide_array_size/div2)),
                                        1),
                                 grid=(div, div2, 1))
                cuda.Context.synchronize()

                # div = 2
                # div = int(self.params.population_size/64)
                div = int(np.ceil(self.params.population_size / 256))
                div2 = int(np.ceil(self.params.position_dim * self.params.personal_guide_array_size / 4))
                create_next_gen1_pb = self.mod.get_function("create_next_gen1_pb")
                create_next_gen1_pb(self.personal_best_fitness_g, self.aux3_g, self.fronts_g,
                                 self.params.population_size_g, self.params.objectives_dim_g,
                                 self.params.personal_guide_array_size_g,
                                 block=(int(np.ceil(self.params.population_size / div)),
                                        int(np.ceil(self.params.objectives_dim[0] *
                                                    self.params.personal_guide_array_size/div2)), 1),
                                 grid=(div, div2, 1))
                cuda.Context.synchronize()

                create_next_gen2_pb = self.mod.get_function("create_next_gen2_pb")
                create_next_gen2_pb(self.personal_best_fitness_g, self.aux3_g,
                                    self.params.population_size_g, self.params.objectives_dim_g,
                                    self.params.personal_guide_array_size_g,
                                 block=(int(np.ceil(self.params.population_size / div)),
                                        int(np.ceil(self.params.objectives_dim[0] *
                                                    self.params.personal_guide_array_size/div2)), 1),
                                 grid=(div, div2, 1))
                cuda.Context.synchronize()

                gpu[7] += (dt.now() - start).total_seconds()

                #atualziacao de memoria
                start = dt.now()
                if self.gpu:
                    #reavalia a nova populacao - depois ver com calma como retirar as copias e as memorias(260525)
                    div = int(np.ceil(self.params.population_size / 1024))
                    function = self.mod.get_function("function1")
                    function(self.params.func_n_g, self.position_g, self.params.position_dim_g,
                             self.fitness_g, self.restrictions_g, self.alpha_g, self.params.population_size_g,
                             block=(int(np.ceil(self.params.population_size / div)), 1, 1),
                             grid=(div, 1, 1))
                    cuda.Context.synchronize()
                    # self.fitness_eval_count += 2 * self.params.population_size + self.params.memory_size

                    # atualizar fronts - testar isso depois(260625)
                    # div1 = int(self.params.population_size / 16)
                    # div2 = int(self.params.population_size / 32)
                    # print(div1, div2)

                    # ordenar fronts
                    # div = int(self.params.population_size/16)
                    div = int(np.ceil(self.params.population_size / 32))

                    fast_nondominated_sort = self.mod.get_function("fast_nondominated_sort")
                    fast_nondominated_sort(self.fitness_g, self.params.objectives_dim_g,
                                           self.domination_counter_g, self.params.population_size_g,
                                           self.params.otimizations_type_g,
                                           # block=(16, 32, 1),
                                           # grid=(div, int(div/2), 1))
                                           block=(int(np.ceil(self.params.population_size / div)),
                                                  int(np.ceil(self.params.population_size / div)), 1),
                                           grid=(div, div, 1))
                    cuda.Context.synchronize()

                    div = int(np.ceil(self.params.population_size / 1024))
                    fast_nondominated_sort2 = self.mod.get_function("fast_nondominated_sort2")
                    fast_nondominated_sort2(self.domination_counter_g, self.params.population_size_g,
                                            # block=(self.params.population_size, 1, 1), grid=(1, 1, 1))
                                            block=(int(np.ceil(self.params.population_size / div)),
                                            1, 1), grid=(div, 1, 1))
                    cuda.Context.synchronize()

                    fast_nondominated_sort3 = self.mod.get_function("fast_nondominated_sort3")
                    fast_nondominated_sort3(self.domination_counter_g,
                                            self.params.population_size_g,
                                            self.fronts_g,
                                            self.tams_fronts_g,
                                            self.rank_g,
                                            block=(1, 1, 1), grid=(1, 1, 1))
                    cuda.Context.synchronize()

                    # tam_front = np.zeros(256, dtype=np.int32)
                    # cuda.memcpy_dtoh(tam_front, self.tams_fronts_g)
                    # print(tam_front)

                    # tam_front = np.zeros(256, dtype=np.int32)
                    # cuda.memcpy_dtoh(tam_front, self.tams_fronts_g)
                    # print(tam_front)
                    # tam_front = np.zeros(256, dtype=np.int32)
                    # cuda.memcpy_dtoh(tam_front, self.fronts_g)
                    # print(tam_front)

                    # inicilaiza o vetor front0+memoria
                    inicialize_front0_mem = self.mod.get_function("inicialize_front0_mem")
                    inicialize_front0_mem(self.fronts_g, self.front0_mem_g, self.tams_fronts_g,
                                           self.tam_front0_mem_g, self.position_g, self.params.memory_size_g,
                                           self.params.population_size_g,
                                           self.params.position_dim_g, self.params.current_memory_size_g,
                                           block=(1, 1, 1), grid=(1, 1, 1))
                    cuda.Context.synchronize()



                    # if tam_front > 32:
                    #     block_x = 32
                    #     grid_x = int(np.ceil(tam_front[0]/32))
                    # else:
                    #     block_x = int(tam_front[0])
                    #     grid_x = 1

                    # print(block_x, grid_x, tam_front)

                    # ordena o conjunto front0 mais memoria
                    tam_front = np.zeros(1, dtype=np.int32)
                    cuda.memcpy_dtoh(tam_front, self.tam_front0_mem_g)
                    div = int(np.ceil(tam_front[0] / 32))

                    # ordena o vetor front0+memoria
                    fast_nondominated_sort4 = self.mod.get_function("fast_nondominated_sort4")
                    fast_nondominated_sort4(self.fitness_g, self.params.objectives_dim_g,
                                            self.domination_counter_g, self.params.population_size_g,
                                            self.params.otimizations_type_g,
                                            self.params.objectives_dim_g,
                                            self.front0_mem_g, self.tam_front0_mem_g,
                                            # block=(int(tam_front[0]), int(tam_front[0]), 1),
                                            # grid=(1, 1, 1))
                                            # block=(block_x, block_x, 1),
                                            # grid=(grid_x, grid_x, 1))
                                            block=(int(np.ceil(tam_front[0] / div)),
                                                   int(np.ceil(tam_front[0] / div)), 1),
                                            grid=(div, div, 1))
                    cuda.Context.synchronize()

                    div = int(np.ceil(tam_front[0] / 1024))
                    fast_nondominated_sort5 = self.mod.get_function("fast_nondominated_sort5")
                    fast_nondominated_sort5(self.domination_counter_g,
                                            self.tam_front0_mem_g,
                                            # block=(int(tam_front[0]), 1, 1),
                                            # grid=(1, 1, 1))
                                            block=(int(np.ceil(tam_front[0] / div)),
                                                   1, 1), grid=(div, 1, 1))
                    cuda.Context.synchronize()

                    fast_nondominated_sort6 = self.mod.get_function("fast_nondominated_sort6")
                    fast_nondominated_sort6(self.domination_counter_g, self.tam_front0_mem_g,
                                            self.front0_mem_g, self.tam_front0_g, self.front0_g,
                                            block=(1, 1, 1), grid=(1, 1, 1))
                    cuda.Context.synchronize()


                    #parei aqui 250725 - 17h36
                    tam_front = np.zeros(1, dtype=np.int32)
                    cuda.memcpy_dtoh(tam_front, self.tam_front0_g)

                    # atualiza memoria pela GPU
                    # se o front 0 do vetor front0+memoria for menor que a memoria
                    if tam_front[0] <= self.params.memory_size:
                        cuda.memcpy_htod(self.params.current_memory_size_g, tam_front[0])

                        # copia as particulas, copias e memoria atual para vetores auxiliares
                        copy3 = self.mod.get_function("copy3")
                        total = 2 * self.params.population_size + self.params.memory_size
                        div = int(np.ceil(total/1024))

                        copy3(self.position_g, self.aux_g, self.params.position_dim_g,
                              self.params.population_size_g, self.params.memory_size_g,
                              block=(int(np.ceil(total/div)), 1, 1),
                              grid=(div, 1, 1))
                        cuda.Context.synchronize()
                        copy3(self.velocity_g, self.aux2_g, self.params.position_dim_g,
                              self.params.population_size_g, self.params.memory_size_g,
                              block=(int(np.ceil(total/div)), 1, 1),
                              grid=(div, 1, 1))
                        cuda.Context.synchronize()

                        # coloca 1e20 em todas as posicoes e fitness de todas as particulas,
                        # copias e memoria
                        div = int(np.ceil(self.params.memory_size / 1024))
                        memory_inicialization4 = self.mod.get_function("memory_inicialization4")
                        memory_inicialization4(self.position_g, self.fitness_g,
                                               self.params.position_dim_g, self.params.objectives_dim_g,
                                               self.params.population_size_g, self.aux_g, self.params.memory_size_g,
                                               block=(int(np.ceil(self.params.memory_size/div)), 1, 1),
                                               grid=(div, 1, 1))
                        cuda.Context.synchronize()

                        #inicializa a nova memoria com o front0
                        div = int(np.ceil(tam_front[0] / 1024))
                        memory_inicialization5 = self.mod.get_function("memory_inicialization5")
                        memory_inicialization5(self.position_g, self.velocity_g, self.fitness_g,
                                               self.aux_g, self.aux2_g, self.front0_g,
                                               self.params.position_dim_g, self.params.objectives_dim_g,
                                               self.params.population_size_g,
                                               block=(int(np.ceil(tam_front[0])/div), 1, 1),
                                               grid=(div, 1, 1))
                        cuda.Context.synchronize()

                        # atualizar fitness da memoria nova
                        div = int(np.ceil(self.params.population_size / 1024))
                        function = self.mod.get_function("function1")
                        function(self.params.func_n_g, self.position_g, self.params.position_dim_g,
                                 self.fitness_g, self.restrictions_g, self.alpha_g, self.params.population_size_g,
                                 block=(int(np.ceil(self.params.population_size / div)), 1, 1),
                                 grid=(div, 1, 1))
                        cuda.Context.synchronize()
                        self.fitness_eval_count += 2*self.params.population_size+self.params.memory_size
                    else:
                        # falta testar o else quando o exemplo chegar aqui

                        # zerar vetor crowding distance
                        div = int(np.ceil(self.params.population_size/1024))
                        crowding_distance_inicialization = self.mod.get_function("crowding_distance_inicialization")
                        crowding_distance_inicialization(self.crowding_distance_g,
                                                        block=(int(np.ceil(self.params.population_size/1024)), 1, 1),
                                                        grid=(div, 1, 1))
                        cuda.Context.synchronize()

                        # para cada dimensao ordena o front 0 de acordo coma  dimensao e
                        # calcula a parcela do cd
                        i_g = cuda.mem_alloc(np.array([1], np.int32).nbytes)
                        total = tam_front[0]-2
                        div = int(np.ceil(total/1024))
                        for i in range(self.params.objectives_dim):
                            # ordena os fronts em ordem crescente de cada coordenada fitness
                            cuda.memcpy_htod(i_g, np.array([i], dtype=np.int32))
                            front_sort = self.mod.get_function("front_sort")
                            front_sort(self.front0_g, self.tam_front0_g, self.fitness_g,
                                       self.params.objectives_dim_g, i_g, self.crowding_distance_g,
                                       block=(1, 1, 1), grid=(1, 1, 1))
                            cuda.Context.synchronize()

                            crowding_distance = self.mod.get_function("crowding_distance")
                            crowding_distance(self.front0_g, self.tam_front0_g, self.fitness_g,
                                              self.params.objectives_dim_g, self.tams_fronts_g, i_g,
                                              self.crowding_distance_g,
                                              block=(int(np.ceil(total/div)), 1, 1),
                                              grid=(div, 1, 1))

                        # ordena por ordem decresente de cd o front0 e o vetor cd
                        front_sort_crowding_distance = self.mod.get_function("front_sort_crowding_distance")
                        front_sort_crowding_distance(self.front0_g, self.tam_front0_g,
                                                     self.crowding_distance_g,
                                                     block=(1, 1, 1), grid=(1, 1, 1))

                        # copia posicao e velocidade para vetores temporarios
                        total = 2 * self.params.population_size + self.params.memory_size
                        div = int(np.ceil(total/1024))

                        cuda.memcpy_htod(self.params.current_memory_size_g, tam_front[0])
                        copy3 = self.mod.get_function("copy3")
                        copy3(self.position_g, self.aux_g, self.params.position_dim_g,
                              self.params.population_size_g, self.params.memory_size_g,
                              block=(int(np.ceil(total/div)), 1, 1),
                              grid=(div, 1, 1))
                        cuda.Context.synchronize()
                        copy3(self.velocity_g, self.aux2_g, self.params.position_dim_g,
                              self.params.population_size_g, self.params.memory_size_g,
                              block=(int(np.ceil(total/div)), 1, 1),
                              grid=(div, 1, 1))
                        cuda.Context.synchronize()

                        # coloca 1e20 na sposicoes e fitness
                        div = int(np.ceil(self.params.memory_size/1024))
                        memory_inicialization4 = self.mod.get_function("memory_inicialization4")
                        memory_inicialization4(self.position_g, self.fitness_g,
                                               self.params.position_dim_g, self.params.objectives_dim_g,
                                               self.params.population_size_g, self.aux_g,
                                               block=(int(np.ceil(self.params.memory_size/div)), 1, 1),
                                               grid=(div, 1, 1))
                        cuda.Context.synchronize()

                        # inicializa a nova memoria
                        memory_inicialization5 = self.mod.get_function("memory_inicialization5")
                        memory_inicialization5(self.position_g, self.velocity_g, self.fitness_g,
                                               self.aux_g, self.aux2_g, self.front0_g,
                                               self.params.position_dim_g, self.params.objectives_dim_g,
                                               self.params.population_size_g,
                                               block=(int(np.ceil(self.params.memory_size/div)), 1, 1),
                                               grid=(div, 1, 1))
                        cuda.Context.synchronize()

                        # atualizar fitness da memoria nova
                        div = int(np.ceil(self.params.population_size / 1024))
                        function = self.mod.get_function("function1")
                        function(self.params.func_n_g, self.position_g, self.params.position_dim_g,
                                 self.fitness_g, self.restrictions_g, self.alpha_g, self.params.population_size_g,
                                 block=(int(np.ceil(self.params.population_size / div)), 1, 1),
                                 grid=(div, 1, 1))
                        cuda.Context.synchronize()

                        self.fitness_eval_count += 2*self.params.population_size +self.params.memory_size
                        cuda.memcpy_htod(self.params.current_memory_size_g,
                                         np.array([self.params.memory_size], dtype=np.int32))

                gpu[8] += (dt.now() - start).total_seconds()

                # Fim do loop principal.

                delta_evals = self.fitness_eval_count - prev_fitness_eval
                pbar.update(delta_evals)
                prev_fitness_eval = self.fitness_eval_count

                self.generation_count = self.generation_count + 1
                self.check_stopping_criteria()

            # armazena os dados dessa semulacao no arquivo pkl
            f = open('results.pkl', 'rb')
            results = pickle.load(f)
            f.close()

            count = results['count']
            count+=1
            results['count'] = count

            fitness = np.zeros((2 * self.params.population_size + self.params.memory_size) * self.params.objectives_dim[0],
                               dtype=np.float64)
            cuda.memcpy_dtoh(fitness, self.fitness_g)
            fitness.shape = 2 * self.params.population_size + self.params.memory_size, self.params.objectives_dim[0]

            position = np.zeros((2 * self.params.population_size + self.params.memory_size) * self.params.position_dim,
                               dtype=np.float64)
            cuda.memcpy_dtoh(position, self.position_g)
            position.shape = 2 * self.params.population_size + self.params.memory_size, self.params.position_dim

            cur = np.zeros(1, dtype=np.int32)
            cuda.memcpy_dtoh(cur, self.params.current_memory_size_g)

            fronts = np.zeros(2*self.params.population_size, dtype=np.int32)
            cuda.memcpy_dtoh(fronts, self.fronts_g)

            tam = np.zeros(2*self.params.population_size, dtype=np.int32)
            cuda.memcpy_dtoh(tam, self.tams_fronts_g)

            results[count] = (position, fitness, cur, fronts, tam, self.params.func_n, self.params.objectives_dim)

            f = open('results.pkl', 'wb')
            pickle.dump(results, f)
            f.close()

            return gpu