#include <curand.h>
#include <curand_kernel.h>

typedef struct{
    int *objectives_dim;
    int *otimizations_type;
    int *max_iterations;
    int *max_fitness_eval;
    int *position_dim;
    double *position_max_value;
    double *position_min_value;
    double *velocity_min_value;
    double *velocity_max_value;
    int *population_size;
    int *memory_size;
    int *memory_update_type;
    int *global_best_attribution_type;
    int *DE_mutation_type;
    int *Xr_pool_type;
    int *crowd_distance_type;
    double *communication_probability;
    double *mutation_rate;
    int *personal_guide_array_size;
    int *secondary_params;
    int *initial_state;
}MESH_PARAMS;

struct particle{
//         int *maximize;
        int *crowd_distance;
        int *rank;
//         double *pos_min;
//         double *pos_max;
//         double *vel_min;
//         double *vel_max;
        double *fitness;
        double *position;
        double *velocity;
//         int *objectives_dim;
        int *domination_counter;
        struct particle *dominated_set;
        double *sigma_value;
        struct particle *global_best;
        struct particle *personal_best;
//         int *secondary_params;
};
typedef struct particle PARTICLE;


typedef struct{
    MESH_PARAMS *params;
    int *stopping_criteria_reached;
    int *generation_count;
//     Particle *population;
//     Particle *population_copy;
//     particle *memory;
//     Particle **fronts;
    int *fitness_eval_count;
    double *weights;
    double *weights_copy;
    int *update_from_differential_mutation;
//     char *log_memory;
    int *copy_pop;
}MESH;

// typedef struct{
//     PARTICLE*population;
//     PARTICLE *population_copy;
//     PARTICLE *memory;
//     PARTICLE *fronts;
//     int *tam;
// }POPULATIONS;

extern "C" {
__global__ void test_mesh_params(MESH_PARAMS *m)
{
//     printf("%d\n", (int)sizeof(int));
    printf("\nobjectives_dim = %d\n", *(m->objectives_dim));
    printf("otimizations_type = %d %d %d\n", m->otimizations_type[0],
    m->otimizations_type[1], m->otimizations_type[2]);
    printf("max_iterations = %d\n", *(m->max_iterations));
    printf("max_fitness_eval = %d\n", *(m->max_fitness_eval));
    printf("position_dim = %d\n", *(m->position_dim));
    printf("position_max_value = %lf %lf %lf\n", m->position_max_value[0],
    m->position_max_value[1], m->position_max_value[2]);
    printf("position_min_value = %lf %lf %lf\n", m->position_min_value[0],
    m->position_min_value[1], m->position_min_value[2]);
    printf("velocity_min_value = %lf %lf %lf\n", m->velocity_min_value[0],
    m->velocity_min_value[1], m->velocity_min_value[2]);
    printf("velocity_max_value = %lf %lf %lf\n", m->velocity_max_value[0],
    m->velocity_max_value[1], m->velocity_max_value[2]);
    printf("population_size = %d\n", *(m->population_size));
    printf("memory_size = %d\n", *(m->memory_size));
    printf("memory_update_type = %d\n", *(m->memory_update_type));
    printf("global_best_attribution_type = %d\n", *(m->global_best_attribution_type));
    printf("DE_mutation_type = %d\n", *(m->DE_mutation_type));
    printf("Xr_pool_type = %d\n", *(m->Xr_pool_type));
    printf("crowd_distance_type = %d\n", *(m->crowd_distance_type));
    printf("communication_probability = %lf\n", *(m->communication_probability));
    printf("mutation_rate = %lf\n", *(m->mutation_rate));
    printf("personal_guide_array_size = %d\n", *(m->personal_guide_array_size));
    printf("secondary_params = %d\n", *(m->secondary_params));
    printf("initial_state = %d\n", *(m->initial_state));
}

__device__ void test_mesh_params2(MESH_PARAMS *m)
{
//     printf("%d\n", (int)sizeof(int));
    printf("objectives_dim = %d\n", *(m->objectives_dim));
    printf("otimizations_type = %d %d %d\n", m->otimizations_type[0],
    m->otimizations_type[1], m->otimizations_type[2]);
    printf("max_iterations = %d\n", *(m->max_iterations));
    printf("max_fitness_eval = %d\n", *(m->max_fitness_eval));
    printf("position_dim = %d\n", *(m->position_dim));
    printf("position_max_value = %lf %lf %lf\n", m->position_max_value[0],
    m->position_max_value[1], m->position_max_value[2]);
    printf("position_min_value = %lf %lf %lf\n", m->position_min_value[0],
    m->position_min_value[1], m->position_min_value[2]);
    printf("velocity_min_value = %lf %lf %lf\n", m->velocity_min_value[0],
    m->velocity_min_value[1], m->velocity_min_value[2]);
    printf("velocity_max_value = %lf %lf %lf\n", m->velocity_max_value[0],
    m->velocity_max_value[1], m->velocity_max_value[2]);
    printf("population_size = %d\n", *(m->population_size));
    printf("memory_size = %d\n", *(m->memory_size));
    printf("memory_update_type = %d\n", *(m->memory_update_type));
    printf("global_best_attribution_type = %d\n", *(m->global_best_attribution_type));
    printf("DE_mutation_type = %d\n", *(m->DE_mutation_type));
    printf("Xr_pool_type = %d\n", *(m->Xr_pool_type));
    printf("crowd_distance_type = %d\n", *(m->crowd_distance_type));
    printf("communication_probability = %lf\n", *(m->communication_probability));
    printf("mutation_rate = %lf\n", *(m->mutation_rate));
    printf("personal_guide_array_size = %d\n", *(m->personal_guide_array_size));
    printf("secondary_params = %d\n", *(m->secondary_params));
    printf("initial_state = %d\n\n", *(m->initial_state));
}

__global__ void test_mesh(MESH *m)
{
    printf("stopping_criteria_reached = %d\n", *(m->stopping_criteria_reached));
    printf("generation_count = %d\n", *(m->generation_count));
    printf("weights1 = %lf %lf %lf\n", m->weights[0*(*m->params->population_size)+0],
    m->weights[0*(*m->params->population_size)+1], m->weights[0*(*m->params->population_size)+2]);
    printf("weights2 = %lf %lf %lf\n", m->weights[1*(*m->params->population_size)+0],
    m->weights[1*(*m->params->population_size)+1], m->weights[1*(*m->params->population_size)+2]);
    printf("weights3 = %lf %lf %lf\n", m->weights[2*(*m->params->population_size)+0],
    m->weights[2*(*m->params->population_size)+1], m->weights[2*(*m->params->population_size)+2]);
    printf("weights4 = %lf %lf %lf\n", m->weights[3*(*m->params->population_size)+0],
    m->weights[3*(*m->params->population_size)+1], m->weights[3*(*m->params->population_size)+2]);
    printf("weights5 = %lf %lf %lf\n", m->weights[4*(*m->params->population_size)+0],
    m->weights[4*(*m->params->population_size)+1], m->weights[4*(*m->params->population_size)+2]);
    printf("weights6 = %lf %lf %lf\n\n", m->weights[5*(*m->params->population_size)+0],
    m->weights[5*(*m->params->population_size)+1], m->weights[5*(*m->params->population_size)+2]);
    printf("weights1c = %lf %lf %lf\n", m->weights_copy[0*(*m->params->population_size)+0],
    m->weights_copy[0*(*m->params->population_size)+1], m->weights_copy[0*(*m->params->population_size)+2]);
    printf("weights2c = %lf %lf %lf\n", m->weights_copy[1*(*m->params->population_size)+0],
    m->weights_copy[1*(*m->params->population_size)+1], m->weights_copy[1*(*m->params->population_size)+2]);
    printf("weights3c = %lf %lf %lf\n", m->weights_copy[2*(*m->params->population_size)+0],
    m->weights_copy[2*(*m->params->population_size)+1], m->weights_copy[2*(*m->params->population_size)+2]);
    printf("weights4c = %lf %lf %lf\n", m->weights_copy[3*(*m->params->population_size)+0],
    m->weights_copy[3*(*m->params->population_size)+1], m->weights_copy[3*(*m->params->population_size)+2]);
    printf("weights5c = %lf %lf %lf\n", m->weights_copy[4*(*m->params->population_size)+0],
    m->weights_copy[4*(*m->params->population_size)+1], m->weights_copy[4*(*m->params->population_size)+2]);
    printf("weights6c = %lf %lf %lf\n\n", m->weights_copy[5*(*m->params->population_size)+0],
    m->weights_copy[5*(*m->params->population_size)+1], m->weights_copy[5*(*m->params->population_size)+2]);
    printf("update_from_differential_mutation = %d\n", *(m->update_from_differential_mutation));
    printf("copy_pop = %d\n", *(m->copy_pop));

//     test_mesh_params2(m->params);
}

__global__ void test_position(double* position, int*dim)
{
    printf("\n");
    for(int i=0;i<3;i+=1)
    {
        for(int j=0;j<dim[0];j++)
        {
            printf("%0.3lf ", position[i*10+j]);
        }
        printf("\n");
    }
}

__device__ void zdt1_device(double *position, int *position_dim, double *fitness, int i)
{
    int j;
    fitness[i*2+0] = position[i*position_dim[0]+0];

    fitness[i*2+1] = 0;
    for(j=1;j<position_dim[0];j++)
    {
        fitness[i*2+1] += position[i*position_dim[0]+j];
    }
    fitness[i*2+1] *= 9;
    fitness[i*2+1] /= (position_dim[0]-1);
    fitness[i*2+1] += 1;
    fitness[i*2+1] *=  1- sqrt(fitness[i*2+0]/fitness[i*2+1]);
}

__device__ void zdt2_device(double *position, int *position_dim, double *fitness, int i)
{
    int j;
    fitness[i*2+0] = position[i*position_dim[0]+0];

    fitness[i*2+1] = 0;
    for(j=1;j<position_dim[0];j++)
    {
        fitness[i*2+1] += position[i*position_dim[0]+j];
    }
    fitness[i*2+1] *= 9;
    fitness[i*2+1] /= (position_dim[0]-1);
    fitness[i*2+1] += 1;
    fitness[i*2+1] *=  1- (fitness[i*2+0]/fitness[i*2+1])*(fitness[i*2+0]/fitness[i*2+1]);
}

__device__ void zdt3_device(double *position, int *position_dim, double *fitness, int i)
{
    double pi = 3.141592;
    int j;
    fitness[i*2+0] = position[i*position_dim[0]+0];

    fitness[i*2+1] = 0;
    for(j=1;j<position_dim[0];j++)
    {
        fitness[i*2+1] += position[i*position_dim[0]+j];
    }
    fitness[i*2+1] *= 9;
    fitness[i*2+1] /= (position_dim[0]-1);
    fitness[i*2+1] += 1;
    fitness[i*2+1] *=  1- sqrt(fitness[i*2+0]/fitness[i*2+1]) -
    fitness[i*2+0]/fitness[i*2+1] * sinf(10*pi*position[i*position_dim[0]+0]);
}

__device__ void zdt4_device(double *position, int *position_dim, double *fitness, int i)
{
    int j;
    double temp;
    double pi = 3.141592;
    fitness[i*2+0] = position[i*position_dim[0]+0];

    fitness[i*2+1] = 0;
    for(j=1;j<position_dim[0];j++)
    {
        temp = position[i*position_dim[0]+j];
        temp *= temp;
        temp -= 10*cosf(4*pi*position[i*position_dim[0]+j]);
        fitness[i*2+1] += temp;
    }
    fitness[i*2+1] += 10*(position_dim[0]-1);
    fitness[i*2+1] += 1;
    fitness[i*2+1] *=  1- sqrt(fitness[i*2+0]/fitness[i*2+1]);
}

__device__ int v_u(int u)
{
    if(u<5)
    {
        return 2+u;
    }
    else
    {
        return 1;
    }
}

__device__ void zdt5(double *position, int *position_dim, double *fitness, int i)
{
    int j;
    double x0 = roundf(position[i*position_dim[0]+0]);
    int ux0 = __popc(x0);
    int v;

//     if(i==0)
//     {
        fitness[i*2+0] = 1+ux0;
//         printf("%lf %lf\n",position[i*position_dim[0]+0], fitness[i*2+0]);

        fitness[i*2+1] = 0;
        for(j=1;j<position_dim[0];j++)
        {
            v = v_u(__popc(roundf(position[i*position_dim[0]+j])));
            fitness[i*2+1] += v;
//             printf("%lf %d\n", position[i*position_dim[0]+j], v);
        }
        fitness[i*2+1] *= 1/fitness[i*2+0];
//     }
}

__device__ void zdt6(double *position, int *position_dim, double *fitness, int i)
{
    int j;
    double pi = 3.141592;
    double x0 = position[i*position_dim[0]+0];
    fitness[i*2+0] = 1-expf(-4*x0)*powf(sinf(6*pi*x0), 6);

    fitness[i*2+1] = 0;
    for(j=1;j<position_dim[0];j++)
    {
        fitness[i*2+1] += position[i*position_dim[0]+j];
    }
    fitness[i*2+1] /=9;
    fitness[i*2+1] = pow(fitness[i*2+1], 0.25);
    fitness[i*2+1] *= 9;
    fitness[i*2+1] += 1;
    fitness[i*2+1] *=  1- pow(fitness[i*2+0]/fitness[i*2+1], 2);
}

__device__ double g_dtlz1(double *position, int *position_dim, int i)
{
    int M = 3,j=0, current=position_dim[0]-1;
    int k = position_dim[0]-M+1;
    double image=0, temp, pi = 3.14159265358979323846;
//     double norm=0;

    while(j<k)
    {
//         if(i==0)
//         {
//             printf("cur = %d\n", current);
//         }
        temp=position[i*position_dim[0]+current]-0.5;
        temp*=temp;
        temp -= cos(20*pi*(position[i*position_dim[0]+current]-0.5));
        image +=temp;
//         norm += position[i*position_dim[0]+current]*position[i*position_dim[0]+current];
        j++;
        current--;
    }
//     norm = sqrtf(norm);
//     image+=norm;
    image+=k;
    image*=100;
    return image;
}

__device__ void dtlz1(double *position, int *position_dim, double *fitness, int i)
{
    double g;

//     if(i==0)
//     {
//         for(int j=0;j<10;j++)
//         {
//             printf("%lf ", position[j]);
//         }
//         printf("\n");
//     }

    g = g_dtlz1(position, position_dim, i);
//     if(i==0)
//     {
//         printf("g=%lf\n",g);
//     }
    g=g+1;
//     if(i==0)
//     {
//         printf("g=%lf\n",g);
//     }
//     printf("%d\n",i);
//     if(i==90)
//     {
//         printf("%lf %lf %lf\n", fitness[i*3+0], fitness[i*3+1], fitness[i*3+2]);
//     }

    fitness[i*3+0] = position[i*position_dim[0]+0]*position[i*position_dim[0]+1];
    fitness[i*3+0] *= 0.5*g;

    fitness[i*3+1] = position[i*position_dim[0]+0];
    fitness[i*3+1] *= (1-position[i*position_dim[0]+1]);
    fitness[i*3+1] *= 0.5*g;

    fitness[i*3+2] = 1-position[i*position_dim[0]+0];
    fitness[i*3+2] *= 0.5*g;

//     if(i==90)
//     {
//         printf("%lf %lf %lf\n", fitness[i*3+0], fitness[i*3+1], fitness[i*3+2]);
//     }
}

__device__ double g_dtlz2(double *position, int *position_dim, int i)
{
    int M = 3,j=0;
    double image=0;

    for(j=M-1;j<position_dim[0];j++)
    {
        image += (position[i*position_dim[0]+j]-0.5)*(position[i*position_dim[0]+j]-0.5);
    }
    return image;
}

__device__ void dtlz2(double *position, int *position_dim, double *fitness, int i)
{
    double g, pi = 3.14159265358979323846;
    g = g_dtlz2(position, position_dim, i);

    g=g+1;

    fitness[i*3+0] =  cos(0.5*pi*position[i*position_dim[0]+0]);
    fitness[i*3+0] *= cos(0.5*pi*position[i*position_dim[0]+1]);
    fitness[i*3+0] *= g;

    fitness[i*3+1] =  cos(0.5*pi*position[i*position_dim[0]+0]);
    fitness[i*3+1] *= sin(0.5*pi*position[i*position_dim[0]+1]);
    fitness[i*3+1] *= g;

    fitness[i*3+2] = sin(0.5*pi*position[i*position_dim[0]+0])*g;
}

__device__ void dtlz3(double *position, int *position_dim, double *fitness, int i)
{
    double g, pi = 3.14159265358979323846;
    g = g_dtlz1(position, position_dim, i);

    g=g+1;

    fitness[i*3+0] =  cos(0.5*pi*position[i*position_dim[0]+0]);
    fitness[i*3+0] *= cos(0.5*pi*position[i*position_dim[0]+1]);
    fitness[i*3+0] *= g;

    fitness[i*3+1] =  cos(0.5*pi*position[i*position_dim[0]+0]);
    fitness[i*3+1] *= sin(0.5*pi*position[i*position_dim[0]+1]);
    fitness[i*3+1] *= g;

    fitness[i*3+2] = sin(0.5*pi*position[i*position_dim[0]+0])*g;
}

__device__ void dtlz4(double *position, int *position_dim, double *fitness, int i)
{
    double g, pi = 3.14159265358979323846, alpha=100;

    g = g_dtlz2(position, position_dim, i);

    g=g+1;

    fitness[i*3+0] =  cos(0.5*pi*pow(position[i*position_dim[0]+0], alpha));
    fitness[i*3+0] *= cos(0.5*pi*pow(position[i*position_dim[0]+1], alpha));
    fitness[i*3+0] *= g;

    fitness[i*3+1] =  cos(0.5*pi*pow(position[i*position_dim[0]+0], alpha));
    fitness[i*3+1] *= sin(0.5*pi*pow(position[i*position_dim[0]+1], alpha));
    fitness[i*3+1] *= g;

    fitness[i*3+2] = sin(0.5*pi*pow(position[i*position_dim[0]+0], alpha))*g;
}

__device__ double theta(double g, double x, int i)
{
    double result=0;
//     double pi = 3.141592;

//     if(i==0)
//     {
//         printf("%lf\n", result);
//         result = 1+2*g*x;
//         printf("%lf %lf %lf\n", result, g, x);
//     }
//     if(i==0)
//     {
//         printf("%lf\n", result);
//         result = 1+2*g*x;
//         printf("%lf %lf %lf\n", result, g, x);
//     }

//     if(i==0)
//     {
//         printf("result = %lf\n", result);
//     }
    result = 1.0+2.0*g*x;
//     if(i==0)
//     {
//         printf("result = %lf %lf %lf\n", result, g, x);
//     }
    result /= (1.0+g);
//     if(i==0)
//     {
//         printf("result = %lf %lf %lf\n", result, g, x);
//         printf("den = %lf\n", (1.0+g)*2);
//     }
    result *= 0.5;
//     result *= pi/4;
//     if(i==0)
//     {
//         printf("result = %lf %lf %lf\n", result, g, x);
//     }

    return result;
}

__device__ void dtlz5(double *position, int *position_dim, double *fitness, int i)
{
    double g=0, pi = 3.14159265358979323846, theta0=0, theta1=0;

    g = g_dtlz2(position, position_dim, i);

    theta0 = position[i*position_dim[0]+0];
    theta1 = theta(g, position[i*position_dim[0]+1],i);

    fitness[i*3+0] =  cos(0.5*pi*theta0);
    fitness[i*3+0] *= cos(0.5*pi*theta1);
    fitness[i*3+0] *= (g+1);

    fitness[i*3+1] =  cos(0.5*pi*theta0);
    fitness[i*3+1] *= sin(0.5*pi*theta1);
    fitness[i*3+1] *= (g+1);

    fitness[i*3+2] = sin(0.5*pi*theta0)*(g+1);
}

__device__ double g_dtlz6(double *position, int *position_dim, int i)
{
    int M = 3,j=0;
    double image=0;

    for(j=M-1;j<position_dim[0];j++)
    {
        image += pow(position[i*position_dim[0]+j], 0.1);
    }
    return image;
}

__device__ void dtlz6(double *position, int *position_dim, double *fitness, int i)
{
    double g=0, pi = 3.141592, theta0=0, theta1=0;

    g = g_dtlz6(position, position_dim, i);

    theta0 = position[i*position_dim[0]+0];
    theta1 = theta(g, position[i*position_dim[0]+1],i);

    fitness[i*3+0] =  cos(0.5*pi*theta0);
    fitness[i*3+0] *= cos(0.5*pi*theta1);
    fitness[i*3+0] *= (g+1);

    fitness[i*3+1] =  cos(0.5*pi*theta0);
    fitness[i*3+1] *= sin(0.5*pi*theta1);
    fitness[i*3+1] *= (g+1);

    fitness[i*3+2] = sin(0.5*pi*theta0)*(g+1);
}

__device__ double g_dtlz7(double *position, int *position_dim, int i)
{
    int M = 3,j=0;
    double image=0;

    for(j=M-1;j<position_dim[0];j++)
    {
        image += position[i*position_dim[0]+j];
    }
    image *=9;
    image/= (position_dim[0] - M + 1);
    image++;

    return image;
}

__device__ double h(double fitness0, double fitness1, double g)
{
    int M = 3;
    double image=0, temp, pi = 3.14159265358979323846;

    temp = fitness0/(1+g);
    temp *= (1+sin(3*pi*fitness0));
    image += temp;

    temp = fitness1/(1+g);
    temp *= (1+sin(3*pi*fitness1));
    image += temp;

    image = M - image;

    return image;
}

__device__ void dtlz7(double *position, int *position_dim, double *fitness, int i)
{
    double g=0, h2;

    g = g_dtlz7(position, position_dim, i);

    fitness[i*3+0] =  position[i*position_dim[0]+0];
    fitness[i*3+1] =  position[i*position_dim[0]+1];

    h2 = h(fitness[i*3+0], fitness[i*3+1], g);

    fitness[i*3+2] = (1+g)*h2;
}

__device__ double g1_mw(int m, int n, double *position, int i)
{
    double g1 = 0, temp;
    int j;

    for(j=m-1;j<n;j++)
    {
        temp = pow(position[i*n+j], (n-m));
//         printf("%lf ", temp);
        temp -= 0.5;
        temp -= ((double)(j)/(2*(double)n));
        temp *= temp;
        temp *= -10;
        temp = 1-exp(temp);
//         printf("%lf ", temp);

        g1 += temp;
    }
//     printf("\n");
    return g1+1;
}

__device__ double g2_mw(int m, int n, double *position, int i)
{
    double g2 = 0, temp, z, pi = 3.14159265358979323846;
    int j;

    for(j=m-1;j<n;j++)
    {
        z = (double)j;
        z = z/((double)n);
        z = position[i*n+j] - z;
        z = z * z;
        z = z * (-10.);
        z = 1.0-exp(z);

        temp = z * z * 0.1;
        temp = temp/((double)n);
        temp = temp - 1.5*cos(2*pi*z);
        temp = temp + 1.5;
        g2 = g2 + temp;
    }
    return g2+1;
}

__device__ double g3_mw(int m, int n, double *position, int i)
{
    double g3 = 0, temp;
    int j;

    for(j=m-1;j<n;j++)
    {
        temp = position[(i-1)*n+j]-0.5;
        temp *= temp;
        temp += position[(i*n+j)];
        temp -= 1;
        temp *= temp;
        temp *= 2;
        g3 += temp;
    }
    return g3+1;
}

__device__ double restricao(double a, double b, double c, double d, double e)
{
    double pi = 3.14159265358979323846;

    return a*pow(sin(b*pi*pow(c, d)), e);
}

__device__ double restricao2(double a, double b, double c, double d, double e)
{
//     return a*powf(sinf(b*powf(c, d)), e);
    return a*pow(sin(b*pow(c, d)), e);
}

__device__ void mw1(double *position, int *position_dim, double *fitness, int i, double *alpha)
{
    double g=0, l, c, pi = 3.14159265358979323846;

    g = g1_mw(2, position_dim[0], position, i);

//     printf("%d p0=%lf p1=%lf %lf\n", i,position[i*10], fitness[i*10+1], g);

    fitness[i*2+0] =  position[i*position_dim[0]+0];
    fitness[i*2+1] =  g*(1-0.85*fitness[i*2+0]/g);

    l = sqrt(2.0)*fitness[i*2+1] - sqrt(2.0) * fitness[i*2+0];
    c = fitness[i*2+1] + fitness[i*2+0] -1 - 0.5*pow(sin(2*pi*l), 8);

//     printf("%d %lf %lf %lf\n", i, fitness[i*2], fitness[i*2+1], g);
    if(c>0)
    {
//         printf("%d %lf %lf\n", i, c, g);
//         printf("%d \n", i);
//         fitness[i*2+0] += alpha[0]*c;
//         fitness[i*2+1] += alpha[0]*c;
//         fitness[i*2+0] = 1.0 + c;
//         fitness[i*2+1] = 1.5 + c;
        fitness[i*2+0] += alpha[0]*c;
        fitness[i*2+1] += alpha[0]*c;
    }
}

__device__ void mw2(double *position, int *position_dim, double *fitness, int i, double *alpha)
{
    double g=0, l, c, pi = 3.141592;

    g = g2_mw(2, position_dim[0], position, i);

    fitness[i*2+0] =  position[i*position_dim[0]+0];
    fitness[i*2+1] =  g*(1-fitness[i*2+0]/g);

    l = sqrt(2.0)*fitness[i*2+1] - sqrt(2.0) * fitness[i*2+0];
    c = fitness[i*2+1] + fitness[i*2+0] -1 - 0.5*pow(sin(3*pi*l), 8);

    if(c>0)
    {
        fitness[i*2+0] = 1.0 + c;
        fitness[i*2+1] = 1.5 + c;
//         fitness[i*2+0] += alpha[0]*c;
//         fitness[i*2+1] += alpha[0]*c;
    }
}

__device__ void mw3(double *position, int *position_dim, double *fitness, int i, double *alpha)
{
    double g=0, l, c[2], pi = 3.141592;

    g = g3_mw(2, position_dim[0], position, i);

    fitness[i*2+0] =  position[i*position_dim[0]+0];
    fitness[i*2+1] =  g*(1-fitness[i*2+0]/g);

    l = sqrt(2.0)*fitness[i*2+1] - sqrt(2.0) * fitness[i*2+0];
    c[0] = fitness[i*2+1] + fitness[i*2+0] -1.05 - 0.45*pow(sin(0.75*pi*l), 6);
    c[1] = 0.85 - fitness[i*2+1] - fitness[i*2+0] + 0.3*pow(sin(0.75*pi*l), 2);

//     if(c[0] > 0 || c[1] > 0)
//     {
//         fitness[i*2+0] = 1.0;
//         fitness[i*2+1] = 1.5;
//     }
    if(c[0]>0)
    {
//         fitness[i*2+0] += alpha[0]*c[0];
//         fitness[i*2+1] += alpha[0]*c[0];
        fitness[i*2+0] += c[0];
        fitness[i*2+1] += c[0];
    }
    if(c[1]>0)
    {
//         fitness[i*2+0] += alpha[0]*c[1];
//         fitness[i*2+1] += alpha[0]*c[1];
        fitness[i*2+0] += c[1];
        fitness[i*2+1] += c[1];
    }
}


__device__ void mw4(double *position, int *position_dim, double *fitness, int i, double *alpha)
{
    double g=0, l, c;

    g = g1_mw(3, position_dim[0], position, i);

    fitness[i*3+0] = g*(1-position[i*position_dim[0]+0])*(1-position[i*position_dim[0]+1]);
    fitness[i*3+1] = g*position[i*position_dim[0]+1] * (1-position[i*position_dim[0]+0]);
    fitness[i*3+2] = g*position[i*position_dim[0]+0];

    l = fitness[i*3+2] - fitness[i*3+0] - fitness[i*3+1];
    c = restricao(-0.4, 2.5, l, 1,  8);
    c += 1;
    c += fitness[i*3+0];
    c += fitness[i*3+1];
    c += fitness[i*3+2];

    if(c>0)
    {
        fitness[i*3+0] += alpha[0]*c;
        fitness[i*3+1] += alpha[0]*c;
        fitness[i*3+2] += alpha[0]*c;
    }
}

__device__ void mw5(double *position, int *position_dim, double *fitness, int i, double *alpha)
{
    double g=0, l[2], c[3], pi = 3.141592;

    g = g1_mw(2, position_dim[0], position, i);

    fitness[i*2+0] =  g*position[i*position_dim[0]+0];
    fitness[i*2+1] =  g*sqrt(1-pow((fitness[i*2+0]/g), 2));

    l[0] = atan(fitness[i*2+1]/fitness[i*2+0]);
    l[1] = 0.5*pi -2*abs(l[0]-0.25*pi);

    c[0] = restricao2(-0.2, 2, l[0], 1, 1) + 1.7;
    c[0] *= c[0];
    c[0] *= -1;
    c[0] += fitness[i*2+0] * fitness[i*2+0];
    c[0] += fitness[i*2+1] * fitness[i*2+1];

    c[1] = restricao2(0.5, 6, l[1], 3, 1) + 1.0;
    c[1] *= c[1];
    c[1] -= fitness[i*2+0] * fitness[i*2+0];
    c[1] -= fitness[i*2+1] * fitness[i*2+1];

    c[2] = restricao2(-0.45, 6, l[1], 3, 1) + 1.0;
    c[2] *= c[2];
    c[2] -= fitness[i*2+0] * fitness[i*2+0];
    c[2] -= fitness[i*2+1] * fitness[i*2+1];


//     if((c[0] > 0 || c[1] > 0 || c[2]>0))
//     {
//         fitness[i*2+0] = 2.0;
//         fitness[i*2+1] = 2.0;
//     }

    if(c[0]>0)
    {
//         fitness[i*2+0] += alpha[0]*c[0];
//         fitness[i*2+1] += alpha[0]*c[0];
        fitness[i*2+0] += c[0];
        fitness[i*2+1] += c[0];
    }
    if(c[1]>0)
    {
//         fitness[i*2+0] += alpha[0]*c[1];
//         fitness[i*2+1] += alpha[0]*c[1];
        fitness[i*2+0] += c[1];
        fitness[i*2+1] += c[1];
    }
    if(c[2]>0)
    {
//         fitness[i*2+0] += alpha[0]*c[2];
//         fitness[i*2+1] += alpha[0]*c[2];
        fitness[i*2+0] += c[2];
        fitness[i*2+1] += c[2];
    }
}

__device__ void mw6(double *position, int *position_dim, double *fitness, int i, double *alpha)
{
    double g=0, l, c=0, temp;

    g = g2_mw(2, position_dim[0], position, i);

    fitness[i*2+0] =  g*position[i*position_dim[0]+0];
    fitness[i*2+1] =  g*sqrt(1.1*1.1-pow((fitness[i*2+0]/g), 2));

//     l = fitness[i*2+1]/fitness[i*2+0];
//     l = powf(l, 4);
//     l = atanf(l);
    l = atan(fitness[i*2+1]/fitness[i*2+0]);
    l = pow(l, 4);
    l = l * 6;
    l = cos(l);
    l = pow(l, 10);

    temp = 1.0+0.15*l;
    temp = fitness[i*2+0]/temp;
    temp = temp * temp;
    c = c + temp;
    temp = 1.0+0.75*l;
    temp = fitness[i*2+1]/temp;
    temp = temp * temp;
    c = c + temp;
    c = c - 1.0;

    if(c>0)
    {
        fitness[i*2+0] = 1.5 + c;
        fitness[i*2+1] = 2.0 + c;
    }
//      if(c>0)
//     {
//         fitness[i*2+0] += alpha[0]*c;
//         fitness[i*2+1] += alpha[0]*c;
//     }
}

__device__ void mw7(double *position, int *position_dim, double *fitness, int i, double *alpha)
{
    double g=0, l, c[2];

    g = g3_mw(2, position_dim[0], position, i);

    fitness[i*2+0] =  g*position[i*position_dim[0]+0];
//     fitness[i*2+1] =  g*sqrtf(1-powf((fitness[i*2+0]/g), 2));
    fitness[i*2+1] =  g*sqrt(1-pow((fitness[i*2+0]/g), 2));

//     l = atanf(fitness[i*2+1]/fitness[i*2+0]);
    l = atan(fitness[i*2+1]/fitness[i*2+0]);

    c[0] = restricao2(0.4, 4, l, 1, 16) + 1.2;
    c[0] *= c[0];
    c[0] *= -1;
    c[0] += fitness[i*2+0] * fitness[i*2+0];
    c[0] += fitness[i*2+1] * fitness[i*2+1];

    c[1] = restricao2(-0.2, 4, l, 1, 8) + 1.15;
    c[1] *= c[1];
    c[1] -= fitness[i*2+0] * fitness[i*2+0];
    c[1] -= fitness[i*2+1] * fitness[i*2+1];

//     if(c[0] > 0 || c[1] > 0)
//     {
//         fitness[i*2+0] = 2.0;
//         fitness[i*2+1] = 2.0;
//     }

    if(c[0]>0)
    {
//         fitness[i*2+0] += alpha[0]*c[0];
//         fitness[i*2+1] += alpha[0]*c[0];
        fitness[i*2+0] += c[0];
        fitness[i*2+1] += c[0];
    }
    if(c[1]>0)
    {
        fitness[i*2+0] += c[1];
        fitness[i*2+1] += c[1];
    }
}

__device__ void mw9(double *position, int *position_dim, double *fitness, int i,
double *alpha)
{
    double g, T[3];

    g = g1_mw(2, position_dim[0], position, i);

    fitness[i*2+0] =  g*position[i*position_dim[0]+0];
    fitness[i*2+1] =  g*(1-powf((fitness[i*2+0]/g), 0.6));

    T[0] = fitness[i*2+0]*fitness[i*2+0];
    T[0] *= -0.64;
    T[0] -= fitness[i*2+1];
    T[0] += 1;
    T[1] = fitness[i*2+0]*fitness[i*2+0];
    T[1] *= -0.36;
    T[1] -= fitness[i*2+1];
    T[1] += 1;
    T[0] = T[0]*T[1];

    T[1] = fitness[i*2+0] + 0.35;
    T[1] *= T[1];
    T[1] *= -1;
    T[1] -= fitness[i*2+1];
    T[1] += powf(1.35, 2);

    T[2] = fitness[i*2+0] + 0.15;
    T[2] *= T[2];
    T[2] *= -1;
    T[2] -= fitness[i*2+1];
    T[2] += powf(1.15, 2);

    T[1] *= T[2];

//     T[0] guarda o minimo entre T[0 e T[1]
    if(T[0]>T[1])
    {
        T[0] = T[1];
    }

    if(T[0]>0)
    {
        if(alpha[0]<0)
        {
            fitness[i*2+0] = 1.1 + T[0];
            fitness[i*2+1] = 1.1 + T[0];
        }
        else
        {
            fitness[i*2+0] += alpha[0]*T[0];
            fitness[i*2+1] += alpha[0]*T[0];
        }
    }
}

__device__ void mw10(double *position, int *position_dim, double *fitness, int i, double *alpha)
{
    double g=0, c[3], temp;

    g = g2_mw(2, position_dim[0], position, i);

    fitness[i*2+0] =  g*powf(position[i*position_dim[0]+0], position_dim[0]);
    fitness[i*2+1] =  g*(1-powf((fitness[i*2+0]/g), 2));

    c[0] = fitness[i*2+0] * fitness[i*2+0];
    c[0] *= -4;
    c[0] -= fitness[i*2+1];
    c[0] += 2;

    temp = fitness[i*2+0] * fitness[i*2+0];
    temp *= -8;
    temp -= fitness[i*2+1];
    temp += 2;
    c[0] *= temp;
    c[0] *= -1;

    c[1] = fitness[i*2+0] * fitness[i*2+0];
    c[1] *= -2;
    c[1] -= fitness[i*2+1];
    c[1] += 2;

    temp = fitness[i*2+0] * fitness[i*2+0];
    temp *= -16;
    temp -= fitness[i*2+1];
    temp += 2;
    c[1] *= temp;

    c[2] = fitness[i*2+0] * fitness[i*2+0];
    c[2] *= -1;
    c[2] -= fitness[i*2+1];
    c[2] += 1;

    temp = fitness[i*2+0] * fitness[i*2+0];
    temp *= -1.2;
    temp -= fitness[i*2+1];
    temp += 1.2;
    c[2] *= temp;

    if((c[0] > 0 || c[1] > 0))
    {
        fitness[i*2+0] = 1.1;
        fitness[i*2+1] = 1.5;
    }

    if(c[0]>0)
    {
//         fitness[i*2+0] += alpha[0]*c[0];
//         fitness[i*2+1] += alpha[0]*c[0];
        fitness[i*2+0] += c[0];
        fitness[i*2+1] += c[0];
    }
    if(c[1]>0)
    {
//         fitness[i*2+0] += alpha[0]*c[1];
//         fitness[i*2+1] += alpha[0]*c[1];
        fitness[i*2+0] += c[1];
        fitness[i*2+1] += c[1];
    }
    if(c[2]>0)
    {
//         fitness[i*2+0] += alpha[0]*c[2];
//         fitness[i*2+1] += alpha[0]*c[2];
        fitness[i*2+0] += c[2];
        fitness[i*2+1] += c[2];
    }
}

// __device__ double s_linear(double y, double a) {
//     double tmp = fabs(y - a);
//     return tmp / fabs(floor(a - y) + a);
// }
//
// __device__ double b_flat(double y, double a, double b, double c) {
//     double tmp = 0.0;
//
//     if (y > b && y < c) {
//         return a;
//     }
//
//     if (y <= b) {
//         tmp = a * (b - y) / b;
//     } else { // y >= c
//         tmp = (1.0 - a) * (y - c) / (1.0 - c);
//     }
//
//     return a + tmp;
// }
//
// __device__ double b_poly(double y, double alpha) {
//     return pow(y, alpha);
// }
//
// __device__ double convex1(double *x) {
//     double pi = 3.141592653589793;
//     return 1.0 - cos(x[0] * pi / 2.0);
// }
//
// __device__ double mixed_m(double *x, double alpha, double A) {
//     double pi = 3.141592653589793;
//     double tmp = cos(2.0 * pi * A * x[0] + pi / 2.0) / (2.0 * pi * A);
//     return pow(1.0 - x[0] - tmp, alpha);
// }
//
// __device__ void wfg1(double *position, int *position_dim, double *fitness, int i) {
//     const int k = 4;
//     const int M = 2;
//
//     double y[100];
//     double x[M];
//     double s[M] = {2.0, 4.0};
//
//     // Step 1: normalize
//     for (int j = 0; j < position_dim[0]; j++) {
//         double upper = 2.0 * (j + 1);
//         y[j] = position[i * position_dim[0] + j] / upper;
//     }
//
//     // Step 2: t1 - s_linear on y[k..n-1]
//     for (int j = k; j < position_dim[0]; j++) {
//         y[j] = s_linear(y[j], 0.35);
//     }
//
//     // Step 3: t2 - b_flat on y[k..n-1]
//     for (int j = k; j < position_dim[0]; j++) {
//         y[j] = b_flat(y[j], 0.8, 0.75, 0.85);
//     }
//
//     // Step 4: t3 - b_poly on all y
//     for (int j = 0; j < position_dim[0]; j++) {
//         y[j] = b_poly(y[j], 0.02);
//     }
//
//     // Step 5: reduction - r_sum for two groups
//
//     // Group 1: j = 0..k-1
//     double num = 0.0, denom = 0.0;
//     for (int j = 0; j < k; j++) {
//         double w = 2.0 * (j + 1);
//         num += w * y[j];
//         denom += w;
//     }
//     y[0] = num / denom;
//
//     // Group 2: j = k..n-1
//     num = 0.0;
//     denom = 0.0;
//     for (int j = k; j < position_dim[0]; j++) {
//         double w = 2.0 * (j + 1);
//         num += w * y[j];
//         denom += w;
//     }
//     y[1] = num / denom;
//
//     // Step 6: construct x
//     x[0] = fmax(y[1], 1.0) * (y[0] - 0.5) + 0.5;
//     x[1] = y[1];
//
//     // Step 7: compute objectives
//     fitness[i * M + 0] = x[1] + s[0] * convex1(x);
//     fitness[i * M + 1] = x[1] + s[1] * mixed_m(x, 1.0, 5.0);
// }


// my code
__device__ double s_linear(double y, double a)
{
    double temp, temp2;
    temp = floor(a-y);
    temp = temp+a;
    temp = fabs(temp);
    temp2 = fabs(y-a);
    return temp2/temp;
}

__device__ double b_flat(double y, double a, double b, double c)
{
    double temp, temp2, temp3;

    temp = b-y;
    temp = temp*a;
    temp = temp/b;
    temp2 = y-b;
    temp2 = floor(temp2);
    temp2 = fmin(0.0, temp2);
    temp = temp*temp2;

    temp2 = (1.0-a)*(y-c);
    temp2 = temp2/(1.0-c);
    temp3 = c-y;
    temp3 = floor(temp3);
    temp3 = fmin(0.0, temp3);
    temp2 = temp2*temp3;

    return a+temp-temp2;

}

__device__ double b_poly(double y, double a)
{
    return pow(y, a);
}

__device__ double convex1(double *x)
{
    double pi = 3.141592653589793;
    return 1-cos(x[0]*pi/2.0);
}

__device__ double mixed_m(double *x, double alpha, double a)
{
    double pi = 3.141592653589793, temp;

    temp = cos(2*a*pi*x[0]+pi/2.0)/(2*a*pi);
    temp = 1-x[0]-temp;
    temp = pow(temp, alpha);
    return temp;
}

__device__ void wfg1(double *position, int *position_dim, double *fitness, int i)
{
    double y[100], xu, temp, x[2], s[2], temp2, a;
    int j, k=4, start, stop;

//     for(j=0;j<10;j++)
//     {
//         position[0] = j*0.1;
//     }

    for(j=0;j<position_dim[0];j++)
    {
        xu = (double)(j+1)*2.0;
        y[j] = position[i*position_dim[0]+j]/xu;
    }

//     for(j=0;j<position_dim[0];j++)
//     {
//         xu = (double)(j+1)*2.0;
//         y[j] = position[i*position_dim[0]+j]*xu;
//     }

//     if(i==0)
//     {
//         printf("z0 = %lf z1 = %lf\n", y[0], y[1]);
//     }

    for(j=k;j<position_dim[0];j++)
    {
        y[j] = s_linear(y[j], 0.35);
    }
//     if(i==0)
//     {
//         printf("y4 = %0.2lf y5 = %0.2lf y6 = %0.2lf y7 = %0.2lf y8 = %0.2lf y9 = %0.2lf\n", y[4], y[5], y[6], y[7], y[8], y[9]);
//     }
    for(j=k;j<position_dim[0];j++)
//     for(j=0;j<k;j++)
    {
        y[j] =b_flat(y[j], 0.8, 0.75, 0.85);
    }
//     if(i==0)
//     {
//         printf("y4 = %0.2lf y5 = %0.2lf y6 = %0.2lf y7 = %0.2lf y8 = %0.2lf y9 = %0.2lf\n", y[4], y[5], y[6], y[7], y[8], y[9]);
//     }
    for(j=0;j<position_dim[0];j++)
    {
        y[j] =b_poly(y[j], 0.02);
    }
//     if(i==0)
//     {
//         printf("y4 = %0.2lf y5 = %0.2lf y6 = %0.2lf y7 = %0.2lf y8 = %0.2lf y9 = %0.2lf\n", y[4], y[5], y[6], y[7], y[8], y[9]);
//     }


//     gap = (double)k/((double)m-1.0);
    temp = 0;
    temp2 = 0;
    start = 0;
    stop = 4;

    for(j=start; j<stop; j++)
    {
        temp = temp + y[j]* (j+1)*2;
        temp2 = temp2 + ((double)j+1.0)*2.0;
    }
    y[0] = temp/temp2;

    temp = 0;
    temp2 = 0;
    for(j=k; j<position_dim[0]; j++)
    {
        temp = temp + y[j]* ((double)j+1.0)*2.0;
        temp2 = temp2 + ((double)j+1.0)*2.0;
    }

    y[1] = temp/temp2;

//     if(i==0)
//     {
//         printf("y0 = %0.2lf y1 = %0.2lf\n", y[0], y[1]);
//     }

//     x[0] = fmaxf(1.0, y[1]);
//     x[0] = x[0]*(y[0]-0.5);
//     x[0] = x[0]+0.5;
//     x[1] = y[1];

//     for(j = 0; j < 2; j++)
//     {
//         double a = 1.0;
//         x[j] = fmax(y[1], a) * (y[j] - 0.5) + 0.5;
//     }
    a = 1.0;
    x[0] = fmax(y[1], a) * (y[0] - 0.5) + 0.5;
    x[1] = y[1];

//     if(i==0)
//     {
//         printf("x0 = %lf x1 = %lf\n", x[0], x[1]);
//     }

    s[0]=2;
    s[1]=4;

    fitness[i*2+0] = x[1]+s[0]*convex1(x);
    fitness[i*2+1] = x[1]+s[1]*mixed_m(x, 1.0, 5.0);
//     if(i==0)
//     {
//         printf("%lf %lf\n", fitness[i*2+0], fitness[i*2+1]);
//     }
}

__device__ void function(int *func_n, double *position, int *position_dim,
 double *fitness, double *alpha, int population_size)
{
//     int i = threadIdx.x;
    int i = blockIdx.x*blockDim.x+threadIdx.x;

    if(i<population_size)
    {
        if(func_n[0] == 1)
        {
            dtlz1(position, position_dim, fitness, i);
        }
        if(func_n[0] == 2)
        {
            dtlz2(position, position_dim, fitness, i);
        }
        if(func_n[0] == 3)
        {
            dtlz3(position, position_dim, fitness, i);
        }
        if(func_n[0] == 4)
        {
            dtlz4(position, position_dim, fitness, i);
        }
        if(func_n[0] == 5)
        {
            dtlz5(position, position_dim, fitness, i);
        }
        if(func_n[0] == 6)
        {
            dtlz6(position, position_dim, fitness, i);
        }
        if(func_n[0] == 7)
        {
            dtlz7(position, position_dim, fitness, i);
        }
        if(func_n[0] == 11)
        {
            zdt1_device(position, position_dim, fitness, i);
        }
        if(func_n[0] == 12)
        {
            zdt2_device(position, position_dim, fitness, i);
        }
        if(func_n[0] == 13)
        {
            zdt3_device(position, position_dim, fitness, i);
        }
        if(func_n[0] == 14)
        {
            zdt4_device(position, position_dim, fitness, i);
        }
        if(func_n[0] == 15)
        {
            zdt5(position, position_dim, fitness, i);
        }
        if(func_n[0] == 16)
        {
            zdt6(position, position_dim, fitness, i);
        }
        if(func_n[0] == 21)
        {
            wfg1(position, position_dim, fitness, i);
        }
        if(func_n[0] == 31)
        {
            mw1(position, position_dim, fitness, i, alpha);
        }
        if(func_n[0] == 32)
        {
            mw2(position, position_dim, fitness, i, alpha);
        }
        if(func_n[0] == 33)
        {
            mw3(position, position_dim, fitness, i, alpha);
        }
        if(func_n[0] == 34)
        {
            mw4(position, position_dim, fitness, i, alpha);
        }
        if(func_n[0] == 35)
        {
            mw5(position, position_dim, fitness, i, alpha);
        }
        if(func_n[0] == 36)
        {
            mw6(position, position_dim, fitness, i, alpha);
        }
        if(func_n[0] == 37)
        {
            mw7(position, position_dim, fitness, i, alpha);
        }
        if(func_n[0] == 39)
        {
            mw9(position, position_dim, fitness, i, alpha);
        }
        if(func_n[0] == 310)
        {
            mw10(position, position_dim, fitness, i, alpha);
        }
    }
}

__global__ void function1(int *func_n, double *position, int *position_dim,
 double *fitness, double *alpha, int *population_size)
{
    function(func_n, position, position_dim, fitness, alpha, population_size[0]);
}

__global__ void function2(int *func_n, double *position, int *position_dim,
 double *fitness, double *alpha, int *population_size)
{
    function(func_n, position, position_dim, fitness, alpha, 2*population_size[0]);
}

__device__ int a_dominate_b(double *fitness1, double *fitness2, int dim, int *maximize)
{
    int i=0, temp=0;

    while(i<dim)
    {
        if(maximize[i] == 0)
        {
//             printf("fitness1[0]=%lf, fitness1[1]=%lf, fitness2[0]=%lf, fitness2[1]=%lf\n",
//              fitness1[0], fitness1[1],fitness2[0], fitness2[1]);
//             if (self.fitness[i]-other.fitness[i])>1e-6
            if(fitness1[i] > fitness2[i])
//             if((fitness1[i] - fitness2[i])>1e-6)
            {
                return 0;
            }
            else
            {
//                 (self.fitness[i]-other.fitness[i])<-1e-6:
                if(fitness1[i] < fitness2[i])
//                 if((fitness1[i] - fitness2[i])<-1e-6)
                {
                    temp = 1;
                }
            }
        }
        else
        {
            if(fitness1[i] < fitness2[i])
//             if((fitness1[i] - fitness2[i])<-1e-6)
            {
                return 0;
            }
            else
            {
                if(fitness1[i] > fitness2[i])
//                 if((fitness1[i] - fitness2[i])>1e-6)
                {
                    temp = 1;
                }
            }
        }
        i++;
    }
    return temp;
}

__global__ void fast_nondominated_sort(double *fitness, int *dim_fitness, int *domination_counter,
int *population_size, int *minimization)
{
    int l = blockIdx.x*blockDim.x+threadIdx.x;
    int c = blockIdx.y*blockDim.y+threadIdx.y;
//     int tam = gridDim.y*blockDim.y;

//     domination_counter[l*tam+c] = 0;
//     domination_counter[l*tam+c] = a_dominate_b(fitness+l*colunas2[0], fitness+c*colunas2[0],
//     dim_fitness[0], minimization);

    if(l<population_size[0] && c<population_size[0])
    {
        domination_counter[l*population_size[0]+c] =
        a_dominate_b(fitness+l*dim_fitness[0], fitness+c*dim_fitness[0], dim_fitness[0], minimization);
    }
}

__global__ void fast_nondominated_sort_copy(double *fitness, int *dim_fitness, int *domination_counter,
int *population_size, int *minimization, int *colunas2)
{
    int l = blockIdx.x*blockDim.x+threadIdx.x;
    int c = blockIdx.y*blockDim.y+threadIdx.y;
//     int tam = gridDim.y*blockDim.y;

//     domination_counter[l*tam+c] = 0;
//     domination_counter[l*tam+c] = a_dominate_b(fitness+l*colunas2[0], fitness+c*colunas2[0],
//     dim_fitness[0], minimization);

    if(l<2*population_size[0] && c<2*population_size[0])
    {
        domination_counter[l*2*population_size[0]+c] =
        a_dominate_b(fitness+l*dim_fitness[0], fitness+c*dim_fitness[0], dim_fitness[0], minimization);
    }
}

__global__ void fast_nondominated_sort2(int *domination_counter, int *population_size)
{
    int i = blockIdx.x*blockDim.x+threadIdx.x;
    domination_counter[population_size[0]*population_size[0]+i] = 0;

    for(int j=0;j<population_size[0];j++)
    {
        domination_counter[population_size[0]*population_size[0]+i] +=
        domination_counter[j*population_size[0]+i];
    }
}

__global__ void fast_nondominated_sort2_copy(int *domination_counter, int *population_size)
{
    int i = blockIdx.x*blockDim.x+threadIdx.x;
    domination_counter[2*population_size[0]*2*population_size[0]+i] = 0;

    for(int j=0;j<2*population_size[0];j++)
    {
        domination_counter[2*population_size[0]*2*population_size[0]+i] +=
        domination_counter[j*2*population_size[0]+i];
    }
}

__global__ void fast_nondominated_sort3(int *domination_counter, int *population_size,
 int *fronts, int *tam, int *rank)
{
    int i, j2=0, k = 0, rank_count = 0;
    int tamP = 0;
    int inicioFrontAnterior;

    for(i=0;i<population_size[0];i++)
    {
        if(domination_counter[population_size[0]*population_size[0]+i]==0)
        {
            domination_counter[population_size[0]*population_size[0]+i]=-1;
            fronts[j2] = i;
            rank[i] = rank_count;
            tamP+=1;
            j2+=1;
        }
    }

    tam[k] = tamP;
    k += 1;

    while(j2<population_size[0])
    {
        rank_count +=1 ;
        tamP = 0;
        inicioFrontAnterior = j2-tam[k-1];
        for(i=0;i<population_size[0];i++)
        {
            for(int j=0;j<tam[k-1];j++)
            {
                if(domination_counter[fronts[inicioFrontAnterior+j]*population_size[0]+i]==1)
                {
                    domination_counter[population_size[0]*population_size[0]+i]-= 1;
                }
            }
            if(domination_counter[population_size[0]*population_size[0]+i] == 0)
            {
                domination_counter[population_size[0]*population_size[0]+i] = -1;
                fronts[j2] = i;
                rank[i] = rank_count;
                j2+=1;
                tamP+=1;
            }
        }
        tam[k] = tamP;
        k+=1;
    }
    tam[k] = -1;
}

__global__ void fast_nondominated_sort3_copy(int *domination_counter, int *population_size,
 int *fronts, int *tam, int *rank)
{
    int i, j2=0, k = 0, rank_count = 0;
    int tamP = 0;
    int inicioFrontAnterior;

    for(i=0;i<2*population_size[0];i++)
    {
        if(domination_counter[2*population_size[0]*2*population_size[0]+i]==0)
        {
            domination_counter[2*population_size[0]*2*population_size[0]+i]=-1;
            fronts[j2] = i;
            rank[i] = rank_count;
            tamP+=1;
            j2+=1;
        }
    }

    tam[k] = tamP;
    k += 1;

    while(j2<2*population_size[0])
    {
        rank_count +=1 ;
        tamP = 0;
        inicioFrontAnterior = j2-tam[k-1];
        for(i=0;i<2*population_size[0];i++)
        {
            for(int j=0;j<tam[k-1];j++)
            {
                if(domination_counter[fronts[inicioFrontAnterior+j]*2*population_size[0]+i]==1)
                {
                    domination_counter[2*population_size[0]*2*population_size[0]+i]-= 1;
                }
            }
            if(domination_counter[2*population_size[0]*2*population_size[0]+i] == 0)
            {
                domination_counter[2*population_size[0]*2*population_size[0]+i] = -1;
                fronts[j2] = i;
                rank[i] = rank_count;
                j2+=1;
                tamP+=1;
            }
        }
        tam[k] = tamP;
        k+=1;
    }
    tam[k] = -1;
}

__global__ void fast_nondominated_sort4(double *fitness, int *dim, int *domination_counter,
int *colunas, int *minimization, int *dim_fit, int *front0_mem, int *tam_front0_mem)
{
    int l = blockIdx.x*blockDim.x+threadIdx.x;
    int c = blockIdx.y*blockDim.y+threadIdx.y;
    int l2, c2;

    if(l < tam_front0_mem[0] && c<tam_front0_mem[0])
    {
        l2 = front0_mem[l];
        c2 = front0_mem[c];
        domination_counter[l*tam_front0_mem[0]+c] = 0;
//         domination_counter[l*tam_front0_mem[0]+c] = a_dominate_b(fitness+l2*colunas2[0],
//         fitness+c2*colunas2[0], dim[0], minimization);
        domination_counter[l*tam_front0_mem[0]+c] = a_dominate_b(fitness+l2*dim_fit[0],
        fitness+c2*dim_fit[0], dim[0], minimization);
    }
}

__global__ void fast_nondominated_sort5(int *domination_counter, int *columns)
{
//     int i = threadIdx.x;
    int i = blockIdx.x*blockDim.x+threadIdx.x;
//     int colunas = blockDim.x;

//     domination_counter[blockDim.x*colunas+i] = 0;
    domination_counter[columns[0]*columns[0]+i] = 0;

//     printf("%d %d ", blockDim.x, blockDim.y);
    if(i<columns[0])
    {
        for(int j=0;j<columns[0];j++)
        {
//          domination_counter[blockDim.x*colunas+i] += domination_counter[j*colunas+i];
            domination_counter[columns[0]*columns[0]+i]  += domination_counter[j*columns[0]+i];
        }
    }
}

__global__ void fast_nondominated_sort6(int *domination_counter, int *tam_front0_mem,
int *front0_mem, int *tam_front0, int *front0)
{
    int i, j=0;

    tam_front0[0]=0;
    for(i=0;i<tam_front0_mem[0];i++)
    {
        if(domination_counter[tam_front0_mem[0]*tam_front0_mem[0]+i]==0)
        {
            front0[j] = front0_mem[i];
            tam_front0[0]+=1;
            j+=1;
        }
    }
    front0[j] = -1;
}

__global__ void fast_nondominated_sort7(double *fitness, int *dim, int *domination_counter,
int *colunas, int *minimization, int *colunas2)
{
    int l = blockIdx.x*blockDim.x+threadIdx.x;
    int c = blockIdx.y*blockDim.y+threadIdx.y;
    int tam = gridDim.y*blockDim.y;

    if(l<colunas[0] && c<colunas[0])
    {
        domination_counter[l*tam+c] = 0;
        domination_counter[l*tam+c] = a_dominate_b(fitness+l*colunas2[0], fitness+c*colunas2[0], dim[0], minimization);
    }
}

__global__ void fast_nondominated_sort3_teste(int *domination_counter, int *colunas, int *population_size,
 int *fronts, int *tam, int *rank, double *fitness)
{
    int i, j2=0, k = 0, rank_count = 0;
    int tamP = 0;
    int inicioFrontAnterior;
    int teste = 0;

//     printf("%lf %lf %lf\n", fitness[0], fitness[1], fitness[2]);

    for(i=0;i<population_size[0];i++)
    {
        if(domination_counter[population_size[0]*colunas[0]+i]==0)
        {
            domination_counter[population_size[0]*colunas[0]+i]=-1;
            fronts[j2] = i;
            rank[i] = rank_count;
            tamP+=1;
            j2+=1;
        }
    }
    tam[k] = tamP;
    k += 1;

    printf("%lf %lf %lf\n", fitness[0], fitness[1], fitness[2]);
    printf("%d %d\n", colunas[0], population_size[0]);
    while(j2<population_size[0])
    {
        printf("%d %d %lf %lf %lf\n", j2, rank_count, fitness[0], fitness[1], fitness[2]);
        teste+=1;
        if(fitness[0]==0 && fitness[1]==0 && fitness[2]==0)
        {
            break;
        }
        rank_count +=1 ;
        tamP = 0;
        inicioFrontAnterior = j2-tam[k-1];
        for(i=0;i<population_size[0];i++)
        {
            for(int j=0;j<tam[k-1];j++)
            {
                if(domination_counter[fronts[inicioFrontAnterior+j]*colunas[0]+i]==1)
                {
                    domination_counter[population_size[0]*colunas[0]+i]-= 1;
                }
            }
            if(domination_counter[population_size[0]*colunas[0]+i] == 0)
            {
                domination_counter[population_size[0]*colunas[0]+i] = -1;
                fronts[j2] = i;
                rank[i] = rank_count;
                j2+=1;
                tamP+=1;
            }
        }
        tam[k] = tamP;
        k+=1;
    }
    tam[k] = -1;
    printf("%lf %lf %lf\n", fitness[0], fitness[1], fitness[2]);
}

__global__ void print_dominantion_counter(int *m, int *colunas, int *population_size)
{
    printf("\n");
    for(int j=0;j<colunas[0];j++)
    {
        printf("%d ", m[population_size[0]*colunas[0]+j]);
    }
    printf("\n");
    for(int i=0;i<colunas[0];i++)
    {
        for(int j=0;j<colunas[0];j++)
        {
            printf("%d ", m[i*colunas[0]+j]);
        }
        printf("\n");
    }
    printf("\n");
}

__global__ void memory_inicialization1(double *position, double *fitness, int *fronts,
 int *position_dim, int *objectives_dim, int *population_size, int *tam_fronts)
{
    int i = blockIdx.x*blockDim.x+threadIdx.x;
    int j, k;
    k = fronts[i];

    if(i<tam_fronts[0])
    {
        for(j=0;j<position_dim[0];j++)
        {
            position[(i+2*population_size[0])*position_dim[0]+j] =
            position[k*position_dim[0]+j];
        }
        for(j=0;j<objectives_dim[0];j++)
        {
            fitness[(i+2*population_size[0])*objectives_dim[0]+j] =
            fitness[k*objectives_dim[0]+j];
        }
    }
}

__global__ void crowding_distance_inicialization(int* c, int *tam_pop)
{
    int i = blockIdx.x*blockDim.x+threadIdx.x;

    if(i<tam_pop[0])
    {
         c[i] = 0;
    }
}

__global__ void crowding_distance_inicialization2(int* c, int *tam_pop)
{
    int i = blockIdx.x*blockDim.x+threadIdx.x;

    if(i<tam_pop[0]*2)
    {
         c[i] = 0;
    }
}

__global__ void front_sort(int* front, int *dim, double *fitness, int *dim_fitness, int *i,
double *crowd_distance)
{
    int j,k,p1,p2, temp;
    double temp2;

    for(j=0;j<dim[0]-1;j++)
    {
        for(k=j+1;k<dim[0];k++)
        {
            p1 = front[j];
            p2 = front[k];
            if(fitness[p1*dim_fitness[0]+i[0]]>fitness[p2*dim_fitness[0]+i[0]])
            {
                temp = front[j];
                front[j] = front[k];
                front[k] = temp;
                temp2 = crowd_distance[j];
                crowd_distance[j] = crowd_distance[k];
                crowd_distance[k] = temp2;
            }
        }
    }
}

__global__ void front_sort2(int* front, int *dim, double *fitness, int *dim_fitness, int *i,
double *crowd_distance, int *tam_mem)
{
    int j,k,p1,p2, temp;
    double temp2;
    for(j=0;j<(dim[0]+tam_mem[0])-1;j++)
    {
        for(k=j+1;k<(dim[0]+tam_mem[0]);k++)
        {
            p1 = front[j];
            p2 = front[k];
            if(fitness[p1*dim_fitness[0]+i[0]]>fitness[p2*dim_fitness[0]+i[0]])
            {
                temp = front[j];
                front[j] = front[k];
                front[k] = temp;
                temp2 = crowd_distance[j];
                crowd_distance[j] = crowd_distance[k];
                crowd_distance[k] = temp2;
            }
        }
    }
}

// de apenas um front
__global__ void front_sort3(int* front, int *dim, double *fitness, int *dim_fitness, int *i,
double *crowd_distance, int *tam_pop)
{
    int j=0,k,p1,p2, temp, inicio=0;
    double temp2;

    while(j<dim[tam_pop[0]-2])
    {
        inicio+=dim[j];
        j++;
    }
//     printf("ini = %d %d\n", inicio, tam_pop[0]);

//     printf("%d\n", dim[dim[tam_pop[0]-2]]);
//     for(j=0;j<dim[dim[tam_pop[0]-2]];j++)
//     {
//         printf("%d ",front[inicio+j]);
//     }
//     printf("\n");

    for(j=0;j<dim[dim[tam_pop[0]-2]]-1;j++)
    {
        for(k=j+1;k<dim[dim[tam_pop[0]-2]];k++)
        {
            p1 = front[inicio+j];
            p2 = front[inicio+k];
            if(fitness[p1*dim_fitness[0]+i[0]]>fitness[p2*dim_fitness[0]+i[0]])
            {
//                 printf("%d %d %lf %lf\n",p1,p2, fitness[p1*dim_fitness[0]+i[0]],fitness[p2*dim_fitness[0]+i[0]]);
                temp = front[inicio+j];
                front[inicio+j] = front[inicio+k];
                front[inicio+k] = temp;
                temp2 = crowd_distance[j];
                crowd_distance[j] = crowd_distance[k];
                crowd_distance[k] = temp2;
            }
        }
    }
}

__global__ void front_sort4(int* front, int *dim, double *fitness, int *dim_fitness, int *i,
double *crowd_distance, int *tam_pop, int *index)
{
    int j,k;
//     double temp2;

//     printf("ini = %d %d\n", inicio, tam_pop[0]);

//     printf("%d\n", dim[dim[tam_pop[0]-2]]);
//     for(j=0;j<dim[dim[tam_pop[0]-2]];j++)
//     {
//         printf("%d ",front[inicio+j]);
//     }
//     printf("\n");

    for(j=0;j<tam_pop[0]-1;j++)
    {
        for(k=j+1;j<tam_pop[0];k++)
        {
//             p1 = front[inicio+j];
//             p2 = front[inicio+k];
            if(fitness[index[j]*dim_fitness[0]+i[0]]>fitness[index[k]*dim_fitness[0]+i[0]])
            {
//                 printf("%d %d %lf %lf\n",p1,p2, fitness[p1*dim_fitness[0]+i[0]],fitness[p2*dim_fitness[0]+i[0]]);
//                 temp = index[j];
//                 index[j] = index[k];
//                 index[k] = temp;
//                 temp2 = crowd_distance[j];
//                 crowd_distance[j] = crowd_distance[k];
//                 crowd_distance[k] = temp2;
            }
        }
    }
}

__global__ void front_sort5_par(double *fitness, int *dim_fitness, int *i,
double *crowd_distance, int *tam_pop, int *index)
{
    int temp;
    double temp2;
//     int j = threadIdx.x*2;
    int j = (blockIdx.x*blockDim.x+threadIdx.x)*2;

    if(j<2*tam_pop[0])
    {
        if(fitness[index[j]*dim_fitness[0]+i[0]]>fitness[index[j+1]*dim_fitness[0]+i[0]])
        {
    //                 printf("%d %d %lf %lf\n",p1,p2, fitness[p1*dim_fitness[0]+i[0]],fitness[p2*dim_fitness[0]+i[0]]);
                    temp = index[j];
                    index[j] = index[j+1];
                    index[j+1] = temp;
    //                 testar depois
                    temp2 = crowd_distance[j];
                    crowd_distance[j] = crowd_distance[j+1];
                    crowd_distance[j+1] = temp2;
        }
    }
}

__global__ void front_sort5_impar(double *fitness, int *dim_fitness, int *i,
double *crowd_distance, int *tam_pop, int *index)
{
    int temp;
    double temp2;
    int j = (blockIdx.x*blockDim.x+threadIdx.x)*2+1;

    if(j<2*tam_pop[0])
    {
        if(fitness[index[j]*dim_fitness[0]+i[0]]>fitness[index[j+1]*dim_fitness[0]+i[0]])
        {
    //                 printf("%d %d %lf %lf\n",p1,p2, fitness[p1*dim_fitness[0]+i[0]],fitness[p2*dim_fitness[0]+i[0]]);
                    temp = index[j];
                    index[j] = index[j+1];
                    index[j+1] = temp;
    //                 testar depois
                    temp2 = crowd_distance[j];
                    crowd_distance[j] = crowd_distance[j+1];
                    crowd_distance[j+1] = temp2;
        }
    }
}

__global__ void front_sort_crowding_distance(int* front, int *dim, double *crowding_distance)
{
    int j, k, temp;
    double temp2;
    for(j=0;j<dim[0]-1;j++)
    {
        for(k=j+1;k<dim[0];k++)
        {
            if(crowding_distance[j]<crowding_distance[k] ||
             (crowding_distance[j]==crowding_distance[k] && front[j]>front[k]))
            {
                temp = front[j];
                front[j] = front[k];
                front[k] = temp;
                temp2 = crowding_distance[j];
                crowding_distance[j] = crowding_distance[k];
                crowding_distance[k] = temp2;
            }
        }
    }
}

__global__ void front_sort_crowding_distance2(int* front, int *dim, double *crowding_distance, int *tam_mem)
{
    int j, k, temp;
//     int p1, p2;
    double temp2;
    for(j=0;j<(dim[0]+tam_mem[0]-1);j++)
    {
        for(k=j+1;k<(dim[0]+tam_mem[0]);k++)
        {
//             p1 = front[j];
//             p2 = front[k];
//             if(crowding_distance[j]<crowding_distance[k] ||
//              (crowding_distance[j]==crowding_distance[k] && front[j]>front[k]))
            if(crowding_distance[j]<crowding_distance[k])
//              (crowding_distance[j]==crowding_distance[k] && front[j]>front[k]))
            {
                temp = front[j];
                front[j] = front[k];
                front[k] = temp;
                temp2 = crowding_distance[j];
                crowding_distance[j] = crowding_distance[k];
                crowding_distance[k] = temp2;
            }
        }
    }
}

__global__ void front_sort_crowding_distance3(int* front, int *dim, double *crowding_distance, int *tam_pop)
{
    int j=0, k, temp, inicio=0;
//     int p1, p2;
    double temp2;
    int tam = dim[dim[tam_pop[0]-2]];

    while(j<dim[tam_pop[0]-2])
    {
        inicio+=dim[j];
        j++;
    }
//     printf("ini = %d\n",inicio);
    for(j=0;j<tam-1;j++)
    {
        for(k=j+1;k<tam;k++)
        {
//             p1 = front[j];
//             p2 = front[k];
            if(crowding_distance[j]<crowding_distance[k] ||
             (crowding_distance[j]==crowding_distance[k] && front[j]>front[k]))
            {
                temp = front[inicio+j];
                front[inicio+j] = front[inicio+k];
                front[inicio+k] = temp;
                temp2 = crowding_distance[j];
                crowding_distance[j] = crowding_distance[k];
                crowding_distance[k] = temp2;
            }
        }
    }
}

__global__ void front_sort_crowding_distance4(int* front, int *dim, double *crowding_distance, int *tam_pop)
{
    int j=0, k, inicio=0, temp;
    int tam = dim[dim[tam_pop[0]-2]];

    while(j<dim[tam_pop[0]-2])
    {
        inicio+=dim[j];
        j++;
    }

    for(j=0;j<tam-1;j++)
    {
        for(k=j+1;k<tam;k++)
        {
            if(crowding_distance[front[inicio+j]]<crowding_distance[front[inicio+k]])
            {
                temp = front[inicio+j];
                front[inicio+j] = front[inicio+k];
                front[inicio+k] = temp;
            }
        }
    }
}

__global__ void crowding_distance(int* front, int *dim, double *fitness,
 int *dim_fitness, int *tam_front, int *i, double *crowd_distance)
{
    int j = threadIdx.x, next, prev, first, last;
    next = front[j+2];
    prev = front[j];
    first = front[0];
    last = front[dim[0]-1];
    crowd_distance[0] = 1e20;
//     crowd_distance[tam_front[0]-1] = 1e20;
    crowd_distance[dim[0]-1] = 1e20;
    if(crowd_distance[j+1]!=1e20)
    {
        crowd_distance[j+1] += (fitness[next*dim_fitness[0]+i[0]] - fitness[prev*dim_fitness[0]+i[0]])/
        (fitness[last*dim_fitness[0]+i[0]] - fitness[first*dim_fitness[0]+i[0]]);
    }
}

__global__ void crowding_distance2(int* front, int *dim, double *fitness,
 int *dim_fitness, int *tam_front, int *i, double *crowd_distance, int *tam_mem)
{
    int j = threadIdx.x, next, prev, first, last;
    next = front[j+2];
    prev = front[j];
    first = front[0];
    last = front[tam_front[0]+tam_mem[0]-1];
    crowd_distance[0] = 1e20;
    crowd_distance[tam_front[0]+tam_mem[0]-1] = 1e20;
    if(crowd_distance[j+1]!=1e20)
    {
        crowd_distance[j+1] += (fitness[next*dim_fitness[0]+i[0]] - fitness[prev*dim_fitness[0]+i[0]])/
        (fitness[last*dim_fitness[0]+i[0]] - fitness[first*dim_fitness[0]+i[0]]);
    }
    if(front[j+1] == 88)
    {
           printf("n = %d, p = %d, f= %d, l= %d\n", next, prev, first, last);
           printf("n = %0.3lf, p = %0.3lf, f= %0.3lf, l= %0.3lf, i = %d\n", fitness[next*dim_fitness[0]+i[0]],
            fitness[prev*dim_fitness[0]+i[0]], fitness[first*dim_fitness[0]+i[0]],
            fitness[last*dim_fitness[0]+i[0]], i[0]);
           printf("%lf, %lf\n", fitness[88*2+0], fitness[88*2+1]);
    }
}

__global__ void crowding_distance3(int* front, int *dim, double *fitness,
 int *dim_fitness, int *tam_front, int *i, double *crowd_distance, int *tam_pop)
{
    int j = threadIdx.x, next, prev, first, last, inicio = 0,k=0;

    while(k<dim[tam_pop[0]-2])
    {
        inicio+=dim[k];
        k++;
    }

    next = front[inicio+j+2];
    prev = front[inicio+j];
    first = front[inicio];
    last = front[inicio+tam_front[tam_front[tam_pop[0]-2]]-1];
    crowd_distance[0] = 1e20;
    crowd_distance[tam_front[tam_front[tam_pop[0]-2]]-1] = 1e20;
    if(crowd_distance[j+1]!=1e20)
    {
//         printf("%d %lf %lf %lf %lf\n",j, fitness[next*dim_fitness[0]+i[0]], fitness[prev*dim_fitness[0]+i[0]],
//         fitness[last*dim_fitness[0]+i[0]], fitness[first*dim_fitness[0]+i[0]]);
        crowd_distance[j+1] += (fitness[next*dim_fitness[0]+i[0]] - fitness[prev*dim_fitness[0]+i[0]])/
        (fitness[last*dim_fitness[0]+i[0]] - fitness[first*dim_fitness[0]+i[0]]);
    }
}

__global__ void crowding_distance4(double *fitness,
 int *dim_fitness, int *tam_front, int *i, double *crowd_distance, int *tam_pop, int *index)
{
    // os indices variam de 0 a tam_pop-2
    // os crwoding distance sao calculados de 1 a tam_pop-2, ja que os extremso sao infinitos
    int j = threadIdx.x, next, prev, first, last;

    next = index[j+2];
    prev = index[j];
    first = index[0];
    last = index[2*tam_pop[0]-1];

    crowd_distance[first] = 1e20;
    crowd_distance[last] = 1e20;
    if(crowd_distance[index[j+1]]!=1e20)
    {
        crowd_distance[index[j+1]] += (fitness[next*dim_fitness[0]+i[0]] - fitness[prev*dim_fitness[0]+i[0]])/
        (fitness[last*dim_fitness[0]+i[0]] - fitness[first*dim_fitness[0]+i[0]]);
    }
}

__global__ void memory_inicialization2(double *position, double *fitness, int *fronts,
 int *position_dim, int *objectives_dim, int *population_size)
{
    int i = threadIdx.x, j, k;
    k = fronts[i];

    for(j=0;j<position_dim[0];j++)
    {
        position[(i+2*population_size[0])*position_dim[0]+j] =
        position[k*position_dim[0]+j];
    }
    for(j=0;j<objectives_dim[0];j++)
    {
        fitness[(i+2*population_size[0])*objectives_dim[0]+j] =
        fitness[k*objectives_dim[0]+j];
    }
}

__global__ void memory_inicialization2_1(double *position, double *fitness, int *fronts,
 int *position_dim, int *objectives_dim, int *population_size, double *aux, double *aux2,
 int *tam)
{
    int i = blockIdx.x*blockDim.x+threadIdx.x,j,k;
//     int i = threadIdx.x, j, k;
    k = fronts[i];

    if(i<tam[0])
    {
        for(j=0;j<position_dim[0];j++)
        {
            aux[i*position_dim[0]+j] =
            position[k*position_dim[0]+j];
        }
        for(j=0;j<objectives_dim[0];j++)
        {
            aux2[i*objectives_dim[0]+j] =
            fitness[k*objectives_dim[0]+j];
        }
    }
}

__global__ void memory_inicialization2_2(double *position, double *fitness, int *fronts,
 int *position_dim, int *objectives_dim, int *population_size, double *aux, double *aux2,
 int *tam)
{
    int i = blockIdx.x*blockDim.x+threadIdx.x,j;
//     int i = threadIdx.x, j;

    if(i<tam[0])
    {
        for(j=0;j<position_dim[0];j++)
        {
            position[(i+2*population_size[0])*position_dim[0]+j] =
            aux[i*position_dim[0]+j];
        }
        for(j=0;j<objectives_dim[0];j++)
        {
            fitness[(i+2*population_size[0])*objectives_dim[0]+j] =
            aux2[i*objectives_dim[0]+j];
        }
    }
}

__global__ void memory_inicialization3(double *position, double *fitness, int *fronts,
 int *position_dim, int *objectives_dim, int *population_size)
{
    int i = threadIdx.x, j;
//     printf("%d %d\n", i+2*population_size[0], k);
    for(j=0;j<position_dim[0];j++)
    {
        position[(i+2*population_size[0])*position_dim[0]+j] =
        position[i*position_dim[0]+j];
    }
    for(j=0;j<objectives_dim[0];j++)
    {
        fitness[(i+2*population_size[0])*objectives_dim[0]+j] =
        fitness[i*objectives_dim[0]+j];
    }
}

__global__ void memory_inicialization4(double *position, double *fitness,
 int *position_dim, int *objectives_dim, int *population_size, double *aux,
 int *memory_size)
{
    int i = blockIdx.x*blockDim.x+threadIdx.x,j;

    if(i<memory_size[0])
    {
        for(j=0;j<position_dim[0];j++)
        {
            position[(i+2*population_size[0])*position_dim[0]+j] = 1e20;
        }
        for(j=0;j<objectives_dim[0];j++)
        {
            fitness[(i+2*population_size[0])*objectives_dim[0]+j] = 1e20;
        }
    }
}

__global__ void memory_inicialization5(double *position, double * velocity, double *fitness,
double *aux, double *aux2, int *fronts, int *position_dim, int *objectives_dim, int *population_size)
{
    int i = threadIdx.x, j;
    for(j=0;j<position_dim[0];j++)
    {
        position[(i+2*population_size[0])*position_dim[0]+j] =
        aux[fronts[i]*position_dim[0]+j];
        velocity[(i+2*population_size[0])*position_dim[0]+j] =
        aux2[fronts[i]*position_dim[0]+j];
    }
    for(j=0;j<objectives_dim[0];j++)
    {
        fitness[(i+2*population_size[0])*objectives_dim[0]+j] =
        fitness[fronts[i]*objectives_dim[0]+j];
    }
}

__global__ void update_personal_best(double *personal_best_p, double *personal_best_v, double *personal_best_f,
int *tam_obj, int *tam_pos, double *position, double *fitness, int *personal_guide_array_size)
{
    int i = threadIdx.x, j;
    int tam1 = personal_guide_array_size[0]*tam_pos[0];
    int tam2 = personal_guide_array_size[0]*tam_obj[0];
//     printf("personal_best_pos[%d][0] =  %lf %d\n", i, personal_best_p[i*tam1+0], tam1);
    if(personal_best_p[i*tam1+0] == 1e10)
    {
//         printf("sim%d\n", i);
        for(j=0;j<tam_pos[0];j++)
        {
//             printf("%d %d %d\n", i, j, i*tam1+j);
            personal_best_p[i*tam1+j] = position[i*tam_pos[0]+j];
        }
        for(j=0;j<tam_obj[0];j++)
        {
//             printf("%d %d %d %lf\n", i, j, i*tam2+j, fitness[i*tam_obj[0]+j]);
            personal_best_f[i*tam2+j] = fitness[i*tam_obj[0]+j];
        }
    }
    else
    {
        printf("falta implementar\n");
    }
}

__global__ void update_personal_best2(double *personal_best_p, double *personal_best_v, double *personal_best_f,
int *tam_obj, int *tam_pos, double *position, double *fitness, int *personal_guide_array_size, int *maximize)
{
    int i = threadIdx.x, j, k;
    int tam1 = personal_guide_array_size[0]*tam_pos[0];
    int tam2 = personal_guide_array_size[0]*tam_obj[0];
    short int different = 0;

    for(j=0;j<tam_pos[0];j++)
    {
        if(position[i*tam_pos[0]+j]!= personal_best_p[i*tam1+j])
        {
            different++;
        }
    }
    if(different>0)
    {
        if(a_dominate_b(fitness+(i*tam_obj[0]), personal_best_f+(i*tam2), tam_obj[0], maximize))
        {
    //         printf("sim %d %d %d %lf %lf %lf %lf\n", i, i*tam_obj[0], i*tam2, fitness[i*2], fitness[i*2+1],
    //                     personal_best_f[i*tam2], personal_best_f[i*tam2+1]);
            printf("sim %d %lf %lf %lf %lf %d %d %lf %lf\n", i, fitness[i*2], fitness[i*2+1],
                        personal_best_f[i*tam2], personal_best_f[i*tam2+1], i*tam1, i*tam2, personal_best_f[4*tam2],
                        personal_best_f[4*tam2+1]);
    //         printf("%lf %lf %lf %lf\n", );
            for(k=0;k<tam_pos[0];k++)
            {
                personal_best_p[i*tam1+k] = position[i*tam_pos[0]+k];
            }
            for(k=0;k<tam_obj[0];k++)
            {
                personal_best_f[i*tam2+k] = fitness[i*tam_obj[0]+k];
            }
            printf("sim %d %lf %lf %lf %lf %d %d %lf %lf\n", i, fitness[i*2], fitness[i*2+1],
                        personal_best_f[i*tam2], personal_best_f[i*tam2+1], i*tam1, i*tam2, personal_best_f[4*tam2],
                        personal_best_f[4*tam2+1]);
        }
        else
        {
            if(a_dominate_b(personal_best_f+(i*tam2), fitness+(i*tam_obj[0]), tam_obj[0], maximize))
            {
                printf("nao sim %d %lf %lf %lf %lf\n", i, fitness[i*2], fitness[i*2+1],
                            personal_best_f[i*tam2], personal_best_f[i*tam2+1]);
            }
            else
            {
                printf("nao nao %d %lf %lf %lf %lf\n", i, fitness[i*2], fitness[i*2+1],
                            personal_best_f[i*tam2], personal_best_f[i*tam2+1]);
                for(k=0;k<tam_pos[0];k++)
                {
                    personal_best_p[i*tam1+1*tam_pos[0]+k] = position[i*tam_pos[0]+k];
                    personal_best_f[i*tam2+1*tam_obj[0]+k] = fitness[i*tam_obj[0]+k];
                }
            }
        }
    }
}

__global__ void update_personal_best3(double *personal_best_p, double *personal_best_v, double *personal_best_f,
int *tam_obj, int *tam_pos, double *position, double *fitness, int *personal_guide_array_size, int *maximize)
{
    int i = blockIdx.x*blockDim.x+threadIdx.x, j, k;
    int tam1 = personal_guide_array_size[0]*tam_pos[0];
    int tam2 = personal_guide_array_size[0]*tam_obj[0];
    int include=0, dominated=0;
    short int exist, different, inset=0;
//     short int follow = 0;

    // exist: se a aquela posicao existe uma particula. O indicador 1e10 na posicao 0 indica que nao existe
    // different: indica se o elemento do eprsonal best e diferent da particula atualiza
    // inset: indica se a particula ja esta no conjunto personal best

//     dominated = personal_guide_array_size[0];
    for(k=0;k<personal_guide_array_size[0];k++)
    {
//         if(i==3)
//         {
//             printf("%d\n", k);
//         }
//         follow = 0;
        if(personal_best_p[i*tam1+k*tam_pos[0]+0] != 1e10)
        {
//             if(i==3)
//             {
//                 printf("%d\n", k);
//             }
            exist=1;
            different=0;
            for(j=0;j<tam_pos[0];j++)
            {
//                 if(i==2)
//                 {
//                     printf("j=%d pos=%lf per=%lf\n",j, position[i*tam_pos[0]+j],personal_best_p[i*tam1+k*tam_pos[0]+j]);
//                 }
                if(position[i*tam_pos[0]+j]!= personal_best_p[i*tam1+k*tam_pos[0]+j])
                {
//                     if(i==2)
//                     {
//                         printf("j=%d pos=%lf per=%lf\n",j, position[i*tam_pos[0]+j],personal_best_p[i*tam1+k*tam_pos[0]+j]);
//                     }
                    different++;
                    break;
                }
            }
            inset+=different;
        }
        else
        {
            exist = 0;
        }
//         if(i==3)
//         {
//             printf("k=%d ex=%d df=%d ins=%d\n", k, exist, different, inset);
//         }
//         dominated--;

//         if(i==2)
//         {
//             printf("i=%d k=%d exist=%d dif = %d\n",i,k,exist, different);
//         }
//         if(i==64+128)
//         {
//             printf("i=%d exist=%d diff=%d\n",i,exist, different);
//         }
        if(exist==1 && different == 1)
        {
            dominated++;
//             if(i==64+128)
//             {
//                 printf("i=%d exist=%d diff=%d\n",i,exist, different);
//             }
            if(a_dominate_b(personal_best_f+(i*tam2+k*tam_obj[0]), fitness+(i*tam_obj[0]),
            tam_obj[0], maximize) == 0)
            {
                dominated--;
            }
            if(a_dominate_b(fitness+(i*tam_obj[0]), personal_best_f+(i*tam2+k*tam_obj[0]),
            tam_obj[0], maximize))
            {
                for(j=0;j<tam_pos[0];j++)
                {
                    personal_best_p[i*tam1+k*tam_pos[0]+j] = 1e10;
                }
                include = 1;
            }
        }
    }
    if((include || !dominated) && inset)
    {
        for(k=0;k<personal_guide_array_size[0];k++)
        {
            if(personal_best_p[i*tam1+k*tam_pos[0]+0] == 1e10)
            {
                for(j=0;j<tam_pos[0];j++)
                {
                    personal_best_p[i*tam1+k*tam_pos[0]+j] = position[i*tam_pos[0]+j];
                }
                for(j=0;j<tam_obj[0];j++)
                {
                    personal_best_f[i*tam2+k*tam_obj[0]+j] = fitness[i*tam_obj[0]+j];
                }
                break;
            }
        }
    }
}

__global__ void update_personal_best3_validation(double *personal_best_p, double *personal_best_v, double *personal_best_f,
int *tam_obj, int *tam_pos, double *position, double *fitness, int *personal_guide_array_size, int *maximize)
{
    int i = blockIdx.x*blockDim.x+threadIdx.x, j, k;
    int tam1 = personal_guide_array_size[0]*tam_pos[0];
    int tam2 = personal_guide_array_size[0]*tam_obj[0];
    int include=0, dominated=0, full=1;
    short int exist, different, inset=0;

    if(i<tam_pos[0])
    {
        for(k=0;k<personal_guide_array_size[0];k++)
        {
            if(personal_best_p[i*tam1+k*tam_pos[0]+0] != 1e10)
            {
                exist=1;
                different=0;
                for(j=0;j<tam_pos[0];j++)
                {
                    if(position[i*tam_pos[0]+j]!= personal_best_p[i*tam1+k*tam_pos[0]+j])
                    {
                        different++;
                        break;
                    }
                }
                inset+=different;
            }
            else
            {
                exist = 0;
            }
            if(exist==1 && different == 1)
            {
                dominated++;
                if(a_dominate_b(personal_best_f+(i*tam2+k*tam_obj[0]), fitness+(i*tam_obj[0]),
                tam_obj[0], maximize) == 0)
                {
                    dominated--;
                }
                if(a_dominate_b(fitness+(i*tam_obj[0]), personal_best_f+(i*tam2+k*tam_obj[0]),
                tam_obj[0], maximize))
                {
                    for(j=0;j<tam_pos[0];j++)
                    {
                        personal_best_p[i*tam1+k*tam_pos[0]+j] = 1e10;
                    }
                    include = 1;
                }
            }
        }

        //para validacao. Apagar depois para maior eficiencia
        for(k=0; k < personal_guide_array_size[0]-1; k++)
        {
            if(personal_best_p[i*tam1+k*tam_pos[0]] == 1e10)
            {
                for(int l=k+1; l<personal_guide_array_size[0]; l++)
                {
                    for(j=0;j<tam_pos[0];j++)
                    {
                        personal_best_p[i*tam1+(l-1)*tam_pos[0]+j] =
                        personal_best_p[i*tam1+(l)*tam_pos[0]+j];
                    }
                    for(j=0;j<tam_obj[0];j++)
                    {
                        personal_best_f[i*tam2+(l-1)*tam_obj[0]+j] =
                        personal_best_f[i*tam2+(l)*tam_obj[0]+j];
                    }
                }
                personal_best_p[i*tam1+(personal_guide_array_size[0]-1)*tam_pos[0]] = 1e10;
            }
        }

        if((include || !dominated) && inset)
        {
            for(k=0;k<personal_guide_array_size[0];k++)
            {
                if(personal_best_p[i*tam1+k*tam_pos[0]+0] == 1e10)
                {
                    full = 0;
                    for(j=0;j<tam_pos[0];j++)
                    {
                        personal_best_p[i*tam1+k*tam_pos[0]+j] = position[i*tam_pos[0]+j];
                    }
                    for(j=0;j<tam_obj[0];j++)
                    {
                        personal_best_f[i*tam2+k*tam_obj[0]+j] = fitness[i*tam_obj[0]+j];
                    }
                    break;
                }
            }
            if(full == 1)
            {
                for(k=0;k<personal_guide_array_size[0]-1;k++)
                {
                    for(j=0;j<tam_pos[0];j++)
                    {
                        personal_best_p[i*tam1+k*tam_pos[0]+j] = personal_best_p[i*tam1+(k+1)*tam_pos[0]+j];
                    }
                    for(j=0;j<tam_obj[0];j++)
                    {
                        personal_best_f[i*tam2+k*tam_obj[0]+j] = personal_best_f[i*tam2+(k+1)*tam_obj[0]+j];
                    }
                }
                k = personal_guide_array_size[0]-1;
                for(j=0;j<tam_pos[0];j++)
                {
                    personal_best_p[i*tam1+k*tam_pos[0]+j] = position[i*tam_pos[0]+j];
                }
                for(j=0;j<tam_obj[0];j++)
                {
                    personal_best_f[i*tam2+k*tam_obj[0]+j] = fitness[i*tam_obj[0]+j];
                }
            }
        }
    }
}

__device__ void update_personal_best4(double *personal_best_p, double *personal_best_v, double *personal_best_f,
int *tam_obj, int *tam_pos, double *position, double *fitness, int *personal_guide_array_size, int *maximize)
{
    int i = threadIdx.x, j, k;
    int tam1 = personal_guide_array_size[0]*tam_pos[0];
    int tam2 = personal_guide_array_size[0]*tam_obj[0];
    int include=0, dominated=0;
    short int exist, different, inset=0;
//     short int follow = 0;

    // exist: se a aquela posicao existe uma particula. O indicador 1e10 na posicao 0 indica que nao existe
    // different: indica se o elemento do eprsonal best e diferent da particula atualiza
    // inset: indica se a particula ja esta no conjunto personal best

//     dominated = personal_guide_array_size[0];
    for(k=0;k<personal_guide_array_size[0];k++)
    {
//         if(i==3)
//         {
//             printf("%d\n", k);
//         }
//         follow = 0;
        if(personal_best_p[i*tam1+k*tam_pos[0]+0] != 1e10)
        {
//             if(i==3)
//             {
//                 printf("%d\n", k);
//             }
            exist=1;
            different=0;
            for(j=0;j<tam_pos[0];j++)
            {
//                 if(i==2)
//                 {
//                     printf("j=%d pos=%lf per=%lf\n",j, position[i*tam_pos[0]+j],personal_best_p[i*tam1+k*tam_pos[0]+j]);
//                 }
                if(position[i*tam_pos[0]+j]!= personal_best_p[i*tam1+k*tam_pos[0]+j])
                {
//                     if(i==2)
//                     {
//                         printf("j=%d pos=%lf per=%lf\n",j, position[i*tam_pos[0]+j],personal_best_p[i*tam1+k*tam_pos[0]+j]);
//                     }
                    different++;
                    break;
                }
            }
            inset+=different;
        }
        else
        {
            exist = 0;
        }
//         if(i==3)
//         {
//             printf("k=%d ex=%d df=%d ins=%d\n", k, exist, different, inset);
//         }
//         dominated--;

//         if(i==2)
//         {
//             printf("i=%d k=%d exist=%d dif = %d\n",i,k,exist, different);
//         }
        if(exist==1 && different == 1)
        {
            dominated++;
//             if(i==2)
//             {
//                 printf("i=%d k=%d\n",i,k);
//             }
            if(a_dominate_b(personal_best_f+(i*tam2+k*tam_obj[0]), fitness+(i*tam_obj[0]),
            tam_obj[0], maximize) == 0)
            {
                dominated--;
            }
            if(a_dominate_b(fitness+(i*tam_obj[0]), personal_best_f+(i*tam2+k*tam_obj[0]),
            tam_obj[0], maximize))
            {
                for(j=0;j<tam_pos[0];j++)
                {
                    personal_best_p[i*tam1+k*tam_pos[0]+j] = 1e10;
                }
                include = 1;
            }
//             if(i==3)
//             {
//                 printf("k=%d ex=%d df=%d ins=%d dom=%d inc=%d\n",
//                  k, exist, different, inset, dominated, include);
//             }
        }
//         if(i==3)
//         {
//             printf("k=%d ex=%d df=%d ins=%d dom=%d inc=%d\n",
//              k, exist, different, inset, dominated, include);
//         }
    }
//     if(i==2)
//     {
//         printf("%d %d %d %d %d\n",i,k, include, follow, dominated);
//     }
    if((include || !dominated) && inset)
    {
//         printf("%d\n", i);
        for(k=0;k<personal_guide_array_size[0];k++)
        {
//             if(i==3)
//             {
//                 for(j=0;j<tam_pos[0];j++)
//                 {
//                     printf("%lf ", personal_best_p[i*tam1+k*tam_pos[0]+j]);
//                 }
//                 printf("%d \n", k);
//             }
            if(personal_best_p[i*tam1+k*tam_pos[0]+0] == 1e10)
            {
                for(j=0;j<tam_pos[0];j++)
                {
                    personal_best_p[i*tam1+k*tam_pos[0]+j] = position[i*tam_pos[0]+j];
                }
                for(j=0;j<tam_obj[0];j++)
                {
                    personal_best_f[i*tam2+k*tam_obj[0]+j] = fitness[i*tam_obj[0]+j];
                }
                break;
            }
        }
    }
}

__device__ void update_personal_best4_validation(double *personal_best_p, double *personal_best_v,
double *personal_best_f, int *tam_obj, int *tam_pos, double *position, double *fitness,
int *personal_guide_array_size, int *maximize)
{
    int i = threadIdx.x, j, k;
    int tam1 = personal_guide_array_size[0]*tam_pos[0];
    int tam2 = personal_guide_array_size[0]*tam_obj[0];
    int include=0, dominated=0, full=1;
    short int exist, different, inset=0;

    // exist: se a aquela posicao existe uma particula. O indicador 1e10 na posicao 0 indica que nao existe
    // different: indica se o elemento do eprsonal best e diferent da particula atualiza
    // inset: indica se a particula ja esta no conjunto personal best

//     dominated = personal_guide_array_size[0];
    for(k=0;k<personal_guide_array_size[0];k++)
    {
        // se existe uma particula nessa posicao na lista de personal best
        if(personal_best_p[i*tam1+k*tam_pos[0]+0] != 1e10)
        {
            // indica que existe uma particula nessa posicao
            exist=1;
            // inicializa contador que verifica se a particula atual e diferente da particula ness aposicao do
            //             eprsonal best
            different=0;
            for(j=0;j<tam_pos[0];j++)
            {
                if(position[i*tam_pos[0]+j]!= personal_best_p[i*tam1+k*tam_pos[0]+j])
                {
                    // se for diferente, different recebe 1 quebra o laco
                    different++;
                    break;
                }
            }
            inset+=different;
        }
        else
        {
            exist = 0;
        }

        if(exist==1 && different == 1)
        {
            // dominated indica se a particle atual e dominada pela particula da lista
            dominated++;

            // particula da lista nao domina a partocula atual, dominated =0
            if(a_dominate_b(personal_best_f+(i*tam2+k*tam_obj[0]), fitness+(i*tam_obj[0]),
            tam_obj[0], maximize) == 0)
            {
                dominated--;
            }

            // se a particula nao e dominada, ela pode dominar ou ser do emsmo front.different
            // se aqui for verdadeiro e ela dominar, colocamos um indicar de vago na posicao da particula
            // e um indicador de inclusao
            // se ela nao dominar nao afz nada, include=0
            if(a_dominate_b(fitness+(i*tam_obj[0]), personal_best_f+(i*tam2+k*tam_obj[0]),
            tam_obj[0], maximize))
            {
                for(j=0;j<tam_pos[0];j++)
                {
                    personal_best_p[i*tam1+k*tam_pos[0]+j] = 1e10;
                }
                include = 1;
            }
        }
    }

    //para validacao. Apagar depois para maior eficiencia
    // erajeitara para os vagos focarem no final, verificar se e isso pode ser retirado depois 270625

    for(k=0; k < personal_guide_array_size[0]-1; k++)
    {
        if(personal_best_p[i*tam1+k*tam_pos[0]] == 1e10)
        {
            for(int l=k+1; l<personal_guide_array_size[0]; l++)
            {
                for(j=0;j<tam_pos[0];j++)
                {
                    personal_best_p[i*tam1+(l-1)*tam_pos[0]+j] =
                    personal_best_p[i*tam1+(l)*tam_pos[0]+j];
                }
                for(j=0;j<tam_obj[0];j++)
                {
                    personal_best_f[i*tam2+(l-1)*tam_obj[0]+j] =
                    personal_best_f[i*tam2+(l)*tam_obj[0]+j];
                }
            }
            personal_best_p[i*tam1+(personal_guide_array_size[0]-1)*tam_pos[0]] = 1e10;
        }
    }

    //     include =1, particula domina pelo menos uma da lista
    // dominated=0, particula nao e dominada por nenhuma
    if((include || !dominated) && inset)
    {
        for(k=0;k<personal_guide_array_size[0];k++)
        {
            if(personal_best_p[i*tam1+k*tam_pos[0]+0] == 1e10)
            {
                full = 0;
                for(j=0;j<tam_pos[0];j++)
                {
                    personal_best_p[i*tam1+k*tam_pos[0]+j] = position[i*tam_pos[0]+j];
                }
                for(j=0;j<tam_obj[0];j++)
                {
                    personal_best_f[i*tam2+k*tam_obj[0]+j] = fitness[i*tam_obj[0]+j];
                }
                break;
            }
        }

        // se estiver cheio, move a lista para a esquerda e add no final
        if(full == 1)
        {
            for(k=0;k<personal_guide_array_size[0]-1;k++)
            {
                for(j=0;j<tam_pos[0];j++)
                {
                    personal_best_p[i*tam1+k*tam_pos[0]+j] = personal_best_p[i*tam1+(k+1)*tam_pos[0]+j];
                }
                for(j=0;j<tam_obj[0];j++)
                {
                    personal_best_f[i*tam2+k*tam_obj[0]+j] = personal_best_f[i*tam2+(k+1)*tam_obj[0]+j];
                }
            }
            k = personal_guide_array_size[0]-1;
            for(j=0;j<tam_pos[0];j++)
            {
                personal_best_p[i*tam1+k*tam_pos[0]+j] = position[i*tam_pos[0]+j];
            }
            for(j=0;j<tam_obj[0];j++)
            {
                personal_best_f[i*tam2+k*tam_obj[0]+j] = fitness[i*tam_obj[0]+j];
            }
        }
    }
}

__device__ int equal(double *position1, double *position2, int *tam)
{
    int i;
    for(i=0;i<tam[0];i++)
    {
//         printf("%lf %lf\n", position1[i], position2[i]);
        if(position1[i]!=position2[i])
        {
            return 0;
        }
    }
    return 1;
}

__device__ void update_personal_best_device(double *personal_best_p, double *personal_best_v,
double *personal_best_f, int *tam_obj, int *tam_pos, double *position, double *fitness,
int *personal_guide_array_size, int *personal_best_tam, int *maximize)
{
    int i = threadIdx.x, j, k, temp, adicionado=0;
    int tam1 = personal_guide_array_size[0]*tam_pos[0];
    int tam2 = personal_guide_array_size[0]*tam_obj[0];
//     printf("personal_best_pos[%d][0] =  %lf %d\n", i, personal_best_p[i*tam1+0], tam1);

    for(j=0;j<personal_guide_array_size[0];j++)
    {
        if(personal_best_p[i*tam1+j*tam_pos[0]]!=1e10)
        {
            temp = a_dominate_b(fitness+(i*tam_obj[0]),
            personal_best_f+(i*tam2+j*tam_obj[0]), tam_obj[0], maximize);
//             printf("i = %d %d %d %d\n, ", i, temp, i*tam_obj[0], i*tam2+j*tam_obj[0]);
            if(temp==1)
            {
                if(adicionado == 0)
                {
                    for(k = 0;k < tam_pos[0]; k++)
                    {
                        personal_best_p[i*tam1+j*tam_pos[0]+k] = position[i*tam_pos[0]+k];
                    }
                    adicionado = 1;
                }
                else
                {
                    personal_best_p[i*tam1+j*tam_pos[0]] = 1e10;
                }
            }
        }
    }
}

__global__ void differential_mutation(int *func_n, int *xr_pool_type, int *tam_pop, int *tam_mem,
 double *position, int *tam_pos, double *personal_best_p, int *personal_guide_array_size,
 double *fitness, int *tam_fit, int *maximize, int *xr_pool, int *DE_mutation_type, int *xr_list,
 double *weights, double *xst, double *pos_min, double *pos_max, int *secondary_params,
 double *xst_fitness, int *xst_dominate, double *personal_best_f, double *personal_best_v,
 int *personal_best_tam, int *update_from_differential_mutation, int *seed, double *alpha)
{
    int i = blockIdx.x*blockDim.x+threadIdx.x;
    int j, tamPersonal, temp1, temp2, k=0, pool_tam=0;
    curandState state;
    int xr_list_l[5];
    int mutation_index_l = -1;
    int whatPersonal_l = 0;
    double mutation_chance_l[1000];

    if(i<tam_pop[0])
    {
        curand_init(seed[0], i, 0, &state);

    //     mutation_index_l = xr_pool[(int)(curand_uniform(&state)*tam_pos[0])];
        mutation_index_l = (int)(curand_uniform(&state)*(tam_pos[0]-1));
        for(j=0;j<tam_pos[0];j++)
        {
            mutation_chance_l[j] = curand_uniform(&state);
        }

        tamPersonal = tam_pos[0]*personal_guide_array_size[0];

        if(xr_pool_type[0] == 1)// Apenas Memoria
        {
            for(j=0;j<tam_mem[0];j++)
            {
    //             personal_best == m
    //             temp1 = equal(personal_best_p+i*tamPersonal+whatPersonal[i]*tam_pos[0],
    //             position+(2*tam_pop[0]+j)*tam_pos[0], tam_pos);
                temp1 = equal(personal_best_p+i*tamPersonal+whatPersonal_l*tam_pos[0],
                position+(2*tam_pop[0]+j)*tam_pos[0], tam_pos);
    //             particle == m
                temp2 = equal(position+(i*tam_pos[0]),
                position+(2*tam_pop[0]+j)*tam_pos[0], tam_pos);

                if(!(temp1 == 1) || !(temp2 == 1))
                {
                    if(a_dominate_b(fitness+i*tam_fit[0], fitness+(2*tam_pop[0]+j)*tam_fit[0],
                     tam_fit[0], maximize) == 0)
                    {
                        xr_pool[i*(2*tam_pop[0]+tam_mem[0])+k] = j;
                        pool_tam+=1;
                        k+=1;
                    }
                }
            }
        }

        if(DE_mutation_type[0] == 0 && k >= 3) //DE\rand\1\Bin
        {
            xr_list_l[0] = xr_pool[(int)(curand_uniform(&state)*pool_tam)];
            xr_list_l[1] = xr_pool[(int)(curand_uniform(&state)*pool_tam)];
            xr_list_l[2] = xr_pool[(int)(curand_uniform(&state)*pool_tam)];

            for(j=0;j<tam_pos[0];j++)
            {
                //faz xst = (xr1-xr2)w5 + xr0
                xst[i*tam_pos[0]+j] = position[(2*tam_pop[0]+xr_list_l[1])*(tam_pos[0])+j] -
                position[(2*tam_pop[0]+xr_list[2])*(tam_pos[0])+j];
                xst[i*tam_pos[0]+j] *= weights[5*tam_pop[0]+i];
                xst[i*tam_pos[0]+j] += position[(2*tam_pop[0]+xr_list[0])*(tam_pos[0])+j];

                if(xst[i*tam_pos[0]+j]<pos_min[0])
                {
                    xst[i*tam_pos[0]+j] = pos_min[0];
                }
                if(xst[i*tam_pos[0]+j]>pos_max[0])
                {
                    xst[i*tam_pos[0]+j] = pos_max[0];
                }

                if(j == mutation_index_l || (mutation_chance_l[j] < weights[4*tam_pop[0]+i]))
                {
                    xst[i*tam_pos[0]+j] =
                    personal_best_p[i*tamPersonal+whatPersonal_l*tam_pos[0]+j];
                }
            }
        }

        // avaliar xst
        if(func_n[0]==11)
        {
            zdt1_device(xst, tam_pos, xst_fitness, i);
        }
        if(func_n[0]==12)
        {
            zdt2_device(xst, tam_pos, xst_fitness, i);
        }
        if(func_n[0]==13)
        {
            zdt3_device(xst, tam_pos, xst_fitness, i);
        }
        if(func_n[0]==14)
        {
            zdt4_device(xst, tam_pos, xst_fitness, i);
        }
        if(func_n[0]==15)
        {
            zdt5(xst, tam_pos, xst_fitness, i);
        }
        if(func_n[0]==16)
        {
            zdt6(xst, tam_pos, xst_fitness, i);
        }
        if(func_n[0]==1)
        {
            dtlz1(xst, tam_pos, xst_fitness, i);
        }
        if(func_n[0]==2)
        {
            dtlz2(xst, tam_pos, xst_fitness, i);
        }
        if(func_n[0]==3)
        {
            dtlz3(xst, tam_pos, xst_fitness, i);
        }
        if(func_n[0]==4)
        {
            dtlz4(xst, tam_pos, xst_fitness, i);
        }
        if(func_n[0]==5)
        {
            dtlz5(xst, tam_pos, xst_fitness, i);
        }
        if(func_n[0]==6)
        {
            dtlz6(xst, tam_pos, xst_fitness, i);
        }
        if(func_n[0]==7)
        {
            dtlz7(xst, tam_pos, xst_fitness, i);
        }
        if(func_n[0] == 21)
        {
            wfg1(xst, tam_pos, xst_fitness, i);
        }
        if(func_n[0] == 31)
        {
            mw1(xst, tam_pos, xst_fitness, i, alpha);
        }
        if(func_n[0] == 32)
        {
            mw2(xst, tam_pos, xst_fitness, i, alpha);
        }
        if(func_n[0] == 33)
        {
            mw3(xst, tam_pos, xst_fitness, i, alpha);
        }
        if(func_n[0] == 34)
        {
            mw4(xst, tam_pos, xst_fitness, i, alpha);
        }
        if(func_n[0] == 35)
        {
            mw5(xst, tam_pos, xst_fitness, i, alpha);
        }
        if(func_n[0] == 36)
        {
            mw6(xst, tam_pos, xst_fitness, i, alpha);
        }
        if(func_n[0] == 37)
        {
            mw7(xst, tam_pos, xst_fitness, i, alpha);
        }
        if(func_n[0] == 39)
        {
            mw9(xst, tam_pos, xst_fitness, i, alpha);
        }
        if(func_n[0] == 310)
        {
            mw10(xst, tam_pos, xst_fitness, i, alpha);
        }

    //     verificar se xst domina a particula i
        if(a_dominate_b(xst_fitness+(i*tam_fit[0]), fitness+(i*tam_fit[0]), tam_fit[0], maximize))
        {
            xst_dominate[i] = 1;
            for(j=0;j<tam_pos[0];j++)
            {
                position[i*tam_pos[0]+j] = xst[i*tam_pos[0]+j];
            }

            for(j=0;j<tam_fit[0];j++)
            {
                fitness[i*tam_fit[0]+j] = xst_fitness[i*tam_fit[0]+j];
            }

            update_from_differential_mutation[i] = 1;

            update_personal_best4_validation(personal_best_p, personal_best_v, personal_best_f, tam_fit,
            tam_pos, position, fitness, personal_guide_array_size, maximize);
        }
    }
}

__global__ void inicialize_front0_mem(int *front, int *front0_mem,
int *tam, int *tam_front0_mem, double *position, int *tam_mem, int *tam_pop, int *tam_pos,
int *current_memory_size)
{
    // talvez usar alguma variavel para indicar quais posicoes da populacao estao na memoria
    int j, k, l, m=0, diff;
    tam_front0_mem[0] = 0;
    for(j=0;j<tam[0];j++)
    {
        front0_mem[j] = front[j];
        tam_front0_mem[0]+=1;
    }
    // para cada membro da memoria
//     for(j=0;j<tam_mem[0];j++)
    for(j=0;j<current_memory_size[0];j++)
    {
        diff = 0;
        // se existir um membor na posicao da memoria
        if(position[(2*tam_pop[0]+j)*tam_pos[0]]!=-1e20)
        {
            // para cada membro do front 0
            for(k=0;k<tam[0];k++)
            {
                // verificar se sao diferentes
                for(l=0;l<tam_pos[0];l++)
                {
                    if(abs(position[front[k]*tam_pos[0]+l]-position[(2*tam_pop[0]+j)*tam_pos[0]+l])>1e-6)
                    {
                        diff+=1;
                        l = 2*tam_pos[0];
                    }
                }
            }
//             if(diff == tam[0])
            if(diff == tam_pos[0])
            {
                front0_mem[tam[0]+m] = (2*tam_pop[0]+j);
                tam_front0_mem[0]+=1;
                m+=1;
            }
        }
    }
}

__global__ void inicialize_front0_mem2(int *front, int *front0_mem,
int *tam, int *tam_front0_mem, double *position, int *tam_mem, int *tam_pop, int *tam_pos,
int *current_memory_size)
{
    int j;
    tam_front0_mem[0] = 0;

    for(j=0;j<tam[0];j++)
    {
        front0_mem[j] = front[j];
        tam_front0_mem[0]+=1;
    }

    for(j=0;j<current_memory_size[0];j++)
    {
        if(position[(2*tam_pop[0]+j)*tam_pos[0]]!=-1e20)
        {
            front0_mem[tam[0]+j] = (2*tam_pop[0]+j);
            tam_front0_mem[0]+=1;
        }
    }
}

// __global__ void copy(double *vector)
// {
//     int l = blockIdx.x*blockDim.x+threadIdx.x;
//     int c = blockIdx.y*blockDim.y+threadIdx.y;
//     vector[(l+blockDim.x*gridDim.x)*blockDim.y+c] = vector[l*blockDim.y+c];
// //     printf("%d %d\n", blockDim.y, blockDim.x*gridDim.x);
// }
//
// __global__ void copy2(double *vector, double *vector2)
// {
//     int l = blockIdx.x*blockDim.x+threadIdx.x;
//     int c = blockIdx.y*blockDim.y+threadIdx.y;
//     vector2[l*blockDim.y*gridDim.y+c] = vector[l*blockDim.y*gridDim.y+c];
// //     printf("%d %d\n", blockDim.y, blockDim.x*gridDim.x);
// }

__global__ void copy_1d(int *vector, int *tam)
{
    int i = blockIdx.x*blockDim.x+threadIdx.x;

    if(i<tam[0])
    {
        vector[i+tam[0]] = vector[i];
    }
}

__global__ void copy_2d(double *vector, int *tam1, int *tam2)
{
    int i = blockIdx.x*blockDim.x+threadIdx.x;

    if(i<tam1[0]*tam2[0])
    {
        vector[i+tam1[0]*tam2[0]] = vector[i];
    }
}

__global__ void copy_2d_vv(double *vector, double *vector2, int *tam)
{
    int i = blockIdx.x*blockDim.x+threadIdx.x;

    if(i<tam[0]*6)
    {
        vector2[i] = vector[i];
    }
}

__global__ void copy_3d(double *vector, int *tam1, int *tam2, int *tam3)
{
    int i = blockIdx.x*blockDim.x+threadIdx.x;

    if(i<tam1[0]*tam2[0]*tam3[0])
    {
        vector[i+tam1[0]*tam2[0]*tam3[0]] = vector[i];
    }
}
__global__ void copy3(double *vector, double *vector2, int *tam, int *tam_pop, int *tam_mem)
{
    int l = blockIdx.x*blockDim.x+threadIdx.x,i;
    int total = 2*tam_pop[0]+tam_mem[0];

    if(l < total)
    {
        for(i=0;i<tam[0];i++)
        {
            vector2[l*tam[0]+i] = vector[l*tam[0]+i];
        }
    }
}

__global__ void sigma_eval(double *sigma_value, double *fitness, int *tam)
{
    // falta implementar maior que 2

    int i = blockIdx.x*blockDim.x+threadIdx.x, j;
    double denominator = 0;

    for(j=0;j<tam[0];j++)
    {
        denominator += fitness[i*tam[0]+j]*fitness[i*tam[0]+j];
    }
    if(tam[0] == 2)
    {
        sigma_value[i*tam[0]] = (
        fitness[i*tam[0]]   * fitness[i*tam[0]] -
        fitness[i*tam[0]+1] * fitness[i*tam[0]+1]
        )/denominator;
    }
    if(tam[0] == 3)
    {
        sigma_value[i*tam[0]] = (
        fitness[i*tam[0]]   * fitness[i*tam[0]] -
        fitness[i*tam[0]+1] * fitness[i*tam[0]+1]
        )/denominator;

        sigma_value[i*tam[0]+1] = (
        fitness[i*tam[0]+1]   * fitness[i*tam[0]+1] -
        fitness[i*tam[0]+2] * fitness[i*tam[0]+2]
        )/denominator;

        sigma_value[i*tam[0]+2] = (
        fitness[i*tam[0]+2]   * fitness[i*tam[0]+2] -
        fitness[i*tam[0]] * fitness[i*tam[0]]
        )/denominator;
    }
}

__global__ void sigma_nearest(double *sigma_value, int *front, int *tam_front,
int *rank, int *tam_pop, int *tam_mem, int *tam_fit, int *global_best, double *fitness)
{
    int i = threadIdx.x, j,p=0, k, different = 0;
    double sigma_distance = 1e20, new_distance=0, temp;

    if(rank[i]==0)
    {
        for(j=0;j<tam_mem[0];j++)
        {
            different = 0;
            for(k=0;k<tam_fit[0];k++)
            {
                if(abs(fitness[i*tam_fit[0]+k]- fitness[(2*tam_pop[0]+j)*tam_fit[0]+k]) > 0)
                {
                    different = 1;
                    break;
                }
            }
            if(tam_fit[0]==2)
            {
                new_distance = abs(sigma_value[i*tam_fit[0]]-sigma_value[(2*tam_pop[0]+j)*tam_fit[0]]);
                //o if(new_distance<sigma_distance && new_distance>0) talvez tenha que ser corrigido
                //se a particula tiver na memoria
//                 if(new_distance<sigma_distance && new_distance>0)
//                 if(i==19)
//                 {
//                     printf("%d %d\n", 256+j, different);
//                 }
                if(new_distance<sigma_distance && different==1)
                {
                    sigma_distance = new_distance;
                    global_best[i] = (2*tam_pop[0]+j);
                }
            }
            if(tam_fit[0]==3)
            {
                new_distance = 0;
                temp = sigma_value[i*tam_fit[0]]-sigma_value[(2*tam_pop[0]+j)*tam_fit[0]];
                temp *= temp;
                new_distance += temp;
                temp = sigma_value[i*tam_fit[0]+1]-sigma_value[(2*tam_pop[0]+j)*tam_fit[0]+1];
                temp *= temp;
                new_distance += temp;
                temp = sigma_value[i*tam_fit[0]+2]-sigma_value[(2*tam_pop[0]+j)*tam_fit[0]+2];
                temp *= temp;
                new_distance += temp;
                new_distance = sqrt(new_distance);

                if(new_distance<sigma_distance && different==1)
                {
                    sigma_distance = new_distance;
                    global_best[i] = (2*tam_pop[0]+j);
                }
            }
        }
    }
    else
    {
        for(j=0;j<rank[i]-1;j++)
        {
            p+=tam_front[j];
        }
//         if(i==82)
//         {
//             printf("i=%d, front=%d %d, p=%d \n",i,rank[i], rank[i]-1,p);
//         }
        for(j=0;j<tam_front[rank[i]-1];j++)
        {
//             if(i == 19)
//             {
//                 printf("%lf ",sigma_value[(2*tam_pop[0]+j)*tam_fit[0]]);
//             }
            //             if(i == 19)
//             if(i == 82)
//             {
//                 printf("%d %lf %lf\n",front[p+j], sigma_value[(front[p+j])*2], sigma_value[(front[p+j])*2]+1);
//             }
            if(tam_fit[0]==2)
            {
                new_distance = abs(sigma_value[i*tam_fit[0]]-sigma_value[(front[p+j]*tam_fit[0])]);
//                 if(i == 6)
//                 {
//                     printf("f=%d n=%lf\n", front[p+j], new_distance);
//                 }
                if(new_distance<sigma_distance && front[p+j]!=i)
                {
                    sigma_distance = new_distance;
                    global_best[i] = front[p+j];
                }
            }
            if(tam_fit[0]==3)
            {
                new_distance = 0;
                temp = sigma_value[i*tam_fit[0]]-sigma_value[(front[p+j])*tam_fit[0]];
                temp *= temp;
                new_distance += temp;
                temp = sigma_value[i*tam_fit[0]+1]-sigma_value[(front[p+j])*tam_fit[0]+1];
                temp *= temp;
                new_distance += temp;
                temp = sigma_value[i*tam_fit[0]+2]-sigma_value[(front[p+j])*tam_fit[0]+2];
                temp *= temp;
                new_distance += temp;
                new_distance = sqrtf(new_distance);

                if(new_distance<sigma_distance && different==1)
                {
                    sigma_distance = new_distance;
                    global_best[i] = front[p+j];;
                }
            }
        }
    }
//     int nearest_particle=-1;
//     double sigma_distance = 1e20;

//      def sigma_nearest(self,particle,search_pool):
//         sigma_distance = sys.maxsize
//         nearest_particle = None
//         for p in search_pool:
//             if particle != p:
//                 new_distance = self.euclidian_distance(particle.sigma_value, p.sigma_value)
//                 if sigma_distance > new_distance:
//                     sigma_distance = new_distance
//                     nearest_particle = p
//         if nearest_particle is None: #se a distancia inicial e maxima, entao
//             # sempre vai ter um melhor pq
//             # 1 distancia nao sera menor que a maxima?
//             nearest_particle = particle
//         nearest_particle = copy.deepcopy(nearest_particle)
//         # entao global best nao e a melhor posicao entre todas as particulas?
//         particle.global_best = nearest_particle
}

__global__ void move_particle(double *weights, double *weights_copy, double *personal_best_p, double *position,
double *velocity, int *personal_guide_array_size,
double *communication_probability, int *global_best, double *velocity_max_value,
double *velocity_min_value, int *seed, int *tam_pop2, int *tam_pos2)
{
    int i = blockIdx.x*blockDim.x+threadIdx.x;
    int j = blockIdx.y*blockDim.y+threadIdx.y;
//     int j = threadIdx.y;
    int k;
    int tam_pop = tam_pop2[0];
    int tam_pos = tam_pos2[0];
    double cooperation_term=0, temp1, temp2, temp3;
    int whatPersonal_l = 0;
    int whatPersonal_l_2 = 0;
    double cooperation_rand_l;
    double cooperation_rand_l_2;
    double communication_l[1000];
    double communication_copy_l[1000];
    curandState state;

    if(i<tam_pop && j<tam_pos)
    {
        curand_init(seed[0], i, 0, &state);

        cooperation_rand_l = curand_uniform(&state);
        cooperation_rand_l_2 = curand_uniform(&state);

        for(k=0;k<tam_pos;k++)
        {
            communication_l[k] = curand_uniform(&state);
            communication_copy_l[k] = curand_uniform(&state);
        }

        if(communication_l[j]<communication_probability[0])
        {
            cooperation_term = weights[3*tam_pop+i]*cooperation_rand_l+1;
            cooperation_term *= position[global_best[i]*tam_pos+j];
            cooperation_term -= position[i*tam_pos+j];
            cooperation_term *= weights[2*tam_pop+i];
        }
        temp1 = velocity[i*tam_pos+j]*weights[0*tam_pop+i];
        temp2 = weights[1*tam_pop+i]*(personal_best_p[i*personal_guide_array_size[0]*tam_pos+
        whatPersonal_l*tam_pos+j] - position[i*tam_pos+j]);
        temp3 = cooperation_term;

        velocity[i*tam_pos+j] = temp1+temp2+temp3;
        if(velocity[i*tam_pos+j]<velocity_min_value[j])
        {
            velocity[i*tam_pos+j]=velocity_min_value[j];
        }
        if(velocity[i*tam_pos+j]>velocity_max_value[j])
        {
            velocity[i*tam_pos+j]=velocity_max_value[j];
        }

        if(communication_copy_l[j]<communication_probability[0])
        {
            cooperation_term = weights_copy[3*tam_pop+i]*cooperation_rand_l_2+1;
            cooperation_term *= position[global_best[i+tam_pop]*tam_pos+j];
            cooperation_term -= position[(i+tam_pop)*tam_pos+j];
            cooperation_term *= weights_copy[2*tam_pop+i];
        }
        else
        {
            cooperation_term = 0;
        }

        temp1  = velocity[(i+tam_pop)*tam_pos+j]*weights_copy[0*tam_pop+i];
        temp2  = weights_copy[1*tam_pop+i]*(personal_best_p[i*
        personal_guide_array_size[0]*tam_pos+whatPersonal_l_2*tam_pos+j] -
        position[(i+tam_pop)*tam_pos+j]);
        velocity[(i+tam_pop)*tam_pos+j] = temp1+temp2+cooperation_term;

        if(velocity[(i+tam_pop)*tam_pos+j]<velocity_min_value[j])
        {
            velocity[(i+tam_pop)*tam_pos+j]=velocity_min_value[j];
        }
        if(velocity[(i+tam_pop)*tam_pos+j]>velocity_max_value[j])
        {
            velocity[(i+tam_pop)*tam_pos+j]=velocity_max_value[j];
        }
    }
}

__global__ void move_particle2(double *position, double *velocity, double *position_min_value,
double *position_max_value, int *tam_pop2, int *tam_pos2)
{
    int i = blockIdx.x*blockDim.x+threadIdx.x;
//     int j = threadIdx.y;
    int j = blockIdx.y*blockDim.y+threadIdx.y;

//     int tam_pos = blockDim.y;
    int tam_pos = tam_pos2[0];
    int tam_pop = tam_pop2[0];

    if(i<tam_pop && j<tam_pos)
    {
        position[i*tam_pos+j]+=velocity[i*tam_pos+j];

        if(position[i*tam_pos+j]<position_min_value[j])
        {
            position[i*tam_pos+j]=position_min_value[j];
        }
        if(position[i*tam_pos+j]>position_max_value[j])
        {
            position[i*tam_pos+j]=position_max_value[j];
        }

        if((position[i*tam_pos+j] == position_min_value[j]) && velocity[i*tam_pos+j]<0)
        {
            velocity[i*tam_pos+j]*=-1;
        }
        if((position[i*tam_pos+j] == position_max_value[j]) && velocity[i*tam_pos+j]>0)
        {
            velocity[i*tam_pos+j]*=-1;
        }
    }
}

__global__ void nextgen1(int *fronts, int *tam, int *tam_pop)
{
    int total=0,  i=0;
    while((total+tam[i])<=tam_pop[0])
    {
        total+=tam[i];
        i++;
    }
//     printf("%d %d %d %d\n",total,i, tam_pop[0], tam[i]);
    //numero do front que sera usado oc rowding distance
    tam[tam_pop[0]-2] = i;

    //numero de particulas ate o front a ser analisado
    tam[tam_pop[0]-1] = tam_pop[0]-total;
    // acheie stranho e vou testar - 260625
    //     tam[tam_pop[0]-1] = total;
}

__global__ void population_index_inicialization(int *index, int* tam_pop)
{
    int i = blockIdx.x*blockDim.x+threadIdx.x;

    if(i<2*tam_pop[0])
    {
        index[i] = threadIdx.x;
    }
}

__global__ void index_sort_par(double *crowd_distance, int *index)
{
    int temp;
    double temp2;
    int j = threadIdx.x*2;

    if(index[j]>index[j+1])
    {
//                 printf("%d %d %lf %lf\n",p1,p2, fitness[p1*dim_fitness[0]+i[0]],fitness[p2*dim_fitness[0]+i[0]]);
                temp = index[j];
                index[j] = index[j+1];
                index[j+1] = temp;
                temp2 = crowd_distance[j];
                crowd_distance[j] = crowd_distance[j+1];
                crowd_distance[j+1] = temp2;
    }
}

__global__ void index_sort_impar(double *crowd_distance, int *index)
{
    int temp;
    double temp2;
    int j = threadIdx.x*2+1;

    if(index[j]>index[j+1])
    {
//                 printf("%d %d %lf %lf\n",p1,p2, fitness[p1*dim_fitness[0]+i[0]],fitness[p2*dim_fitness[0]+i[0]]);
                temp = index[j];
                index[j] = index[j+1];
                index[j+1] = temp;
                temp2 = crowd_distance[j];
                crowd_distance[j] = crowd_distance[j+1];
                crowd_distance[j+1] = temp2;
    }
}

__global__ void create_next_gen1_pop(double *position1, double *position2, int *selected,
int *tam_pop, int *pos_dim)
{
    int l = blockIdx.x*blockDim.x+threadIdx.x;
    int c = blockIdx.y*blockDim.y+threadIdx.y;
//     int tam = blockDim.y;
    if(l<tam_pop[0] && c<pos_dim[0])
    {
        position2[l*pos_dim[0]+c] = position1[selected[l]*pos_dim[0]+c];
    }
}

__global__ void create_next_gen2_pop(double *position1, double *position2,
int *tam_pop, int *pos_dim)
{
    int l = blockIdx.x*blockDim.x+threadIdx.x;
    int c = blockIdx.y*blockDim.y+threadIdx.y;
//     int tam = blockDim.y;

    if(l<tam_pop[0] && c<pos_dim[0])
    {
        position1[l*pos_dim[0]+c] = position2[l*pos_dim[0]+c];
    }
}

__global__ void create_next_gen1_pb(double *position1, double *position2, int *selected,
int *tam_pop, int *pos_dim, int *personal_guide_array_size)
{
    int l = blockIdx.x*blockDim.x+threadIdx.x;
    int c = blockIdx.y*blockDim.y+threadIdx.y;
//     int tam = blockDim.y;

    if(l<tam_pop[0] && c<pos_dim[0]*personal_guide_array_size[0])
    {
        position2[l*pos_dim[0]*personal_guide_array_size[0]+c] =
        position1[selected[l]*pos_dim[0]*personal_guide_array_size[0]+c];
    }
}

__global__ void create_next_gen2_pb(double *position1, double *position2,
int *tam_pop, int *pos_dim, int *personal_guide_array_size)
{
    int l = blockIdx.x*blockDim.x+threadIdx.x;
    int c = blockIdx.y*blockDim.y+threadIdx.y;
//     int tam = blockDim.y;

    if(l<tam_pop[0] && c<pos_dim[0]*personal_guide_array_size[0])
    {
        position1[l*pos_dim[0]*personal_guide_array_size[0]+c] =
        position2[l*pos_dim[0]*personal_guide_array_size[0]+c];
    }
}

__global__ void create_next_gen1(double *position1, double *position2, int *selected)
{
    int l = blockIdx.x*blockDim.x+threadIdx.x;
    int c = blockIdx.y*blockDim.y+threadIdx.y;
    int tam = blockDim.y;

    position2[l*tam+c] = position1[selected[l]*tam+c];
}

__global__ void create_next_gen2(double *position1, double *position2)
{
    int l = blockIdx.x*blockDim.x+threadIdx.x;
    int c = blockIdx.y*blockDim.y+threadIdx.y;
    int tam = blockDim.y;

    position1[l*tam+c] = position2[l*tam+c];
}

// __global__ void orderTest1(double *position1, double *velocity, int *index, int *index2, int *linhas, int *colunas)
// {
//     double temp;
//     int temp2, temp3;
//
//     for(int i=0;i<linhas[0];i++)
//     {
//         if(index[i]!=i)
//         {
//             for(int j=0;j<colunas[0];j++)
//             {
//                 temp = position1[i*colunas[0]+j];
//                 position1[i*colunas[0]+j] = position1[index2[i]*colunas[0]+j];
//                 position1[index2[i]*colunas[0]+j] = temp;
//             }
//             temp3 = index2[i];
//             temp2 = index[i];
//             index[i] = index[temp3];
//             index[temp3] = temp2;
//             temp2 = index2[i];
//             index2[i] = index2[temp3];
//             index2[temp3] = temp2;
//         }
//     }
// }

__global__ void index_sort_par2(int *index, double *position, double *velocity, int *tam)
{
    int temp;
    double temp2;
    int j = threadIdx.x*2;

    if(index[j]>index[j+1])
    {
        for(int i=0;i<tam[0];i++)
        {
//             if(j==6)
//             {
//                 printf("%lf %lf", position[index[j]*tam[0]+i], position[(index[j+1])*tam[0]+i]);
//             }
            temp2 = position[index[j]*tam[0]+i];
            position[index[j]*tam[0]+i] = position[(index[j+1])*tam[0]+i];
            position[(index[j+1])*tam[0]+i] = temp2;
            temp2 = velocity[index[j]*tam[0]+i];
            velocity[index[j]*tam[0]+i] = velocity[(index[j+1])*tam[0]+i];
            velocity[(index[j+1])*tam[0]+i] = temp2;
//             if(j==6)
//             {
//                 printf(" %lf %lf\n", position[j*tam[0]+i], position[(j+1)*tam[0]+i]);
//             }
        }
        temp = index[j];
        index[j] = index[j+1];
        index[j+1] = temp;
    }
}

__global__ void index_sort_impar2(int *index, double *position, double *velocity, int *tam)
{
    int temp;
    double temp2;
    int j = threadIdx.x*2+1;

    if(index[j]>index[j+1])
    {
        for(int i=0;i<tam[0];i++)
        {
            temp2 = position[index[j]*tam[0]+i];
            position[index[j]*tam[0]+i] = position[index[j+1]*tam[0]+i];
            position[index[j+1]*tam[0]+i] = temp2;
            temp2 = velocity[index[j]*tam[0]+i];
            velocity[index[j]*tam[0]+i] = velocity[(index[j+1])*tam[0]+i];
            velocity[(index[j+1])*tam[0]+i] = temp2;
        }
        temp = index[j];
        index[j] = index[j+1];
        index[j+1] = temp;
    }
}

// __global__ void initial_memory_velocity1(double *velocity, int *fronts, int *tam_pop, int *tam_pos)
// {
//     int l = blockIdx.x*blockDim.x+threadIdx.x;
//     int c = blockIdx.y*blockDim.y+threadIdx.y;
//
//     velocity[(l+2*tam_pop[0])*tam_pos[0]] =
//     (double)fronts[l];
// }
__global__ void initial_memory_velocity1(int *initial_mem, int *fronts)
{
    int l = blockIdx.x*blockDim.x+threadIdx.x;

    initial_mem[l] = fronts[l];
}

// __global__ void initial_memory_velocity2(double *velocity, int *fronts, int *tam_pop, int *tam_pos)
// {
//     int l = blockIdx.x*blockDim.x+threadIdx.x;
//     int i;
//     int temp = (int)velocity[(l+2*tam_pop[0])*tam_pos[0]];
//
//     for(i=0;i<tam_pos[0];i++)
//     {
//         velocity[(l+2*tam_pop[0])*tam_pos[0]+i] = velocity[temp*tam_pos[0]+i];
//     }
// }

__global__ void initial_memory_velocity2(double *velocity, int *initial, int *tam_pop, int *tam_pos)
{
    int l = blockIdx.x*blockDim.x+threadIdx.x;
    int i;

    for(i=0;i<tam_pos[0];i++)
    {
        velocity[(l+2*tam_pop[0])*tam_pos[0]+i] = velocity[initial[l]*tam_pos[0]+i];
    }
}

__global__ void initial_memory_velocity3(double *velocity, int *fronts, int *tam_pop, int *tam_pos)
{
    int l = blockIdx.x*blockDim.x+threadIdx.x;
    int i;

    for(i=0;i<tam_pos[0];i++)
    {
        velocity[(l+2*tam_pop[0])*tam_pos[0]+i] = velocity[fronts[l]*tam_pos[0]+i];
    }
}

__global__ void initial_memory_velocity(double *vector, int *tam_pop)
{
    int l = blockIdx.x*blockDim.x+threadIdx.x;
    int c = blockIdx.y*blockDim.y+threadIdx.y;

    vector[(l+2*tam_pop[0])*blockDim.y*gridDim.y+c] = vector[l*blockDim.y*gridDim.y+c];
}

__global__ void population_init(double *position, int *position_dim, int *seed, double *min, double *max)
{
    int i = threadIdx.x, j;
//     int j = blockIdx.y*blockDim.y+threadIdx.y;

    curandState state;
    curand_init(seed[0], i, 0, &state);
//     printf("%lf %lf\n", min[0], max[0]);
    for(j=0;j<position_dim[0];j++)
    {
        position[i*position_dim[0]+j] = curand_uniform(&state);
        position[i*position_dim[0]+j] = position[i*position_dim[0]+j]*(max[0]-min[0])+min[0];
    }
}

__global__ void init_population(double *position, int *position_dim, int *seed, double *min,
 double *max, int *population_size)
{
//     int i = threadIdx.x, j;
//     int j = blockIdx.y*blockDim.y+threadIdx.y;
    int i = blockIdx.x*blockDim.x+threadIdx.x;
    int j = blockIdx.y*blockDim.y+threadIdx.y;

    curandState state;
//     curand_init(seed[0], i, 0, &state);
//     printf("%lf %lf\n", min[0], max[0]);

//     for(j=0;j<position_dim[0];j++)
//     {
//         position[i*position_dim[0]+j] = curand_uniform(&state);
//         position[i*position_dim[0]+j] = position[i*position_dim[0]+j]*(max[j]-min[j])+min[j];
//     }

    curand_init(seed[0], i, j, &state);
    if(i< population_size[0] && j < position_dim[0])
    {
        position[i*position_dim[0]+j] = curand_uniform(&state);
        position[i*position_dim[0]+j] = position[i*position_dim[0]+j]*(max[j]-min[j])+min[j];
    }
}

// __global__ void mutate_weights(double *weights, int *seed, int *tam_pop, double *mutation_rate)
// {
//     int i = threadIdx.x;
//     int j = threadIdx.y;
//     curandState state;
//
//     curand_init(seed[0], i*tam_pop[0]+j, 0, &state);
//     weights[i*tam_pop[0]+j] = curand_normal(&state)*mutation_rate[0];
// }

__global__ void mutate_weights(double *weights, int *seed, int *tam_pop, double *mutation_rate)
{
    int i = blockIdx.x*blockDim.x+threadIdx.x;
    curandState state;

    if(i < 6 * tam_pop[0])
    {
           curand_init(seed[0], i, 0, &state);
           weights[i] = curand_normal(&state)*mutation_rate[0];
    }
}

__global__ void mutate_weights2(double *weights, int *tam_pop)
{
    int i = blockIdx.x*blockDim.x+threadIdx.x;
    int j = blockIdx.y*blockDim.y+threadIdx.y;

    if(j<tam_pop[0])
    {
        if(weights[i*tam_pop[0]+j]<0)
        {
            weights[i*tam_pop[0]+j] = 0;
        }
        if(weights[i*tam_pop[0]+j]>1)
        {
            weights[i*tam_pop[0]+j] = 1;
        }
   }
}

__global__ void mutate_weights3(double *weights, int *tam_pop)
{
    int i = blockIdx.x*blockDim.x+threadIdx.x;

    if(i<tam_pop[0])
    {
        if(weights[4*tam_pop[0]+i]<0)
        {
            weights[4*tam_pop[0]+i] = 0;
        }
        if(weights[4*tam_pop[0]+i]>0.5)
        {
            weights[4*tam_pop[0]+i] = 0.5;
        }

        if(weights[5*tam_pop[0]+i]<0)
        {
            weights[5*tam_pop[0]+i] = 0;
        }
        if(weights[5*tam_pop[0]+i]>2.0)
        {
            weights[5*tam_pop[0]+i] = 2.0;
        }
    }
}
}