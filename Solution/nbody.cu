#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include "timer.h"

#define SOFTENING 0.01f 

// Optimized constants
#define THREADS_PER_BLOCK 256
#define TILE_SIZE 256
#define NUM_STREAMS 16

#define ABS(val)    ((val)<0.0 ? (-(val)) : (val))
#define IS_NULL(ptr) (((ptr) == (NULL)) ? 1 : 0)

// CUDA error checking macro
#define CUDA_CHECK(call) \
    do { \
        cudaError_t err = call; \
        if (err != cudaSuccess) { \
            fprintf(stderr, "CUDA Error: %s (err_num=%d) at %s:%d\n", \
                cudaGetErrorString(err), err, __FILE__, __LINE__); \
            exit(EXIT_FAILURE); \
        } \
    } while (0)

#ifdef DEBUG
    int debug = 1;
#else
    int debug = 0;
#endif

// CPU structures (AoS - unchanged)
typedef struct {
    float x, y, z;
} coords_T;

typedef struct {
    float vx, vy, vz;
} velocity_T;

typedef struct {
    coords_T *all_coords;
    velocity_T *all_velocities;
} Universe;

// GPU Structure-of-Arrays (SoA)
typedef struct {
    float *x, *y, *z;
    float *vx, *vy, *vz;
} Body_Arrays;

// Per device context for multiple GPU
typedef struct {
    Body_Arrays d_bodies;
    coords_T *h_out_coords;
    cudaStream_t streams[NUM_STREAMS];
    int num_systems_local;
    int sys_start_global;
} DeviceContext;

Universe h_data, h_device_out, omp_out;

/* CPU CODE */
void bodyForce(coords_T * coor_p, velocity_T * vel_p, float dt, int n) {
    int i, j;
    float Fx, Fy, Fz, dx, dy, dz, distSqr, invDist, invDist3;

    for (i = 0; i < n; i++) {
	    Fx = 0.0f;
    	Fy = 0.0f;
    	Fz = 0.0f;
        
        coords_T curr = coor_p[i];
    	for (j = 0; j < n; j++) {
	        dx = coor_p[j].x - coor_p[i].x;
            dy = coor_p[j].y - coor_p[i].y;
            dz = coor_p[j].z - coor_p[i].z;
            distSqr = dx * dx + dy * dy + dz * dz + SOFTENING;
            invDist = 1.0f / sqrtf(distSqr);
            invDist3 = invDist * invDist * invDist;

            Fx += dx * invDist3;
            Fy += dy * invDist3;
            Fz += dz * invDist3;
        }

        vel_p[i].vx += dt * Fx;
        vel_p[i].vy += dt * Fy;
        vel_p[i].vz += dt * Fz;
    }
}

/* Integrate positions.
    - array of bodies
    - time step
    - number of bodies
*/
void integrate(coords_T * coor_p, velocity_T * vel_p, float dt, int n) {
    int i;
    
    for (i = 0; i < n; i++) {
	    coor_p[i].x += vel_p[i].vx * dt;
        coor_p[i].y += vel_p[i].vy * dt;
        coor_p[i].z += vel_p[i].vz * dt;
    }
}

/* OMP CODE */
void run_omp(int num_systems, int nIters, int bodies_per_system, float dt) {
    double totalTime;
    int iter, sys;
    double interactions_per_system, total_interactions;

    printf("\n----------------------------\n");
    printf("Running parallel CPU simulation for %d systems...\n",
           num_systems);

    totalTime = 0.0;
    // CPU code
    StartTimer();

    /* Time-steps */        
    for (iter = 1; iter <= nIters; iter++) {
        /* Galaxies */    
        #pragma omp parallel for
	    for (sys = 0; sys < num_systems; sys++) {
	        /* Calculate offset for the galaxy */
	        // private pointer for each thread. One thread per system
            coords_T *coords_ptr = &(omp_out.all_coords[sys * bodies_per_system]);
            velocity_T *veloc_ptr = &(omp_out.all_velocities[sys * bodies_per_system]);
	        
	        /* Compute forces & integrate for the galaxy */
	        bodyForce(coords_ptr, veloc_ptr, dt, bodies_per_system);
	        integrate(coords_ptr, veloc_ptr, dt, bodies_per_system);
        }
    }
    totalTime = GetTimer() / 1000.0;

    /* Metrics calculation */
    interactions_per_system = (double) bodies_per_system * bodies_per_system;
    total_interactions = interactions_per_system * num_systems * nIters;
    printf("Total Time CPU: %.3f seconds\n", totalTime);
    printf("Average Throughput CPU: %0.3f Billion Interactions / second\n\n",
           1e-9 * total_interactions / totalTime);

    /* Dump final state of System 0, Body 0 and 1 for verification comparison */
    printf("Final position of System 0, Body 0: %.4f, %.4f, %.4f\n",
           omp_out.all_coords[0].x, omp_out.all_coords[0].y, omp_out.all_coords[0].z);
    printf("Final position of System 0, Body 1: %.4f, %.4f, %.4f\n",
           omp_out.all_coords[1].x, omp_out.all_coords[1].y, omp_out.all_coords[1].z);
    printf("----------------------------\n");
}


/* GPU KERNELS */
__global__ void update_velocities(
    const float* __restrict__ x, const float* __restrict__ y, const float* __restrict__ z,
    float* __restrict__ vx, float* __restrict__ vy, float* __restrict__ vz,
    int bodies_per_system, float dt)
{       
    // Body index within this system
    int body_id = threadIdx.x + blockIdx.x * blockDim.x;
    if (body_id >= bodies_per_system) return;

    // Shared memory for tiling
    __shared__ float shared_x[TILE_SIZE];
    __shared__ float shared_y[TILE_SIZE];
    __shared__ float shared_z[TILE_SIZE];

    // Load this body position
    float body_x = x[body_id];
    float body_y = y[body_id];
    float body_z = z[body_id];

    // Accumulate forces
    float fx = 0.0f, fy = 0.0f, fz = 0.0f;

    int num_tiles = (bodies_per_system + TILE_SIZE - 1) / TILE_SIZE;

    // Loop over tiles
    #pragma unroll 4
    for (int tile = 0; tile < num_tiles; tile++) {
        int tile_idx = tile * TILE_SIZE + threadIdx.x;

        // Cooperatively load tile into shared memory
        if (tile_idx < bodies_per_system) {
            shared_x[threadIdx.x] = x[tile_idx];
            shared_y[threadIdx.x] = y[tile_idx];
            shared_z[threadIdx.x] = z[tile_idx];
        }
        __syncthreads();

        // Compute interactions with all bodies in this tile
        #pragma unroll 8
        for (int j = 0; j < TILE_SIZE; j++) {
            int idx = tile * TILE_SIZE + j;
            if (idx >= bodies_per_system) break;

            float dx = shared_x[j] - body_x;
            float dy = shared_y[j] - body_y;
            float dz = shared_z[j] - body_z;

            float distSqr  = fmaf(dx, dx, fmaf(dy, dy, fmaf(dz, dz, SOFTENING)));
            float invDist  = rsqrtf(distSqr);
            float invDist3 = invDist * invDist * invDist;

            fx = fmaf(dx, invDist3, fx);
            fy = fmaf(dy, invDist3, fy);
            fz = fmaf(dz, invDist3, fz);
        }
        __syncthreads();
    }

    // Update velocities
    vx[body_id] += dt * fx;
    vy[body_id] += dt * fy;
    vz[body_id] += dt * fz;
}

__global__ void update_positions(
    float* __restrict__ x, float* __restrict__ y, float* __restrict__ z,
    const float* __restrict__ vx, const float* __restrict__ vy, const float* __restrict__ vz,
    int bodies_per_system, float dt)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= bodies_per_system) return;

    x[i] += vx[i] * dt;
    y[i] += vy[i] * dt;
    z[i] += vz[i] * dt;
}

/* MULTIPLE GPU RUNNER */
void run_device_multi_gpu(int num_systems, int bodies_per_system, int nIters, float dt){
    double totalTime;
    int total_bodies = num_systems * bodies_per_system;
    int bytes_coords = total_bodies * sizeof(coords_T);
    int bytes_vels = total_bodies * sizeof(velocity_T);

    // Detect GPUs
    int dev_count = 0;
    CUDA_CHECK(cudaGetDeviceCount(&dev_count));
    if (dev_count == 0) {
        printf("No CUDA devices found\n");
        exit(1);
    }

    printf("Using %d GPU(s) for simulation\n", dev_count);

    // Allocate per-device contexts
    DeviceContext *dev_context = (DeviceContext*)malloc(dev_count * sizeof(DeviceContext));

    // Partition systems across GPUs
    int base_sys_per_dev = num_systems / dev_count;
    int extra = num_systems % dev_count;

    int sys_start = 0;
    for (int dev = 0; dev < dev_count; dev++) {
        CUDA_CHECK(cudaSetDevice(dev));

        int nsys = base_sys_per_dev + (dev < extra ? 1 : 0);
        dev_context[dev].num_systems_local = nsys;
        dev_context[dev].sys_start_global = sys_start;

        int bodies_local = nsys * bodies_per_system;

        if (nsys > 0) {
            // Allocate SoA device memory
            CUDA_CHECK(cudaMalloc((void**)&dev_context[dev].d_bodies.x,  bodies_local * sizeof(float)));
            CUDA_CHECK(cudaMalloc((void**)&dev_context[dev].d_bodies.y,  bodies_local * sizeof(float)));
            CUDA_CHECK(cudaMalloc((void**)&dev_context[dev].d_bodies.z,  bodies_local * sizeof(float)));
            CUDA_CHECK(cudaMalloc((void**)&dev_context[dev].d_bodies.vx, bodies_local * sizeof(float)));
            CUDA_CHECK(cudaMalloc((void**)&dev_context[dev].d_bodies.vy, bodies_local * sizeof(float)));
            CUDA_CHECK(cudaMalloc((void**)&dev_context[dev].d_bodies.vz, bodies_local * sizeof(float)));

            // Allocate pinned host output buffer
            CUDA_CHECK(cudaHostAlloc((void**)&dev_context[dev].h_out_coords,
                                     bodies_local * sizeof(coords_T),
                                     cudaHostAllocMapped));

            // Create streams
            for (int s = 0; s < NUM_STREAMS; s++) {
                CUDA_CHECK(cudaStreamCreate(&dev_context[dev].streams[s]));
            }
        } else {
            dev_context[dev].d_bodies.x = NULL;
            dev_context[dev].h_out_coords = NULL;
        }

        sys_start += nsys;
    }

    // Temporary host SoA buffers for packing
    Body_Arrays body_array;
    CUDA_CHECK(cudaHostAlloc((void**)&body_array.x,  total_bodies*sizeof(float), cudaHostAllocDefault));
    CUDA_CHECK(cudaHostAlloc((void**)&body_array.y,  total_bodies*sizeof(float), cudaHostAllocDefault));
    CUDA_CHECK(cudaHostAlloc((void**)&body_array.z,  total_bodies*sizeof(float), cudaHostAllocDefault));
    CUDA_CHECK(cudaHostAlloc((void**)&body_array.vx, total_bodies*sizeof(float), cudaHostAllocDefault));
    CUDA_CHECK(cudaHostAlloc((void**)&body_array.vy, total_bodies*sizeof(float), cudaHostAllocDefault));
    CUDA_CHECK(cudaHostAlloc((void**)&body_array.vz, total_bodies*sizeof(float), cudaHostAllocDefault));

    // Pack AoS -> SoA on host
    for (int i = 0; i < total_bodies; i++) {
        body_array.x[i]  = h_data.all_coords[i].x;
        body_array.y[i]  = h_data.all_coords[i].y;
        body_array.z[i]  = h_data.all_coords[i].z;
        body_array.vx[i] = h_data.all_velocities[i].vx;
        body_array.vy[i] = h_data.all_velocities[i].vy;
        body_array.vz[i] = h_data.all_velocities[i].vz;
    }
    // H2D: Copy initial data to each GPU
    for (int dev = 0; dev < dev_count; dev++) {
    if (dev_context[dev].num_systems_local == 0) continue;
    CUDA_CHECK(cudaSetDevice(dev));

    int nsys = dev_context[dev].num_systems_local;
    int sys_start_global = dev_context[dev].sys_start_global;
    size_t bytes_sys = (size_t)bodies_per_system * sizeof(float);

    for (int local_sys = 0; local_sys < nsys; local_sys++) {
        int stream_id = local_sys % NUM_STREAMS;

        int global_sys = sys_start_global + local_sys;
        int body_offset_global = global_sys * bodies_per_system;
        int body_offset_local  = local_sys  * bodies_per_system;

        CUDA_CHECK(cudaMemcpyAsync(dev_context[dev].d_bodies.x  + body_offset_local,
                                   body_array.x + body_offset_global,
                                   bytes_sys, cudaMemcpyHostToDevice,
                                   dev_context[dev].streams[stream_id]));
        CUDA_CHECK(cudaMemcpyAsync(dev_context[dev].d_bodies.y  + body_offset_local,
                                   body_array.y + body_offset_global,
                                   bytes_sys, cudaMemcpyHostToDevice,
                                   dev_context[dev].streams[stream_id]));
        CUDA_CHECK(cudaMemcpyAsync(dev_context[dev].d_bodies.z  + body_offset_local,
                                   body_array.z + body_offset_global,
                                   bytes_sys, cudaMemcpyHostToDevice,
                                   dev_context[dev].streams[stream_id]));
        CUDA_CHECK(cudaMemcpyAsync(dev_context[dev].d_bodies.vx + body_offset_local,
                                   body_array.vx + body_offset_global,
                                   bytes_sys, cudaMemcpyHostToDevice,
                                   dev_context[dev].streams[stream_id]));
        CUDA_CHECK(cudaMemcpyAsync(dev_context[dev].d_bodies.vy + body_offset_local,
                                   body_array.vy + body_offset_global,
                                   bytes_sys, cudaMemcpyHostToDevice,
                                   dev_context[dev].streams[stream_id]));
        CUDA_CHECK(cudaMemcpyAsync(dev_context[dev].d_bodies.vz + body_offset_local,
                                   body_array.vz + body_offset_global,
                                   bytes_sys, cudaMemcpyHostToDevice,
                                   dev_context[dev].streams[stream_id]));
    }
}

// computation
StartTimer();

for (int dev = 0; dev < dev_count; dev++) {
    if (dev_context[dev].num_systems_local == 0) continue;
    CUDA_CHECK(cudaSetDevice(dev));

    int nsys = dev_context[dev].num_systems_local;

    int blocks_per_sys = (bodies_per_system + THREADS_PER_BLOCK - 1) / THREADS_PER_BLOCK;
    dim3 grid(blocks_per_sys, 1);
    dim3 block(THREADS_PER_BLOCK);

    for (int local_sys = 0; local_sys < nsys; local_sys++) {
        int stream_id = local_sys % NUM_STREAMS;
        int off = local_sys * bodies_per_system;

        for (int iter = 0; iter < nIters; iter++) {
            update_velocities<<<grid, block, 0, dev_context[dev].streams[stream_id]>>>(
                dev_context[dev].d_bodies.x  + off,
                dev_context[dev].d_bodies.y  + off,
                dev_context[dev].d_bodies.z  + off,
                dev_context[dev].d_bodies.vx + off,
                dev_context[dev].d_bodies.vy + off,
                dev_context[dev].d_bodies.vz + off,
                bodies_per_system, dt);
            CUDA_CHECK(cudaGetLastError());

            update_positions<<<grid, block, 0, dev_context[dev].streams[stream_id]>>>(
                dev_context[dev].d_bodies.x  + off,
                dev_context[dev].d_bodies.y  + off,
                dev_context[dev].d_bodies.z  + off,
                dev_context[dev].d_bodies.vx + off,
                dev_context[dev].d_bodies.vy + off,
                dev_context[dev].d_bodies.vz + off,
                bodies_per_system, dt);
            CUDA_CHECK(cudaGetLastError());
        }
    }
}

// Copy results back to host
for (int dev = 0; dev < dev_count; dev++) {
    if (dev_context[dev].num_systems_local == 0) continue;
    CUDA_CHECK(cudaSetDevice(dev));

    int nsys = dev_context[dev].num_systems_local;
    int sys_start_global = dev_context[dev].sys_start_global;
    size_t bytes_sys = (size_t)bodies_per_system * sizeof(float);

    for (int local_sys = 0; local_sys < nsys; local_sys++) {
        int stream_id = local_sys % NUM_STREAMS;

        int global_sys = sys_start_global + local_sys;
        int body_offset_global = global_sys * bodies_per_system;
        int body_offset_local  = local_sys  * bodies_per_system;

        CUDA_CHECK(cudaMemcpyAsync(body_array.x + body_offset_global,
                                   dev_context[dev].d_bodies.x + body_offset_local,
                                   bytes_sys, cudaMemcpyDeviceToHost,
                                   dev_context[dev].streams[stream_id]));
        CUDA_CHECK(cudaMemcpyAsync(body_array.y + body_offset_global,
                                   dev_context[dev].d_bodies.y + body_offset_local,
                                   bytes_sys, cudaMemcpyDeviceToHost,
                                   dev_context[dev].streams[stream_id]));
        CUDA_CHECK(cudaMemcpyAsync(body_array.z + body_offset_global,
                                   dev_context[dev].d_bodies.z + body_offset_local,
                                   bytes_sys, cudaMemcpyDeviceToHost,
                                   dev_context[dev].streams[stream_id]));
    }
}

// Synchronize all GPUs
for (int dev = 0; dev < dev_count; dev++) {
    CUDA_CHECK(cudaSetDevice(dev));
    CUDA_CHECK(cudaDeviceSynchronize());
}

totalTime = GetTimer() / 1000.0;

    // Unpack SoA -> AoS
    for (int i = 0; i < total_bodies; i++) {
        h_device_out.all_coords[i].x = body_array.x[i];
        h_device_out.all_coords[i].y = body_array.y[i];
        h_device_out.all_coords[i].z = body_array.z[i];
    }

    // Cleanup
    for (int dev = 0; dev < dev_count; dev++) {
        if (dev_context[dev].num_systems_local == 0) continue;

        CUDA_CHECK(cudaSetDevice(dev));
        for (int s = 0; s < NUM_STREAMS; s++) {
            CUDA_CHECK(cudaStreamDestroy(dev_context[dev].streams[s]));
        }
        CUDA_CHECK(cudaFree(dev_context[dev].d_bodies.x));
        CUDA_CHECK(cudaFree(dev_context[dev].d_bodies.y));
        CUDA_CHECK(cudaFree(dev_context[dev].d_bodies.z));
        CUDA_CHECK(cudaFree(dev_context[dev].d_bodies.vx));
        CUDA_CHECK(cudaFree(dev_context[dev].d_bodies.vy));
        CUDA_CHECK(cudaFree(dev_context[dev].d_bodies.vz));
        CUDA_CHECK(cudaFreeHost(dev_context[dev].h_out_coords));
    }

    free(dev_context);
    CUDA_CHECK(cudaFreeHost(body_array.x));
    CUDA_CHECK(cudaFreeHost(body_array.y));
    CUDA_CHECK(cudaFreeHost(body_array.z));
    CUDA_CHECK(cudaFreeHost(body_array.vx));
    CUDA_CHECK(cudaFreeHost(body_array.vy));
    CUDA_CHECK(cudaFreeHost(body_array.vz));

    // Metrics
    double interactions_per_system = (double)bodies_per_system * bodies_per_system;
    double total_interactions = interactions_per_system * num_systems * nIters;

    printf("Multi-GPU run on %d device(s)\n", dev_count);
    printf("Total Time (GPU): %.3f seconds\n", totalTime);
    printf("Average Throughput: %0.3f Billion Interactions / second\n\n",
           1e-9 * total_interactions / totalTime);

    printf("Final position of System 0, Body 0: %.4f, %.4f, %.4f\n",
           h_device_out.all_coords[0].x,
           h_device_out.all_coords[0].y,
           h_device_out.all_coords[0].z);
    printf("Final position of System 0, Body 1: %.4f, %.4f, %.4f\n",
           h_device_out.all_coords[1].x,
           h_device_out.all_coords[1].y,
           h_device_out.all_coords[1].z);
    printf("----------------------------\n");
}

/* RESULT COMPARISON */
void compare_results(int total_bodies, int bodies_per_system) {
    int errors = 0;
    for (int i = 0; i < total_bodies; i++) {
        float dx = ABS(omp_out.all_coords[i].x - h_device_out.all_coords[i].x);
        float dy = ABS(omp_out.all_coords[i].y - h_device_out.all_coords[i].y);
        float dz = ABS(omp_out.all_coords[i].z - h_device_out.all_coords[i].z);
        const float eps = 0.001f;
        if (dx > eps || dy > eps || dz > eps) {
            errors++;
            if (errors <= 5) {
                printf("Mismatch at Body %d (System %d):\n", i, i / bodies_per_system);
                printf("  CPU: x=%.4f, y=%.4f, z=%.4f\n",
                       omp_out.all_coords[i].x, omp_out.all_coords[i].y, omp_out.all_coords[i].z);
                printf("  GPU: x=%.4f, y=%.4f, z=%.4f\n",
                       h_device_out.all_coords[i].x, h_device_out.all_coords[i].y, h_device_out.all_coords[i].z);
                printf("  Diff: %.4f, %.4f, %.4f\n", dx, dy, dz);
            }
        }
    }
    if (errors == 0) {
        printf(" Correct\n");
    } else {
        printf(" Found %d mismatches\n", errors);
    }
}


int main(const int argc, const char *argv[]) {
    int num_systems = 32;
    int bodies_per_system = 8192;
    int nIters = 20;
    const float dt = 0.01f;
    FILE *fp;
    int total_bodies = num_systems * bodies_per_system;
    float *buf;
    // Load dataset
    fp = fopen("galaxy_data.bin", "rb");
    if (fp) {
        fread(&num_systems, sizeof(int), 1, fp);
        fread(&bodies_per_system, sizeof(int), 1, fp);
        printf("Found dataset: %d systems of %d bodies.\n", num_systems, bodies_per_system);
    } else {
        printf("No dataset found. Using random initialization.\n");
        /* Random initialization if file missing */
        buf = (float*)malloc(6 * total_bodies * sizeof(float));
        for (int i = 0; i < 6 * total_bodies; i++) {
            buf[i] = 2.0f * (rand() / (float)RAND_MAX) - 1.0f;
        }
        for (int i = 0; i < total_bodies; i++) {
            h_data.all_coords[i].x = buf[6 * i + 0];
            h_data.all_coords[i].y = buf[6 * i + 1];
            h_data.all_coords[i].z = buf[6 * i + 2];    
            h_data.all_velocities[i].vx = buf[6 * i + 3];
            h_data.all_velocities[i].vy = buf[6 * i + 4];
            h_data.all_velocities[i].vz = buf[6 * i + 5];
        }
    }

    // Allocate memory
    total_bodies = num_systems * bodies_per_system;
    int bytes_coords = total_bodies * sizeof(coords_T);
    int bytes_vels = total_bodies * sizeof(velocity_T);

    CUDA_CHECK(cudaHostAlloc((void**)&h_data.all_coords, bytes_coords, cudaHostAllocMapped));
    CUDA_CHECK(cudaHostAlloc((void**)&h_data.all_velocities, bytes_vels, cudaHostAllocMapped));
    omp_out.all_coords = (coords_T*)malloc(bytes_coords);
    omp_out.all_velocities = (velocity_T*)malloc(bytes_vels);
    CUDA_CHECK(cudaHostAlloc((void**)&h_device_out.all_coords, bytes_coords, cudaHostAllocMapped));
    h_device_out.all_velocities = NULL;

    if (IS_NULL(omp_out.all_coords) || IS_NULL(omp_out.all_velocities)) {
        printf("Host: Memory allocation failed!\n");
        return 1;
    }

    // Read data
    if (fp) {
        for (int i = 0; i < total_bodies; i++) { 
            fread(&(h_data.all_coords[i]), sizeof(coords_T), 1, fp);
            fseek(fp, sizeof(velocity_T), SEEK_CUR);
        }    
        fseek(fp, 2*sizeof(int), SEEK_SET);
        for (int i = 0; i < total_bodies; i++) {
            fseek(fp, sizeof(coords_T), SEEK_CUR); 
            fread(&(h_data.all_velocities[i]), sizeof(velocity_T), 1, fp);
        }
        fclose(fp);
    } 

    // Save copy for CPU
    memcpy(omp_out.all_coords, h_data.all_coords, bytes_coords);   
    memcpy(omp_out.all_velocities, h_data.all_velocities, bytes_vels);

    // Check debug flag
    char *d = getenv("DEBUG");
    if (d) debug = atoi(d);

    // Run CPU (if debug)
    if (debug)
        run_omp(num_systems, nIters, bodies_per_system, dt);
    
    // Run GPU
    run_device_multi_gpu(num_systems, bodies_per_system, nIters, dt);

    // Compare results
    if (debug)
        compare_results(total_bodies, bodies_per_system);

    // Cleanup
    cudaFreeHost(h_data.all_coords);
    cudaFreeHost(h_data.all_velocities);   
    free(omp_out.all_coords);
    free(omp_out.all_velocities);
    cudaFreeHost(h_device_out.all_coords);
    
    cudaDeviceReset();
    return 0;
}