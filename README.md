## Project summary
Performance-engineered an O(N²) N-body gravitational simulation on CPU and GPU. Starting from a sequential reference implementation, I introduced coarse-grained CPU parallelism with **OpenMP** across independent systems (galaxies), then offloaded the computational hotspot to **CUDA** and iteratively optimized the GPU implementation through memory-layout redesign, shared-memory tiling and asynchronous execution. The work emphasizes profiling-driven optimization, memory efficiency and end-to-end throughput.

## Highlights
- **CPU parallelization (OpenMP):** parallelized at the *system (galaxy)* level to exploit independence with minimal synchronization.
- **CUDA offload:** mapped one galaxy per CUDA block and used a strided body assignment to cover all bodies, enforced per-timestep correctness with `__syncthreads()`.
- **Memory layout optimizations:** evolved from **AoS** to split **SoA** (coords/velocities) and ultimately full **SoA** (`x[]/y[]/z[]/vx[]/vy[]/vz[]`) to improve coalescing and reduce wasted bandwidth.
- **Shared-memory tiling:** staged coordinate tiles in shared memory to reduce redundant global loads in the inner interaction loop (best tile size found experimentally).
- **Instruction-level optimizations:** applied manual unrolling and fast-math primitives (`rsqrtf`, `fmaf`) where the error tolerance permitted.
- **Concurrency & overlap:** used multiple **CUDA streams**, **pinned host memory** and `cudaMemcpyAsync` to overlap H2D/D2H transfers with kernel execution.
- **Multi-GPU scaling:** partitioned systems across **two GPUs** with per-device streams and synchronization, improving throughput substantially.

## Performance (relative to CPU OpenMP baseline)
Best result on **csl-venus server** (2× Intel Xeon E5-2695 v3, 128 GiB RAM, 2× NVIDIA Tesla K80):
- **CPU OpenMP:** 5.7137 s (7.5175 GInter/s)
- **Final (Multi-GPU + SoA):** **0.1016 s**, **56.25× speedup** compared to OpenMP version, **423.3374 GInter/s**

See `docs/report.pdf` for methodology, plots and the full optimization breakdown.
