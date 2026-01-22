# N-body Simulation (OpenMP + CUDA) — GPU Performance Engineering

Parallelization and optimization of an N-body gravitational simulation (O(N²) interactions) on CPU (OpenMP) and GPU (CUDA), with a focus on memory efficiency and overlapping transfers/compute.

## Key results (relative to CPU OpenMP baseline)
- **CPU OpenMP baseline:** 5.7137s (7.52 GInter/s)
- **Best single-node result (2× Tesla K80):** **0.1016s, 56.25× speedup**, **423.34 GInter/s**
- Multi-GPU + streams + SoA achieved the highest throughput in the project.

## Optimizations implemented
- **Data layout redesign:** AoS → split SoA (coords/velocities), later full SoA (x/y/z/vx/vy/vz arrays)
- **Shared memory tiling:** reduced global memory traffic (best TILE_SIZE = 512 in experiments)
- **Loop unrolling:** `#pragma unroll 8` to increase ILP in the inner interaction loop
- **CUDA streams:** concurrent execution across independent systems (galaxies)
- **Overlap transfers/compute:** pinned host memory + `cudaMemcpyAsync` to hide H2D/D2H behind kernels
- **Fast math & caching:** `__ldg`, `__restrict__`, `rsqrtf`, `fmaf`, robust tiling bounds
- **Multiple GPUs:** partitioned systems across GPUs with per-device streams and synchronization

## Correctness
Validated GPU output against CPU OpenMP with absolute error < **1e-3** on selected body positions (per report).

## Build
Requires CUDA toolkit (`nvcc`):
```bash
make -C src
