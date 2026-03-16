**A performance-engineered O(N²) N-body gravitational simulation optimized for CPU and multi-GPU architectures.**

This project started as a sequential reference implementation and was iteratively optimized using OpenMP for the CPU and CUDA for the GPU. The focus is on profiling driven optimization, memory efficiency and maximizing end-to-end throughput.

> **Status:** Completed. Achieved a **56.25× speedup** over the OpenMP baseline using multi-GPU execution and memory layout optimizations.

---

##  What It Does

This simulation calculates the gravitational forces and updates the positions/velocities of bodies across multiple independent systems (galaxies). By isolating systems, the workload avoids heavy synchronization, allowing for massive parallelization across CPU threads and GPU blocks.

See `docs/report.pdf` for the complete methodology, optimization breakdown and performance plots.

---

##  Project Structure

```text
nbody-openmp-cuda/
├── src/
│   ├── Makefile            # Build configuration
│   ├── nbody.cu            # Main CUDA implementation
│   └── ...                 # Additional source files
├── docs/
│   └── report.pdf          # Detailed methodology and profiling plots
└── README.md
