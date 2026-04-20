# Milestone 2: Parallel Tree Construction with GPU-Accelerated Split Finding

## High-Level Summary
- Built a hybrid CPU-GPU decision tree training pipeline.
- Kept tree structure and control flow on CPU.
- Offloaded split evaluation to GPU for parallel speedup.
- Implemented level-wise training so all nodes at the same depth are processed together.

## What We Implemented
- One-file integrated implementation for Milestone 2 in `milestone2_code.cu`.
- CPU-side level-wise orchestration (frontier expansion and stopping checks).
- GPU histogram-based split evaluation using CUDA kernels.
- Batched node processing per depth level for better parallel utilization.
- One-time dataset transfer strategy: keep binned features and labels resident on GPU.
- Reusable GPU scratch buffers to reduce repeated allocation overhead.
- Timing breakdown instrumentation:
  - data preparation (CPU)
  - split evaluation (GPU)
  - split application (CPU)

## Benchmark and Artifacts
- Ran Milestone 2 benchmarking across selected datasets from Milestone 1.
- Produced CSV outputs for:
  - per-dataset training and accuracy metrics
  - scalability runs
  - CPU vs GPU comparison
- Generated runtime, speedup, scalability, and accuracy plots.

## Outcome (Milestone 2)
- Achieved clear training-time speedups.
- Maintained strong accuracy overall, with expected trade-offs from histogram-based approximation.
- Completed the Milestone 2 objective: measurable hybrid CPU-GPU acceleration with reportable metrics.
