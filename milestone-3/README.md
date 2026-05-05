# Milestone 3: Random Forest with GPU-Accelerated Training

## High-Level Summary
- Extended Milestone 2 single-tree training to a forest of 10 trees using bootstrap sampling.
- Each tree is trained independently using the same GPU split-evaluation kernels from M2.
- Implemented three training variants to measure parallelism effects:
  - **Sequential**: baseline CPU-side control, one tree at a time
  - **Mutex**: CPU threads with shared GPU access via mutex lock
  - **Streams**: experimental CUDA streams scaffold (currently serialized)
- Inference (prediction) fully parallelized across samples and trees using CPU threads.
- Compact tree representation for memory-efficient prediction on large datasets.

## What We Implemented
- Executables:
  - `milestone3_complete.cu`: GPU-accelerated forest with all three variants
  - `milestone3_cpu_randomforest.cpp`: CPU-only baseline for comparison
- Forest training loop with bootstrap resampling (different seed per tree).
- Tree collection and compact serialization for fast inference.
- Parallel inference benchmarking (sequential vs parallel-samples vs parallel-trees).
- CSV output for training times, inference throughput, and scalability runs.

## Benchmarks and Artifacts
- Ran all three variants across 5 datasets (Iris, Shuttle, Letter, Skin_NonSkin, Synthetic_1M).
- Produced CSV outputs:
  - `benchmark_forest_train_m3.csv`: training times and accuracy for all variants
  - `benchmark_throughput_m3.csv`: inference throughput metrics
  - `benchmark_speedup_vs_trees_m3.csv`: speedup vs increasing tree count
  - `benchmark_scalability_m3.csv`: scalability across dataset sizes
  - `benchmark_forest_train_m3_cpu.csv`: CPU-only baseline results
- Generated runtime, accuracy, and speedup plots via Jupyter notebook analysis.

## Outcome (Milestone 3)
- Demonstrated forest training with GPU acceleration and three parallelism strategies.
- Measured both training times and inference throughput across variants.
- Validated accuracy against CPU baseline.
- Completed Milestone 3 objective: measurable GPU-accelerated random forest with empirical variant comparison.

