# Milestone 1 Results Summary (5 Datasets)

This file summarizes outputs from:
- `seq approach results/`
- `sklearn results/`

Datasets used (smallest to largest):
- Iris
- Shuttle
- LetterRecognition
- Skin_NonSkin
- Synthetic_dataset_1000k_200f

## 1. CSV Columns Explained

### Sequential main metrics (`seq approach results/benchmark_metrics.csv`)
- `dataset`: Dataset name.
- `n_samples`: Total rows used.
- `n_features`: Number of input features.
- `n_classes`: Number of target classes.
- `train_samples`: Rows in training split.
- `test_samples`: Rows in test split.
- `max_depth`: Fixed tree depth used in this run.
- `min_samples_split`: Minimum samples required to split a node.
- `bin_count`: Histogram bins used by sequential split finder.
- `train_time_sec`: Total training time.
- `split_eval_sec`: Time spent specifically in split evaluation.
- `tree_overhead_sec`: Remaining training time (`train_time_sec - split_eval_sec`).
- `predict_time_sec`: Total prediction time on test set.
- `infer_time_per_sample_us`: Per-sample inference latency in microseconds.
- `train_accuracy`: Accuracy on training split.
- `test_accuracy`: Accuracy on test split.

### Sequential scalability (`seq approach results/benchmark_scalability.csv`)
- `dataset`: Dataset name.
- `fraction`: Fraction of full dataset used (`0.10, 0.25, 0.50, 0.75, 1.00`).
- `n_samples`: Actual sampled row count for that fraction.
- `train_time_sec`: Training time at this scale.
- `predict_time_sec`: Prediction time at this scale.
- `test_accuracy`: Test accuracy at this scale.

### Sklearn main metrics (`sklearn results/benchmark_metrics_sklearn.csv`)
- Same general meaning as above for shared fields.
- `split_eval_sec` and `tree_overhead_sec` are not exposed by sklearn internals in this file.

### Sklearn scalability (`sklearn results/benchmark_scalability_sklearn.csv`)
- Same meaning as sequential scalability CSV.

## 2. Fraction-Wise Meaning

`fraction` controls how much of each dataset is used for scalability checks:
- `0.10` = 10% of full dataset
- `1.00` = 100% of full dataset

This is used to observe runtime growth and accuracy stability as dataset size increases.

## 3. Key Inferences From Results

### Sequential approach
- Strong accuracy on Iris, Shuttle, Skin_NonSkin, and Synthetic datasets.
- Lower test accuracy on LetterRecognition (multi-class complexity is higher).
- Training time increases strongly with dataset size and feature count.
- The synthetic 1M x 200 dataset dominates runtime cost.
- `split_eval_sec` is substantial, but `tree_overhead_sec` is also large on big data, so both contribute to total cost.

### Sklearn approach
- Very high accuracy overall, with better LetterRecognition accuracy than sequential in these runs.
- Much faster training than sequential on most datasets, especially large datasets.
- Inference time is also low and scales reasonably with data size.

### Overall comparison (sequential vs sklearn)
- Accuracy is similar on easier datasets.
- Sklearn is generally faster in training for the same fixed depth/split settings.
- The runtime gap becomes most visible on the largest/high-dimensional dataset.

## 4. PNG Files: What They Show

### Sequential plots (`seq approach results/`)
- `runtime_comparison.png`: Train/predict/split-eval runtime per dataset.
- `accuracy_comparison.png`: Train vs test accuracy per dataset.
- `scalability_<dataset>.png`: For each dataset, left panel shows runtime vs sample size; right panel shows test accuracy vs sample size.

### Sklearn plots (`sklearn results/`)
- `runtime_comparison_sklearn.png`: Train/predict runtime per dataset for sklearn model.
- `accuracy_comparison_sklearn.png`: Train vs test accuracy per dataset for sklearn model.
- `scalability_sklearn_<dataset>.png`: Runtime and test-accuracy trends versus sample size for sklearn.

## 5. Practical Conclusion

- Keeping fixed hyperparameters across all 5 datasets is good for fair benchmarking.
- Sequential implementation is correct and scalable, runtime rises sharply on very large/high-dimensional data.
- Sklearn provides a strong baseline with faster runtime and competitive or better accuracy on harder datasets.
