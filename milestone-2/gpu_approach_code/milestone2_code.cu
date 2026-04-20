// ============================================================================
// MILESTONE 2 — FINAL INTEGRATED CPU-GPU DECISION TREE TRAINER
// ============================================================================
// Team mapping for the final submission:
//   Set 1 (ZUHAA): level-wise tree construction and CPU orchestration
//   Set 2 (FATIMA): GPU histogram split evaluation
//   Set 3 (YAMAN): GPU data management / resident buffers / batch packing
//
// Workflow summary (matches milestone2_workflow.pdf):
//   Step 0) CPU loads dataset; Yaman pre-bins X→X_bin and copies X_bin+y to GPU ONCE.
//   Step 1) Zuhaa creates root node with all training sample indices.
//   Step 2) Each level: Zuhaa collects active nodes; Yaman packs their indices for GPU.
//   Step 3) Fatima's GPU kernels build histograms + compute Gini + return best split per node.
//   Step 4) CPU applies returned splits and creates left/right children.
//   Step 5) Repeat Steps 2-4 for next depth level until stopping criteria met.
//
// Speedup comparison: use split_eval_sec from this output vs M1 CSV on same dataset.
// This file is intentionally self-contained. Compile with nvcc on Colab/T4.
// ============================================================================

#include <algorithm>
#include <cassert>
#include <cctype>
#include <chrono>
#include <cmath>
#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <limits>
#include <map>
#include <numeric>
#include <random>
#include <sstream>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

#ifdef __CUDACC__
#include <cuda_runtime.h>
#endif

// ============================================================================
// FARAZ'S PART — DATA STRUCTURES + CSV LOADING
// ============================================================================

struct Dataset {
    std::vector<std::vector<float>> features;
    std::vector<int>                labels;
    std::vector<std::string>        feature_names;
};

struct RunConfig {
    int max_depth = 8;
    int min_samples_split = 2;
    int bin_count = 8;
    float train_ratio = 0.8f;
    int seed = 42;
};

struct DatasetSpec {
    std::string name;
    std::string file;
    bool has_header = true;
};

// GPU returns one of these per node per level (Step 3 output → Step 4 input)
struct SplitResult {
    int    best_feature   = -1;
    int    best_bin_index = -1;
    double best_threshold = 0.0;
    double best_gini      = std::numeric_limits<double>::infinity();
    bool   found          = false;
};

// All timing fields are real measured values — no placeholders.
// speedup vs M1 is computed externally using split_eval_sec from both CSVs.
struct BenchmarkResult {
    std::string dataset_name;
    int n_samples    = 0;
    int n_features   = 0;
    int n_classes    = 0;
    int train_samples = 0;
    int test_samples  = 0;
    double total_train_sec  = 0.0; // wall time for full build_tree_levelwise call
    double data_prep_sec    = 0.0; // time packing node index batches (CPU, Yaman)
    double split_eval_sec   = 0.0; // time inside GPU kernels (Fatima)
    double apply_split_sec  = 0.0; // time CPU applies splits and creates children (Zuhaa)
    double predict_sec      = 0.0; // inference time on test set
    double train_acc = 0.0;
    double test_acc  = 0.0;
};

struct ScalabilityRow {
    std::string dataset_name;
    double fraction        = 1.0;
    int    n_samples       = 0;
    double total_train_sec = 0.0;
    double predict_sec     = 0.0;
    double test_acc        = 0.0;
};

// Per-level timing breakdown collected inside build_tree_levelwise
struct LevelwiseBuildStats {
    double total_time_ms = 0.0;
    double data_prep_ms  = 0.0; // Step 2: packing indices for GPU
    double split_eval_ms = 0.0; // Step 3: GPU kernel execution time
    double apply_ms      = 0.0; // Step 4: CPU applying splits
    int levels_processed     = 0;
    int total_nodes_created  = 0;
};

// Tree node — CPU side only; GPU never touches Node objects
struct Node {
    std::vector<int> sample_indices; // indices into Dataset for samples at this node
    int depth           = 0;
    int feature_index   = -1;
    float threshold     = 0.0f;
    Node* left          = nullptr;
    Node* right         = nullptr;
    bool is_leaf        = false;
    int predicted_class = -1;

    explicit Node(int d = 0) : depth(d) {}
};

static std::string resolve_dataset_path(
    const std::string& file_name,
    const std::vector<std::string>& search_roots)
{
    namespace fs = std::filesystem;
    if (fs::exists(file_name)) return file_name;
    for (const auto& root : search_roots) {
        fs::path candidate = fs::path(root) / file_name;
        if (fs::exists(candidate)) return candidate.string();
    }
    return file_name;
}

// Faraz: CSV loader — auto-detects label column (first if non-numeric, else last)
static Dataset load_csv(const std::string& filename, bool has_header = true) {
    Dataset data;
    std::ifstream file(filename);
    if (!file.is_open()) {
        throw std::runtime_error("Cannot open dataset: " + filename);
    }

    auto trim = [](std::string s) {
        auto not_space = [](unsigned char ch) { return !std::isspace(ch); };
        s.erase(s.begin(), std::find_if(s.begin(), s.end(), not_space));
        s.erase(std::find_if(s.rbegin(), s.rend(), not_space).base(), s.end());
        return s;
    };

    auto is_numeric = [](const std::string& s) {
        if (s.empty()) return false;
        char* end_ptr = nullptr;
        std::strtod(s.c_str(), &end_ptr);
        return end_ptr != s.c_str() && *end_ptr == '\0';
    };

    std::string line;
    int line_number = 0;
    std::map<std::string, int> label_encoder;
    int next_label_id = 0;
    std::vector<std::string> header_tokens;

    while (std::getline(file, line)) {
        ++line_number;
        if (line.empty()) continue;

        if (has_header && line_number == 1) {
            std::stringstream ss(line);
            std::string item;
            while (std::getline(ss, item, ',')) header_tokens.push_back(trim(item));
            continue;
        }

        std::stringstream ss(line);
        std::string token;
        std::vector<std::string> tokens;
        while (std::getline(ss, token, ',')) tokens.push_back(trim(token));

        if (tokens.size() < 2) continue;

        const bool first_is_label = !is_numeric(tokens.front());
        const int label_idx = first_is_label ? 0 : static_cast<int>(tokens.size()) - 1;

        if (data.feature_names.empty() && !header_tokens.empty() && header_tokens.size() == tokens.size()) {
            data.feature_names.reserve(tokens.size() - 1);
            for (int i = 0; i < static_cast<int>(header_tokens.size()); ++i) {
                if (i == label_idx) continue;
                data.feature_names.push_back(header_tokens[i]);
            }
        }

        std::vector<float> row;
        row.reserve(tokens.size() - 1);
        for (int i = 0; i < static_cast<int>(tokens.size()); ++i) {
            if (i == label_idx) continue;
            if (!is_numeric(tokens[i])) { row.clear(); break; }
            row.push_back(static_cast<float>(std::stod(tokens[i])));
        }
        if (row.empty()) continue;

        // String labels get integer-encoded; numeric labels are rounded
        int label_id = 0;
        if (is_numeric(tokens[label_idx])) {
            label_id = static_cast<int>(std::lround(std::stod(tokens[label_idx])));
        } else {
            auto it = label_encoder.find(tokens[label_idx]);
            if (it == label_encoder.end()) {
                label_id = next_label_id;
                label_encoder[tokens[label_idx]] = next_label_id++;
            } else {
                label_id = it->second;
            }
        }

        data.features.push_back(std::move(row));
        data.labels.push_back(label_id);
    }

    return data;
}

static int count_unique_classes(const std::vector<int>& labels) {
    std::map<int, int> counts;
    for (int y : labels) counts[y]++;
    return static_cast<int>(counts.size());
}

// Faraz: reproducible train/test split using fixed seed
static std::pair<std::vector<int>, std::vector<int>>
train_test_split(const Dataset& data, float train_ratio = 0.8f, int seed = 42) {
    const int n = static_cast<int>(data.features.size());
    std::vector<int> idx(n);
    std::iota(idx.begin(), idx.end(), 0);
    std::shuffle(idx.begin(), idx.end(), std::default_random_engine(seed));
    const int train_size = static_cast<int>(n * train_ratio);
    return {
        std::vector<int>(idx.begin(), idx.begin() + train_size),
        std::vector<int>(idx.begin() + train_size, idx.end())
    };
}

// Zuhaa (Step 1): create root node holding all training sample indices
static Node* create_root_node(const std::vector<int>& train_indices) {
    Node* root = new Node(0);
    root->sample_indices = train_indices;
    return root;
}

static const std::vector<int>& get_samples_for_node(Node* node) {
    return node->sample_indices;
}

// Zuhaa (Step 4): attach left/right children after GPU returns the split
static std::pair<Node*, Node*>
split_node(
    Node* parent,
    const std::vector<int>& left_indices,
    const std::vector<int>& right_indices)
{
    Node* left  = new Node(parent->depth + 1);
    Node* right = new Node(parent->depth + 1);
    left->sample_indices  = left_indices;
    right->sample_indices = right_indices;
    parent->left  = left;
    parent->right = right;
    return {left, right};
}

// ============================================================================
// YAMAN'S PART — GPU DATA MANAGEMENT / BATCH ORCHESTRATION
// ============================================================================
// Workflow Step 0: allocate GPU memory once, pre-bin X→X_bin, copy to GPU.
// Workflow Step 2: pack per-node index batches into contiguous GPU buffer.
// Workflow Step 5 (cleanup): release all GPU memory after training finishes.
// ============================================================================

struct GPUDataManager {
    bool initialized = false;
    int bin_count = 8;
    explicit GPUDataManager(int bins = 8) : bin_count(bins) {}
    void init(const Dataset& data);
    void release();
    ~GPUDataManager();
};

// ============================================================================
// FATIMA'S PART — GPU SPLIT EVALUATION (CUDA)
// ============================================================================
// Workflow Step 3: GPU compute engine.
//   Kernel 1 (build_histograms_batched_kernel): for each node × feature,
//     scatter samples into bin×class histogram using atomicAdd.
//   Kernel 2 (evaluate_splits_batched_kernel): for each node × feature,
//     prefix-scan histogram to compute weighted Gini at each split boundary,
//     return best bin index and Gini score per node×feature slot.
//   CPU reduction (gpu_find_best_splits_batch): pick best feature per node
//     from GPU results and convert bin index → actual threshold value.
// ============================================================================

#ifdef __CUDACC__

constexpr int kMaxClassesPerNode = 128;

#define CUDA_CHECK(call) \
    do { \
        cudaError_t err__ = (call); \
        if (err__ != cudaSuccess) throw std::runtime_error(cudaGetErrorString(err__)); \
    } while (0)

// Yaman: global GPU-resident state — allocated once in GPUDataManager::init,
// freed once in GPUDataManager::release. Reusable buffers grow via ensure_device_capacity.
static int g_n_samples  = 0;
static int g_n_features = 0;
static int g_n_classes  = 0;
static int g_bin_count  = 0;
static std::vector<int>   g_h_x_bin;       // [n_samples * n_features] pre-binned on CPU
static std::vector<int>   g_h_y;           // [n_samples] labels (host copy for reference)
static std::vector<float> g_h_min;         // [n_features] per-feature minimum value
static std::vector<float> g_h_bin_width;   // [n_features] per-feature bin width
static int*   d_x_bin     = nullptr;       // GPU: pre-binned features (Step 0, permanent)
static int*   d_y         = nullptr;       // GPU: labels (Step 0, permanent)
static int*   d_indices   = nullptr;       // GPU: packed node sample indices (Step 2, reused)
static int*   d_offsets   = nullptr;       // GPU: per-node start offsets (Step 2, reused)
static int*   d_hist      = nullptr;       // GPU: histogram buffer (Step 3, reused)
static float* d_best_gini = nullptr;       // GPU: best Gini per node×feature (Step 3 output)
static int*   d_best_bin  = nullptr;       // GPU: best bin index per node×feature (Step 3 output)
static std::size_t d_indices_capacity   = 0;
static std::size_t d_offsets_capacity   = 0;
static std::size_t d_hist_capacity      = 0;
static std::size_t d_best_gini_capacity = 0;
static std::size_t d_best_bin_capacity  = 0;

// Yaman: grow GPU buffer only when needed — avoids cudaMalloc on every level call
template <typename T>
static void ensure_device_capacity(T*& ptr, std::size_t& capacity, std::size_t required) {
    if (required == 0) return;
    if (ptr != nullptr && capacity >= required) return;
    if (ptr != nullptr) {
        CUDA_CHECK(cudaFree(ptr));
        ptr = nullptr;
        capacity = 0;
    }
    CUDA_CHECK(cudaMalloc(&ptr, sizeof(T) * required));
    capacity = required;
}

// Fatima: device helper — Gini impurity from class count array
__device__ float device_gini_from_counts(const int* counts, int nc, int total) {
    if (total == 0) return 0.0f;
    float sum_sq = 0.0f;
    for (int c = 0; c < nc; ++c) {
        const float p = static_cast<float>(counts[c]) / static_cast<float>(total);
        sum_sq += p * p;
    }
    return 1.0f - sum_sq;
}

// Fatima (Step 3 — Kernel 1): histogram construction.
// Grid: x=sample chunks, y=features, z=nodes. Each thread processes one sample.
// atomicAdd safely accumulates class counts into hist[node][feature][bin][class].
__global__ void build_histograms_batched_kernel(
    const int* x_bin,
    const int* y,
    const int* packed_indices,
    const int* offsets,
    int n_nodes,
    int n_features,
    int bins,
    int n_classes,
    int* hist)
{
    const int local_i   = blockIdx.x * blockDim.x + threadIdx.x;
    const int feature_id = blockIdx.y;
    const int node_id   = blockIdx.z;

    if (node_id >= n_nodes || feature_id >= n_features) return;

    const int start = offsets[node_id];
    const int end   = offsets[node_id + 1];
    const int len   = end - start;
    if (local_i >= len) return;

    const int sample_idx = packed_indices[start + local_i];
    const int bin_id     = x_bin[sample_idx * n_features + feature_id];
    const int cls        = y[sample_idx];
    if (bin_id < 0 || bin_id >= bins || cls < 0 || cls >= n_classes) return;

    // Scatter: increment hist[node_id][feature_id][bin_id][cls]
    const int idx = (((node_id * n_features + feature_id) * bins + bin_id) * n_classes + cls);
    atomicAdd(&hist[idx], 1);
}

// Fatima (Step 3 — Kernel 2): Gini evaluation via prefix scan over bin splits.
// Each thread owns one (node, feature) pair and scans all split boundaries.
// Moves counts left→right incrementally; computes weighted Gini at each split.
__global__ void evaluate_splits_batched_kernel(
    const int* hist,
    const int* offsets,
    int n_nodes,
    int n_features,
    int bins,
    int n_classes,
    float* out_best_gini,
    int* out_best_bin)
{
    const int feature_id = blockIdx.x * blockDim.x + threadIdx.x;
    const int node_id    = blockIdx.y;
    if (node_id >= n_nodes || feature_id >= n_features) return;

    const int node_count = offsets[node_id + 1] - offsets[node_id];
    if (node_count <= 1 || n_classes > kMaxClassesPerNode) {
        out_best_gini[node_id * n_features + feature_id] = 1e30f;
        out_best_bin [node_id * n_features + feature_id] = -1;
        return;
    }

    // Initialize: all samples start on the right side
    int left[kMaxClassesPerNode];
    int right[kMaxClassesPerNode];
    for (int c = 0; c < n_classes; ++c) { left[c] = 0; right[c] = 0; }

    int total = 0;
    for (int b = 0; b < bins; ++b) {
        for (int c = 0; c < n_classes; ++c) {
            const int v = hist[(((node_id * n_features + feature_id) * bins + b) * n_classes + c)];
            right[c] += v;
            total    += v;
        }
    }

    float best_g  = 1e30f;
    int   best_b  = -1;
    int   left_n  = 0;
    int   right_n = total;

    // Prefix scan: try each bin boundary as the split point
    for (int sb = 0; sb < bins - 1; ++sb) {
        for (int c = 0; c < n_classes; ++c) {
            const int v = hist[(((node_id * n_features + feature_id) * bins + sb) * n_classes + c)];
            left[c]  += v;
            right[c] -= v;
            left_n   += v;
            right_n  -= v;
        }

        if (left_n == 0 || right_n == 0) continue;

        // Weighted Gini = (n_left/n_total)*Gini_left + (n_right/n_total)*Gini_right
        const float g_left  = device_gini_from_counts(left,  n_classes, left_n);
        const float g_right = device_gini_from_counts(right, n_classes, right_n);
        const float wg =
            (static_cast<float>(left_n)  / static_cast<float>(total)) * g_left +
            (static_cast<float>(right_n) / static_cast<float>(total)) * g_right;

        if (wg < best_g) { best_g = wg; best_b = sb; }
    }

    out_best_gini[node_id * n_features + feature_id] = best_g;
    out_best_bin [node_id * n_features + feature_id] = best_b;
}

static int find_num_classes_host(const std::vector<int>& y) {
    int mx = -1;
    for (int v : y) {
        if (v < 0) throw std::invalid_argument("Negative labels are not supported.");
        if (v > mx) mx = v;
    }
    return mx + 1;
}

// Yaman (Step 0): one-time setup — pre-bin X→X_bin on CPU, copy X_bin+y to GPU
void GPUDataManager::init(const Dataset& data) {
    if (initialized) return;
    if (data.features.empty() || data.labels.empty())
        throw std::invalid_argument("Dataset is empty.");
    if (data.features.size() != data.labels.size())
        throw std::invalid_argument("features/labels size mismatch.");

    g_n_samples  = static_cast<int>(data.features.size());
    g_n_features = static_cast<int>(data.features[0].size());
    g_n_classes  = find_num_classes_host(data.labels);
    g_bin_count  = bin_count;
    g_h_y        = data.labels;
    g_h_x_bin.assign(g_n_samples * g_n_features, 0);
    g_h_min.assign(g_n_features, 0.0f);
    g_h_bin_width.assign(g_n_features, 0.0f);

    // Pre-binning (Step 0): convert each float feature value to integer bin index
    for (int f = 0; f < g_n_features; ++f) {
        float mn = data.features[0][f];
        float mx = data.features[0][f];
        for (int i = 1; i < g_n_samples; ++i) {
            mn = std::min(mn, data.features[i][f]);
            mx = std::max(mx, data.features[i][f]);
        }
        g_h_min[f]       = mn;
        g_h_bin_width[f] = (mx > mn) ? ((mx - mn) / static_cast<float>(bin_count)) : 0.0f;
        for (int i = 0; i < g_n_samples; ++i) {
            int b = 0;
            if (g_h_bin_width[f] > 0.0f) {
                const float raw = (data.features[i][f] - mn) / g_h_bin_width[f];
                b = static_cast<int>(std::floor(raw));
                b = std::clamp(b, 0, bin_count - 1);
            }
            g_h_x_bin[i * g_n_features + f] = b;
        }
    }

    // One-time transfer: X_bin and y go to GPU and stay there for all levels
    CUDA_CHECK(cudaMalloc(&d_x_bin, sizeof(int) * g_h_x_bin.size()));
    CUDA_CHECK(cudaMalloc(&d_y,     sizeof(int) * g_h_y.size()));
    CUDA_CHECK(cudaMemcpy(d_x_bin, g_h_x_bin.data(), sizeof(int) * g_h_x_bin.size(), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_y,     g_h_y.data(),     sizeof(int) * g_h_y.size(),     cudaMemcpyHostToDevice));
    initialized = true;
}

// Yaman (Step 5 / cleanup): free all GPU memory after training finishes
void GPUDataManager::release() {
    if (d_x_bin)     CUDA_CHECK(cudaFree(d_x_bin));
    if (d_y)         CUDA_CHECK(cudaFree(d_y));
    if (d_indices)   CUDA_CHECK(cudaFree(d_indices));
    if (d_offsets)   CUDA_CHECK(cudaFree(d_offsets));
    if (d_hist)      CUDA_CHECK(cudaFree(d_hist));
    if (d_best_gini) CUDA_CHECK(cudaFree(d_best_gini));
    if (d_best_bin)  CUDA_CHECK(cudaFree(d_best_bin));

    d_x_bin = nullptr; d_y = nullptr; d_indices = nullptr;
    d_offsets = nullptr; d_hist = nullptr;
    d_best_gini = nullptr; d_best_bin = nullptr;
    d_indices_capacity = 0; d_offsets_capacity = 0;
    d_hist_capacity = 0; d_best_gini_capacity = 0; d_best_bin_capacity = 0;

    g_h_x_bin.clear(); g_h_y.clear(); g_h_min.clear(); g_h_bin_width.clear();
    g_n_samples = 0; g_n_features = 0; g_n_classes = 0; g_bin_count = 0;
    initialized = false;
}

GPUDataManager::~GPUDataManager() { release(); }

// Yaman + Fatima (Steps 2+3): pack node indices → GPU, launch kernels, unpack results.
// Called once per depth level with ALL active nodes batched together.
static std::vector<SplitResult> gpu_find_best_splits_batch(
    const std::vector<std::vector<int>>& node_index_batches,
    int bin_count)
{
    if (g_n_features == 0)
        throw std::runtime_error("GPU data buffers are not initialized.");
    if (bin_count != g_bin_count && g_bin_count != 0)
        throw std::invalid_argument("bin_count mismatch with initialized GPU buffers.");
    if (node_index_batches.empty()) return {};

    const int n_nodes = static_cast<int>(node_index_batches.size());
    std::vector<SplitResult> out(n_nodes);

    // Yaman (Step 2): build CSR-style offset array and flat packed index array
    std::vector<int> h_offsets(n_nodes + 1, 0);
    int total_indices = 0;
    for (int i = 0; i < n_nodes; ++i) {
        h_offsets[i]  = total_indices;
        total_indices += static_cast<int>(node_index_batches[i].size());
    }
    h_offsets[n_nodes] = total_indices;

    std::vector<int> h_packed(total_indices);
    int p = 0;
    for (const auto& v : node_index_batches)
        for (int idx : v) h_packed[p++] = idx;

    // Yaman (Step 2): transfer packed indices to GPU (reusing buffers, no cudaMalloc per level)
    ensure_device_capacity(d_indices, d_indices_capacity, h_packed.size());
    ensure_device_capacity(d_offsets, d_offsets_capacity, h_offsets.size());
    if (!h_packed.empty()) {
        CUDA_CHECK(cudaMemcpy(d_indices, h_packed.data(), sizeof(int) * h_packed.size(), cudaMemcpyHostToDevice));
    }
    CUDA_CHECK(cudaMemcpy(d_offsets, h_offsets.data(), sizeof(int) * h_offsets.size(), cudaMemcpyHostToDevice));

    // Fatima (Step 3 — Kernel 1): build histogram[node][feature][bin][class]
    const std::size_t hist_size = static_cast<std::size_t>(n_nodes) * g_n_features * bin_count * g_n_classes;
    ensure_device_capacity(d_hist, d_hist_capacity, hist_size);
    CUDA_CHECK(cudaMemset(d_hist, 0, sizeof(int) * hist_size));

    const std::size_t feat_out = static_cast<std::size_t>(n_nodes) * g_n_features;
    ensure_device_capacity(d_best_gini, d_best_gini_capacity, feat_out);
    ensure_device_capacity(d_best_bin,  d_best_bin_capacity,  feat_out);

    const int threads = 256;
    int max_len = 1;
    for (const auto& v : node_index_batches) max_len = std::max(max_len, static_cast<int>(v.size()));
    dim3 grid((max_len + threads - 1) / threads, g_n_features, n_nodes);
    dim3 block(threads, 1, 1);
    build_histograms_batched_kernel<<<grid, block>>>(
        d_x_bin, d_y, d_indices, d_offsets, n_nodes, g_n_features, bin_count, g_n_classes, d_hist);
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaDeviceSynchronize());

    // Fatima (Step 3 — Kernel 2): scan splits, compute weighted Gini, find best bin per node×feature
    const int feat_threads = 128;
    dim3 sgrid((g_n_features + feat_threads - 1) / feat_threads, n_nodes, 1);
    evaluate_splits_batched_kernel<<<sgrid, feat_threads>>>(
        d_hist, d_offsets, n_nodes, g_n_features, bin_count, g_n_classes, d_best_gini, d_best_bin);
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaDeviceSynchronize());

    // Yaman (Step 3 → Step 4): copy small result arrays back to CPU
    std::vector<float> h_best_gini(feat_out, 1e30f);
    std::vector<int>   h_best_bin(feat_out, -1);
    CUDA_CHECK(cudaMemcpy(h_best_gini.data(), d_best_gini, sizeof(float) * feat_out, cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(h_best_bin.data(),  d_best_bin,  sizeof(int)   * feat_out, cudaMemcpyDeviceToHost));

    // CPU reduction: pick best feature per node; convert bin index → actual threshold
    for (int node = 0; node < n_nodes; ++node) {
        SplitResult best;
        for (int f = 0; f < g_n_features; ++f) {
            const int idx = node * g_n_features + f;
            if (h_best_bin[idx] < 0) continue;
            const double g = static_cast<double>(h_best_gini[idx]);
            if (g < best.best_gini) {
                best.best_gini      = g;
                best.best_feature   = f;
                best.best_bin_index = h_best_bin[idx];
                // threshold = right edge of the best bin
                best.best_threshold = static_cast<double>(g_h_min[f]) +
                                      static_cast<double>(h_best_bin[idx] + 1) *
                                      static_cast<double>(g_h_bin_width[f]);
                best.found = true;
            }
        }
        out[node] = best;
    }

    return out;
}

#else

// Non-CUDA stub — fails loudly so it is obvious if compiled without nvcc
void GPUDataManager::init(const Dataset&) {
    throw std::runtime_error("Compile with nvcc on Colab/T4 to use the GPU split backend.");
}
void GPUDataManager::release() {}
GPUDataManager::~GPUDataManager() = default;
static std::vector<SplitResult> gpu_find_best_splits_batch(
    const std::vector<std::vector<int>>&, int)
{
    throw std::runtime_error("Compile with nvcc on Colab/T4 to use the GPU split backend.");
}

#endif

// ============================================================================
// ZUHAA'S PART — LEVEL-WISE TREE CONSTRUCTION, PREDICTION, EVALUATION
// ============================================================================

// CPU-side Gini helper used for display/debugging only (not in hot path)
static double node_gini(const std::vector<int>& indices, const std::vector<int>& labels) {
    if (indices.empty()) return 0.0;
    std::map<int, int> counts;
    for (int idx : indices) counts[labels[idx]]++;
    const int total = static_cast<int>(indices.size());
    double sum_sq = 0.0;
    for (const auto& [cls, cnt] : counts) {
        const double p = static_cast<double>(cnt) / total;
        sum_sq += p * p;
    }
    return 1.0 - sum_sq;
}

// Zuhaa: majority vote label assigned to every node (leaf prediction + pre-pruning fallback)
static int majority_class(const std::vector<int>& indices, const std::vector<int>& labels) {
    std::map<int, int> counts;
    for (int idx : indices) counts[labels[idx]]++;
    return std::max_element(counts.begin(), counts.end(),
        [](const auto& a, const auto& b) { return a.second < b.second; })->first;
}

// Zuhaa: stopping condition check (depth, min_samples, purity)
static bool is_pure(const std::vector<int>& indices, const std::vector<int>& labels) {
    if (indices.empty()) return true;
    const int first = labels[indices[0]];
    for (int idx : indices)
        if (labels[idx] != first) return false;
    return true;
}

static bool should_split_node(const Node* node, const Dataset& data, int max_depth, int min_samples_split) {
    const auto& indices = node->sample_indices;
    if (indices.empty()) return false;
    if (node->depth >= max_depth) return false;
    if (static_cast<int>(indices.size()) < min_samples_split) return false;
    if (is_pure(indices, data.labels)) return false;
    return true;
}

// Zuhaa: recursive tree traversal for inference (CPU only)
static int predict(const Node* node, const std::vector<float>& sample) {
    if (node->is_leaf || node->left == nullptr || node->right == nullptr)
        return node->predicted_class;
    if (sample[node->feature_index] <= node->threshold)
        return predict(node->left, sample);
    return predict(node->right, sample);
}

static std::vector<int> predict_batch(
    const Node* root,
    const Dataset& data,
    const std::vector<int>& indices)
{
    std::vector<int> preds;
    preds.reserve(indices.size());
    for (int idx : indices) preds.push_back(predict(root, data.features[idx]));
    return preds;
}

static double compute_accuracy(
    const std::vector<int>& predictions,
    const std::vector<int>& true_labels)
{
    if (predictions.size() != true_labels.size() || predictions.empty())
        throw std::invalid_argument("Size mismatch or empty vectors.");
    int correct = 0;
    for (int i = 0; i < static_cast<int>(predictions.size()); ++i)
        if (predictions[i] == true_labels[i]) ++correct;
    return static_cast<double>(correct) / static_cast<double>(predictions.size());
}

// Used by sanity tests to verify tree was built with at least one internal node
struct TreeStats { int leaves = 0; int internals = 0; int max_depth = 0; };

static void collect_stats(const Node* node, TreeStats& s) {
    if (!node) return;
    if (node->depth > s.max_depth) s.max_depth = node->depth;
    if (node->is_leaf || (!node->left && !node->right)) ++s.leaves;
    else {
        ++s.internals;
        collect_stats(node->left, s);
        collect_stats(node->right, s);
    }
}

// Zuhaa: optional tree printer for debugging (not called during benchmarking)
static void print_tree(
    const Node* node,
    const Dataset& data,
    const std::vector<std::string>& feat_names,
    const std::string& prefix = "",
    bool is_left = true)
{
    if (!node) return;

    const std::string connector  = prefix.empty() ? "" : (is_left ? "├── " : "└── ");
    const std::string new_prefix = prefix + (prefix.empty() ? "" : (is_left ? "│   " : "    "));
    const int n    = static_cast<int>(node->sample_indices.size());
    const double g = node_gini(node->sample_indices, data.labels);

    if (node->is_leaf || (!node->left && !node->right)) {
        std::cout << prefix << connector
                  << "[LEAF] class=" << node->predicted_class
                  << "  samples=" << n
                  << "  gini=" << std::fixed << std::setprecision(4) << g << "\n";
        return;
    }

    const std::string fname = (node->feature_index >= 0 && node->feature_index < static_cast<int>(feat_names.size()))
        ? feat_names[node->feature_index]
        : ("f" + std::to_string(node->feature_index));

    std::cout << prefix << connector
              << "[depth " << node->depth << "]  "
              << fname << " <= " << std::fixed << std::setprecision(4) << node->threshold
              << "  |  samples=" << n << "  gini=" << g << "\n";
    print_tree(node->left,  data, feat_names, new_prefix, true);
    print_tree(node->right, data, feat_names, new_prefix, false);
}

static void delete_tree(Node* node) {
    if (!node) return;
    delete_tree(node->left);
    delete_tree(node->right);
    delete node;
}

// ============================================================================
// ZUHAA (Steps 1-5): main level-wise BFS training loop
// ============================================================================
static Node* build_tree_levelwise(
    const Dataset& data,
    const std::vector<int>& train_indices,
    int max_depth,
    int min_samples_split,
    int bin_count,
    LevelwiseBuildStats& stats)
{
    using clock = std::chrono::high_resolution_clock;
    const auto wall_start = clock::now();

    // Step 1: create root node; Step 0: initialise GPU (pre-bin + one-time transfer)
    Node* root = create_root_node(train_indices);
    stats.total_nodes_created = 1;

    GPUDataManager gpu_manager(bin_count);
    gpu_manager.init(data); // Yaman: Step 0 — pre-bin X→X_bin, copy to GPU

    std::vector<Node*> frontier = { root };

    // Steps 2-5: BFS loop — one iteration per depth level
    while (!frontier.empty()) {

        // Zuhaa: assign majority-class label to every node; filter out nodes that should not split
        std::vector<Node*> active_nodes;
        active_nodes.reserve(frontier.size());
        for (Node* node : frontier) {
            const auto& indices = get_samples_for_node(node);
            node->predicted_class = majority_class(indices, data.labels);
            if (!should_split_node(node, data, max_depth, min_samples_split)) {
                node->is_leaf = true;
                continue;
            }
            active_nodes.push_back(node);
        }
        if (active_nodes.empty()) break;

        // Step 2: collect sample index lists for all active nodes (Yaman packs inside gpu_find_best_splits_batch)
        const auto pack_start = clock::now();
        std::vector<std::vector<int>> node_batches;
        node_batches.reserve(active_nodes.size());
        for (Node* node : active_nodes)
            node_batches.push_back(node->sample_indices);
        stats.data_prep_ms += std::chrono::duration<double, std::milli>(clock::now() - pack_start).count();

        // Step 3: GPU evaluates all nodes in one batched call (Yaman packs + Fatima kernels)
        const auto split_start = clock::now();
        std::vector<SplitResult> results = gpu_find_best_splits_batch(node_batches, bin_count);
        stats.split_eval_ms += std::chrono::duration<double, std::milli>(clock::now() - split_start).count();

        // Step 4: CPU applies returned splits; partition samples into left/right children
        const auto apply_start = clock::now();
        std::vector<Node*> next_frontier;
        next_frontier.reserve(active_nodes.size() * 2);

        for (int i = 0; i < static_cast<int>(active_nodes.size()); ++i) {
            Node* node = active_nodes[i];
            const SplitResult& sr = results[i];

            if (!sr.found) { node->is_leaf = true; continue; }

            std::vector<int> left_idx, right_idx;
            left_idx.reserve(node->sample_indices.size());
            right_idx.reserve(node->sample_indices.size());

            // Partition: feature[sample] <= threshold goes left, else right
            for (int sample : node->sample_indices) {
                if (data.features[sample][sr.best_feature] <= sr.best_threshold)
                    left_idx.push_back(sample);
                else
                    right_idx.push_back(sample);
            }

            if (left_idx.empty() || right_idx.empty()) { node->is_leaf = true; continue; }

            node->feature_index = sr.best_feature;
            node->threshold     = static_cast<float>(sr.best_threshold);
            auto [left_child, right_child] = split_node(node, left_idx, right_idx);
            next_frontier.push_back(left_child);
            next_frontier.push_back(right_child);
            stats.total_nodes_created += 2;
        }

        stats.apply_ms += std::chrono::duration<double, std::milli>(clock::now() - apply_start).count();
        frontier.swap(next_frontier); // Step 5: advance to next depth level
        ++stats.levels_processed;
    }

    stats.total_time_ms = std::chrono::duration<double, std::milli>(clock::now() - wall_start).count();
    // GPUDataManager destructor handles cleanup (RAII).
    return root;
}

// ============================================================================
// BENCHMARK + REPORTING
// ============================================================================

// All timing values written to CSV are real measured values from LevelwiseBuildStats.
// speedup_vs_cpu column is omitted — compute externally from M1 vs M2 split_eval_sec.
static void save_metrics_csv(const std::string& path, const std::vector<BenchmarkResult>& rows) {
    std::ofstream out(path);
    out << "dataset,n_samples,n_features,n_classes,train_samples,test_samples,"
           "total_train_sec,data_prep_sec,split_eval_sec,apply_split_sec,predict_sec,"
           "train_accuracy,test_accuracy\n";
    out << std::fixed << std::setprecision(6);
    for (const auto& r : rows) {
        out << r.dataset_name << ',' << r.n_samples << ',' << r.n_features << ','
            << r.n_classes << ',' << r.train_samples << ',' << r.test_samples << ','
            << r.total_train_sec << ',' << r.data_prep_sec << ',' << r.split_eval_sec << ','
            << r.apply_split_sec << ',' << r.predict_sec << ','
            << r.train_acc << ',' << r.test_acc << '\n';
    }
}

static void save_scalability_csv(const std::string& path, const std::vector<ScalabilityRow>& rows) {
    std::ofstream out(path);
    out << "dataset,fraction,n_samples,total_train_sec,predict_sec,test_accuracy\n";
    out << std::fixed << std::setprecision(6);
    for (const auto& r : rows) {
        out << r.dataset_name << ',' << r.fraction << ',' << r.n_samples << ','
            << r.total_train_sec << ',' << r.predict_sec << ',' << r.test_acc << '\n';
    }
}

static void print_result_summary(const BenchmarkResult& r) {
    std::cout << "\n[Dataset: " << r.dataset_name << " | Mode: GPU (Hybrid CPU-GPU)]\n";
    std::cout << "  samples/features/classes : " << r.n_samples << " / " << r.n_features << " / " << r.n_classes << "\n";
    std::cout << "  train/test samples       : " << r.train_samples << " / " << r.test_samples << "\n";
    std::cout << std::fixed << std::setprecision(6);
    std::cout << "  total_train_sec          : " << r.total_train_sec << "\n";
    std::cout << "  data_prep_sec            : " << r.data_prep_sec << "  (Step 2: index packing)\n";
    std::cout << "  split_eval_sec           : " << r.split_eval_sec << "  (Step 3: GPU kernels — compare vs M1)\n";
    std::cout << "  apply_split_sec          : " << r.apply_split_sec << "  (Step 4: CPU apply splits)\n";
    std::cout << "  predict_sec              : " << r.predict_sec << "\n";
    std::cout << "  train_acc                : " << r.train_acc * 100.0 << "%\n";
    std::cout << "  test_acc                 : " << r.test_acc  * 100.0 << "%\n";
}

static BenchmarkResult run_benchmark_on_dataset(
    const DatasetSpec& spec,
    const Dataset& data,
    const RunConfig& cfg)
{
    BenchmarkResult r;
    r.dataset_name = spec.name;
    r.n_samples    = static_cast<int>(data.features.size());
    r.n_features   = static_cast<int>(data.features[0].size());
    r.n_classes    = count_unique_classes(data.labels);

    auto [train_idx, test_idx] = train_test_split(data, cfg.train_ratio, cfg.seed);
    r.train_samples = static_cast<int>(train_idx.size());
    r.test_samples  = static_cast<int>(test_idx.size());

    // Full hybrid CPU-GPU training — all timing captured inside build_tree_levelwise
    LevelwiseBuildStats stats;
    Node* root = build_tree_levelwise(data, train_idx, cfg.max_depth, cfg.min_samples_split, cfg.bin_count, stats);

    r.total_train_sec  = stats.total_time_ms / 1000.0;
    r.data_prep_sec    = stats.data_prep_ms  / 1000.0;
    r.split_eval_sec   = stats.split_eval_ms / 1000.0;
    r.apply_split_sec  = stats.apply_ms      / 1000.0;

    // Inference timing on test set
    const auto p_start = std::chrono::high_resolution_clock::now();
    auto preds = predict_batch(root, data, test_idx);
    const auto p_end = std::chrono::high_resolution_clock::now();
    r.predict_sec = std::chrono::duration<double>(p_end - p_start).count();

    std::vector<int> truth;
    truth.reserve(test_idx.size());
    for (int idx : test_idx) truth.push_back(data.labels[idx]);

    const auto train_preds = predict_batch(root, data, train_idx);
    std::vector<int> train_truth;
    train_truth.reserve(train_idx.size());
    for (int idx : train_idx) train_truth.push_back(data.labels[idx]);

    r.train_acc = compute_accuracy(train_preds, train_truth);
    r.test_acc  = compute_accuracy(preds, truth);

    delete_tree(root);
    return r;
}

static std::vector<ScalabilityRow> run_scalability(
    const DatasetSpec& spec,
    const Dataset& full_data,
    const RunConfig& cfg,
    const std::vector<double>& fractions)
{
    std::vector<ScalabilityRow> rows;
    const int n_total = static_cast<int>(full_data.features.size());
    for (double frac : fractions) {
        int n_sub = static_cast<int>(std::round(frac * n_total));
        n_sub = std::clamp(n_sub, 200, n_total);

        Dataset sub;
        sub.feature_names = full_data.feature_names;
        std::vector<int> idx(n_total);
        std::iota(idx.begin(), idx.end(), 0);
        std::shuffle(idx.begin(), idx.end(), std::default_random_engine(cfg.seed + static_cast<int>(frac * 1000)));

        sub.features.reserve(n_sub);
        sub.labels.reserve(n_sub);
        for (int i = 0; i < n_sub; ++i) {
            sub.features.push_back(full_data.features[idx[i]]);
            sub.labels.push_back(full_data.labels[idx[i]]);
        }

        BenchmarkResult r = run_benchmark_on_dataset(spec, sub, cfg);
        rows.push_back({spec.name, frac, r.n_samples, r.total_train_sec, r.predict_sec, r.test_acc});
    }
    return rows;
}

// ============================================================================
// SANITY TESTS
// ============================================================================

static void test_pure_node_is_leaf() {
    std::cout << "\n[Test: pure node -> leaf]\n";
    Dataset data;
    for (int i = 0; i < 8; ++i) {
        data.features.push_back({static_cast<float>(i), 1.0f});
        data.labels.push_back(0);
    }
    std::vector<int> idx = {0,1,2,3,4,5,6,7};
    LevelwiseBuildStats stats;
    Node* root = build_tree_levelwise(data, idx, 10, 2, 4, stats);
    assert(root->is_leaf);
    assert(root->predicted_class == 0);
    delete_tree(root);
    std::cout << "  [PASS]\n";
}

static void test_levelwise_split_on_toy_data() {
    std::cout << "\n[Test: level-wise toy split]\n";
    Dataset data;
    for (int i = 0; i < 10; ++i) {
        data.features.push_back({static_cast<float>(i), 0.0f});
        data.labels.push_back(i < 5 ? 0 : 1);
    }
    std::vector<int> idx(10);
    std::iota(idx.begin(), idx.end(), 0);
    LevelwiseBuildStats stats;
    Node* root = build_tree_levelwise(data, idx, 3, 1, 8, stats);
    TreeStats ts;
    collect_stats(root, ts);
    assert(ts.internals > 0);
    delete_tree(root);
    std::cout << "  [PASS]\n";
}

// ============================================================================
// MAIN
// ============================================================================

int main() {
    std::cout << "=============================================================\n";
    std::cout << "   MILESTONE 2 — Final Integrated CPU-GPU Tree Trainer\n";
    std::cout << "=============================================================\n";

    try {
#ifdef __CUDACC__
        test_pure_node_is_leaf();
        test_levelwise_split_on_toy_data();

        RunConfig cfg;
        // Add/remove datasets here; missing files are skipped with a warning
        std::vector<DatasetSpec> datasets = {
            {"Iris",                          "Iris.csv",                  true},
            {"Shuttle",                       "shuttle.csv",               true},
            {"LetterRecognition",             "letter-recognition.csv",    true},
            {"Skin_NonSkin",                  "Skin_NonSkin.csv",          true},
            {"Synthetic_dataset_1000k_200f",  "synthetic_1000k_200f.csv",  true}
        };

        std::vector<std::string> search_roots = {
            std::filesystem::current_path().string(),
            (std::filesystem::current_path() / "datasets").string(),
            (std::filesystem::current_path() / "milestone-1").string()
        };

        std::vector<BenchmarkResult> metrics;
        std::vector<ScalabilityRow> scalability_rows;
        const std::vector<double> fractions = {0.10, 0.25, 0.50, 0.75, 1.00};

        for (const auto& spec : datasets) {
            std::string path = resolve_dataset_path(spec.file, search_roots);
            std::cout << "\n[Loading] " << spec.name << " from: " << path << "\n";
            Dataset data;
            try {
                data = load_csv(path, spec.has_header);
            } catch (const std::exception& ex) {
                std::cout << "  [Skipped] " << ex.what() << "\n";
                continue;
            }

            BenchmarkResult res = run_benchmark_on_dataset(spec, data, cfg);
            print_result_summary(res);
            metrics.push_back(res);

            auto scale = run_scalability(spec, data, cfg, fractions);
            scalability_rows.insert(scalability_rows.end(), scale.begin(), scale.end());
        }

        save_metrics_csv("benchmark_metrics_m2.csv", metrics);
        save_scalability_csv("benchmark_scalability_m2.csv", scalability_rows);

        std::cout << "\n[Artifacts saved]\n";
        std::cout << "  - benchmark_metrics_m2.csv\n";
        std::cout << "  - benchmark_scalability_m2.csv\n";
        std::cout << "\nDone. Compare split_eval_sec here vs M1 CSV for speedup figures.\n";
#else
        std::cerr << "ERROR: Compile with nvcc (not g++) — __CUDACC__ not defined.\n"
                    << "  Command: nvcc -O2 -std=c++17 -arch=sm_75 milestone2_code.cu -o milestone2\n";
        return 1;
#endif
    } catch (const std::exception& ex) {
        std::cerr << "Error: " << ex.what() << "\n";
        return 1;
    }

    return 0;
}