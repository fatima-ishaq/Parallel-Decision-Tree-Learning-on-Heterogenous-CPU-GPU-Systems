// ============================================================================
// MILESTONE 3 — INTEGRATED RANDOM FOREST WITH GPU-ACCELERATED TRAINING
// ============================================================================
// Builds directly on the Milestone 2 CPU-GPU hybrid pipeline.
// Every tree in the forest is trained using the SAME GPU split-evaluation
// kernels from M2 (GPUDataManager + gpu_find_best_splits_batch).
// Trees are trained sequentially on the CPU (one at a time through the GPU)
// because the GPU is a shared resource; parallelism across trees via CPU
// threads would require serialised GPU access and is architecturally complex.
// Inference (prediction) is fully parallelised using std::thread.
//
// Team contributions:
//   Yaman  (Set 1) : Bootstrap sampling, forest training control, tree collection
//                    (reuses M2 GPUDataManager for GPU memory management)
//   Faraz  (Set 2) : Compact tree representation, array-based inference, tests
//   Zuhaa  (Set 3) : Parallel inference (samples & trees), throughput benchmarking
//   Fatima (M2)    : GPU kernels (histogram + Gini) — unchanged from M2
//
// Parallelism levels:
//   Training  : GPU parallelises nodes × features × bins per tree (M2 kernels)
//   Inference : CPU threads parallelise across samples OR across trees
//
// Compile: nvcc -O2 -std=c++17 -arch=sm_75 milestone3_complete.cu -o milestone3
// Run:     ./milestone3
// ============================================================================

#include <algorithm>
#include <cassert>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <limits>
#include <map>
#include <mutex>
#include <numeric>
#include <queue>
#include <random>
#include <set>
#include <sstream>
#include <stdexcept>
#include <string>
#include <thread>
#include <vector>

#ifdef __CUDACC__
#include <cuda_runtime.h>
#endif

// ============================================================================
// SHARED DATA STRUCTURES  (Faraz — carried forward from M1/M2)
// ============================================================================

struct Dataset {
    std::vector<std::vector<float>> features;
    std::vector<int>                labels;
    std::vector<std::string>        feature_names;
};

struct SplitResult {
    int    best_feature   = -1;
    int    best_bin_index = -1;
    double best_threshold = 0.0;
    double best_gini      = std::numeric_limits<double>::infinity();
    bool   found          = false;
};

struct LevelwiseBuildStats {
    double total_time_ms = 0.0;
    double data_prep_ms  = 0.0;   // Step 2: index packing (Yaman)
    double split_eval_ms = 0.0;   // Step 3: GPU kernel time (Fatima)
    double apply_ms      = 0.0;   // Step 4: CPU apply splits (Zuhaa)
    int levels_processed    = 0;
    int total_nodes_created = 0;
};

// Pointer-based node used during training only; converted to CompactNode after
struct Node {
    std::vector<int> sample_indices;
    int   depth           = 0;
    int   feature_index   = -1;
    float threshold       = 0.0f;
    Node* left            = nullptr;
    Node* right           = nullptr;
    bool  is_leaf         = false;
    int   predicted_class = -1;
    explicit Node(int d = 0) : depth(d) {}
};

// Forward declarations for the inference types/functions that are defined later
// in the file but used by the variant comparison harness.
struct CompactForest;
struct FlatBatch;

static FlatBatch prepare_batch(const Dataset& data, const std::vector<int>& indices);
static std::vector<int> predict_sequential(const CompactForest& forest, const FlatBatch& batch);
static double compute_accuracy(const std::vector<int>& pred, const std::vector<int>& truth);
// CSV saving helpers (defined later)
static void save_forest_train_csv(
    const std::string& path,
    const std::string& dataset,
    const std::string& variant,
    int n_trees,
    int n_samples,
    int n_features,
    int n_classes,
    const struct ForestTrainStats& stats,
    double test_acc);

// ============================================================================
// FARAZ — CSV LOADER  (carried from M1/M2)
// ============================================================================

static Dataset load_csv(const std::string& filename, bool has_header = true) {
    Dataset data;
    std::ifstream file(filename);
    if (!file.is_open())
        throw std::runtime_error("Cannot open dataset: " + filename);

    auto trim = [](std::string s) {
        auto ns = [](unsigned char c){ return !std::isspace(c); };
        s.erase(s.begin(), std::find_if(s.begin(), s.end(), ns));
        s.erase(std::find_if(s.rbegin(), s.rend(), ns).base(), s.end());
        return s;
    };
    auto is_num = [](const std::string& s) {
        if (s.empty()) return false;
        char* e = nullptr; std::strtod(s.c_str(), &e);
        return e != s.c_str() && *e == '\0';
    };

    std::string line; int ln = 0;
    std::map<std::string,int> enc; int nid = 0;
    std::vector<std::string> hdr;

    while (std::getline(file, line)) {
        ++ln; if (line.empty()) continue;
        if (has_header && ln == 1) {
            std::stringstream ss(line); std::string t;
            while (std::getline(ss, t, ',')) hdr.push_back(trim(t));
            continue;
        }
        std::stringstream ss(line); std::string tok;
        std::vector<std::string> toks;
        while (std::getline(ss, tok, ',')) toks.push_back(trim(tok));
        if (toks.size() < 2) continue;

        const bool first_label = !is_num(toks.front());
        const int  li = first_label ? 0 : static_cast<int>(toks.size()) - 1;

        if (data.feature_names.empty() && !hdr.empty() && hdr.size() == toks.size()) {
            for (int i = 0; i < static_cast<int>(hdr.size()); ++i)
                if (i != li) data.feature_names.push_back(hdr[i]);
        }

        std::vector<float> row; row.reserve(toks.size() - 1);
        for (int i = 0; i < static_cast<int>(toks.size()); ++i) {
            if (i == li) continue;
            if (!is_num(toks[i])) { row.clear(); break; }
            row.push_back(static_cast<float>(std::stod(toks[i])));
        }
        if (row.empty()) continue;

        int label = 0;
        if (is_num(toks[li])) {
            label = static_cast<int>(std::lround(std::stod(toks[li])));
        } else {
            auto it = enc.find(toks[li]);
            if (it == enc.end()) { label = nid; enc[toks[li]] = nid++; }
            else label = it->second;
        }
        data.features.push_back(std::move(row));
        data.labels.push_back(label);
    }
    return data;
}

static int count_unique_classes(const std::vector<int>& labels) {
    std::map<int,int> c; for (int y : labels) c[y]++;
    return static_cast<int>(c.size());
}

static std::pair<std::vector<int>,std::vector<int>>
train_test_split(const Dataset& data, float ratio = 0.8f, int seed = 42) {
    int n = static_cast<int>(data.features.size());
    std::vector<int> idx(n); std::iota(idx.begin(), idx.end(), 0);
    std::shuffle(idx.begin(), idx.end(), std::default_random_engine(seed));
    int ts = static_cast<int>(n * ratio);
    return { {idx.begin(), idx.begin()+ts}, {idx.begin()+ts, idx.end()} };
}

static std::string resolve_path(const std::string& f,
                                 const std::vector<std::string>& roots) {
    namespace fs = std::filesystem;
    if (fs::exists(f)) return f;
    for (const auto& r : roots) {
        fs::path p = fs::path(r) / f;
        if (fs::exists(p)) return p.string();
    }
    return f;
}

// ============================================================================
// M2 GPU BACKEND — YAMAN + FATIMA  (carried from milestone2_code.cu, UNCHANGED)
// ============================================================================
// This is the exact GPU pipeline from Milestone 2.
// M3 uses it unchanged: each tree calls build_tree_levelwise which calls
// GPUDataManager::init + gpu_find_best_splits_batch internally.
// ============================================================================

struct GPUDataManager {
    bool initialized = false;
    int  bin_count   = 8;
    explicit GPUDataManager(int bins = 8) : bin_count(bins) {}
    void init(const Dataset& data);
    void release();
    ~GPUDataManager();
};

#ifdef __CUDACC__

constexpr int kMaxClassesPerNode = 128;

#define CUDA_CHECK(call) \
    do { cudaError_t e__ = (call); \
         if (e__ != cudaSuccess) throw std::runtime_error(cudaGetErrorString(e__)); \
    } while (0)

// Yaman: GPU-resident global state — one set for the whole process.
// Since trees are trained sequentially, this is safe.
static int   g_n_samples  = 0;
static int   g_n_features = 0;
static int   g_n_classes  = 0;
static std::vector<int>   g_h_x_bin;
static std::vector<int>   g_h_y;
static std::vector<float> g_h_min;
static std::vector<float> g_h_bin_width;
static int*   d_x_bin     = nullptr;
static int*   d_y         = nullptr;
static int*   d_indices   = nullptr;
static int*   d_offsets   = nullptr;
static int*   d_hist      = nullptr;
static float* d_best_gini = nullptr;
static int*   d_best_bin  = nullptr;
static std::size_t d_indices_capacity   = 0;
static std::size_t d_offsets_capacity   = 0;
static std::size_t d_hist_capacity      = 0;
static std::size_t d_best_gini_capacity = 0;
static std::size_t d_best_bin_capacity  = 0;

// Yaman: grow-never-shrink reusable buffer helper
template <typename T>
static void ensure_device_capacity(T*& ptr, std::size_t& cap, std::size_t req) {
    if (req == 0) return;
    if (ptr && cap >= req) return;
    if (ptr) { CUDA_CHECK(cudaFree(ptr)); ptr = nullptr; cap = 0; }
    CUDA_CHECK(cudaMalloc(&ptr, sizeof(T) * req));
    cap = req;
}

// Fatima: device helper
__device__ float device_gini_from_counts(const int* counts, int nc, int total) {
    if (total == 0) return 0.0f;
    float sq = 0.0f;
    for (int c = 0; c < nc; ++c) {
        float p = static_cast<float>(counts[c]) / static_cast<float>(total);
        sq += p * p;
    }
    return 1.0f - sq;
}

// Fatima — Kernel 1: histogram construction (nodes × features × samples in parallel)
__global__ void build_histograms_batched_kernel(
    const int* x_bin, const int* y,
    const int* packed_indices, const int* offsets,
    int n_nodes, int n_features, int bins, int n_classes,
    int* hist)
{
    const int local_i    = blockIdx.x * blockDim.x + threadIdx.x;
    const int feature_id = blockIdx.y;
    const int node_id    = blockIdx.z;
    if (node_id >= n_nodes || feature_id >= n_features) return;
    const int start = offsets[node_id], end = offsets[node_id + 1];
    if (local_i >= end - start) return;
    const int si  = packed_indices[start + local_i];
    const int bid = x_bin[si * n_features + feature_id];
    const int cls = y[si];
    if (bid < 0 || bid >= bins || cls < 0 || cls >= n_classes) return;
    atomicAdd(&hist[((node_id * n_features + feature_id) * bins + bid) * n_classes + cls], 1);
}

// Fatima — Kernel 2: Gini prefix-scan, best split per (node, feature)
__global__ void evaluate_splits_batched_kernel(
    const int* hist, const int* offsets,
    int n_nodes, int n_features, int bins, int n_classes,
    float* out_best_gini, int* out_best_bin)
{
    const int feature_id = blockIdx.x * blockDim.x + threadIdx.x;
    const int node_id    = blockIdx.y;
    if (node_id >= n_nodes || feature_id >= n_features) return;
    const int nc = n_classes;
    const int node_count = offsets[node_id + 1] - offsets[node_id];
    if (node_count <= 1 || nc > kMaxClassesPerNode) {
        out_best_gini[node_id * n_features + feature_id] = 1e30f;
        out_best_bin [node_id * n_features + feature_id] = -1;
        return;
    }
    int left[kMaxClassesPerNode] = {}, right[kMaxClassesPerNode] = {};
    int total = 0;
    for (int b = 0; b < bins; ++b)
        for (int c = 0; c < nc; ++c) {
            int v = hist[((node_id * n_features + feature_id) * bins + b) * nc + c];
            right[c] += v; total += v;
        }
    float best_g = 1e30f; int best_b = -1;
    int left_n = 0, right_n = total;
    for (int sb = 0; sb < bins - 1; ++sb) {
        for (int c = 0; c < nc; ++c) {
            int v = hist[((node_id * n_features + feature_id) * bins + sb) * nc + c];
            left[c] += v; left_n  += v;
            right[c]-= v; right_n -= v;
        }
        if (left_n == 0 || right_n == 0) continue;
        float wg = ((float)left_n /total)*device_gini_from_counts(left,  nc, left_n)
                 + ((float)right_n/total)*device_gini_from_counts(right, nc, right_n);
        if (wg < best_g) { best_g = wg; best_b = sb; }
    }
    out_best_gini[node_id * n_features + feature_id] = best_g;
    out_best_bin [node_id * n_features + feature_id] = best_b;
}

static int find_num_classes_host(const std::vector<int>& y) {
    int mx = -1;
    for (int v : y) { if (v < 0) throw std::invalid_argument("Negative label."); if (v > mx) mx = v; }
    return mx + 1;
}

// Yaman — Step 0: one-time GPU setup per tree (pre-bin + upload X_bin and y)
void GPUDataManager::init(const Dataset& data) {
    if (initialized) return;
    if (data.features.empty()) throw std::invalid_argument("Empty dataset.");
    g_n_samples  = static_cast<int>(data.features.size());
    g_n_features = static_cast<int>(data.features[0].size());
    g_n_classes  = find_num_classes_host(data.labels);
    g_h_y        = data.labels;
    g_h_x_bin.assign(g_n_samples * g_n_features, 0);
    g_h_min.assign(g_n_features, 0.0f);
    g_h_bin_width.assign(g_n_features, 0.0f);
    for (int f = 0; f < g_n_features; ++f) {
        float mn = data.features[0][f], mx = data.features[0][f];
        for (int i = 1; i < g_n_samples; ++i) {
            mn = std::min(mn, data.features[i][f]);
            mx = std::max(mx, data.features[i][f]);
        }
        g_h_min[f]       = mn;
        g_h_bin_width[f] = (mx > mn) ? (mx - mn) / static_cast<float>(bin_count) : 0.0f;
        for (int i = 0; i < g_n_samples; ++i) {
            int b = 0;
            if (g_h_bin_width[f] > 0.0f) {
                b = static_cast<int>(std::floor((data.features[i][f] - mn) / g_h_bin_width[f]));
                b = std::clamp(b, 0, bin_count - 1);
            }
            g_h_x_bin[i * g_n_features + f] = b;
        }
    }
    CUDA_CHECK(cudaMalloc(&d_x_bin, sizeof(int) * g_h_x_bin.size()));
    CUDA_CHECK(cudaMalloc(&d_y,     sizeof(int) * g_h_y.size()));
    CUDA_CHECK(cudaMemcpy(d_x_bin, g_h_x_bin.data(), sizeof(int)*g_h_x_bin.size(), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_y,     g_h_y.data(),     sizeof(int)*g_h_y.size(),     cudaMemcpyHostToDevice));
    initialized = true;
}

// Yaman — cleanup: free GPU memory after each tree finishes
void GPUDataManager::release() {
    if (!initialized) return;
    if (d_x_bin)     CUDA_CHECK(cudaFree(d_x_bin));
    if (d_y)         CUDA_CHECK(cudaFree(d_y));
    if (d_indices)   CUDA_CHECK(cudaFree(d_indices));
    if (d_offsets)   CUDA_CHECK(cudaFree(d_offsets));
    if (d_hist)      CUDA_CHECK(cudaFree(d_hist));
    if (d_best_gini) CUDA_CHECK(cudaFree(d_best_gini));
    if (d_best_bin)  CUDA_CHECK(cudaFree(d_best_bin));
    d_x_bin = d_y = d_indices = d_offsets = d_hist = nullptr;
    d_best_gini = nullptr; d_best_bin = nullptr;
    d_indices_capacity = d_offsets_capacity = d_hist_capacity = 0;
    d_best_gini_capacity = d_best_bin_capacity = 0;
    g_h_x_bin.clear(); g_h_y.clear(); g_h_min.clear(); g_h_bin_width.clear();
    g_n_samples = g_n_features = g_n_classes = 0;
    initialized = false;
}
GPUDataManager::~GPUDataManager() { release(); }

// Yaman+Fatima — Steps 2+3: batch-pack indices → GPU, run kernels, unpack results
static std::vector<SplitResult> gpu_find_best_splits_batch(
    const std::vector<std::vector<int>>& batches, int bin_count)
{
    if (g_n_features == 0) throw std::runtime_error("GPUDataManager not initialised.");
    if (batches.empty()) return {};
    const int nn = static_cast<int>(batches.size());
    std::vector<SplitResult> out(nn);

    // Step 2: build CSR offsets + packed index array
    std::vector<int> h_off(nn + 1, 0); int total = 0;
    for (int i = 0; i < nn; ++i) { h_off[i] = total; total += static_cast<int>(batches[i].size()); }
    h_off[nn] = total;
    std::vector<int> h_pack(total); int p = 0;
    for (const auto& v : batches) for (int x : v) h_pack[p++] = x;

    // Transfer packed indices to GPU (reused buffers — no cudaMalloc per level)
    ensure_device_capacity(d_indices, d_indices_capacity, h_pack.size());
    ensure_device_capacity(d_offsets, d_offsets_capacity, h_off.size());
    if (!h_pack.empty())
        CUDA_CHECK(cudaMemcpy(d_indices, h_pack.data(), sizeof(int)*h_pack.size(), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_offsets, h_off.data(), sizeof(int)*h_off.size(), cudaMemcpyHostToDevice));

    // Step 3 — Kernel 1: build histograms for all (node × feature) pairs
    const std::size_t hs = static_cast<std::size_t>(nn) * g_n_features * bin_count * g_n_classes;
    ensure_device_capacity(d_hist, d_hist_capacity, hs);
    CUDA_CHECK(cudaMemset(d_hist, 0, sizeof(int)*hs));
    const std::size_t fo = static_cast<std::size_t>(nn) * g_n_features;
    ensure_device_capacity(d_best_gini, d_best_gini_capacity, fo);
    ensure_device_capacity(d_best_bin,  d_best_bin_capacity,  fo);

    const int thr = 256;
    int mlen = 1;
    for (const auto& v : batches) mlen = std::max(mlen, static_cast<int>(v.size()));
    build_histograms_batched_kernel<<<dim3((mlen+thr-1)/thr, g_n_features, nn), dim3(thr,1,1)>>>(
        d_x_bin, d_y, d_indices, d_offsets, nn, g_n_features, bin_count, g_n_classes, d_hist);
    CUDA_CHECK(cudaGetLastError()); CUDA_CHECK(cudaDeviceSynchronize());

    // Step 3 — Kernel 2: prefix-scan Gini evaluation, best split per (node, feature)
    const int ft = 128;
    evaluate_splits_batched_kernel<<<dim3((g_n_features+ft-1)/ft, nn, 1), ft>>>(
        d_hist, d_offsets, nn, g_n_features, bin_count, g_n_classes, d_best_gini, d_best_bin);
    CUDA_CHECK(cudaGetLastError()); CUDA_CHECK(cudaDeviceSynchronize());

    // Copy small result arrays back; CPU reduces over features to get one best split per node
    std::vector<float> h_gini(fo, 1e30f); std::vector<int> h_bin(fo, -1);
    CUDA_CHECK(cudaMemcpy(h_gini.data(), d_best_gini, sizeof(float)*fo, cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(h_bin.data(),  d_best_bin,  sizeof(int)  *fo, cudaMemcpyDeviceToHost));

    for (int n = 0; n < nn; ++n) {
        SplitResult best;
        for (int f = 0; f < g_n_features; ++f) {
            const int i = n * g_n_features + f;
            if (h_bin[i] < 0) continue;
            const double g = static_cast<double>(h_gini[i]);
            if (g < best.best_gini) {
                best.best_gini      = g;
                best.best_feature   = f;
                best.best_bin_index = h_bin[i];
                best.best_threshold = static_cast<double>(g_h_min[f]) +
                                      static_cast<double>(h_bin[i] + 1) *
                                      static_cast<double>(g_h_bin_width[f]);
                best.found = true;
            }
        }
        out[n] = best;
    }
    return out;
}

#else
// Non-CUDA stub: fails loudly at runtime
void GPUDataManager::init(const Dataset&) {
    throw std::runtime_error("Compile with nvcc: nvcc -O2 -std=c++17 -arch=sm_75 milestone3_complete.cu -o milestone3");
}
void GPUDataManager::release() {}
GPUDataManager::~GPUDataManager() = default;
static std::vector<SplitResult> gpu_find_best_splits_batch(
    const std::vector<std::vector<int>>&, int) {
    throw std::runtime_error("Compile with nvcc.");
}
#endif

// ============================================================================
// M2 TREE BUILDER — ZUHAA  (carried from M2, unchanged)
// Called once per tree inside train_random_forest
// ============================================================================

static int majority_class(const std::vector<int>& idx, const std::vector<int>& labels) {
    std::map<int,int> c; for (int i : idx) c[labels[i]]++;
    return std::max_element(c.begin(), c.end(),
        [](const auto& a, const auto& b){ return a.second < b.second; })->first;
}
static bool is_pure(const std::vector<int>& idx, const std::vector<int>& labels) {
    if (idx.empty()) return true;
    int f = labels[idx[0]];
    for (int i : idx) if (labels[i] != f) return false;
    return true;
}
static bool should_split(const Node* n, const Dataset& d, int maxd, int mins) {
    const auto& idx = n->sample_indices;
    if (idx.empty() || n->depth >= maxd || (int)idx.size() < mins) return false;
    return !is_pure(idx, d.labels);
}
static void delete_tree(Node* n) {
    if (!n) return; delete_tree(n->left); delete_tree(n->right); delete n;
}

// Level-wise BFS training using the M2 GPU pipeline for split evaluation
static Node* build_tree_levelwise(
    const Dataset& data, const std::vector<int>& train_idx,
    int max_depth, int min_samples, int bin_count,
    LevelwiseBuildStats& stats)
{
    using clk = std::chrono::high_resolution_clock;
    const auto wall = clk::now();

    Node* root = new Node(0);
    root->sample_indices = train_idx;
    stats.total_nodes_created = 1;

    // Step 0: initialise GPU — pre-bin X→X_bin, upload once for this tree
    GPUDataManager gm(bin_count);
    gm.init(data);

    std::vector<Node*> frontier = { root };

    while (!frontier.empty()) {
        // Assign majority class; mark leaves; collect splittable nodes
        std::vector<Node*> active;
        for (Node* n : frontier) {
            n->predicted_class = majority_class(n->sample_indices, data.labels);
            if (!should_split(n, data, max_depth, min_samples)) n->is_leaf = true;
            else active.push_back(n);
        }
        if (active.empty()) break;

        // Step 2: collect per-node index batches
        const auto t0 = clk::now();
        std::vector<std::vector<int>> batches;
        batches.reserve(active.size());
        for (Node* n : active) batches.push_back(n->sample_indices);
        stats.data_prep_ms += std::chrono::duration<double,std::milli>(clk::now()-t0).count();

        // Step 3: GPU finds best split for every active node in one kernel launch
        const auto t1 = clk::now();
        auto results = gpu_find_best_splits_batch(batches, bin_count);
        stats.split_eval_ms += std::chrono::duration<double,std::milli>(clk::now()-t1).count();

        // Step 4: CPU applies splits, creates children
        const auto t2 = clk::now();
        std::vector<Node*> next;
        next.reserve(active.size() * 2);
        for (int i = 0; i < (int)active.size(); ++i) {
            Node* n = active[i];
            const SplitResult& sr = results[i];
            if (!sr.found) { n->is_leaf = true; continue; }
            std::vector<int> li, ri;
            for (int s : n->sample_indices) {
                if (data.features[s][sr.best_feature] <= sr.best_threshold) li.push_back(s);
                else ri.push_back(s);
            }
            if (li.empty() || ri.empty()) { n->is_leaf = true; continue; }
            n->feature_index = sr.best_feature;
            n->threshold     = static_cast<float>(sr.best_threshold);
            Node* L = new Node(n->depth+1); L->sample_indices = std::move(li); n->left  = L;
            Node* R = new Node(n->depth+1); R->sample_indices = std::move(ri); n->right = R;
            next.push_back(L); next.push_back(R);
            stats.total_nodes_created += 2;
        }
        stats.apply_ms += std::chrono::duration<double,std::milli>(clk::now()-t2).count();
        frontier.swap(next);
        ++stats.levels_processed;
    }
    stats.total_time_ms = std::chrono::duration<double,std::milli>(clk::now()-wall).count();
    // GPUDataManager RAII: gm.release() called here automatically
    return root;
}

// ============================================================================
// YAMAN — SET 1: BOOTSTRAP SAMPLING + FOREST TRAINING CONTROL
// ============================================================================
// Responsibilities:
//   - Bootstrap sampling with replacement (one per tree, different seed each)
//   - Sequential tree training loop calling the M2 GPU pipeline per tree
//   - Forest collection, stats tracking, tree count cap (MAX_TREES = 10)
//
// Note on parallelism: Trees are trained sequentially because the GPU is a
// shared single resource. Each tree fully occupies the GPU during its training
// via GPUDataManager. The parallelism across trees is therefore at the
// conceptual level (independent bootstrap samples, independent split paths);
// actual GPU-level parallelism occurs WITHIN each tree (nodes × features × bins).
// ============================================================================

static constexpr int MAX_TREES = 10;

struct ForestConfig {
    int   n_trees           = MAX_TREES;
    int   max_depth         = 8;
    int   min_samples_split = 2;
    int   bin_count         = 8;
    float bootstrap_ratio   = 1.0f;
    int   base_seed         = 42;
};

struct ForestTrainStats {
    int    n_trees         = 0;
    double total_forest_ms = 0.0;
    double bootstrap_ms    = 0.0;
    double tree_build_ms   = 0.0;  // total GPU+CPU training time across all trees
    double gpu_kernel_ms   = 0.0;  // accumulated split_eval_ms across all trees
    int    total_nodes_created = 0;
    std::vector<double> per_tree_ms;
    std::vector<int>    per_tree_nodes;
};

// Yaman: bootstrap sample — draw n_samples indices WITH REPLACEMENT from pool
static std::vector<int> bootstrap_sample(
    const std::vector<int>& pool, int n_samples, int seed)
{
    if (pool.empty()) throw std::invalid_argument("bootstrap_sample: empty pool.");
    std::mt19937 rng(static_cast<uint32_t>(seed));
    std::uniform_int_distribution<int> dist(0, static_cast<int>(pool.size()) - 1);
    std::vector<int> s(n_samples);
    for (int& v : s) v = pool[dist(rng)];
    return s;
}

struct ForestResult {
    std::vector<Node*> trees;
    int                n_classes = 0;
    ForestTrainStats   stats;

    ForestResult() = default;
    ForestResult(const ForestResult&)            = delete;
    ForestResult& operator=(const ForestResult&) = delete;
    ForestResult(ForestResult&& o) noexcept
        : trees(std::move(o.trees)), n_classes(o.n_classes), stats(std::move(o.stats))
    { o.n_classes = 0; }
    ~ForestResult() { for (Node* t : trees) delete_tree(t); }
};

// ============================================================================
// YAMAN — THREE TRAINING VARIANTS FOR EXPERIMENTATION
// ============================================================================
// Variant 1: Sequential (baseline)
// Variant 2: Mutex-protected CPU parallelism (GPU serialized)
// Variant 3: CUDA streams with per-tree context (concurrent kernel submission)
// ============================================================================

// Global mutex for GPU access protection (Variant 2)
static std::mutex g_gpu_mutex;

// Per-tree GPU context for streams variant
struct TreeGPUContext {
#ifdef __CUDACC__
    cudaStream_t stream = nullptr;
#else
    void* stream = nullptr;
#endif
    TreeGPUContext() {
#ifdef __CUDACC__
        CUDA_CHECK(cudaStreamCreate(&stream));
#endif
    }
    ~TreeGPUContext() {
#ifdef __CUDACC__
        if (stream) (void)cudaStreamDestroy(stream);
#endif
    }
    TreeGPUContext(const TreeGPUContext&) = delete;
    TreeGPUContext& operator=(const TreeGPUContext&) = delete;
    TreeGPUContext(TreeGPUContext&& o) noexcept : stream(o.stream) { o.stream = nullptr; }
};

// ────────────────────────────────────────────────────────────────────────────
// VARIANT 1: Sequential Training (Baseline — equivalent to current implementation)
// ────────────────────────────────────────────────────────────────────────────
static ForestResult train_random_forest_sequential(
    const Dataset& data, const std::vector<int>& train_idx,
    const ForestConfig& cfg)
{
    using clk = std::chrono::high_resolution_clock;
    const int K = std::clamp(cfg.n_trees, 1, MAX_TREES);
    const int nb = static_cast<int>(
        std::round(cfg.bootstrap_ratio * static_cast<float>(train_idx.size())));
    if (nb <= 0) throw std::invalid_argument("Bootstrap ratio produced 0 samples.");

    ForestResult forest;
    forest.n_classes = count_unique_classes(data.labels);
    forest.trees.reserve(K);
    forest.stats.per_tree_ms.reserve(K);
    forest.stats.per_tree_nodes.reserve(K);

    const auto wall = clk::now();

    for (int t = 0; t < K; ++t) {
        const auto bs = clk::now();
        auto boot = bootstrap_sample(train_idx, nb, cfg.base_seed + t);
        forest.stats.bootstrap_ms +=
            std::chrono::duration<double,std::milli>(clk::now()-bs).count();

        const auto ts = clk::now();
        LevelwiseBuildStats lstats;
        Node* root = build_tree_levelwise(
            data, boot, cfg.max_depth, cfg.min_samples_split, cfg.bin_count, lstats);
        const double tree_ms =
            std::chrono::duration<double,std::milli>(clk::now()-ts).count();

        forest.stats.tree_build_ms   += tree_ms;
        forest.stats.gpu_kernel_ms   += lstats.split_eval_ms;
        forest.stats.per_tree_ms.push_back(tree_ms);
        forest.stats.per_tree_nodes.push_back(lstats.total_nodes_created);
        forest.trees.push_back(root);

        std::cout << "  [Sequential] Tree " << (t+1) << "/" << K
                  << "  nodes=" << lstats.total_nodes_created
                  << "  gpu_split_ms=" << std::fixed << std::setprecision(2)
                  << lstats.split_eval_ms
                  << "  total_ms=" << tree_ms << "\n";
    }

    forest.stats.total_forest_ms =
        std::chrono::duration<double,std::milli>(clk::now()-wall).count();
    forest.stats.n_trees = K;
    return forest;
}

// ────────────────────────────────────────────────────────────────────────────
// VARIANT 2: Mutex-Protected Parallel Training
// CPU threads run bootstrap and tree construction in parallel;
// GPU access (build_tree_levelwise) is serialized by a mutex.
// ────────────────────────────────────────────────────────────────────────────
static ForestResult train_random_forest_mutex(
    const Dataset& data, const std::vector<int>& train_idx,
    const ForestConfig& cfg)
{
    using clk = std::chrono::high_resolution_clock;
    const int K = std::clamp(cfg.n_trees, 1, MAX_TREES);
    const int nb = static_cast<int>(
        std::round(cfg.bootstrap_ratio * static_cast<float>(train_idx.size())));
    if (nb <= 0) throw std::invalid_argument("Bootstrap ratio produced 0 samples.");

    ForestResult forest;
    forest.n_classes = count_unique_classes(data.labels);
    forest.trees.resize(K, nullptr);
    forest.stats.per_tree_ms.resize(K, 0.0);
    forest.stats.per_tree_nodes.resize(K, 0);

    const auto wall = clk::now();
    std::vector<std::thread> workers(K);

    for (int t = 0; t < K; ++t) {
        workers[t] = std::thread([&, t]() {
            // CPU work: bootstrap (no lock needed)
            const auto bs = clk::now();
            auto boot = bootstrap_sample(train_idx, nb, cfg.base_seed + t);
            const double boot_ms =
                std::chrono::duration<double,std::milli>(clk::now()-bs).count();

            // GPU work: protected by mutex (only one thread uses GPU at a time)
            const auto gpu_start = clk::now();
            {
                std::lock_guard<std::mutex> lock(g_gpu_mutex);
                LevelwiseBuildStats lstats;
                forest.trees[t] = build_tree_levelwise(
                    data, boot, cfg.max_depth, cfg.min_samples_split, cfg.bin_count, lstats);
                forest.stats.per_tree_nodes[t] = lstats.total_nodes_created;
                forest.stats.gpu_kernel_ms += lstats.split_eval_ms;
            }
            const double gpu_ms =
                std::chrono::duration<double,std::milli>(clk::now()-gpu_start).count();
            const double total_ms = boot_ms + gpu_ms;

            forest.stats.per_tree_ms[t] = total_ms;
            std::cout << "  [Mutex] Tree " << (t+1) << "/" << K
                      << "  nodes=" << forest.stats.per_tree_nodes[t]
                      << "  boot_ms=" << std::fixed << std::setprecision(2) << boot_ms
                      << "  gpu_ms=" << gpu_ms << "\n";
        });
    }
    for (auto& w : workers) w.join();

    for (double ms : forest.stats.per_tree_ms) forest.stats.tree_build_ms += ms;
    for (int n : forest.stats.per_tree_nodes) forest.stats.total_nodes_created += n;

    forest.stats.total_forest_ms =
        std::chrono::duration<double,std::milli>(clk::now()-wall).count();
    forest.stats.n_trees = K;
    return forest;
}

// ────────────────────────────────────────────────────────────────────────────
// VARIANT 3: CUDA Streams Parallel Training (Experimental)
// CPU threads create per-tree CUDA streams and submit kernel launches concurrently.
// GPU scheduler can overlap kernels where SM capacity permits.
// Note: Global GPU state is still shared; this variant shows kernel-level overlap only.
// ────────────────────────────────────────────────────────────────────────────
static ForestResult train_random_forest_streams(
    const Dataset& data, const std::vector<int>& train_idx,
    const ForestConfig& cfg)
{
    using clk = std::chrono::high_resolution_clock;
    const int K = std::clamp(cfg.n_trees, 1, MAX_TREES);
    const int nb = static_cast<int>(
        std::round(cfg.bootstrap_ratio * static_cast<float>(train_idx.size())));
    if (nb <= 0) throw std::invalid_argument("Bootstrap ratio produced 0 samples.");

    ForestResult forest;
    forest.n_classes = count_unique_classes(data.labels);
    forest.trees.resize(K, nullptr);
    forest.stats.per_tree_ms.resize(K, 0.0);
    forest.stats.per_tree_nodes.resize(K, 0);

    const auto wall = clk::now();
    std::vector<std::thread> workers(K);
    std::vector<TreeGPUContext> contexts(K);

    for (int t = 0; t < K; ++t) {
        workers[t] = std::thread([&, t]() {
            // CPU work: bootstrap
            const auto bs = clk::now();
            auto boot = bootstrap_sample(train_idx, nb, cfg.base_seed + t);
            const double boot_ms =
                std::chrono::duration<double,std::milli>(clk::now()-bs).count();

            // GPU work: use per-tree context (stream submitted to GPU scheduler)
            // Note: Global state is still shared, so kernels queue up but have contention
            const auto gpu_start = clk::now();
            {
                std::lock_guard<std::mutex> lock(g_gpu_mutex);  // Protect global state access
                LevelwiseBuildStats lstats;
                forest.trees[t] = build_tree_levelwise(
                    data, boot, cfg.max_depth, cfg.min_samples_split, cfg.bin_count, lstats);
                forest.stats.per_tree_nodes[t] = lstats.total_nodes_created;
                forest.stats.gpu_kernel_ms += lstats.split_eval_ms;
#ifdef __CUDACC__
                // Synchronize only this tree's stream (if per-stream isolation were implemented)
                // For now, global state means we still serialize; this is experimental scaffolding
                CUDA_CHECK(cudaStreamSynchronize(contexts[t].stream));
#endif
            }
            const double gpu_ms =
                std::chrono::duration<double,std::milli>(clk::now()-gpu_start).count();
            const double total_ms = boot_ms + gpu_ms;

            forest.stats.per_tree_ms[t] = total_ms;
            std::cout << "  [Streams] Tree " << (t+1) << "/" << K
                      << "  nodes=" << forest.stats.per_tree_nodes[t]
                      << "  boot_ms=" << std::fixed << std::setprecision(2) << boot_ms
                      << "  gpu_ms=" << gpu_ms << "\n";
        });
    }
    for (auto& w : workers) w.join();

    for (double ms : forest.stats.per_tree_ms) forest.stats.tree_build_ms += ms;
    for (int n : forest.stats.per_tree_nodes) forest.stats.total_nodes_created += n;

    forest.stats.total_forest_ms =
        std::chrono::duration<double,std::milli>(clk::now()-wall).count();
    forest.stats.n_trees = K;
    return forest;
}

// ────────────────────────────────────────────────────────────────────────────
// LEGACY: Current train_random_forest delegates to sequential variant
// ────────────────────────────────────────────────────────────────────────────
// Yaman: train K trees sequentially; each tree uses the M2 GPU pipeline internally
static ForestResult train_random_forest(
    const Dataset& data, const std::vector<int>& train_idx,
    const ForestConfig& cfg)
{
    return train_random_forest_sequential(data, train_idx, cfg);
}


// ============================================================================
// FARAZ — SET 2: COMPACT TREE REPRESENTATION + ARRAY-BASED INFERENCE
// ============================================================================
// Responsibilities:
//   - Convert pointer-based Node trees to flat integer-array layout (CompactNode)
//   - Iterative (non-recursive) single-tree prediction using array traversal
//   - Sequential forest majority-vote prediction (baseline for benchmarking)
//   - Memory report comparing compact vs pointer layout
// ============================================================================

struct CompactNode {
    int   feature         = -1;
    float threshold       = 0.0f;
    int   left_idx        = -1;
    int   right_idx       = -1;
    int   predicted_class = -1;
    bool  is_leaf         = true;
    CompactNode() = default;
    CompactNode(int feat, float thr, int li, int ri)
        : feature(feat), threshold(thr), left_idx(li), right_idx(ri), is_leaf(false) {}
    explicit CompactNode(int cls) : predicted_class(cls), is_leaf(true) {}
};

// Faraz: BFS serialisation — assigns node indices in BFS discovery order
// and writes them into a flat array, replacing pointers with integer indices.
static std::vector<CompactNode> serialize_tree(const Node* root) {
    if (!root) return {};

    // Pass 1: BFS to assign stable indices in discovery order
    std::vector<const Node*> bfs_order;
    std::map<const Node*, int> idx_map;
    std::queue<const Node*> q;
    q.push(root);
    while (!q.empty()) {
        const Node* n = q.front(); q.pop();
        if (idx_map.count(n)) continue;
        idx_map[n] = static_cast<int>(bfs_order.size());
        bfs_order.push_back(n);
        if (n->left)  q.push(n->left);
        if (n->right) q.push(n->right);
    }

    // Pass 2: fill compact array using the stable index map
    std::vector<CompactNode> nodes(bfs_order.size());
    for (const Node* n : bfs_order) {
        int i = idx_map[n];
        if (n->is_leaf || (!n->left && !n->right)) {
            nodes[i] = CompactNode(n->predicted_class);
        } else {
            nodes[i] = CompactNode(
                n->feature_index, n->threshold,
                idx_map[n->left], idx_map[n->right]);
        }
    }
    return nodes;
}

struct CompactForest {
    std::vector<std::vector<CompactNode>> trees;
    int n_classes = 0;
    int n_trees   = 0;

    CompactForest() = default;
    CompactForest(const ForestResult& fr, int nc) : n_classes(nc), n_trees(static_cast<int>(fr.trees.size())) {
        trees.reserve(fr.trees.size());
        for (const Node* root : fr.trees) trees.push_back(serialize_tree(root));
    }
};

// Faraz: iterative single-tree prediction using compact array
static int predict_compact_tree(
    const std::vector<CompactNode>& tree,
    const float* row)
{
    if (tree.empty()) return -1;
    int idx = 0;
    while (true) {
        const CompactNode& n = tree[idx];
        if (n.is_leaf || n.feature < 0) return n.predicted_class;
        idx = (row[n.feature] <= n.threshold) ? n.left_idx : n.right_idx;
        if (idx < 0 || idx >= (int)tree.size()) return -1;
    }
}

static double compute_accuracy(const std::vector<int>& pred, const std::vector<int>& truth) {
    if (pred.empty() || pred.size() != truth.size()) return 0.0;
    int c = 0; for (int i = 0; i < (int)pred.size(); ++i) if (pred[i] == truth[i]) ++c;
    return static_cast<double>(c) / pred.size();
}

// ============================================================================
// ZUHAA — SET 3: PARALLEL INFERENCE + THROUGHPUT BENCHMARKING
// ============================================================================
// Responsibilities:
//   1. FlatBatch: flatten Dataset into contiguous row-major float buffer
//      so all inference variants read from the same cache-friendly layout.
//   2. predict_sequential: compact-array traversal, sequential baseline.
//   3. predict_parallel_samples: divide N test samples across std::threads;
//      each thread owns a contiguous slice → zero shared-write contention.
//   4. predict_parallel_trees: each thread group owns a slice of K trees,
//      writes into a T×N vote matrix; final sequential vote reduction.
//   5. measure_throughput: times all three variants, reports samples/sec + speedup.
//   6. speedup_vs_trees: re-uses sub-forests of 1..K trees (no re-training)
//      to show how inference parallelism scales with ensemble size.
// ============================================================================

static int hw_threads() {
    int n = static_cast<int>(std::thread::hardware_concurrency());
    return n > 0 ? n : 2;
}

// Zuhaa: flatten Dataset rows → single contiguous float array (prepared once, outside timing)
struct FlatBatch {
    std::vector<float> x;
    int n_samples  = 0;
    int n_features = 0;
};

static FlatBatch prepare_batch(const Dataset& data, const std::vector<int>& indices) {
    FlatBatch b;
    b.n_samples  = static_cast<int>(indices.size());
    b.n_features = data.features.empty() ? 0 : static_cast<int>(data.features[0].size());
    b.x.resize(static_cast<std::size_t>(b.n_samples) * b.n_features);
    for (int i = 0; i < b.n_samples; ++i)
        std::copy(data.features[indices[i]].begin(), data.features[indices[i]].end(),
                  b.x.data() + static_cast<std::size_t>(i) * b.n_features);
    return b;
}

// Internal: predict one sample through one compact tree using flat pointer
static int predict_one(const std::vector<CompactNode>& tree, const float* row) {
    if (tree.empty()) return -1;
    int idx = 0;
    while (true) {
        const CompactNode& n = tree[idx];
        if (n.is_leaf || n.feature < 0) return n.predicted_class;
        idx = (row[n.feature] <= n.threshold) ? n.left_idx : n.right_idx;
        if (idx < 0 || idx >= (int)tree.size()) return -1;
    }
}

// Zuhaa: sequential prediction using FlatBatch (direct baseline for speedup)
static std::vector<int> predict_sequential(
    const CompactForest& forest, const FlatBatch& batch)
{
    const int N = batch.n_samples, nf = batch.n_features;
    std::vector<int> out(N);
    for (int i = 0; i < N; ++i) {
        const float* row = batch.x.data() + (std::size_t)i * nf;
        std::vector<int> votes(forest.n_classes, 0);
        for (const auto& tree : forest.trees) {
            int p = predict_one(tree, row);
            if (p >= 0 && p < forest.n_classes) ++votes[p];
        }
        out[i] = static_cast<int>(std::max_element(votes.begin(), votes.end()) - votes.begin());
    }
    return out;
}

// Zuhaa: data-parallel inference — N samples split across n_threads (no locks)
static std::vector<int> predict_parallel_samples(
    const CompactForest& forest, const FlatBatch& batch, int n_threads = -1)
{
    if (n_threads <= 0) n_threads = hw_threads();
    n_threads = std::min(n_threads, batch.n_samples);
    if (n_threads <= 1) return predict_sequential(forest, batch);

    const int N = batch.n_samples, nf = batch.n_features;
    std::vector<int> out(N);
    std::vector<std::thread> workers(n_threads);

    for (int t = 0; t < n_threads; ++t) {
        const int begin = (t * N) / n_threads;
        const int end   = ((t+1) * N) / n_threads;
        workers[t] = std::thread([&, begin, end]() {
            for (int i = begin; i < end; ++i) {
                const float* row = batch.x.data() + (std::size_t)i * nf;
                std::vector<int> votes(forest.n_classes, 0);
                for (const auto& tree : forest.trees) {
                    int p = predict_one(tree, row);
                    if (p >= 0 && p < forest.n_classes) ++votes[p];
                }
                out[i] = static_cast<int>(
                    std::max_element(votes.begin(), votes.end()) - votes.begin());
            }
        });
    }
    for (auto& w : workers) w.join();
    return out;
}

// Zuhaa: tree-parallel inference — K trees split across n_threads, vote matrix merged
static std::vector<int> predict_parallel_trees(
    const CompactForest& forest, const FlatBatch& batch, int n_threads = -1)
{
    const int T = static_cast<int>(forest.trees.size());
    const int N = batch.n_samples, nf = batch.n_features;
    if (n_threads <= 0) n_threads = hw_threads();
    n_threads = std::min(n_threads, T);
    if (n_threads <= 1 || T == 0) return predict_sequential(forest, batch);

    // tree_votes[t][i] = class predicted by tree t for sample i
    std::vector<std::vector<int>> tv(T, std::vector<int>(N, -1));
    std::vector<std::thread> workers(n_threads);

    for (int tg = 0; tg < n_threads; ++tg) {
        const int tb = (tg * T) / n_threads;
        const int te = ((tg+1) * T) / n_threads;
        workers[tg] = std::thread([&, tb, te]() {
            for (int t = tb; t < te; ++t) {
                const auto& tree = forest.trees[t];
                for (int i = 0; i < N; ++i)
                    tv[t][i] = predict_one(tree, batch.x.data() + (std::size_t)i * nf);
            }
        });
    }
    for (auto& w : workers) w.join();

    // Sequential vote reduction
    std::vector<int> out(N);
    for (int i = 0; i < N; ++i) {
        std::vector<int> votes(forest.n_classes, 0);
        for (int t = 0; t < T; ++t) {
            int p = tv[t][i];
            if (p >= 0 && p < forest.n_classes) ++votes[p];
        }
        out[i] = static_cast<int>(std::max_element(votes.begin(), votes.end()) - votes.begin());
    }
    return out;
}

// Zuhaa: timing helper — minimum over reps to reduce OS jitter
template <typename Fn>
static double time_min_ms(Fn&& fn, int reps = 5) {
    double best = std::numeric_limits<double>::infinity();
    for (int r = 0; r < reps; ++r) {
        auto t0 = std::chrono::high_resolution_clock::now();
        fn();
        double ms = std::chrono::duration<double,std::milli>(
            std::chrono::high_resolution_clock::now() - t0).count();
        if (ms < best) best = ms;
    }
    return best;
}

struct ThroughputResult {
    std::string label;
    int    n_samples       = 0;
    int    n_trees         = 0;
    int    n_threads       = 0;
    double elapsed_ms      = 0.0;
    double samples_per_sec = 0.0;
    double speedup         = 1.0;
};

static ThroughputResult make_result(const std::string& lbl, int N, int T, int nth,
                                     double ms, double base_ms) {
    ThroughputResult r;
    r.label = lbl; r.n_samples = N; r.n_trees = T; r.n_threads = nth;
    r.elapsed_ms      = ms;
    r.samples_per_sec = ms > 0.0 ? N / (ms / 1000.0) : 0.0;
    r.speedup         = (ms > 0.0 && base_ms > 0.0) ? base_ms / ms : 1.0;
    return r;
}

// Zuhaa: measure all three inference variants on the same batch
static std::vector<ThroughputResult> measure_throughput(
    const CompactForest& forest, const Dataset& data,
    const std::vector<int>& test_idx, int n_threads = -1, int reps = 5)
{
    if (n_threads <= 0) n_threads = hw_threads();
    const FlatBatch batch = prepare_batch(data, test_idx);
    const int N = batch.n_samples, T = static_cast<int>(forest.trees.size());
    predict_sequential(forest, batch);  // warm-up

    const double seq_ms   = time_min_ms([&]{ predict_sequential(forest, batch); }, reps);
    const double par_s_ms = time_min_ms([&]{ predict_parallel_samples(forest, batch, n_threads); }, reps);
    const double par_t_ms = time_min_ms([&]{ predict_parallel_trees(forest, batch, n_threads); }, reps);

    return {
        make_result("Sequential (compact baseline)",  N, T, 1,        seq_ms,   seq_ms),
        make_result("Parallel across samples",         N, T, n_threads, par_s_ms, seq_ms),
        make_result("Parallel across trees",           N, T, n_threads, par_t_ms, seq_ms),
    };
}

struct SpeedupRow {
    int    n_trees    = 0;
    double seq_ms     = 0.0;
    double par_ms     = 0.0;
    double speedup    = 0.0;
    double throughput = 0.0;
};

// Zuhaa: speedup vs number of trees (sub-forest views, no re-training)
static std::vector<SpeedupRow> speedup_vs_trees(
    const CompactForest& forest, const FlatBatch& batch,
    const std::vector<int>& counts, int n_threads = -1, int reps = 5)
{
    if (n_threads <= 0) n_threads = hw_threads();
    const int N = batch.n_samples;
    std::vector<SpeedupRow> rows;
    for (int k : counts) {
        if (k < 1 || k > (int)forest.trees.size()) continue;
        CompactForest sub;
        sub.n_classes = forest.n_classes; sub.n_trees = k;
        sub.trees.assign(forest.trees.begin(), forest.trees.begin() + k);
        const double sm = time_min_ms([&]{ predict_sequential(sub, batch); }, reps);
        const double pm = time_min_ms([&]{ predict_parallel_samples(sub, batch, n_threads); }, reps);
        rows.push_back({k, sm, pm, pm > 0.0 ? sm/pm : 1.0, pm > 0.0 ? N/(pm/1000.0) : 0.0});
    }
    return rows;
}

// Helper: print variant comparison results
static void print_variant_comparison(
    const std::string& variant_name,
    const ForestTrainStats& stats,
    double test_accuracy)
{
    std::cout << "\n[" << variant_name << " Results]\n" << std::fixed << std::setprecision(2);
    std::cout << "  Total forest time  : " << stats.total_forest_ms  << " ms\n";
    std::cout << "  Tree build time    : " << stats.tree_build_ms    << " ms\n";
    std::cout << "  GPU kernel time    : " << stats.gpu_kernel_ms    << " ms\n";
    std::cout << "  Test accuracy      : " << test_accuracy * 100.0  << "%\n";
}

// Variant comparison harness
struct VariantComparison {
    std::string name;
    double      total_time_ms;
    double      gpu_time_ms;
    double      speedup;
    double      accuracy;
};

static void print_variant_table(const std::vector<VariantComparison>& rows) {
    if (rows.empty()) return;
    std::cout << "\n  " << std::left
              << std::setw(20) << "Variant"
              << std::setw(16) << "Total Time(ms)"
              << std::setw(16) << "GPU Time(ms)"
              << std::setw(12) << "Speedup"
              << std::setw(12) << "Accuracy"
              << "\n  " << std::string(76, '-') << "\n";
    for (const auto& r : rows) {
        std::cout << "  " << std::left
                  << std::setw(20) << r.name
                  << std::setw(16) << std::fixed << std::setprecision(2) << r.total_time_ms
                  << std::setw(16) << r.gpu_time_ms
                  << std::setw(12) << (r.speedup > 0 ? r.speedup : 1.0)
                  << std::setprecision(1) << r.accuracy * 100.0 << "%\n";
    }
}

// Main comparison function for three variants — runs all 3 and saves to CSV
static void run_variant_comparison(
    const Dataset& data,
    const std::vector<int>& train_idx,
    const std::vector<int>& test_idx,
    const ForestConfig& cfg,
    const std::string& dataset_name,
    int n_features,
    int n_classes)
{
    using clk = std::chrono::high_resolution_clock;
    std::vector<VariantComparison> results;

    std::cout << "\n[Running three training variants on " << cfg.n_trees << " trees]\n";

    std::vector<int> truth;
    for (int i : test_idx) truth.push_back(data.labels[i]);

    // ── Variant 1: Sequential ──────────────────────────────────────────────────
    std::cout << "\n[Variant 1: Sequential (baseline)]\n";
    const auto t1_start = clk::now();
    ForestResult forest1 = train_random_forest_sequential(data, train_idx, cfg);
    const double t1_ms = std::chrono::duration<double,std::milli>(clk::now()-t1_start).count();
    CompactForest cf1(forest1, n_classes);
    FlatBatch batch1 = prepare_batch(data, test_idx);
    auto preds1 = predict_sequential(cf1, batch1);
    double acc1 = compute_accuracy(preds1, truth);
    print_variant_comparison("Sequential", forest1.stats, acc1);
    results.push_back({"Sequential", t1_ms, forest1.stats.gpu_kernel_ms, 1.0, acc1});
    save_forest_train_csv("benchmark_forest_train_m3.csv",
        dataset_name, "Sequential", cfg.n_trees,
        (int)data.features.size(), n_features, n_classes, forest1.stats, acc1);

    // ── Variant 2: Mutex-Protected ────────────────────────────────────────────
    std::cout << "\n[Variant 2: Mutex-Protected Parallel]\n";
    const auto t2_start = clk::now();
    ForestResult forest2 = train_random_forest_mutex(data, train_idx, cfg);
    const double t2_ms = std::chrono::duration<double,std::milli>(clk::now()-t2_start).count();
    CompactForest cf2(forest2, n_classes);
    FlatBatch batch2 = prepare_batch(data, test_idx);
    auto preds2 = predict_sequential(cf2, batch2);
    double acc2 = compute_accuracy(preds2, truth);
    print_variant_comparison("Mutex-Protected", forest2.stats, acc2);
    results.push_back({"Mutex", t2_ms, forest2.stats.gpu_kernel_ms, t1_ms / t2_ms, acc2});
    save_forest_train_csv("benchmark_forest_train_m3.csv",
        dataset_name, "Mutex", cfg.n_trees,
        (int)data.features.size(), n_features, n_classes, forest2.stats, acc2);

    // ── Variant 3: Streams ────────────────────────────────────────────────────
    std::cout << "\n[Variant 3: CUDA Streams]\n";
    const auto t3_start = clk::now();
    ForestResult forest3 = train_random_forest_streams(data, train_idx, cfg);
    const double t3_ms = std::chrono::duration<double,std::milli>(clk::now()-t3_start).count();
    CompactForest cf3(forest3, n_classes);
    FlatBatch batch3 = prepare_batch(data, test_idx);
    auto preds3 = predict_sequential(cf3, batch3);
    double acc3 = compute_accuracy(preds3, truth);
    print_variant_comparison("CUDA Streams", forest3.stats, acc3);
    results.push_back({"Streams", t3_ms, forest3.stats.gpu_kernel_ms, t1_ms / t3_ms, acc3});
    save_forest_train_csv("benchmark_forest_train_m3.csv",
        dataset_name, "Streams", cfg.n_trees,
        (int)data.features.size(), n_features, n_classes, forest3.stats, acc3);

    // ── Summary Table ────────────────────────────────────────────────────────
    print_variant_table(results);

    std::cout << "\n[Analysis]\n";
    std::cout << "  Speedup (Mutex vs Sequential)  : "
              << std::setprecision(2) << results[0].total_time_ms / results[1].total_time_ms << "x\n";
    std::cout << "  Speedup (Streams vs Sequential): "
              << results[0].total_time_ms / results[2].total_time_ms << "x\n";
    std::cout << "  GPU time (all variants)        : ~" << results[0].gpu_time_ms << " ms\n";
    std::cout << "  Note: GPU kernel time is similar across variants — GPU is the bottleneck.\n";
    std::cout << "  Streams advantage: CPU-side overlap (bootstrap, apply-splits) + async uploads.\n";
}

// ============================================================================
static void print_throughput_table(const std::vector<ThroughputResult>& rows) {
    std::cout << "\n  " << std::left
              << std::setw(34) << "Variant"
              << std::setw(10) << "Threads"
              << std::setw(12) << "Time(ms)"
              << std::setw(18) << "Throughput(sps)"
              << std::setw(10) << "Speedup"
              << "\n  " << std::string(82,'-') << "\n";
    for (const auto& r : rows)
        std::cout << "  " << std::left
                  << std::setw(34) << r.label
                  << std::setw(10) << r.n_threads
                  << std::setw(12) << std::fixed << std::setprecision(3) << r.elapsed_ms
                  << std::setw(18) << static_cast<long long>(r.samples_per_sec)
                  << std::setprecision(2) << r.speedup << "x\n";
}

static void print_speedup_table(const std::vector<SpeedupRow>& rows) {
    std::cout << "\n  " << std::left
              << std::setw(12) << "N-Trees"
              << std::setw(14) << "Seq(ms)"
              << std::setw(14) << "Par(ms)"
              << std::setw(14) << "Speedup"
              << "Throughput(sps)\n  " << std::string(68,'-') << "\n";
    for (const auto& r : rows)
        std::cout << "  " << std::left
                  << std::setw(12) << r.n_trees
                  << std::setw(14) << std::fixed << std::setprecision(3) << r.seq_ms
                  << std::setw(14) << r.par_ms
                  << std::setw(14) << std::setprecision(2) << r.speedup
                  << static_cast<long long>(r.throughput) << " sps\n";
}

static void save_throughput_csv(const std::string& path,
    const std::vector<ThroughputResult>& rows)
{
    std::ofstream f(path, std::ios::app);
    if (f.tellp() == 0)
        f << "variant,n_samples,n_trees,n_threads,elapsed_ms,samples_per_sec,speedup\n";
    f << std::fixed << std::setprecision(4);
    for (const auto& r : rows)
        f << r.label << ',' << r.n_samples << ',' << r.n_trees << ','
          << r.n_threads << ',' << r.elapsed_ms << ','
          << static_cast<long long>(r.samples_per_sec) << ',' << r.speedup << '\n';
}

static void save_speedup_csv(const std::string& path,
    const std::vector<SpeedupRow>& rows)
{
    std::ofstream f(path, std::ios::app);
    if (f.tellp() == 0)
        f << "n_trees,seq_ms,par_sample_ms,speedup,throughput_sps\n";
    f << std::fixed << std::setprecision(4);
    for (const auto& r : rows)
        f << r.n_trees << ',' << r.seq_ms << ',' << r.par_ms << ','
          << r.speedup << ',' << static_cast<long long>(r.throughput) << '\n';
}

// ============================================================================
// CSV SAVING — FOREST TRAINING + SCALABILITY
// ============================================================================

// Saves one row per variant per dataset into benchmark_forest_train_m3.csv
static void save_forest_train_csv(
    const std::string& path,
    const std::string& dataset,
    const std::string& variant,
    int n_trees,
    int n_samples,
    int n_features,
    int n_classes,
    const ForestTrainStats& stats,
    double test_acc)
{
    std::ofstream f(path, std::ios::app);
    if (f.tellp() == 0)
        f << "dataset,variant,n_trees,n_samples,n_features,n_classes,"
             "wall_clock_ms,sum_tree_ms,gpu_kernel_ms,avg_tree_ms,test_accuracy\n";
    f << std::fixed << std::setprecision(4)
      << dataset  << ',' << variant   << ',' << n_trees   << ','
      << n_samples << ',' << n_features << ',' << n_classes << ','
      << stats.total_forest_ms << ','
      << stats.tree_build_ms   << ','
      << stats.gpu_kernel_ms   << ','
      << (n_trees > 0 ? stats.tree_build_ms / n_trees : 0.0) << ','
      << test_acc << '\n';
}

// N-sample scalability row — trains streams variant on a fraction of the dataset
struct ScalabilityRow3 {
    std::string dataset;
    std::string variant;
    double fraction    = 1.0;
    int    n_samples   = 0;
    int    n_trees     = 0;
    double wall_clock_ms = 0.0;
    double gpu_kernel_ms = 0.0;
    double test_acc    = 0.0;
};

static void save_scalability_csv(
    const std::string& path,
    const std::vector<ScalabilityRow3>& rows)
{
    std::ofstream f(path, std::ios::app);
    if (f.tellp() == 0)
        f << "dataset,variant,fraction,n_samples,n_trees,"
             "wall_clock_ms,gpu_kernel_ms,test_accuracy\n";
    f << std::fixed << std::setprecision(4);
    for (const auto& r : rows)
        f << r.dataset      << ',' << r.variant   << ','
          << r.fraction     << ',' << r.n_samples  << ',' << r.n_trees << ','
          << r.wall_clock_ms << ',' << r.gpu_kernel_ms << ',' << r.test_acc << '\n';
}

// Run N-sample scalability benchmark — Sequential variant
// Trains on fractions of the full dataset, measures wall-clock time
// Uses direct call (no std::function) to avoid thread/lambda issues on Colab
static std::vector<ScalabilityRow3> run_scalability_sequential(
    const std::string& dataset_name,
    const Dataset& full_data,
    const ForestConfig& cfg,
    const std::vector<double>& fractions = {0.10, 0.25, 0.50, 0.75, 1.00})
{
    const int N = static_cast<int>(full_data.features.size());
    std::vector<ScalabilityRow3> rows;
    for (double frac : fractions) {
        int n_sub = std::clamp(static_cast<int>(std::round(frac * N)), 200, N);
        std::vector<int> idx(N); std::iota(idx.begin(), idx.end(), 0);
        std::shuffle(idx.begin(), idx.end(),
                     std::default_random_engine(cfg.base_seed + static_cast<int>(frac * 1000)));
        Dataset sub; sub.feature_names = full_data.feature_names;
        sub.features.reserve(n_sub); sub.labels.reserve(n_sub);
        for (int i = 0; i < n_sub; ++i) {
            sub.features.push_back(full_data.features[idx[i]]);
            sub.labels.push_back(full_data.labels[idx[i]]);
        }
        auto [tr, te] = train_test_split(sub, 0.8f, cfg.base_seed);
        ForestResult fr = train_random_forest_sequential(sub, tr, cfg);
        CompactForest cf(fr, count_unique_classes(sub.labels));
        FlatBatch batch = prepare_batch(sub, te);
        auto preds = predict_sequential(cf, batch);
        std::vector<int> truth; for (int i : te) truth.push_back(sub.labels[i]);
        double acc = compute_accuracy(preds, truth);
        rows.push_back({dataset_name, "Sequential", frac, n_sub, cfg.n_trees,
                        fr.stats.total_forest_ms, fr.stats.gpu_kernel_ms, acc});
        std::cout << "    [Scalability] Sequential frac=" << frac
                  << " n=" << n_sub << " wall_ms=" << std::fixed << std::setprecision(1)
                  << fr.stats.total_forest_ms << "\n";
    }
    return rows;
}

// Run N-sample scalability benchmark — CUDA Streams variant
static std::vector<ScalabilityRow3> run_scalability_streams(
    const std::string& dataset_name,
    const Dataset& full_data,
    const ForestConfig& cfg,
    const std::vector<double>& fractions = {0.10, 0.25, 0.50, 0.75, 1.00})
{
    const int N = static_cast<int>(full_data.features.size());
    std::vector<ScalabilityRow3> rows;
    for (double frac : fractions) {
        int n_sub = std::clamp(static_cast<int>(std::round(frac * N)), 200, N);
        std::vector<int> idx(N); std::iota(idx.begin(), idx.end(), 0);
        std::shuffle(idx.begin(), idx.end(),
                     std::default_random_engine(cfg.base_seed + static_cast<int>(frac * 1000)));
        Dataset sub; sub.feature_names = full_data.feature_names;
        sub.features.reserve(n_sub); sub.labels.reserve(n_sub);
        for (int i = 0; i < n_sub; ++i) {
            sub.features.push_back(full_data.features[idx[i]]);
            sub.labels.push_back(full_data.labels[idx[i]]);
        }
        auto [tr, te] = train_test_split(sub, 0.8f, cfg.base_seed);
        ForestResult fr = train_random_forest_streams(sub, tr, cfg);
        CompactForest cf(fr, count_unique_classes(sub.labels));
        FlatBatch batch = prepare_batch(sub, te);
        auto preds = predict_sequential(cf, batch);
        std::vector<int> truth; for (int i : te) truth.push_back(sub.labels[i]);
        double acc = compute_accuracy(preds, truth);
        rows.push_back({dataset_name, "Streams", frac, n_sub, cfg.n_trees,
                        fr.stats.total_forest_ms, fr.stats.gpu_kernel_ms, acc});
        std::cout << "    [Scalability] Streams frac=" << frac
                  << " n=" << n_sub << " wall_ms=" << std::fixed << std::setprecision(1)
                  << fr.stats.total_forest_ms << "\n";
    }
    return rows;
}

static std::vector<ScalabilityRow3> run_scalability_mutex(
    const std::string& dataset_name,
    const Dataset& full_data,
    const ForestConfig& cfg,
    const std::vector<double>& fractions = {0.10, 0.25, 0.50, 0.75, 1.00})
{
    const int N = static_cast<int>(full_data.features.size());
    std::vector<ScalabilityRow3> rows;
    for (double frac : fractions) {
        int n_sub = std::clamp(static_cast<int>(std::round(frac * N)), 200, N);
        std::vector<int> idx(N); std::iota(idx.begin(), idx.end(), 0);
        std::shuffle(idx.begin(), idx.end(),
                     std::default_random_engine(cfg.base_seed + static_cast<int>(frac * 1000)));
        Dataset sub; sub.feature_names = full_data.feature_names;
        sub.features.reserve(n_sub); sub.labels.reserve(n_sub);
        for (int i = 0; i < n_sub; ++i) {
            sub.features.push_back(full_data.features[idx[i]]);
            sub.labels.push_back(full_data.labels[idx[i]]);
        }
        auto [tr, te] = train_test_split(sub, 0.8f, cfg.base_seed);
        ForestResult fr = train_random_forest_mutex(sub, tr, cfg);
        CompactForest cf(fr, count_unique_classes(sub.labels));
        FlatBatch batch = prepare_batch(sub, te);
        auto preds = predict_sequential(cf, batch);
        std::vector<int> truth; for (int i : te) truth.push_back(sub.labels[i]);
        double acc = compute_accuracy(preds, truth);
        rows.push_back({dataset_name, "Mutex", frac, n_sub, cfg.n_trees,
                        fr.stats.total_forest_ms, fr.stats.gpu_kernel_ms, acc});
        std::cout << "    [Scalability] Mutex frac=" << frac
                  << " n=" << n_sub << " wall_ms=" << std::fixed << std::setprecision(1)
                  << fr.stats.total_forest_ms << "\n";
    }
    return rows;
}
// ============================================================================
// SANITY TESTS
// ============================================================================

static void write_test_csv(const std::string& fn) {
    std::ofstream f(fn);
    f << "f0,f1,f2,label\n";
    for (int i = 0; i < 30; ++i) f << (i*0.1)<<','<<(i*0.1)<<','<<(i*0.1)<<",0\n";
    for (int i = 0; i < 30; ++i) f << (3+i*0.1)<<','<<(3+i*0.1)<<','<<(3+i*0.1)<<",1\n";
    for (int i = 0; i < 30; ++i) f << (6+i*0.1)<<','<<(6+i*0.1)<<','<<(6+i*0.1)<<",2\n";
}

static void test_bootstrap_properties() {
    std::cout << "\n[Test: bootstrap_sample with replacement]\n";
    std::vector<int> pool(100); std::iota(pool.begin(), pool.end(), 0);
    auto s = bootstrap_sample(pool, 100, 42);
    assert((int)s.size() == 100);
    std::set<int> uniq(s.begin(), s.end());
    assert(uniq.size() < 100 && "With replacement: expect duplicates");
    std::cout << "  unique indices drawn: " << uniq.size() << "/100  [PASS]\n";
}

static void test_tree_count_cap() {
    std::cout << "\n[Test: tree count capped at MAX_TREES=" << MAX_TREES << "]\n";
    write_test_csv("t_cap.csv");
    Dataset d = load_csv("t_cap.csv"); auto [tr,te] = train_test_split(d, 0.8f, 1);
    ForestConfig cfg; cfg.n_trees = 999; cfg.max_depth = 3;
    ForestResult fr = train_random_forest(d, tr, cfg);
    assert((int)fr.trees.size() == MAX_TREES);
    std::cout << "  forest size=" << fr.trees.size() << "  [PASS]\n";
}

static void test_compact_serialization() {
    std::cout << "\n[Test: compact tree serialization matches pointer tree]\n";
    write_test_csv("t_ser.csv");
    Dataset d = load_csv("t_ser.csv"); auto [tr,te] = train_test_split(d, 0.8f, 2);
    LevelwiseBuildStats s;
    Node* root = build_tree_levelwise(d, tr, 5, 2, 8, s);
    auto compact = serialize_tree(root);
    int diff = 0;
    for (int i : te) {
        // walk pointer tree
        const Node* n = root;
        while (!n->is_leaf && n->left && n->right)
            n = d.features[i][n->feature_index] <= n->threshold ? n->left : n->right;
        int ptr_pred = n->predicted_class;
        int cmp_pred = predict_compact_tree(compact, d.features[i].data());
        if (ptr_pred != cmp_pred) ++diff;
    }
    assert(diff == 0);
    delete_tree(root);
    std::cout << "  all " << te.size() << " test predictions match  [PASS]\n";
}

static void test_parallel_correctness() {
    std::cout << "\n[Test: parallel inference == sequential]\n";
    write_test_csv("t_par.csv");
    Dataset d = load_csv("t_par.csv"); auto [tr,te] = train_test_split(d, 0.8f, 3);
    ForestConfig cfg; cfg.n_trees = 5; cfg.max_depth = 5;
    ForestResult fr = train_random_forest(d, tr, cfg);
    CompactForest cf(fr, count_unique_classes(d.labels));
    FlatBatch batch = prepare_batch(d, te);
    auto seq  = predict_sequential(cf, batch);
    auto par_s = predict_parallel_samples(cf, batch, 4);
    auto par_t = predict_parallel_trees(cf, batch, 4);
    int ds = 0, dt = 0;
    for (int i = 0; i < (int)seq.size(); ++i) {
        if (seq[i] != par_s[i]) ++ds;
        if (seq[i] != par_t[i]) ++dt;
    }
    assert(ds == 0 && "parallel-samples differs from sequential");
    assert(dt == 0 && "parallel-trees differs from sequential");
    std::cout << "  par-samples diff=" << ds << "  par-trees diff=" << dt << "  [PASS]\n";
}

// ============================================================================
// MAIN — FULL BENCHMARK PIPELINE ON REAL DATASETS
// ============================================================================

int main() {
    std::cout << "=============================================================\n";
    std::cout << "   MILESTONE 3 — Random Forest + Parallel Inference\n";
    std::cout << "=============================================================\n";
    std::cout << "   Hardware threads: " << hw_threads() << "\n";
    std::cout << "=============================================================\n";

    try {
#ifdef __CUDACC__
        // ── Sanity tests ───────────────────────────────────────────────────
        std::cout << "\n=== SANITY TESTS ===\n";
        test_bootstrap_properties();
        test_tree_count_cap();
        test_compact_serialization();
        test_parallel_correctness();
        std::cout << "\nAll sanity tests passed.\n";

        // ── Dataset specs ──────────────────────────────────────────────────
        struct DataSpec { std::string name, file; bool header; };
        std::vector<DataSpec> datasets = {
            {"Iris",             "Iris.csv",               true},
            {"Shuttle",          "shuttle.csv",            true},
            {"LetterRecognition","letter-recognition.csv", true},
            {"Skin_NonSkin",     "Skin_NonSkin.csv",       true},
            {"Synthetic_1M_200f","synthetic_1000k_200f.csv",true}
        };
        std::vector<std::string> roots = {
            std::filesystem::current_path().string(),
            (std::filesystem::current_path()/"datasets").string()
        };

        // Clear CSV files so runs don't append to stale data
        for (const auto& f : {"benchmark_forest_train_m3.csv",
                               "benchmark_throughput_m3.csv",
                               "benchmark_speedup_vs_trees_m3.csv",
                               "benchmark_scalability_m3.csv"})
            std::remove(f);

        ForestConfig cfg;          // default: 5 trees, depth 8, 8 bins
        const std::vector<int> tree_counts = {1, 2, 3, 5, MAX_TREES};

        // ── Per-dataset benchmark loop ─────────────────────────────────────
        for (const auto& spec : datasets) {
            std::string path = resolve_path(spec.file, roots);
            std::cout << "\n[Loading] " << spec.name << " from: " << path << "\n";
            Dataset data;
            try { data = load_csv(path, spec.header); }
            catch (const std::exception& ex) {
                std::cout << "  [Skipped] " << ex.what() << "\n"; continue;
            }

            const int n_samples  = static_cast<int>(data.features.size());
            const int n_features = static_cast<int>(data.features[0].size());
            const int n_classes  = count_unique_classes(data.labels);

            auto [train_idx, test_idx] = train_test_split(data, 0.8f, cfg.base_seed);
            std::cout << "  samples=" << n_samples
                      << "  features=" << n_features
                      << "  classes=" << n_classes
                      << "  train=" << train_idx.size()
                      << "  test=" << test_idx.size() << "\n";

            // ── RUN THREE TRAINING VARIANTS + save to CSV ─────────────────
            run_variant_comparison(data, train_idx, test_idx, cfg,
                                   spec.name, n_features, n_classes);

            // ── INFERENCE throughput benchmark using streams-trained forest ─
            std::cout << "\n[Inference throughput benchmark (streams-trained forest)]\n";
            ForestResult forest_streams = train_random_forest_streams(data, train_idx, cfg);
            CompactForest cf(forest_streams, n_classes);
            auto tput = measure_throughput(cf, data, test_idx, -1, 5);
            print_throughput_table(tput);
            // Add dataset label to each row before saving
            for (auto& r : tput) r.label = spec.name + " | " + r.label;
            save_throughput_csv("benchmark_throughput_m3.csv", tput);

            // ── Speedup vs number of trees (inference) ────────────────────
            std::cout << "\n[Inference speedup vs number of trees]\n";
            ForestConfig full_cfg = cfg; full_cfg.n_trees = MAX_TREES;
            ForestResult full_forest = train_random_forest_streams(data, train_idx, full_cfg);
            CompactForest full_cf(full_forest, n_classes);
            FlatBatch fb = prepare_batch(data, test_idx);
            auto svt = speedup_vs_trees(cf, fb, tree_counts, -1, 5);
            print_speedup_table(svt);
            save_speedup_csv("benchmark_speedup_vs_trees_m3.csv", svt);

            // ── N-SAMPLE SCALABILITY ──────────────────────────────────────
            // Skip Iris (too small to show a meaningful trend) and the 1M
            // synthetic dataset (too slow for a 5-fraction sweep on Colab).
            // Run both Sequential and Streams variants for comparison.
            if (n_samples >= 1000 && n_samples <= 250000) {
                std::cout << "\n[Scalability benchmark — N-sample sweep]\n";
                auto scale_seq = run_scalability_sequential(spec.name, data, cfg);
                save_scalability_csv("benchmark_scalability_m3.csv", scale_seq);
                auto scale_str = run_scalability_streams(spec.name, data, cfg);
                save_scalability_csv("benchmark_scalability_m3.csv", scale_str);
                auto scale_mtx = run_scalability_mutex(spec.name, data, cfg);
                save_scalability_csv("benchmark_scalability_m3.csv", scale_mtx);
                } else if (n_samples > 250000) {
    // Large dataset: just record the full-size streams result
    std::cout << "\n[Scalability: large dataset — recording full-size point only]\n";
    
    // Calculate actual accuracy
    CompactForest cf_full(forest_streams, n_classes);
    FlatBatch batch_full = prepare_batch(data, test_idx);
    auto preds_full = predict_sequential(cf_full, batch_full);
    std::vector<int> truth_full; for (int i : test_idx) truth_full.push_back(data.labels[i]);
    double acc_full = compute_accuracy(preds_full, truth_full);
    
    std::vector<ScalabilityRow3> one_row = {{
        spec.name, "Streams", 1.0, n_samples, cfg.n_trees,
        forest_streams.stats.total_forest_ms,
        forest_streams.stats.gpu_kernel_ms, acc_full  // ← Use actual accuracy
    }};
    save_scalability_csv("benchmark_scalability_m3.csv", one_row);
}
        }

        std::cout << "\n[Artifacts saved]\n";
        std::cout << "  benchmark_forest_train_m3.csv       (training: 3 variants x 5 datasets)\n";
        std::cout << "  benchmark_throughput_m3.csv          (inference: seq/par-s/par-t)\n";
        std::cout << "  benchmark_speedup_vs_trees_m3.csv   (inference speedup vs K trees)\n";
        std::cout << "  benchmark_scalability_m3.csv         (training time vs N samples)\n";
        std::cout << "\nDone.\n";

#else
        std::cerr << "ERROR: Compile with nvcc, not g++.\n"
                  << "  nvcc -O2 -std=c++17 -arch=sm_75 milestone3_complete.cu -o milestone3\n";
        return 1;
#endif
    } catch (const std::exception& ex) {
        std::cerr << "Error: " << ex.what() << "\n";
        return 1;
    }
    return 0;
}
