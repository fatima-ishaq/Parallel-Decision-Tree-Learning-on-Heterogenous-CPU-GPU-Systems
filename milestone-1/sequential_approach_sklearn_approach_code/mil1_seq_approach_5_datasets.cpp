// ===================================================
// MILESTONE 1 — Sequential CPU Decision Tree Builder
// ===================================================

#include <algorithm>
#include <cassert>
#include <cctype>
#include <cmath>
#include <cstdlib>
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
#include <tuple>
#include <vector>
#include <chrono>
#include <filesystem>

// ── Timing accumulators (global) ─────────────────────────────────────────
double g_split_eval_time_sec = 0.0;

// =======================================
// ── FARAZ'S DATA STRUCTURES & UTILITIES
// =======================================

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
    bool has_header;
};

struct BenchmarkResult {
    std::string dataset_name;
    int n_samples = 0;
    int n_features = 0;
    int n_classes = 0;
    int train_samples = 0;
    int test_samples = 0;
    double train_time_sec = 0.0;
    double split_eval_sec = 0.0;
    double tree_overhead_sec = 0.0;
    double predict_time_sec = 0.0;
    double infer_us_per_sample = 0.0;
    double train_acc = 0.0;
    double test_acc = 0.0;
};

struct ScalabilityPoint {
    std::string dataset_name;
    double fraction = 1.0;
    int n_samples = 0;
    double train_time_sec = 0.0;
    double predict_time_sec = 0.0;
    double test_acc = 0.0;
};

static std::string resolve_dataset_path(
    const std::string& file_name,
    const std::vector<std::string>& search_roots)
{
    namespace fs = std::filesystem;
    fs::path direct(file_name);
    if (fs::exists(direct)) return direct.string();

    for (const auto& root : search_roots) {
        fs::path candidate = fs::path(root) / file_name;
        if (fs::exists(candidate)) return candidate.string();
    }
    return file_name;
}

// Node structure — Faraz defined this; Yaman FILLS and USES it.
struct Node {
    std::vector<int> sample_indices;   // which training samples live here
    int              depth;
    int              feature_index;    // split feature (-1 if leaf)
    float            threshold;        // split threshold
    Node*            left;
    Node*            right;
    bool             is_leaf;
    int              predicted_class;  // majority class at this node

    Node(int d = 0)
        : depth(d), feature_index(-1), threshold(0.0f),
          left(nullptr), right(nullptr),
          is_leaf(false), predicted_class(-1) {}
};

// ── Faraz: load CSV (last column = label) ───────────────────────────────────
Dataset load_csv(const std::string& filename, bool has_header = true) {
    Dataset data;
    std::ifstream file(filename);
    if (!file.is_open()) {
        std::cerr << "[!] Cannot open: " << filename << "\n";
        return data;
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

    std::map<std::string, int> label_encoder;
    int next_label_id = 0;

    std::string line;
    int line_number = 0;
    while (std::getline(file, line)) {
        ++line_number;
        if (line.empty()) continue;
        if (has_header && line_number == 1) {
            std::stringstream ss(line);
            std::string item;
            while (std::getline(ss, item, ','))
                data.feature_names.push_back(item);
            if (!data.feature_names.empty())
                data.feature_names.pop_back();
            continue;
        }
        std::stringstream ss(line);
        std::string value;
        std::vector<std::string> tokens;
        while (std::getline(ss, value, ',')) {
            tokens.push_back(trim(value));
        }

        if (tokens.size() < 2) continue;

        bool first_is_label = !is_numeric(tokens.front());
        int label_idx = first_is_label ? 0 : static_cast<int>(tokens.size()) - 1;

        std::vector<float> row;
        row.reserve(tokens.size() - 1);
        for (int i = 0; i < static_cast<int>(tokens.size()); ++i) {
            if (i == label_idx) continue;
            if (!is_numeric(tokens[i])) {
                row.clear();
                break;
            }
            row.push_back(static_cast<float>(std::stod(tokens[i])));
        }
        if (row.empty()) continue;

        int label_id = 0;
        if (is_numeric(tokens[label_idx])) {
            label_id = static_cast<int>(std::lround(std::stod(tokens[label_idx])));
        } else {
            auto it = label_encoder.find(tokens[label_idx]);
            if (it == label_encoder.end()) {
                label_encoder[tokens[label_idx]] = next_label_id;
                label_id = next_label_id;
                ++next_label_id;
            } else {
                label_id = it->second;
            }
        }

        data.features.push_back(row);
        data.labels.push_back(label_id);
    }
    return data;
}

static bool validate_dataset(const Dataset& data, const std::string& name) {
    if (data.features.empty() || data.labels.empty()) {
        std::cerr << "[!] Dataset '" << name << "' is empty.\n";
        return false;
    }
    if (data.features.size() != data.labels.size()) {
        std::cerr << "[!] Dataset '" << name << "' has mismatched features/labels sizes.\n";
        return false;
    }
    const std::size_t nf = data.features[0].size();
    if (nf == 0) {
        std::cerr << "[!] Dataset '" << name << "' has zero feature columns.\n";
        return false;
    }
    for (std::size_t i = 1; i < data.features.size(); ++i) {
        if (data.features[i].size() != nf) {
            std::cerr << "[!] Dataset '" << name << "' has inconsistent feature counts across rows.\n";
            return false;
        }
    }
    return true;
}

static int count_unique_classes(const std::vector<int>& labels) {
    std::map<int, int> counts;
    for (int y : labels) counts[y]++;
    return static_cast<int>(counts.size());
}

static Dataset subset_dataset(const Dataset& data, int subset_size, int seed) {
    Dataset out;
    out.feature_names = data.feature_names;
    int n = static_cast<int>(data.features.size());
    if (subset_size <= 0 || subset_size >= n) return data;

    std::vector<int> idx(n);
    std::iota(idx.begin(), idx.end(), 0);
    std::shuffle(idx.begin(), idx.end(), std::default_random_engine(seed));

    out.features.reserve(subset_size);
    out.labels.reserve(subset_size);
    for (int i = 0; i < subset_size; ++i) {
        out.features.push_back(data.features[idx[i]]);
        out.labels.push_back(data.labels[idx[i]]);
    }
    return out;
}

// ── Faraz: train/test split by indices ──────────────────────────────────────
std::pair<std::vector<int>, std::vector<int>>
train_test_split(const Dataset& data, float train_ratio = 0.8f, int seed = 42) {
    int n = static_cast<int>(data.features.size());
    std::vector<int> idx(n);
    std::iota(idx.begin(), idx.end(), 0);
    std::shuffle(idx.begin(), idx.end(), std::default_random_engine(seed));
    int train_size = static_cast<int>(n * train_ratio);
    return { std::vector<int>(idx.begin(), idx.begin() + train_size),
             std::vector<int>(idx.begin() + train_size, idx.end()) };
}

// ── Faraz: create root node ──────────────────────────────────────────────────
Node* create_root_node(const std::vector<int>& train_indices) {
    Node* root = new Node(0);
    root->sample_indices = train_indices;
    return root;
}

// ── Faraz: get samples at a node ────────────────────────────────────────────
const std::vector<int>& get_samples_for_node(Node* node) {
    return node->sample_indices;
}

// ── Faraz: split a node into two children ───────────────────────────────────
std::pair<Node*, Node*> split_node(Node* parent,
                                   const std::vector<int>& left_indices,
                                   const std::vector<int>& right_indices) {
    Node* left  = new Node(parent->depth + 1);
    Node* right = new Node(parent->depth + 1);
    left->sample_indices  = left_indices;
    right->sample_indices = right_indices;
    parent->left  = left;
    parent->right = right;
    return {left, right};
}

// =========================
// ── FATIMA'S SPLIT-FINDER
// =========================

struct SplitResult {
    int    best_feature   = -1;
    int    best_bin_index = -1;
    double best_threshold = 0.0;
    double best_gini      = std::numeric_limits<double>::infinity();
    bool   found          = false;
};

static double gini_from_counts(const std::vector<int>& counts) {
    int total = 0;
    for (int c : counts) total += c;
    if (total == 0) return 0.0;
    double sum_sq = 0.0;
    for (int c : counts) {
        double p = static_cast<double>(c) / total;
        sum_sq += p * p;
    }
    return 1.0 - sum_sq;
}

static int find_num_classes(const std::vector<int>& y) {
    int mx = -1;
    for (int v : y) {
        if (v < 0) throw std::invalid_argument("Negative label.");
        if (v > mx) mx = v;
    }
    return mx + 1;
}

static std::vector<double> equal_width_boundaries(const std::vector<double>& vals, int bins) {
    auto [mn_it, mx_it] = std::minmax_element(vals.begin(), vals.end());
    double mn = *mn_it, mx = *mx_it;
    std::vector<double> b(bins + 1);
    if (mn == mx) { for (auto& x : b) x = mn; return b; }
    double w = (mx - mn) / bins;
    for (int i = 0; i <= bins; ++i) b[i] = mn + i * w;
    b.back() = mx;
    return b;
}

static int val_to_bin(double x, const std::vector<double>& b) {
    int bc = static_cast<int>(b.size()) - 1;
    if (x <= b.front()) return 0;
    if (x >= b.back())  return bc - 1;
    auto it = std::upper_bound(b.begin(), b.end(), x);
    int idx = static_cast<int>(it - b.begin()) - 1;
    return std::clamp(idx, 0, bc - 1);
}

SplitResult find_best_histogram_split(
    const std::vector<std::vector<double>>& X,
    const std::vector<int>& y,
    int bin_count)
{
    if (X.empty() || X.size() != y.size() || bin_count < 2) return {};
    int n  = static_cast<int>(X.size());
    int nf = static_cast<int>(X[0].size());
    int nc = find_num_classes(y);
    if (nc < 2) return {};

    SplitResult best;
    for (int f = 0; f < nf; ++f) {
        std::vector<double> fv(n);
        for (int i = 0; i < n; ++i) fv[i] = X[i][f];
        auto bounds = equal_width_boundaries(fv, bin_count);
        int bc = bin_count;
        std::vector<std::vector<int>> hist(bc, std::vector<int>(nc, 0));
        for (int i = 0; i < n; ++i)
            hist[val_to_bin(fv[i], bounds)][y[i]]++;

        std::vector<int> lc(nc, 0), tot(nc, 0);
        int ln = 0, tn = 0;
        for (int b2 = 0; b2 < bc; ++b2)
            for (int c = 0; c < nc; ++c) { tot[c] += hist[b2][c]; tn += hist[b2][c]; }
        auto rc = tot; int rn = tn;

        for (int sb = 0; sb < bc - 1; ++sb) {
            for (int c = 0; c < nc; ++c) {
                lc[c] += hist[sb][c]; rc[c] -= hist[sb][c];
                ln    += hist[sb][c]; rn    -= hist[sb][c];
            }
            if (ln == 0 || rn == 0) continue;
            double wg = (double)ln/tn * gini_from_counts(lc)
                      + (double)rn/tn * gini_from_counts(rc);
            if (wg < best.best_gini) {
                best.best_gini      = wg;
                best.best_feature   = f;
                best.best_bin_index = sb;
                best.best_threshold = bounds[sb + 1];
                best.found          = true;
            }
        }
    }
    return best;
}

// =========================
// ══ YAMAN'S COMPONENTS ══
// =========================

// ----------------------------------------------------------------------------
// Y-1. GINI IMPURITY FOR A NODE
//      Computes the Gini impurity of the label distribution at one node.
//      Pure node → 0.0   Maximally mixed → close to 1.0
//      Used to report node quality and to verify splits reduce impurity.
// ----------------------------------------------------------------------------
double node_gini(const std::vector<int>& indices, const std::vector<int>& labels) {
    if (indices.empty()) return 0.0;
    std::map<int, int> counts;
    for (int idx : indices) counts[labels[idx]]++;
    int total = static_cast<int>(indices.size());
    double sum_sq = 0.0;
    for (const auto& [cls, cnt] : counts) {
        double p = static_cast<double>(cnt) / total;
        sum_sq += p * p;
    }
    return 1.0 - sum_sq;
}

// ----------------------------------------------------------------------------
// Y-2. MAJORITY CLASS
//      Returns the most frequent label among the samples at a node.
//      This becomes the node's predicted_class — used both for leaves
//      and as a fallback if traversal somehow reaches a non-leaf.
// ----------------------------------------------------------------------------
int majority_class(const std::vector<int>& indices, const std::vector<int>& labels) {
    std::map<int, int> counts;
    for (int idx : indices) counts[labels[idx]]++;
    return std::max_element(counts.begin(), counts.end(),
        [](const auto& a, const auto& b){ return a.second < b.second; })->first;
}

// ----------------------------------------------------------------------------
// Y-3. PURE-NODE CHECK
//      A node is pure if every sample in it shares the same class label.
//      Pure nodes must become leaves — splitting them further is pointless.
// ----------------------------------------------------------------------------
bool is_pure(const std::vector<int>& indices, const std::vector<int>& labels) {
    if (indices.empty()) return true;
    int first = labels[indices[0]];
    for (int idx : indices)
        if (labels[idx] != first) return false;
    return true;
}

// ----------------------------------------------------------------------------
// Y-4. BUILD DECISION TREE  ← Yaman's core contribution
//
//      Recursively grows the decision tree from a given node downwards.
//
//      Algorithm:
//        1. Record majority class at this node (always, even for internal nodes)
//        2. Check stopping conditions:
//              a. Node is pure → make leaf
//              b. Depth reached max_depth → make leaf
//              c. Too few samples to split → make leaf
//        3. Call Fatima's find_best_histogram_split() on the samples here
//        4. If no valid split found → make leaf
//        5. Partition sample indices: left (< threshold), right (> = threshold)
//        6. Call Faraz's split_node() to attach children
//        7. Recurse into left and right children
//
//      Parameters:
//        node              – current node (already has sample_indices set)
//        data              – full dataset (Faraz's Dataset struct)
//        max_depth         – maximum tree depth (prevents overfitting)
//        min_samples_split – don't split if node has fewer samples than this
//        bin_count         – number of histogram bins for Fatima's finder
// ----------------------------------------------------------------------------
void build_tree(
    Node*           node,
    const Dataset&  data,
    int             max_depth,
    int             min_samples_split,
    int             bin_count)
{
    // ── Step 1: Always record majority class ─────────────────────────────────
    const std::vector<int>& indices = get_samples_for_node(node);
    node->predicted_class = majority_class(indices, data.labels);

    // ── Step 2: Stopping conditions ──────────────────────────────────────────
    if (is_pure(indices, data.labels) ||
        node->depth >= max_depth      ||
        (int)indices.size() < min_samples_split)
    {
        node->is_leaf = true;
        return;
    }

    // ── Step 3: Build X / y sub-arrays for Fatima's split-finder ─────────────
    int nf = static_cast<int>(data.features[0].size());
    std::vector<std::vector<double>> X(indices.size(), std::vector<double>(nf));
    std::vector<int> y(indices.size());
    for (int i = 0; i < (int)indices.size(); ++i) {
        int s = indices[i];
        for (int f = 0; f < nf; ++f)
            X[i][f] = static_cast<double>(data.features[s][f]);
        y[i] = data.labels[s];
    }

    // ── Step 4: Find best split (Fatima's function) ───────────────────────────
    auto split_start = std::chrono::high_resolution_clock::now();
    SplitResult sr = find_best_histogram_split(X, y, bin_count);
    auto split_end = std::chrono::high_resolution_clock::now();
    g_split_eval_time_sec += std::chrono::duration<double>(split_end - split_start).count();

    if (!sr.found) {
        node->is_leaf = true;
        return;
    }

    // ── Step 5: Record split info in node ────────────────────────────────────
    node->feature_index = sr.best_feature;
    node->threshold     = static_cast<float>(sr.best_threshold);

    // ── Step 6: Partition sample indices ─────────────────────────────────────
    std::vector<int> left_idx, right_idx;
    for (int s : indices) {
        if (data.features[s][sr.best_feature] < node->threshold)
            left_idx.push_back(s);
        else
            right_idx.push_back(s);
    }

    // Guard: degenerate split → make leaf
    if (left_idx.empty() || right_idx.empty()) {
        node->is_leaf = true;
        return;
    }

    // ── Step 7: Create children (Faraz's split_node) and recurse ─────────────
    auto [left_child, right_child] = split_node(node, left_idx, right_idx);
    build_tree(left_child,  data, max_depth, min_samples_split, bin_count);
    build_tree(right_child, data, max_depth, min_samples_split, bin_count);
}

// -------------------------------------
// ZUHAA'S PREDICTION AND EVALUATION
// PREDICT — single sample
// -------------------------------------
int predict(const Node* node, const std::vector<float>& sample) {
    // Leaf or safety fallback
    if (node->is_leaf || node->left == nullptr || node->right == nullptr)
        return node->predicted_class;

    if (sample[node->feature_index] < node->threshold)
        return predict(node->left,  sample);
    else
        return predict(node->right, sample);
}

// -------------------------------------
// PREDICT BATCH — all test-set samples
// -------------------------------------
std::vector<int> predict_batch(
    const Node*             root,
    const Dataset&          data,
    const std::vector<int>& indices)
{
    std::vector<int> preds;
    preds.reserve(indices.size());
    for (int idx : indices)
        preds.push_back(predict(root, data.features[idx]));
    return preds;
}

// ---------
// ACCURACY
// ---------
double compute_accuracy(
    const std::vector<int>& predictions,
    const std::vector<int>& true_labels)
{
    if (predictions.size() != true_labels.size() || predictions.empty())
        throw std::invalid_argument("Size mismatch or empty vectors.");
    int correct = 0;
    for (int i = 0; i < (int)predictions.size(); ++i)
        if (predictions[i] == true_labels[i]) ++correct;
    return static_cast<double>(correct) / static_cast<double>(predictions.size());
}

// -------------
// TREE PRINTER
// -------------
void print_tree(
    const Node*             node,
    const Dataset&          data,
    const std::vector<std::string>& feat_names,
    const std::string&      prefix = "",
    bool                    is_left = true)
{
    if (!node) return;

    std::string connector = prefix.empty() ? "" : (is_left ? "├── " : "└── ");
    std::string new_prefix = prefix + (prefix.empty() ? "" : (is_left ? "│   " : "    "));

    int n = static_cast<int>(node->sample_indices.size());
    double g = node_gini(node->sample_indices, data.labels);

    if (node->is_leaf || (!node->left && !node->right)) {
        std::cout << prefix << connector
                  << "[LEAF] class=" << node->predicted_class
                  << "  samples=" << n
                  << "  gini=" << std::fixed << std::setprecision(4) << g
                  << "\n";
    } else {
        std::string fname = (node->feature_index >= 0 &&
                             node->feature_index < (int)feat_names.size())
                            ? feat_names[node->feature_index]
                            : "f" + std::to_string(node->feature_index);
        std::cout << prefix << connector
                  << "[depth " << node->depth << "]  "
                  << fname << " < " << std::fixed << std::setprecision(4) << node->threshold
                  << "  |  samples=" << n
                  << "  gini=" << g
                  << "\n";
        print_tree(node->left,  data, feat_names, new_prefix, true);
        print_tree(node->right, data, feat_names, new_prefix, false);
    }
}

// ----------------
// TREE STATISTICS
// ----------------
struct TreeStats { int leaves = 0; int internals = 0; int max_depth = 0; };

void collect_stats(const Node* node, TreeStats& s) {
    if (!node) return;
    if (node->depth > s.max_depth) s.max_depth = node->depth;
    if (node->is_leaf || (!node->left && !node->right)) {
        ++s.leaves;
    } else {
        ++s.internals;
        collect_stats(node->left,  s);
        collect_stats(node->right, s);
    }
}

void print_tree_stats(const Node* root) {
    TreeStats s;
    collect_stats(root, s);
    std::cout << "\n[Tree Statistics]\n";
    std::cout << "  Leaf nodes    : " << s.leaves    << "\n";
    std::cout << "  Internal nodes: " << s.internals << "\n";
    std::cout << "  Max depth     : " << s.max_depth << "\n";
    std::cout << "  Total nodes   : " << (s.leaves + s.internals) << "\n";
}

// ----------------
// MEMORY CLEANUP
// ----------------
void delete_tree(Node* node) {
    if (!node) return;
    delete_tree(node->left);
    delete_tree(node->right);
    delete node;
}

static BenchmarkResult run_benchmark_on_dataset(const DatasetSpec& spec, const Dataset& data, const RunConfig& cfg) {
    BenchmarkResult r;
    r.dataset_name = spec.name;
    r.n_samples = static_cast<int>(data.features.size());
    r.n_features = static_cast<int>(data.features[0].size());
    r.n_classes = count_unique_classes(data.labels);

    auto [train_idx, test_idx] = train_test_split(data, cfg.train_ratio, cfg.seed);
    r.train_samples = static_cast<int>(train_idx.size());
    r.test_samples = static_cast<int>(test_idx.size());

    Node* root = create_root_node(train_idx);

    g_split_eval_time_sec = 0.0;
    auto train_start = std::chrono::high_resolution_clock::now();
    build_tree(root, data, cfg.max_depth, cfg.min_samples_split, cfg.bin_count);
    auto train_end = std::chrono::high_resolution_clock::now();
    r.train_time_sec = std::chrono::duration<double>(train_end - train_start).count();
    r.split_eval_sec = g_split_eval_time_sec;
    r.tree_overhead_sec = std::max(0.0, r.train_time_sec - r.split_eval_sec);

    auto train_preds = predict_batch(root, data, train_idx);
    auto test_start = std::chrono::high_resolution_clock::now();
    auto test_preds = predict_batch(root, data, test_idx);
    auto test_end = std::chrono::high_resolution_clock::now();
    r.predict_time_sec = std::chrono::duration<double>(test_end - test_start).count();

    std::vector<int> train_true;
    std::vector<int> test_true;
    train_true.reserve(train_idx.size());
    test_true.reserve(test_idx.size());
    for (int i : train_idx) train_true.push_back(data.labels[i]);
    for (int i : test_idx) test_true.push_back(data.labels[i]);

    r.train_acc = compute_accuracy(train_preds, train_true);
    r.test_acc = compute_accuracy(test_preds, test_true);
    r.infer_us_per_sample = (r.test_samples > 0)
        ? (r.predict_time_sec * 1e6 / static_cast<double>(r.test_samples))
        : 0.0;

    delete_tree(root);
    return r;
}

static std::vector<ScalabilityPoint> run_scalability(
    const DatasetSpec& spec,
    const Dataset& full_data,
    const RunConfig& cfg,
    const std::vector<double>& fractions)
{
    std::vector<ScalabilityPoint> out;
    int n_total = static_cast<int>(full_data.features.size());
    for (double frac : fractions) {
        int n_sub = static_cast<int>(std::round(frac * n_total));
        n_sub = std::clamp(n_sub, 200, n_total);
        Dataset sub = subset_dataset(full_data, n_sub, cfg.seed + static_cast<int>(frac * 1000));

        BenchmarkResult r = run_benchmark_on_dataset(spec, sub, cfg);
        ScalabilityPoint p;
        p.dataset_name = spec.name;
        p.fraction = frac;
        p.n_samples = r.n_samples;
        p.train_time_sec = r.train_time_sec;
        p.predict_time_sec = r.predict_time_sec;
        p.test_acc = r.test_acc;
        out.push_back(p);
    }
    return out;
}

static void save_metrics_csv(const std::string& path, const std::vector<BenchmarkResult>& rows, const RunConfig& cfg) {
    std::ofstream out(path);
    out << "dataset,n_samples,n_features,n_classes,train_samples,test_samples,max_depth,min_samples_split,bin_count,train_time_sec,split_eval_sec,tree_overhead_sec,predict_time_sec,infer_time_per_sample_us,train_accuracy,test_accuracy\n";
    out << std::fixed << std::setprecision(6);
    for (const auto& r : rows) {
        out << r.dataset_name << ","
            << r.n_samples << ","
            << r.n_features << ","
            << r.n_classes << ","
            << r.train_samples << ","
            << r.test_samples << ","
            << cfg.max_depth << ","
            << cfg.min_samples_split << ","
            << cfg.bin_count << ","
            << r.train_time_sec << ","
            << r.split_eval_sec << ","
            << r.tree_overhead_sec << ","
            << r.predict_time_sec << ","
            << r.infer_us_per_sample << ","
            << r.train_acc << ","
            << r.test_acc << "\n";
    }
}

static void save_scalability_csv(const std::string& path, const std::vector<ScalabilityPoint>& rows) {
    std::ofstream out(path);
    out << "dataset,fraction,n_samples,train_time_sec,predict_time_sec,test_accuracy\n";
    out << std::fixed << std::setprecision(6);
    for (const auto& r : rows) {
        out << r.dataset_name << ","
            << r.fraction << ","
            << r.n_samples << ","
            << r.train_time_sec << ","
            << r.predict_time_sec << ","
            << r.test_acc << "\n";
    }
}

static void print_dataset_summary(const BenchmarkResult& r) {
    std::cout << "\n[Dataset: " << r.dataset_name << "]\n";
    std::cout << "  samples/features/classes : " << r.n_samples << " / " << r.n_features << " / " << r.n_classes << "\n";
    std::cout << "  train/test samples       : " << r.train_samples << " / " << r.test_samples << "\n";
    std::cout << std::fixed << std::setprecision(6);
    std::cout << "  train_time_sec           : " << r.train_time_sec << "\n";
    std::cout << "  split_eval_sec           : " << r.split_eval_sec << "\n";
    std::cout << "  tree_overhead_sec        : " << r.tree_overhead_sec << "\n";
    std::cout << "  predict_time_sec         : " << r.predict_time_sec << "\n";
    std::cout << "  infer_us_per_sample      : " << r.infer_us_per_sample << "\n";
    std::cout << "  train_acc                : " << r.train_acc * 100.0 << "%\n";
    std::cout << "  test_acc                 : " << r.test_acc * 100.0 << "%\n";
}

static void print_scalability_comment(const std::string& dataset_name, const std::vector<ScalabilityPoint>& points) {
    if (points.size() < 2) return;
    const auto& first = points.front();
    const auto& last = points.back();

    double size_ratio = static_cast<double>(last.n_samples) / std::max(1, first.n_samples);
    double train_ratio = last.train_time_sec / std::max(1e-9, first.train_time_sec);
    double pred_ratio = last.predict_time_sec / std::max(1e-9, first.predict_time_sec);
    double acc_delta_pp = (last.test_acc - first.test_acc) * 100.0;

    std::cout << "\n[Scalability Comment: " << dataset_name << "]\n";
    std::cout << std::fixed << std::setprecision(3);
    std::cout << "  Size x" << size_ratio
              << " => Train time x" << train_ratio
              << ", Predict time x" << pred_ratio << "\n";
    std::cout << "  Test accuracy change from smallest to largest subset: "
              << acc_delta_pp << " percentage points.\n";

    if (train_ratio <= size_ratio * 1.2)
        std::cout << "  Comment: training scales close to linearly with sample count for this configuration.\n";
    else
        std::cout << "  Comment: training grows faster than linear; depth/bin settings may be amplifying split cost.\n";
}

int main(int argc, char** argv) {
    std::cout << "=============================================================\n";
    std::cout << "   MILESTONE 1 — Sequential Decision Tree Benchmark Runner\n";
    std::cout << "=============================================================\n";

    try {
        RunConfig cfg;

        namespace fs = std::filesystem;
        std::vector<std::string> search_roots;
        search_roots.push_back(fs::current_path().string());
        search_roots.push_back((fs::current_path() / "milestone-1").string());
        if (argc > 0) {
            fs::path exe_path(argv[0]);
            if (exe_path.has_parent_path()) {
                search_roots.push_back(exe_path.parent_path().string());
                search_roots.push_back((exe_path.parent_path() / "milestone-1").string());
            }
        }

        std::vector<DatasetSpec> datasets = {
            {"Iris", "Iris.csv", true},
            {"Shuttle", "shuttle.csv", true},
            {"LetterRecognition", "letter-recognition.csv", true},
            {"Skin_NonSkin", "Skin_NonSkin.csv", true},
            {"Synthetic_dataset_1000k_200f", "synthetic_1000k_200f.csv", true}
        };

        std::vector<BenchmarkResult> metrics;
        std::vector<ScalabilityPoint> scalability_rows;
        const std::vector<double> fractions = {0.10, 0.25, 0.50, 0.75, 1.00};

        for (const auto& ds : datasets) {
            std::string resolved_path = resolve_dataset_path(ds.file, search_roots);
            std::cout << "\n[Loading] " << ds.name << " from: " << resolved_path << "\n";

            Dataset data = load_csv(resolved_path, ds.has_header);
            if (!validate_dataset(data, ds.name)) {
                std::cerr << "[!] Skipping dataset: " << ds.name << "\n";
                continue;
            }

            BenchmarkResult r = run_benchmark_on_dataset(ds, data, cfg);
            metrics.push_back(r);
            print_dataset_summary(r);

            auto s_points = run_scalability(ds, data, cfg, fractions);
            scalability_rows.insert(scalability_rows.end(), s_points.begin(), s_points.end());
            print_scalability_comment(ds.name, s_points);
        }

        save_metrics_csv("benchmark_metrics.csv", metrics, cfg);
        save_scalability_csv("benchmark_scalability.csv", scalability_rows);

        std::cout << "\n[Artifacts]\n";
        std::cout << "  - benchmark_metrics.csv\n";
        std::cout << "  - benchmark_scalability.csv\n";
        std::cout << "\nUse these CSV files to generate runtime/accuracy plots.\n";
    } catch (const std::exception& ex) {
        std::cerr << "Error: " << ex.what() << "\n";
        return 1;
    }
    return 0;
}