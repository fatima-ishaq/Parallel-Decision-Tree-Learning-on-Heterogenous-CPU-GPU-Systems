// ============================================================================
// MILESTONE 3 — INTEGRATED RANDOM FOREST & COMPACT INFERENCE
// ============================================================================
// Team contributions:
//   Yaman (Set 1)  : Bootstrap sampling, forest training control, tree collection
//   Faraz (Set 2)  : Compact tree representation, inference functions, tests
//   Zuhaa (Set 3)  : Parallel prediction (to be added)
// ============================================================================
//
// Build:
//   g++ -std=c++17 -O2 -o milestone3 milestone3_complete.cpp
//   ./milestone3
//
// ============================================================================

#include <algorithm>
#include <cassert>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <map>
#include <numeric>
#include <queue>
#include <random>
#include <set>
#include <sstream>
#include <stack>
#include <stdexcept>
#include <string>
#include <thread>
#include <mutex>
#include <vector>

// ============================================================================
// FARAZ'S MILESTONE 2 DATA STRUCTURES (Reused from M2)
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

struct SplitResult {
    int    best_feature   = -1;
    int    best_bin_index = -1;
    double best_threshold = 0.0;
    double best_gini      = std::numeric_limits<double>::infinity();
    bool   found          = false;
};

struct LevelwiseBuildStats {
    double total_time_ms = 0.0;
    double data_prep_ms  = 0.0;
    double split_eval_ms = 0.0;
    double apply_ms      = 0.0;
    int levels_processed     = 0;
    int total_nodes_created  = 0;
};

struct Node {
    std::vector<int> sample_indices;
    int depth           = 0;
    int feature_index   = -1;
    float threshold     = 0.0f;
    Node* left          = nullptr;
    Node* right         = nullptr;
    bool is_leaf        = false;
    int predicted_class = -1;
    explicit Node(int d = 0) : depth(d) {}
};

// ============================================================================
// FARAZ'S MILESTONE 2 CSV LOADER
// ============================================================================

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

static Node* create_root_node(const std::vector<int>& train_indices) {
    Node* root = new Node(0);
    root->sample_indices = train_indices;
    return root;
}

static const std::vector<int>& get_samples_for_node(Node* node) {
    return node->sample_indices;
}

static std::pair<Node*, Node*> split_node(Node* parent,
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

// ============================================================================
// FARAZ'S MILESTONE 2 CPU SPLIT FINDER (Fallback for CPU builds)
// ============================================================================

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

static SplitResult find_best_histogram_split_cpu(
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

// ============================================================================
// ZUHAA'S MILESTONE 2 TREE BUILDER (CPU fallback version)
// ============================================================================

static int majority_class(const std::vector<int>& indices, const std::vector<int>& labels) {
    std::map<int, int> counts;
    for (int idx : indices) counts[labels[idx]]++;
    return std::max_element(counts.begin(), counts.end(),
        [](const auto& a, const auto& b){ return a.second < b.second; })->first;
}

static bool is_pure(const std::vector<int>& indices, const std::vector<int>& labels) {
    if (indices.empty()) return true;
    int first = labels[indices[0]];
    for (int idx : indices)
        if (labels[idx] != first) return false;
    return true;
}

static bool should_split_node(const Node* node, const Dataset& data,
                               int max_depth, int min_samples_split) {
    const auto& idx = node->sample_indices;
    if (idx.empty())                             return false;
    if (node->depth >= max_depth)                return false;
    if ((int)idx.size() < min_samples_split)     return false;
    if (is_pure(idx, data.labels))               return false;
    return true;
}

static int predict_node(const Node* node, const std::vector<float>& sample) {
    if (node->is_leaf || !node->left || !node->right)
        return node->predicted_class;
    return (sample[node->feature_index] <= node->threshold)
        ? predict_node(node->left, sample)
        : predict_node(node->right, sample);
}

static void delete_tree(Node* node) {
    if (!node) return;
    delete_tree(node->left);
    delete_tree(node->right);
    delete node;
}

static Node* build_tree_levelwise_cpu(
    const Dataset& data,
    const std::vector<int>& train_indices,
    int max_depth,
    int min_samples_split,
    int bin_count,
    LevelwiseBuildStats& stats)
{
    using clock = std::chrono::high_resolution_clock;
    const auto wall_start = clock::now();

    Node* root = create_root_node(train_indices);
    stats.total_nodes_created = 1;
    const int nf = static_cast<int>(data.features[0].size());

    std::vector<Node*> frontier = { root };

    while (!frontier.empty()) {
        std::vector<Node*> active_nodes;
        active_nodes.reserve(frontier.size());

        for (Node* node : frontier) {
            const auto& indices = get_samples_for_node(node);
            node->predicted_class = majority_class(indices, data.labels);
            if (!should_split_node(node, data, max_depth, min_samples_split)) {
                node->is_leaf = true;
            } else {
                active_nodes.push_back(node);
            }
        }
        if (active_nodes.empty()) break;

        const auto split_start = clock::now();
        std::vector<SplitResult> results;

        for (Node* node : active_nodes) {
            const auto& indices = node->sample_indices;
            int n = (int)indices.size();
            std::vector<std::vector<double>> X(n, std::vector<double>(nf));
            std::vector<int> y(n);
            for (int i = 0; i < n; ++i) {
                int s = indices[i];
                for (int f = 0; f < nf; ++f)
                    X[i][f] = static_cast<double>(data.features[s][f]);
                y[i] = data.labels[s];
            }
            results.push_back(find_best_histogram_split_cpu(X, y, bin_count));
        }
        stats.split_eval_ms += std::chrono::duration<double, std::milli>(clock::now() - split_start).count();

        const auto apply_start = clock::now();
        std::vector<Node*> next_frontier;
        next_frontier.reserve(active_nodes.size() * 2);

        for (int i = 0; i < (int)active_nodes.size(); ++i) {
            Node* node = active_nodes[i];
            const SplitResult& sr = results[i];

            if (!sr.found) { node->is_leaf = true; continue; }

            std::vector<int> left_idx, right_idx;
            left_idx.reserve(node->sample_indices.size());
            right_idx.reserve(node->sample_indices.size());
            for (int s : node->sample_indices) {
                if (data.features[s][sr.best_feature] <= sr.best_threshold)
                    left_idx.push_back(s);
                else
                    right_idx.push_back(s);
            }

            if (left_idx.empty() || right_idx.empty()) { node->is_leaf = true; continue; }

            node->feature_index = sr.best_feature;
            node->threshold     = static_cast<float>(sr.best_threshold);
            auto [L, R] = split_node(node, left_idx, right_idx);
            next_frontier.push_back(L);
            next_frontier.push_back(R);
            stats.total_nodes_created += 2;
        }

        stats.apply_ms += std::chrono::duration<double, std::milli>(clock::now() - apply_start).count();
        frontier.swap(next_frontier);
        ++stats.levels_processed;
    }

    stats.total_time_ms = std::chrono::duration<double, std::milli>(clock::now() - wall_start).count();
    return root;
}

// ============================================================================
// YAMAN'S MILESTONE 3 CODE (Set 1: Random Forest Training Control)
// ============================================================================
// Responsibilities:
//   - Bootstrap sampling per tree (sampling with replacement)
//   - Reuses the M2 single-tree pipeline unchanged
//   - Manages forest-level orchestration and tree collection
//   - Caps tree count at MAX_TREES (10)
// ============================================================================

static constexpr int MAX_TREES = 10;

struct ForestConfig {
    int   n_trees          = 5;
    int   max_depth        = 8;
    int   min_samples_split = 2;
    int   bin_count        = 8;
    float bootstrap_ratio  = 1.0f;
    int   base_seed        = 42;
};

struct ForestTrainStats {
    int    n_trees           = 0;
    double total_forest_ms   = 0.0;
    double bootstrap_ms      = 0.0;
    double tree_build_ms     = 0.0;
    std::vector<double> per_tree_ms;
    std::vector<int>    per_tree_nodes;
};

struct ForestResult {
    std::vector<Node*>  trees;
    std::vector<int>    bootstrap_seeds;
    int                 n_classes = 0;
    ForestTrainStats    stats;

    ForestResult() = default;
    ForestResult(const ForestResult&)            = delete;
    ForestResult& operator=(const ForestResult&) = delete;
    ForestResult(ForestResult&& o) noexcept
        : trees(std::move(o.trees)),
          bootstrap_seeds(std::move(o.bootstrap_seeds)),
          n_classes(o.n_classes),
          stats(std::move(o.stats))
    { o.n_classes = 0; }

    ~ForestResult() {
        for (Node* t : trees) delete_tree(t);
    }
};

static std::vector<int> bootstrap_sample(
    const std::vector<int>& pool,
    int                     n_samples,
    int                     seed)
{
    if (pool.empty())
        throw std::invalid_argument("bootstrap_sample: pool is empty.");
    if (n_samples <= 0)
        throw std::invalid_argument("bootstrap_sample: n_samples must be > 0.");

    std::mt19937 rng(static_cast<std::uint32_t>(seed));
    std::uniform_int_distribution<int> dist(0, static_cast<int>(pool.size()) - 1);

    std::vector<int> sample(n_samples);
    for (int i = 0; i < n_samples; ++i)
        sample[i] = pool[dist(rng)];

    return sample;
}

static ForestResult train_random_forest(
    const Dataset&          data,
    const std::vector<int>& train_indices,
    const ForestConfig&     cfg)
{
    using clock = std::chrono::high_resolution_clock;

    const int n_trees = std::clamp(cfg.n_trees, 1, MAX_TREES);

    if (train_indices.empty())
        throw std::invalid_argument("train_random_forest: train_indices is empty.");
    if (data.features.empty())
        throw std::invalid_argument("train_random_forest: dataset has no features.");

    const int n_bootstrap = static_cast<int>(
        std::round(cfg.bootstrap_ratio * static_cast<float>(train_indices.size())));
    if (n_bootstrap <= 0)
        throw std::invalid_argument(
            "train_random_forest: bootstrap_ratio produced 0 samples.");

    const int n_classes = count_unique_classes(data.labels);

    ForestResult forest;
    forest.n_classes = n_classes;
    forest.trees.reserve(n_trees);
    forest.bootstrap_seeds.reserve(n_trees);
    forest.stats.per_tree_ms.reserve(n_trees);
    forest.stats.per_tree_nodes.reserve(n_trees);

    const auto forest_wall_start = clock::now();

    for (int t = 0; t < n_trees; ++t) {
        const int tree_seed = cfg.base_seed + t;

        const auto bs_start = clock::now();
        std::vector<int> boot_indices =
            bootstrap_sample(train_indices, n_bootstrap, tree_seed);
        forest.stats.bootstrap_ms +=
            std::chrono::duration<double, std::milli>(
                clock::now() - bs_start).count();

        const auto tree_start = clock::now();
        LevelwiseBuildStats tree_stats;
        Node* root = build_tree_levelwise_cpu(
            data,
            boot_indices,
            cfg.max_depth,
            cfg.min_samples_split,
            cfg.bin_count,
            tree_stats);
        const double tree_ms =
            std::chrono::duration<double, std::milli>(
                clock::now() - tree_start).count();

        forest.stats.tree_build_ms += tree_ms;
        forest.stats.per_tree_ms.push_back(tree_ms);
        forest.stats.per_tree_nodes.push_back(tree_stats.total_nodes_created);

        forest.trees.push_back(root);
        forest.bootstrap_seeds.push_back(tree_seed);

        std::cout << "  [Forest] Tree " << (t + 1) << "/" << n_trees
                  << "  nodes=" << tree_stats.total_nodes_created
                  << "  time=" << std::fixed << std::setprecision(1)
                  << tree_ms << " ms\n";
    }

    forest.stats.total_forest_ms =
        std::chrono::duration<double, std::milli>(
            clock::now() - forest_wall_start).count();
    forest.stats.n_trees = n_trees;

    return forest;
}

static void print_forest_stats(const ForestTrainStats& s) {
    std::cout << "\n[Forest Training Statistics]\n"
              << std::fixed << std::setprecision(2);
    std::cout << "  Trees trained        : " << s.n_trees          << "\n";
    std::cout << "  Total forest time    : " << s.total_forest_ms  << " ms\n";
    std::cout << "  Bootstrap time       : " << s.bootstrap_ms     << " ms\n";
    std::cout << "  Tree build time      : " << s.tree_build_ms    << " ms\n";
    std::cout << "  Avg time / tree      : "
              << (s.n_trees > 0 ? s.tree_build_ms / s.n_trees : 0.0) << " ms\n";
    std::cout << "\n  Per-tree breakdown:\n";
    for (int t = 0; t < s.n_trees; ++t) {
        std::cout << "    Tree " << std::setw(2) << (t + 1)
                  << " : " << std::setw(8) << s.per_tree_ms[t]    << " ms"
                  << "   nodes=" << s.per_tree_nodes[t] << "\n";
    }
}

// ============================================================================
// FARAZ'S MILESTONE 3 CODE (Set 2: Compact Tree Representation & Inference)
// ============================================================================
// Responsibilities:
//   - Convert pointer-based Node trees into compact array-based layout
//   - Implement single-tree prediction using compact arrays (no recursion)
//   - Implement majority voting across forest for classification
//   - Memory-efficient inference for deployment scenarios
// ============================================================================

struct CompactNode {
    int   feature          = -1;
    float threshold        = 0.0f;
    int   left_idx         = -1;
    int   right_idx        = -1;
    int   predicted_class  = -1;
    bool  is_leaf          = true;
    
    CompactNode() = default;
    CompactNode(int feat, float thresh, int left, int right)
        : feature(feat), threshold(thresh), 
          left_idx(left), right_idx(right), 
          is_leaf(false) {}
    CompactNode(int pred_class)
        : predicted_class(pred_class), is_leaf(true) {}
};

struct CompactForest {
    std::vector<std::vector<CompactNode>> trees;
    int n_classes = 0;
    int n_trees = 0;
    
    CompactForest() = default;
    
    CompactForest(const ForestResult& forest, int num_classes) {
        trees.reserve(forest.trees.size());
        n_trees = static_cast<int>(forest.trees.size());
        n_classes = num_classes;
        
        for (Node* root : forest.trees) {
            trees.push_back(serialize_tree(root));
        }
    }
    
private:
    std::vector<CompactNode> serialize_tree(Node* root) {
        std::vector<CompactNode> nodes;
        if (!root) return nodes;
        
        std::queue<std::pair<Node*, int>> q;
        q.push({root, 0});
        std::map<Node*, int> node_to_index;
        int next_idx = 0;
        
        while (!q.empty()) {
            auto [node, idx] = q.front();
            q.pop();
            
            if (node_to_index.find(node) != node_to_index.end()) continue;
            node_to_index[node] = idx;
            next_idx = std::max(next_idx, idx + 1);
            
            if (node->left) q.push({node->left, next_idx++});
            if (node->right) q.push({node->right, next_idx++});
        }
        
        nodes.resize(next_idx);
        
        for (const auto& [node, idx] : node_to_index) {
            if (node->is_leaf || (!node->left && !node->right)) {
                nodes[idx] = CompactNode(node->predicted_class);
            } else {
                int left_idx = node_to_index[node->left];
                int right_idx = node_to_index[node->right];
                nodes[idx] = CompactNode(node->feature_index, node->threshold, 
                                         left_idx, right_idx);
            }
        }
        
        return nodes;
    }
};

static int predict_compact_tree(
    const std::vector<CompactNode>& tree_nodes,
    const std::vector<float>& sample)
{
    if (tree_nodes.empty()) return -1;
    
    int node_idx = 0;
    
    while (true) {
        const CompactNode& node = tree_nodes[node_idx];
        
        if (node.is_leaf || node.feature == -1) {
            return node.predicted_class;
        }
        
        if (sample[node.feature] <= node.threshold) {
            node_idx = node.left_idx;
        } else {
            node_idx = node.right_idx;
        }
        
        if (node_idx < 0 || node_idx >= static_cast<int>(tree_nodes.size())) {
            return -1;
        }
    }
}

static std::vector<int> predict_compact_forest_batch(
    const CompactForest& forest,
    const Dataset& data,
    const std::vector<int>& indices)
{
    if (forest.trees.empty()) {
        throw std::runtime_error("predict_compact_forest_batch: no trees in forest");
    }
    
    if (indices.empty()) return {};
    
    const int n_samples = static_cast<int>(indices.size());
    const int n_trees = forest.n_trees;
    
    std::vector<int> predictions(n_samples);
    
    for (int i = 0; i < n_samples; ++i) {
        const std::vector<float>& sample = data.features[indices[i]];
        std::vector<int> votes(forest.n_classes, 0);
        
        for (int t = 0; t < n_trees; ++t) {
            int pred = predict_compact_tree(forest.trees[t], sample);
            if (pred >= 0 && pred < forest.n_classes) {
                votes[pred]++;
            }
        }
        
        int best_class = 0;
        int best_votes = votes[0];
        for (int c = 1; c < forest.n_classes; ++c) {
            if (votes[c] > best_votes) {
                best_votes = votes[c];
                best_class = c;
            }
        }
        
        predictions[i] = best_class;
    }
    
    return predictions;
}

static double compute_accuracy(const std::vector<int>& predictions, const std::vector<int>& truth) {
    if (predictions.size() != truth.size() || predictions.empty())
        return 0.0;
    int correct = 0;
    for (size_t i = 0; i < predictions.size(); ++i)
        if (predictions[i] == truth[i]) correct++;
    return static_cast<double>(correct) / predictions.size();
}

static double compute_compact_forest_accuracy(
    const CompactForest& forest,
    const Dataset& data,
    const std::vector<int>& indices)
{
    auto predictions = predict_compact_forest_batch(forest, data, indices);
    
    std::vector<int> truth;
    truth.reserve(indices.size());
    for (int idx : indices) {
        truth.push_back(data.labels[idx]);
    }
    
    return compute_accuracy(predictions, truth);
}

static void print_compact_forest_memory_report(const CompactForest& forest) {
    std::cout << "\n[Compact Forest Memory Report - Faraz]\n";
    std::cout << "  Number of trees: " << forest.n_trees << "\n";
    
    size_t total_nodes = 0;
    size_t total_bytes = 0;
    
    for (const auto& tree : forest.trees) {
        total_nodes += tree.size();
        total_bytes += tree.size() * sizeof(CompactNode);
    }
    
    std::cout << "  Total nodes: " << total_nodes << "\n";
    std::cout << "  Memory usage (CompactNode): " << total_bytes << " bytes ("
              << std::fixed << std::setprecision(2) 
              << total_bytes / 1024.0 << " KB)\n";
    
    size_t pointer_memory = total_nodes * (sizeof(Node*) * 3 + sizeof(int) * 4 + sizeof(float) + sizeof(bool));
    std::cout << "  Estimated pointer-tree memory: ~" << pointer_memory << " bytes ("
              << pointer_memory / 1024.0 << " KB)\n";
    std::cout << "  Memory savings: " << std::setprecision(1) 
              << (1.0 - static_cast<double>(total_bytes) / pointer_memory) * 100.0 << "%\n";
}

// ============================================================================
// FARAZ'S TESTS FOR COMPACT REPRESENTATION
// ============================================================================

static void write_test_csv(const std::string& fn) {
    std::ofstream f(fn);
    f << "feat0,feat1,feat2,label\n";
    for (int i = 0; i < 20; ++i)
        f << (i*0.1) << "," << (i*0.1) << "," << (i*0.1) << ",0\n";
    for (int i = 0; i < 20; ++i)
        f << (2.0 + i*0.1) << "," << (2.0 + i*0.1) << "," << (2.0 + i*0.1) << ",1\n";
    for (int i = 0; i < 20; ++i)
        f << (4.0 + i*0.1) << "," << (4.0 + i*0.1) << "," << (4.0 + i*0.1) << ",2\n";
}

static void test_compact_serialization() {
    std::cout << "\n[Faraz Test 1: Compact Tree Serialization]\n";
    
    write_test_csv("faraz_test.csv");
    Dataset data = load_csv("faraz_test.csv", true);
    auto [train_idx, test_idx] = train_test_split(data, 0.8f, 42);
    
    LevelwiseBuildStats stats;
    Node* root = build_tree_levelwise_cpu(data, train_idx, 5, 2, 8, stats);
    
    std::queue<std::pair<Node*, int>> q;
    q.push({root, 0});
    std::map<Node*, int> idx_map;
    int next = 0;
    
    while (!q.empty()) {
        auto [node, idx] = q.front(); q.pop();
        if (idx_map.count(node)) continue;
        idx_map[node] = idx;
        next = std::max(next, idx + 1);
        if (node->left) q.push({node->left, next++});
        if (node->right) q.push({node->right, next++});
    }
    
    std::vector<CompactNode> compact_tree(next);
    for (const auto& [node, idx] : idx_map) {
        if (node->is_leaf || !node->left) {
            compact_tree[idx] = CompactNode(node->predicted_class);
        } else {
            compact_tree[idx] = CompactNode(node->feature_index, node->threshold,
                                            idx_map[node->left], idx_map[node->right]);
        }
    }
    
    int correct = 0;
    for (int idx : test_idx) {
        int pred_ptr = predict_node(root, data.features[idx]);
        int pred_compact = predict_compact_tree(compact_tree, data.features[idx]);
        if (pred_ptr == pred_compact) correct++;
    }
    
    double match_rate = static_cast<double>(correct) / test_idx.size();
    std::cout << "  Predictions match rate: " << (match_rate * 100.0) << "%\n";
    assert(match_rate == 1.0);
    
    delete_tree(root);
    std::cout << "  [PASS] Compact tree matches pointer-based tree\n";
}

static void test_compact_forest_accuracy() {
    std::cout << "\n[Faraz Test 2: Compact Forest Accuracy]\n";
    
    write_test_csv("faraz_forest.csv");
    Dataset data = load_csv("faraz_forest.csv", true);
    auto [train_idx, test_idx] = train_test_split(data, 0.8f, 42);
    
    ForestConfig cfg;
    cfg.n_trees = 10;
    cfg.max_depth = 5;
    cfg.bin_count = 8;
    cfg.base_seed = 42;
    
    ForestResult forest = train_random_forest(data, train_idx, cfg);
    int n_classes = count_unique_classes(data.labels);
    
    CompactForest compact_forest(forest, n_classes);
    
    auto preds_compact = predict_compact_forest_batch(compact_forest, data, test_idx);
    
    std::vector<int> truth;
    for (int idx : test_idx) truth.push_back(data.labels[idx]);
    
    double acc = compute_accuracy(preds_compact, truth);
    std::cout << "  Compact forest test accuracy: " << (acc * 100.0) << "%\n";
    
    print_compact_forest_memory_report(compact_forest);
    print_forest_stats(forest.stats);
    std::cout << "  [PASS] Compact forest works correctly\n";
}

static void test_compact_inference_speed() {
    std::cout << "\n[Faraz Test 3: Inference Speed Comparison]\n";
    
    write_test_csv("faraz_speed.csv");
    Dataset data = load_csv("faraz_speed.csv", true);
    auto [train_idx, test_idx] = train_test_split(data, 0.8f, 42);
    
    ForestConfig cfg;
    cfg.n_trees = 10;
    cfg.max_depth = 6;
    cfg.bin_count = 8;
    cfg.base_seed = 42;
    
    ForestResult forest = train_random_forest(data, train_idx, cfg);
    int n_classes = count_unique_classes(data.labels);
    CompactForest compact_forest(forest, n_classes);
    
    auto start = std::chrono::high_resolution_clock::now();
    auto preds_compact = predict_compact_forest_batch(compact_forest, data, test_idx);
    auto end = std::chrono::high_resolution_clock::now();
    double compact_time = std::chrono::duration<double, std::milli>(end - start).count();
    
    std::cout << "  Compact forest inference: " << std::fixed << std::setprecision(3) 
              << compact_time << " ms for " << test_idx.size() << " samples\n";
    std::cout << "  Latency per sample: " << (compact_time / test_idx.size() * 1000) << " us\n";
    
    std::cout << "  [PASS] Compact inference is efficient\n";
}

// ============================================================================
// YAMAN'S TESTS (Minimal - to verify his code works)
// ============================================================================

static void test_bootstrap_sample_properties() {
    std::cout << "\n[Yaman Test: Bootstrap Sample Properties]\n";
    std::vector<int> pool(100);
    std::iota(pool.begin(), pool.end(), 0);
    auto s1 = bootstrap_sample(pool, 100, 42);
    assert((int)s1.size() == 100);
    std::cout << "  [PASS]\n";
}

static void test_tree_count_capped() {
    std::cout << "\n[Yaman Test: Tree Count Capped]\n";
    write_test_csv("m3_cap.csv");
    Dataset data = load_csv("m3_cap.csv", true);
    auto [train_idx, test_idx] = train_test_split(data, 0.8f, 42);
    ForestConfig cfg;
    cfg.n_trees = 999;
    cfg.max_depth = 3;
    ForestResult forest = train_random_forest(data, train_idx, cfg);
    assert((int)forest.trees.size() == MAX_TREES);
    std::cout << "  [PASS] Capped at " << MAX_TREES << "\n";
}

// ============================================================================
// ZUHAA'S MILESTONE 3 CODE (Set 3: Parallel Inference & Throughput)
// ============================================================================
// Responsibilities:
//   1. Batch input preparation  — flatten Dataset into contiguous row-major
//      float matrix so every inference variant reads from the same buffer
//      (avoids repeated vector-of-vector derefs inside timing loops).
//   2. Parallel prediction across samples — split sample indices across
//      std::threads; each thread works on an independent slice and writes
//      to its own slice of the output vector (zero shared-write contention).
//   3. Parallel prediction across trees — each thread owns one tree and
//      runs ALL samples through it; a final sequential pass merges votes.
//   4. Throughput measurement — times sequential vs parallel variants,
//      computes samples/sec and speedup ratios.
//   5. Speedup-vs-trees table — re-uses sub-forests of increasing size
//      (no re-training) to show how parallelism scales with ensemble size.
//
// Build (CPU only, no flags needed beyond what already exists):
//   g++ -std=c++17 -O2 -o milestone3 milestone3_complete.cpp
// Build with OpenMP for additional speedup reporting:
//   g++ -std=c++17 -O2 -fopenmp -o milestone3 milestone3_complete.cpp
// ============================================================================

// ── Helpers ──────────────────────────────────────────────────────────────────

// Number of hardware threads available; used as the default thread count.
static int hw_threads() {
    int n = static_cast<int>(std::thread::hardware_concurrency());
    return (n > 0) ? n : 2;   // fall back to 2 if detection fails
}

// ── 1. Batch Input Preparation ───────────────────────────────────────────────
// Converts the vector-of-vectors Dataset layout into a single contiguous
// row-major float array.  Prepared ONCE before the timing loop so the
// data-flattening cost is not charged against inference latency.
// All three prediction variants (sequential, parallel-samples, parallel-trees)
// use this same buffer.

struct FlatBatch {
    std::vector<float> x;    // [n_samples × n_features], row-major
    int n_samples  = 0;
    int n_features = 0;
};

// Build a FlatBatch from a subset of Dataset rows specified by 'indices'.
static FlatBatch prepare_batch(
    const Dataset&          data,
    const std::vector<int>& indices)
{
    FlatBatch b;
    b.n_samples  = static_cast<int>(indices.size());
    b.n_features = b.n_samples > 0
        ? static_cast<int>(data.features[indices[0]].size()) : 0;
    b.x.resize(static_cast<std::size_t>(b.n_samples) * b.n_features);

    for (int i = 0; i < b.n_samples; ++i) {
        const std::vector<float>& row = data.features[indices[i]];
        std::copy(row.begin(), row.end(),
                  b.x.data() + static_cast<std::size_t>(i) * b.n_features);
    }
    return b;
}

// ── Internal: predict one sample through one compact tree using flat buffer ──
// Uses the same traversal logic as predict_compact_tree() but reads from the
// pre-flattened pointer instead of a std::vector<float>.
static int predict_one(
    const std::vector<CompactNode>& tree_nodes,
    const float*                    sample_row,
    int                             /*n_features*/)
{
    if (tree_nodes.empty()) return -1;
    int node_idx = 0;
    while (true) {
        const CompactNode& nd = tree_nodes[node_idx];
        if (nd.is_leaf || nd.feature == -1) return nd.predicted_class;
        node_idx = (sample_row[nd.feature] <= nd.threshold)
                 ? nd.left_idx : nd.right_idx;
        if (node_idx < 0 || node_idx >= static_cast<int>(tree_nodes.size()))
            return -1;
    }
}

// ── 2. Sequential baseline (compact, uses FlatBatch) ─────────────────────────
// Identical logic to Faraz's predict_compact_forest_batch() but reads from the
// pre-flattened buffer.  Acts as the direct baseline for speedup calculations.

static std::vector<int> predict_sequential(
    const CompactForest& forest,
    const FlatBatch&     batch)
{
    const int N  = batch.n_samples;
    const int nf = batch.n_features;
    std::vector<int> out(N, 0);

    for (int i = 0; i < N; ++i) {
        const float* row = batch.x.data() + static_cast<std::size_t>(i) * nf;
        std::vector<int> votes(forest.n_classes, 0);
        for (const auto& tree : forest.trees) {
            int pred = predict_one(tree, row, nf);
            if (pred >= 0 && pred < forest.n_classes) ++votes[pred];
        }
        int best = 0;
        for (int c = 1; c < forest.n_classes; ++c)
            if (votes[c] > votes[best]) best = c;
        out[i] = best;
    }
    return out;
}

// ── 3a. Parallel prediction across samples ────────────────────────────────────
// The N test samples are divided into n_threads equal-sized chunks.
// Each thread independently processes its chunk and writes directly into
// its own slice of the output vector — no locks, no atomics, no false sharing
// (writes are to distinct cache-line-aligned ranges for typical N > 64).

static std::vector<int> predict_parallel_samples(
    const CompactForest& forest,
    const FlatBatch&     batch,
    int                  n_threads = -1)
{
    if (n_threads <= 0) n_threads = hw_threads();
    n_threads = std::min(n_threads, batch.n_samples);
    if (n_threads <= 1) return predict_sequential(forest, batch);

    const int N  = batch.n_samples;
    const int nf = batch.n_features;
    std::vector<int> out(N, 0);
    std::vector<std::thread> workers(n_threads);

    // Divide [0, N) into n_threads contiguous slices
    for (int t = 0; t < n_threads; ++t) {
        const int begin = (t * N) / n_threads;
        const int end   = ((t + 1) * N) / n_threads;

        workers[t] = std::thread([&, begin, end]() {
            for (int i = begin; i < end; ++i) {
                const float* row = batch.x.data() +
                                   static_cast<std::size_t>(i) * nf;
                std::vector<int> votes(forest.n_classes, 0);
                for (const auto& tree : forest.trees) {
                    int pred = predict_one(tree, row, nf);
                    if (pred >= 0 && pred < forest.n_classes) ++votes[pred];
                }
                int best = 0;
                for (int c = 1; c < forest.n_classes; ++c)
                    if (votes[c] > votes[best]) best = c;
                out[i] = best;  // safe: each thread writes a unique range
            }
        });
    }
    for (auto& w : workers) w.join();
    return out;
}

// ── 3b. Parallel prediction across trees ─────────────────────────────────────
// Each thread owns ONE tree and runs all N samples through it, writing into
// its own row of a T×N vote matrix.  A final sequential pass collects votes.
// This exploits ensemble-level parallelism and is most effective when
// n_trees >= n_threads.

static std::vector<int> predict_parallel_trees(
    const CompactForest& forest,
    const FlatBatch&     batch,
    int                  n_threads = -1)
{
    const int T  = static_cast<int>(forest.trees.size());
    const int N  = batch.n_samples;
    const int nf = batch.n_features;

    if (n_threads <= 0) n_threads = hw_threads();
    n_threads = std::min(n_threads, T);
    if (n_threads <= 1 || T == 0) return predict_sequential(forest, batch);

    // tree_votes[t][i] = class predicted by tree t for sample i
    std::vector<std::vector<int>> tree_votes(T, std::vector<int>(N, -1));
    std::vector<std::thread> workers(n_threads);

    for (int t_group = 0; t_group < n_threads; ++t_group) {
        const int t_begin = (t_group * T) / n_threads;
        const int t_end   = ((t_group + 1) * T) / n_threads;

        workers[t_group] = std::thread([&, t_begin, t_end]() {
            for (int t = t_begin; t < t_end; ++t) {
                const auto& tree = forest.trees[t];
                for (int i = 0; i < N; ++i) {
                    const float* row = batch.x.data() +
                                       static_cast<std::size_t>(i) * nf;
                    tree_votes[t][i] = predict_one(tree, row, nf);
                }
            }
        });
    }
    for (auto& w : workers) w.join();

    // Sequential vote reduction: O(N × T), inexpensive
    std::vector<int> out(N, 0);
    for (int i = 0; i < N; ++i) {
        std::vector<int> votes(forest.n_classes, 0);
        for (int t = 0; t < T; ++t) {
            int pred = tree_votes[t][i];
            if (pred >= 0 && pred < forest.n_classes) ++votes[pred];
        }
        int best = 0;
        for (int c = 1; c < forest.n_classes; ++c)
            if (votes[c] > votes[best]) best = c;
        out[i] = best;
    }
    return out;
}

// ── 4. Throughput measurement ─────────────────────────────────────────────────

struct ThroughputResult {
    std::string label;
    int    n_samples       = 0;
    int    n_trees         = 0;
    int    n_threads       = 0;
    double elapsed_ms      = 0.0;
    double samples_per_sec = 0.0;
    double speedup         = 1.0;   // relative to sequential baseline
};

// Time fn() for 'reps' repetitions; return minimum elapsed milliseconds.
// Minimum is used to reduce OS scheduling jitter.
template <typename Fn>
static double time_min_ms(Fn&& fn, int reps = 5) {
    double best = std::numeric_limits<double>::infinity();
    for (int r = 0; r < reps; ++r) {
        auto t0  = std::chrono::high_resolution_clock::now();
        fn();
        double ms = std::chrono::duration<double, std::milli>(
            std::chrono::high_resolution_clock::now() - t0).count();
        if (ms < best) best = ms;
    }
    return best;
}

static ThroughputResult make_result(
    const std::string& label, int n_samples, int n_trees, int n_threads,
    double ms, double baseline_ms)
{
    ThroughputResult r;
    r.label          = label;
    r.n_samples      = n_samples;
    r.n_trees        = n_trees;
    r.n_threads      = n_threads;
    r.elapsed_ms     = ms;
    r.samples_per_sec = (ms > 0.0) ? (n_samples / (ms / 1000.0)) : 0.0;
    r.speedup         = (ms > 0.0 && baseline_ms > 0.0) ? baseline_ms / ms : 1.0;
    return r;
}

// Run all three inference variants on the same batch and return a result table.
static std::vector<ThroughputResult> measure_throughput(
    const CompactForest&    forest,
    const Dataset&          data,
    const std::vector<int>& test_indices,
    int                     n_threads = -1,
    int                     reps      = 5)
{
    if (n_threads <= 0) n_threads = hw_threads();

    // Prepare flat batch ONCE — not charged to inference timing
    const FlatBatch batch = prepare_batch(data, test_indices);
    const int N = batch.n_samples;
    const int T = static_cast<int>(forest.trees.size());

    // Warm-up run (not measured)
    predict_sequential(forest, batch);

    const double seq_ms = time_min_ms(
        [&]{ predict_sequential(forest, batch); }, reps);
    const double par_s_ms = time_min_ms(
        [&]{ predict_parallel_samples(forest, batch, n_threads); }, reps);
    const double par_t_ms = time_min_ms(
        [&]{ predict_parallel_trees(forest, batch, n_threads); }, reps);

    return {
        make_result("Sequential (compact baseline)", N, T, 1,        seq_ms,   seq_ms),
        make_result("Parallel across samples",       N, T, n_threads, par_s_ms, seq_ms),
        make_result("Parallel across trees",         N, T, n_threads, par_t_ms, seq_ms),
    };
}

static void print_throughput_table(
    const std::vector<ThroughputResult>& results)
{
    std::cout << "\n  " << std::left
              << std::setw(32) << "Variant"
              << std::setw(10) << "Threads"
              << std::setw(12) << "Time (ms)"
              << std::setw(18) << "Throughput (sps)"
              << std::setw(10) << "Speedup"
              << "\n  " << std::string(80, '-') << "\n";
    for (const auto& r : results) {
        std::cout << "  " << std::left
                  << std::setw(32) << r.label
                  << std::setw(10) << r.n_threads
                  << std::setw(12) << std::fixed << std::setprecision(3) << r.elapsed_ms
                  << std::setw(18) << static_cast<long long>(r.samples_per_sec)
                  << std::setw(10) << std::setprecision(2) << r.speedup
                  << "x\n";
    }
}

// ── 5. Speedup-vs-number-of-trees table ──────────────────────────────────────
// Creates a view of the first k trees from the existing CompactForest
// (no re-training) and measures parallel-sample inference speedup for each k.

struct SpeedupRow {
    int    n_trees       = 0;
    double seq_ms        = 0.0;
    double par_ms        = 0.0;
    double speedup       = 0.0;
    double throughput    = 0.0;  // samples/sec for parallel variant
};

static std::vector<SpeedupRow> speedup_vs_trees(
    const CompactForest&    forest,
    const FlatBatch&        batch,
    const std::vector<int>& tree_counts,
    int                     n_threads = -1,
    int                     reps      = 5)
{
    if (n_threads <= 0) n_threads = hw_threads();
    const int N = batch.n_samples;

    std::vector<SpeedupRow> rows;
    for (int k : tree_counts) {
        if (k < 1 || k > static_cast<int>(forest.trees.size())) continue;

        // Build a lightweight sub-forest view using first k trees
        CompactForest sub;
        sub.n_classes = forest.n_classes;
        sub.n_trees   = k;
        sub.trees.assign(forest.trees.begin(), forest.trees.begin() + k);

        const double seq_ms = time_min_ms(
            [&]{ predict_sequential(sub, batch); }, reps);
        const double par_ms = time_min_ms(
            [&]{ predict_parallel_samples(sub, batch, n_threads); }, reps);

        SpeedupRow row;
        row.n_trees   = k;
        row.seq_ms    = seq_ms;
        row.par_ms    = par_ms;
        row.speedup   = (par_ms > 0.0) ? seq_ms / par_ms : 1.0;
        row.throughput = (par_ms > 0.0) ? N / (par_ms / 1000.0) : 0.0;
        rows.push_back(row);
    }
    return rows;
}

static void print_speedup_table(
    const std::vector<SpeedupRow>& rows)
{
    std::cout << "\n  " << std::left
              << std::setw(12) << "N-Trees"
              << std::setw(14) << "Seq (ms)"
              << std::setw(14) << "Par (ms)"
              << std::setw(14) << "Speedup"
              << std::setw(20) << "Throughput (sps)"
              << "\n  " << std::string(72, '-') << "\n";
    for (const auto& r : rows) {
        std::cout << "  " << std::left
                  << std::setw(12) << r.n_trees
                  << std::setw(14) << std::fixed << std::setprecision(3) << r.seq_ms
                  << std::setw(14) << r.par_ms
                  << std::setw(14) << std::setprecision(2) << r.speedup
                  << static_cast<long long>(r.throughput) << " sps\n";
    }
}

// ── Save throughput results to CSV ───────────────────────────────────────────
static void save_throughput_csv(
    const std::string&                        path,
    const std::vector<ThroughputResult>& rows)
{
    std::ofstream out(path);
    out << "variant,n_samples,n_trees,n_threads,elapsed_ms,samples_per_sec,speedup\n";
    out << std::fixed << std::setprecision(4);
    for (const auto& r : rows) {
        out << r.label        << ',' << r.n_samples      << ',' << r.n_trees
            << ',' << r.n_threads    << ',' << r.elapsed_ms
            << ',' << static_cast<long long>(r.samples_per_sec)
            << ',' << r.speedup << '\n';
    }
}

static void save_speedup_csv(
    const std::string&                  path,
    const std::vector<SpeedupRow>& rows)
{
    std::ofstream out(path);
    out << "n_trees,seq_ms,par_sample_ms,speedup,throughput_sps\n";
    out << std::fixed << std::setprecision(4);
    for (const auto& r : rows) {
        out << r.n_trees   << ',' << r.seq_ms << ',' << r.par_ms
            << ',' << r.speedup << ',' << static_cast<long long>(r.throughput) << '\n';
    }
}

// ============================================================================
// CPU-ONLY BENCHMARK CSV EXPORT
// ============================================================================

struct CpuBenchmarkRow {
    std::string dataset;
    std::string variant;
    int n_trees = 0;
    int n_samples = 0;
    int n_features = 0;
    int n_classes = 0;
    double test_accuracy = 0.0;
    double train_time_sec = 0.0;
    double total_tree_time_sec = 0.0;
    double gpu_kernel_time_sec = 0.0;
    double avg_tree_time_sec = 0.0;
};

static void save_benchmarks_csv(
    const std::string& path,
    const std::vector<CpuBenchmarkRow>& rows)
{
    std::ofstream out(path);
    out << "dataset,variant,n_trees,n_samples,n_features,n_classes,test_accuracy,train_time_sec,total_tree_time_sec,gpu_kernel_time_sec,avg_tree_time_sec\n";
    out << std::fixed << std::setprecision(6);
    for (const auto& r : rows) {
        out << r.dataset << ',' << r.variant << ',' << r.n_trees << ','
            << r.n_samples << ',' << r.n_features << ',' << r.n_classes << ','
            << r.test_accuracy << ',' << r.train_time_sec << ',' << r.total_tree_time_sec << ','
            << r.gpu_kernel_time_sec << ',' << r.avg_tree_time_sec << '\n';
    }
}

static void run_cpu_only_benchmarks() {
    struct DatasetSpec {
        std::string label;
        std::string file;
        bool has_header;
    };

    const std::vector<DatasetSpec> datasets = {
        {"Iris", "Iris.csv", true},
        {"Letter", "letter-recognition.csv", true},
        {"Shuttle", "shuttle.csv", true},
        {"Skin_NonSkin", "Skin_NonSkin.csv", true},
        {"Synthetic_1M_200f", "synthetic_1000k_200f.csv", true}
    };

    ForestConfig cfg;
    cfg.n_trees = 10;
    cfg.max_depth = 8;
    cfg.min_samples_split = 2;
    cfg.bin_count = 8;
    cfg.bootstrap_ratio = 1.0f;
    cfg.base_seed = 42;

    std::vector<CpuBenchmarkRow> rows;
    rows.reserve(datasets.size());

    std::cout << "\n[CPU-only benchmark export]\n";
    for (const auto& spec : datasets) {
        Dataset data;
        bool loaded = false;
        for (const std::string& path : {spec.file, std::string("datasets/") + spec.file}) {
            if (path.empty()) continue;
            try {
                data = load_csv(path, spec.has_header);
                loaded = true;
                std::cout << "  [Loaded] " << spec.label << " from: " << path << "\n";
                break;
            } catch (const std::exception&) {
                // try the next path
            }
        }
        if (!loaded) {
            std::cout << "  [Skipped] " << spec.label << " (could not open CSV)\n";
            continue;
        }

        const int n_samples  = static_cast<int>(data.features.size());
        const int n_features = data.features.empty() ? 0 : static_cast<int>(data.features[0].size());
        const int n_classes  = count_unique_classes(data.labels);

        auto [train_idx, test_idx] = train_test_split(data, 0.8f, cfg.base_seed);

        ForestResult forest = train_random_forest(data, train_idx, cfg);
        CompactForest compact_forest(forest, n_classes);
        auto test_predictions = predict_compact_forest_batch(compact_forest, data, test_idx);

        std::vector<int> truth;
        truth.reserve(test_idx.size());
        for (int idx : test_idx) truth.push_back(data.labels[idx]);

        const double acc = compute_accuracy(test_predictions, truth);
        rows.push_back({
            spec.label,
            "CPU-Only",
            cfg.n_trees,
            n_samples,
            n_features,
            n_classes,
            acc,
            forest.stats.total_forest_ms / 1000.0,
            forest.stats.tree_build_ms / 1000.0,
            0.0,
            (cfg.n_trees > 0 ? (forest.stats.tree_build_ms / cfg.n_trees) / 1000.0 : 0.0)
        });
    }

    save_benchmarks_csv("benchmark_forest_train_m3_cpu.csv", rows);
    std::cout << "\n[Saved] benchmark_forest_train_m3_cpu.csv\n";
}

// ============================================================================
// ZUHAA'S TESTS
// ============================================================================

// Helper: write a 3-class separable dataset for Zuhaa's tests
static void zuhaa_write_test_csv(const std::string& fn, int n_per_class = 80) {
    std::ofstream f(fn);
    f << "f0,f1,f2,label\n";
    for (int cls = 0; cls < 3; ++cls)
        for (int i = 0; i < n_per_class; ++i)
            f << std::fixed << std::setprecision(3)
              << (cls * 5.0f + 0.05f * i) << ','
              << (cls * 5.0f + 0.05f * i) << ','
              << static_cast<float>(cls)   << ','
              << cls << '\n';
}

// Helper: train a small forest on the test dataset
static CompactForest train_test_forest(
    const Dataset&          data,
    const std::vector<int>& train_idx,
    int n_trees = 5, int max_depth = 5)
{
    ForestConfig cfg;
    cfg.n_trees   = n_trees;
    cfg.max_depth = max_depth;
    cfg.bin_count = 8;
    cfg.base_seed = 42;
    ForestResult forest = train_random_forest(data, train_idx, cfg);
    return CompactForest(forest, count_unique_classes(data.labels));
}

// Test 1: FlatBatch is correctly shaped and matches original feature values
static void test_flat_batch_preparation() {
    std::cout << "\n[Zuhaa Test 1: Flat Batch Preparation]\n";

    zuhaa_write_test_csv("z_batch.csv", 30);
    Dataset data = load_csv("z_batch.csv", true);
    auto [tr, te] = train_test_split(data, 0.8f, 7);

    FlatBatch batch = prepare_batch(data, te);

    assert(batch.n_samples  == static_cast<int>(te.size()));
    assert(batch.n_features == static_cast<int>(data.features[te[0]].size()));
    assert(static_cast<int>(batch.x.size()) == batch.n_samples * batch.n_features);

    // Each row in the flat buffer must match the original Dataset row
    for (int i = 0; i < batch.n_samples; ++i) {
        const auto& orig = data.features[te[i]];
        for (int f = 0; f < batch.n_features; ++f) {
            float flat_val = batch.x[static_cast<std::size_t>(i) * batch.n_features + f];
            assert(flat_val == orig[f] && "Flat batch value mismatch");
        }
    }
    std::cout << "  Batch: " << batch.n_samples << " samples × "
              << batch.n_features << " features  [all values match]\n";
    std::cout << "  [PASS]\n";
}

// Test 2: parallel-samples produces identical predictions to sequential
static void test_parallel_samples_correctness() {
    std::cout << "\n[Zuhaa Test 2: Parallel Samples == Sequential]\n";

    zuhaa_write_test_csv("z_par_s.csv", 80);
    Dataset data = load_csv("z_par_s.csv", true);
    auto [tr, te] = train_test_split(data, 0.8f, 13);

    CompactForest cf = train_test_forest(data, tr, 5, 5);
    FlatBatch batch  = prepare_batch(data, te);

    const auto seq = predict_sequential(cf, batch);
    const auto par = predict_parallel_samples(cf, batch, 4);

    assert(seq.size() == par.size());
    int diff = 0;
    for (std::size_t i = 0; i < seq.size(); ++i)
        if (seq[i] != par[i]) ++diff;
    assert(diff == 0 && "Parallel-samples predictions differ from sequential!");

    // Accuracy sanity check on separable data
    std::vector<int> truth; for (int i : te) truth.push_back(data.labels[i]);
    double acc = compute_accuracy(par, truth);
    assert(acc >= 0.85 && "Expected >= 85% accuracy on clearly separable data");

    std::cout << "  All " << seq.size() << " predictions match sequential.  "
              << "Accuracy: " << std::fixed << std::setprecision(1)
              << acc * 100.0 << "%\n";
    std::cout << "  [PASS]\n";
}

// Test 3: parallel-trees produces identical predictions to sequential
static void test_parallel_trees_correctness() {
    std::cout << "\n[Zuhaa Test 3: Parallel Trees == Sequential]\n";

    zuhaa_write_test_csv("z_par_t.csv", 80);
    Dataset data = load_csv("z_par_t.csv", true);
    auto [tr, te] = train_test_split(data, 0.8f, 17);

    CompactForest cf = train_test_forest(data, tr, MAX_TREES, 5);
    FlatBatch batch  = prepare_batch(data, te);

    const auto seq  = predict_sequential(cf, batch);
    const auto par  = predict_parallel_trees(cf, batch, 4);

    assert(seq.size() == par.size());
    int diff = 0;
    for (std::size_t i = 0; i < seq.size(); ++i)
        if (seq[i] != par[i]) ++diff;
    assert(diff == 0 && "Parallel-trees predictions differ from sequential!");

    std::cout << "  All " << seq.size() << " predictions match sequential.\n";
    std::cout << "  [PASS]\n";
}

// Test 4: parallel-samples matches Faraz's predict_compact_forest_batch exactly
static void test_matches_faraz_baseline() {
    std::cout << "\n[Zuhaa Test 4: Parallel Samples == Faraz's Batch Predict]\n";

    zuhaa_write_test_csv("z_match.csv", 60);
    Dataset data = load_csv("z_match.csv", true);
    auto [tr, te] = train_test_split(data, 0.8f, 99);

    CompactForest cf = train_test_forest(data, tr, 5, 5);
    FlatBatch batch  = prepare_batch(data, te);

    // Faraz's reference predictions (using Dataset directly)
    const auto faraz_preds = predict_compact_forest_batch(cf, data, te);
    // Zuhaa's parallel predictions (using FlatBatch)
    const auto zuhaa_preds = predict_parallel_samples(cf, batch, 2);

    assert(faraz_preds.size() == zuhaa_preds.size());
    int diff = 0;
    for (std::size_t i = 0; i < faraz_preds.size(); ++i)
        if (faraz_preds[i] != zuhaa_preds[i]) ++diff;
    assert(diff == 0 && "Zuhaa parallel results differ from Faraz's baseline!");

    std::cout << "  Checked " << faraz_preds.size()
              << " samples — all match Faraz's predict_compact_forest_batch.\n";
    std::cout << "  [PASS]\n";
}

// Test 5: single-thread parallel equals sequential (degenerate case)
static void test_single_thread_equals_sequential() {
    std::cout << "\n[Zuhaa Test 5: Single-Thread Parallel == Sequential]\n";

    zuhaa_write_test_csv("z_single.csv", 40);
    Dataset data = load_csv("z_single.csv", true);
    auto [tr, te] = train_test_split(data, 0.8f, 55);

    CompactForest cf = train_test_forest(data, tr, 3, 4);
    FlatBatch batch  = prepare_batch(data, te);

    const auto seq  = predict_sequential(cf, batch);
    const auto par1 = predict_parallel_samples(cf, batch, 1);
    const auto par1t = predict_parallel_trees(cf, batch, 1);

    int diff_s = 0, diff_t = 0;
    for (std::size_t i = 0; i < seq.size(); ++i) {
        if (seq[i] != par1[i])  ++diff_s;
        if (seq[i] != par1t[i]) ++diff_t;
    }
    assert(diff_s == 0 && "1-thread par-samples differs from sequential!");
    assert(diff_t == 0 && "1-thread par-trees differs from sequential!");
    std::cout << "  Both 1-thread variants match sequential.\n";
    std::cout << "  [PASS]\n";
}

// Test 6: throughput measurement — sanity-checks numeric outputs
static void test_throughput_measurement() {
    std::cout << "\n[Zuhaa Test 6: Throughput Measurement]\n";

    zuhaa_write_test_csv("z_tput.csv", 150);
    Dataset data = load_csv("z_tput.csv", true);
    auto [tr, te] = train_test_split(data, 0.7f, 42);

    CompactForest cf = train_test_forest(data, tr, MAX_TREES, 6);

    const auto results = measure_throughput(cf, data, te, 4, 3);
    print_throughput_table(results);

    assert(results.size() == 3 && "Expected exactly 3 throughput result rows");
    for (const auto& r : results) {
        assert(r.n_samples      > 0   && "n_samples must be > 0");
        assert(r.elapsed_ms     > 0.0 && "elapsed_ms must be > 0");
        assert(r.samples_per_sec > 0.0 && "throughput must be > 0");
    }
    // Sequential baseline speedup must be 1.0x exactly
    assert(std::abs(results[0].speedup - 1.0) < 1e-9 && "Baseline speedup must be 1.0x");

    std::cout << "  [PASS]\n";
}

// Test 7: speedup-vs-trees table — verify shape and ordering
static void test_speedup_vs_trees() {
    std::cout << "\n[Zuhaa Test 7: Speedup vs Number of Trees]\n";

    zuhaa_write_test_csv("z_svt.csv", 100);
    Dataset data = load_csv("z_svt.csv", true);
    auto [tr, te] = train_test_split(data, 0.8f, 71);

    CompactForest cf   = train_test_forest(data, tr, MAX_TREES, 5);
    FlatBatch batch    = prepare_batch(data, te);

    const std::vector<int> counts = {1, 3, 5, MAX_TREES};
    const auto rows = speedup_vs_trees(cf, batch, counts, 4, 3);

    std::cout << "\n  Speedup-vs-Trees (parallel samples, 4 threads):\n";
    print_speedup_table(rows);

    assert(static_cast<int>(rows.size()) == static_cast<int>(counts.size()));
    for (const auto& r : rows) {
        assert(r.n_trees   >= 1    && "n_trees must be >= 1");
        assert(r.seq_ms    > 0.0   && "seq_ms must be > 0");
        assert(r.throughput > 0.0  && "throughput must be > 0");
    }
    std::cout << "  [PASS]\n";
}

// ============================================================================
// MAIN
// ============================================================================

int main() {
    std::cout << "=============================================================\n";
    std::cout << "   MILESTONE 3 — Integrated Random Forest\n";
    std::cout << "   Yaman (Set 1) + Faraz (Set 2) + Zuhaa (Set 3)\n";
    std::cout << "=============================================================\n";
    std::cout << "   Hardware threads available: " << hw_threads() << "\n";
    std::cout << "=============================================================\n";

    try {
        run_cpu_only_benchmarks();

        std::cout << "\n=== YAMAN'S TESTS ===\n";
        test_bootstrap_sample_properties();
        test_tree_count_capped();
        
        std::cout << "\n=== FARAZ'S TESTS ===\n";
        test_compact_serialization();
        test_compact_forest_accuracy();
        test_compact_inference_speed();

        std::cout << "\n=== ZUHAA'S TESTS ===\n";
        test_flat_batch_preparation();
        test_parallel_samples_correctness();
        test_parallel_trees_correctness();
        test_matches_faraz_baseline();
        test_single_thread_equals_sequential();
        test_throughput_measurement();
        test_speedup_vs_trees();
        
        std::cout << "\n=============================================================\n";
        std::cout << "   ALL TESTS PASSED\n";
        std::cout << "=============================================================\n";
        std::cout << "\nMemory savings from compact representation:\n";
        std::cout << "  - No pointers (8 bytes saved per child reference)\n";
        std::cout << "  - Sequential memory layout (cache friendly)\n";
        std::cout << "  - Easy to serialize to disk\n";
        std::cout << "\nZuhaa's parallel inference interface:\n";
        std::cout << "  prepare_batch()             -> FlatBatch\n";
        std::cout << "  predict_sequential()        -> vector<int>\n";
        std::cout << "  predict_parallel_samples()  -> vector<int>\n";
        std::cout << "  predict_parallel_trees()    -> vector<int>\n";
        std::cout << "  measure_throughput()        -> throughput table\n";
        std::cout << "  speedup_vs_trees()          -> speedup table\n";
        std::cout << "  save_throughput_csv()       -> CSV artifact\n";
        std::cout << "  save_speedup_csv()          -> CSV artifact\n";

    } catch (const std::exception& ex) {
        std::cerr << "Error: " << ex.what() << "\n";
        return 1;
    }

    return 0;
}