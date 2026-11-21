// Batched tree traversal/expansion; neural inference and exploration RNG stay in Python.
#include <array>
#include <cmath>
#include <cstdint>
#include <exception>
#include <stdexcept>
#include <string>
#include <vector>

namespace {
thread_local std::string error;
bool won(uint64_t bits) {
    for (int shift : {7, 8, 6, 1}) {
        uint64_t pair = bits & (bits >> shift);
        if (pair & (pair >> (2 * shift))) return true;
    }
    return false;
}
struct Node {
    uint64_t mask, position;
    int moves, parent, visits = 0;
    double prior, value_sum = 0.;
    std::array<int, 7> children;
    bool expanded = false, terminal;
    double result;
    Node(uint64_t m, uint64_t p, int ply, int par, double probability)
        : mask(m), position(p), moves(ply), parent(par), prior(probability) {
        children.fill(-1);
        bool win = moves >= 7 && won(mask ^ position);
        terminal = win || moves == 42;
        result = win ? -1. : 0.;
    }
    bool legal(int col) const { return !(mask & (uint64_t(1) << (7 * col + 5))); }
};
struct Forest {
    std::vector<Node> nodes;
    std::vector<int> leaves, evaluated_trees;
    std::vector<double> values;
    int trees, evaluated = 0;
    double c_puct;
    Forest(int count, int simulations, double cp) : leaves(count), evaluated_trees(count),
        values(count), trees(count), c_puct(cp) {
        nodes.reserve(size_t(count) * (1 + 7 * (size_t(simulations) + 1)));
    }
    int select(int id) const {
        while (nodes[id].expanded) {
            const Node& parent = nodes[id];
            double best = -INFINITY, sqrt_visits = std::sqrt(double(parent.visits));
            int chosen = -1;
            for (int child_id : parent.children) {
                if (child_id < 0) continue;
                const Node& child = nodes[child_id];
                double q = child.visits ? child.value_sum / child.visits : 0.;
                double score = -q + c_puct * child.prior * sqrt_visits / (1 + child.visits);
                if (score > best) { best = score; chosen = child_id; }
            }
            id = chosen;
        }
        return id;
    }
};
}

extern "C" {
const char* cx_error() { return error.c_str(); }
int cx_version() { return 1; }
void* cx_create(const uint64_t* masks, const uint64_t* positions, const int* moves,
                int count, int simulations, double c_puct) {
    Forest* forest = nullptr;
    try {
        if (count <= 0 || simulations < 0 || !std::isfinite(c_puct) || c_puct < 0)
            throw std::invalid_argument("Invalid search dimensions or PUCT constant");
        forest = new Forest(count, simulations, c_puct);
        uint64_t valid = 0;
        for (int col = 0; col < 7; ++col) valid |= uint64_t(63) << (col * 7);
        for (int i = 0; i < count; ++i) {
            if ((masks[i] & ~valid) || (positions[i] & ~masks[i]) || moves[i] < 0 || moves[i] > 42
                || __builtin_popcountll(masks[i]) != moves[i])
                throw std::invalid_argument("Invalid bitboard");
            for (int col = 0; col < 7; ++col) {
                uint64_t occupied = (masks[i] >> (col * 7)) & 63;
                if (occupied & (occupied + 1)) throw std::invalid_argument("Invalid bitboard gravity");
            }
            forest->nodes.emplace_back(masks[i], positions[i], moves[i], -1, 0.);
            if (forest->nodes.back().terminal) throw std::invalid_argument("Terminal search root");
        }
        return forest;
    } catch (const std::exception& e) {
        error = e.what(); delete forest; return nullptr;
    }
}
void cx_destroy(void* handle) { delete static_cast<Forest*>(handle); }
int cx_prepare(void* handle, float* states, uint8_t* legal) {
    auto& f = *static_cast<Forest*>(handle);
    f.evaluated = 0;
    for (int tree = 0; tree < f.trees; ++tree) {
        int id = f.select(tree);
        f.leaves[tree] = id;
        const Node& node = f.nodes[id];
        f.values[tree] = node.result;
        if (node.terminal) continue;
        int batch = f.evaluated++;
        f.evaluated_trees[batch] = tree;
        for (int col = 0; col < 7; ++col) {
            legal[batch * 7 + col] = node.legal(col);
            for (int row = 0; row < 6; ++row) {
                uint64_t bit = uint64_t(1) << (col * 7 + row);
                int cell = batch * 126 + row * 7 + col;
                states[cell] = (node.position & bit) != 0;
                states[cell + 42] = ((node.mask ^ node.position) & bit) != 0;
                states[cell + 84] = node.moves % 2 == 0;
            }
        }
    }
    return f.evaluated;
}
int cx_finish(void* handle, const float* policies, const float* values,
              const double* noise, double epsilon, int backup) {
    try {
        auto& f = *static_cast<Forest*>(handle);
        for (int i = 0; i < f.evaluated; ++i) {
            if (!std::isfinite(values[i])) throw std::invalid_argument("Non-finite network value");
            int tree = f.evaluated_trees[i], id = f.leaves[tree];
            // Copy because appending children can invalidate references into the vector.
            Node parent = f.nodes[id];
            for (int col = 0; col < 7; ++col) {
                if (!parent.legal(col)) continue;
                uint64_t bottom = uint64_t(1) << (col * 7);
                uint64_t piece = (parent.mask + bottom) & (uint64_t(63) << (col * 7));
                double prior = policies[i * 7 + col];
                if (noise) prior = (1. - epsilon) * prior + epsilon * noise[i * 7 + col];
                if (!std::isfinite(prior) || prior < 0.) throw std::invalid_argument("Invalid network prior");
                int child = int(f.nodes.size());
                f.nodes.emplace_back(parent.mask | piece, parent.mask ^ parent.position,
                                     parent.moves + 1, id, prior);
                f.nodes[id].children[col] = child;
            }
            f.nodes[id].expanded = true;
            f.values[tree] = values[i];
        }
        if (backup) for (int tree = 0; tree < f.trees; ++tree) {
            double value = f.values[tree];
            for (int id = f.leaves[tree]; id >= 0; id = f.nodes[id].parent) {
                f.nodes[id].visits++;
                f.nodes[id].value_sum += value;
                value = -value;
            }
        }
        return 0;
    } catch (const std::exception& e) { error = e.what(); return -1; }
}
void cx_visits(void* handle, float* visits) {
    auto& f = *static_cast<Forest*>(handle);
    for (int tree = 0; tree < f.trees; ++tree) for (int col = 0; col < 7; ++col) {
        int id = f.nodes[tree].children[col];
        visits[tree * 7 + col] = id < 0 ? 0 : f.nodes[id].visits;
    }
}
}
