// Adaptive sharp-feature dual contouring on a sparse, 2:1 balanced octree
// (SDF-DESIGN.md, "Meshing").
//
// Lattice points are named by 60-bit keys of their coordinates on the finest
// level (20 bits per axis, from a grid origin on the shared world lattice).
// A leaf of level l is a cube of 2^(D - l) fine units; leaves never share a
// minimum corner, so that corner names a leaf. A leaf's boundary is read on
// the grid of points half its size apart: its corners, the midpoints of its
// edges that finer leaves split, and the centres of its faces whose
// neighbour is split. Corner values, crossings and face-centre values are
// cached by lattice key across refinement passes, and leaves whose boundary
// did not change keep their vertices and measured triangles.
#include "mesh.h"

#include <algorithm>

namespace rsdf {

void MeshField::evaluate(int n, const double *x, const double *y, const double *z, double *out) const {
    // eval_batch takes at most kBatch points per call.
    for (int first = 0; first < n; first += kBatch)
        eval_batch(*graph, node, std::min(kBatch, n - first), x + first, y + first, z + first, out + first);
    for (int i = 0; i < n; ++i) {
        const double qx = std::fmax(lo[0] - x[i], x[i] - hi[0]);
        const double qy = std::fmax(lo[1] - y[i], y[i] - hi[1]);
        const double qz = std::fmax(lo[2] - z[i], z[i] - hi[2]);
        const double ox = std::fmax(qx, 0.0), oy = std::fmax(qy, 0.0), oz = std::fmax(qz, 0.0);
        const double box = std::sqrt(ox * ox + oy * oy + oz * oz) + std::fmin(std::fmax(qx, std::fmax(qy, qz)), 0.0);
        out[i] = std::fmax(out[i], box);
    }
}

BatchField MeshField::batch() const {
    return [this](int n, const double *x, const double *y, const double *z, double *out) { evaluate(n, x, y, z, out); };
}

namespace {

constexpr uint32_t kNone = UINT32_MAX;
constexpr int kAxisBits = 20;
constexpr uint64_t kAxisMask = (uint64_t(1) << kAxisBits) - 1;
constexpr int kMaxRepairPasses = 64;
constexpr double kHalfDiagonal = 0.8660254037844386;  // sqrt(3) / 2
// Work goes to the threads in chunks small enough that a pass over a few
// thousand leaves of a costly field (microseconds per evaluation) still
// spreads over every core; results never depend on the chunking.
constexpr size_t kChunk = 1024;
constexpr size_t kLeafChunk = 256;
constexpr int kSlots = 48;
constexpr uint8_t kNoComponent = 0xff;
// Grid planes lie at (phi + k / 2^l) h for level l, phi = 2 - golden ratio:
// the offset is in base cells so that the levels nest, and every mesh of
// the same cell size shares the lattice. Faces at simple coordinates never
// lie on a grid plane: a face through grid corners makes crossings land
// exactly on sharp edges, where finite-difference normals are blends that
// misplace vertices.
constexpr double kGridShift = 0.3819660112501051;
constexpr double kBudgetMargin = 1.25;  // growth allowance for balance splits when budgeting refinement
constexpr double kTopology = 1e300;     // refinement score of leaves at non-manifold edges
// Refinement rounds beyond the depth: splits spread from refined regions to
// their coarser neighbours for a few rounds (on the Minotaur's strap pocket
// the last triangles beyond the tolerance went in the fourth), then each
// round splits a handful of cells.
constexpr int kExtraRounds = 4;
constexpr double kNormalAngle = 20;     // degrees a triangle may turn from the field normal at its centroid
constexpr double kFeatureAngle = 8;     // the same for a triangle with a vertex on a sharp feature
// QEF eigenvalues below 5% of the largest are dropped: tangent planes of a
// curved patch then meet near its mass point, and two planes meeting at an
// angle theta (a ratio of tan^2(theta / 2)) are resolved from 26 degrees.
// Shallower creases are resolved by crease_point.
constexpr double kTruncation = 0.05;
constexpr double kCreaseTurn = 4;  // degrees: the shallowest crease a vertex is placed on

inline uint64_t point_key(uint64_t x, uint64_t y, uint64_t z) {
    return x | y << kAxisBits | z << (2 * kAxisBits);
}
inline uint64_t point_key(const uint64_t c[3]) { return point_key(c[0], c[1], c[2]); }
inline void point_coords(uint64_t key, uint64_t c[3]) {
    c[0] = key & kAxisMask;
    c[1] = key >> kAxisBits & kAxisMask;
    c[2] = key >> (2 * kAxisBits) & kAxisMask;
}
inline uint64_t node_key(uint64_t point, int level) { return point | uint64_t(level) << 60; }
inline uint64_t morton_of(uint64_t key) {
    uint64_t c[3];
    point_coords(key, c);
    return morton(c);
}

template <class T>
std::vector<T> concatenate(std::vector<std::vector<T>> &pieces) {
    size_t total = 0;
    for (const auto &piece : pieces) total += piece.size();
    std::vector<T> all;
    all.reserve(total);
    for (auto &piece : pieces) {
        all.insert(all.end(), piece.begin(), piece.end());
        std::vector<T>().swap(piece);
    }
    return all;
}

[[noreturn]] void too_many_cells(uint64_t limit) {
    throw Error(format("the narrow band needs more than max_cells (%llu) cells; increase cell_size",
                       static_cast<unsigned long long>(limit)));
}

// Open-addressing map from 64-bit keys (never UINT64_MAX) to 32-bit values.
// Lookups may run concurrently; insertions are serial.
class KeyMap {
public:
    KeyMap() { reset(0); }

    void reset(size_t expected) {
        size_t capacity = 16;
        while (capacity < 2 * expected + 2) capacity *= 2;
        keys_.assign(capacity, kEmpty);
        values_.assign(capacity, 0);
        mask_ = capacity - 1;
        size_ = 0;
    }

    void reserve(size_t count) {
        while (2 * count + 2 > keys_.size()) grow();
    }

    size_t size() const { return size_; }

    uint32_t find(uint64_t key) const {
        for (size_t i = slot(key);; i = (i + 1) & mask_) {
            const uint64_t k = keys_[i];
            if (k == key) return values_[i];
            if (k == kEmpty) return kNone;
        }
    }

    // Stores key -> value unless the key is present; returns the stored value.
    uint32_t insert(uint64_t key, uint32_t value) {
        if (2 * (size_ + 1) > keys_.size()) grow();
        for (size_t i = slot(key);; i = (i + 1) & mask_) {
            if (keys_[i] == key) return values_[i];
            if (keys_[i] == kEmpty) {
                keys_[i] = key;
                values_[i] = value;
                ++size_;
                return value;
            }
        }
    }

private:
    static constexpr uint64_t kEmpty = UINT64_MAX;

    size_t slot(uint64_t key) const {
        key ^= key >> 33;
        key *= 0xff51afd7ed558ccdULL;
        key ^= key >> 33;
        key *= 0xc4ceb9fe1a85ec53ULL;
        key ^= key >> 33;
        return static_cast<size_t>(key) & mask_;
    }

    void grow() {
        std::vector<uint64_t> keys(2 * keys_.size(), kEmpty);
        std::vector<uint32_t> values(keys.size());
        keys.swap(keys_);
        values.swap(values_);
        mask_ = keys_.size() - 1;
        for (size_t i = 0; i < keys.size(); ++i) {
            if (keys[i] == kEmpty) continue;
            size_t j = slot(keys[i]);
            while (keys_[j] != kEmpty) j = (j + 1) & mask_;
            keys_[j] = keys[i];
            values_[j] = values[i];
        }
    }

    std::vector<uint64_t> keys_;
    std::vector<uint32_t> values_;
    size_t mask_ = 0, size_ = 0;
};

// ------------------------------------------------------------------ leaf boundary tables
//
// Boundary grid point g = i + 3 j + 9 k, (i, j, k) in {0, 1, 2}^3 half-leaf
// steps from the minimum corner. Leaf edge e = 4 axis + o_lo + 2 o_hi, with
// o_lo, o_hi the edge's offsets along the other two axes in increasing axis
// order. Face f = 2 axis + side, with in-face axes u = axis + 1, v = axis + 2
// (mod 3); its corners k = 0..3 sit at (u, v) = (0, 0), (1, 0), (1, 1),
// (0, 1) and face edge k joins corners k and k + 1. Slots (sub-edges):
// 2 e and 2 e + 1 for the halves of edge e (2 e alone when it is not split),
// 24 + 4 f + k for the half-edge from the centre of a split face f to the
// midpoint of its face edge k.

struct Tables {
    uint8_t edge_axis[12], edge_lo_axis[12], edge_hi_axis[12], edge_lo_offset[12], edge_hi_offset[12];
    uint8_t edge_start[12], edge_mid[12], edge_end[12];  // grid points
    uint8_t face_edge[6][4];      // leaf edge of face edge k
    uint8_t face_corner[6][4];    // grid point of face corner k
    uint8_t face_mid[6][4];       // grid point of the midpoint of face edge k
    uint8_t face_centre[6];
    uint8_t start_half[6][4], end_half[6][4];  // slot of the half of face edge k at its start / end corner
    uint8_t cross_lo[6][4], cross_hi[6][4], cross_axis[6][4];
};

inline int grid_point(const int c[3]) { return c[0] + 3 * c[1] + 9 * c[2]; }

int local_edge(int axis, const int offset[3]) {
    const int lo = axis == 0 ? 1 : 0, hi = axis == 2 ? 1 : 2;
    return 4 * axis + offset[lo] + 2 * offset[hi];
}

const Tables &tables() {
    static const Tables table = [] {
        Tables t{};
        for (int e = 0; e < 12; ++e) {
            const int axis = e / 4, lo = axis == 0 ? 1 : 0, hi = axis == 2 ? 1 : 2;
            t.edge_axis[e] = static_cast<uint8_t>(axis);
            t.edge_lo_axis[e] = static_cast<uint8_t>(lo);
            t.edge_hi_axis[e] = static_cast<uint8_t>(hi);
            t.edge_lo_offset[e] = static_cast<uint8_t>(e & 1);
            t.edge_hi_offset[e] = static_cast<uint8_t>(e >> 1 & 1);
            int c[3] = {0, 0, 0};
            c[lo] = 2 * (e & 1);
            c[hi] = 2 * (e >> 1 & 1);
            for (int step = 0; step < 3; ++step) {
                c[axis] = step;
                const uint8_t g = static_cast<uint8_t>(grid_point(c));
                (step == 0 ? t.edge_start : step == 1 ? t.edge_mid : t.edge_end)[e] = g;
            }
        }
        static const int kCycle[4][2] = {{0, 0}, {1, 0}, {1, 1}, {0, 1}};
        for (int axis = 0; axis < 3; ++axis)
            for (int side = 0; side < 2; ++side) {
                const int f = 2 * axis + side, u = (axis + 1) % 3, v = (axis + 2) % 3;
                int centre[3];
                centre[axis] = 2 * side;
                centre[u] = centre[v] = 1;
                t.face_centre[f] = static_cast<uint8_t>(grid_point(centre));
                for (int k = 0; k < 4; ++k) {
                    int corner[3];
                    corner[axis] = 2 * side;
                    corner[u] = 2 * kCycle[k][0];
                    corner[v] = 2 * kCycle[k][1];
                    t.face_corner[f][k] = static_cast<uint8_t>(grid_point(corner));
                    const int along = k % 2 == 0 ? u : v, across = k % 2 == 0 ? v : u;
                    const int across_side = k == 1 || k == 2 ? 1 : 0;
                    int offset[3];
                    offset[axis] = side;
                    offset[along] = 0;
                    offset[across] = across_side;
                    const int e = local_edge(along, offset);
                    t.face_edge[f][k] = static_cast<uint8_t>(e);
                    int mid[3];
                    mid[axis] = 2 * side;
                    mid[along] = 1;
                    mid[across] = 2 * across_side;
                    t.face_mid[f][k] = static_cast<uint8_t>(grid_point(mid));
                    // Corner k lies at the lower end of face edge k for k = 0, 1.
                    const bool starts_low = k < 2;
                    t.start_half[f][k] = static_cast<uint8_t>(2 * e + (starts_low ? 0 : 1));
                    t.end_half[f][k] = static_cast<uint8_t>(2 * e + (starts_low ? 1 : 0));
                    // The half-edge from the centre to that midpoint runs along `across`.
                    t.cross_axis[f][k] = static_cast<uint8_t>(across);
                    const bool mid_low = across_side == 0;
                    t.cross_lo[f][k] = mid_low ? t.face_mid[f][k] : t.face_centre[f];
                    t.cross_hi[f][k] = mid_low ? t.face_centre[f] : t.face_mid[f][k];
                }
            }
        return t;
    }();
    return table;
}

// Status bits of a leaf: bit e for a split edge e, bit 12 + f for a split face f.
inline bool edge_split(uint32_t status, int e) { return status >> e & 1; }
inline bool face_split(uint32_t status, int f) { return status >> (12 + f) & 1; }

// The grid points at the lower and upper end of an active slot along its
// axis, its axis and whether it is half the leaf. False for inactive slots.
inline bool slot_geometry(int slot, uint32_t status, int &lo, int &hi, int &axis, bool &half) {
    const Tables &t = tables();
    if (slot < 24) {
        const int e = slot / 2;
        const bool split = edge_split(status, e);
        axis = t.edge_axis[e];
        half = split;
        if (!split) {
            if (slot & 1) return false;
            lo = t.edge_start[e];
            hi = t.edge_end[e];
        } else {
            lo = slot & 1 ? t.edge_mid[e] : t.edge_start[e];
            hi = slot & 1 ? t.edge_end[e] : t.edge_mid[e];
        }
        return true;
    }
    const int f = (slot - 24) / 4, k = (slot - 24) % 4;
    if (!face_split(status, f)) return false;
    lo = t.cross_lo[f][k];
    hi = t.cross_hi[f][k];
    axis = t.cross_axis[f][k];
    half = true;
    return true;
}

inline void grid_coords(int g, int c[3]) {
    c[0] = g % 3;
    c[1] = g / 3 % 3;
    c[2] = g / 9;
}

// ------------------------------------------------------------------ extraction

struct Crossing {
    double p[3];
    double n[3];
    // The field jumps across the bracket by more than its Lipschitz bound
    // allows: a discontinuity, not a root, which refinement cannot resolve.
    bool jump;
};

struct EdgeJob {
    uint64_t point;  // lower end
    uint32_t level, axis;
    double f0, f1;   // values at the lower and upper end
};

struct VertexRecord {
    double position[3];
    double crease[3];   // the point on a shallow crease (crease_point)
    double guide[3];
    double spread[4];   // widest pair of crossing normals: cosine, difference
    double box[6];      // projection box: lo, hi
    double side;        // sample offset for the sharpness test
    bool jump;          // a crossing is a discontinuity of the field
    double residual;    // largest crossing-plane distance from the QEF point
};

// Previous-pass data kept for leaves whose boundary does not change.
struct PassCache {
    KeyMap leaf_index;  // min corner -> leaf
    std::vector<uint8_t> level;
    std::vector<uint32_t> status;
    std::vector<uint32_t> first_vertex;
    std::vector<uint8_t> vertex_count;
    std::vector<uint32_t> table;        // into tables, or kNone
    std::vector<uint8_t> tables;        // kSlots per active leaf
    std::vector<double> positions;      // 3 per vertex
    std::vector<uint8_t> marks;
    std::vector<double> residuals, values, slopes, normals;
    // Measured triangles by crossing index of their sign-change edge.
    std::vector<uint32_t> result_of_crossing;
    std::vector<uint8_t> results_per_crossing;
    std::vector<TriangleResult> results;
    std::vector<uint8_t> flipped;
};

class Extractor {
public:
    Extractor(const MeshField &field, const MeshSettings &settings) : field_(field), s_(settings) {
        depth_ = std::max(0, settings.max_depth);
        h_ = settings.cell_size;
        fine_ = std::ldexp(h_, -depth_);
        const uint64_t limit = (kAxisMask >> depth_) - 8;
        for (int a = 0; a < 3; ++a) {
            const double first = std::floor(field.lo[a] / h_ - kGridShift) - 2;
            const double span = std::ceil(field.hi[a] / h_ - kGridShift - first) + 2;
            if (!(span <= double(limit)) || !(std::fabs(first) < 0x1p40))
                throw Error(format("meshing needs too many cells along %c (more than %llu); increase cell_size",
                                   "xyz"[a], static_cast<unsigned long long>(limit)));
            base_origin_[a] = static_cast<int64_t>(first);
            origin_[a] = base_origin_[a] * (int64_t(1) << depth_);
            cells_[a] = static_cast<uint32_t>(span);
        }
        morton_bits_ = 3 * (bit_width(std::max({cells_[0], cells_[1], cells_[2]})) + depth_);
        cos_normal_ = std::cos(kNormalAngle * M_PI / 180);
        cos_feature_ = std::cos(kFeatureAngle * M_PI / 180);
        cos_crease_ = std::cos(settings.crease_angle * M_PI / 180);
    }

    Surface run() {
        roots_ = narrow_band();
        for (uint64_t root : roots_) add_node(root, 0);
        for (;;) {
            balance();
            collect_leaves();
            build_points();
            build_status();
            build_edges_and_faces();
            build_vertices();
            build_triangles();
            measure();
            ++passes_;
            if (!missing_.empty() && repairs_ < kMaxRepairPasses) {
                // Cells around sign changes that the narrow band missed: add them and extract again.
                ++repairs_;
                for (uint64_t root : missing_) add_node(root, 0);
                if (roots_.size() + missing_.size() > s_.max_cells) too_many_cells(s_.max_cells);
                roots_.insert(roots_.end(), missing_.begin(), missing_.end());
                keep_cache();
                continue;
            }
            if (!refine(rounds_ < depth_ + kExtraRounds)) break;
            ++rounds_;
            keep_cache();
        }
        return finish();
    }

private:
    // ---------------------------------------------------------------- lattice

    uint64_t size_of(int level) const { return uint64_t(1) << (depth_ - level); }

    // World coordinate of fine lattice index `c` along `axis`, from the global
    // index so that meshes of the same cell size agree to the bit.
    double coordinate(int axis, uint64_t c) const {
        return h_ * (kGridShift + std::ldexp(static_cast<double>(origin_[axis] + static_cast<int64_t>(c)), -depth_));
    }

    bool in_range(const int64_t c[3]) const {
        for (int a = 0; a < 3; ++a)
            if (c[a] < 0 || c[a] >= (int64_t(cells_[a]) << depth_)) return false;
        return true;
    }

    // ---------------------------------------------------------------- narrow band

    // Hierarchical narrow band at the base level: blocks whose centre value
    // proves they cannot reach the surface are dropped before subdividing.
    // Returns the min-corner keys of the candidate base cells.
    std::vector<uint64_t> narrow_band() {
        uint32_t side = 1;
        auto blocks = [&](int a, uint32_t size) { return (cells_[a] + size - 1) / size; };
        while (blocks(0, side) > 64 || blocks(1, side) > 64 || blocks(2, side) > 64) side *= 2;
        std::vector<uint64_t> level;
        for (uint64_t code = 0; code < (uint64_t(1) << 18); ++code) {
            const uint64_t c[3] = {compact3(code), compact3(code >> 1), compact3(code >> 2)};
            if (c[0] < blocks(0, side) && c[1] < blocks(1, side) && c[2] < blocks(2, side)) level.push_back(code);
        }
        std::vector<uint8_t> keep;
        while (true) {
            const double size = side * h_;
            const double reach = s_.lipschitz * kHalfDiagonal * size * (1 + 1e-9);
            keep.assign(level.size(), 0);
            parallel_chunks(chunks_of(level.size(), kChunk), s_.threads, [&](size_t chunk) {
                double x[kBatch], y[kBatch], z[kBatch], v[kBatch];
                const size_t end = std::min(level.size(), (chunk + 1) * kChunk);
                for (size_t first = chunk * kChunk; first < end; first += kBatch) {
                    const int n = static_cast<int>(std::min<size_t>(kBatch, end - first));
                    for (int i = 0; i < n; ++i) {
                        const uint64_t code = level[first + i];
                        // Block centre: half a block from its minimum corner.
                        const uint64_t c[3] = {compact3(code), compact3(code >> 1), compact3(code >> 2)};
                        double *out[3] = {x, y, z};
                        for (int a = 0; a < 3; ++a)
                            out[a][i] = h_ * (kGridShift + (static_cast<double>(base_origin_[a]) + (c[a] + 0.5) * side));
                    }
                    field_.evaluate(n, x, y, z, v);
                    for (int i = 0; i < n; ++i) keep[first + i] = std::fabs(v[i]) <= reach;
                }
            });
            size_t kept = 0;
            for (uint8_t k : keep) kept += k;
            if (side == 1) {
                if (kept > s_.max_cells) too_many_cells(s_.max_cells);
                std::vector<uint64_t> cells;
                cells.reserve(kept);
                for (size_t i = 0; i < level.size(); ++i) {
                    if (!keep[i]) continue;
                    const uint64_t c[3] = {compact3(level[i]) << depth_, compact3(level[i] >> 1) << depth_,
                                           compact3(level[i] >> 2) << depth_};
                    cells.push_back(point_key(c));
                }
                return cells;
            }
            if (kept > s_.max_cells / 2) too_many_cells(s_.max_cells);  // children would exceed 4 x max_cells
            const uint32_t half = side / 2;
            std::vector<uint64_t> next;
            next.reserve(kept * 8);
            for (size_t i = 0; i < level.size(); ++i) {
                if (!keep[i]) continue;
                for (uint64_t d = 0; d < 8; ++d) {  // Morton order is preserved by subdivision
                    const uint64_t child = level[i] << 3 | d;
                    if (compact3(child) * half < cells_[0] && compact3(child >> 1) * half < cells_[1] &&
                        compact3(child >> 2) * half < cells_[2])
                        next.push_back(child);
                }
            }
            level.swap(next);
            side = half;
        }
    }

    // ---------------------------------------------------------------- octree

    bool node_exists(uint64_t point, int level) const { return nodes_.find(node_key(point, level)) != kNone; }

    void add_node(uint64_t point, int level) {
        if (nodes_.insert(node_key(point, level), 0) == 0 && nodes_.size() > node_count_) {
            node_count_ = nodes_.size();
            if (level_nodes_.size() <= size_t(level)) level_nodes_.resize(level + 1);
            level_nodes_[level].push_back(point);
        }
    }

    // Splits node (point, level) into its eight children.
    void split(uint64_t point, int level) {
        const uint64_t half = size_of(level + 1);
        uint64_t c[3];
        point_coords(point, c);
        for (int d = 0; d < 8; ++d)
            add_node(point_key(c[0] + (d & 1) * half, c[1] + (d >> 1 & 1) * half, c[2] + (d >> 2) * half), level + 1);
    }

    // Makes node (point, level) exist by splitting its ancestors, unless its
    // base cell is not a root. Returns the splits made.
    uint64_t ensure(uint64_t point, int level) {
        uint64_t c[3];
        point_coords(point, c);
        auto ancestor = [&](int m) {
            const uint64_t mask = ~(size_of(m) - 1);
            return point_key(c[0] & mask, c[1] & mask, c[2] & mask);
        };
        if (!node_exists(ancestor(0), 0)) return 0;
        uint64_t splits = 0;
        for (int m = 0; m < level; ++m) {
            const uint64_t a = ancestor(m);
            if (!node_exists(a, m + 1)) {
                split(a, m);
                ++splits;
            }
        }
        return splits;
    }

    // 2:1 balance across faces, edges and corners: the 26 neighbours of every
    // internal node exist at its level, so leaves that touch differ by at most
    // one level. Finest internal nodes first; splits only create coarser
    // internal nodes, which later sweeps visit.
    void balance() {
        for (int level = depth_ - 1; level >= 1; --level) {
            if (level_nodes_.size() <= size_t(level + 1)) continue;
            const std::vector<uint64_t> &children = level_nodes_[level + 1];
            // Internal nodes at `level`: the first child of each sibling group.
            std::vector<uint64_t> parents;
            parents.reserve(children.size() / 8);
            for (size_t i = 0; i < children.size(); i += 8) parents.push_back(children[i]);
            const int64_t size = static_cast<int64_t>(size_of(level));
            const size_t parts = chunks_of(parents.size(), kLeafChunk);
            std::vector<std::vector<uint64_t>> pieces(parts);
            parallel_chunks(parts, s_.threads, [&](size_t part) {
                const size_t end = std::min(parents.size(), (part + 1) * kLeafChunk);
                for (size_t i = part * kLeafChunk; i < end; ++i) {
                    uint64_t c[3];
                    point_coords(parents[i], c);
                    for (int d = 0; d < 27; ++d) {
                        if (d == 13) continue;
                        const int64_t q[3] = {int64_t(c[0]) + (d % 3 - 1) * size, int64_t(c[1]) + (d / 3 % 3 - 1) * size,
                                              int64_t(c[2]) + (d / 9 - 1) * size};
                        if (!in_range(q)) continue;
                        const uint64_t key = point_key(uint64_t(q[0]), uint64_t(q[1]), uint64_t(q[2]));
                        if (!node_exists(key, level)) pieces[part].push_back(key);
                    }
                }
            });
            std::vector<uint64_t> wanted = concatenate(pieces);
            std::sort(wanted.begin(), wanted.end());
            wanted.erase(std::unique(wanted.begin(), wanted.end()), wanted.end());
            for (uint64_t key : wanted) balanced_ += ensure(key, level);
        }
    }

    // Leaves (nodes without children) in Morton order of their min corners.
    void collect_leaves() {
        std::vector<uint64_t> items;  // Morton code of the min corner; index into `points` below
        std::vector<uint64_t> points;
        std::vector<uint8_t> levels;
        for (int level = 0; level < int(level_nodes_.size()); ++level)
            for (uint64_t p : level_nodes_[level]) {
                if (level < depth_ && node_exists(p, level + 1)) continue;
                points.push_back(p);
                levels.push_back(static_cast<uint8_t>(level));
            }
        if (points.size() >= kNone) throw Error("meshing needs too many octree cells; increase cell_size");
        std::vector<uint64_t> codes(points.size());
        std::vector<uint32_t> order(points.size());
        for (size_t i = 0; i < points.size(); ++i) {
            codes[i] = morton_of(points[i]);
            order[i] = static_cast<uint32_t>(i);
        }
        radix_sort(codes, &order, morton_bits_);
        const size_t count = points.size();
        leaf_key_.resize(count);
        leaf_level_.resize(count);
        for (size_t i = 0; i < count; ++i) {
            leaf_key_[i] = points[order[i]];
            leaf_level_[i] = levels[order[i]];
        }
        leaf_index_.reset(count);
        for (size_t i = 0; i < count; ++i) leaf_index_.insert(leaf_key_[i], static_cast<uint32_t>(i));
        max_level_ = 0;
        for (uint8_t level : leaf_level_) max_level_ = std::max<uint32_t>(max_level_, level);
    }

    // The leaf containing the fine cell with minimum corner c, or kNone.
    uint32_t locate(const int64_t c[3]) const {
        if (!in_range(c)) return kNone;
        for (int level = 0; level <= depth_; ++level) {
            const uint64_t mask = ~(size_of(level) - 1);
            const uint32_t leaf = leaf_index_.find(point_key(uint64_t(c[0]) & mask, uint64_t(c[1]) & mask,
                                                             uint64_t(c[2]) & mask));
            if (leaf == kNone) continue;
            if (leaf_level_[leaf] == level) return leaf;
        }
        return kNone;
    }

    // ---------------------------------------------------------------- corner values

    // Every boundary point a leaf reads is a corner of some leaf: the
    // midpoint of a split edge and the centre of a split face are corners of
    // the finer leaves that split them. New corners are evaluated in Morton order.
    void build_points() {
        // Corners of leaves new to this pass that are not known yet, as
        // Morton codes (sorted, unique).
        const size_t leaves = leaf_key_.size();
        const size_t parts = chunks_of(leaves, kLeafChunk);
        std::vector<std::vector<uint64_t>> pieces(parts);
        parallel_chunks(parts, s_.threads, [&](size_t part) {
            for (size_t i = part * kLeafChunk, end = std::min(leaves, (part + 1) * kLeafChunk); i < end; ++i) {
                if (!cache_.level.empty()) {  // a leaf of the previous pass has its corners already
                    const uint32_t previous = cache_.leaf_index.find(leaf_key_[i]);
                    if (previous != kNone && cache_.level[previous] == leaf_level_[i]) continue;
                }
                const uint64_t size = size_of(leaf_level_[i]);
                uint64_t c[3];
                point_coords(leaf_key_[i], c);
                for (int d = 0; d < 8; ++d) {
                    const uint64_t corner[3] = {c[0] + (d & 1) * size, c[1] + (d >> 1 & 1) * size, c[2] + (d >> 2) * size};
                    if (points_.find(point_key(corner)) == kNone) pieces[part].push_back(morton(corner));
                }
            }
        });
        std::vector<uint64_t> codes = concatenate(pieces);
        radix_sort_items(codes, 0, 3 * kAxisBits, s_.threads);
        codes.erase(std::unique(codes.begin(), codes.end()), codes.end());
        std::vector<uint64_t> fresh(codes.size());
        for (size_t i = 0; i < codes.size(); ++i) {
            const uint64_t c[3] = {compact3(codes[i]), compact3(codes[i] >> 1), compact3(codes[i] >> 2)};
            fresh[i] = point_key(c);
            points_.insert(fresh[i], static_cast<uint32_t>(point_values_.size() + i));
        }
        if (point_values_.size() + fresh.size() >= kNone) throw Error("meshing needs too many grid corners; increase cell_size");
        const size_t first = point_values_.size();
        point_values_.resize(first + fresh.size());
        std::vector<uint32_t> todo(fresh.size());  // already in Morton order
        for (size_t i = 0; i < fresh.size(); ++i) todo[i] = static_cast<uint32_t>(i);
        parallel_chunks(chunks_of(todo.size(), kChunk), s_.threads, [&](size_t chunk) {
            double x[kBatch], y[kBatch], z[kBatch], v[kBatch];
            const size_t end = std::min(todo.size(), (chunk + 1) * kChunk);
            for (size_t begin = chunk * kChunk; begin < end; begin += kBatch) {
                const int n = static_cast<int>(std::min<size_t>(kBatch, end - begin));
                for (int i = 0; i < n; ++i) {
                    uint64_t c[3];
                    point_coords(fresh[todo[begin + i]], c);
                    x[i] = coordinate(0, c[0]);
                    y[i] = coordinate(1, c[1]);
                    z[i] = coordinate(2, c[2]);
                }
                field_.evaluate(n, x, y, z, v);
                for (int i = 0; i < n; ++i) point_values_[first + todo[begin + i]] = v[i];
            }
        });
    }

    double point_value(uint64_t key) const {
        const uint32_t index = points_.find(key);
        if (index == kNone) throw Error("internal error: a leaf boundary point is missing");
        return point_values_[index];
    }

    // ---------------------------------------------------------------- leaf boundaries

    // Which edges and faces of each leaf finer leaves split, and whether the
    // leaf can keep its vertices from the previous pass.
    void build_status() {
        const Tables &t = tables();
        const size_t count = leaf_key_.size();
        leaf_status_.assign(count, 0);
        leaf_reused_.assign(count, kNone);
        parallel_chunks(chunks_of(count, kLeafChunk), s_.threads, [&](size_t chunk) {
            const size_t end = std::min(count, (chunk + 1) * kLeafChunk);
            for (size_t i = chunk * kLeafChunk; i < end; ++i) {
                const int level = leaf_level_[i];
                uint32_t status = 0;
                if (level < depth_) {
                    const int64_t size = static_cast<int64_t>(size_of(level));
                    uint64_t m[3];
                    point_coords(leaf_key_[i], m);
                    auto finer = [&](const int64_t c[3]) {
                        const uint32_t leaf = locate(c);
                        return leaf != kNone && leaf_level_[leaf] > level;
                    };
                    for (int f = 0; f < 6; ++f) {
                        int64_t c[3] = {int64_t(m[0]), int64_t(m[1]), int64_t(m[2])};
                        c[f / 2] += f % 2 ? size : -1;
                        if (finer(c)) status |= 1u << (12 + f);
                    }
                    for (int e = 0; e < 12; ++e) {
                        const int b = t.edge_lo_axis[e], d = t.edge_hi_axis[e];
                        const int ob = t.edge_lo_offset[e], od = t.edge_hi_offset[e];
                        if (face_split(status, 2 * b + ob) || face_split(status, 2 * d + od)) {
                            status |= 1u << e;
                            continue;
                        }
                        int64_t c[3] = {int64_t(m[0]), int64_t(m[1]), int64_t(m[2])};
                        c[b] += ob ? size : -1;
                        c[d] += od ? size : -1;
                        if (finer(c)) status |= 1u << e;
                    }
                }
                leaf_status_[i] = status;
                if (cache_.level.empty()) continue;
                const uint32_t previous = cache_.leaf_index.find(leaf_key_[i]);
                if (previous != kNone && cache_.level[previous] == level && cache_.status[previous] == status)
                    leaf_reused_[i] = previous;
            }
        });
    }

    // Values at the active points of leaf i's boundary grid (NaN elsewhere).
    void boundary_values(size_t i, double value[27]) const {
        const int level = leaf_level_[i];
        const uint64_t size = size_of(level);
        const uint32_t status = leaf_status_[i];
        const Tables &t = tables();
        uint64_t m[3];
        point_coords(leaf_key_[i], m);
        bool active[27] = {};
        for (int c = 0; c < 8; ++c) active[(c & 1) * 2 + (c >> 1 & 1) * 6 + (c >> 2) * 18] = true;
        for (int e = 0; e < 12; ++e)
            if (edge_split(status, e)) active[t.edge_mid[e]] = true;
        for (int f = 0; f < 6; ++f)
            if (face_split(status, f)) active[t.face_centre[f]] = true;
        for (int g = 0; g < 27; ++g) {
            if (!active[g]) {
                value[g] = NAN;
                continue;
            }
            int c[3];
            grid_coords(g, c);
            value[g] = point_value(point_key(m[0] + (c[0] * size >> 1), m[1] + (c[1] * size >> 1),
                                             m[2] + (c[2] * size >> 1)));
        }
    }

    // Global key of an active slot: (level, lower end point * 4 + axis).
    void slot_key(size_t i, int lo, int axis, bool half, uint32_t &level, uint64_t &key, uint64_t &point) const {
        const uint64_t size = size_of(leaf_level_[i]);
        uint64_t m[3];
        point_coords(leaf_key_[i], m);
        int c[3];
        grid_coords(lo, c);
        point = point_key(m[0] + (c[0] * size >> 1), m[1] + (c[1] * size >> 1), m[2] + (c[2] * size >> 1));
        level = leaf_level_[i] + (half ? 1 : 0);
        key = point * 4 + uint64_t(axis);
    }

    // The boundary polygons of leaf i: per face, one cycle of slots through
    // its corners and the midpoints of its split edges, or four squares of a
    // split face. Calls polygon(slots, points, count, level, key) with the
    // cycle's slots, the grid points before each slot, and the key of the
    // polygon (min corner * 4 + face axis) at its level.
    template <class Polygon>
    void boundary_polygons(size_t i, const Polygon &polygon) const {
        const Tables &t = tables();
        const uint32_t status = leaf_status_[i];
        const int level = leaf_level_[i];
        const uint64_t size = size_of(level);
        uint64_t m[3];
        point_coords(leaf_key_[i], m);
        for (int f = 0; f < 6; ++f) {
            const int axis = f / 2, side = f % 2, u = (axis + 1) % 3, v = (axis + 2) % 3;
            uint64_t corner[3] = {m[0], m[1], m[2]};
            corner[axis] += side * size;
            int slots[8], points[8], count = 0;
            if (!face_split(status, f)) {
                for (int k = 0; k < 4; ++k) {
                    const int e = t.face_edge[f][k];
                    if (edge_split(status, e)) {
                        points[count] = t.face_corner[f][k];
                        slots[count++] = t.start_half[f][k];
                        points[count] = t.face_mid[f][k];
                        slots[count++] = t.end_half[f][k];
                    } else {
                        points[count] = t.face_corner[f][k];
                        slots[count++] = 2 * e;
                    }
                }
                polygon(slots, points, count, level, point_key(corner) * 4 + uint64_t(axis));
                continue;
            }
            static const int kCycle[4][2] = {{0, 0}, {1, 0}, {1, 1}, {0, 1}};
            for (int q = 0; q < 4; ++q) {
                const int previous = (q + 3) % 4;
                points[0] = t.face_corner[f][q];
                slots[0] = t.start_half[f][q];
                points[1] = t.face_mid[f][q];
                slots[1] = 24 + 4 * f + q;
                points[2] = t.face_centre[f];
                slots[2] = 24 + 4 * f + previous;
                points[3] = t.face_mid[f][previous];
                slots[3] = t.end_half[f][previous];
                uint64_t square[3] = {corner[0], corner[1], corner[2]};
                square[u] += kCycle[q][0] * (size >> 1);
                square[v] += kCycle[q][1] * (size >> 1);
                polygon(slots, points, 4, level + 1, point_key(square) * 4 + uint64_t(axis));
            }
        }
    }

    // Crossings of new sign-change slots and centre values of new ambiguous
    // polygons of the leaves that changed.
    void build_edges_and_faces() {
        const size_t count = leaf_key_.size();
        const size_t parts = chunks_of(count, kLeafChunk);
        std::vector<std::vector<EdgeJob>> edge_pieces(parts);
        std::vector<std::vector<std::pair<uint32_t, uint64_t>>> face_pieces(parts);
        parallel_chunks(parts, s_.threads, [&](size_t part) {
            const size_t end = std::min(count, (part + 1) * kLeafChunk);
            for (size_t i = part * kLeafChunk; i < end; ++i) {
                if (leaf_reused_[i] != kNone) continue;
                double value[27];
                boundary_values(i, value);
                const uint32_t status = leaf_status_[i];
                bool any = false;
                for (int slot = 0; slot < kSlots; ++slot) {
                    int lo, hi, axis;
                    bool half;
                    if (!slot_geometry(slot, status, lo, hi, axis, half)) continue;
                    if ((value[lo] < 0) == (value[hi] < 0)) continue;
                    any = true;
                    EdgeJob job;
                    uint64_t key;
                    slot_key(i, lo, axis, half, job.level, key, job.point);
                    job.axis = static_cast<uint32_t>(axis);
                    job.f0 = value[lo];
                    job.f1 = value[hi];
                    if (crossing_maps_.size() <= job.level || crossing_maps_[job.level].find(key) == kNone)
                        edge_pieces[part].push_back(job);
                }
                if (!any) continue;
                boundary_polygons(i, [&](const int *, const int *points, int n, int level, uint64_t key) {
                    int changes = 0;
                    for (int k = 0; k < n; ++k) changes += (value[points[k]] < 0) != (value[points[(k + 1) % n]] < 0);
                    if (changes < 4) return;
                    if (face_maps_.size() <= size_t(level) || face_maps_[level].find(key) == kNone)
                        face_pieces[part].emplace_back(static_cast<uint32_t>(level), key);
                });
            }
        });
        std::vector<EdgeJob> edges = concatenate(edge_pieces);
        std::sort(edges.begin(), edges.end(), [](const EdgeJob &a, const EdgeJob &b) {
            return a.level != b.level ? a.level < b.level : a.point != b.point ? a.point < b.point : a.axis < b.axis;
        });
        edges.erase(std::unique(edges.begin(), edges.end(), [](const EdgeJob &a, const EdgeJob &b) {
                        return a.level == b.level && a.point == b.point && a.axis == b.axis;
                    }),
                    edges.end());
        if (crossings_.size() + edges.size() >= kNone) throw Error("meshing needs too many surface crossings; increase cell_size");
        const size_t first = crossings_.size();
        crossings_.resize(first + edges.size());
        for (size_t e = 0; e < edges.size(); ++e) {
            if (crossing_maps_.size() <= edges[e].level) crossing_maps_.resize(edges[e].level + 1);
            crossing_maps_[edges[e].level].insert(edges[e].point * 4 + edges[e].axis, static_cast<uint32_t>(first + e));
        }
        // Refine in Morton order of the edges' lower ends: batches of edges in
        // key order are long strips that defeat culling.
        std::vector<uint32_t> todo(edges.size());
        for (size_t e = 0; e < edges.size(); ++e) todo[e] = static_cast<uint32_t>(e);
        if (s_.spatial_order) {
            std::vector<uint64_t> codes(edges.size());
            for (size_t e = 0; e < edges.size(); ++e) codes[e] = morton_of(edges[e].point);
            radix_sort(codes, &todo, morton_bits_);
        }
        const BatchField gradient_field = field_.batch();
        parallel_chunks(chunks_of(todo.size(), kChunk / 4), s_.threads, [&](size_t chunk) {
            const size_t end = std::min(todo.size(), (chunk + 1) * (kChunk / 4));
            for (size_t begin = chunk * (kChunk / 4); begin < end; begin += kBatch)
                refine(edges, todo.data() + begin, static_cast<int>(std::min<size_t>(kBatch, end - begin)),
                       gradient_field, &crossings_[first]);
        });

        std::vector<std::pair<uint32_t, uint64_t>> faces = concatenate(face_pieces);
        std::sort(faces.begin(), faces.end());
        faces.erase(std::unique(faces.begin(), faces.end()), faces.end());
        const size_t base = face_values_.size();
        face_values_.resize(base + faces.size());
        for (size_t i = 0; i < faces.size(); ++i) {
            if (face_maps_.size() <= faces[i].first) face_maps_.resize(faces[i].first + 1);
            face_maps_[faces[i].first].insert(faces[i].second, static_cast<uint32_t>(base + i));
        }
        parallel_chunks(chunks_of(faces.size(), kChunk), s_.threads, [&](size_t chunk) {
            double x[kBatch], y[kBatch], z[kBatch], v[kBatch];
            const size_t end = std::min(faces.size(), (chunk + 1) * kChunk);
            for (size_t begin = chunk * kChunk; begin < end; begin += kBatch) {
                const int n = static_cast<int>(std::min<size_t>(kBatch, end - begin));
                for (int i = 0; i < n; ++i) {
                    const int level = static_cast<int>(faces[begin + i].first);
                    const uint64_t key = faces[begin + i].second;
                    const int axis = static_cast<int>(key % 4);
                    uint64_t c[3];
                    point_coords(key / 4, c);
                    const uint64_t half = size_of(level) >> 1;  // polygons are at least two fine units wide
                    double *out[3] = {x, y, z};
                    for (int a = 0; a < 3; ++a) out[a][i] = coordinate(a, c[a] + (a == axis ? 0 : half));
                }
                field_.evaluate(n, x, y, z, v);
                for (int i = 0; i < n; ++i) face_values_[base + begin + i] = v[i];
            }
        });
    }

    // Crossing refinement along each edge, then the gradient normal. Steps
    // alternate between Illinois false position (fast on smooth surfaces) and
    // a Lipschitz step from the bracket end with the larger |f|: the root is
    // at least |f| / L from that end, which pins crossings at once where the
    // field is flat along the edge near a sharp corner and false position
    // creeps. Halved Illinois values only shorten those steps.
    void refine(const std::vector<EdgeJob> &edges, const uint32_t *ids, int n, const BatchField &gradient_field,
                Crossing *out) const {
        double a[kBatch], fa[kBatch], b[kBatch], fb[kBatch], t[kBatch], ft[kBatch], length[kBatch];
        double va[kBatch], vb[kBatch];  // the field at the bracket's ends (fa, fb may be halved)
        double base[3][kBatch], x[kBatch], y[kBatch], z[kBatch];
        int side[kBatch];
        double *coords[3] = {x, y, z};
        for (int i = 0; i < n; ++i) {
            const EdgeJob &edge = edges[ids[i]];
            uint64_t c[3];
            point_coords(edge.point, c);
            for (int axis = 0; axis < 3; ++axis) base[axis][i] = coordinate(axis, c[axis]);
            length[i] = std::ldexp(fine_, depth_ - static_cast<int>(edge.level));
            a[i] = 0;
            fa[i] = va[i] = edge.f0;
            b[i] = 1;
            fb[i] = vb[i] = edge.f1;
            side[i] = 0;
        }
        // False position; rounding can put it just past an end of the
        // bracket (when an end value is 0, the root itself), which stays.
        auto position = [&](int i) {
            const double fraction = (a[i] * fb[i] - b[i] * fa[i]) / (fb[i] - fa[i]);
            return std::isfinite(fraction) ? std::fmin(std::fmax(fraction, a[i]), b[i]) : 0.5 * (a[i] + b[i]);
        };
        auto place = [&](const double *along) {
            for (int axis = 0; axis < 3; ++axis) std::copy(base[axis], base[axis] + n, coords[axis]);
            for (int i = 0; i < n; ++i) coords[edges[ids[i]].axis][i] += along[i] * length[i];
        };
        for (int iteration = 0; iteration < s_.refine_steps; ++iteration) {
            const bool lipschitz_step = iteration % 2 == 0;
            for (int i = 0; i < n; ++i) {
                t[i] = position(i);
                if (!lipschitz_step) continue;
                const double reach = 1 / (s_.lipschitz * length[i]);  // edge fraction per unit of |f|
                const double step = std::fabs(fa[i]) >= std::fabs(fb[i]) ? a[i] + std::fabs(fa[i]) * reach
                                                                          : b[i] - std::fabs(fb[i]) * reach;
                if (step > a[i] && step < b[i]) t[i] = step;
            }
            place(t);
            field_.evaluate(n, x, y, z, ft);
            for (int i = 0; i < n; ++i) {
                if ((ft[i] < 0) == (fb[i] < 0)) {  // root in [a, t]
                    b[i] = t[i];
                    fb[i] = vb[i] = ft[i];
                    if (!lipschitz_step && side[i] == -1) fa[i] *= 0.5;
                    side[i] = -1;
                } else {  // root in [t, b]
                    a[i] = t[i];
                    fa[i] = va[i] = ft[i];
                    if (!lipschitz_step && side[i] == 1) fb[i] *= 0.5;
                    side[i] = 1;
                }
            }
        }
        for (int i = 0; i < n; ++i) t[i] = position(i);
        place(t);
        double normal[3][kBatch];
        for (int first = 0; first < n; first += kBatch / 4) {
            const int m = std::min(kBatch / 4, n - first);
            batch_gradients(gradient_field, m, x + first, y + first, z + first, s_.gradient_step, normal[0] + first,
                            normal[1] + first, normal[2] + first);
        }
        for (int i = 0; i < n; ++i) {
            Crossing &crossing = out[ids[i]];
            crossing.jump = std::fabs(vb[i] - va[i]) > 2 * s_.lipschitz * (b[i] - a[i]) * length[i] + 1e-9 * length[i];
            crossing.p[0] = x[i];
            crossing.p[1] = y[i];
            crossing.p[2] = z[i];
            for (int axis = 0; axis < 3; ++axis) crossing.n[axis] = normal[axis][i];
            if (normal[0][i] == 0 && normal[1][i] == 0 && normal[2][i] == 0) {
                // Vanishing gradient: use the edge direction, from inside to outside.
                const EdgeJob &edge = edges[ids[i]];
                crossing.n[edge.axis] = edge.f0 < 0 ? 1 : -1;
            }
        }
    }

    const Crossing &crossing(uint32_t level, uint64_t key) const {
        const uint32_t index = level < crossing_maps_.size() ? crossing_maps_[level].find(key) : kNone;
        if (index == kNone) throw Error("internal error: a surface crossing is missing");
        return crossings_[index];
    }

    uint32_t crossing_index(uint32_t level, uint64_t key) const {
        const uint32_t index = level < crossing_maps_.size() ? crossing_maps_[level].find(key) : kNone;
        if (index == kNone) throw Error("internal error: a surface crossing is missing");
        return index;
    }

    double face_value(int level, uint64_t key) const {
        const uint32_t index = size_t(level) < face_maps_.size() ? face_maps_[level].find(key) : kNone;
        if (index == kNone) throw Error("internal error: an ambiguous face is missing");
        return face_values_[index];
    }

    // Groups leaf i's sign-change slots into surface components, like the
    // patches of marching cubes on its subdivided boundary: on each boundary
    // polygon the crossings pair into segments, and slots linked through
    // segments share a component. With two crossings they pair; with more,
    // the field at the polygon's centre decides: inside joins the inside runs
    // of the polygon's boundary (each outside run is cut off by a segment
    // joining its two crossings), outside joins the outside runs. Both leaves
    // sharing a polygon read the same centre value and pair it alike. Near a
    // sharp edge the field is the max or min of smooth fields, not bilinear,
    // so corner values alone misjudge such faces. Writes each sign-change
    // slot's component (numbered by lowest slot) and returns the count.
    int components(size_t i, const double value[27], uint8_t component[kSlots]) const {
        const uint32_t status = leaf_status_[i];
        uint8_t parent[kSlots];
        for (int k = 0; k < kSlots; ++k) parent[k] = static_cast<uint8_t>(k);
        auto root = [&](int k) {
            while (parent[k] != k) k = parent[k] = parent[parent[k]];
            return k;
        };
        auto unite = [&](int a, int b) {
            a = root(a);
            b = root(b);
            if (a != b) parent[std::max(a, b)] = static_cast<uint8_t>(std::min(a, b));
        };
        boundary_polygons(i, [&](const int *slots, const int *points, int n, int level, uint64_t key) {
            int crossings[8], count = 0;
            bool after_inside[8];
            for (int k = 0; k < n; ++k) {
                const bool here = value[points[k]] < 0, next = value[points[(k + 1) % n]] < 0;
                if (here == next) continue;
                crossings[count] = slots[k];
                after_inside[count++] = next;
            }
            if (count == 0) return;
            const bool cut_inside = count == 2 || !(face_value(level, key) < 0);
            for (int j = 0; j < count; ++j)
                if (after_inside[j] == cut_inside) unite(crossings[j], crossings[(j + 1) % count]);
        });
        int count = 0;
        uint8_t label[kSlots];
        std::fill(label, label + kSlots, kNoComponent);
        std::fill(component, component + kSlots, kNoComponent);
        for (int slot = 0; slot < kSlots; ++slot) {
            int lo, hi, axis;
            bool half;
            if (!slot_geometry(slot, status, lo, hi, axis, half)) continue;
            if ((value[lo] < 0) == (value[hi] < 0)) continue;
            const int r = root(slot);
            if (label[r] == kNoComponent) label[r] = static_cast<uint8_t>(count++);
            component[slot] = label[r];
        }
        return count;
    }

    // QEF vertices of the leaves that changed (one per surface component),
    // their sharpness marks and projections; reused leaves copy theirs.
    void build_vertices() {
        const size_t count = leaf_key_.size();
        const size_t parts = chunks_of(count, kLeafChunk);
        std::vector<std::vector<VertexRecord>> records(parts);
        std::vector<std::vector<uint8_t>> part_tables(parts);
        std::vector<uint8_t> counts(count, 0);
        std::vector<uint8_t> has_table(count, 0);
        parallel_chunks(parts, s_.threads, [&](size_t part) {
            const size_t end = std::min(count, (part + 1) * kLeafChunk);
            for (size_t i = part * kLeafChunk; i < end; ++i) {
                const uint32_t previous = leaf_reused_[i];
                if (previous != kNone) {
                    counts[i] = cache_.vertex_count[previous];
                    has_table[i] = cache_.table[previous] != kNone;
                    continue;
                }
                double value[27];
                boundary_values(i, value);
                uint8_t component[kSlots];
                const int n = components(i, value, component);
                if (n == 0) continue;
                if (n > 255) throw Error("internal error: too many surface components in a cell");
                counts[i] = static_cast<uint8_t>(n);
                has_table[i] = 1;
                part_tables[part].insert(part_tables[part].end(), component, component + kSlots);
                const int level = leaf_level_[i];
                const double size = std::ldexp(fine_, depth_ - level);
                uint64_t m[3];
                point_coords(leaf_key_[i], m);
                const double lo[3] = {coordinate(0, m[0]), coordinate(1, m[1]), coordinate(2, m[2])};
                const uint32_t status = leaf_status_[i];
                for (int j = 0; j < n; ++j) {
                    double p[kSlots][3], nrm[kSlots][3];
                    int k = 0;
                    bool jump = false;
                    for (int slot = 0; slot < kSlots; ++slot) {
                        if (component[slot] != j) continue;
                        int a, b, axis;
                        bool half;
                        slot_geometry(slot, status, a, b, axis, half);
                        uint32_t edge_level;
                        uint64_t key, point;
                        slot_key(i, a, axis, half, edge_level, key, point);
                        const Crossing &c = crossing(edge_level, key);
                        std::copy(c.p, c.p + 3, p[k]);
                        std::copy(c.n, c.n + 3, nrm[k]);
                        jump = jump || c.jump;
                        ++k;
                    }
                    VertexRecord r;
                    solve_qef(k, p, nrm, lo, size, r.position);
                    crease_point(k, p, nrm, r.position, lo, size, r.crease);
                    r.residual = 0;
                    for (int q = 0; q < k; ++q)
                        r.residual = std::fmax(r.residual, std::fabs(nrm[q][0] * (r.position[0] - p[q][0]) +
                                                                     nrm[q][1] * (r.position[1] - p[q][1]) +
                                                                     nrm[q][2] * (r.position[2] - p[q][2])));
                    // spread: the widest pair's cosine and its difference.
                    r.spread[0] = 1;
                    r.spread[1] = r.spread[2] = r.spread[3] = 0;
                    r.guide[0] = r.guide[1] = r.guide[2] = 0;
                    for (int q = 0; q < k; ++q) {
                        for (int a = 0; a < 3; ++a) r.guide[a] += nrm[q][a];
                        for (int l = q + 1; l < k; ++l) {
                            const double cosine = nrm[q][0] * nrm[l][0] + nrm[q][1] * nrm[l][1] + nrm[q][2] * nrm[l][2];
                            if (!(cosine < r.spread[0])) continue;
                            r.spread[0] = cosine;
                            for (int a = 0; a < 3; ++a) r.spread[1 + a] = nrm[q][a] - nrm[l][a];
                        }
                    }
                    for (int a = 0; a < 3; ++a) {
                        r.box[a] = lo[a] - 0.5 * size;
                        r.box[3 + a] = lo[a] + 1.5 * size;
                    }
                    r.side = 0.05 * size;
                    r.jump = jump;
                    records[part].push_back(r);
                }
            }
        });
        std::vector<VertexRecord> fresh = concatenate(records);
        std::vector<uint8_t> fresh_tables = concatenate(part_tables);
        std::vector<uint8_t> fresh_marks;
        std::vector<double> fresh_positions;
        project_vertices(fresh, fresh_marks, fresh_positions);
        std::vector<double> fresh_values = field_values(field_, s_.threads, s_.spatial_order, fresh_positions);
        std::vector<double> fresh_slopes;
        const std::vector<double> fresh_normals = gradients(field_, fresh.size(), s_.gradient_step, s_.threads,
            s_.spatial_order, [&](size_t v, double *p) { std::copy_n(&fresh_positions[3 * v], 3, p); }, &fresh_slopes);

        // Assemble in leaf order.
        leaf_first_vertex_.assign(count, 0);
        leaf_table_.assign(count, kNone);
        size_t vertices = 0, active = 0;
        for (size_t i = 0; i < count; ++i) {
            leaf_first_vertex_[i] = static_cast<uint32_t>(vertices);
            vertices += counts[i];
            if (has_table[i]) leaf_table_[i] = static_cast<uint32_t>(active++);
            if (vertices >= kNone) throw Error("the mesh has too many vertices");
        }
        leaf_vertex_count_.swap(counts);
        active_cells_ = active;
        vertex_positions_.assign(3 * vertices, 0.0);
        vertex_marks_.assign(vertices, 0);
        vertex_residuals_.assign(vertices, 0.0);
        vertex_values_.assign(vertices, 0.0);
        vertex_slopes_.assign(vertices, 0.0);
        vertex_normals_.assign(3 * vertices, 0.0);
        vertex_leaf_.assign(vertices, 0);
        tables_.assign(size_t(kSlots) * active, kNoComponent);
        size_t next_fresh = 0, next_table = 0;
        for (size_t i = 0; i < count; ++i) {
            const uint32_t first = leaf_first_vertex_[i], n = leaf_vertex_count_[i];
            for (uint32_t v = first; v < first + n; ++v) vertex_leaf_[v] = static_cast<uint32_t>(i);
            const uint32_t previous = leaf_reused_[i];
            if (previous != kNone) {
                const uint32_t from = cache_.first_vertex[previous];
                std::copy_n(&cache_.positions[3 * size_t(from)], 3 * n, &vertex_positions_[3 * size_t(first)]);
                std::copy_n(&cache_.marks[from], n, &vertex_marks_[first]);
                std::copy_n(&cache_.residuals[from], n, &vertex_residuals_[first]);
                std::copy_n(&cache_.values[from], n, &vertex_values_[first]);
                std::copy_n(&cache_.slopes[from], n, &vertex_slopes_[first]);
                std::copy_n(&cache_.normals[3 * size_t(from)], 3 * n, &vertex_normals_[3 * size_t(first)]);
                if (leaf_table_[i] != kNone)
                    std::copy_n(&cache_.tables[size_t(kSlots) * cache_.table[previous]], kSlots,
                                &tables_[size_t(kSlots) * leaf_table_[i]]);
                continue;
            }
            if (n == 0) continue;
            for (uint32_t k = 0; k < n; ++k, ++next_fresh) {
                std::copy_n(&fresh_positions[3 * next_fresh], 3, &vertex_positions_[3 * size_t(first + k)]);
                vertex_marks_[first + k] = fresh_marks[next_fresh] | (fresh[next_fresh].jump ? kJumpVertex : 0);
                vertex_residuals_[first + k] = fresh[next_fresh].residual;
                vertex_values_[first + k] = fresh_values[next_fresh];
                vertex_slopes_[first + k] = fresh_slopes[next_fresh];
                std::copy_n(&fresh_normals[3 * next_fresh], 3, &vertex_normals_[3 * size_t(first + k)]);
            }
            std::copy_n(&fresh_tables[size_t(kSlots) * next_table++], kSlots, &tables_[size_t(kSlots) * leaf_table_[i]]);
        }
    }

    // Moves QEF vertices onto the surface and marks them. Across a blend
    // narrower than a cell the crossings' tangent planes meet off the
    // surface, above a rounded ridge or inside it depending on which grid
    // edges the blend crosses, so its vertices alternate into dents along the
    // ridge. Such a ridge can also bulge slightly past the cell whose
    // crossings found it, so a vertex may move up to half a cell beyond its
    // cell. On a sharp edge or corner the QEF vertex is the feature point and
    // the gradient there blends the faces, so a step would slide it off the
    // feature: a vertex whose crossing normals differ by more than kSpread
    // stays (kSharpVertex) where the gradients a twentieth of its cell either
    // side of it, along the difference of its widest pair of normals, differ
    // by more than half that pair's angle and by more than kTurn. Across a
    // sub-cell blend they differ by a few degrees; such a vertex is a
    // kSpreadVertex. Vertices whose crossing normals agree move too: a blend
    // can pass through a cell without crossing its grid edges.
    void project_vertices(std::vector<VertexRecord> &records, std::vector<uint8_t> &marks,
                          std::vector<double> &positions) {
        const double kSpread = std::cos(10 * M_PI / 180), kTurn = std::cos(20 * M_PI / 180);
        const double kShallow = std::cos(kCreaseTurn * M_PI / 180);
        refine_creases(records);
        const size_t count = records.size();
        positions.resize(3 * count);
        for (size_t v = 0; v < count; ++v) std::copy_n(records[v].position, 3, &positions[3 * v]);
        // Vertices whose crossing normals spread are tested at their QEF
        // point and, where it differs, at the point that resolves shallow
        // creases, there also a quarter as far either side: across a crease
        // the gradients differ by its angle at any distance, across a curve
        // in proportion to the distance (a blend narrower than a cell can
        // pass the first test).
        std::vector<size_t> features;
        std::vector<uint8_t> at_crease;
        std::vector<double> points, across, sides;
        for (size_t v = 0; v < count; ++v) {
            const VertexRecord &r = records[v];
            for (int crease = 0; crease < 2; ++crease) {
                const double *point = crease ? r.crease : r.position;
                if (crease ? !(r.spread[0] < kShallow) || std::equal(r.crease, r.crease + 3, r.position)
                           : !(r.spread[0] < kSpread))
                    continue;
                for (int near = 0; near <= crease; ++near) {
                    features.push_back(v);
                    at_crease.push_back(static_cast<uint8_t>(crease + near));
                    points.insert(points.end(), point, point + 3);
                    across.insert(across.end(), r.spread + 1, r.spread + 4);
                    sides.push_back(near ? 0.25 * r.side : r.side);
                }
            }
        }
        const std::vector<double> gaps = side_cosines(field_, s_, points, across, sides);
        marks.assign(count, 0);
        for (size_t i = 0; i < features.size(); ++i) {
            const size_t v = features[i];
            const VertexRecord &r = records[v];
            const double half = std::sqrt(0.5 * (1 + r.spread[0]));  // cosine of half the widest pair's angle
            if (at_crease[i] == 0) {
                // Half the widest pair's angle, at least kTurn.
                const bool sharp = gaps[i] < std::fmin(half, kTurn);
                marks[v] = sharp ? kSharpVertex : kSpreadVertex;
            } else if (at_crease[i] == 1 && !(marks[v] & kSharpVertex) && gaps[i] < std::fmin(half, kShallow) &&
                       std::acos(std::fmin(1.0, gaps[i + 1])) > 0.6 * std::acos(gaps[i])) {
                // A crease too shallow for the general truncation: stay on it.
                marks[v] = kSharpVertex;
                std::copy_n(r.crease, 3, &positions[3 * v]);
            }
        }
        std::vector<double> boxes(6 * count), guides(3 * count);
        std::vector<uint8_t> fixed(count);
        for (size_t v = 0; v < count; ++v) {
            std::copy_n(records[v].box, 6, &boxes[6 * v]);
            std::copy_n(records[v].guide, 3, &guides[3 * v]);
            fixed[v] = marks[v] & kSharpVertex;
        }
        project_points(field_, s_, 1e-6 * fine_, positions, guides, boxes, fixed);
    }

    // Re-fits each shallow crease point to the field: the planes through the
    // surface points (one Newton step) a tenth of the cell either side of
    // it across the crease, with the gradients there, meet on a line close
    // to a curved crease, where the planes of the cell's crossings bend
    // away from it. Clamped to the cell; points whose planes do not differ
    // by kCreaseTurn stay.
    void refine_creases(std::vector<VertexRecord> &records) const {
        static const double kTetra[4][3] = {{1, -1, -1}, {-1, -1, 1}, {-1, 1, -1}, {1, 1, 1}};
        constexpr int kPerPoint = 10, kPoints = kBatch / kPerPoint;  // two sides: the value and four stencil samples
        const double shallow = std::cos(kCreaseTurn * M_PI / 180), eps = s_.gradient_step;
        std::vector<uint32_t> ids;
        for (size_t v = 0; v < records.size(); ++v)
            if (records[v].spread[0] < shallow && !std::equal(records[v].crease, records[v].crease + 3, records[v].position))
                ids.push_back(static_cast<uint32_t>(v));
        parallel_chunks(chunks_of(ids.size(), kChunk), s_.threads, [&](size_t chunk) {
            double x[kBatch], y[kBatch], z[kBatch], f[kBatch], across[kPoints][3];
            const size_t end = std::min(ids.size(), (chunk + 1) * kChunk);
            for (size_t first = chunk * kChunk; first < end; first += kPoints) {
                const int n = static_cast<int>(std::min<size_t>(kPoints, end - first));
                for (int i = 0; i < n; ++i) {
                    const VertexRecord &r = records[ids[first + i]];
                    // Across the crease: the widest pair's difference, in the tangent plane.
                    double normal[3], length = 0;
                    for (int a = 0; a < 3; ++a) length += r.guide[a] * r.guide[a];
                    length = std::sqrt(length);
                    for (int a = 0; a < 3; ++a) normal[a] = length > 0 ? r.guide[a] / length : 0;
                    const double along = r.spread[1] * normal[0] + r.spread[2] * normal[1] + r.spread[3] * normal[2];
                    double t[3], norm = 0;
                    for (int a = 0; a < 3; ++a) {
                        t[a] = r.spread[1 + a] - along * normal[a];
                        norm += t[a] * t[a];
                    }
                    norm = std::sqrt(norm);
                    for (int a = 0; a < 3; ++a) across[i][a] = norm > 0 ? 2 * r.side * t[a] / norm : 0;
                    for (int side = 0; side < 2; ++side) {
                        double q[3];
                        for (int a = 0; a < 3; ++a) q[a] = r.crease[a] + (side ? -across[i][a] : across[i][a]);
                        const int base = kPerPoint * i + 5 * side;
                        x[base] = q[0];
                        y[base] = q[1];
                        z[base] = q[2];
                        for (int k = 0; k < 4; ++k) {
                            x[base + 1 + k] = q[0] + eps * kTetra[k][0];
                            y[base + 1 + k] = q[1] + eps * kTetra[k][1];
                            z[base + 1 + k] = q[2] + eps * kTetra[k][2];
                        }
                    }
                }
                field_.evaluate(kPerPoint * n, x, y, z, f);
                for (int i = 0; i < n; ++i) {
                    VertexRecord &r = records[ids[first + i]];
                    if (across[i][0] == 0 && across[i][1] == 0 && across[i][2] == 0) continue;
                    double normal[2][3], point[2][3];
                    bool usable = true;
                    for (int side = 0; side < 2 && usable; ++side) {
                        const int base = kPerPoint * i + 5 * side;
                        double g[3] = {0, 0, 0};
                        for (int k = 0; k < 4; ++k)
                            for (int a = 0; a < 3; ++a) g[a] += f[base + 1 + k] * kTetra[k][a] / (4 * eps);
                        const double gg = g[0] * g[0] + g[1] * g[1] + g[2] * g[2];
                        if (!(gg > 0) || !std::isfinite(gg)) {
                            usable = false;
                            break;
                        }
                        const double length = std::sqrt(gg);
                        for (int a = 0; a < 3; ++a) {
                            normal[side][a] = g[a] / length;
                            point[side][a] = (side ? r.crease[a] - across[i][a] : r.crease[a] + across[i][a]) -
                                             f[base] * g[a] / gg;
                        }
                    }
                    if (!usable) continue;
                    const double c = normal[0][0] * normal[1][0] + normal[0][1] * normal[1][1] + normal[0][2] * normal[1][2];
                    if (!(c < shallow)) continue;
                    double offset[2];
                    for (int g = 0; g < 2; ++g)
                        offset[g] = normal[g][0] * (r.crease[0] - point[g][0]) + normal[g][1] * (r.crease[1] - point[g][1]) +
                                    normal[g][2] * (r.crease[2] - point[g][2]);
                    const double det = 1 - c * c;
                    const double a = (offset[0] - c * offset[1]) / det, b = (offset[1] - c * offset[0]) / det;
                    const double size = 0.5 * (r.box[3] - r.box[0]);
                    for (int k = 0; k < 3; ++k) {
                        const double lo = r.box[k] + 0.5 * size;
                        r.crease[k] = std::fmin(std::fmax(r.crease[k] - a * normal[0][k] - b * normal[1][k], lo), lo + size);
                    }
                }
            }
        });
    }

    // Minimises sum (n_i . (x - p_i))^2 about the mass point with a truncated
    // pseudo-inverse (eigenvalues below kTruncation of the largest dropped),
    // clamped to the cell.
    static void solve_qef(int m, const double (*p)[3], const double (*n)[3], const double lo[3], double h,
                          double out[3]) {
        double mass[3] = {0, 0, 0};
        for (int i = 0; i < m; ++i)
            for (int a = 0; a < 3; ++a) mass[a] += p[i][a];
        for (int a = 0; a < 3; ++a) mass[a] /= m;
        double ata[3][3] = {}, atb[3] = {};
        for (int i = 0; i < m; ++i) {
            const double d = n[i][0] * (p[i][0] - mass[0]) + n[i][1] * (p[i][1] - mass[1]) + n[i][2] * (p[i][2] - mass[2]);
            for (int r = 0; r < 3; ++r) {
                atb[r] += n[i][r] * d;
                for (int c = 0; c < 3; ++c) ata[r][c] += n[i][r] * n[i][c];
            }
        }
        double vectors[3][3], values[3];
        jacobi_eigen(ata, vectors, values);
        const double largest = std::max({values[0], values[1], values[2]});
        double x[3] = {0, 0, 0};
        for (int k = 0; k < 3; ++k) {
            if (!(values[k] > kTruncation * largest) || !(values[k] > 0)) continue;
            const double projection =
                (vectors[0][k] * atb[0] + vectors[1][k] * atb[1] + vectors[2][k] * atb[2]) / values[k];
            for (int a = 0; a < 3; ++a) x[a] += vectors[a][k] * projection;
        }
        for (int a = 0; a < 3; ++a) out[a] = std::fmin(std::fmax(mass[a] + x[a], lo[a]), lo[a] + h);
    }

    // The point nearest `from` on the line where the planes of two groups
    // of crossings meet: each crossing joins the nearer of the widest pair
    // of normals, and each group's plane passes through its mean point
    // with its mean normal (so curvature within a group averages out).
    // Clamped to the cell; `from` when the groups do not differ by
    // kCreaseTurn.
    static void crease_point(int m, const double (*p)[3], const double (*n)[3], const double from[3],
                             const double lo[3], double h, double out[3]) {
        std::copy_n(from, 3, out);
        int first = -1, second = -1;
        double widest = 2;
        for (int i = 0; i < m; ++i)
            for (int j = i + 1; j < m; ++j) {
                const double c = n[i][0] * n[j][0] + n[i][1] * n[j][1] + n[i][2] * n[j][2];
                if (c < widest) {
                    widest = c;
                    first = i;
                    second = j;
                }
            }
        if (first < 0) return;
        double normal[2][3] = {}, point[2][3] = {};
        int size[2] = {0, 0};
        for (int i = 0; i < m; ++i) {
            const double a = n[i][0] * n[first][0] + n[i][1] * n[first][1] + n[i][2] * n[first][2];
            const double b = n[i][0] * n[second][0] + n[i][1] * n[second][1] + n[i][2] * n[second][2];
            const int g = a >= b ? 0 : 1;
            for (int k = 0; k < 3; ++k) {
                normal[g][k] += n[i][k];
                point[g][k] += p[i][k];
            }
            ++size[g];
        }
        double offset[2];
        for (int g = 0; g < 2; ++g) {
            const double length = std::sqrt(normal[g][0] * normal[g][0] + normal[g][1] * normal[g][1] +
                                            normal[g][2] * normal[g][2]);
            if (!(length > 0)) return;
            for (int k = 0; k < 3; ++k) {
                normal[g][k] /= length;
                point[g][k] /= size[g];
            }
            offset[g] = normal[g][0] * (from[0] - point[g][0]) + normal[g][1] * (from[1] - point[g][1]) +
                        normal[g][2] * (from[2] - point[g][2]);
        }
        const double c = normal[0][0] * normal[1][0] + normal[0][1] * normal[1][1] + normal[0][2] * normal[1][2];
        if (!(c < std::cos(kCreaseTurn * M_PI / 180))) return;
        // x = from - a n0 - b n1 with n_g . (x - point_g) = 0.
        const double det = 1 - c * c;
        const double a = (offset[0] - c * offset[1]) / det, b = (offset[1] - c * offset[0]) / det;
        for (int k = 0; k < 3; ++k)
            out[k] = std::fmin(std::fmax(from[k] - a * normal[0][k] - b * normal[1][k], lo[k]), lo[k] + h);
    }

    // Cyclic Jacobi eigen-decomposition of a symmetric 3x3 matrix (destroyed).
    // Eigenvectors are the columns of `vectors`.
    static void jacobi_eigen(double a[3][3], double vectors[3][3], double values[3]) {
        for (int r = 0; r < 3; ++r)
            for (int c = 0; c < 3; ++c) vectors[r][c] = r == c ? 1 : 0;
        const double scale = std::fabs(a[0][0]) + std::fabs(a[1][1]) + std::fabs(a[2][2]);
        for (int sweep = 0; sweep < 32; ++sweep) {
            const double off = std::fabs(a[0][1]) + std::fabs(a[0][2]) + std::fabs(a[1][2]);
            if (!(off > 1e-15 * scale)) break;
            for (int p = 0; p < 2; ++p)
                for (int q = p + 1; q < 3; ++q) {
                    if (a[p][q] == 0) continue;
                    const double theta = (a[q][q] - a[p][p]) / (2 * a[p][q]);
                    const double t = std::fabs(theta) > 1e150 ? 0.5 / theta
                                                              : (theta >= 0 ? 1.0 : -1.0) /
                                                                    (std::fabs(theta) + std::sqrt(theta * theta + 1));
                    const double c = 1 / std::sqrt(t * t + 1), s = t * c;
                    const double apq = a[p][q];
                    a[p][p] -= t * apq;
                    a[q][q] += t * apq;
                    a[p][q] = a[q][p] = 0;
                    for (int r = 0; r < 3; ++r) {
                        if (r != p && r != q) {
                            const double arp = a[r][p], arq = a[r][q];
                            a[r][p] = a[p][r] = c * arp - s * arq;
                            a[r][q] = a[q][r] = s * arp + c * arq;
                        }
                        const double vrp = vectors[r][p], vrq = vectors[r][q];
                        vectors[r][p] = c * vrp - s * vrq;
                        vectors[r][q] = s * vrp + c * vrq;
                    }
                }
        }
        for (int k = 0; k < 3; ++k) values[k] = a[k][k];
    }

    // ---------------------------------------------------------------- polygons

    // Slot of the minimal edge (point p, axis, size) on leaf `leaf`'s boundary, or -1.
    int slot_of(uint32_t leaf, const uint64_t p[3], int axis, uint64_t size) const {
        const uint64_t leaf_size = size_of(leaf_level_[leaf]);
        uint64_t m[3];
        point_coords(leaf_key_[leaf], m);
        const int b = (axis + 1) % 3, d = (axis + 2) % 3;
        const uint64_t oa = p[axis] - m[axis], ob = p[b] - m[b], od = p[d] - m[d];
        const uint64_t half = leaf_size >> 1;
        const bool b_end = ob == 0 || ob == leaf_size, d_end = od == 0 || od == leaf_size;
        if (b_end && d_end) {
            int offset[3] = {0, 0, 0};
            offset[b] = ob ? 1 : 0;
            offset[d] = od ? 1 : 0;
            const int e = local_edge(axis, offset);
            if (size == leaf_size) return oa == 0 ? 2 * e : -1;
            if (size == half && (oa == 0 || oa == half)) return 2 * e + (oa ? 1 : 0);
            return -1;
        }
        if (size != half || (oa != 0 && oa != half)) return -1;
        int normal;
        uint64_t normal_offset;
        if (b_end && od == half) {
            normal = b;
            normal_offset = ob;
        } else if (d_end && ob == half) {
            normal = d;
            normal_offset = od;
        } else {
            return -1;
        }
        const int f = 2 * normal + (normal_offset ? 1 : 0);
        const int u = (normal + 1) % 3;
        // Cross halves along u are 1 (upper) and 3 (lower); along v, 2 and 0.
        const int k = axis == u ? (oa ? 1 : 3) : (oa ? 2 : 0);
        return 24 + 4 * f + k;
    }

    // One polygon per minimal sign-change edge, owned by the leaf above it
    // in both other axes, joining the vertices of the leaves around it that
    // hold its component (a leaf spanning two positions makes a triangle),
    // oriented outward and split along the shorter diagonal unless only the
    // other keeps both triangles facing the crossing's normal. Positions the
    // narrow band missed record their base cells.
    void build_triangles() {
        const size_t count = leaf_key_.size();
        const size_t parts = chunks_of(count, kLeafChunk);
        std::vector<std::vector<uint32_t>> tris(parts), owners(parts);
        std::vector<std::vector<uint8_t>> cached(parts);
        std::vector<std::vector<uint64_t>> missing(parts);
        parallel_chunks(parts, s_.threads, [&](size_t part) {
            for (size_t i = part * kLeafChunk, end = std::min(count, (part + 1) * kLeafChunk); i < end; ++i) {
                if (leaf_table_[i] == kNone) continue;
                const uint8_t *table = &tables_[size_t(kSlots) * leaf_table_[i]];
                const uint32_t status = leaf_status_[i];
                const uint64_t leaf_size = size_of(leaf_level_[i]);
                uint64_t m[3];
                point_coords(leaf_key_[i], m);
                for (int slot = 0; slot < kSlots; ++slot) {
                    if (table[slot] == kNoComponent) continue;
                    int lo, hi, axis;
                    bool half;
                    slot_geometry(slot, status, lo, hi, axis, half);
                    int g[3];
                    grid_coords(lo, g);
                    const uint64_t p[3] = {m[0] + (g[0] * leaf_size >> 1), m[1] + (g[1] * leaf_size >> 1),
                                           m[2] + (g[2] * leaf_size >> 1)};
                    const uint64_t size = half ? leaf_size >> 1 : leaf_size;
                    const int b = (axis + 1) % 3, d = (axis + 2) % 3;
                    // Positions counter-clockwise about +axis: (+b, +d), (-b, +d), (-b, -d), (+b, -d).
                    static const int kSigns[4][2] = {{1, 1}, {-1, 1}, {-1, -1}, {1, -1}};
                    uint32_t leaves[4];
                    bool complete = true;
                    for (int q = 0; q < 4; ++q) {
                        int64_t c[3] = {int64_t(p[0]), int64_t(p[1]), int64_t(p[2])};
                        if (kSigns[q][0] < 0) c[b] -= 1;
                        if (kSigns[q][1] < 0) c[d] -= 1;
                        leaves[q] = q == 0 && c[b] >= int64_t(m[b]) && c[d] >= int64_t(m[d]) &&
                                            c[b] < int64_t(m[b] + leaf_size) && c[d] < int64_t(m[d] + leaf_size)
                                        ? static_cast<uint32_t>(i)
                                        : locate(c);
                        if (leaves[q] == kNone) {
                            complete = false;
                            if (in_range(c))
                                missing[part].push_back(point_key(uint64_t(c[0]) >> depth_ << depth_,
                                                                  uint64_t(c[1]) >> depth_ << depth_,
                                                                  uint64_t(c[2]) >> depth_ << depth_));
                        }
                    }
                    if (leaves[0] != i) continue;  // another leaf owns this edge
                    if (!complete) continue;
                    uint32_t vertex[4];
                    bool reused = true;
                    for (int q = 0; q < 4; ++q) {
                        const uint32_t leaf = leaves[q];
                        reused = reused && leaf_reused_[leaf] != kNone;
                        const int s = slot_of(leaf, p, axis, size);
                        if (s < 0 || leaf_table_[leaf] == kNone)
                            throw Error("internal error: a leaf around a surface edge does not hold it");
                        const uint8_t component = tables_[size_t(kSlots) * leaf_table_[leaf] + s];
                        if (component == kNoComponent)
                            throw Error("internal error: a leaf around a surface edge has no crossing there");
                        vertex[q] = leaf_first_vertex_[leaf] + component;
                    }
                    // Drop a repeated leaf (it spans two positions).
                    uint32_t polygon[4];
                    int n = 0;
                    for (int q = 0; q < 4; ++q)
                        if (leaves[q] != leaves[(q + 3) % 4]) polygon[n++] = vertex[q];
                    if (n < 3) throw Error("internal error: a surface edge has fewer than three cells around it");
                    uint32_t edge_level;
                    uint64_t key, point;
                    slot_key(i, lo, axis, half, edge_level, key, point);
                    const uint32_t index = crossing_index(edge_level, key);
                    const Crossing &c = crossings_[index];
                    // Outward is -axis when the edge's lower end is outside.
                    if (!(point_value(point) < 0)) std::reverse(polygon, polygon + n);
                    std::vector<uint32_t> &out = tris[part];
                    const size_t before = out.size();
                    if (n == 3) {
                        out.insert(out.end(), {polygon[0], polygon[1], polygon[2]});
                    } else {
                        const double *pos[4];
                        for (int q = 0; q < 4; ++q) pos[q] = &vertex_positions_[3 * size_t(polygon[q])];
                        auto distance2 = [](const double *u, const double *v) {
                            const double dx = u[0] - v[0], dy = u[1] - v[1], dz = u[2] - v[2];
                            return dx * dx + dy * dy + dz * dz;
                        };
                        // Whether triangle (a, b, c) faces against the crossing's normal.
                        auto folds = [&](int a, int b2, int c2) {
                            double u[3], v[3];
                            for (int k = 0; k < 3; ++k) {
                                u[k] = pos[b2][k] - pos[a][k];
                                v[k] = pos[c2][k] - pos[a][k];
                            }
                            return (u[1] * v[2] - u[2] * v[1]) * c.n[0] + (u[2] * v[0] - u[0] * v[2]) * c.n[1] +
                                       (u[0] * v[1] - u[1] * v[0]) * c.n[2] < 0;
                        };
                        bool first_diagonal = distance2(pos[0], pos[2]) <= distance2(pos[1], pos[3]);
                        const bool first_folds = folds(0, 1, 2) || folds(0, 2, 3);
                        const bool second_folds = folds(0, 1, 3) || folds(1, 2, 3);
                        if (first_folds != second_folds) first_diagonal = second_folds;
                        if (first_diagonal) {
                            out.insert(out.end(), {polygon[0], polygon[1], polygon[2], polygon[0], polygon[2], polygon[3]});
                        } else {
                            out.insert(out.end(), {polygon[0], polygon[1], polygon[3], polygon[1], polygon[2], polygon[3]});
                        }
                    }
                    for (size_t t = before; t < out.size(); t += 3) {
                        owners[part].push_back(index);
                        cached[part].push_back(reused ? 1 : 0);
                    }
                }
            }
        });
        triangles_ = concatenate(tris);
        triangle_edge_ = concatenate(owners);
        triangle_reusable_ = concatenate(cached);
        separate_diagonals();
        missing_ = concatenate(missing);
        std::sort(missing_.begin(), missing_.end());
        missing_.erase(std::unique(missing_.begin(), missing_.end()), missing_.end());
        // Positions inside an existing root are leaves balance has yet to create; only new roots repair.
        missing_.erase(std::remove_if(missing_.begin(), missing_.end(),
                                      [&](uint64_t key) { return node_exists(key, 0); }),
                       missing_.end());
    }

    // Two quads that share three vertices (a fold of the polygons around
    // nearby edges, as at thin sharp corners) must not both be cut along the
    // diagonal joining two of them: that edge would have four triangles.
    // Such a quad, the later one first, takes its other diagonal.
    void separate_diagonals() {
        const size_t count = triangles_.size() / 3;
        triangle_flipped_.assign(count, 0);
        std::vector<uint64_t> overused = overused_edges();
        if (!overused.empty()) {
            // Rare: count every edge exactly while flipping.
            KeyMap index;
            index.reserve(3 * count / 2);
            std::vector<uint32_t> uses;
            auto edge = [&](uint32_t a, uint32_t b) -> uint32_t & {
                const uint32_t slot = index.insert(surface_edge_key(a, b), static_cast<uint32_t>(uses.size()));
                if (slot == uses.size()) uses.push_back(0);
                return uses[slot];
            };
            auto count_triangle = [&](const uint32_t *v, int delta) {
                for (int k = 0; k < 3; ++k) edge(v[k], v[(k + 1) % 3]) += delta;
            };
            for (size_t t = 0; t < count; ++t) count_triangle(&triangles_[3 * t], 1);
            bool flipped = false;
            for (size_t t = count; t-- > 1;) {
                if (triangle_edge_[t] != triangle_edge_[t - 1]) continue;
                uint32_t *first = &triangles_[3 * (t - 1)], *second = &triangles_[3 * t];
                // Layouts: (p0 p1 p2)(p0 p2 p3) cut along p0-p2, (p0 p1 p3)(p1 p2 p3) along p1-p3.
                const bool along_02 = second[0] == first[0];
                const uint32_t p[4] = {first[0], first[1], along_02 ? first[2] : second[1],
                                       along_02 ? second[2] : first[2]};
                const uint32_t a = along_02 ? p[0] : p[1], b = along_02 ? p[2] : p[3];
                if (edge(a, b) <= 2) continue;
                count_triangle(first, -1);
                count_triangle(second, -1);
                const uint32_t other[6] = {p[0], p[1], p[3], p[1], p[2], p[3]};
                const uint32_t straight[6] = {p[0], p[1], p[2], p[0], p[2], p[3]};
                std::copy_n(along_02 ? other : straight, 3, first);
                std::copy_n((along_02 ? other : straight) + 3, 3, second);
                count_triangle(first, 1);
                count_triangle(second, 1);
                triangle_flipped_[t - 1] = triangle_flipped_[t] = 1;
                flipped = true;
            }
            if (flipped) overused = overused_edges();
        }
        // Edges still shared by more than two triangles: the two leaves'
        // components meet in two segments of one face (an ambiguous face
        // whose crossings each leaf joins into one component, as where a
        // coarse leaf meets finer ones). Their leaves split first.
        overused_.clear();
        for (uint64_t key : overused) {
            overused_.push_back(vertex_leaf_[key >> 32]);
            overused_.push_back(vertex_leaf_[static_cast<uint32_t>(key)]);
        }
    }

    // Edges (surface_edge_key) of more than two triangles, sorted: a
    // parallel sort of every triangle's edges.
    std::vector<uint64_t> overused_edges() const {
        const size_t count = triangles_.size() / 3;
        const int width = std::max(1, bit_width(vertex_positions_.size() / 3));
        std::vector<uint64_t> edges(3 * count);
        parallel_chunks(chunks_of(count, kChunk), s_.threads, [&](size_t chunk) {
            for (size_t t = chunk * kChunk, end = std::min(count, (chunk + 1) * kChunk); t < end; ++t)
                for (int k = 0; k < 3; ++k) {
                    const uint64_t key = surface_edge_key(triangles_[3 * t + k], triangles_[3 * t + (k + 1) % 3]);
                    edges[3 * t + k] = (key >> 32) << width | (key & 0xffffffffu);
                }
        });
        radix_sort_items(edges, 0, 2 * width, s_.threads);
        std::vector<uint64_t> overused;
        const uint64_t mask = (uint64_t(1) << width) - 1;
        for (size_t i = 0; i < edges.size();) {
            size_t j = i + 1;
            while (j < edges.size() && edges[j] == edges[i]) ++j;
            if (j - i > 2) overused.push_back((edges[i] >> width) << 32 | (edges[i] & mask));
            i = j;
        }
        return overused;
    }

    // ---------------------------------------------------------------- measurement

    // Measures every triangle (reusing the previous pass's results for
    // triangles of edges whose cells all kept their vertices): the largest
    // estimated distance |f| / |grad f| at its corners, edge midpoints,
    // centroid and three interior points, with |grad f| from the centroid,
    // and the field normal at the centroid.
    void measure() {
        const size_t count = triangles_.size() / 3;
        results_.assign(count, TriangleResult{0, 1, {0, 0, 0}});
        std::vector<uint32_t> todo;
        for (size_t t = 0; t < count; ++t) {
            const uint32_t edge = triangle_edge_[t];
            if (triangle_reusable_[t] && edge < cache_.result_of_crossing.size() &&
                cache_.result_of_crossing[edge] != kNone &&
                cache_.flipped[cache_.result_of_crossing[edge]] == triangle_flipped_[t]) {
                // The edge's triangles come in order; find this one's offset.
                size_t offset = 0;
                while (t >= offset + 1 && triangle_edge_[t - offset - 1] == edge) ++offset;
                if (offset < cache_.results_per_crossing[edge]) {
                    results_[t] = cache_.results[cache_.result_of_crossing[edge] + offset];
                    continue;
                }
            }
            todo.push_back(static_cast<uint32_t>(t));
        }
        measure_triangles(field_, s_.threads, s_.spatial_order, s_.gradient_step, vertex_positions_.data(),
                          triangles_.data(), vertex_values_.data(), vertex_slopes_.data(), todo, results_);
    }

    // ---------------------------------------------------------------- refinement

    // Measures the pass (largest deviation, flagged triangles) and, when
    // `allowed`, splits the leaves that own the vertices of triangles outside
    // the tolerance or the normal angle, leaves whose sharp vertex leaves a
    // QEF residual above the tolerance, and leaves at non-manifold edges,
    // largest error first within the triangle budget and max_cells.
    // Returns whether any leaf split.
    bool refine(bool allowed) {
        const size_t count = leaf_key_.size();
        const double tolerance = s_.tolerance;
        std::vector<double> score(count, 0.0);
        std::vector<uint32_t> incident(count, 0);
        const size_t triangles = triangles_.size() / 3;
        std::vector<uint8_t> flagged(triangles, 0);
        max_deviation_ = 0;
        for (size_t t = 0; t < triangles; ++t) {
            const TriangleResult &r = results_[t];
            max_deviation_ = std::fmax(max_deviation_, r.deviation);
            const uint32_t *v = &triangles_[3 * t];
            for (int k = 0; k < 3; ++k) ++incident[vertex_leaf_[v[k]]];
            double s = r.deviation / tolerance;
            const uint8_t all = vertex_marks_[v[0]] & vertex_marks_[v[1]] & vertex_marks_[v[2]];
            const uint8_t any = vertex_marks_[v[0]] | vertex_marks_[v[1]] | vertex_marks_[v[2]];
            // Beside a sharp feature a triangle must lie on one face: across a
            // crease of angle theta its facet turns by about theta / 2 from both.
            const double limit = any & kSharpVertex ? cos_feature_ : cos_normal_;
            if (!(all & kSharpVertex) && r.cosine < limit) s = std::fmax(s, 1 + (limit - r.cosine));
            if (!(s > 1)) continue;
            flagged[t] = 1;
            for (int k = 0; k < 3; ++k) {
                double &leaf_score = score[vertex_leaf_[v[k]]];
                leaf_score = std::fmax(leaf_score, s);
            }
        }
        for (uint32_t leaf : overused_) score[leaf] = kTopology;
        for (size_t v = 0; v < vertex_residuals_.size(); ++v)
            if ((vertex_marks_[v] & kSharpVertex) && vertex_residuals_[v] > tolerance) {
                double &leaf_score = score[vertex_leaf_[v]];
                leaf_score = std::fmax(leaf_score, vertex_residuals_[v] / tolerance);
            }
        // Display creases: the interior normals of the triangles around a
        // vertex that is not on a sharp feature differ by more than the
        // crease angle (a smooth feature too narrow for its triangles, which
        // would shade as a broken crease).
        {
            const size_t vertices = vertex_marks_.size();
            std::vector<uint32_t> first(vertices + 1, 0), fan(triangles_.size());
            for (uint32_t v : triangles_) ++first[v + 1];
            for (size_t v = 0; v < vertices; ++v) first[v + 1] += first[v];
            std::vector<uint32_t> fill(first.begin(), first.end() - 1);
            for (size_t c = 0; c < triangles_.size(); ++c) fan[fill[triangles_[c]]++] = static_cast<uint32_t>(c / 3);
            std::vector<double> turn(vertices, 1.0);
            parallel_chunks(chunks_of(vertices, kChunk), s_.threads, [&](size_t chunk) {
                for (size_t v = chunk * kChunk, end = std::min(vertices, (chunk + 1) * kChunk); v < end; ++v) {
                    if (vertex_marks_[v] & kSharpVertex) continue;
                    double lowest = 1;
                    for (uint32_t i = first[v]; i < first[v + 1]; ++i)
                        for (uint32_t j = i + 1; j < first[v + 1]; ++j) {
                            const double *a = results_[fan[i]].normal, *b = results_[fan[j]].normal;
                            lowest = std::fmin(lowest, a[0] * b[0] + a[1] * b[1] + a[2] * b[2]);
                        }
                    turn[v] = lowest;
                }
            });
            for (size_t v = 0; v < vertices; ++v) {
                if (!(turn[v] < cos_crease_)) continue;
                double &leaf_score = score[vertex_leaf_[v]];
                leaf_score = std::fmax(leaf_score, 1 + (cos_crease_ - turn[v]));
                for (uint32_t i = first[v]; i < first[v + 1]; ++i) flagged[fan[i]] = 1;
            }
        }
        flagged_ = 0;
        for (uint8_t f : flagged) flagged_ += f;
        // Discontinuities of the field never meet a tolerance: their leaves stay.
        for (size_t v = 0; v < vertex_marks_.size(); ++v)
            if (vertex_marks_[v] & kJumpVertex) score[vertex_leaf_[v]] = 0;
        // After the last round only leaves at edges of more than two
        // triangles split (for as many rounds again).
        if (!allowed && rounds_ >= 2 * (depth_ + kExtraRounds)) return false;
        std::vector<uint32_t> candidates;
        for (size_t i = 0; i < count; ++i)
            if (score[i] > 1 && leaf_level_[i] < depth_ && (allowed || score[i] == kTopology))
                candidates.push_back(static_cast<uint32_t>(i));
        if (candidates.empty()) return false;
        std::stable_sort(candidates.begin(), candidates.end(),
                         [&](uint32_t a, uint32_t b) { return score[a] > score[b]; });
        // A split leaf's share of the triangles (a third of those at its
        // vertices) roughly quadruples; the growth the last round's splits
        // actually caused (balance included) scales that estimate up.
        if (predicted_ > 0) growth_scale_ = std::fmax(growth_scale_, (double(triangles) - before_) / predicted_);
        // A round spends at most half the room left, so later rounds can
        // refine the coarser neighbours of the cells this one splits.
        double estimate = static_cast<double>(triangles);
        double leaves = static_cast<double>(count);
        const double budget = s_.max_triangles ? estimate + 0.5 * (double(s_.max_triangles) - estimate) : INFINITY;
        size_t chosen = 0;
        double predicted = 0;
        for (; chosen < candidates.size(); ++chosen) {
            const double growth = growth_scale_ * incident[candidates[chosen]];
            const bool topology = score[candidates[chosen]] == kTopology;  // never held back by the budget
            if (!topology && (estimate + growth > budget || leaves + growth_scale_ * 7 > double(s_.max_cells))) break;
            estimate += growth;
            predicted += growth;
            leaves += growth_scale_ * 7;
        }
        if (chosen < candidates.size()) budget_limited_ = true;
        if (chosen == 0) return false;
        before_ = static_cast<double>(triangles);
        predicted_ = predicted;
        candidates.resize(chosen);
        std::sort(candidates.begin(), candidates.end());
        for (uint32_t i : candidates) split(leaf_key_[i], leaf_level_[i]);
        refined_ += chosen;
        return true;
    }

    // Keeps this pass's leaves, vertices and measurements for the next.
    void keep_cache() {
        PassCache next;
        next.leaf_index.reset(leaf_key_.size());
        for (size_t i = 0; i < leaf_key_.size(); ++i) next.leaf_index.insert(leaf_key_[i], static_cast<uint32_t>(i));
        next.level = leaf_level_;
        next.status = leaf_status_;
        next.first_vertex = leaf_first_vertex_;
        next.vertex_count = leaf_vertex_count_;
        next.table = leaf_table_;
        next.tables = tables_;
        next.positions = vertex_positions_;
        next.marks = vertex_marks_;
        next.residuals = vertex_residuals_;
        next.values = vertex_values_;
        next.slopes = vertex_slopes_;
        next.normals = vertex_normals_;
        next.result_of_crossing.assign(crossings_.size(), kNone);
        next.results_per_crossing.assign(crossings_.size(), 0);
        next.results = results_;
        next.flipped = triangle_flipped_;
        for (size_t t = 0; t < triangle_edge_.size(); ++t) {
            const uint32_t edge = triangle_edge_[t];
            if (next.result_of_crossing[edge] == kNone) next.result_of_crossing[edge] = static_cast<uint32_t>(t);
            ++next.results_per_crossing[edge];
        }
        cache_ = std::move(next);
    }

    // Where two sheets of the surface touch along an edge narrower than the
    // finest cells (more than two triangles at an edge after refinement),
    // gives each vertex of such an edge one copy per fan of triangles joined
    // across its two-triangle edges, so the surface is edge-manifold with
    // coincident vertices along the pinch. Kept only if no edge of the
    // changed triangles then has an odd number of triangles.
    void separate_pinches() {
        if (overused_.empty()) return;  // this pass's triangles have no edge of more than two
        // Each round can leave a pinch whose sheets only the copies it made
        // separate around one end: repeat while that helps.
        for (int round = 0; round < 4 && !overused_edges().empty(); ++round)
            if (!separate_pinches_once()) break;
    }

    bool separate_pinches_once() {
        const size_t count = triangles_.size() / 3;
        KeyMap index;
        index.reserve(3 * count / 2);
        std::vector<uint32_t> uses;
        auto slot = [&](uint32_t a, uint32_t b) {
            const uint32_t i = index.insert(surface_edge_key(a, b), static_cast<uint32_t>(uses.size()));
            if (i == uses.size()) uses.push_back(0);
            return i;
        };
        for (size_t c = 0; c < triangles_.size(); ++c) ++uses[slot(triangles_[c], triangles_[c - c % 3 + (c + 1) % 3])];
        std::vector<uint32_t> pinched;
        for (size_t c = 0; c < triangles_.size(); ++c) {
            const uint32_t a = triangles_[c], b = triangles_[c - c % 3 + (c + 1) % 3];
            if (uses[index.find(surface_edge_key(a, b))] > 2) pinched.insert(pinched.end(), {a, b});
        }
        if (pinched.empty()) return false;
        std::sort(pinched.begin(), pinched.end());
        pinched.erase(std::unique(pinched.begin(), pinched.end()), pinched.end());
        const std::vector<uint32_t> before = triangles_;
        const size_t vertices = vertex_positions_.size() / 3;
        std::vector<uint32_t> corners;  // corners (3 t + k) at pinched vertices, by vertex
        for (size_t c = 0; c < triangles_.size(); ++c)
            if (std::binary_search(pinched.begin(), pinched.end(), triangles_[c])) corners.push_back(static_cast<uint32_t>(c));
        std::stable_sort(corners.begin(), corners.end(), [&](uint32_t x, uint32_t y) { return triangles_[x] < triangles_[y]; });
        auto fan = [&](uint32_t v) {
            const auto lo = std::lower_bound(corners.begin(), corners.end(), v,
                                             [&](uint32_t c, uint32_t value) { return triangles_[c] < value; });
            auto hi = lo;
            while (hi != corners.end() && triangles_[*hi] == v) ++hi;
            return std::make_pair(lo, hi);
        };
        // Whether triangles t and u are joined around pinched vertex v
        // through edges at v that have two triangles.
        auto fan_linked = [&](uint32_t v, uint32_t t, uint32_t u) {
            const auto range = fan(v);
            std::vector<uint32_t> stack{t}, seen{t};
            while (!stack.empty()) {
                const uint32_t current = stack.back();
                stack.pop_back();
                if (current == u) return true;
                const uint32_t *tri = &triangles_[3 * size_t(current)];
                for (auto it = range.first; it != range.second; ++it) {
                    const uint32_t next = *it / 3;
                    if (next == current || std::find(seen.begin(), seen.end(), next) != seen.end()) continue;
                    const uint32_t *other = &triangles_[3 * size_t(next)];
                    for (int k = 0; k < 3; ++k) {
                        const uint32_t x = tri[k];
                        if (x == v || std::find(other, other + 3, x) == other + 3) continue;
                        if (uses[index.find(surface_edge_key(v, x))] != 2) continue;
                        seen.push_back(next);
                        stack.push_back(next);
                        break;
                    }
                }
            }
            return false;
        };
        // At an edge with four triangles, two run along it each way: each
        // sheet pairs one of each, those joined around an end of the edge,
        // else those whose facets agree best.
        std::vector<std::pair<uint32_t, uint32_t>> sheets;  // triangle pairs joined across a pinched edge
        {
            std::vector<std::pair<uint64_t, uint32_t>> around;  // (edge key, triangle * 2 + runs from lower vertex)
            for (size_t c = 0; c < triangles_.size(); ++c) {
                const uint32_t a = triangles_[c], b = triangles_[c - c % 3 + (c + 1) % 3];
                if (uses[index.find(surface_edge_key(a, b))] == 4)
                    around.emplace_back(surface_edge_key(a, b), static_cast<uint32_t>(2 * (c / 3) + (a < b ? 1 : 0)));
            }
            std::sort(around.begin(), around.end());
            auto facet = [&](uint32_t t, double *n) {
                const double *p0 = &vertex_positions_[3 * size_t(triangles_[3 * t])];
                const double *p1 = &vertex_positions_[3 * size_t(triangles_[3 * t + 1])];
                const double *p2 = &vertex_positions_[3 * size_t(triangles_[3 * t + 2])];
                const double u[3] = {p1[0] - p0[0], p1[1] - p0[1], p1[2] - p0[2]};
                const double w[3] = {p2[0] - p0[0], p2[1] - p0[1], p2[2] - p0[2]};
                n[0] = u[1] * w[2] - u[2] * w[1];
                n[1] = u[2] * w[0] - u[0] * w[2];
                n[2] = u[0] * w[1] - u[1] * w[0];
                const double l = std::sqrt(n[0] * n[0] + n[1] * n[1] + n[2] * n[2]);
                for (int k = 0; k < 3; ++k) n[k] = l > 0 ? n[k] / l : 0;
            };
            for (size_t i = 0; i + 3 < around.size(); i += 4) {
                uint32_t forward[2], backward[2];
                int nf = 0, nb = 0;
                for (size_t j = i; j < i + 4; ++j) {
                    if (around[j].second & 1) {
                        if (nf < 2) forward[nf] = around[j].second >> 1;
                        ++nf;
                    } else {
                        if (nb < 2) backward[nb] = around[j].second >> 1;
                        ++nb;
                    }
                }
                if (nf != 2 || nb != 2) continue;
                // A sheet's two triangles are joined around an end of the
                // edge through two-triangle edges; else pair by facets.
                const uint32_t a = static_cast<uint32_t>(around[i].first >> 32), b = static_cast<uint32_t>(around[i].first);
                int linked = -1;  // 0: forward[0] with backward[0], 1: with backward[1]
                for (uint32_t end : {a, b}) {
                    if (linked >= 0) break;
                    const bool l0 = fan_linked(end, forward[0], backward[0]);
                    const bool l1 = fan_linked(end, forward[0], backward[1]);
                    if (l0 != l1) linked = l0 ? 0 : 1;
                }
                if (linked >= 0) {
                    sheets.emplace_back(forward[0], backward[linked]);
                    sheets.emplace_back(forward[1], backward[1 - linked]);
                    continue;
                }
                double f0[3], f1[3], b0[3], b1[3];
                facet(forward[0], f0);
                facet(forward[1], f1);
                facet(backward[0], b0);
                facet(backward[1], b1);
                const double straight = f0[0] * b0[0] + f0[1] * b0[1] + f0[2] * b0[2] + f1[0] * b1[0] + f1[1] * b1[1] + f1[2] * b1[2];
                const double crossed = f0[0] * b1[0] + f0[1] * b1[1] + f0[2] * b1[2] + f1[0] * b0[0] + f1[1] * b0[1] + f1[2] * b0[2];
                const bool keep = straight >= crossed;
                sheets.emplace_back(forward[0], keep ? backward[0] : backward[1]);
                sheets.emplace_back(forward[1], keep ? backward[1] : backward[0]);
            }
            for (auto &pair : sheets)
                if (pair.first > pair.second) std::swap(pair.first, pair.second);
            std::sort(sheets.begin(), sheets.end());
        }
        std::vector<uint32_t> copies;  // vertex each new copy duplicates
        std::vector<uint32_t> parent;
        for (size_t first = 0; first < corners.size();) {
            const uint32_t v = triangles_[corners[first]];
            size_t end = first;
            while (end < corners.size() && triangles_[corners[end]] == v) ++end;
            const size_t n = end - first;
            parent.resize(n);
            for (size_t i = 0; i < n; ++i) parent[i] = static_cast<uint32_t>(i);
            auto root = [&](uint32_t i) {
                while (parent[i] != i) i = parent[i] = parent[parent[i]];
                return i;
            };
            for (size_t i = 0; i < n; ++i)
                for (size_t j = i + 1; j < n; ++j) {
                    const uint32_t ci = corners[first + i], cj = corners[first + j];
                    const uint32_t *ti = &before[ci - ci % 3], *tj = &before[cj - cj % 3];
                    const std::pair<uint32_t, uint32_t> pair(std::min(ci, cj) / 3, std::max(ci, cj) / 3);
                    for (int k = 0; k < 3; ++k) {
                        const uint32_t x = ti[k];
                        if (x == v || std::find(tj, tj + 3, x) == tj + 3) continue;
                        const uint32_t n = uses[index.find(surface_edge_key(v, x))];
                        if (n != 2 && !(n == 4 && std::binary_search(sheets.begin(), sheets.end(), pair))) continue;
                        const uint32_t ri = root(static_cast<uint32_t>(i)), rj = root(static_cast<uint32_t>(j));
                        if (ri != rj) parent[std::max(ri, rj)] = std::min(ri, rj);
                    }
                }
            // Fans after the first (by first corner) take new copies.
            std::vector<uint32_t> copy_of(n, kNone);
            for (size_t i = 0; i < n; ++i) {
                const uint32_t r = root(static_cast<uint32_t>(i));
                if (r == root(0)) continue;
                if (copy_of[r] == kNone) {
                    copy_of[r] = static_cast<uint32_t>(vertices + copies.size());
                    copies.push_back(v);
                }
                triangles_[corners[first + i]] = copy_of[r];
            }
            first = end;
        }
        if (copies.empty()) return false;
        // Verify: every edge of a changed triangle has an even number of
        // triangles, two where the sheets came apart.
        KeyMap check;
        std::vector<uint32_t> counts;
        for (size_t c = 0; c < triangles_.size(); ++c) {
            const uint32_t i = check.insert(surface_edge_key(triangles_[c], triangles_[c - c % 3 + (c + 1) % 3]),
                                            static_cast<uint32_t>(counts.size()));
            if (i == counts.size()) counts.push_back(0);
            ++counts[i];
        }
        for (uint32_t c : corners)
            for (int k = 0; k < 3; ++k) {
                const size_t t = c - c % 3;
                if (counts[check.find(surface_edge_key(triangles_[t + k], triangles_[t + (k + 1) % 3]))] % 2) {
                    triangles_ = before;
                    return false;
                }
            }
        for (uint32_t v : copies) {
            for (int a = 0; a < 3; ++a) {
                vertex_positions_.push_back(vertex_positions_[3 * size_t(v) + a]);
                vertex_normals_.push_back(vertex_normals_[3 * size_t(v) + a]);
            }
            vertex_marks_.push_back(vertex_marks_[v]);
        }
        return true;
    }

    // The surface: vertices that triangles reference, in leaf order.
    Surface finish() {
        separate_pinches();
        Surface surface;
        surface.active_cells = active_cells_;
        std::vector<uint32_t> remap(vertex_positions_.size() / 3, kNone);
        for (uint32_t v : triangles_) remap[v] = 0;
        uint32_t next = 0;
        for (size_t v = 0; v < remap.size(); ++v) {
            if (remap[v] == kNone) continue;
            remap[v] = next++;
            surface.positions.insert(surface.positions.end(), &vertex_positions_[3 * v], &vertex_positions_[3 * v] + 3);
            surface.vertex_normals.insert(surface.vertex_normals.end(), &vertex_normals_[3 * v], &vertex_normals_[3 * v] + 3);
            surface.marks.push_back(vertex_marks_[v]);
        }
        surface.triangles.resize(triangles_.size());
        for (size_t i = 0; i < triangles_.size(); ++i) surface.triangles[i] = remap[triangles_[i]];
        surface.interior_normals.resize(3 * results_.size());
        for (size_t t = 0; t < results_.size(); ++t) std::copy_n(results_[t].normal, 3, &surface.interior_normals[3 * t]);
        surface.max_deviation = max_deviation_;
        surface.flagged_triangles = flagged_;
        surface.refined_cells = refined_;
        surface.balanced_cells = balanced_;
        surface.max_level = max_level_;
        surface.passes = passes_;
        surface.budget_limited = budget_limited_;
        return surface;
    }

    const MeshField &field_;
    const MeshSettings &s_;
    int depth_ = 0;
    double h_ = 0, fine_ = 0, cos_normal_ = 1, cos_feature_ = 1, cos_crease_ = -1;
    int64_t origin_[3] = {0, 0, 0};       // global fine index of the grid origin
    int64_t base_origin_[3] = {0, 0, 0};  // the same in base cells
    uint32_t cells_[3] = {0, 0, 0};  // base cells per axis
    int morton_bits_ = 0;

    // Octree.
    std::vector<uint64_t> roots_;
    KeyMap nodes_;
    size_t node_count_ = 0;
    std::vector<std::vector<uint64_t>> level_nodes_;
    // Leaves of the current pass (Morton order of min corners).
    std::vector<uint64_t> leaf_key_;
    std::vector<uint8_t> leaf_level_;
    KeyMap leaf_index_;
    std::vector<uint32_t> leaf_status_, leaf_reused_, leaf_first_vertex_, leaf_table_;
    std::vector<uint8_t> leaf_vertex_count_;
    std::vector<uint8_t> tables_;  // kSlots per active leaf: component per slot
    // Caches by lattice key.
    KeyMap points_;
    std::vector<double> point_values_;
    std::vector<KeyMap> crossing_maps_, face_maps_;  // by level
    std::vector<Crossing> crossings_;
    std::vector<double> face_values_;
    // Vertices and triangles of the current pass.
    std::vector<double> vertex_positions_, vertex_residuals_, vertex_values_, vertex_slopes_, vertex_normals_;
    std::vector<uint8_t> vertex_marks_;
    std::vector<uint32_t> vertex_leaf_;
    std::vector<uint32_t> triangles_, triangle_edge_;
    std::vector<uint8_t> triangle_reusable_, triangle_flipped_;
    std::vector<TriangleResult> results_;
    std::vector<uint64_t> missing_;
    std::vector<uint32_t> overused_;  // leaves at edges of more than two triangles
    PassCache cache_;
    // Report.
    uint64_t active_cells_ = 0, refined_ = 0, balanced_ = 0, flagged_ = 0;
    uint32_t max_level_ = 0, passes_ = 0;
    int repairs_ = 0, rounds_ = 0;
    bool budget_limited_ = false;
    double max_deviation_ = 0;
    // Refinement budgeting: triangles before the last round's splits, the
    // growth they were predicted to add, and the ratio of actual to predicted.
    double before_ = 0, predicted_ = 0, growth_scale_ = kBudgetMargin;
};

}  // namespace

Surface extract_surface(const MeshField &field, const MeshSettings &settings) {
    return Extractor(field, settings).run();
}

}  // namespace rsdf
