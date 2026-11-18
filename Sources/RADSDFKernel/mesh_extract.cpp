// Sharp-feature dual contouring on a sparse grid (SDF-DESIGN.md, "Meshing").
//
// Cells and corners are identified by linear keys i + N0 j + N0 N1 k on the
// corner lattice (N = cells + 1 per axis), so a fixed lattice offset is a
// constant key offset. Every per-cell or per-edge lookup of a neighbour is
// then a monotone scan ("cursor") through sorted key arrays, and the unique
// corners of the candidate cells are a merge of eight shifted copies of the
// sorted cell list. Batches that evaluate the field are formed in Morton
// order so they stay spatially compact for culling.
#include "mesh.h"

#include <algorithm>

namespace rsdf {

void MeshField::evaluate(int n, const double *x, const double *y, const double *z, double *out) const {
    eval_batch(*graph, node, n, x, y, z, out);
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
constexpr uint32_t kMaxAxisCells = (1u << 20) - 8;  // Morton codes keep 21 bits per axis
constexpr int kMaxRepairPasses = 64;
constexpr double kHalfDiagonal = 0.8660254037844386;  // sqrt(3) / 2
constexpr size_t kChunk = 8192;
// Grid planes sit this irrational fraction of a cell off the bounds, so faces
// at simple multiples of the cell size from the bounds never lie on a grid
// plane: a face through grid corners makes crossings land exactly on sharp
// edges, where finite-difference normals are blends that misplace vertices.
constexpr double kGridShift = 0.3819660112501051;  // 2 - golden ratio

// ------------------------------------------------------------------ keys

struct Grid {
    double origin[3];
    double h;
    uint32_t cells[3];
    uint64_t stride[3];  // 1, N0, N0 N1

    uint64_t key(const uint64_t c[3]) const { return c[0] + stride[1] * c[1] + stride[2] * c[2]; }
    void coords(uint64_t key, uint64_t c[3]) const {
        c[2] = key / stride[2];
        key -= c[2] * stride[2];
        c[1] = key / stride[1];
        c[0] = key - c[1] * stride[1];
    }
    double at(int axis, uint64_t index) const { return origin[axis] + h * static_cast<double>(index); }
};


// Finds keys in a sorted array for nondecreasing queries.
class Cursor {
public:
    Cursor(const std::vector<uint64_t> &keys, uint64_t first)
        : keys_(keys.data()), size_(keys.size()),
          pos_(static_cast<size_t>(std::lower_bound(keys.begin(), keys.end(), first) - keys.begin())) {}

    uint32_t find(uint64_t key) {
        while (pos_ < size_ && keys_[pos_] < key) ++pos_;
        return pos_ < size_ && keys_[pos_] == key ? static_cast<uint32_t>(pos_) : kNone;
    }

private:
    const uint64_t *keys_;
    size_t size_;
    size_t pos_;
};

inline uint64_t saturating_sub(uint64_t a, uint64_t b) { return a > b ? a - b : 0; }

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

// ------------------------------------------------------------------ extraction

struct Edge {
    uint32_t c0, c1;  // corner indices; c1 = c0 + one lattice step along axis
    uint32_t axis;
};

struct Crossing {
    double p[3];
    double n[3];
};

// Cell topology. Local corner index: dx + 2 dy + 4 dz. Local edge index:
// 4 axis + o_lo + 2 o_hi, with o_lo, o_hi the edge's offsets along the other
// two axes in increasing axis order.
struct CellTopology {
    uint8_t edge_start[12];     // local corner at the lower end of each edge
    uint8_t edge_axis[12];
    uint8_t face_corners[6][4]; // cyclic order
    uint8_t face_edges[6][4];   // face_edges[f][k] joins face corners k and k + 1
};

int local_edge(int axis, const int offset[3]) {
    const int lo = axis == 0 ? 1 : 0, hi = axis == 2 ? 1 : 2;
    return 4 * axis + offset[lo] + 2 * offset[hi];
}

const CellTopology &topology() {
    static const CellTopology table = [] {
        CellTopology t{};
        for (int e = 0; e < 12; ++e) {
            const int axis = e / 4, lo = axis == 0 ? 1 : 0, hi = axis == 2 ? 1 : 2;
            int offset[3] = {0, 0, 0};
            offset[lo] = e & 1;
            offset[hi] = (e >> 1) & 1;
            t.edge_start[e] = static_cast<uint8_t>(offset[0] + 2 * offset[1] + 4 * offset[2]);
            t.edge_axis[e] = static_cast<uint8_t>(axis);
        }
        static const int kCycle[4][2] = {{0, 0}, {1, 0}, {1, 1}, {0, 1}};
        for (int axis = 0; axis < 3; ++axis)
            for (int side = 0; side < 2; ++side) {
                const int face = 2 * axis + side, u = (axis + 1) % 3, v = (axis + 2) % 3;
                for (int k = 0; k < 4; ++k) {
                    int corner[3];
                    corner[axis] = side;
                    corner[u] = kCycle[k][0];
                    corner[v] = kCycle[k][1];
                    t.face_corners[face][k] = static_cast<uint8_t>(corner[0] + 2 * corner[1] + 4 * corner[2]);
                    // Even k runs along u, odd k along v; the start is the lower of the two corners.
                    const int along = k % 2 == 0 ? u : v, across = k % 2 == 0 ? v : u;
                    int start[3];
                    start[axis] = side;
                    start[along] = 0;
                    start[across] = k == 1 || k == 2 ? 1 : 0;
                    t.face_edges[face][k] = static_cast<uint8_t>(local_edge(along, start));
                }
            }
        return t;
    }();
    return table;
}

// Whether a face's corner signs alternate (inside corners diagonal).
inline bool ambiguous_face(const bool inside[8], const uint8_t corners[4]) {
    return inside[corners[0]] == inside[corners[2]] && inside[corners[1]] == inside[corners[3]] &&
           inside[corners[0]] != inside[corners[1]];
}

// Groups a cell's sign-change edges into surface components, like the patches
// of marching cubes: on each face the crossings pair into segments, and edges
// linked through segments share a component. An ambiguous face (inside corners
// diagonal) joins its inside corners when the field at the face centre
// (centre[face]) is inside. Near sharp edges the field is a max or min of
// smooth fields, not bilinear, so corner values alone misjudge such faces.
// Both cells sharing a face read the same centre value and pair it the same
// way. Writes each sign-change edge's component (numbered by lowest edge) and
// returns the count.
int cell_components(const double value[8], const double centre[6], uint8_t component[12]) {
    const CellTopology &t = topology();
    bool inside[8];
    for (int c = 0; c < 8; ++c) inside[c] = value[c] < 0;
    uint8_t parent[12];
    for (uint8_t e = 0; e < 12; ++e) parent[e] = e;
    auto root = [&](int e) {
        while (parent[e] != e) e = parent[e] = parent[parent[e]];
        return e;
    };
    auto unite = [&](int a, int b) {
        a = root(a);
        b = root(b);
        if (a != b) parent[std::max(a, b)] = static_cast<uint8_t>(std::min(a, b));
    };
    for (int face = 0; face < 6; ++face) {
        const uint8_t *corners = t.face_corners[face], *edges = t.face_edges[face];
        bool change[4];
        int changes = 0;
        for (int k = 0; k < 4; ++k) changes += change[k] = inside[corners[k]] != inside[corners[(k + 1) % 4]];
        if (changes == 2) {
            int first = -1;
            for (int k = 0; k < 4; ++k) {
                if (!change[k]) continue;
                if (first < 0) first = k;
                else unite(edges[first], edges[k]);
            }
        } else if (changes == 4) {
            const int in = inside[corners[0]] ? 0 : 1;  // inside corners: in and in + 2
            const bool join_inside = centre[face] < 0;
            // Each corner the segments cut off pairs its two adjacent edges.
            const int separated = join_inside ? in + 1 : in;
            for (int k : {separated % 4, (separated + 2) % 4}) unite(edges[(k + 3) % 4], edges[k]);
        }
    }
    uint8_t label[12];
    std::fill(label, label + 12, 0xff);
    int count = 0;
    for (int e = 0; e < 12; ++e) {
        const int start = t.edge_start[e];
        if (inside[start] == inside[start + (1 << t.edge_axis[e])]) continue;
        const int r = root(e);
        if (label[r] == 0xff) label[r] = static_cast<uint8_t>(count++);
        component[e] = label[r];
    }
    return count;
}

class Extractor {
public:
    Extractor(const MeshField &field, const MeshSettings &settings) : field_(field), s_(settings) {
        for (int a = 0; a < 3; ++a) {
            const double span = (field.hi[a] - field.lo[a]) / settings.cell_size + kGridShift;
            if (!(span <= kMaxAxisCells - 4))
                throw Error(format("meshing needs too many cells along %c (more than %u); increase cell_size",
                                   "xyz"[a], kMaxAxisCells));
            grid_.cells[a] = static_cast<uint32_t>(std::ceil(span)) + 4;  // at least two cells of padding per side
            grid_.origin[a] = field.lo[a] - (2 + kGridShift) * settings.cell_size;
        }
        grid_.h = settings.cell_size;
        grid_.stride[0] = 1;
        grid_.stride[1] = grid_.cells[0] + 1ull;
        grid_.stride[2] = grid_.stride[1] * (grid_.cells[1] + 1ull);
        key_bits_ = bit_width(grid_.stride[2] * (grid_.cells[2] + 1ull));
        morton_bits_ = 3 * bit_width(std::max({grid_.cells[0], grid_.cells[1], grid_.cells[2]}) + 1ull);
    }

    Surface run() {
        cells_ = narrow_band();
        for (int pass = 0;; ++pass) {
            build_corners();
            build_edges();
            build_faces();
            build_vertices();
            std::vector<uint64_t> missing = build_triangles();
            if (missing.empty() || pass + 1 == kMaxRepairPasses) break;
            // Cells around sign changes that the narrow band missed: add them and extract again.
            std::vector<uint64_t> merged;
            merged.reserve(cells_.size() + missing.size());
            std::merge(cells_.begin(), cells_.end(), missing.begin(), missing.end(), std::back_inserter(merged));
            merged.erase(std::unique(merged.begin(), merged.end()), merged.end());
            if (merged.size() > s_.max_cells) too_many_cells(s_.max_cells);
            cells_.swap(merged);
        }
        return finish();
    }

private:
    // Hierarchical narrow band: blocks whose centre value proves they cannot
    // reach the surface are dropped before subdividing (step 2).
    std::vector<uint64_t> narrow_band() {
        uint32_t side = 1;
        auto blocks = [&](int a, uint32_t size) { return (grid_.cells[a] + size - 1) / size; };
        while (blocks(0, side) > 64 || blocks(1, side) > 64 || blocks(2, side) > 64) side *= 2;
        std::vector<uint64_t> level;
        for (uint64_t code = 0; code < (uint64_t(1) << 18); ++code) {
            const uint64_t c[3] = {compact3(code), compact3(code >> 1), compact3(code >> 2)};
            if (c[0] < blocks(0, side) && c[1] < blocks(1, side) && c[2] < blocks(2, side)) level.push_back(code);
        }
        std::vector<uint8_t> keep;
        while (true) {
            const double size = side * grid_.h;
            const double reach = s_.lipschitz * kHalfDiagonal * size * (1 + 1e-9);
            keep.assign(level.size(), 0);
            parallel_chunks(chunks_of(level.size(), kChunk), s_.threads, [&](size_t chunk) {
                double x[kBatch], y[kBatch], z[kBatch], v[kBatch];
                const size_t end = std::min(level.size(), (chunk + 1) * kChunk);
                for (size_t first = chunk * kChunk; first < end; first += kBatch) {
                    const int n = static_cast<int>(std::min<size_t>(kBatch, end - first));
                    for (int i = 0; i < n; ++i) {
                        const uint64_t code = level[first + i];
                        x[i] = grid_.origin[0] + size * (compact3(code) + 0.5);
                        y[i] = grid_.origin[1] + size * (compact3(code >> 1) + 0.5);
                        z[i] = grid_.origin[2] + size * (compact3(code >> 2) + 0.5);
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
                    const uint64_t c[3] = {compact3(level[i]), compact3(level[i] >> 1), compact3(level[i] >> 2)};
                    cells.push_back(grid_.key(c));
                }
                radix_sort(cells, nullptr, key_bits_);
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
                    if (compact3(child) * half < grid_.cells[0] && compact3(child >> 1) * half < grid_.cells[1] &&
                        compact3(child >> 2) * half < grid_.cells[2])
                        next.push_back(child);
                }
            }
            level.swap(next);
            side = half;
        }
    }

    // The unique corners of the candidate cells (step 3) with their values;
    // values known from an earlier repair pass are reused.
    void build_corners() {
        const uint64_t s1 = grid_.stride[1], s2 = grid_.stride[2];
        const uint64_t offsets[8] = {0, 1, s1, s1 + 1, s2, s2 + 1, s2 + s1, s2 + s1 + 1};
        const size_t parts = chunks_of(cells_.size(), kChunk);
        std::vector<std::vector<uint64_t>> pieces(parts);
        parallel_chunks(parts, s_.threads, [&](size_t part) {
            // This part owns corner keys in [lo, hi): a merge of the shifted cell keys there.
            const uint64_t lo = cells_[part * kChunk];
            const bool last = part + 1 == parts;
            const uint64_t hi = last ? UINT64_MAX : cells_[(part + 1) * kChunk];
            auto at_or_after = [&](uint64_t key) {
                return cells_.data() + (std::lower_bound(cells_.begin(), cells_.end(), key) - cells_.begin());
            };
            const uint64_t *begin[8], *end[8];
            for (int d = 0; d < 8; ++d) {
                begin[d] = at_or_after(saturating_sub(lo, offsets[d]));
                end[d] = last ? cells_.data() + cells_.size() : at_or_after(saturating_sub(hi, offsets[d]));
            }
            std::vector<uint64_t> &out = pieces[part];
            while (true) {
                uint64_t best = UINT64_MAX;
                for (int d = 0; d < 8; ++d)
                    if (begin[d] < end[d]) best = std::min(best, *begin[d] + offsets[d]);
                if (best == UINT64_MAX) break;
                out.push_back(best);
                for (int d = 0; d < 8; ++d)
                    if (begin[d] < end[d] && *begin[d] + offsets[d] == best) ++begin[d];
            }
        });
        std::vector<uint64_t> keys = concatenate(pieces);
        if (keys.size() >= kNone) throw Error("meshing needs too many grid corners; increase cell_size");

        // Reuse values from the previous pass and evaluate the rest in Morton order.
        std::vector<double> values(keys.size());
        std::vector<uint32_t> todo;
        size_t old = 0;
        for (size_t i = 0; i < keys.size(); ++i) {
            while (old < corner_keys_.size() && corner_keys_[old] < keys[i]) ++old;
            if (old < corner_keys_.size() && corner_keys_[old] == keys[i]) values[i] = corner_values_[old];
            else todo.push_back(static_cast<uint32_t>(i));
        }
        std::vector<uint64_t> codes(todo.size());
        for (size_t i = 0; i < todo.size(); ++i) {
            uint64_t c[3];
            grid_.coords(keys[todo[i]], c);
            codes[i] = morton(c);
        }
        radix_sort(codes, &todo, morton_bits_);
        parallel_chunks(chunks_of(todo.size(), kChunk), s_.threads, [&](size_t chunk) {
            double x[kBatch], y[kBatch], z[kBatch], v[kBatch];
            const size_t end = std::min(todo.size(), (chunk + 1) * kChunk);
            for (size_t first = chunk * kChunk; first < end; first += kBatch) {
                const int n = static_cast<int>(std::min<size_t>(kBatch, end - first));
                for (int i = 0; i < n; ++i) {
                    uint64_t c[3];
                    grid_.coords(keys[todo[first + i]], c);
                    x[i] = grid_.at(0, c[0]);
                    y[i] = grid_.at(1, c[1]);
                    z[i] = grid_.at(2, c[2]);
                }
                field_.evaluate(n, x, y, z, v);
                for (int i = 0; i < n; ++i) values[todo[first + i]] = v[i];
            }
        });
        corner_keys_.swap(keys);
        corner_values_.swap(values);
    }

    // Sign-change grid edges between known corners (step 4) and their
    // refined crossings; crossings known from an earlier pass are reused.
    void build_edges() {
        const std::vector<uint64_t> &keys = corner_keys_;
        const std::vector<double> &values = corner_values_;
        const size_t parts = chunks_of(keys.size(), 2 * kChunk);
        std::vector<std::vector<Edge>> pieces(parts);
        parallel_chunks(parts, s_.threads, [&](size_t part) {
            const size_t begin = part * 2 * kChunk, end = std::min(keys.size(), begin + 2 * kChunk);
            Cursor next_y(keys, keys[begin] + grid_.stride[1]), next_z(keys, keys[begin] + grid_.stride[2]);
            std::vector<Edge> &out = pieces[part];
            for (size_t c = begin; c < end; ++c) {
                const bool inside = values[c] < 0;
                const uint32_t neighbours[3] = {
                    c + 1 < keys.size() && keys[c + 1] == keys[c] + 1 ? static_cast<uint32_t>(c + 1) : kNone,
                    next_y.find(keys[c] + grid_.stride[1]), next_z.find(keys[c] + grid_.stride[2])};
                for (uint32_t axis = 0; axis < 3; ++axis) {
                    const uint32_t other = neighbours[axis];
                    if (other != kNone && (values[other] < 0) != inside)
                        out.push_back(Edge{static_cast<uint32_t>(c), other, axis});
                }
            }
        });
        std::vector<Edge> edges = concatenate(pieces);

        corner_edges_.assign(3 * keys.size(), kNone);
        for (size_t e = 0; e < edges.size(); ++e) corner_edges_[3 * size_t(edges[e].c0) + edges[e].axis] = uint32_t(e);

        std::vector<uint64_t> edge_keys(edges.size());
        for (size_t e = 0; e < edges.size(); ++e) edge_keys[e] = keys[edges[e].c0] * 3 + edges[e].axis;
        std::vector<Crossing> crossings(edges.size());
        std::vector<uint32_t> todo;
        size_t old = 0;
        for (size_t e = 0; e < edges.size(); ++e) {
            while (old < edge_keys_.size() && edge_keys_[old] < edge_keys[e]) ++old;
            if (old < edge_keys_.size() && edge_keys_[old] == edge_keys[e]) crossings[e] = crossings_[old];
            else todo.push_back(static_cast<uint32_t>(e));
        }
        // Refine in Morton order of the edges' lower corners: batches of
        // edges along key order are long strips that defeat culling.
        if (s_.spatial_order) {
            std::vector<uint64_t> codes(todo.size());
            for (size_t i = 0; i < todo.size(); ++i) {
                uint64_t c[3];
                grid_.coords(keys[edges[todo[i]].c0], c);
                codes[i] = morton(c);
            }
            radix_sort(codes, &todo, morton_bits_);
        }
        const BatchField gradient_field = field_.batch();
        parallel_chunks(chunks_of(todo.size(), kChunk / 4), s_.threads, [&](size_t chunk) {
            const size_t end = std::min(todo.size(), (chunk + 1) * (kChunk / 4));
            for (size_t first = chunk * (kChunk / 4); first < end; first += kBatch)
                refine(edges, todo.data() + first, static_cast<int>(std::min<size_t>(kBatch, end - first)),
                       gradient_field, crossings);
        });
        edges_.swap(edges);
        edge_keys_.swap(edge_keys);
        crossings_.swap(crossings);
    }

    // Crossing refinement along each edge, then the gradient normal. Steps
    // alternate between Illinois false position (fast on smooth surfaces) and
    // a Lipschitz step from the bracket end with the larger |f|: the root is
    // at least |f| / L from that end, which pins crossings at once where the
    // field is flat along the edge near a sharp corner and false position
    // creeps. Halved Illinois values only shorten those steps.
    void refine(const std::vector<Edge> &edges, const uint32_t *ids, int n, const BatchField &gradient_field,
                std::vector<Crossing> &crossings) const {
        double a[kBatch], fa[kBatch], b[kBatch], fb[kBatch], t[kBatch], ft[kBatch];
        double base[3][kBatch], x[kBatch], y[kBatch], z[kBatch];
        int side[kBatch];
        double *coords[3] = {x, y, z};
        for (int i = 0; i < n; ++i) {
            const Edge &edge = edges[ids[i]];
            uint64_t c[3];
            grid_.coords(corner_keys_[edge.c0], c);
            for (int axis = 0; axis < 3; ++axis) base[axis][i] = grid_.at(axis, c[axis]);
            a[i] = 0;
            fa[i] = corner_values_[edge.c0];
            b[i] = 1;
            fb[i] = corner_values_[edge.c1];
            side[i] = 0;
        }
        auto position = [&](int i) {
            const double fraction = (a[i] * fb[i] - b[i] * fa[i]) / (fb[i] - fa[i]);
            return fraction >= a[i] && fraction <= b[i] ? fraction : 0.5 * (a[i] + b[i]);
        };
        auto place = [&](const double *along) {
            for (int axis = 0; axis < 3; ++axis) std::copy(base[axis], base[axis] + n, coords[axis]);
            for (int i = 0; i < n; ++i) coords[edges[ids[i]].axis][i] += along[i] * grid_.h;
        };
        const double reach = 1 / (s_.lipschitz * grid_.h);  // edge fraction per unit of |f|
        for (int iteration = 0; iteration < s_.refine_steps; ++iteration) {
            const bool lipschitz_step = iteration % 2 == 0;
            for (int i = 0; i < n; ++i) {
                t[i] = position(i);
                if (!lipschitz_step) continue;
                const double step = std::fabs(fa[i]) >= std::fabs(fb[i]) ? a[i] + std::fabs(fa[i]) * reach
                                                                          : b[i] - std::fabs(fb[i]) * reach;
                if (step > a[i] && step < b[i]) t[i] = step;
            }
            place(t);
            field_.evaluate(n, x, y, z, ft);
            for (int i = 0; i < n; ++i) {
                if ((ft[i] < 0) == (fb[i] < 0)) {  // root in [a, t]
                    b[i] = t[i];
                    fb[i] = ft[i];
                    if (!lipschitz_step && side[i] == -1) fa[i] *= 0.5;
                    side[i] = -1;
                } else {  // root in [t, b]
                    a[i] = t[i];
                    fa[i] = ft[i];
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
            Crossing &crossing = crossings[ids[i]];
            crossing.p[0] = x[i];
            crossing.p[1] = y[i];
            crossing.p[2] = z[i];
            for (int axis = 0; axis < 3; ++axis) crossing.n[axis] = normal[axis][i];
            if (normal[0][i] == 0 && normal[1][i] == 0 && normal[2][i] == 0) {
                // Vanishing gradient: use the edge direction, from inside to outside.
                const Edge &edge = edges[ids[i]];
                crossing.n[edge.axis] = corner_values_[edge.c0] < 0 ? 1 : -1;
            }
        }
    }

    // The corner indices and values of candidate cell `key` (local corner
    // index dx + 2 dy + 4 dz), found through monotone row cursors.
    void cell_corners(Cursor rows[4], uint64_t key, uint32_t local[8], double value[8]) const {
        const uint64_t s1 = grid_.stride[1], s2 = grid_.stride[2];
        const uint64_t row_offsets[4] = {0, s1, s2, s1 + s2};  // corners (0, j, k); (1, j, k) follow directly
        for (int r = 0; r < 4; ++r) {
            local[2 * r] = rows[r].find(key + row_offsets[r]);
            if (local[2 * r] == kNone) throw Error("internal error: a candidate cell corner is missing");
            local[2 * r + 1] = local[2 * r] + 1;
        }
        for (int c = 0; c < 8; ++c) value[c] = corner_values_[local[c]];
    }

    // Key of face `face` (2 axis + side) of the cell with minimum corner `key`:
    // its minimum corner's key times three plus its axis, shared by both cells.
    uint64_t face_key(uint64_t key, int face) const {
        const int axis = face / 2, side = face % 2;
        return (key + (side ? grid_.stride[axis] : 0)) * 3 + static_cast<uint64_t>(axis);
    }

    // Field values at the centres of the ambiguous faces of active cells
    // (step 5); values known from an earlier repair pass are reused.
    void build_faces() {
        const CellTopology &t = topology();
        const uint64_t s1 = grid_.stride[1], s2 = grid_.stride[2];
        const size_t parts = chunks_of(cells_.size(), kChunk);
        std::vector<std::vector<uint64_t>> pieces(parts);
        parallel_chunks(parts, s_.threads, [&](size_t part) {
            const size_t begin = part * kChunk, end = std::min(cells_.size(), begin + kChunk);
            Cursor rows[4] = {Cursor(corner_keys_, cells_[begin]), Cursor(corner_keys_, cells_[begin] + s1),
                              Cursor(corner_keys_, cells_[begin] + s2), Cursor(corner_keys_, cells_[begin] + s1 + s2)};
            for (size_t i = begin; i < end; ++i) {
                uint32_t local[8];
                double value[8];
                cell_corners(rows, cells_[i], local, value);
                bool inside[8];
                for (int c = 0; c < 8; ++c) inside[c] = value[c] < 0;
                for (int face = 0; face < 6; ++face)
                    if (ambiguous_face(inside, t.face_corners[face])) pieces[part].push_back(face_key(cells_[i], face));
            }
        });
        std::vector<uint64_t> faces = concatenate(pieces);
        std::sort(faces.begin(), faces.end());
        faces.erase(std::unique(faces.begin(), faces.end()), faces.end());
        std::vector<double> values(faces.size());
        std::vector<uint32_t> todo;
        size_t old = 0;
        for (size_t i = 0; i < faces.size(); ++i) {
            while (old < face_keys_.size() && face_keys_[old] < faces[i]) ++old;
            if (old < face_keys_.size() && face_keys_[old] == faces[i]) values[i] = face_values_[old];
            else todo.push_back(static_cast<uint32_t>(i));
        }
        parallel_chunks(chunks_of(todo.size(), kChunk), s_.threads, [&](size_t chunk) {
            double x[kBatch], y[kBatch], z[kBatch], v[kBatch];
            const size_t end = std::min(todo.size(), (chunk + 1) * kChunk);
            for (size_t first = chunk * kChunk; first < end; first += kBatch) {
                const int n = static_cast<int>(std::min<size_t>(kBatch, end - first));
                for (int i = 0; i < n; ++i) {
                    const uint64_t key = faces[todo[first + i]];
                    const int axis = static_cast<int>(key % 3);
                    uint64_t c[3];
                    grid_.coords(key / 3, c);
                    double p[3];
                    for (int a = 0; a < 3; ++a) p[a] = grid_.at(a, c[a]) + (a == axis ? 0.0 : 0.5 * grid_.h);
                    x[i] = p[0];
                    y[i] = p[1];
                    z[i] = p[2];
                }
                field_.evaluate(n, x, y, z, v);
                for (int i = 0; i < n; ++i) values[todo[first + i]] = v[i];
            }
        });
        face_keys_.swap(faces);
        face_values_.swap(values);
    }

    double face_value(uint64_t key) const {
        const auto found = std::lower_bound(face_keys_.begin(), face_keys_.end(), key);
        if (found == face_keys_.end() || *found != key) throw Error("internal error: an ambiguous face is missing");
        return face_values_[static_cast<size_t>(found - face_keys_.begin())];
    }

    // QEF vertices of the active candidate cells (step 5): one per surface
    // component of the cell, so features thinner than a cell stay manifold.
    void build_vertices() {
        const std::vector<uint64_t> &keys = corner_keys_;
        const CellTopology &t = topology();
        const uint64_t s1 = grid_.stride[1], s2 = grid_.stride[2];
        const size_t parts = chunks_of(cells_.size(), kChunk);
        std::vector<std::vector<uint32_t>> owners(parts), tables(parts);
        std::vector<std::vector<uint8_t>> counts(parts);
        // Per vertex: its QEF position, its cell's minimum corner and the sum
        // of its component's crossing normals.
        std::vector<std::vector<double>> points(parts), corners(parts), guides(parts);
        parallel_chunks(parts, s_.threads, [&](size_t part) {
            const size_t begin = part * kChunk, end = std::min(cells_.size(), begin + kChunk);
            Cursor rows[4] = {Cursor(keys, cells_[begin]), Cursor(keys, cells_[begin] + s1),
                              Cursor(keys, cells_[begin] + s2), Cursor(keys, cells_[begin] + s1 + s2)};
            for (size_t i = begin; i < end; ++i) {
                const uint64_t key = cells_[i];
                uint32_t local[8];  // corner indices by local corner index
                double value[8];
                cell_corners(rows, key, local, value);
                int inside = 0;
                bool signs[8];
                for (int c = 0; c < 8; ++c) inside += signs[c] = value[c] < 0;
                if (inside == 0 || inside == 8) continue;
                double centre[6];
                for (int face = 0; face < 6; ++face)
                    centre[face] = ambiguous_face(signs, t.face_corners[face]) ? face_value(face_key(key, face)) : 0.0;
                uint8_t component[12];
                const int components = cell_components(value, centre, component);
                uint64_t c[3];
                grid_.coords(key, c);
                const double lo[3] = {grid_.at(0, c[0]), grid_.at(1, c[1]), grid_.at(2, c[2])};
                uint32_t table = 0;
                for (int j = 0; j < components; ++j) {
                    double p[12][3], nrm[12][3];
                    int m = 0;
                    for (int e = 0; e < 12; ++e) {
                        const int start = t.edge_start[e];
                        if ((value[start] < 0) == (value[start + (1 << t.edge_axis[e])] < 0) || component[e] != j)
                            continue;
                        const uint32_t edge = corner_edges_[3 * size_t(local[start]) + t.edge_axis[e]];
                        std::copy(crossings_[edge].p, crossings_[edge].p + 3, p[m]);
                        std::copy(crossings_[edge].n, crossings_[edge].n + 3, nrm[m]);
                        table |= static_cast<uint32_t>(j) << (2 * e);
                        ++m;
                    }
                    double vertex[3], guide[3] = {0, 0, 0};
                    solve_qef(m, p, nrm, lo, grid_.h, vertex);
                    for (int k = 0; k < m; ++k)
                        for (int a = 0; a < 3; ++a) guide[a] += nrm[k][a];
                    points[part].insert(points[part].end(), vertex, vertex + 3);
                    corners[part].insert(corners[part].end(), lo, lo + 3);
                    guides[part].insert(guides[part].end(), guide, guide + 3);
                }
                owners[part].push_back(local[0]);
                counts[part].push_back(static_cast<uint8_t>(components));
                tables[part].push_back(table);
            }
        });
        vertex_positions_ = concatenate(points);
        project_vertices(concatenate(corners), concatenate(guides));
        const std::vector<uint32_t> owner = concatenate(owners), table = concatenate(tables);
        const std::vector<uint8_t> count = concatenate(counts);
        active_cells_ = owner.size();
        cell_vertex_.assign(keys.size(), kNone);
        cell_table_.assign(keys.size(), 0);
        uint32_t next = 0;
        for (size_t i = 0; i < owner.size(); ++i) {
            cell_vertex_[owner[i]] = next;
            cell_table_[owner[i]] = table[i];
            next += count[i];
        }
    }

    // Moves the QEF vertices onto the surface (step 5): Newton steps along the
    // field gradient. Tangent planes meet on a sharp edge, so its QEF vertex
    // already lies on the surface and stays, but across a blend narrower than
    // a cell they meet off it: above a rounded ridge or inside it, depending
    // on which grid edges the blend crosses, so its vertices alternate into
    // dents along the ridge. Such a ridge can also bulge slightly past the
    // cell whose crossings found it, so a vertex may move up to half a cell
    // beyond its cell. A step is taken only while the gradient faces the
    // component's crossing normals, so a vertex never moves to the opposite
    // side of a thin wall, and kept only if it reduces |f|.
    void project_vertices(const std::vector<double> &cell_lo, const std::vector<double> &guide) {
        constexpr int kSamples = 3;  // field samples per vertex: at most two steps
        constexpr int kPoints = kBatch / 4;
        static const double kTetra[4][3] = {{1, -1, -1}, {-1, -1, 1}, {-1, 1, -1}, {1, 1, 1}};
        const size_t count = vertex_positions_.size() / 3;
        const double h = grid_.h, eps = s_.gradient_step, settled = 1e-9 * h;
        std::vector<uint32_t> order;
        if (s_.spatial_order) {
            order = morton_order(count, s_.threads, [&](size_t v, double *xyz) {
                std::copy(&vertex_positions_[3 * v], &vertex_positions_[3 * v] + 3, xyz);
            });
        } else {
            order.resize(count);
            for (size_t v = 0; v < count; ++v) order[v] = static_cast<uint32_t>(v);
        }
        parallel_chunks(chunks_of(count, kChunk), s_.threads, [&](size_t chunk) {
            double x[kPoints][3], best[kPoints][3], best_f[kPoints];
            double sx[kBatch], sy[kBatch], sz[kBatch], values[kBatch];
            int active[kPoints];
            const size_t end = std::min(count, (chunk + 1) * kChunk);
            for (size_t first = chunk * kChunk; first < end; first += kPoints) {
                const int n = static_cast<int>(std::min<size_t>(kPoints, end - first));
                int live = n;
                for (int i = 0; i < n; ++i) {
                    std::copy(&vertex_positions_[3 * size_t(order[first + i])],
                              &vertex_positions_[3 * size_t(order[first + i])] + 3, x[i]);
                    std::copy(x[i], x[i] + 3, best[i]);
                    best_f[i] = INFINITY;
                    active[i] = i;
                }
                for (int sample = 0; sample < kSamples && live > 0; ++sample) {
                    // The value (mean of the stencil) and gradient (tetrahedral differences).
                    for (int j = 0; j < live; ++j)
                        for (int k = 0; k < 4; ++k) {
                            sx[4 * j + k] = x[active[j]][0] + eps * kTetra[k][0];
                            sy[4 * j + k] = x[active[j]][1] + eps * kTetra[k][1];
                            sz[4 * j + k] = x[active[j]][2] + eps * kTetra[k][2];
                        }
                    field_.evaluate(4 * live, sx, sy, sz, values);
                    int next = 0;
                    for (int j = 0; j < live; ++j) {
                        const int i = active[j];
                        double f = 0, g[3] = {0, 0, 0};
                        for (int k = 0; k < 4; ++k) {
                            f += values[4 * j + k];
                            for (int a = 0; a < 3; ++a) g[a] += values[4 * j + k] * kTetra[k][a];
                        }
                        f *= 0.25;
                        if (!(std::fabs(f) < best_f[i])) continue;  // the last step did not help
                        std::copy(x[i], x[i] + 3, best[i]);
                        best_f[i] = std::fabs(f);
                        const double *towards = &guide[3 * size_t(order[first + i])];
                        const double gg = g[0] * g[0] + g[1] * g[1] + g[2] * g[2];
                        if (sample + 1 == kSamples || best_f[i] <= settled || !(gg > 0) || !std::isfinite(gg) ||
                            g[0] * towards[0] + g[1] * towards[1] + g[2] * towards[2] <= 0)
                            continue;
                        const double step = f * 4 * eps / gg;  // the gradient is g / (4 eps)
                        const double *lo = &cell_lo[3 * size_t(order[first + i])];
                        for (int a = 0; a < 3; ++a)
                            x[i][a] = std::fmin(std::fmax(x[i][a] - step * g[a], lo[a] - 0.5 * h), lo[a] + 1.5 * h);
                        active[next++] = i;
                    }
                    live = next;
                }
                for (int i = 0; i < n; ++i)
                    std::copy(best[i], best[i] + 3, &vertex_positions_[3 * size_t(order[first + i])]);
            }
        });
    }

    // Minimises sum (n_i . (x - p_i))^2 about the mass point with a truncated
    // pseudo-inverse (eigenvalues below 5% of the largest dropped), clamped to the cell.
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
            if (!(values[k] > 0.05 * largest) || !(values[k] > 0)) continue;
            const double projection =
                (vectors[0][k] * atb[0] + vectors[1][k] * atb[1] + vectors[2][k] * atb[2]) / values[k];
            for (int a = 0; a < 3; ++a) x[a] += vectors[a][k] * projection;
        }
        for (int a = 0; a < 3; ++a) out[a] = std::fmin(std::fmax(mass[a] + x[a], lo[a]), lo[a] + h);
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

    // Quads around every sign-change edge, oriented outward and split along
    // the shorter diagonal (step 6). Returns the keys of missing cells.
    std::vector<uint64_t> build_triangles() {
        const std::vector<uint64_t> &keys = corner_keys_;
        const uint64_t s1 = grid_.stride[1], s2 = grid_.stride[2];
        // Neighbour cell offsets, indexed by axis: e_b, e_b + e_d, e_d with b, d the other axes.
        const uint64_t offsets[6] = {1, s1, s2, s1 + s2, 1 + s2, 1 + s1};
        static const int kCursorOf[3][3] = {{1, 3, 2}, {2, 4, 0}, {0, 5, 1}};
        // Local index of the quad's grid edge within each of its four cells.
        int quad_edges[3][4];
        for (int axis = 0; axis < 3; ++axis) {
            static const int kOffsets[4][2] = {{0, 0}, {1, 0}, {1, 1}, {0, 1}};  // along e_b, e_d
            for (int q = 0; q < 4; ++q) {
                int offset[3] = {0, 0, 0};
                offset[(axis + 1) % 3] = kOffsets[q][0];
                offset[(axis + 2) % 3] = kOffsets[q][1];
                quad_edges[axis][q] = local_edge(axis, offset);
            }
        }
        const size_t parts = chunks_of(edges_.size(), 2 * kChunk);
        std::vector<std::vector<uint32_t>> tris(parts);
        std::vector<std::vector<uint64_t>> missing(parts);
        parallel_chunks(parts, s_.threads, [&](size_t part) {
            const size_t begin = part * 2 * kChunk, end = std::min(edges_.size(), begin + 2 * kChunk);
            const uint64_t first = keys[edges_[begin].c0];
            Cursor cursors[6] = {Cursor(keys, saturating_sub(first, offsets[0])),
                                 Cursor(keys, saturating_sub(first, offsets[1])),
                                 Cursor(keys, saturating_sub(first, offsets[2])),
                                 Cursor(keys, saturating_sub(first, offsets[3])),
                                 Cursor(keys, saturating_sub(first, offsets[4])),
                                 Cursor(keys, saturating_sub(first, offsets[5]))};
            for (size_t e = begin; e < end; ++e) {
                const Edge &edge = edges_[e];
                const uint64_t key = keys[edge.c0];
                uint64_t c[3];
                grid_.coords(key, c);
                const int b = (edge.axis + 1) % 3, d = (edge.axis + 2) % 3;
                if (c[b] == 0 || c[d] == 0) continue;  // cannot happen: the padding is outside
                // Cells in counter-clockwise order about +axis.
                const uint64_t cell_keys[4] = {key, key - offsets[kCursorOf[edge.axis][0]],
                                               key - offsets[kCursorOf[edge.axis][1]],
                                               key - offsets[kCursorOf[edge.axis][2]]};
                uint32_t vertex[4];
                bool complete = true;
                for (int q = 0; q < 4; ++q) {
                    const uint32_t corner = q == 0 ? edge.c0 : cursors[kCursorOf[edge.axis][q - 1]].find(cell_keys[q]);
                    vertex[q] = corner == kNone ? kNone : cell_vertex_[corner];
                    if (vertex[q] == kNone) {
                        missing[part].push_back(cell_keys[q]);
                        complete = false;
                        continue;
                    }
                    // The vertex of the cell's surface component that holds this edge.
                    vertex[q] += (cell_table_[corner] >> (2 * quad_edges[edge.axis][q])) & 3;
                }
                if (!complete) continue;
                if (corner_values_[edge.c0] >= 0) std::swap(vertex[1], vertex[3]);  // outward is -axis
                const double *p[4];
                for (int q = 0; q < 4; ++q) p[q] = &vertex_positions_[3 * size_t(vertex[q])];
                auto distance2 = [](const double *u, const double *v) {
                    const double dx = u[0] - v[0], dy = u[1] - v[1], dz = u[2] - v[2];
                    return dx * dx + dy * dy + dz * dz;
                };
                // Whether triangle (a, b, c) faces against the crossing's normal.
                const double *normal = crossings_[e].n;
                auto folds = [&](int a, int b, int c) {
                    double u[3], v[3];
                    for (int axis = 0; axis < 3; ++axis) {
                        u[axis] = p[b][axis] - p[a][axis];
                        v[axis] = p[c][axis] - p[a][axis];
                    }
                    return (u[1] * v[2] - u[2] * v[1]) * normal[0] + (u[2] * v[0] - u[0] * v[2]) * normal[1] +
                               (u[0] * v[1] - u[1] * v[0]) * normal[2] < 0;
                };
                // The shorter diagonal, unless only the other one keeps both
                // triangles facing outward (sharp and twisted edges).
                bool first_diagonal = distance2(p[0], p[2]) <= distance2(p[1], p[3]);
                const bool first_folds = folds(0, 1, 2) || folds(0, 2, 3);
                const bool second_folds = folds(0, 1, 3) || folds(1, 2, 3);
                if (first_folds != second_folds) first_diagonal = second_folds;
                std::vector<uint32_t> &out = tris[part];
                if (first_diagonal) {
                    out.insert(out.end(), {vertex[0], vertex[1], vertex[2], vertex[0], vertex[2], vertex[3]});
                } else {
                    out.insert(out.end(), {vertex[0], vertex[1], vertex[3], vertex[1], vertex[2], vertex[3]});
                }
            }
        });
        triangles_ = concatenate(tris);
        std::vector<uint64_t> cells = concatenate(missing);
        std::sort(cells.begin(), cells.end());
        cells.erase(std::unique(cells.begin(), cells.end()), cells.end());
        return cells;
    }

    // Keeps the vertices that triangles reference, in cell order, after
    // releasing the grid data.
    Surface finish() {
        for (auto *keys : {&cells_, &corner_keys_, &edge_keys_, &face_keys_}) std::vector<uint64_t>().swap(*keys);
        std::vector<double>().swap(face_values_);
        for (auto *table : {&corner_edges_, &cell_vertex_, &cell_table_}) std::vector<uint32_t>().swap(*table);
        std::vector<double>().swap(corner_values_);
        std::vector<Edge>().swap(edges_);
        std::vector<Crossing>().swap(crossings_);
        Surface surface;
        surface.active_cells = active_cells_;
        std::vector<uint32_t> remap(vertex_positions_.size() / 3, kNone);
        for (uint32_t v : triangles_) remap[v] = 0;
        uint32_t next = 0;
        for (size_t v = 0; v < remap.size(); ++v) {
            if (remap[v] == kNone) continue;
            remap[v] = next++;
            surface.positions.insert(surface.positions.end(), &vertex_positions_[3 * v], &vertex_positions_[3 * v] + 3);
        }
        surface.triangles.resize(triangles_.size());
        for (size_t i = 0; i < triangles_.size(); ++i) surface.triangles[i] = remap[triangles_[i]];
        return surface;
    }

    const MeshField &field_;
    const MeshSettings &s_;
    Grid grid_;
    int key_bits_ = 0, morton_bits_ = 0;

    std::vector<uint64_t> cells_;          // candidate cells (sorted keys)
    std::vector<uint64_t> corner_keys_;    // their unique corners (sorted)
    std::vector<double> corner_values_;
    std::vector<uint32_t> corner_edges_;   // 3 per corner: edge index along each axis, or kNone
    std::vector<Edge> edges_;              // sign-change edges, sorted by (corner key, axis)
    std::vector<uint64_t> edge_keys_;      // corner key * 3 + axis
    std::vector<Crossing> crossings_;
    std::vector<uint64_t> face_keys_;      // ambiguous faces of active cells (sorted, see face_key)
    std::vector<double> face_values_;      // the field at their centres
    std::vector<uint32_t> cell_vertex_;    // per corner: first vertex of the cell it is the minimum corner of
    std::vector<uint32_t> cell_table_;     // per corner: 2-bit component (vertex offset) per local edge
    std::vector<double> vertex_positions_;
    uint64_t active_cells_ = 0;
    std::vector<uint32_t> triangles_;
};

}  // namespace

Surface extract_surface(const MeshField &field, const MeshSettings &settings) {
    return Extractor(field, settings).run();
}

}  // namespace rsdf
