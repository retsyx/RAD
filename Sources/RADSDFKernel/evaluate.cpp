// Batched interpretation of the node DAG.
//
// A batch of at most kBatch points flows down the graph on an explicit stack
// of frames (graphs may nest far deeper than a thread's call stack allows).
// Each frame owns structure-of-arrays buffers taken from a per-thread pool in
// LIFO order; the pool keeps its blocks between batches, so steady-state
// evaluation never allocates.
#include "kernel.h"

#include <algorithm>
#include <cstdlib>
#include <limits>
#include <new>

#include <sys/mman.h>

namespace rsdf {
namespace {

struct Frame {
    uint32_t node;
    uint32_t stage;            // children dispatched so far
    int n;
    const double *x, *y, *z;   // the node's points (z is null for 2D nodes)
    double *out;               // the node's values
    double *tmp;               // a child's values awaiting combination
    size_t mark;               // pool position to release when the frame ends
    double box_lo[3], box_hi[3];
    double feature_scale;     // model units per local unit (preview detail probes only)
    double current;            // largest value (union) or interior depth (difference)
    size_t candidate_mark;     // Scratch::candidates size to restore when the frame ends
    size_t next_candidate, candidates_end;  // this frame's candidate children (ChildTree nodes)
    bool has_box, has_current, pending, has_candidates;
};

class Scratch {
public:
    ~Scratch() {
        for (double *slab : slabs_) munmap(slab, kSlabBytes);
    }

    double *take() {
        if (top_ == blocks_.size()) grow();
        return blocks_[top_++];
    }
    size_t top() const { return top_; }
    void release(size_t mark) { top_ = mark; }
    double *block(size_t index) const { return blocks_[index]; }  // a buffer taken earlier (index < top())

    void reset() {
        top_ = 0;
        frames.clear();
        candidates.clear();
    }

    // Returns memory beyond a modest working set after an unusually deep
    // graph. Buffers live in mapped slabs, so unmapping them gives the memory
    // back to the system at once.
    void trim() {
        constexpr size_t kKeepSlabs = 32, kKeepFrames = 65536;  // 8 MB of buffers, ~8 MB of frames
        if (slabs_.size() > kKeepSlabs) {
            for (size_t i = kKeepSlabs; i < slabs_.size(); ++i) munmap(slabs_[i], kSlabBytes);
            slabs_.resize(kKeepSlabs);
            slabs_.shrink_to_fit();
            blocks_.resize(kKeepSlabs * kSlabBlocks);
            blocks_.shrink_to_fit();
        }
        if (frames.capacity() > kKeepFrames) {
            frames.clear();
            frames.shrink_to_fit();
        }
        if (candidates.capacity() > 16 * kKeepFrames) {
            candidates.clear();
            candidates.shrink_to_fit();
        }
    }

    // Optional preview probe. Track the narrowest nearby input facet at
    // each query while leaving CSG distances and branch choices unchanged.
    double *facet_sizes = nullptr;
    double facet_radius = 0;
    std::vector<Frame> frames;
    std::vector<uint32_t> candidates;  // child positions per boolean frame, stacked like the frames

private:
    static constexpr size_t kSlabBlocks = 128;
    static constexpr size_t kSlabBytes = kSlabBlocks * kBatch * sizeof(double);  // 256 KB

    void grow() {
        slabs_.reserve(slabs_.size() + 1);
        blocks_.reserve(blocks_.size() + kSlabBlocks);
        void *memory = mmap(nullptr, kSlabBytes, PROT_READ | PROT_WRITE, MAP_PRIVATE | MAP_ANON, -1, 0);
        if (memory == MAP_FAILED) throw std::bad_alloc();
        slabs_.push_back(static_cast<double *>(memory));
        for (size_t i = 0; i < kSlabBlocks; ++i) blocks_.push_back(slabs_.back() + i * kBatch);
    }

    std::vector<double *> slabs_;
    std::vector<double *> blocks_;  // kBatch doubles each, inside the slabs
    size_t top_ = 0;
};

thread_local Scratch scratch;

void push(Scratch &s, uint32_t node, int n, const double *x, const double *y, const double *z, double *out, double scale = 1) {
    Frame frame;
    frame.node = node;
    frame.stage = 0;
    frame.n = n;
    frame.x = x;
    frame.y = y;
    frame.z = z;
    frame.out = out;
    frame.tmp = nullptr;
    frame.mark = s.top();
    frame.current = 0;
    frame.feature_scale = s.facet_sizes && !s.frames.empty() ? s.frames.back().feature_scale * scale : 1;
    frame.candidate_mark = s.candidates.size();
    frame.next_candidate = frame.candidates_end = 0;
    frame.has_box = frame.has_current = frame.pending = frame.has_candidates = false;
    s.frames.push_back(frame);
}

inline double smin(double a, double b, double k) {
    const double h = std::fmax(k - std::fabs(a - b), 0.0) / k;
    return std::fmin(a, b) - h * h * k * 0.25;
}

inline double smax(double a, double b, double k) {
    const double h = std::fmax(k - std::fabs(a - b), 0.0) / k;
    return std::fmax(a, b) + h * h * k * 0.25;
}

// Folds a child's values into the running result of a boolean node and
// returns the culling level of the result (see cull_threshold): the largest
// value of a union, the largest interior depth (at least 0) of a difference.
double combine(uint32_t op, double k, int n, double *out, const double *child) {
    const bool smooth = k > 0;
    double level = -INFINITY;
    switch (op) {
    case OP_UNION: case OP_UNION2:
        if (smooth) for (int i = 0; i < n; ++i) level = std::fmax(level, out[i] = smin(out[i], child[i], k));
        else for (int i = 0; i < n; ++i) level = std::fmax(level, out[i] = std::fmin(out[i], child[i]));
        break;
    case OP_INTERSECTION: case OP_INTERSECTION2:
        if (smooth) for (int i = 0; i < n; ++i) out[i] = smax(out[i], child[i], k);
        else for (int i = 0; i < n; ++i) out[i] = std::fmax(out[i], child[i]);
        break;
    default:  // DIFFERENCE
        level = 0;
        if (smooth) for (int i = 0; i < n; ++i) level = std::fmax(level, -(out[i] = smax(out[i], -child[i], k)));
        else for (int i = 0; i < n; ++i) level = std::fmax(level, -(out[i] = std::fmax(out[i], -child[i])));
        break;
    }
    return level;
}

// Whole-batch culling for unions and differences (see SDF-DESIGN.md): at
// every point of the batch a child is at least m D, with D the distance from
// the batch box to the child's box and m the child's distance ratio. When
// m D is at least the batch's largest value (union) or interior depth
// (difference) plus the blend radius, the child cannot change any value. The
// boxes must also be strictly separated (D > 0): a batch overlapping a
// cutter's box may lie inside it. Folding children only lowers a union's
// values and raises a difference's, so the threshold only tightens while a
// node's children are folded.
double cull_threshold(Frame &f, const Node &node, double k) {
    if (!f.has_box) {
        const double *coords[3] = {f.x, f.y, f.z};
        for (uint32_t a = 0; a < node.dim; ++a) {
            double lo = coords[a][0], hi = coords[a][0];
            for (int i = 1; i < f.n; ++i) {
                lo = std::fmin(lo, coords[a][i]);
                hi = std::fmax(hi, coords[a][i]);
            }
            f.box_lo[a] = lo;
            f.box_hi[a] = hi;
        }
        f.has_box = true;
    }
    if (!f.has_current) {
        double current = -INFINITY;
        if (node.op == OP_UNION || node.op == OP_UNION2) {
            for (int i = 0; i < f.n; ++i) current = std::fmax(current, f.out[i]);
        } else {
            current = 0;
            for (int i = 0; i < f.n; ++i) current = std::fmax(current, -f.out[i]);
        }
        f.current = current;
        f.has_current = true;
    }
    return std::fmax(f.current + k, std::numeric_limits<double>::denorm_min());
}

bool culled(const rsdf_graph &graph, Frame &f, const Node &node, uint32_t child, double k) {
    const double threshold = cull_threshold(f, node, k);
    const Node &c = graph.nodes[child];
    return c.ratio * box_distance(f.box_lo, f.box_hi, c.lo, c.hi, static_cast<int>(node.dim)) >= threshold;
}

// Appends, in child order, the children of a large boolean that the current
// threshold does not cull; since the threshold only tightens, every other
// child would be culled when its turn came.
void collect_candidates(const rsdf_graph &graph, Frame &f, const Node &node, double k,
                        std::vector<uint32_t> &out) {
    const ChildTree &tree = graph.child_trees[node.child_tree - 1];
    const uint32_t *kids = graph.child_ids(node);
    const int dim = static_cast<int>(node.dim);
    const double threshold = cull_threshold(f, node, k);
    const size_t first = out.size();
    if (!(threshold < INFINITY)) {  // nothing can be culled
        for (uint32_t slot = 1; slot < node.child_count; ++slot) out.push_back(slot);
        return;
    }
    out.insert(out.end(), tree.always.begin(), tree.always.end());
    uint32_t stack[64];
    int top = 0;
    if (!tree.branches.empty()) stack[top++] = 0;
    while (top > 0) {
        const uint32_t index = stack[--top];
        const ChildTree::Branch &branch = tree.branches[index];
        if (!(branch.ratio * box_distance(f.box_lo, f.box_hi, branch.lo, branch.hi, dim) < threshold)) continue;
        if (branch.right != 0) {
            stack[top++] = branch.right;
            stack[top++] = index + 1;
            continue;
        }
        for (uint32_t i = branch.first; i < branch.first + branch.count; ++i) {
            const Node &c = graph.nodes[kids[tree.slots[i]]];
            if (c.ratio * box_distance(f.box_lo, f.box_hi, c.lo, c.hi, dim) < threshold) out.push_back(tree.slots[i]);
        }
    }
    std::sort(out.begin() + static_cast<std::ptrdiff_t>(first), out.end());
}

inline double clamp01(double t) { return std::fmin(std::fmax(t, 0.0), 1.0); }

// Distance of an extrusion-style slab given the profile value d and the
// signed slab distance w.
inline double slab(double d, double w) {
    const double od = std::fmax(d, 0.0), ow = std::fmax(w, 0.0);
    return std::fmin(std::fmax(d, w), 0.0) + std::sqrt(od * od + ow * ow);
}

// Evaluates primitives directly into f.out.
void eval_leaf(const rsdf_graph &graph, const Node &node, const double *p, int n, const double *x,
               const double *y, const double *z, double *out) {
    switch (node.op) {
    case OP_SPHERE:
        for (int i = 0; i < n; ++i) out[i] = std::sqrt(x[i] * x[i] + y[i] * y[i] + z[i] * z[i]) - p[0];
        break;
    case OP_BOX: {
        const double r = p[3], bx = p[0] - r, by = p[1] - r, bz = p[2] - r;
        for (int i = 0; i < n; ++i) {
            const double qx = std::fabs(x[i]) - bx, qy = std::fabs(y[i]) - by, qz = std::fabs(z[i]) - bz;
            const double ox = std::fmax(qx, 0.0), oy = std::fmax(qy, 0.0), oz = std::fmax(qz, 0.0);
            out[i] = std::sqrt(ox * ox + oy * oy + oz * oz) + std::fmin(std::fmax(qx, std::fmax(qy, qz)), 0.0) - r;
        }
        break;
    }
    case OP_CYLINDER:
        for (int i = 0; i < n; ++i)
            out[i] = slab(std::sqrt(x[i] * x[i] + y[i] * y[i]) - p[0], std::fabs(z[i]) - p[1]);
        break;
    case OP_CONE: {
        const double r1 = p[0], r2 = p[1], h = p[2];
        const double kx = r2 - r1, ky = 2 * h, inv = 1 / (kx * kx + ky * ky);
        for (int i = 0; i < n; ++i) {
            const double qx = std::sqrt(x[i] * x[i] + y[i] * y[i]), qy = z[i];
            const double cax = qx - std::fmin(qx, qy < 0 ? r1 : r2), cay = std::fabs(qy) - h;
            const double t = clamp01(((r2 - qx) * kx + (h - qy) * ky) * inv);
            const double cbx = qx - r2 + kx * t, cby = qy - h + ky * t;
            const double sign = (cbx < 0 && cay < 0) ? -1.0 : 1.0;
            out[i] = sign * std::sqrt(std::fmin(cax * cax + cay * cay, cbx * cbx + cby * cby));
        }
        break;
    }
    case OP_CAPSULE: {
        const double bax = p[3] - p[0], bay = p[4] - p[1], baz = p[5] - p[2], inv = node.c0;
        for (int i = 0; i < n; ++i) {
            const double px = x[i] - p[0], py = y[i] - p[1], pz = z[i] - p[2];
            const double t = clamp01((px * bax + py * bay + pz * baz) * inv);
            const double dx = px - t * bax, dy = py - t * bay, dz = pz - t * baz;
            out[i] = std::sqrt(dx * dx + dy * dy + dz * dz) - p[6];
        }
        break;
    }
    case OP_TORUS:
        for (int i = 0; i < n; ++i) {
            const double q = std::sqrt(x[i] * x[i] + y[i] * y[i]) - p[0];
            out[i] = std::sqrt(q * q + z[i] * z[i]) - p[1];
        }
        break;
    case OP_ELLIPSOID: {
        const double ix = 1 / p[0], iy = 1 / p[1], iz = 1 / p[2];
        for (int i = 0; i < n; ++i) {
            const double qx = x[i] * ix, qy = y[i] * iy, qz = z[i] * iz;
            out[i] = (std::sqrt(qx * qx + qy * qy + qz * qz) - 1) * node.c0;
        }
        break;
    }
    case OP_HALFSPACE:
        for (int i = 0; i < n; ++i) out[i] = p[0] * x[i] + p[1] * y[i] + p[2] * z[i] - p[3];
        break;
    case OP_MESH: eval_triangle_mesh(graph.meshes[node.mesh], n, x, y, z, out); break;
    case OP_CIRCLE:
        for (int i = 0; i < n; ++i) out[i] = std::sqrt(x[i] * x[i] + y[i] * y[i]) - p[0];
        break;
    case OP_RECT: {
        const double r = p[2], bx = p[0] - r, by = p[1] - r;
        for (int i = 0; i < n; ++i) {
            const double qx = std::fabs(x[i]) - bx, qy = std::fabs(y[i]) - by;
            const double ox = std::fmax(qx, 0.0), oy = std::fmax(qy, 0.0);
            out[i] = std::sqrt(ox * ox + oy * oy) + std::fmin(std::fmax(qx, qy), 0.0) - r;
        }
        break;
    }
    case OP_POLYGON: eval_polygon(graph.polygons[node.polygon], n, x, y, out); break;
    case OP_SEGMENT: {
        const double bax = p[2] - p[0], bay = p[3] - p[1], inv = node.c0;
        for (int i = 0; i < n; ++i) {
            const double px = x[i] - p[0], py = y[i] - p[1];
            const double t = clamp01((px * bax + py * bay) * inv);
            const double dx = px - t * bax, dy = py - t * bay;
            out[i] = std::sqrt(dx * dx + dy * dy) - p[4];
        }
        break;
    }
    case OP_ELLIPSE: {
        const double ix = 1 / p[0], iy = 1 / p[1];
        for (int i = 0; i < n; ++i) {
            const double qx = x[i] * ix, qy = y[i] * iy;
            out[i] = (std::sqrt(qx * qx + qy * qy) - 1) * node.c0;
        }
        break;
    }
    case OP_HALFPLANE:
        for (int i = 0; i < n; ++i) out[i] = p[0] * x[i] + p[1] * y[i] - p[2];
        break;
    }
}

// Writes the child points of a point-mapping node into fresh buffers and
// returns them through cx, cy, cz.
void map_points(Scratch &s, const Node &node, const double *p, int n, const double *x, const double *y,
                const double *z, const double *&cx, const double *&cy, const double *&cz) {
    double *bx = s.take(), *by = s.take();
    double *bz = nullptr;
    cx = bx;
    cy = by;
    cz = z;
    switch (node.op) {
    case OP_TRANSFORM: {  // local = R^T (p - t) / s
        bz = s.take();
        cz = bz;
        const double inv = 1 / p[12];
        for (int i = 0; i < n; ++i) {
            const double dx = x[i] - p[9], dy = y[i] - p[10], dz = z[i] - p[11];
            bx[i] = (p[0] * dx + p[3] * dy + p[6] * dz) * inv;
            by[i] = (p[1] * dx + p[4] * dy + p[7] * dz) * inv;
            bz[i] = (p[2] * dx + p[5] * dy + p[8] * dz) * inv;
        }
        break;
    }
    case OP_STRETCH: {
        bz = s.take();
        cz = bz;
        const double ix = 1 / p[0], iy = 1 / p[1], iz = 1 / p[2];
        for (int i = 0; i < n; ++i) {
            bx[i] = x[i] * ix;
            by[i] = y[i] * iy;
            bz[i] = z[i] * iz;
        }
        break;
    }
    case OP_MIRROR: {
        bz = s.take();
        cz = bz;
        for (int i = 0; i < n; ++i) {
            const double m = 2 * (p[0] * x[i] + p[1] * y[i] + p[2] * z[i] - p[3]);
            bx[i] = x[i] - m * p[0];
            by[i] = y[i] - m * p[1];
            bz[i] = z[i] - m * p[2];
        }
        break;
    }
    case OP_TWIST:  // rotate by -rate * z about Z; z is unchanged
        for (int i = 0; i < n; ++i) {
            const double angle = -p[0] * z[i];
            const double c = std::cos(angle), sn = std::sin(angle);
            bx[i] = c * x[i] - sn * y[i];
            by[i] = sn * x[i] + c * y[i];
        }
        break;
    case OP_ELONGATE: {
        bz = s.take();
        cz = bz;
        for (int i = 0; i < n; ++i) {
            bx[i] = x[i] - std::fmin(std::fmax(x[i], -p[0]), p[0]);
            by[i] = y[i] - std::fmin(std::fmax(y[i], -p[1]), p[1]);
            bz[i] = z[i] - std::fmin(std::fmax(z[i], -p[2]), p[2]);
        }
        break;
    }
    case OP_REVOLVE:  // the 2D profile sees (radius - offset, z)
        cz = nullptr;
        for (int i = 0; i < n; ++i) {
            bx[i] = std::sqrt(x[i] * x[i] + y[i] * y[i]) - p[0];
            by[i] = z[i];
        }
        break;
    case OP_TRANSFORM2: {
        const double inv = 1 / p[6];
        for (int i = 0; i < n; ++i) {
            const double dx = x[i] - p[4], dy = y[i] - p[5];
            bx[i] = (p[0] * dx + p[2] * dy) * inv;
            by[i] = (p[1] * dx + p[3] * dy) * inv;
        }
        break;
    }
    case OP_STRETCH2: {
        const double ix = 1 / p[0], iy = 1 / p[1];
        for (int i = 0; i < n; ++i) {
            bx[i] = x[i] * ix;
            by[i] = y[i] * iy;
        }
        break;
    }
    case OP_MIRROR2:
        for (int i = 0; i < n; ++i) {
            const double m = 2 * (p[0] * x[i] + p[1] * y[i] - p[2]);
            bx[i] = x[i] - m * p[0];
            by[i] = y[i] - m * p[1];
        }
        break;
    }
}

// Applies a unary node to its child's values (already in out).
void finish_unary(const Node &node, const double *p, int n, const double *z, double *out) {
    switch (node.op) {
    case OP_TRANSFORM: for (int i = 0; i < n; ++i) out[i] *= p[12]; break;
    case OP_TRANSFORM2: for (int i = 0; i < n; ++i) out[i] *= p[6]; break;
    case OP_STRETCH: case OP_STRETCH2: case OP_TWIST:
        for (int i = 0; i < n; ++i) out[i] *= node.c0;
        break;
    case OP_OFFSET: case OP_OFFSET2: for (int i = 0; i < n; ++i) out[i] -= p[0]; break;
    case OP_SHELL: {
        const double t = p[0];
        if (p[1] == 0) for (int i = 0; i < n; ++i) out[i] = std::fmax(out[i], -out[i] - t);
        else if (p[1] == 1) for (int i = 0; i < n; ++i) out[i] = std::fmax(-out[i], out[i] - t);
        else for (int i = 0; i < n; ++i) out[i] = std::fabs(out[i]) - 0.5 * t;
        break;
    }
    case OP_OUTLINE2: for (int i = 0; i < n; ++i) out[i] = std::fabs(out[i]) - 0.5 * p[0]; break;
    case OP_EXTRUDE: case OP_TAPERED_EXTRUDE: {
        const double z0 = p[0], z1 = p[1], center = 0.5 * (z0 + z1), half = 0.5 * (z1 - z0);
        if (node.op == OP_EXTRUDE) {
            for (int i = 0; i < n; ++i) out[i] = slab(out[i], std::fabs(z[i] - center) - half);
        } else {
            const double inv = 1 / (z1 - z0), taper = p[2];
            for (int i = 0; i < n; ++i) {
                const double d = (out[i] - taper * clamp01((z[i] - z0) * inv)) * node.c0;
                out[i] = slab(d, std::fabs(z[i] - center) - half);
            }
        }
        break;
    }
    default: break;  // MIRROR, ELONGATE, REVOLVE and their 2D forms pass values through
    }
}

// |(a mod 2P) - P| with the floored remainder: fmod is exact and cannot
// overflow (a / 2P can for tiny periods), so values lie in [0, P].
inline double triwave(double a, double period) {
    const double twice = 2 * period;
    double rest = std::fmod(a, twice);
    if (rest < 0) rest += twice;
    return std::fabs(rest - period);
}

// One instruction of a field tape (not a shape read) over n points.
template <class Operand>
void run_instruction(const FieldTape::Instr &ins, const uint32_t *args, int n, const Operand &operand, double *out) {
    const double *a = ins.count ? operand(args[0]) : nullptr;
    const double *b = ins.count > 1 ? operand(args[1]) : nullptr;
    switch (ins.op) {
    case OP_CONST: for (int i = 0; i < n; ++i) out[i] = ins.p0; break;
    case OP_ADD: for (int i = 0; i < n; ++i) out[i] = a[i] + b[i]; break;
    case OP_SUB: for (int i = 0; i < n; ++i) out[i] = a[i] - b[i]; break;
    case OP_MUL: for (int i = 0; i < n; ++i) out[i] = a[i] * b[i]; break;
    case OP_DIV: for (int i = 0; i < n; ++i) out[i] = a[i] / b[i]; break;
    case OP_NEG: for (int i = 0; i < n; ++i) out[i] = -a[i]; break;
    case OP_ABS: for (int i = 0; i < n; ++i) out[i] = std::fabs(a[i]); break;
    case OP_SQUARE: for (int i = 0; i < n; ++i) out[i] = a[i] * a[i]; break;
    case OP_SQRT: for (int i = 0; i < n; ++i) out[i] = std::sqrt(a[i]); break;
    case OP_SIN: for (int i = 0; i < n; ++i) out[i] = std::sin(a[i]); break;
    case OP_COS: for (int i = 0; i < n; ++i) out[i] = std::cos(a[i]); break;
    case OP_EXP: for (int i = 0; i < n; ++i) out[i] = std::exp(a[i]); break;
    case OP_LOG: for (int i = 0; i < n; ++i) out[i] = std::log(a[i]); break;
    case OP_POWC: {
        const double e = ins.p0;
        if (e == 2) for (int i = 0; i < n; ++i) out[i] = a[i] * a[i];
        else for (int i = 0; i < n; ++i) out[i] = std::pow(a[i], e);
        break;
    }
    case OP_ATAN2: for (int i = 0; i < n; ++i) out[i] = std::atan2(a[i], b[i]); break;
    case OP_MIN: case OP_MAX: {
        const bool lowest = ins.op == OP_MIN;
        for (int i = 0; i < n; ++i) out[i] = a[i];
        for (uint32_t k = 1; k < ins.count; ++k) {
            const double *c = operand(args[k]);
            if (lowest) for (int i = 0; i < n; ++i) out[i] = std::fmin(out[i], c[i]);
            else for (int i = 0; i < n; ++i) out[i] = std::fmax(out[i], c[i]);
        }
        break;
    }
    case OP_CLAMP:
        for (int i = 0; i < n; ++i) out[i] = std::fmin(std::fmax(a[i], ins.p0), ins.p1);
        break;
    case OP_LENGTH: {
        for (int i = 0; i < n; ++i) out[i] = a[i] * a[i];
        for (uint32_t k = 1; k < ins.count; ++k) {
            const double *c = operand(args[k]);
            for (int i = 0; i < n; ++i) out[i] += c[i] * c[i];
        }
        for (int i = 0; i < n; ++i) out[i] = std::sqrt(out[i]);
        break;
    }
    case OP_TRIWAVE: for (int i = 0; i < n; ++i) out[i] = triwave(a[i], ins.p0); break;
    case OP_SMIN: case OP_SMAX: {  // the polynomial smooth minimum of the booleans; smax(a, b) = -smin(-a, -b)
        const double k = ins.p0, sign = ins.op == OP_SMIN ? 1.0 : -1.0;
        for (int i = 0; i < n; ++i) {
            const double u = sign * a[i], v = sign * b[i];
            const double h = k > 0 ? std::fmax(k - std::fabs(u - v), 0.0) / k : 0.0;
            out[i] = sign * (std::fmin(u, v) - h * h * k * 0.25);
        }
        break;
    }
    default: throw Error(format("internal error: field opcode %u in a tape", ins.op));
    }
}

// Advances the frame of an IMPLICIT node or a queried field root through its
// tape. The frame's buffers are the clamped point (IMPLICIT only) and the
// tape's slots; a shape read (at the base point, or at EVAL_AT's point)
// pushes the shape's frame, writing into the instruction's slot, and the
// frame resumes after it (Frame::stage is 1 + the next instruction).
bool run_tape(const rsdf_graph &graph, Scratch &s, Frame &f, const Node &node, const double *p) {
    const FieldTape &tape = graph.tapes[node.tape - 1];
    const bool implicit = node.op == OP_IMPLICIT;
    const int n = f.n;
    const size_t first_slot = f.mark + (implicit ? 3 : 0);
    if (f.stage == 0) {
        if (implicit) {
            double *c[3] = {s.take(), s.take(), s.take()};
            const double *q[3] = {f.x, f.y, f.z};
            for (int a = 0; a < 3; ++a)
                for (int i = 0; i < n; ++i) c[a][i] = std::fmin(std::fmax(q[a][i], p[a]), p[3 + a]);
        }
        for (uint32_t k = 0; k < tape.slots; ++k) s.take();
        f.stage = 1;
    }
    const double *base[3] = {f.x, f.y, f.z};
    if (implicit)
        for (int a = 0; a < 3; ++a) base[a] = s.block(f.mark + a);
    const auto operand = [&](uint32_t k) -> const double * {
        return k < 3 ? base[k] : s.block(first_slot + k - 3);
    };
    const uint32_t *args = tape.args.data();
    if (!implicit && f.stage >= 2 && tape.code[f.stage - 2].op == OP_EVAL_AT) {
        // A field root's warp is unproven: a shape read at a point that is
        // not finite or beyond 1e150 (the query limit) has no value there.
        // (A destination never shares a slot with its own operands.)
        const FieldTape::Instr &at = tape.code[f.stage - 2];
        double *dst = s.block(first_slot + at.dst - 3);
        for (int k = 0; k < 3; ++k) {
            const double *c = operand(args[at.first + k]);
            for (int i = 0; i < n; ++i)
                if (!(std::fabs(c[i]) <= 1e150)) dst[i] = NAN;
        }
    }
    for (size_t pc = f.stage - 1; pc < tape.code.size(); ++pc) {
        const FieldTape::Instr &ins = tape.code[pc];
        double *dst = s.block(first_slot + ins.dst - 3);
        if (ins.op == FieldTape::kShape) {
            f.stage = static_cast<uint32_t>(pc + 2);
            push(s, ins.node, n, base[0], base[1], base[2], dst);
            return false;
        }
        if (ins.op == OP_EVAL_AT) {
            const uint32_t *at = args + ins.first;
            f.stage = static_cast<uint32_t>(pc + 2);
            push(s, ins.shape, n, operand(at[0]), operand(at[1]), operand(at[2]), dst);
            return false;
        }
        run_instruction(ins, args + ins.first, n, operand, dst);
    }
    const double *value = operand(tape.result);
    double *out = f.out;
    if (!implicit) {
        std::copy(value, value + n, out);
        return true;
    }
    // h = f(c) / L at c = clamp(p, B), and the exact distance d = box_B(p) to
    // the box B: max(h, d), except outside B where h > 0, which takes
    // hypot(h, d) = |(h, p - c)|. That stays 1-Lipschitz (h is, in c, and
    // projecting onto a convex set gives |c - c'|^2 + |(p - c) - (p' - c')|^2
    // <= |p - p'|^2) and keeps a gradient away from the solid outside B.
    const double L = node.c0;
    double center[3], half[3];
    for (int a = 0; a < 3; ++a) {
        center[a] = 0.5 * (p[a] + p[3 + a]);
        half[a] = 0.5 * (p[3 + a] - p[a]);
    }
    for (int i = 0; i < n; ++i) {
        const double qx = std::fabs(f.x[i] - center[0]) - half[0];
        const double qy = std::fabs(f.y[i] - center[1]) - half[1];
        const double qz = std::fabs(f.z[i] - center[2]) - half[2];
        const double ox = std::fmax(qx, 0.0), oy = std::fmax(qy, 0.0), oz = std::fmax(qz, 0.0);
        const double box = std::sqrt(ox * ox + oy * oy + oz * oz) + std::fmin(std::fmax(qx, std::fmax(qy, qz)), 0.0);
        const double h = value[i] / L;
        out[i] = h > 0 && box > 0 ? std::hypot(h, box) : std::fmax(h, box);
    }
    return true;
}

// Advances frame `f`. Returns true when the frame's values are complete.
// Pushing a child invalidates `f`, so every push is the last action.
bool step(const rsdf_graph &graph, Scratch &s, Frame &f) {
    const Node &node = graph.nodes[f.node];
    const double *p = graph.param_values(node);
    if (node.tape) return run_tape(graph, s, f, node, p);
    if (node.field)
        throw Error(format("node %u (%s) is a field read by other fields; query its root field or its IMPLICIT node",
                           f.node, op_name(node.op)));
    const int n = f.n;
    const double *x = f.x, *y = f.y, *z = f.z;
    double *out = f.out;
    switch (node.op) {
    case OP_UNION: case OP_INTERSECTION: case OP_DIFFERENCE:
    case OP_UNION2: case OP_INTERSECTION2: case OP_DIFFERENCE2: {
        const uint32_t *kids = graph.child_ids(node);
        const double k = p[0];
        if (f.stage == 0) {
            f.stage = 1;
            push(s, kids[0], n, x, y, z, out);
            return false;
        }
        if (f.pending) {
            f.current = combine(node.op, k, n, out, f.tmp);
            f.pending = false;
            f.has_current = true;
        }
        const bool cullable = node.op != OP_INTERSECTION && node.op != OP_INTERSECTION2;
        const bool indexed = cullable && node.child_tree != 0;
        if (indexed && !f.has_candidates) {
            f.has_candidates = true;
            f.next_candidate = s.candidates.size();
            collect_candidates(graph, f, node, k, s.candidates);
            f.candidates_end = s.candidates.size();
        }
        while (indexed ? f.next_candidate < f.candidates_end : f.stage < node.child_count) {
            const uint32_t child = kids[indexed ? s.candidates[f.next_candidate++] : f.stage++];
            if (cullable && culled(graph, f, node, child, k)) continue;
            if (!f.tmp) f.tmp = s.take();
            f.pending = true;
            push(s, child, n, x, y, z, f.tmp);
            return false;
        }
        return true;
    }
    case OP_LOFT: {
        const uint32_t *kids = graph.child_ids(node);
        if (f.stage == 0) {
            f.stage = 1;
            push(s, kids[0], n, x, y, nullptr, out);
            return false;
        }
        if (f.stage == 1) {
            f.stage = 2;
            f.tmp = s.take();
            push(s, kids[1], n, x, y, nullptr, f.tmp);
            return false;
        }
        const double z0 = p[0], z1 = p[1], inv = 1 / (z1 - z0);
        const double center = 0.5 * (z0 + z1), half = 0.5 * (z1 - z0);
        for (int i = 0; i < n; ++i) {
            const double t = clamp01((z[i] - z0) * inv);
            out[i] = slab(((1 - t) * out[i] + t * f.tmp[i]) * node.c0, std::fabs(z[i] - center) - half);
        }
        return true;
    }
    case OP_OFFSET: case OP_SHELL: case OP_OFFSET2: case OP_OUTLINE2:
    case OP_EXTRUDE: case OP_TAPERED_EXTRUDE:
        if (f.stage == 0) {
            f.stage = 1;
            // Extrusions hand their 2D profile the (x, y) coordinates.
            const bool profile = node.op == OP_EXTRUDE || node.op == OP_TAPERED_EXTRUDE;
            push(s, graph.child_ids(node)[0], n, x, y, profile ? nullptr : z, out);
            return false;
        }
        finish_unary(node, p, n, z, out);
        return true;
    case OP_TRANSFORM: case OP_STRETCH: case OP_MIRROR: case OP_TWIST: case OP_ELONGATE: case OP_REVOLVE:
    case OP_TRANSFORM2: case OP_STRETCH2: case OP_MIRROR2:
        if (f.stage == 0) {
            f.stage = 1;
            const double *cx, *cy, *cz;
            map_points(s, node, p, n, x, y, z, cx, cy, cz);
            double scale = 1;
            if (s.facet_sizes) {
                if (node.op == OP_TRANSFORM) scale = p[12];
                if (node.op == OP_STRETCH) scale = std::min({p[0], p[1], p[2]});
                if (node.op == OP_TWIST) scale = node.c0;
            }
            push(s, graph.child_ids(node)[0], n, cx, cy, cz, out, scale);
            return false;
        }
        finish_unary(node, p, n, z, out);
        return true;
    default:
        if (node.op == OP_MESH && s.facet_sizes)
            eval_triangle_mesh(graph.meshes[node.mesh], n, x, y, z, out, s.facet_sizes, s.facet_radius, f.feature_scale);
        else
            eval_leaf(graph, node, p, n, x, y, z, out);
        return true;
    }
}

}  // namespace

void eval_batch(const rsdf_graph &graph, uint32_t node, int n, const double *x, const double *y, const double *z,
                double *out, double *facet_sizes, double facet_radius) {
    if (n <= 0) return;
    Scratch &s = scratch;
    s.reset();
    s.facet_sizes = facet_sizes;
    s.facet_radius = facet_radius;
    if (facet_sizes) std::fill_n(facet_sizes, n, INFINITY);
    push(s, node, n, x, y, z, out);
    while (!s.frames.empty()) {
        Frame &frame = s.frames.back();
        if (step(graph, s, frame)) {
            // `frame` is still the top: steps that push return false.
            s.release(s.frames.back().mark);
            s.candidates.resize(s.frames.back().candidate_mark);
            s.frames.pop_back();
        }
    }
    s.trim();
}

void batch_gradients(const BatchField &field, int n, const double *x, const double *y, const double *z, double eps,
                     double *nx, double *ny, double *nz) {
    // Tetrahedral differences: g = sum_k f(p + eps v_k) v_k for the four
    // vertices v_k of a regular tetrahedron.
    static const double kTetra[4][3] = {{1, -1, -1}, {-1, -1, 1}, {-1, 1, -1}, {1, 1, 1}};
    double sx[kBatch], sy[kBatch], sz[kBatch], values[kBatch];
    for (int i = 0; i < n; ++i)
        for (int k = 0; k < 4; ++k) {
            sx[4 * i + k] = x[i] + eps * kTetra[k][0];
            sy[4 * i + k] = y[i] + eps * kTetra[k][1];
            sz[4 * i + k] = z[i] + eps * kTetra[k][2];
        }
    field(4 * n, sx, sy, sz, values);
    for (int i = 0; i < n; ++i) {
        double gx = 0, gy = 0, gz = 0;
        for (int k = 0; k < 4; ++k) {
            gx += values[4 * i + k] * kTetra[k][0];
            gy += values[4 * i + k] * kTetra[k][1];
            gz += values[4 * i + k] * kTetra[k][2];
        }
        const double length = std::sqrt(gx * gx + gy * gy + gz * gz);
        if (length > 0 && std::isfinite(length)) {
            nx[i] = gx / length;
            ny[i] = gy / length;
            nz[i] = gz / length;
        } else {
            nx[i] = ny[i] = nz[i] = 0;
        }
    }
}

}  // namespace rsdf
