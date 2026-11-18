// Internal declarations shared by the RAD SDF kernel sources.
//
// The decoded graph keeps the RSDF topological order: every child index is
// smaller than its parent's index, so per-node data can be derived in one
// forward pass and evaluation never needs recursion limits beyond the
// explicit evaluation stack in evaluate.cpp.
#pragma once

#include "radsdf.h"

#include <atomic>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <functional>
#include <mutex>
#include <stdexcept>
#include <string>
#include <vector>

namespace rsdf {

enum Op : uint32_t {
    OP_SPHERE = 1, OP_BOX = 2, OP_CYLINDER = 3, OP_CONE = 4, OP_CAPSULE = 5,
    OP_TORUS = 6, OP_ELLIPSOID = 7, OP_HALFSPACE = 8,
    OP_MESH = 9, OP_IMPLICIT = 10,
    OP_TRANSFORM = 20, OP_STRETCH = 21, OP_MIRROR = 22, OP_TWIST = 23,
    OP_ELONGATE = 24, OP_OFFSET = 25, OP_SHELL = 26,
    OP_UNION = 30, OP_INTERSECTION = 31, OP_DIFFERENCE = 32,
    OP_EXTRUDE = 40, OP_TAPERED_EXTRUDE = 41, OP_REVOLVE = 42, OP_LOFT = 43,
    OP_CIRCLE = 100, OP_RECT = 101, OP_POLYGON = 102, OP_SEGMENT = 103,
    OP_ELLIPSE = 104, OP_HALFPLANE = 105,
    OP_TRANSFORM2 = 120, OP_STRETCH2 = 121, OP_MIRROR2 = 122, OP_OFFSET2 = 123,
    OP_OUTLINE2 = 124,
    OP_UNION2 = 130, OP_INTERSECTION2 = 131, OP_DIFFERENCE2 = 132,
    // Field nodes (traced expressions): 3D, no bounds of their own, read only
    // by other field nodes or by IMPLICIT (see field.cpp).
    OP_CONST = 200, OP_COORD = 201, OP_ADD = 202, OP_SUB = 203, OP_MUL = 204, OP_DIV = 205,
    OP_NEG = 206, OP_ABS = 207, OP_SQUARE = 208, OP_SQRT = 209, OP_SIN = 210, OP_COS = 211,
    OP_EXP = 212, OP_LOG = 213, OP_POWC = 214, OP_ATAN2 = 215, OP_MIN = 216, OP_MAX = 217,
    OP_CLAMP = 218, OP_LENGTH = 219, OP_TRIWAVE = 220, OP_EVAL_AT = 221, OP_SMIN = 222, OP_SMAX = 223,
};

// Points per evaluation batch.
constexpr int kBatch = 256;

// Failures inside the kernel; the C boundary turns them into rsdf_last_error.
struct Error : std::runtime_error {
    using std::runtime_error::runtime_error;
};

std::string format(const char *fmt, ...) __attribute__((format(printf, 1, 2)));

struct Node {
    uint32_t op = 0;
    uint32_t dim = 3;             // 2 or 3
    uint32_t first_child = 0;     // into rsdf_graph::children
    uint32_t child_count = 0;
    uint64_t first_param = 0;     // into rsdf_graph::params
    uint32_t param_count = 0;
    uint32_t polygon = 0;         // into rsdf_graph::polygons (POLYGON only)
    uint32_t mesh = 0;            // into rsdf_graph::meshes (MESH only)
    uint32_t child_tree = 0;      // 1 + index into rsdf_graph::child_trees, or 0 (large UNION/DIFFERENCE)
    // 1 + index into rsdf_graph::tapes, or 0: the compiled field of an
    // IMPLICIT node, or of a field node that IMPLICIT or nothing reads (only
    // those field nodes can be queried).
    uint32_t tape = 0;
    uint32_t proof = 0;           // IMPLICIT: 1 + index into rsdf_graph::proofs
    bool field = false;           // a field node (opcodes 200-223)
    // Conservative bounds of the solid (may be infinite). An empty solid may
    // keep an inverted box (lo > hi along some axis): box_distance() to it is
    // positive everywhere, which keeps `ratio` meaningful for empty solids.
    double lo[3] = {0, 0, 0};
    double hi[3] = {0, 0, 0};
    // Distance ratio m in (0, 1]: outside the box, the field is at least m
    // times the Euclidean distance to the box (1 for exact distances). It
    // sizes the bounds of offsets and blends and keeps whole-batch culling
    // value-preserving for fields that underestimate distance.
    double ratio = 1;
    double lipschitz = 1;         // the specified Lipschitz bound (reported by rsdf_bounds)
    double step_lipschitz = 1;    // >= lipschitz; used for pruning and ray stepping
    // An exact distance outside its solid (exact primitives and meshes, rigid
    // transforms and mirrors of exact fields, their extrusions and exact
    // unions): its gradient there points away from a nearest surface point.
    bool exact = false;
    // Its stepping bound holds only near its box: the subtree has a TWIST of
    // an inexact child or a LOFT of inexact profiles, whose slope grows with
    // the distance from their box. Fields reading it use shape_lipschitz().
    bool regional = false;
    double c0 = 0, c1 = 0;        // derived constants, see derive() in graph.cpp (IMPLICIT: L used, proven L)
    // Point queries of this node visit points in Morton order: its evaluation
    // costs enough (64 node visits per point, or a large polygon) that compact
    // batches, which culling and polygon pruning need, repay the sort.
    bool sorts_queries = false;
    uint32_t depth = 0;           // nesting levels below the node
    uint32_t live_buffers = 0;    // point buffers held alive along its deepest path (see graph.cpp)
};

// One node of a bounding volume hierarchy over a contiguous polygon edge range.
struct PolygonTree {
    double lo[2], hi[2];
    uint32_t begin, end;  // edges [begin, end)
    uint32_t right;       // right child index (the left child follows this node); 0 for a leaf
};

// A simple polygon normalised to counter-clockwise order. Edge i joins vertex
// i to vertex i + 1 (cyclically).
struct Polygon {
    std::vector<double> x, y;     // vertices
    std::vector<double> ex, ey;   // edge vectors
    std::vector<double> inv_len2; // 1 / |edge|^2
    std::vector<double> nx, ny;   // vertex pseudonormals: sums of the two edges' unit outward normals
    std::vector<PolygonTree> tree;
};

void build_polygon(Polygon &polygon, const double *coords, size_t point_count);

// A bounding volume hierarchy over the boxes of a large boolean node's
// children after the first, used to find the children a batch cannot cull
// without testing every child. Children with infinite boxes are always
// candidates.
struct ChildTree {
    struct Branch {
        double lo[3], hi[3];
        double ratio;           // smallest distance ratio of the children below
        uint32_t first, count;  // leaf: slots[first, first + count)
        uint32_t right;         // inner node: right child index (the left child follows); 0 for a leaf
    };
    std::vector<Branch> branches;
    std::vector<uint32_t> slots;   // positions in the node's child list
    std::vector<uint32_t> always;  // positions of unbounded children
};

// The closed triangle mesh of a MESH node, validated, oriented outward and
// indexed for nearest-triangle queries (trimesh.cpp). Its field is the union
// of its bodies' exact signed distances.
struct TriangleMesh {
    // A bounding volume hierarchy node (triangles, or bodies in body_tree).
    struct Branch {
        double lo[3], hi[3];
        uint32_t right;         // inner node: right child index (the left child follows); 0 for a leaf
        uint32_t first, count;  // leaf: items [first, first + count)
    };
    // A positive shell and the cavities it encloses.
    struct Body {
        double lo[3], hi[3];
        uint32_t first_root, root_count;  // its shells' subtree roots in body_roots
    };
    std::vector<double> positions;        // x y z per vertex
    std::vector<double> vertex_normals;   // angle-weighted pseudonormal per vertex
    std::vector<uint32_t> triangles;      // outward counter-clockwise corners, in hierarchy order
    std::vector<uint32_t> neighbours;     // per triangle, the triangle across edges 01, 12 and 20
    std::vector<double> normals;          // unit outward normal per triangle
    std::vector<uint32_t> triangle_body;  // body of each triangle
    std::vector<Branch> tree;             // root 0; every shell is a subtree
    std::vector<Body> bodies;
    std::vector<uint32_t> body_roots;
    std::vector<Branch> body_tree;        // over the bodies' boxes (leaves index body_order); several bodies only
    std::vector<uint32_t> body_order;
    std::vector<Branch> mixed;            // over all triangles, shells mixed (leaves index mixed_order); several bodies only
    std::vector<uint32_t> mixed_order;
    double lo[3] = {0, 0, 0}, hi[3] = {0, 0, 0};
};

// Decodes MESH parameters (vertex count, triangle count, vertices, triangles)
// into `mesh`; throws Error describing an invalid mesh.
void build_triangle_mesh(TriangleMesh &mesh, const double *params, size_t param_count);

// A field DAG compiled to straight-line code (field.cpp). Every node of the
// field runs once per batch, in topological order, into a value slot that is
// reused after its last read, so shared subexpressions cost nothing extra.
// Operands 0-2 are the base point's coordinates (an IMPLICIT's clamped point,
// or the query point of a field root); operand 3 + k is slot k.
struct FieldTape {
    static constexpr uint32_t kShape = 0;  // Instr::op of a shape read as a field at the base point
    struct Instr {
        uint32_t op;           // field opcode (never COORD, which is an operand), or kShape
        uint32_t node;         // the graph node computed (kShape: the shape node)
        uint32_t dst;          // operand written (>= 3)
        uint32_t first, count; // operands in args
        uint32_t shape;        // EVAL_AT: the shape node
        double p0, p1;         // CONST c, POWC e, CLAMP lo hi, TRIWAVE P
    };
    std::vector<Instr> code;
    std::vector<uint32_t> args;
    uint32_t slots = 0;
    uint32_t result = 0;       // operand holding the field's value
    uint32_t root = 0;         // the field's root node
    uint64_t visits = 1;       // node visits per evaluated point (instructions plus the shapes read)
};

// Compiles the field rooted at `root` (a field node or a 3D shape node).
void build_tape(const rsdf_graph &graph, uint32_t root, FieldTape &tape);

// The proven slope bound of a field over a box (field.cpp, "Custom fields"
// in SDF-DESIGN.md): forward-mode interval arithmetic of values and
// gradients with adaptive subdivision. Deterministic and single-threaded.
struct FieldBound {
    bool ok = false;
    double bound = INFINITY;              // upper bound of |grad f| over the box
    double value_lo = -INFINITY, value_hi = INFINITY;
    uint64_t boxes = 0;                   // boxes evaluated
    uint32_t failed_node = 0;             // without a bound: the node whose slope is unbounded
    double failed_lo[3] = {0, 0, 0}, failed_hi[3] = {0, 0, 0};  // and the region
    std::string problem;                  // without a bound: what failed, naming the node and region
};

constexpr uint64_t kFieldBoxes = 32768;  // default subdivision budget

// One proven IMPLICIT field (graph.cpp), shared by the IMPLICIT nodes whose
// fields compile to the same tape over the same box. The sampled gradient
// statistics are filled in by the first rsdf_field_analyze of its bounds.
struct Proof {
    FieldBound bound;
    mutable bool sampled = false;  // the statistics below, under rsdf_graph::proofs_mutex
    mutable uint64_t samples = 0;
    mutable double gradient_min = 0, gradient_median = 0, gradient_max = 0;
};

// The default budget of boxes for a tape: 32,768, fewer for large fields so
// that an analysis visits at most about 2^24 nodes (a box and its centre are
// two evaluations).
uint64_t field_budget(const FieldTape &tape);

FieldBound analyze_field(const rsdf_graph &graph, const FieldTape &tape, const double *lo, const double *hi,
                         uint64_t max_boxes);

// The box IMPLICIT analyses: its bounds grown by 1e-6 of their largest extent.
void implicit_analysis_box(const double *lo, const double *hi, double *analysis_lo, double *analysis_hi);

// A Lipschitz bound of the 3D shape `node` over the box [lo, hi] of its own
// coordinates: its stepping bound unless it is regional (field.cpp).
double shape_lipschitz(const rsdf_graph &graph, uint32_t node, const double *lo, const double *hi);

// Proof limits: a field costing more node visits per point cannot be proven
// (one box and its centre would exceed the 2^24 visits of a default proof),
// and the IMPLICIT proofs of one graph may visit at most kProofWork nodes:
// each distinct proof gets its default budget or one box plus an equal share
// of the work left beyond the later proofs' first boxes, whichever is
// smaller, and a graph fails at once when its proofs cannot all evaluate one
// box (graph.cpp).
constexpr uint64_t kMaxProvenVisits = uint64_t(1) << 23;
constexpr uint64_t kProofWork = uint64_t(1) << 28;

// The RSDF name of an opcode ("?" when unknown).
const char *op_name(uint32_t op);

}  // namespace rsdf

struct rsdf_graph {
    std::vector<rsdf::Node> nodes;
    std::vector<uint32_t> children;
    std::vector<double> params;
    std::vector<rsdf::Polygon> polygons;
    std::vector<rsdf::ChildTree> child_trees;
    std::vector<rsdf::TriangleMesh> meshes;
    std::vector<rsdf::FieldTape> tapes;
    std::vector<rsdf::Proof> proofs;
    mutable std::mutex proofs_mutex;  // guards Proof::sampled and the statistics

    const uint32_t *child_ids(const rsdf::Node &node) const { return children.data() + node.first_child; }
    const double *param_values(const rsdf::Node &node) const { return params.data() + node.first_param; }

    // Unique among the graphs of this process: a mesh records the graph it
    // was built from, so simplification can refuse another graph's mesh.
    uint64_t identity = next_identity();

    static uint64_t next_identity() {
        static std::atomic<uint64_t> created{0};
        return created.fetch_add(1, std::memory_order_relaxed) + 1;
    }
};

namespace rsdf {

// ---------------------------------------------------------------- errors

void set_error(const std::string &message);
void clear_error();

// Runs `body` and converts any exception into rsdf_last_error. Returns false on failure.
bool guarded(const std::function<void()> &body);

const rsdf_graph &checked_graph(const rsdf_graph *graph);
const Node &checked_node(const rsdf_graph &graph, uint32_t node);

// ---------------------------------------------------------------- threading

// The worker count for a requested thread count (<= 0 means all cores).
int resolve_threads(int32_t requested);

// The worker count for evaluating `node`: resolve_threads(requested), limited
// so that the per-thread evaluation buffers and frames of very deep graphs
// (up to 128 MB per thread) stay within about 1 GB in total.
int evaluation_threads(const Node &node, int32_t requested);

// Runs body(chunk) for every chunk in [0, chunks) on up to `threads` workers
// with libdispatch. Work assignment never changes what a chunk computes, so
// results are identical for any thread count. The first exception thrown by a
// chunk is rethrown on the calling thread after all workers stop.
void parallel_chunks(size_t chunks, int threads, const std::function<void(size_t)> &body);

// ---------------------------------------------------------------- evaluation

// Evaluates `node` at n <= kBatch points given as structure-of-arrays
// coordinates. z is ignored (and may be null) for 2D nodes.
void eval_batch(const rsdf_graph &graph, uint32_t node, int n, const double *x, const double *y,
                const double *z, double *out);

// Distances to a polygon (signed, negative inside) for one batch.
void eval_polygon(const Polygon &polygon, int n, const double *x, const double *y, double *out);

// Signed distances to a triangle mesh for one batch.
void eval_triangle_mesh(const TriangleMesh &mesh, int n, const double *x, const double *y, const double *z,
                        double *out);

// Tetrahedral finite-difference gradients of a 3D field at n <= kBatch / 4
// points, normalised; a vanishing gradient yields (0, 0, 0). `field` evaluates
// one batch of at most kBatch points.
using BatchField = std::function<void(int n, const double *x, const double *y, const double *z, double *out)>;
void batch_gradients(const BatchField &field, int n, const double *x, const double *y, const double *z,
                     double eps, double *nx, double *ny, double *nz);

// ---------------------------------------------------------------- spatial order

// Interleaves the low 21 bits of v with two zero bits per bit (and back).
inline uint64_t spread3(uint64_t v) {
    v &= 0x1fffff;
    v = (v | v << 32) & 0x1f00000000ffffULL;
    v = (v | v << 16) & 0x1f0000ff0000ffULL;
    v = (v | v << 8) & 0x100f00f00f00f00fULL;
    v = (v | v << 4) & 0x10c30c30c30c30c3ULL;
    v = (v | v << 2) & 0x1249249249249249ULL;
    return v;
}

inline uint64_t compact3(uint64_t v) {
    v &= 0x1249249249249249ULL;
    v = (v ^ (v >> 2)) & 0x10c30c30c30c30c3ULL;
    v = (v ^ (v >> 4)) & 0x100f00f00f00f00fULL;
    v = (v ^ (v >> 8)) & 0x1f0000ff0000ffULL;
    v = (v ^ (v >> 16)) & 0x1f00000000ffffULL;
    v = (v ^ (v >> 32)) & 0x1fffffULL;
    return v;
}

inline uint64_t morton(const uint64_t c[3]) { return spread3(c[0]) | spread3(c[1]) << 1 | spread3(c[2]) << 2; }

inline int bit_width(uint64_t value) { return value ? 64 - __builtin_clzll(value) : 0; }

// Stable LSD radix sort of keys < 2^bits, carrying optional 32-bit values.
void radix_sort(std::vector<uint64_t> &keys, std::vector<uint32_t> *values, int bits);

// Sorts items by their bits [shift, shift + bits): a stable LSD radix sort
// whose passes split the items into fixed chunks run on up to `threads`
// workers (the result does not depend on the thread count).
void radix_sort_items(std::vector<uint64_t> &items, int shift, int bits, int threads);

// A visiting order of `count` (< 2^32) points, point(i, xyz) writing point i:
// the Morton order of the points within their bounding box (8 bits per
// axis). Batches taken in this order stay spatially compact, which
// whole-batch culling and polygon pruning need; results never depend on it.
template <class PointAt>
std::vector<uint32_t> morton_order(size_t count, int threads, const PointAt &point) {
    constexpr size_t kPart = 65536;
    constexpr double kTop = 255;
    const size_t parts = (count + kPart - 1) / kPart;
    std::vector<double> boxes(6 * parts);
    parallel_chunks(parts, threads, [&](size_t part) {
        double lo[3] = {INFINITY, INFINITY, INFINITY}, hi[3] = {-INFINITY, -INFINITY, -INFINITY}, p[3];
        const size_t end = std::min(count, (part + 1) * kPart);
        for (size_t i = part * kPart; i < end; ++i) {
            point(i, p);
            for (int a = 0; a < 3; ++a) {
                if (!std::isfinite(p[a])) continue;
                lo[a] = std::fmin(lo[a], p[a]);
                hi[a] = std::fmax(hi[a], p[a]);
            }
        }
        for (int a = 0; a < 3; ++a) {
            boxes[6 * part + a] = lo[a];
            boxes[6 * part + 3 + a] = hi[a];
        }
    });
    double lo[3] = {INFINITY, INFINITY, INFINITY}, hi[3] = {-INFINITY, -INFINITY, -INFINITY}, scale[3];
    for (size_t part = 0; part < parts; ++part)
        for (int a = 0; a < 3; ++a) {
            lo[a] = std::fmin(lo[a], boxes[6 * part + a]);
            hi[a] = std::fmax(hi[a], boxes[6 * part + 3 + a]);
        }
    for (int a = 0; a < 3; ++a) scale[a] = hi[a] > lo[a] ? kTop / (hi[a] - lo[a]) : 0.0;
    std::vector<uint64_t> items(count);
    parallel_chunks(parts, threads, [&](size_t part) {
        double p[3];
        const size_t end = std::min(count, (part + 1) * kPart);
        for (size_t i = part * kPart; i < end; ++i) {
            point(i, p);
            uint64_t c[3];
            for (int a = 0; a < 3; ++a) {
                const double q = (p[a] - lo[a]) * scale[a];
                c[a] = q >= 0 && q <= kTop ? static_cast<uint64_t>(q) : 0;  // NaN: 0
            }
            items[i] = morton(c) << 32 | i;
        }
    });
    radix_sort_items(items, 32, 24, threads);
    std::vector<uint32_t> order(count);
    for (size_t i = 0; i < count; ++i) order[i] = static_cast<uint32_t>(items[i]);
    return order;
}

// Minimum Euclidean distance between two axis-aligned boxes (0 when they overlap).
inline double box_distance(const double *alo, const double *ahi, const double *blo, const double *bhi, int dim) {
    double sum = 0;
    for (int a = 0; a < dim; ++a) {
        double gap = std::fmax(blo[a] - ahi[a], alo[a] - bhi[a]);
        if (gap > 0) sum += gap * gap;
    }
    return std::sqrt(sum);
}

}  // namespace rsdf
