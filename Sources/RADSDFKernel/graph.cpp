// RSDF decoding: validation, derived bounds and Lipschitz bounds.
#include "kernel.h"

#include <algorithm>
#include <cstring>
#include <memory>

namespace rsdf {
namespace {

constexpr uint32_t kMaxNodes = 4000000;
constexpr uint32_t kMaxChildren = 65536;
constexpr uint32_t kMaxParams = 1000000;
// MESH: counts, 2,000,000 vertices and 2,000,000 triangles (trimesh.cpp).
constexpr uint32_t kMaxMeshParams = 2 + 3 * 2000000 + 3 * 2000000;
constexpr size_t kMaxBytes = size_t(512) << 20;
constexpr double kTolerance = 1e-6;
constexpr double kInf = INFINITY;
// Evaluation keeps one frame per nesting level and the point buffers of every
// point-mapping ancestor alive, so nesting is bounded to keep that memory
// (per evaluating thread) modest.
constexpr uint32_t kMaxDepth = 200000;
constexpr uint32_t kMaxLiveBuffers = 65536;
// Evaluation visits a shared subgraph once per path to it, so a few doublings
// of `s = s | s.translate(...)` describe more work than any query can finish.
// Graphs whose evaluation could visit more nodes per point than this (before
// culling) are refused.
constexpr uint64_t kMaxVisits = uint64_t(1) << 30;

enum ChildRule : uint8_t { NO_CHILDREN, ONE_CHILD, TWO_CHILDREN, CHILDREN };

struct OpInfo {
    uint32_t op;
    const char *name;
    uint8_t dim;        // dimension of the node
    ChildRule children;
    uint8_t child_dim;  // dimension required of its children
    int params;         // exact parameter count; -1: POLYGON (even, at least six); -2: MESH (checked when decoded)
};

const OpInfo kOps[] = {
    {OP_SPHERE, "SPHERE", 3, NO_CHILDREN, 0, 1},
    {OP_BOX, "BOX", 3, NO_CHILDREN, 0, 4},
    {OP_CYLINDER, "CYLINDER", 3, NO_CHILDREN, 0, 2},
    {OP_CONE, "CONE", 3, NO_CHILDREN, 0, 3},
    {OP_CAPSULE, "CAPSULE", 3, NO_CHILDREN, 0, 7},
    {OP_TORUS, "TORUS", 3, NO_CHILDREN, 0, 2},
    {OP_ELLIPSOID, "ELLIPSOID", 3, NO_CHILDREN, 0, 3},
    {OP_HALFSPACE, "HALFSPACE", 3, NO_CHILDREN, 0, 4},
    {OP_MESH, "MESH", 3, NO_CHILDREN, 0, -2},
    {OP_TRANSFORM, "TRANSFORM", 3, ONE_CHILD, 3, 13},
    {OP_STRETCH, "STRETCH", 3, ONE_CHILD, 3, 3},
    {OP_MIRROR, "MIRROR", 3, ONE_CHILD, 3, 4},
    {OP_TWIST, "TWIST", 3, ONE_CHILD, 3, 1},
    {OP_ELONGATE, "ELONGATE", 3, ONE_CHILD, 3, 3},
    {OP_OFFSET, "OFFSET", 3, ONE_CHILD, 3, 1},
    {OP_SHELL, "SHELL", 3, ONE_CHILD, 3, 2},
    {OP_UNION, "UNION", 3, CHILDREN, 3, 1},
    {OP_INTERSECTION, "INTERSECTION", 3, CHILDREN, 3, 1},
    {OP_DIFFERENCE, "DIFFERENCE", 3, CHILDREN, 3, 1},
    {OP_EXTRUDE, "EXTRUDE", 3, ONE_CHILD, 2, 2},
    {OP_TAPERED_EXTRUDE, "TAPERED_EXTRUDE", 3, ONE_CHILD, 2, 3},
    {OP_REVOLVE, "REVOLVE", 3, ONE_CHILD, 2, 1},
    {OP_LOFT, "LOFT", 3, TWO_CHILDREN, 2, 2},
    {OP_CIRCLE, "CIRCLE", 2, NO_CHILDREN, 0, 1},
    {OP_RECT, "RECT", 2, NO_CHILDREN, 0, 3},
    {OP_POLYGON, "POLYGON", 2, NO_CHILDREN, 0, -1},
    {OP_SEGMENT, "SEGMENT", 2, NO_CHILDREN, 0, 5},
    {OP_ELLIPSE, "ELLIPSE", 2, NO_CHILDREN, 0, 2},
    {OP_HALFPLANE, "HALFPLANE", 2, NO_CHILDREN, 0, 3},
    {OP_TRANSFORM2, "TRANSFORM2", 2, ONE_CHILD, 2, 7},
    {OP_STRETCH2, "STRETCH2", 2, ONE_CHILD, 2, 2},
    {OP_MIRROR2, "MIRROR2", 2, ONE_CHILD, 2, 3},
    {OP_OFFSET2, "OFFSET2", 2, ONE_CHILD, 2, 1},
    {OP_OUTLINE2, "OUTLINE2", 2, ONE_CHILD, 2, 1},
    {OP_UNION2, "UNION2", 2, CHILDREN, 2, 1},
    {OP_INTERSECTION2, "INTERSECTION2", 2, CHILDREN, 2, 1},
    {OP_DIFFERENCE2, "DIFFERENCE2", 2, CHILDREN, 2, 1},
};

const OpInfo *find_op(uint32_t op) {
    for (const OpInfo &info : kOps)
        if (info.op == op) return &info;
    return nullptr;
}

struct Reader {
    const uint8_t *data;
    size_t length;
    size_t offset = 0;

    size_t remaining() const { return length - offset; }
    uint32_t u32() {
        uint32_t value;
        std::memcpy(&value, data + offset, 4);
        offset += 4;
        return value;
    }
    double f64() {
        double value;
        std::memcpy(&value, data + offset, 8);
        offset += 8;
        return value;
    }
};

// Validation of one node's parameters; `where` names the node in messages.
class Checker {
public:
    Checker(const std::string &where, const double *params) : where_(where), p_(params) {}

    void fail(const std::string &what) const { throw Error(where_ + ": " + what); }
    void positive(int i, const char *name) const {
        if (!(p_[i] > 0)) fail(format("%s must be positive (got %.17g)", name, p_[i]));
    }
    void nonnegative(int i, const char *name) const {
        if (!(p_[i] >= 0)) fail(format("%s must not be negative (got %.17g)", name, p_[i]));
    }
    void unit(int first, int dim) const {
        double sum = 0;
        for (int a = 0; a < dim; ++a) sum += p_[first + a] * p_[first + a];
        if (!(std::fabs(std::sqrt(sum) - 1) <= kTolerance))
            fail(format("normal must have unit length (got length %.17g)", std::sqrt(sum)));
    }
    void ordered(int lo, int hi) const {
        if (!(p_[hi] > p_[lo])) fail(format("z1 (%.17g) must be greater than z0 (%.17g)", p_[hi], p_[lo]));
    }
    // A row-major dim x dim proper rotation: orthonormal with determinant +1.
    void rotation(int first, int dim) const {
        const double *r = p_ + first;
        for (int i = 0; i < dim; ++i)
            for (int j = 0; j < dim; ++j) {
                double dot = 0;
                for (int k = 0; k < dim; ++k) dot += r[k * dim + i] * r[k * dim + j];
                if (!(std::fabs(dot - (i == j ? 1.0 : 0.0)) <= kTolerance))
                    fail("rotation matrix is not orthonormal");
            }
        double det = dim == 2 ? r[0] * r[3] - r[1] * r[2]
                              : r[0] * (r[4] * r[8] - r[5] * r[7]) - r[1] * (r[3] * r[8] - r[5] * r[6]) +
                                    r[2] * (r[3] * r[7] - r[4] * r[6]);
        if (!(std::fabs(det - 1) <= kTolerance))
            fail(format("rotation matrix must have determinant +1 (got %.17g); use MIRROR for reflections", det));
    }

private:
    const std::string &where_;
    const double *p_;
};

void validate_params(const Node &node, const double *p, const Checker &check) {
    switch (node.op) {
    case OP_SPHERE: check.positive(0, "radius"); break;
    case OP_BOX:
        check.positive(0, "half extent x");
        check.positive(1, "half extent y");
        check.positive(2, "half extent z");
        check.nonnegative(3, "edge radius");
        if (p[3] > std::min({p[0], p[1], p[2]})) check.fail("edge radius must not exceed the smallest half extent");
        break;
    case OP_CYLINDER:
        check.positive(0, "radius");
        check.positive(1, "half height");
        break;
    case OP_CONE:
        check.nonnegative(0, "radius r1");
        check.nonnegative(1, "radius r2");
        if (!(std::max(p[0], p[1]) > 0)) check.fail("at least one cone radius must be positive");
        check.positive(2, "half height");
        break;
    case OP_CAPSULE: check.positive(6, "radius"); break;
    case OP_TORUS:
        check.positive(0, "major radius");
        check.positive(1, "minor radius");
        break;
    case OP_ELLIPSOID:
        check.positive(0, "radius x");
        check.positive(1, "radius y");
        check.positive(2, "radius z");
        break;
    case OP_HALFSPACE: check.unit(0, 3); break;
    case OP_TRANSFORM:
        check.rotation(0, 3);
        check.positive(12, "scale");
        break;
    case OP_STRETCH:
        check.positive(0, "stretch x");
        check.positive(1, "stretch y");
        check.positive(2, "stretch z");
        break;
    case OP_MIRROR: check.unit(0, 3); break;
    case OP_ELONGATE:
        check.nonnegative(0, "elongation x");
        check.nonnegative(1, "elongation y");
        check.nonnegative(2, "elongation z");
        break;
    case OP_SHELL:
        check.positive(0, "thickness");
        if (!(p[1] == 0 || p[1] == 1 || p[1] == 2)) check.fail("mode must be 0 (inside), 1 (outside) or 2 (center)");
        break;
    case OP_UNION: case OP_INTERSECTION: case OP_DIFFERENCE:
    case OP_UNION2: case OP_INTERSECTION2: case OP_DIFFERENCE2:
        check.nonnegative(0, "blend radius k");
        break;
    case OP_EXTRUDE: case OP_TAPERED_EXTRUDE: case OP_LOFT: check.ordered(0, 1); break;
    case OP_CIRCLE: check.positive(0, "radius"); break;
    case OP_RECT:
        check.positive(0, "half extent x");
        check.positive(1, "half extent y");
        check.nonnegative(2, "corner radius");
        if (p[2] > std::min(p[0], p[1])) check.fail("corner radius must not exceed the smaller half extent");
        break;
    case OP_SEGMENT: check.positive(4, "radius"); break;
    case OP_ELLIPSE:
        check.positive(0, "radius x");
        check.positive(1, "radius y");
        break;
    case OP_HALFPLANE: check.unit(0, 2); break;
    case OP_TRANSFORM2:
        check.rotation(0, 2);
        check.positive(6, "scale");
        break;
    case OP_STRETCH2:
        check.positive(0, "stretch x");
        check.positive(1, "stretch y");
        break;
    case OP_MIRROR2: check.unit(0, 2); break;
    case OP_OUTLINE2: check.positive(0, "thickness"); break;
    default: break;  // TWIST, OFFSET, REVOLVE, OFFSET2: any finite value; POLYGON: build_polygon
    }
}

// Point buffers a node keeps alive while evaluating its child `index`
// (mirrors step() in evaluate.cpp).
uint32_t held_buffers(uint32_t op, uint32_t index) {
    switch (op) {
    case OP_TRANSFORM: case OP_STRETCH: case OP_MIRROR: case OP_ELONGATE: return 3;
    case OP_TWIST: case OP_REVOLVE: case OP_TRANSFORM2: case OP_STRETCH2: case OP_MIRROR2: return 2;
    case OP_LOFT: case OP_UNION: case OP_INTERSECTION: case OP_DIFFERENCE:
    case OP_UNION2: case OP_INTERSECTION2: case OP_DIFFERENCE2: return index == 0 ? 0 : 1;
    default: return 0;
    }
}

// ------------------------------------------------------------------ bounds
//
// Every node gets a box B and a distance ratio m (Node::ratio): outside B its
// field is at least m times the Euclidean distance to B. Exact distances have
// m = 1; fields that underestimate distance (ellipsoids, stretches, twists,
// tapers, lofts, intersections) have m < 1. The solid {f <= d} of a field
// grown by d > 0 then lies in B expanded by d / m, since outside the box
// expanded by e the distance to B exceeds the distance to the expanded box by
// at least e.
//
// Boxes are kept raw: an operation that proves its solid empty (such as an
// intersection of disjoint boxes) leaves lo > hi along some axis. Distances
// to an inverted box are positive everywhere and every rule below holds for
// them, so a grown empty solid (which need not be empty) keeps correct bounds.

bool finite_box(const Node &node) {
    for (uint32_t a = 0; a < node.dim; ++a)
        if (!std::isfinite(node.lo[a]) || !std::isfinite(node.hi[a])) return false;
    return true;
}

// How far a box must grow to hold {f <= d} of a node with distance ratio m.
double growth(const Node &child, double d) { return d > 0 ? d / child.ratio : 0.0; }

void set_symmetric(Node &node, double x, double y, double z = 0) {
    const double half[3] = {x, y, z};
    for (uint32_t a = 0; a < node.dim; ++a) {
        node.lo[a] = -half[a];
        node.hi[a] = half[a];
    }
}

void copy_box(Node &node, const Node &child) {
    std::copy(child.lo, child.lo + 3, node.lo);
    std::copy(child.hi, child.hi + 3, node.hi);
}

void expand(Node &node, double amount) {
    for (uint32_t a = 0; a < node.dim; ++a) {
        node.lo[a] -= amount;
        node.hi[a] += amount;
    }
}

// Bounds of the image of the child box under the similarity p -> m p + t (m
// row-major, dim x dim). Interval arithmetic gives the exact box of the
// transformed corners and handles infinite extents (a zero coefficient never
// meets an infinity). An inverted (empty) child box stays inverted when every
// row has one nonzero coefficient (translations, quarter turns, axis-aligned
// mirrors); other rotations map its normalised box, which is conservative.
void map_box(Node &node, const Node &child, const double *m, const double *t) {
    const int dim = static_cast<int>(node.dim);
    bool monomial = true;
    for (int i = 0; i < dim; ++i) {
        int nonzero = 0;
        for (int j = 0; j < dim; ++j) nonzero += m[i * dim + j] != 0;
        monomial = monomial && nonzero == 1;
    }
    double lo[3], hi[3];
    for (int j = 0; j < dim; ++j) {
        lo[j] = monomial ? child.lo[j] : std::min(child.lo[j], child.hi[j]);
        hi[j] = monomial ? child.hi[j] : std::max(child.lo[j], child.hi[j]);
    }
    for (int i = 0; i < dim; ++i) {
        double a = t[i], b = t[i];
        for (int j = 0; j < dim; ++j) {
            const double c = m[i * dim + j];
            if (c > 0) {
                a += c * lo[j];
                b += c * hi[j];
            } else if (c < 0) {
                a += c * hi[j];
                b += c * lo[j];
            }
        }
        node.lo[i] = a;
        node.hi[i] = b;
    }
}

// How far, in units of the blend radius k, a smooth union's left fold over
// `children` children can reach outside the hull of their boxes. Outside the
// hull every child is at least D, the distance to the hull, and smin is
// monotone, so the fold is at least D - k x_(n-1) with x_0 = 0 and
// x_(m+1) = x_m + (1 - x_m)^2 / 4, the smin of D - k x_m and D. Coincident
// children reach exactly this far: x_1 = 1/4, x_2 = 0.390625, and
// x_m <= min(1, m / 4).
double smooth_union_reach(uint32_t children) {
    double x = 0;
    for (uint32_t m = 1; m < children; ++m) x += 0.25 * (1 - x) * (1 - x);
    return x;
}

// {p : n.p - d <= 0} is bounded only along an axis-aligned normal.
void half_space_box(Node &node, const double *n, double d) {
    const int dim = static_cast<int>(node.dim);
    for (int a = 0; a < dim; ++a) {
        node.lo[a] = -kInf;
        node.hi[a] = kInf;
    }
    for (int a = 0; a < dim; ++a) {
        bool aligned = n[a] != 0;
        for (int b = 0; b < dim; ++b)
            if (b != a && n[b] != 0) aligned = false;
        if (!aligned) continue;
        if (n[a] > 0) node.hi[a] = d / n[a];
        else node.lo[a] = d / n[a];
    }
}

// Derives bounds, distance ratios, Lipschitz bounds and evaluation constants
// of a validated node whose children are already derived (see "bounds" above).
void derive(rsdf_graph &graph, Node &node, const std::string &where) {
    const double *p = graph.param_values(node);
    const uint32_t *kids = graph.child_ids(node);
    const Node *child = node.child_count ? &graph.nodes[kids[0]] : nullptr;
    double lipschitz = 1, step = 1;
    for (uint32_t i = 0; i < node.child_count; ++i) {
        lipschitz = std::max(lipschitz, graph.nodes[kids[i]].lipschitz);
        step = std::max(step, graph.nodes[kids[i]].step_lipschitz);
    }
    // Exact primitives, isometries, elongations, revolutions, extrusions,
    // offsets, shells and differences keep their (first) child's ratio.
    double ratio = child ? child->ratio : 1.0;

    switch (node.op) {
    case OP_SPHERE: set_symmetric(node, p[0], p[0], p[0]); break;
    case OP_BOX: set_symmetric(node, p[0], p[1], p[2]); break;
    case OP_CYLINDER: set_symmetric(node, p[0], p[0], p[1]); break;
    case OP_CONE: {
        const double r = std::max(p[0], p[1]);
        set_symmetric(node, r, r, p[2]);
        break;
    }
    case OP_CAPSULE: case OP_SEGMENT: {
        const int dim = static_cast<int>(node.dim);
        const double *a = p, *b = p + dim, r = p[2 * dim];
        double len2 = 0;
        for (int i = 0; i < dim; ++i) {
            node.lo[i] = std::min(a[i], b[i]) - r;
            node.hi[i] = std::max(a[i], b[i]) + r;
            len2 += (b[i] - a[i]) * (b[i] - a[i]);
        }
        node.c0 = len2 > 0 ? 1 / len2 : 0;
        break;
    }
    case OP_TORUS: set_symmetric(node, p[0] + p[1], p[0] + p[1], p[1]); break;
    case OP_ELLIPSOID: case OP_ELLIPSE: {
        // (|p/r| - 1) min r >= (min r / max r) * distance to the ellipsoid.
        const int dim = static_cast<int>(node.dim);
        set_symmetric(node, p[0], p[1], dim == 3 ? p[2] : 0.0);
        node.c0 = *std::min_element(p, p + dim);
        ratio = node.c0 / *std::max_element(p, p + dim);
        break;
    }
    case OP_HALFSPACE: half_space_box(node, p, p[3]); break;
    case OP_HALFPLANE: half_space_box(node, p, p[2]); break;
    case OP_TRANSFORM: {
        double m[9];
        for (int i = 0; i < 9; ++i) m[i] = p[12] * p[i];
        map_box(node, *child, m, p + 9);
        break;
    }
    case OP_TRANSFORM2: {
        const double m[4] = {p[6] * p[0], p[6] * p[1], p[6] * p[2], p[6] * p[3]};
        map_box(node, *child, m, p + 4);
        break;
    }
    case OP_STRETCH: case OP_STRETCH2: {
        copy_box(node, *child);
        double smallest = kInf, largest = 0;
        for (uint32_t a = 0; a < node.dim; ++a) {
            node.lo[a] *= p[a];
            node.hi[a] *= p[a];
            smallest = std::min(smallest, p[a]);
            largest = std::max(largest, p[a]);
        }
        node.c0 = smallest;
        ratio *= smallest / largest;
        break;
    }
    case OP_MIRROR: case OP_MIRROR2: {
        const int dim = static_cast<int>(node.dim);
        const double d = p[dim];
        double m[9], t[3];
        for (int i = 0; i < dim; ++i) {
            for (int j = 0; j < dim; ++j) m[i * dim + j] = (i == j ? 1.0 : 0.0) - 2 * p[i] * p[j];
            t[i] = 2 * d * p[i];
        }
        map_box(node, *child, m, t);
        break;
    }
    case OP_TWIST: {
        if (!finite_box(*child)) throw Error(where + ": TWIST needs a bounded child");
        // The child box lies in the cylinder of radius rho; the twist keeps
        // radii and heights, so distances to that cylinder bound the ratio.
        const double rx = std::max(std::fabs(child->lo[0]), std::fabs(child->hi[0]));
        const double ry = std::max(std::fabs(child->lo[1]), std::fabs(child->hi[1]));
        const double rho = std::hypot(rx, ry);
        const double w = std::fabs(p[0]) * rho;
        set_symmetric(node, rho, rho);
        node.lo[2] = child->lo[2];
        node.hi[2] = child->hi[2];
        node.c0 = 1 / std::sqrt(1 + w * w);
        ratio *= node.c0;
        // The specified correction keeps the reported bound at the child's;
        // the twist map's exact stretch factor is (w + sqrt(w^2 + 4)) / 2.
        step *= 0.5 * (w + std::sqrt(w * w + 4)) * node.c0;
        break;
    }
    case OP_ELONGATE:
        copy_box(node, *child);
        for (int a = 0; a < 3; ++a) {
            node.lo[a] -= p[a];
            node.hi[a] += p[a];
        }
        break;
    case OP_OFFSET: case OP_OFFSET2:
        copy_box(node, *child);
        expand(node, growth(*child, p[0]));
        break;
    case OP_SHELL:  // inside: max(f, -f - t) >= f; outside: >= f - t; center: >= f - t / 2
        copy_box(node, *child);
        expand(node, growth(*child, p[1] == 1 ? p[0] : p[1] == 2 ? 0.5 * p[0] : 0.0));
        break;
    case OP_OUTLINE2:
        copy_box(node, *child);
        expand(node, growth(*child, 0.5 * p[0]));
        break;
    case OP_UNION: case OP_UNION2:
        // Outside the hull H of the children's boxes every child is at least
        // m dist(p, H) with m the smallest ratio, so the fold is at least
        // m dist(p, H) - k x_(n-1) (see smooth_union_reach).
        for (uint32_t a = 0; a < node.dim; ++a) {
            node.lo[a] = kInf;
            node.hi[a] = -kInf;
        }
        for (uint32_t i = 0; i < node.child_count; ++i) {
            const Node &c = graph.nodes[kids[i]];
            for (uint32_t a = 0; a < node.dim; ++a) {
                node.lo[a] = std::min(node.lo[a], c.lo[a]);
                node.hi[a] = std::max(node.hi[a], c.hi[a]);
            }
            ratio = std::min(ratio, c.ratio);
        }
        if (p[0] > 0) expand(node, p[0] * smooth_union_reach(node.child_count) / ratio);
        break;
    case OP_INTERSECTION: case OP_INTERSECTION2: {
        // Along each axis the gap to the intersection of the boxes is one
        // child's gap, so dist(p, box) <= sqrt(min(n, dim)) max_i dist(p, box_i);
        // a child whose box is the intersection bounds the ratio directly.
        copy_box(node, *child);
        for (uint32_t i = 1; i < node.child_count; ++i) {
            const Node &c = graph.nodes[kids[i]];
            for (uint32_t a = 0; a < node.dim; ++a) {
                node.lo[a] = std::max(node.lo[a], c.lo[a]);
                node.hi[a] = std::min(node.hi[a], c.hi[a]);
            }
            ratio = std::min(ratio, c.ratio);
        }
        ratio /= std::sqrt(static_cast<double>(std::min(node.child_count, node.dim)));
        for (uint32_t i = 0; i < node.child_count; ++i) {
            const Node &c = graph.nodes[kids[i]];
            bool same = true;
            for (uint32_t a = 0; a < node.dim; ++a) same = same && c.lo[a] == node.lo[a] && c.hi[a] == node.hi[a];
            if (same) ratio = std::max(ratio, c.ratio);
        }
        break;
    }
    case OP_DIFFERENCE: case OP_DIFFERENCE2: copy_box(node, *child); break;
    case OP_EXTRUDE: case OP_TAPERED_EXTRUDE: {
        copy_box(node, *child);
        node.lo[2] = p[0];
        node.hi[2] = p[1];
        if (node.op == OP_TAPERED_EXTRUDE) {
            // The profile grows by at most max(o, 0); the corrected slab is
            // 1-Lipschitz (outside the z range t is constant), with ratio c0 m.
            const double slope = p[2] / (p[1] - p[0]);
            node.c0 = 1 / std::sqrt(1 + slope * slope);
            const double grow = growth(*child, p[2]);
            for (int a = 0; a < 2; ++a) {
                node.lo[a] -= grow;
                node.hi[a] += grow;
            }
            ratio *= node.c0;
        }
        break;
    }
    case OP_REVOLVE: {
        const double radius = std::max(child->hi[0] + p[0], 0.0);
        set_symmetric(node, radius, radius);
        node.lo[2] = child->lo[1];
        node.hi[2] = child->hi[1];
        break;
    }
    case OP_LOFT: {
        // The profiles' blend changes by |b - a| along z, at most the diagonal
        // D of the box around both profiles inside it (for exact profiles
        // everywhere): c0 = 1 / sqrt(1 + (D / (z1 - z0))^2) keeps the bound.
        const Node &a = graph.nodes[kids[0]], &b = graph.nodes[kids[1]];
        if (!finite_box(a) || !finite_box(b)) throw Error(where + ": LOFT needs bounded profiles");
        double extent[2] = {0, 0};
        for (int i = 0; i < 2; ++i) {
            node.lo[i] = std::min(a.lo[i], b.lo[i]);
            node.hi[i] = std::max(a.hi[i], b.hi[i]);
            const double low = std::min({a.lo[i], a.hi[i], b.lo[i], b.hi[i]});
            const double high = std::max({a.lo[i], a.hi[i], b.lo[i], b.hi[i]});
            extent[i] = high - low;
        }
        node.lo[2] = p[0];
        node.hi[2] = p[1];
        const double slope = std::hypot(extent[0], extent[1]) / (p[1] - p[0]);
        node.c0 = 1 / std::sqrt(1 + slope * slope);
        ratio = std::min(a.ratio, b.ratio) * node.c0;
        // Margin for inexact profiles, whose |b - a| may exceed D outside the box.
        step *= std::sqrt(2.0);
        break;
    }
    case OP_MESH: {  // an exact distance: the box of the vertices, ratio 1
        const TriangleMesh &mesh = graph.meshes[node.mesh];
        std::copy(mesh.lo, mesh.lo + 3, node.lo);
        std::copy(mesh.hi, mesh.hi + 3, node.hi);
        break;
    }
    case OP_CIRCLE: set_symmetric(node, p[0], p[0]); break;
    case OP_RECT: set_symmetric(node, p[0], p[1]); break;
    case OP_POLYGON: {
        const Polygon &polygon = graph.polygons[node.polygon];
        node.lo[0] = *std::min_element(polygon.x.begin(), polygon.x.end());
        node.hi[0] = *std::max_element(polygon.x.begin(), polygon.x.end());
        node.lo[1] = *std::min_element(polygon.y.begin(), polygon.y.end());
        node.hi[1] = *std::max_element(polygon.y.begin(), polygon.y.end());
        break;
    }
    default: break;
    }
    // Ratios only shrink through products; keep them positive so growths
    // stay finite (a vanishing ratio would make bounds unusable anyway).
    node.ratio = std::max(ratio, 1e-300);
    node.lipschitz = lipschitz;
    node.step_lipschitz = std::max(step, lipschitz);
}

// ------------------------------------------------------------------ child trees

// Boolean nodes with at least this many children get a ChildTree.
constexpr uint32_t kChildTreeMinimum = 16;
constexpr uint32_t kChildTreeLeaf = 4;

struct TreeItem {
    uint32_t slot;
    double center[3];
};

uint32_t build_branch(ChildTree &tree, const rsdf_graph &graph, const uint32_t *kids, std::vector<TreeItem> &items,
                      size_t begin, size_t end) {
    const uint32_t index = static_cast<uint32_t>(tree.branches.size());
    tree.branches.emplace_back();
    ChildTree::Branch branch{};
    double center_lo[3] = {kInf, kInf, kInf}, center_hi[3] = {-kInf, -kInf, -kInf};
    for (int a = 0; a < 3; ++a) {
        branch.lo[a] = kInf;
        branch.hi[a] = -kInf;
    }
    branch.ratio = 1;
    for (size_t i = begin; i < end; ++i) {
        const Node &child = graph.nodes[kids[items[i].slot]];
        branch.ratio = std::min(branch.ratio, child.ratio);
        for (int a = 0; a < 3; ++a) {
            branch.lo[a] = std::min(branch.lo[a], child.lo[a]);
            branch.hi[a] = std::max(branch.hi[a], child.hi[a]);
            center_lo[a] = std::min(center_lo[a], items[i].center[a]);
            center_hi[a] = std::max(center_hi[a], items[i].center[a]);
        }
    }
    if (end - begin <= kChildTreeLeaf) {
        branch.first = static_cast<uint32_t>(tree.slots.size());
        branch.count = static_cast<uint32_t>(end - begin);
        for (size_t i = begin; i < end; ++i) tree.slots.push_back(items[i].slot);
    } else {
        // Median split along the widest spread of box centres.
        int axis = 0;
        for (int a = 1; a < 3; ++a)
            if (center_hi[a] - center_lo[a] > center_hi[axis] - center_lo[axis]) axis = a;
        const size_t middle = begin + (end - begin) / 2;
        std::nth_element(items.begin() + begin, items.begin() + middle, items.begin() + end,
                         [axis](const TreeItem &a, const TreeItem &b) {
                             return a.center[axis] < b.center[axis] ||
                                    (a.center[axis] == b.center[axis] && a.slot < b.slot);
                         });
        build_branch(tree, graph, kids, items, begin, middle);
        branch.right = build_branch(tree, graph, kids, items, middle, end);
    }
    tree.branches[index] = branch;
    return index;
}

// Indexes the children after the first of a large UNION or DIFFERENCE.
void build_child_tree(rsdf_graph &graph, Node &node) {
    const bool cullable = node.op == OP_UNION || node.op == OP_DIFFERENCE || node.op == OP_UNION2 ||
                          node.op == OP_DIFFERENCE2;
    if (!cullable || node.child_count < kChildTreeMinimum) return;
    const uint32_t *kids = graph.child_ids(node);
    ChildTree tree;
    std::vector<TreeItem> items;
    for (uint32_t slot = 1; slot < node.child_count; ++slot) {
        const Node &child = graph.nodes[kids[slot]];
        if (!finite_box(child)) {
            tree.always.push_back(slot);
            continue;
        }
        TreeItem item{slot, {}};
        for (int a = 0; a < 3; ++a) item.center[a] = 0.5 * (child.lo[a] + child.hi[a]);
        items.push_back(item);
    }
    if (!items.empty()) build_branch(tree, graph, kids, items, 0, items.size());
    graph.child_trees.push_back(std::move(tree));
    node.child_tree = static_cast<uint32_t>(graph.child_trees.size());
}

std::unique_ptr<rsdf_graph> parse(const uint8_t *bytes, size_t length) {
    if (!bytes) throw Error("RSDF data pointer is null");
    if (length < 16) throw Error(format("RSDF data is truncated: %zu bytes is shorter than the 16-byte header", length));
    if (length > kMaxBytes) throw Error(format("RSDF data is %zu bytes; the limit is %zu bytes", length, kMaxBytes));
    if (std::memcmp(bytes, "RSDF", 4) != 0) throw Error("RSDF data does not start with the \"RSDF\" magic");
    Reader reader{bytes, length, 4};
    const uint32_t version = reader.u32();
    const uint32_t count = reader.u32();
    const uint32_t reserved = reader.u32();
    if (version != 1) throw Error(format("unsupported RSDF version %u (expected 1)", version));
    if (reserved != 0) throw Error(format("RSDF header reserved field is %u (expected 0)", reserved));
    if (count == 0) throw Error("RSDF graph has no nodes");
    if (count > kMaxNodes) throw Error(format("RSDF graph has %u nodes; the limit is %u", count, kMaxNodes));
    if (count > reader.remaining() / 16)
        throw Error(format("RSDF data is truncated: %u nodes need at least %llu bytes of records but %zu remain",
                           count, 16ull * count, reader.remaining()));

    auto graph = std::make_unique<rsdf_graph>();
    graph->nodes.resize(count);
    std::vector<uint32_t> depth(count), live(count);  // nesting levels and live buffers below each node
    std::vector<uint64_t> visits(count);                // node visits per evaluated point, saturating
    for (uint32_t index = 0; index < count; ++index) {
        if (reader.remaining() < 16) throw Error(format("RSDF data is truncated in the record of node %u", index));
        Node &node = graph->nodes[index];
        node.op = reader.u32();
        node.child_count = reader.u32();
        node.param_count = reader.u32();
        const uint32_t node_reserved = reader.u32();
        const OpInfo *info = find_op(node.op);
        if (!info) throw Error(format("node %u: unknown opcode %u", index, node.op));
        const std::string where = format("node %u (%s)", index, info->name);
        node.dim = info->dim;
        if (node_reserved != 0) throw Error(where + format(": reserved field is %u (expected 0)", node_reserved));
        if (node.child_count > kMaxChildren)
            throw Error(where + format(": %u children exceed the limit of %u", node.child_count, kMaxChildren));
        if (node.param_count > (node.op == OP_MESH ? kMaxMeshParams : kMaxParams))
            throw Error(where + format(": %u parameters exceed the limit of %u", node.param_count,
                                       node.op == OP_MESH ? kMaxMeshParams : kMaxParams));
        static const uint32_t kExactChildren[] = {0, 1, 2};
        if (info->children == CHILDREN ? node.child_count == 0 : node.child_count != kExactChildren[info->children])
            throw Error(where + format(": has %u children; expected %s", node.child_count,
                                       info->children == CHILDREN ? "at least 1"
                                       : info->children == NO_CHILDREN ? "0"
                                       : info->children == ONE_CHILD ? "1" : "2"));
        if (info->params == -2 ? false
            : info->params >= 0 ? node.param_count != static_cast<uint32_t>(info->params)
                              : (node.param_count < 6 || node.param_count % 2 != 0))
            throw Error(where + (info->params >= 0
                                     ? format(": has %u parameters; expected %d", node.param_count, info->params)
                                     : format(": has %u parameters; expected an even count of at least 6 "
                                              "(three or more points)", node.param_count)));
        const uint64_t needed = 4ull * node.child_count + 8ull * node.param_count;
        if (reader.remaining() < needed)
            throw Error(where + format(": record is truncated (needs %llu more bytes, %zu remain)",
                                       static_cast<unsigned long long>(needed), reader.remaining()));

        node.first_child = static_cast<uint32_t>(graph->children.size());
        for (uint32_t i = 0; i < node.child_count; ++i) {
            const uint32_t child = reader.u32();
            if (child >= index)
                throw Error(where + format(": child %u refers to node %u, which does not precede it", i, child));
            const uint32_t child_dim = graph->nodes[child].dim;
            if (child_dim != info->child_dim)
                throw Error(where + format(": child %u (node %u) is %uD but must be %uD", i, child, child_dim,
                                           static_cast<uint32_t>(info->child_dim)));
            graph->children.push_back(child);
        }
        node.first_param = graph->params.size();
        for (uint32_t i = 0; i < node.param_count; ++i) {
            const double value = reader.f64();
            if (!std::isfinite(value)) throw Error(where + format(": parameter %u is not finite", i));
            graph->params.push_back(value);
        }
        const double *params = graph->param_values(node);
        validate_params(node, params, Checker(where, params));
        if (node.op == OP_POLYGON) {
            node.polygon = static_cast<uint32_t>(graph->polygons.size());
            graph->polygons.emplace_back();
            try {
                build_polygon(graph->polygons.back(), params, node.param_count / 2);
            } catch (const Error &error) {
                throw Error(where + ": " + error.what());
            }
        }
        if (node.op == OP_MESH) {
            node.mesh = static_cast<uint32_t>(graph->meshes.size());
            graph->meshes.emplace_back();
            try {
                build_triangle_mesh(graph->meshes.back(), params, node.param_count);
            } catch (const Error &error) {
                throw Error(where + ": " + error.what());
            }
        }
        derive(*graph, node, where);
        build_child_tree(*graph, node);
        visits[index] = 1;
        for (uint32_t i = 0; i < node.child_count; ++i) {
            const uint32_t child = graph->children[node.first_child + i];
            depth[index] = std::max(depth[index], depth[child] + 1);
            live[index] = std::max(live[index], live[child] + held_buffers(node.op, i));
            visits[index] = std::min(visits[index] + visits[child], kMaxVisits + 1);
        }
        node.sorts_queries = visits[index] >= 64 || (node.op == OP_POLYGON && node.param_count > 64);
        if (node.op == OP_MESH) node.sorts_queries = true;  // a hierarchy search per point
        for (uint32_t i = 0; i < node.child_count; ++i)
            node.sorts_queries = node.sorts_queries || graph->nodes[graph->children[node.first_child + i]].sorts_queries;
        if (visits[index] > kMaxVisits)
            throw Error(where + format(": evaluating it would visit more than %llu nodes per point (a shared "
                                       "subgraph is evaluated once for every path to it); share less or "
                                       "flatten repeated booleans",
                                       static_cast<unsigned long long>(kMaxVisits)));
        node.depth = depth[index];
        node.live_buffers = live[index];
        if (depth[index] > kMaxDepth)
            throw Error(where + format(": nesting depth exceeds the limit of %u levels", kMaxDepth));
        if (live[index] > kMaxLiveBuffers)
            throw Error(where + format(": nesting keeps more than %u point buffers alive during evaluation",
                                       kMaxLiveBuffers));
    }
    if (reader.remaining() != 0)
        throw Error(format("RSDF data has %zu unexpected trailing bytes after the last node", reader.remaining()));
    return graph;
}

}  // namespace
}  // namespace rsdf

using namespace rsdf;

extern "C" {

rsdf_graph *rsdf_graph_create(const uint8_t *bytes, size_t length) {
    rsdf_graph *graph = nullptr;
    guarded([&] { graph = parse(bytes, length).release(); });
    return graph;
}

void rsdf_graph_free(rsdf_graph *graph) { delete graph; }

uint32_t rsdf_graph_node_count(const rsdf_graph *graph) {
    return graph ? static_cast<uint32_t>(graph->nodes.size()) : 0;
}

int rsdf_node_dimension(const rsdf_graph *graph, uint32_t node) {
    int dimension = -1;
    guarded([&] { dimension = static_cast<int>(checked_node(checked_graph(graph), node).dim); });
    return dimension;
}

int rsdf_bounds(const rsdf_graph *graph, uint32_t node, double *lo, double *hi, double *lipschitz) {
    return guarded([&] {
               const Node &n = checked_node(checked_graph(graph), node);
               for (uint32_t a = 0; a < n.dim; ++a) {
                   if (lo) lo[a] = n.lo[a];
                   if (hi) hi[a] = n.hi[a];
               }
               if (lipschitz) *lipschitz = n.lipschitz;
           })
               ? 0
               : -1;
}

}  // extern "C"
