// Exact signed distance to a closed triangle mesh: the MESH node.
//
// Decoding validates the mesh and prepares it for distance queries:
// - every triangle has three distinct vertices and a nonzero area, and every
//   vertex is used by a triangle;
// - every edge joins exactly two triangles that traverse it in opposite
//   directions (closed, consistently oriented and edge-manifold); an edge of
//   bodies that touch along it, used by more triangles, is paired around the
//   edge so that each body keeps its own surface, and a vertex whose
//   triangles form several fans (surfaces touching at a point) gets a copy
//   per fan;
// - connected shells are oriented: when the shells enclose a negative total
//   volume, every triangle is turned around (an inward mesh); a shell that
//   then encloses a negative volume is a cavity of an outward shell whose
//   solid holds it entirely (see classify_shells), or else is turned around
//   too. Every outward shell with its cavities is a body.
// The field is the union (minimum) of the bodies' exact signed distances, so
// bodies may overlap or touch; a shell must not intersect itself. Outside the
// box of the vertices the field is the (positive) distance.
//
// Distances come from a bounding volume hierarchy whose subtrees are the
// shells. The sign is the side of the angle-weighted pseudonormal of the
// nearest feature (Baerentzen and Aanaes, 2005): the triangle normal for a face
// interior, the sum of both triangles' normals for an edge, and the sum of the
// incident triangles' normals weighted by their angles at a vertex. Every
// nearest point gives the correct side, so a distance tie between features
// cannot flip the sign. With several bodies, a second hierarchy over all
// triangles serves the nearest feature, and a ray's winding numbers tell which
// other bodies hold the point. A query depends on its point alone, not on its
// batch.
#include "kernel.h"

#include <algorithm>
#include <numeric>

namespace rsdf {
namespace {

using Branch = TriangleMesh::Branch;

constexpr uint32_t kMaxTriangles = 2000000;
constexpr uint32_t kMaxVertices = 2000000;
constexpr double kMaxCoordinate = 1e12;
// A triangle whose doubled area is at most this fraction of its longest edge
// squared is degenerate: its corners are (nearly) collinear and its normal is
// meaningless.
constexpr double kFlat = 1e-12;
// A shell enclosing at most this fraction of its box diagonal cubed (rounding
// of the volume sum included) encloses no space, like a sheet whose two sides
// are both triangulated.
constexpr double kEmptyShell = 1e-9;
constexpr uint32_t kLeafTriangles = 4;
constexpr uint32_t kLeafBodies = 4;
constexpr uint32_t kNone = UINT32_MAX;
constexpr int kStack = 128;  // hierarchy depth is at most about 2 log2(triangles) + 2

std::string point_text(const double *p) { return format("(%.6g, %.6g, %.6g)", p[0], p[1], p[2]); }

// Union-find with path halving; the smaller index becomes the root, so roots
// (and everything numbered from them) do not depend on the order of unions.
struct DisjointSets {
    std::vector<uint32_t> parent;
    explicit DisjointSets(size_t count) : parent(count) { std::iota(parent.begin(), parent.end(), 0u); }
    uint32_t find(uint32_t i) {
        while (parent[i] != i) {
            parent[i] = parent[parent[i]];
            i = parent[i];
        }
        return i;
    }
    void unite(uint32_t a, uint32_t b) {
        a = find(a);
        b = find(b);
        if (a != b) parent[std::max(a, b)] = std::min(a, b);
    }
};

inline void cross(const double *a, const double *b, double *out) {
    out[0] = a[1] * b[2] - a[2] * b[1];
    out[1] = a[2] * b[0] - a[0] * b[2];
    out[2] = a[0] * b[1] - a[1] * b[0];
}

inline double dot(const double *a, const double *b) { return a[0] * b[0] + a[1] * b[1] + a[2] * b[2]; }

inline double box_distance2(const double *p, const Branch &box) {
    double sum = 0;
    for (int a = 0; a < 3; ++a) {
        const double gap = std::fmax(std::fmax(box.lo[a] - p[a], p[a] - box.hi[a]), 0.0);
        sum += gap * gap;
    }
    return sum;
}

inline bool box_contains(const double *lo, const double *hi, const double *p) {
    return p[0] >= lo[0] && p[0] <= hi[0] && p[1] >= lo[1] && p[1] <= hi[1] && p[2] >= lo[2] && p[2] <= hi[2];
}

// ------------------------------------------------------------------ queries

// The nearest feature found so far: 0 a face interior, 1 + k the edge from
// corner k to corner k + 1, 4 + k corner k.
struct Nearest {
    double d2 = INFINITY;
    uint32_t triangle = 0;
    int feature = 0;
};

// Squared distance from p to triangle t and its nearest feature (the Voronoi
// regions of Ericson, Real-Time Collision Detection, 5.1.5). Keeps `best`
// when the triangle is not strictly nearer.
inline void test_triangle(const TriangleMesh &m, uint32_t t, const double *p, Nearest &best) {
    const uint32_t *v = &m.triangles[3 * size_t(t)];
    const double *a = &m.positions[3 * size_t(v[0])], *b = &m.positions[3 * size_t(v[1])],
                 *c = &m.positions[3 * size_t(v[2])];
    const double ap[3] = {p[0] - a[0], p[1] - a[1], p[2] - a[2]};
    const double plane = dot(ap, &m.normals[3 * size_t(t)]);
    const double plane2 = plane * plane;
    if (!(plane2 < best.d2)) return;  // the plane is no nearer
    const double ab[3] = {b[0] - a[0], b[1] - a[1], b[2] - a[2]};
    const double ac[3] = {c[0] - a[0], c[1] - a[1], c[2] - a[2]};
    double d2;
    int feature;
    const double s1 = dot(ab, ap), s2 = dot(ac, ap);
    const double bp[3] = {p[0] - b[0], p[1] - b[1], p[2] - b[2]};
    const double s3 = dot(ab, bp), s4 = dot(ac, bp);
    const double cp[3] = {p[0] - c[0], p[1] - c[1], p[2] - c[2]};
    const double s5 = dot(ab, cp), s6 = dot(ac, cp);
    const double vc = s1 * s4 - s3 * s2, vb = s5 * s2 - s1 * s6, va = s3 * s6 - s5 * s4;
    if (s1 <= 0 && s2 <= 0) {
        d2 = dot(ap, ap);
        feature = 4;
    } else if (s3 >= 0 && s4 <= s3) {
        d2 = dot(bp, bp);
        feature = 5;
    } else if (vc <= 0 && s1 >= 0 && s3 <= 0) {
        const double u = s1 / (s1 - s3);
        const double e[3] = {ap[0] - u * ab[0], ap[1] - u * ab[1], ap[2] - u * ab[2]};
        d2 = dot(e, e);
        feature = 1;
    } else if (s6 >= 0 && s5 <= s6) {
        d2 = dot(cp, cp);
        feature = 6;
    } else if (vb <= 0 && s2 >= 0 && s6 <= 0) {
        const double u = s2 / (s2 - s6);
        const double e[3] = {ap[0] - u * ac[0], ap[1] - u * ac[1], ap[2] - u * ac[2]};
        d2 = dot(e, e);
        feature = 3;
    } else if (va <= 0 && s4 - s3 >= 0 && s5 - s6 >= 0) {
        const double u = (s4 - s3) / ((s4 - s3) + (s5 - s6));
        const double bc[3] = {c[0] - b[0], c[1] - b[1], c[2] - b[2]};
        const double e[3] = {bp[0] - u * bc[0], bp[1] - u * bc[1], bp[2] - u * bc[2]};
        d2 = dot(e, e);
        feature = 2;
    } else {
        d2 = plane2;
        feature = 0;
    }
    if (d2 < best.d2) best = Nearest{d2, t, feature};
}

// The nearest triangle of the subtree at `root` of `tree` (whose leaves index
// `order` when given, else the triangles), by nearest-first branch and bound;
// `best` may already hold a nearer feature of another subtree.
void nearest_in(const TriangleMesh &m, const std::vector<Branch> &tree, const uint32_t *order, uint32_t root,
                const double *p, Nearest &best) {
    struct Entry {
        uint32_t node;
        double d2;
    };
    Entry stack[kStack];
    int top = 0;
    stack[top++] = Entry{root, box_distance2(p, tree[root])};
    while (top > 0) {
        const Entry entry = stack[--top];
        if (!(entry.d2 < best.d2)) continue;
        const Branch &node = tree[entry.node];
        if (node.right == 0) {
            for (uint32_t i = node.first; i < node.first + node.count; ++i) test_triangle(m, order ? order[i] : i, p, best);
            continue;
        }
        const uint32_t left = entry.node + 1, right = node.right;
        const double dl = box_distance2(p, tree[left]), dr = box_distance2(p, tree[right]);
        const Entry nearer = dl <= dr ? Entry{left, dl} : Entry{right, dr};
        const Entry farther = dl <= dr ? Entry{right, dr} : Entry{left, dl};
        if (farther.d2 < best.d2) stack[top++] = farther;
        if (nearer.d2 < best.d2) stack[top++] = nearer;
    }
}

// The nearest triangle of the shell hierarchy's subtree at `root`.
inline void nearest(const TriangleMesh &m, uint32_t root, const double *p, Nearest &best) {
    nearest_in(m, m.tree, nullptr, root, p, best);
}

// Whether p lies inside, judged by the pseudonormal of its nearest feature.
inline bool inside(const TriangleMesh &m, const double *p, const Nearest &best) {
    const size_t t = best.triangle;
    const uint32_t *v = &m.triangles[3 * t];
    double normal[3];
    const double *q;
    if (best.feature == 0) {
        std::copy(&m.normals[3 * t], &m.normals[3 * t] + 3, normal);
        q = &m.positions[3 * size_t(v[0])];
    } else if (best.feature <= 3) {
        const int k = best.feature - 1;
        const double *n1 = &m.normals[3 * t], *n2 = &m.normals[3 * size_t(m.neighbours[3 * t + k])];
        for (int a = 0; a < 3; ++a) normal[a] = n1[a] + n2[a];
        q = &m.positions[3 * size_t(v[k])];  // the edge's direction is normal to both triangles' normals
    } else {
        const size_t vertex = v[best.feature - 4];
        std::copy(&m.vertex_normals[3 * vertex], &m.vertex_normals[3 * vertex] + 3, normal);
        q = &m.positions[3 * vertex];
    }
    const double d[3] = {p[0] - q[0], p[1] - q[1], p[2] - q[2]};
    return dot(d, normal) < 0;
}

inline double signed_value(const TriangleMesh &m, const double *p, const Nearest &best) {
    if (!(best.d2 > 0)) return best.d2 == 0 ? 0.0 : best.d2;  // on the surface (or NaN input)
    const double distance = std::sqrt(best.d2);
    return inside(m, p, best) ? -distance : distance;
}

// The signed distance of body `body` alone.
double body_value(const TriangleMesh &m, uint32_t body, const double *p) {
    const TriangleMesh::Body &b = m.bodies[body];
    Nearest best;
    for (uint32_t i = b.first_root; i < b.first_root + b.root_count; ++i) nearest(m, m.body_roots[i], p, best);
    return signed_value(m, p, best);
}

// (a - p) x (b - p) in (y, z), exactly opposite for b, a: both products are
// rounded on their own (a fused multiply-add would round the two orders
// differently).
inline double ray_cross(const double *a, const double *b, const double *p) {
#pragma clang fp contract(off)
    const double first = (a[1] - p[1]) * (b[2] - p[2]);
    const double second = (a[2] - p[2]) * (b[1] - p[1]);
    return first - second;
}

// The side of the ray from p along +x that edge ab passes in the plane x = 0
// (the projection along the ray is exact): the sign of ray_cross, exactly
// opposite for the edge ba of the neighbouring triangle. A ray through the
// edge's line counts as if p had moved by (0, e, e^2), so it passes on one
// definite side.
inline int ray_side(const double *a, const double *b, const double *p) {
    const double e = ray_cross(a, b, p);
    if (e != 0) return e > 0 ? 1 : -1;
    if (a[2] != b[2]) return a[2] > b[2] ? 1 : -1;
    if (a[1] != b[1]) return b[1] > a[1] ? 1 : -1;
    return 0;
}

// Whether the ray from p along +x crosses triangle t: +1 leaving through a
// triangle facing +x, -1 entering through one facing -x, 0 missing it. The
// projected tests are exact up to the sign of each edge's cross product, so a
// ray never passes between neighbouring triangles or counts both; whether the
// crossing lies ahead of p is decided by rounding only within rounding of the
// surface.
inline int ray_crossing(const TriangleMesh &m, uint32_t t, const double *p) {
    const uint32_t *v = &m.triangles[3 * size_t(t)];
    const double *a = &m.positions[3 * size_t(v[0])], *b = &m.positions[3 * size_t(v[1])],
                 *c = &m.positions[3 * size_t(v[2])];
    const int side = ray_side(a, b, p);
    if (side == 0 || ray_side(b, c, p) != side || ray_side(c, a, p) != side) return 0;
    const double wa = ray_cross(b, c, p), wb = ray_cross(c, a, p), wc = ray_cross(a, b, p);
    const double ahead = wa * (a[0] - p[0]) + wb * (b[0] - p[0]) + wc * (c[0] - p[0]);  // barycentric, times the area
    return side * ahead > 0 ? side : 0;
}

constexpr int kMaxCrossedBodies = 32;

// The winding number about p of the surface of every body that the ray from
// p along +x crosses (1 for a body holding p, 0 otherwise). Returns false
// when it crosses more than kMaxCrossedBodies bodies.
struct Windings {
    int count = 0;
    uint32_t body[kMaxCrossedBodies];
    int winding[kMaxCrossedBodies];
};

bool ray_windings(const TriangleMesh &m, const double *p, Windings &w) {
    uint32_t stack[kStack];
    int top = 0;
    stack[top++] = 0;
    while (top > 0) {
        const uint32_t index = stack[--top];
        const Branch &node = m.mixed[index];
        if (!(node.hi[0] >= p[0] && node.lo[1] <= p[1] && p[1] <= node.hi[1] && node.lo[2] <= p[2] &&
              p[2] <= node.hi[2]))
            continue;
        if (node.right != 0) {
            stack[top++] = node.right;
            stack[top++] = index + 1;
            continue;
        }
        for (uint32_t i = node.first; i < node.first + node.count; ++i) {
            const uint32_t t = m.mixed_order[i];
            const int crossing = ray_crossing(m, t, p);
            if (crossing == 0) continue;
            const uint32_t body = m.triangle_body[t];
            int k = 0;
            while (k < w.count && w.body[k] != body) ++k;
            if (k == w.count) {
                if (w.count == kMaxCrossedBodies) return false;
                w.body[w.count] = body;
                w.winding[w.count++] = 0;
            }
            w.winding[k] += crossing;
        }
    }
    return true;
}

// Lowers `value` (the signed distance of the body nearest to p) to the union
// of every body. Only a body that holds p can lower it (all surfaces are at
// least |value| away): those the ray from p crosses with a nonzero winding
// number, or without the hierarchy over all triangles or when the ray crosses
// too many bodies, those whose boxes hold p.
double union_of_bodies(const TriangleMesh &m, const double *p, uint32_t nearest_body, double value) {
    Windings w;
    if (!m.mixed.empty() && ray_windings(m, p, w)) {
        for (int k = 0; k < w.count; ++k)
            if (w.winding[k] != 0 && w.body[k] != nearest_body) value = std::fmin(value, body_value(m, w.body[k], p));
        return value;
    }
    uint32_t stack[kStack];
    int top = 0;
    stack[top++] = 0;
    while (top > 0) {
        const uint32_t index = stack[--top];
        const Branch &node = m.body_tree[index];
        if (!box_contains(node.lo, node.hi, p)) continue;
        if (node.right != 0) {
            stack[top++] = node.right;
            stack[top++] = index + 1;
            continue;
        }
        for (uint32_t i = node.first; i < node.first + node.count; ++i) {
            const uint32_t body = m.body_order[i];
            if (body == nearest_body || !box_contains(m.bodies[body].lo, m.bodies[body].hi, p)) continue;
            value = std::fmin(value, body_value(m, body, p));
        }
    }
    return value;
}

}  // namespace

void eval_triangle_mesh(const TriangleMesh &m, int n, const double *x, const double *y, const double *z,
                        double *out) {
    for (int i = 0; i < n; ++i) {
        const double p[3] = {x[i], y[i], z[i]};
        Nearest best;
        if (m.mixed.empty())
            nearest(m, 0, p, best);
        else
            nearest_in(m, m.mixed, m.mixed_order.data(), 0, p, best);
        if (!box_contains(m.lo, m.hi, p)) {  // the solid lies within the vertices' box
            out[i] = std::sqrt(best.d2);
            continue;
        }
        double value = signed_value(m, p, best);
        if (m.bodies.size() > 1 && !std::isnan(value))
            value = union_of_bodies(m, p, m.triangle_body[best.triangle], value);
        out[i] = value;
    }
}

// ------------------------------------------------------------------ decoding

namespace {

uint32_t count_parameter(double value, const char *what, uint32_t limit) {
    if (!(value >= 0 && value <= limit && value == std::floor(value)))
        throw Error(format("the %s count must be a whole number of at most %u (got %.17g)", what, limit, value));
    return static_cast<uint32_t>(value);
}

inline size_t next_slot(size_t slot) { return slot % 3 == 2 ? slot - 2 : slot + 1; }

// Parameters: vertex count V, triangle count T, V vertices (x y z), then T
// triangles (three vertex indices each, whole numbers).
void read_mesh(TriangleMesh &m, const double *params, size_t param_count, uint32_t &vertex_count) {
    if (param_count < 2) throw Error("parameters must start with the vertex and triangle counts");
    vertex_count = count_parameter(params[0], "vertex", kMaxVertices);
    const uint32_t triangle_count = count_parameter(params[1], "triangle", kMaxTriangles);
    const uint64_t needed = 2 + 3ull * vertex_count + 3ull * triangle_count;
    if (param_count != needed)
        throw Error(format("has %zu parameters; %u vertices and %u triangles need %llu", param_count, vertex_count,
                           triangle_count, static_cast<unsigned long long>(needed)));
    if (triangle_count < 4)
        throw Error(format("a closed mesh needs at least four triangles (got %u)", triangle_count));
    const double *coords = params + 2;
    m.positions.assign(coords, coords + 3 * size_t(vertex_count));
    for (size_t i = 0; i < m.positions.size(); ++i)
        if (!(std::fabs(m.positions[i]) <= kMaxCoordinate))
            throw Error(format("vertex %zu has a coordinate beyond 1e12 in magnitude", i / 3));
    const double *indices = coords + 3 * size_t(vertex_count);
    m.triangles.resize(3 * size_t(triangle_count));
    for (size_t i = 0; i < m.triangles.size(); ++i) {
        const double value = indices[i];
        if (!(value >= 0 && value < vertex_count && value == std::floor(value)))
            throw Error(format("triangle %zu refers to vertex %.17g, which is not one of the %u vertices", i / 3,
                               value, vertex_count));
        m.triangles[i] = static_cast<uint32_t>(value);
    }
}

void check_triangles(const TriangleMesh &m, uint32_t vertex_count) {
    const size_t count = m.triangles.size() / 3;
    std::vector<uint8_t> used(vertex_count, 0);
    size_t degenerate = 0, example = 0;
    for (size_t t = 0; t < count; ++t) {
        const uint32_t *v = &m.triangles[3 * t];
        for (int k = 0; k < 3; ++k) used[v[k]] = 1;
        bool flat = v[0] == v[1] || v[1] == v[2] || v[2] == v[0];
        if (!flat) {
            const double *a = &m.positions[3 * size_t(v[0])], *b = &m.positions[3 * size_t(v[1])],
                         *c = &m.positions[3 * size_t(v[2])];
            const double ab[3] = {b[0] - a[0], b[1] - a[1], b[2] - a[2]};
            const double ac[3] = {c[0] - a[0], c[1] - a[1], c[2] - a[2]};
            const double bc[3] = {c[0] - b[0], c[1] - b[1], c[2] - b[2]};
            double n[3];
            cross(ab, ac, n);
            const double longest = std::max({dot(ab, ab), dot(ac, ac), dot(bc, bc)});
            flat = !(std::sqrt(dot(n, n)) > kFlat * longest);
        }
        if (flat && degenerate++ == 0) example = t;
    }
    if (degenerate) {
        double center[3] = {0, 0, 0};
        for (int k = 0; k < 3; ++k)
            for (int a = 0; a < 3; ++a) center[a] += m.positions[3 * size_t(m.triangles[3 * example + k]) + a] / 3;
        throw Error(format("%zu triangle%s degenerate (a repeated corner or zero area), e.g. triangle %zu near %s; "
                           "remove degenerate triangles or export the mesh again",
                           degenerate, degenerate == 1 ? " is" : "s are", example, point_text(center).c_str()));
    }
    for (uint32_t v = 0; v < vertex_count; ++v)
        if (!used[v]) throw Error(format("vertex %u is not used by any triangle", v));
}

std::string edge_text(const TriangleMesh &m, size_t slot) {
    return "the edge from " + point_text(&m.positions[3 * size_t(m.triangles[slot])]) + " to " +
           point_text(&m.positions[3 * size_t(m.triangles[next_slot(slot)])]);
}

// Six times the signed volume enclosed by all triangles, about the first
// vertex: positive for a mesh facing outward as a whole.
double soup_volume(const TriangleMesh &m) {
    const double *o = &m.positions[0];
    double total = 0;
    for (size_t t = 0; t < m.triangles.size() / 3; ++t) {
        double corner[3][3], side[3];
        for (int k = 0; k < 3; ++k)
            for (int a = 0; a < 3; ++a) corner[k][a] = m.positions[3 * size_t(m.triangles[3 * t + k]) + a] - o[a];
        cross(corner[1], corner[2], side);
        total += dot(corner[0], side);
    }
    return total;
}

// Pairs the uses `slots[0, count)` of one edge shared by more than two
// triangles, as left by bodies that touch along it. Turning about the edge,
// every triangle either opens a wedge of solid (the solid lies ahead of it)
// or closes one; the uses must alternate between the two, and each opening
// use pairs with the next use, which closes its wedge, so every body keeps
// its own surface. Returns false when the uses do not alternate (surfaces
// that cross, or inconsistent orientations). `outward` tells how the mesh
// faces as a whole.
bool pair_around_edge(const TriangleMesh &m, const uint32_t *slots, size_t count, bool outward,
                      std::vector<uint32_t> &partner) {
    if (count % 2) return false;
    const uint32_t from = m.triangles[slots[0]], to = m.triangles[next_slot(slots[0])];
    const uint32_t lo = std::min(from, to), hi = std::max(from, to);
    const double *a = &m.positions[3 * size_t(lo)], *b = &m.positions[3 * size_t(hi)];
    double axis[3] = {b[0] - a[0], b[1] - a[1], b[2] - a[2]};
    const double length = std::sqrt(dot(axis, axis));
    for (double &value : axis) value /= length;
    struct Use {
        double angle;
        bool closes;
        uint32_t slot;
    };
    std::vector<Use> uses(count);
    double u[3] = {0, 0, 0}, v[3] = {0, 0, 0};
    for (size_t i = 0; i < count; ++i) {
        const uint32_t s = slots[i];
        const double *c = &m.positions[3 * size_t(m.triangles[next_slot(next_slot(s))])];
        const double ac[3] = {c[0] - a[0], c[1] - a[1], c[2] - a[2]};
        const double along = dot(ac, axis);
        const double w[3] = {ac[0] - along * axis[0], ac[1] - along * axis[1], ac[2] - along * axis[2]};
        if (i == 0) {
            const double size = std::sqrt(dot(w, w));
            for (int k = 0; k < 3; ++k) u[k] = w[k] / size;
            cross(axis, u, v);
        }
        // A use running from the lower vertex index to the higher has the
        // edge's direction, so an outward triangle's normal (axis x w) points
        // ahead and its solid lies behind it: it closes a wedge.
        uses[i] = Use{std::atan2(dot(w, v) + 0.0, dot(w, u)), (m.triangles[s] == lo) == outward, s};
    }
    std::sort(uses.begin(), uses.end(), [](const Use &x, const Use &y) {
        return x.angle < y.angle || (x.angle == y.angle && x.slot < y.slot);
    });
    // Start after the widest gap, so that nearly equal angles stay together,
    // and put a closing use before an opening one at (nearly) equal angles:
    // coincident faces of touching bodies close one wedge, then open the next.
    constexpr double kTwoPi = 6.283185307179586;
    size_t start = 0;
    double widest = -1;
    for (size_t i = 0; i < count; ++i) {
        const double gap = (i + 1 < count ? uses[i + 1].angle : uses[0].angle + kTwoPi) - uses[i].angle;
        if (gap > widest) {
            widest = gap;
            start = (i + 1) % count;
        }
    }
    std::rotate(uses.begin(), uses.begin() + static_cast<std::ptrdiff_t>(start), uses.end());
    for (size_t i = 1; i < count; ++i)
        if (uses[i].angle < uses[i - 1].angle) uses[i].angle += kTwoPi;
    constexpr double kSameAngle = 1e-9;
    for (size_t i = 0; i < count;) {
        size_t j = i + 1;
        while (j < count && uses[j].angle - uses[j - 1].angle <= kSameAngle) ++j;
        std::sort(uses.begin() + static_cast<std::ptrdiff_t>(i), uses.begin() + static_cast<std::ptrdiff_t>(j),
                  [](const Use &x, const Use &y) { return x.closes > y.closes || (x.closes == y.closes && x.slot < y.slot); });
        i = j;
    }
    for (size_t i = 0; i < count; ++i)
        if (uses[i].closes == uses[(i + 1) % count].closes) return false;
    for (size_t i = 0; i < count; ++i) {
        if (uses[i].closes) continue;
        const uint32_t opens = uses[i].slot, closes = uses[(i + 1) % count].slot;
        partner[opens] = closes;
        partner[closes] = opens;
    }
    return true;
}

// The opposite directed edge of every directed edge (slot 3t + k runs from
// corner k of triangle t to corner k + 1). Every undirected edge must have
// exactly two uses in opposite directions, or more uses that pair around the
// edge (bodies touching along it, see pair_around_edge).
std::vector<uint32_t> pair_edges(const TriangleMesh &m, bool outward) {
    const size_t slots = m.triangles.size();
    std::vector<uint64_t> keys(slots);
    std::vector<uint32_t> order(slots);
    for (size_t s = 0; s < slots; ++s) {
        const uint32_t a = m.triangles[s], b = m.triangles[next_slot(s)];
        keys[s] = uint64_t(std::min(a, b)) << 21 | std::max(a, b);  // vertex indices < 2^21
        order[s] = static_cast<uint32_t>(s);
    }
    radix_sort(keys, &order, 42);
    std::vector<uint32_t> partner(slots, kNone);
    size_t open = 0, crowded = 0, turned = 0, open_at = 0, crowded_at = 0, turned_at = 0;
    for (size_t i = 0; i < slots;) {
        size_t j = i + 1;
        while (j < slots && keys[j] == keys[i]) ++j;
        if (j - i == 1) {
            if (open++ == 0) open_at = order[i];
        } else if (j - i > 2) {
            if (!pair_around_edge(m, &order[i], j - i, outward, partner) && crowded++ == 0) crowded_at = order[i];
        } else if (m.triangles[order[i]] == m.triangles[order[i + 1]]) {  // both start at the same vertex
            if (turned++ == 0) turned_at = order[i];
        } else {
            partner[order[i]] = order[i + 1];
            partner[order[i + 1]] = order[i];
        }
        i = j;
    }
    if (open || crowded || turned) {
        std::string problems;
        const auto add = [&](size_t count, const char *one, const char *many, size_t slot, const char *remedy) {
            if (!count) return;
            if (!problems.empty()) problems += "; ";
            problems += format("%zu %s (e.g. %s); %s", count, count == 1 ? one : many, edge_text(m, slot).c_str(),
                               remedy);
        };
        add(open, "edge belongs to only one triangle", "edges belong to only one triangle", open_at,
            "close the holes");
        add(crowded, "edge is shared by more than two triangles that do not pair up around it",
            "edges are shared by more than two triangles that do not pair up around them", crowded_at,
            "remove duplicated or crossing surfaces");
        add(turned, "edge joins triangles that face opposite ways", "edges join triangles that face opposite ways",
            turned_at, "orient all triangles consistently");
        throw Error("the mesh is not a closed, consistently oriented manifold: " + problems);
    }
    return partner;
}

// Corners at a vertex are joined across the edges they share into fans. A
// vertex whose triangles form several fans (surfaces touching at a point, or
// bodies touching along edges) gets a copy for every fan after its first, so
// each fan has its own vertex. Returns the new vertex count.
uint32_t split_vertex_fans(TriangleMesh &m, const std::vector<uint32_t> &partner, uint32_t vertex_count) {
    const size_t slots = partner.size();
    DisjointSets corners(slots);
    for (size_t s = 0; s < slots; ++s) {
        const size_t o = partner[s];
        if (o < s) continue;
        // s runs a -> b and o runs b -> a: corner s and the corner after o lie
        // at a, the corner after s and corner o at b.
        corners.unite(static_cast<uint32_t>(s), static_cast<uint32_t>(next_slot(o)));
        corners.unite(static_cast<uint32_t>(next_slot(s)), static_cast<uint32_t>(o));
    }
    std::vector<uint32_t> fan(vertex_count, kNone);
    bool split = false;
    for (size_t s = 0; s < slots && !split; ++s) {
        const uint32_t v = m.triangles[s], root = corners.find(static_cast<uint32_t>(s));
        if (fan[v] == kNone) fan[v] = root;
        split = fan[v] != root;
    }
    if (!split) return vertex_count;
    std::fill(fan.begin(), fan.end(), kNone);
    std::vector<uint32_t> copy(slots, kNone);  // by fan root
    for (size_t s = 0; s < slots; ++s) {
        const uint32_t v = m.triangles[s], root = corners.find(static_cast<uint32_t>(s));
        if (fan[v] == kNone) fan[v] = root;
        if (fan[v] == root) continue;
        if (copy[root] == kNone) {
            copy[root] = vertex_count++;
            const double p[3] = {m.positions[3 * size_t(v)], m.positions[3 * size_t(v) + 1], m.positions[3 * size_t(v) + 2]};
            m.positions.insert(m.positions.end(), p, p + 3);
        }
        m.triangles[s] = copy[root];
    }
    return vertex_count;
}

inline void flip_triangle(TriangleMesh &m, size_t t) {
    std::swap(m.triangles[3 * t + 1], m.triangles[3 * t + 2]);
    std::swap(m.neighbours[3 * t], m.neighbours[3 * t + 2]);  // edges 0 and 2 trade places
    if (!m.normals.empty())
        for (int a = 0; a < 3; ++a) m.normals[3 * t + a] = -m.normals[3 * t + a];
}

// Median-split hierarchies: one subtree per shell, joined by a hierarchy over
// the shells' boxes, so a query can cover one shell or the whole mesh.
class TreeBuilder {
public:
    TreeBuilder(TriangleMesh &m, std::vector<uint32_t> &order, const std::vector<double> &centroids,
                const std::vector<uint32_t> &first, const std::vector<double> &boxes)
        : m_(m), tree_(m.tree), order_(order), centroids_(centroids), first_(first), boxes_(boxes) {}
    // A builder of triangles(...) alone into `tree`.
    TreeBuilder(TriangleMesh &m, std::vector<Branch> &tree, std::vector<uint32_t> &order,
                const std::vector<double> &centroids)
        : m_(m), tree_(tree), order_(order), centroids_(centroids), first_(order), boxes_(centroids) {}

    // The subtree of the shells ids[begin, end); records each shell's root.
    uint32_t shells(std::vector<uint32_t> &ids, size_t begin, size_t end, std::vector<uint32_t> &roots, int depth) {
        if (end - begin == 1) {
            const uint32_t shell = ids[begin];
            return roots[shell] = triangles(first_[shell], first_[shell + 1], depth);
        }
        check(depth);
        const uint32_t index = static_cast<uint32_t>(tree_.size());
        tree_.emplace_back();
        Branch node{};
        double lo[3] = {INFINITY, INFINITY, INFINITY}, hi[3] = {-INFINITY, -INFINITY, -INFINITY};
        empty(node);
        for (size_t i = begin; i < end; ++i) {
            const double *box = &boxes_[6 * size_t(ids[i])];
            for (int a = 0; a < 3; ++a) {
                node.lo[a] = std::fmin(node.lo[a], box[a]);
                node.hi[a] = std::fmax(node.hi[a], box[3 + a]);
                lo[a] = std::fmin(lo[a], box[a] + box[3 + a]);
                hi[a] = std::fmax(hi[a], box[a] + box[3 + a]);
            }
        }
        const int axis = widest(lo, hi);
        const size_t middle = begin + (end - begin) / 2;
        std::nth_element(ids.begin() + begin, ids.begin() + middle, ids.begin() + end, [&](uint32_t a, uint32_t b) {
            const double ca = boxes_[6 * size_t(a) + axis] + boxes_[6 * size_t(a) + 3 + axis];
            const double cb = boxes_[6 * size_t(b) + axis] + boxes_[6 * size_t(b) + 3 + axis];
            return ca < cb || (ca == cb && a < b);
        });
        shells(ids, begin, middle, roots, depth + 1);
        node.right = shells(ids, middle, end, roots, depth + 1);
        tree_[index] = node;
        return index;
    }

    // The subtree of the triangles order_[begin, end).
    uint32_t triangles(uint32_t begin, uint32_t end, int depth) {
        check(depth);
        const uint32_t index = static_cast<uint32_t>(tree_.size());
        tree_.emplace_back();
        Branch node{};
        double lo[3] = {INFINITY, INFINITY, INFINITY}, hi[3] = {-INFINITY, -INFINITY, -INFINITY};
        empty(node);
        for (uint32_t i = begin; i < end; ++i) {
            const size_t t = order_[i];
            for (int k = 0; k < 3; ++k) {
                const double *p = &m_.positions[3 * size_t(m_.triangles[3 * t + k])];
                for (int a = 0; a < 3; ++a) {
                    node.lo[a] = std::fmin(node.lo[a], p[a]);
                    node.hi[a] = std::fmax(node.hi[a], p[a]);
                }
            }
            for (int a = 0; a < 3; ++a) {
                lo[a] = std::fmin(lo[a], centroids_[3 * t + a]);
                hi[a] = std::fmax(hi[a], centroids_[3 * t + a]);
            }
        }
        if (end - begin <= kLeafTriangles) {
            node.first = begin;
            node.count = end - begin;
        } else {
            const int axis = widest(lo, hi);
            const uint32_t middle = begin + (end - begin) / 2;
            std::nth_element(order_.begin() + begin, order_.begin() + middle, order_.begin() + end,
                             [&](uint32_t a, uint32_t b) {
                                 const double ca = centroids_[3 * size_t(a) + axis], cb = centroids_[3 * size_t(b) + axis];
                                 return ca < cb || (ca == cb && a < b);
                             });
            triangles(begin, middle, depth + 1);
            node.right = triangles(middle, end, depth + 1);
        }
        tree_[index] = node;
        return index;
    }

private:
    static void empty(Branch &node) {
        for (int a = 0; a < 3; ++a) {
            node.lo[a] = INFINITY;
            node.hi[a] = -INFINITY;
        }
    }
    static int widest(const double *lo, const double *hi) {
        int axis = 0;
        for (int a = 1; a < 3; ++a)
            if (hi[a] - lo[a] > hi[axis] - lo[axis]) axis = a;
        return axis;
    }
    static void check(int depth) {
        // Median splits halve every range: shells and triangles each add at
        // most about 21 levels within the limits.
        if (depth >= kStack / 2 - 2) throw Error("mesh hierarchy is too deep");
    }

    TriangleMesh &m_;
    std::vector<Branch> &tree_;
    std::vector<uint32_t> &order_;
    const std::vector<double> &centroids_;
    const std::vector<uint32_t> &first_;
    const std::vector<double> &boxes_;
};

// A median-split hierarchy over the bodies' boxes (leaves index body_order).
uint32_t build_body_tree(TriangleMesh &m, uint32_t begin, uint32_t end) {
    const uint32_t index = static_cast<uint32_t>(m.body_tree.size());
    m.body_tree.emplace_back();
    Branch node{};
    double lo[3] = {INFINITY, INFINITY, INFINITY}, hi[3] = {-INFINITY, -INFINITY, -INFINITY};
    for (int a = 0; a < 3; ++a) {
        node.lo[a] = INFINITY;
        node.hi[a] = -INFINITY;
    }
    for (uint32_t i = begin; i < end; ++i) {
        const TriangleMesh::Body &body = m.bodies[m.body_order[i]];
        for (int a = 0; a < 3; ++a) {
            node.lo[a] = std::fmin(node.lo[a], body.lo[a]);
            node.hi[a] = std::fmax(node.hi[a], body.hi[a]);
            lo[a] = std::fmin(lo[a], body.lo[a] + body.hi[a]);
            hi[a] = std::fmax(hi[a], body.lo[a] + body.hi[a]);
        }
    }
    if (end - begin <= kLeafBodies) {
        node.first = begin;
        node.count = end - begin;
    } else {
        int axis = 0;
        for (int a = 1; a < 3; ++a)
            if (hi[a] - lo[a] > hi[axis] - lo[axis]) axis = a;
        const uint32_t middle = begin + (end - begin) / 2;
        std::nth_element(m.body_order.begin() + begin, m.body_order.begin() + middle, m.body_order.begin() + end,
                         [&](uint32_t a, uint32_t b) {
                             const double ca = m.bodies[a].lo[axis] + m.bodies[a].hi[axis];
                             const double cb = m.bodies[b].lo[axis] + m.bodies[b].hi[axis];
                             return ca < cb || (ca == cb && a < b);
                         });
        build_body_tree(m, begin, middle);
        node.right = build_body_tree(m, middle, end);
    }
    m.body_tree[index] = node;
    return index;
}

// Unit triangle normals and angle-weighted vertex pseudonormals.
void compute_normals(TriangleMesh &m) {
    const size_t count = m.triangles.size() / 3;
    m.normals.resize(3 * count);
    m.vertex_normals.assign(m.positions.size(), 0.0);
    for (size_t t = 0; t < count; ++t) {
        const uint32_t *v = &m.triangles[3 * t];
        const double *p[3] = {&m.positions[3 * size_t(v[0])], &m.positions[3 * size_t(v[1])],
                              &m.positions[3 * size_t(v[2])]};
        const double ab[3] = {p[1][0] - p[0][0], p[1][1] - p[0][1], p[1][2] - p[0][2]};
        const double ac[3] = {p[2][0] - p[0][0], p[2][1] - p[0][1], p[2][2] - p[0][2]};
        double *n = &m.normals[3 * t];
        cross(ab, ac, n);
        const double length = std::sqrt(dot(n, n));
        for (int a = 0; a < 3; ++a) n[a] /= length;
        for (int k = 0; k < 3; ++k) {
            const double *o = p[k], *b = p[(k + 1) % 3], *c = p[(k + 2) % 3];
            const double e1[3] = {b[0] - o[0], b[1] - o[1], b[2] - o[2]};
            const double e2[3] = {c[0] - o[0], c[1] - o[1], c[2] - o[2]};
            double side[3];
            cross(e1, e2, side);
            const double angle = std::atan2(std::sqrt(dot(side, side)), dot(e1, e2));
            for (int a = 0; a < 3; ++a) m.vertex_normals[3 * size_t(v[k]) + a] += angle * n[a];
        }
    }
}

// ------------------------------------------------------------------ shell nesting

// (b - a) x (c - a) . (d - a): positive when d lies on the side of plane abc
// that its counter-clockwise normal faces.
inline double orient3(const double *a, const double *b, const double *c, const double *d) {
    const double ab[3] = {b[0] - a[0], b[1] - a[1], b[2] - a[2]};
    const double ac[3] = {c[0] - a[0], c[1] - a[1], c[2] - a[2]};
    const double ad[3] = {d[0] - a[0], d[1] - a[1], d[2] - a[2]};
    double n[3];
    cross(ab, ac, n);
    return dot(n, ad);
}

inline double orient2(const double *a, const double *b, const double *c) {
    return (b[0] - a[0]) * (c[1] - a[1]) - (b[1] - a[1]) * (c[0] - a[0]);
}

// Whether the closed segments pq and rs of a plane meet (2D points).
bool segments_meet2(const double *p, const double *q, const double *r, const double *s) {
    const double d1 = orient2(r, s, p), d2 = orient2(r, s, q), d3 = orient2(p, q, r), d4 = orient2(p, q, s);
    if (((d1 > 0 && d2 < 0) || (d1 < 0 && d2 > 0)) && ((d3 > 0 && d4 < 0) || (d3 < 0 && d4 > 0))) return true;
    // A collinear endpoint meets the other segment when it lies within its box.
    const auto within = [](const double *a, const double *b, const double *x) {
        return std::fmin(a[0], b[0]) <= x[0] && x[0] <= std::fmax(a[0], b[0]) && std::fmin(a[1], b[1]) <= x[1] &&
               x[1] <= std::fmax(a[1], b[1]);
    };
    return (d1 == 0 && within(r, s, p)) || (d2 == 0 && within(r, s, q)) || (d3 == 0 && within(p, q, r)) ||
           (d4 == 0 && within(p, q, s));
}

// Whether the closed segment pq meets the closed triangle abc, up to rounding.
bool segment_meets_triangle(const double *p, const double *q, const double *a, const double *b, const double *c) {
    const double sp = orient3(a, b, c, p), sq = orient3(a, b, c, q);
    if ((sp > 0 && sq > 0) || (sp < 0 && sq < 0)) return false;
    if (sp == 0 && sq == 0) {  // in the triangle's plane: compare them in 2D
        const double ab[3] = {b[0] - a[0], b[1] - a[1], b[2] - a[2]};
        const double ac[3] = {c[0] - a[0], c[1] - a[1], c[2] - a[2]};
        double n[3];
        cross(ab, ac, n);
        int drop = 0;
        for (int k = 1; k < 3; ++k)
            if (std::fabs(n[k]) > std::fabs(n[drop])) drop = k;
        const int i = drop == 0 ? 1 : 0, j = drop == 2 ? 1 : 2;
        const double P[2] = {p[i], p[j]}, Q[2] = {q[i], q[j]}, A[2] = {a[i], a[j]}, B[2] = {b[i], b[j]},
                     C[2] = {c[i], c[j]};
        for (const double *x : {P, Q}) {
            const double e1 = orient2(A, B, x), e2 = orient2(B, C, x), e3 = orient2(C, A, x);
            if ((e1 >= 0 && e2 >= 0 && e3 >= 0) || (e1 <= 0 && e2 <= 0 && e3 <= 0)) return true;
        }
        return segments_meet2(P, Q, A, B) || segments_meet2(P, Q, B, C) || segments_meet2(P, Q, C, A);
    }
    // The line pq passes through the triangle when it passes every edge on the same side.
    const double s1 = orient3(p, q, a, b), s2 = orient3(p, q, b, c), s3 = orient3(p, q, c, a);
    return (s1 >= 0 && s2 >= 0 && s3 >= 0) || (s1 <= 0 && s2 <= 0 && s3 <= 0);
}

// Whether closed triangles t and u share a point, up to rounding: two
// triangles that meet have an edge of one meeting the other.
bool triangles_meet(const TriangleMesh &m, uint32_t t, uint32_t u) {
    const double *a[3], *b[3];
    for (int k = 0; k < 3; ++k) {
        a[k] = &m.positions[3 * size_t(m.triangles[3 * size_t(t) + k])];
        b[k] = &m.positions[3 * size_t(m.triangles[3 * size_t(u) + k])];
    }
    const double da[3] = {orient3(b[0], b[1], b[2], a[0]), orient3(b[0], b[1], b[2], a[1]), orient3(b[0], b[1], b[2], a[2])};
    if ((da[0] > 0 && da[1] > 0 && da[2] > 0) || (da[0] < 0 && da[1] < 0 && da[2] < 0)) return false;
    const double db[3] = {orient3(a[0], a[1], a[2], b[0]), orient3(a[0], a[1], a[2], b[1]), orient3(a[0], a[1], a[2], b[2])};
    if ((db[0] > 0 && db[1] > 0 && db[2] > 0) || (db[0] < 0 && db[1] < 0 && db[2] < 0)) return false;
    for (int k = 0; k < 3; ++k)
        if (segment_meets_triangle(a[k], a[(k + 1) % 3], b[0], b[1], b[2]) ||
            segment_meets_triangle(b[k], b[(k + 1) % 3], a[0], a[1], a[2]))
            return true;
    return false;
}

inline bool boxes_overlap(const double *lo1, const double *hi1, const double *lo2, const double *hi2) {
    return lo1[0] <= hi2[0] && lo2[0] <= hi1[0] && lo1[1] <= hi2[1] && lo2[1] <= hi1[1] && lo1[2] <= hi2[2] &&
           lo2[2] <= hi1[2];
}

// Classification gives up beyond this much work (hierarchy nodes and pairs
// visited) per triangle, so crafted nesting cannot stall decoding.
constexpr size_t kNestingWorkPerTriangle = 64;
constexpr size_t kNestingWorkBase = 20000000;

struct NestingWork {
    size_t done = 0, limit = 0;
    void add(size_t amount) {
        done += amount;
        if (done > limit)
            throw Error("the mesh has too many shells facing inward nested inside (or crowded around) others to "
                        "classify them; merge or separate its shells");
    }
};

// Whether the surfaces of the hierarchy subtrees at `a` and `b` (two shells)
// share a point.
bool subtrees_meet(const TriangleMesh &m, uint32_t a, uint32_t b, NestingWork &work) {
    std::vector<std::pair<uint32_t, uint32_t>> stack{{a, b}};
    while (!stack.empty()) {
        const auto [x, y] = stack.back();
        stack.pop_back();
        const Branch &bx = m.tree[x], &by = m.tree[y];
        if (!boxes_overlap(bx.lo, bx.hi, by.lo, by.hi)) continue;
        work.add(1);
        const bool leaf_x = bx.right == 0, leaf_y = by.right == 0;
        if (leaf_x && leaf_y) {
            work.add(size_t(bx.count) * by.count);
            for (uint32_t t = bx.first; t < bx.first + bx.count; ++t)
                for (uint32_t u = by.first; u < by.first + by.count; ++u)
                    if (triangles_meet(m, t, u)) return true;
            continue;
        }
        double extent_x = 0, extent_y = 0;
        for (int k = 0; k < 3; ++k) {
            extent_x = std::fmax(extent_x, bx.hi[k] - bx.lo[k]);
            extent_y = std::fmax(extent_y, by.hi[k] - by.lo[k]);
        }
        if (leaf_y || (!leaf_x && extent_x >= extent_y)) {
            stack.emplace_back(x + 1, y);
            stack.emplace_back(bx.right, y);
        } else {
            stack.emplace_back(x, y + 1);
            stack.emplace_back(x, by.right);
        }
    }
    return false;
}

inline bool box_holds(const double *lo, const double *hi, const double *box) {
    return lo[0] <= box[0] && lo[1] <= box[1] && lo[2] <= box[2] && box[3] <= hi[0] && box[4] <= hi[1] &&
           box[5] <= hi[2];
}

// A median-split hierarchy over the shells' boxes, for the shells whose boxes
// hold or overlap a box (6 values: lo, hi).
class ShellBoxes {
public:
    ShellBoxes(const std::vector<double> &boxes, uint32_t count) : boxes_(boxes), order_(count) {
        std::iota(order_.begin(), order_.end(), 0u);
        if (count) build(0, count);
    }

    // Appends the shells whose boxes hold `box` (holding = true) or overlap it.
    void find(const double *box, bool holding, std::vector<uint32_t> &out, NestingWork &work) const {
        const auto wanted = [&](const double *lo, const double *hi) {
            return holding ? box_holds(lo, hi, box) : boxes_overlap(lo, hi, box, box + 3);
        };
        uint32_t stack[kStack];
        int top = 0;
        stack[top++] = 0;
        while (top > 0) {
            const uint32_t index = stack[--top];
            const Branch &node = nodes_[index];
            work.add(1);
            if (!wanted(node.lo, node.hi)) continue;
            if (node.right != 0) {
                stack[top++] = node.right;
                stack[top++] = index + 1;
                continue;
            }
            for (uint32_t i = node.first; i < node.first + node.count; ++i) {
                const double *other = &boxes_[6 * size_t(order_[i])];
                if (wanted(other, other + 3)) out.push_back(order_[i]);
            }
        }
    }

private:
    uint32_t build(uint32_t begin, uint32_t end) {
        const uint32_t index = static_cast<uint32_t>(nodes_.size());
        nodes_.emplace_back();
        Branch node{};
        double lo[3] = {INFINITY, INFINITY, INFINITY}, hi[3] = {-INFINITY, -INFINITY, -INFINITY};
        for (int a = 0; a < 3; ++a) {
            node.lo[a] = INFINITY;
            node.hi[a] = -INFINITY;
        }
        for (uint32_t i = begin; i < end; ++i) {
            const double *box = &boxes_[6 * size_t(order_[i])];
            for (int a = 0; a < 3; ++a) {
                node.lo[a] = std::fmin(node.lo[a], box[a]);
                node.hi[a] = std::fmax(node.hi[a], box[3 + a]);
                lo[a] = std::fmin(lo[a], box[a] + box[3 + a]);
                hi[a] = std::fmax(hi[a], box[a] + box[3 + a]);
            }
        }
        if (end - begin <= kLeafBodies) {
            node.first = begin;
            node.count = end - begin;
        } else {
            int axis = 0;
            for (int a = 1; a < 3; ++a)
                if (hi[a] - lo[a] > hi[axis] - lo[axis]) axis = a;
            const uint32_t middle = begin + (end - begin) / 2;
            std::nth_element(order_.begin() + begin, order_.begin() + middle, order_.begin() + end,
                             [&](uint32_t a, uint32_t b) {
                                 const double ca = boxes_[6 * size_t(a) + axis] + boxes_[6 * size_t(a) + 3 + axis];
                                 const double cb = boxes_[6 * size_t(b) + axis] + boxes_[6 * size_t(b) + 3 + axis];
                                 return ca < cb || (ca == cb && a < b);
                             });
            build(begin, middle);
            node.right = build(middle, end);
        }
        nodes_[index] = node;
        return index;
    }

    const std::vector<double> &boxes_;
    std::vector<uint32_t> order_;
    std::vector<Branch> nodes_;
};

// The side of shell `root`'s surface that p lies on: 1 inside (by the
// shell's current orientation), -1 outside, 0 on it.
int side_of(const TriangleMesh &m, uint32_t root, const double *p) {
    Nearest best;
    nearest(m, root, p, best);
    if (!(best.d2 > 0)) return 0;
    return inside(m, p, best) ? 1 : -1;
}

// Decides which shells facing inward (negative volume, after the mesh as a
// whole faces outward) are cavities. Shells are taken from the largest
// enclosed volume down, so every shell that could hold a shell is settled
// before it. A shell is a cavity of a shell X facing outward when it lies in
// X's solid: a vertex lies strictly inside X, their surfaces do not meet, and
// it neither lies in nor meets one of X's cavities. Containers are tried from
// the smallest volume up. A shell inside such an X that meets one of X's
// cavities (two cavities crossing) cannot be represented and is an error; any
// other shell facing inward (an inverted body, one crossing other shells, or
// an island in a cavity) is turned outward and becomes a body. Returns each
// cavity's container (kNone for bodies) and turns the other shells.
std::vector<uint32_t> classify_shells(TriangleMesh &m, const std::vector<uint32_t> &first,
                                      const std::vector<uint32_t> &roots, const std::vector<double> &boxes,
                                      std::vector<double> &volume, uint32_t vertex_count) {
    const uint32_t shells = static_cast<uint32_t>(volume.size());
    std::vector<uint32_t> owner(shells, kNone), negative;
    for (uint32_t s = 0; s < shells; ++s)
        if (volume[s] < 0) negative.push_back(s);
    if (negative.empty()) return owner;
    std::sort(negative.begin(), negative.end(),
              [&](uint32_t a, uint32_t b) { return volume[a] < volume[b] || (volume[a] == volume[b] && a < b); });
    NestingWork work;
    work.limit = kNestingWorkBase + kNestingWorkPerTriangle * (m.triangles.size() / 3);
    const ShellBoxes tree(boxes, shells);
    std::vector<uint8_t> flipped_vertex(vertex_count, 0);
    std::vector<uint32_t> nearby, candidates;
    for (const uint32_t s : negative) {
        const double *box = &boxes[6 * size_t(s)];
        candidates.clear();
        tree.find(box, true, candidates, work);
        candidates.erase(std::remove_if(candidates.begin(), candidates.end(),
                                        [&](uint32_t x) { return x == s || !(volume[x] > 0); }),
                         candidates.end());
        std::sort(candidates.begin(), candidates.end(),
                  [&](uint32_t a, uint32_t b) { return volume[a] < volume[b] || (volume[a] == volume[b] && a < b); });
        nearby.clear();
        if (!candidates.empty()) tree.find(box, false, nearby, work);  // cavities that may meet s
        const double *q = &m.positions[3 * size_t(m.triangles[3 * size_t(first[s])])];
        uint32_t container = kNone;
        bool crossing = false;
        for (const uint32_t x : candidates) {
            work.add(1);
            if (side_of(m, roots[x], q) != 1 || subtrees_meet(m, roots[s], roots[x], work)) continue;
            bool in_cavity = false, meets_cavity = false;
            for (const uint32_t c : nearby) {
                if (owner[c] != x) continue;
                work.add(1);
                if (subtrees_meet(m, roots[s], roots[c], work)) {
                    meets_cavity = true;
                    break;
                }
                if (side_of(m, roots[c], q) == -1) {  // a cavity faces inward: q lies in its space
                    in_cavity = true;
                    break;
                }
            }
            crossing = crossing || meets_cavity;
            if (meets_cavity || in_cavity) continue;
            container = x;
            break;
        }
        if (container != kNone) {
            owner[s] = container;
            continue;
        }
        if (crossing) {
            const double center[3] = {(box[0] + box[3]) / 2, (box[1] + box[4]) / 2, (box[2] + box[5]) / 2};
            throw Error(format("two shells facing inward (cavities) inside one body intersect, e.g. near %s; merge "
                               "them into one cavity",
                               point_text(center).c_str()));
        }
        volume[s] = -volume[s];  // an inverted body, a shell crossing others, or an island in a cavity
        for (uint32_t t = first[s]; t < first[s + 1]; ++t) {
            flip_triangle(m, t);
            for (int k = 0; k < 3; ++k) {
                const uint32_t v = m.triangles[3 * size_t(t) + k];
                if (flipped_vertex[v]) continue;
                flipped_vertex[v] = 1;
                for (int a = 0; a < 3; ++a) m.vertex_normals[3 * size_t(v) + a] = -m.vertex_normals[3 * size_t(v) + a];
            }
        }
    }
    return owner;
}

// Whether the bodies' boxes overlap so often (more than twice per body on
// average) that the shell hierarchy prunes queries poorly; a hierarchy over
// all triangles then serves them.
bool crowded_bodies(const TriangleMesh &m) {
    const size_t limit = 2 * m.bodies.size() + 16;
    size_t pairs = 0;
    for (uint32_t b = 0; b < m.bodies.size(); ++b) {
        const TriangleMesh::Body &body = m.bodies[b];
        uint32_t stack[kStack];
        int top = 0;
        stack[top++] = 0;
        while (top > 0) {
            const uint32_t index = stack[--top];
            const Branch &node = m.body_tree[index];
            if (!boxes_overlap(node.lo, node.hi, body.lo, body.hi)) continue;
            if (node.right != 0) {
                stack[top++] = node.right;
                stack[top++] = index + 1;
                continue;
            }
            for (uint32_t i = node.first; i < node.first + node.count; ++i) {
                const TriangleMesh::Body &other = m.bodies[m.body_order[i]];
                if (m.body_order[i] != b && boxes_overlap(other.lo, other.hi, body.lo, body.hi) && ++pairs > limit)
                    return true;
            }
        }
    }
    return false;
}

}  // namespace

void build_triangle_mesh(TriangleMesh &m, const double *params, size_t param_count) {
    uint32_t vertex_count = 0;
    read_mesh(m, params, param_count, vertex_count);
    check_triangles(m, vertex_count);
    const size_t count = m.triangles.size() / 3;
    {
        const std::vector<uint32_t> partner = pair_edges(m, soup_volume(m) >= 0);
        vertex_count = split_vertex_fans(m, partner, vertex_count);
        m.neighbours.resize(partner.size());
        for (size_t s = 0; s < partner.size(); ++s) m.neighbours[s] = partner[s] / 3;
    }

    // Connected shells (numbered by their first triangle), their boxes and
    // enclosed volumes (about a vertex of each shell, for accuracy).
    std::vector<uint32_t> shell(count);
    uint32_t shells = 0;
    {
        DisjointSets sets(count);
        for (size_t s = 0; s < m.neighbours.size(); ++s) sets.unite(static_cast<uint32_t>(s / 3), m.neighbours[s]);
        std::vector<uint32_t> number(count, kNone);
        for (size_t t = 0; t < count; ++t) {
            const uint32_t root = sets.find(static_cast<uint32_t>(t));
            if (number[root] == kNone) number[root] = shells++;
            shell[t] = number[root];
        }
    }
    std::vector<double> volume(shells, 0.0), boxes(6 * size_t(shells));
    std::vector<uint32_t> origin(shells, kNone), first(size_t(shells) + 1, 0);
    for (uint32_t s = 0; s < shells; ++s)
        for (int a = 0; a < 3; ++a) {
            boxes[6 * size_t(s) + a] = INFINITY;
            boxes[6 * size_t(s) + 3 + a] = -INFINITY;
        }
    for (size_t t = 0; t < count; ++t) {
        const uint32_t s = shell[t], *v = &m.triangles[3 * t];
        if (origin[s] == kNone) origin[s] = v[0];
        const double *o = &m.positions[3 * size_t(origin[s])];
        double corner[3][3];
        for (int k = 0; k < 3; ++k)
            for (int a = 0; a < 3; ++a) {
                const double value = m.positions[3 * size_t(v[k]) + a];
                corner[k][a] = value - o[a];
                boxes[6 * size_t(s) + a] = std::fmin(boxes[6 * size_t(s) + a], value);
                boxes[6 * size_t(s) + 3 + a] = std::fmax(boxes[6 * size_t(s) + 3 + a], value);
            }
        double side[3];
        cross(corner[1], corner[2], side);
        volume[s] += dot(corner[0], side) / 6;
        ++first[s + 1];
    }
    double total = 0;
    for (uint32_t s = 0; s < shells; ++s) {
        const double *box = &boxes[6 * size_t(s)];
        const double size[3] = {box[3] - box[0], box[4] - box[1], box[5] - box[2]};
        const double diagonal = std::sqrt(dot(size, size));
        if (!(std::fabs(volume[s]) > kEmptyShell * diagonal * diagonal * diagonal)) {
            const double center[3] = {(box[0] + box[3]) / 2, (box[1] + box[4]) / 2, (box[2] + box[5]) / 2};
            throw Error(format("a connected shell of %u triangles around %s encloses no volume (such as a sheet "
                               "triangulated on both sides); give it thickness or remove it",
                               first[s + 1], point_text(center).c_str()));
        }
        total += volume[s];
    }
    if (total < 0) {  // an inward mesh: turn it outward
        for (size_t t = 0; t < count; ++t) flip_triangle(m, t);
        for (double &v : volume) v = -v;
    }

    // Shells' triangles are contiguous in hierarchy order, each shell a subtree.
    for (uint32_t s = 0; s < shells; ++s) first[s + 1] += first[s];
    std::vector<uint32_t> order(count), roots(shells);
    {
        std::vector<uint32_t> next(first.begin(), first.end() - 1);
        for (size_t t = 0; t < count; ++t) order[next[shell[t]]++] = static_cast<uint32_t>(t);
        std::vector<double> centroids(3 * count);
        for (size_t t = 0; t < count; ++t)
            for (int a = 0; a < 3; ++a)
                centroids[3 * t + a] = (m.positions[3 * size_t(m.triangles[3 * t]) + a] +
                                        m.positions[3 * size_t(m.triangles[3 * t + 1]) + a] +
                                        m.positions[3 * size_t(m.triangles[3 * t + 2]) + a]) / 3;
        std::vector<uint32_t> ids(shells);
        std::iota(ids.begin(), ids.end(), 0u);
        TreeBuilder(m, order, centroids, first, boxes).shells(ids, 0, shells, roots, 0);
    }
    // Store triangles in hierarchy order and number vertices by first use, so
    // nearby queries touch nearby memory.
    {
        std::vector<uint32_t> position(count);
        for (size_t i = 0; i < count; ++i) position[order[i]] = static_cast<uint32_t>(i);
        std::vector<uint32_t> triangles(3 * count), neighbours(3 * count), shells_in_order(count);
        for (size_t i = 0; i < count; ++i) {
            const size_t t = order[i];
            for (int k = 0; k < 3; ++k) {
                triangles[3 * i + k] = m.triangles[3 * t + k];
                neighbours[3 * i + k] = position[m.neighbours[3 * t + k]];
            }
            shells_in_order[i] = shell[t];
        }
        std::vector<uint32_t> number(vertex_count, kNone);
        std::vector<double> positions(m.positions.size());
        uint32_t next = 0;
        for (uint32_t &v : triangles) {
            if (number[v] == kNone) {
                std::copy(&m.positions[3 * size_t(v)], &m.positions[3 * size_t(v)] + 3, &positions[3 * size_t(next)]);
                number[v] = next++;
            }
            v = number[v];
        }
        m.triangles.swap(triangles);
        m.neighbours.swap(neighbours);
        m.positions.swap(positions);
        shell.swap(shells_in_order);
    }
    compute_normals(m);
    const std::vector<uint32_t> owner = classify_shells(m, first, roots, boxes, volume, vertex_count);

    // Bodies: every positive shell with its cavities.
    std::vector<uint32_t> body(shells, kNone);
    uint32_t bodies = 0;
    for (uint32_t s = 0; s < shells; ++s)
        if (owner[s] == kNone) body[s] = bodies++;
    std::vector<uint32_t> root_count(bodies, 0);
    for (uint32_t s = 0; s < shells; ++s) {
        if (owner[s] != kNone) body[s] = body[owner[s]];
        ++root_count[body[s]];
    }
    m.bodies.resize(bodies);
    m.body_roots.resize(shells);
    for (uint32_t b = 0, offset = 0; b < bodies; offset += root_count[b], ++b) {
        m.bodies[b].first_root = offset;
        m.bodies[b].root_count = 0;
        for (int a = 0; a < 3; ++a) {
            m.bodies[b].lo[a] = INFINITY;
            m.bodies[b].hi[a] = -INFINITY;
        }
    }
    for (uint32_t s = 0; s < shells; ++s) {
        TriangleMesh::Body &b = m.bodies[body[s]];
        m.body_roots[b.first_root + b.root_count++] = roots[s];
        for (int a = 0; a < 3; ++a) {
            b.lo[a] = std::fmin(b.lo[a], boxes[6 * size_t(s) + a]);
            b.hi[a] = std::fmax(b.hi[a], boxes[6 * size_t(s) + 3 + a]);
        }
    }
    m.triangle_body.resize(count);
    for (size_t t = 0; t < count; ++t) m.triangle_body[t] = body[shell[t]];
    if (bodies > 1) {
        m.body_order.resize(bodies);
        std::iota(m.body_order.begin(), m.body_order.end(), 0u);
        build_body_tree(m, 0, bodies);
    }
    if (bodies > 1 && crowded_bodies(m)) {
        // A hierarchy over all triangles whatever their shells: overlapping
        // shells' boxes cannot prune the nearest feature, nor the ray that
        // finds the bodies holding a point.
        std::vector<double> centroids(3 * count);
        for (size_t t = 0; t < count; ++t)
            for (int a = 0; a < 3; ++a)
                centroids[3 * t + a] = (m.positions[3 * size_t(m.triangles[3 * t]) + a] +
                                        m.positions[3 * size_t(m.triangles[3 * t + 1]) + a] +
                                        m.positions[3 * size_t(m.triangles[3 * t + 2]) + a]) / 3;
        m.mixed_order.resize(count);
        std::iota(m.mixed_order.begin(), m.mixed_order.end(), 0u);
        TreeBuilder(m, m.mixed, m.mixed_order, centroids).triangles(0, static_cast<uint32_t>(count), 0);
    }
    for (int a = 0; a < 3; ++a) {
        m.lo[a] = m.tree[0].lo[a];
        m.hi[a] = m.tree[0].hi[a];
    }
}

}  // namespace rsdf
