// GPU field programs (SDF-DESIGN.md, "GPU field programs").
//
// A part compiles to a sparse grid of bricks. Each brick whose region (the
// brick grown by a margin) may hold the surface gets a straight-line register
// tape of the part's DAG pruned to the nodes that can change a value in that
// region; elsewhere the program stores a conservative signed distance bound.
//
// Pruning walks the graph with the region mapped into each node's own
// coordinates (a ball and a box, both conservative) and applies the rules of
// the kernel's batch culling, which keep values bitwise (evaluate.cpp): a
// union operand, difference cutter or intersection operand is removed where
// its value bound puts the fold's blend weight at zero and its min/max on the
// other side; an operand that dominates a smooth fold's whole prefix
// restarts the fold at itself; exact minima and maxima drop any operand
// another one dominates, wherever it stands. An extrusion's or loft's
// profile is replaced by a constant where the slab ignores its exact value.
// Bounds come from the node's value at the region's centre plus or minus its
// Lipschitz bound times the region's radius, and from the node's distance
// ratio times the distance to its box, with a relative slack that covers
// rounding of both values compared.
//
// Tapes are register code: value registers hold floats, point registers hold
// the points that point-mapping nodes produce. Boolean folds evaluate the
// operand that needs more registers first (smooth minima and maxima are
// symmetric in their operands, so the order changes no value), a fold,
// scale or offset joins the instruction computing its operand (a post
// operation), and a linear scan assigns registers. A brick whose tape is
// long tries its octants, which prune its pruned tree further, and keeps
// them when their tapes are shorter (down to a minimum size). Tapes are
// encoded as 16-byte records with their parameters in a per-part pool of
// 16-byte aligned runs; cells are one word. The float64 reference interpreter
// below runs the same tapes with the parameters the kernel itself uses, for
// tests.
#include "kernel.h"
#include "radsdf_gpu.h"

#include <pthread.h>

#include <algorithm>
#include <array>
#include <atomic>
#include <chrono>
#include <cstdio>
#include <cstring>
#include <exception>
#include <memory>
#include <mutex>
#include <string>
#include <unordered_map>

namespace rsdf {
namespace gpu {
namespace {

// ------------------------------------------------------------------ format

constexpr uint32_t kMagic = 0x47504652;  // "RFPG"
constexpr uint32_t kVersion = 2;
constexpr uint32_t kHeaderWords = 16;
constexpr uint32_t kPartWords = 24;
constexpr uint32_t kMeshWords = 24;
constexpr uint32_t kPolygonWords = 8;
// Cells are one word: a far cell has bit 31 set, its sign in bit 30 and the
// float bits of its distance bound shifted right by one (rounded toward 0);
// a refined cell is kRefined | the word offset of its 8 octant cells / 8;
// otherwise the word is the tape's word offset / 4 (a header record, then
// its instructions).
constexpr uint32_t kFarCell = 0x80000000u;
constexpr uint32_t kRefined = 0x40000000u;
constexpr uint32_t kDefaultRefine = 4;        // tapes longer than this try their brick's octants (down to the minimum)
constexpr int kMaxLevels = 8;
constexpr uint32_t kTreeSearch = 0xFFFFFFFFu; // POLYGON range count: search the edge hierarchy per point

constexpr uint32_t kValueRegisters = 16;
constexpr uint32_t kPointRegisters = 8;
constexpr uint32_t kDefaultInstructions = 4096;
constexpr uint64_t kDefaultWords = uint64_t(1) << 26;  // 256 MB
constexpr uint64_t kMaxGridCells = uint64_t(1) << 21;  // per part
constexpr int kMaxDepth = 4096;                        // nesting of the pruned walk
constexpr uint32_t kPolygonAll = 16;                   // smaller polygons test every edge
constexpr uint32_t kPolygonCandidates = 64;            // more candidates: per-point hierarchy search
constexpr size_t kStackBytes = size_t(64) << 20;       // compiler threads
constexpr double kSlack = 1e-9;                        // relative slack of pruning decisions
constexpr double kDeepInside = -1e30;                  // profile value where an extrusion ignores its profile

enum GOp : uint8_t {
    G_END = 0,
    // point maps: P[dst] = map(P[a])
    G_AFFINE3 = 1, G_STRETCH3, G_MIRROR3, G_TWIST, G_ELONGATE, G_REVOLVE, G_AFFINE2, G_STRETCH2, G_MIRROR2,
    G_CLAMP3, G_POINT3, G_TRANSLATE3,
    // leaves: V[dst] = f(P[a])
    G_SPHERE = 20, G_BOX, G_CYLINDER, G_CONE, G_CAPSULE, G_TORUS, G_ELLIPSOID, G_HALFSPACE, G_MESH, G_CIRCLE,
    G_RECT, G_POLYGON, G_SEGMENT, G_ELLIPSE, G_HALFPLANE, G_COORD,
    // values
    G_SCALE = 40, G_OFFSET, G_SHELL_IN, G_SHELL_OUT, G_SHELL_MID, G_OUTLINE, G_EXTRUDE, G_TAPER, G_LOFT, G_MIN,
    G_MAX, G_SMIN, G_SMAX, G_DIFF, G_SDIFF, G_IMPLICIT,
    // field expressions
    G_CONST = 60, G_ADD, G_SUB, G_MUL, G_DIV, G_NEG, G_ABS, G_SQUARE, G_SQRT, G_SIN, G_COS, G_EXP, G_LOG, G_POWC,
    G_ATAN2, G_CLAMP, G_LENSQ, G_LENACC, G_TRIWAVE, G_FSMIN, G_FSMAX,
};

enum Kind : uint8_t { K_NONE = 0, K_VALUE = 1, K_POINT = 2 };

struct OpSpec {
    uint8_t dst, a, b, c;  // operand kinds (c is an extra word after the header)
    uint8_t params;        // float parameters after any auxiliary words
};

OpSpec spec(uint8_t op) {
    switch (op) {
    case G_AFFINE3: return {K_POINT, K_POINT, 0, 0, 13};
    case G_STRETCH3: return {K_POINT, K_POINT, 0, 0, 3};
    case G_MIRROR3: return {K_POINT, K_POINT, 0, 0, 4};
    case G_TWIST: return {K_POINT, K_POINT, 0, 0, 1};
    case G_ELONGATE: return {K_POINT, K_POINT, 0, 0, 3};
    case G_REVOLVE: return {K_POINT, K_POINT, 0, 0, 1};
    case G_AFFINE2: return {K_POINT, K_POINT, 0, 0, 7};
    case G_STRETCH2: return {K_POINT, K_POINT, 0, 0, 2};
    case G_MIRROR2: return {K_POINT, K_POINT, 0, 0, 3};
    case G_CLAMP3: return {K_POINT, K_POINT, 0, 0, 6};
    case G_TRANSLATE3: return {K_POINT, K_POINT, 0, 0, 3};
    case G_POINT3: return {K_POINT, K_VALUE, K_VALUE, K_VALUE, 0};
    case G_SPHERE: return {K_VALUE, K_POINT, 0, 0, 1};
    case G_BOX: return {K_VALUE, K_POINT, 0, 0, 4};
    case G_CYLINDER: return {K_VALUE, K_POINT, 0, 0, 2};
    case G_CONE: return {K_VALUE, K_POINT, 0, 0, 6};
    case G_CAPSULE: return {K_VALUE, K_POINT, 0, 0, 8};
    case G_TORUS: return {K_VALUE, K_POINT, 0, 0, 2};
    case G_ELLIPSOID: return {K_VALUE, K_POINT, 0, 0, 4};
    case G_HALFSPACE: return {K_VALUE, K_POINT, 0, 0, 4};
    case G_MESH: return {K_VALUE, K_POINT, 0, 0, 0};
    case G_CIRCLE: return {K_VALUE, K_POINT, 0, 0, 1};
    case G_RECT: return {K_VALUE, K_POINT, 0, 0, 3};
    case G_POLYGON: return {K_VALUE, K_POINT, 0, 0, 0};
    case G_SEGMENT: return {K_VALUE, K_POINT, 0, 0, 6};
    case G_ELLIPSE: return {K_VALUE, K_POINT, 0, 0, 3};
    case G_HALFPLANE: return {K_VALUE, K_POINT, 0, 0, 3};
    case G_COORD: return {K_VALUE, K_POINT, 0, 0, 0};
    case G_SCALE: case G_OFFSET: case G_SHELL_IN: case G_SHELL_OUT: case G_SHELL_MID: case G_OUTLINE:
        return {K_VALUE, K_VALUE, 0, 0, 1};
    case G_EXTRUDE: return {K_VALUE, K_VALUE, K_POINT, 0, 2};
    case G_TAPER: return {K_VALUE, K_VALUE, K_POINT, 0, 6};
    case G_LOFT: return {K_VALUE, K_VALUE, K_VALUE, K_POINT, 5};
    case G_MIN: case G_MAX: case G_DIFF: return {K_VALUE, K_VALUE, K_VALUE, 0, 0};
    case G_SMIN: case G_SMAX: case G_SDIFF: return {K_VALUE, K_VALUE, K_VALUE, 0, 1};
    case G_IMPLICIT: return {K_VALUE, K_VALUE, K_POINT, 0, 7};
    case G_CONST: return {K_VALUE, 0, 0, 0, 1};
    case G_ADD: case G_SUB: case G_MUL: case G_DIV: case G_ATAN2: case G_LENACC:
        return {K_VALUE, K_VALUE, K_VALUE, 0, 0};
    case G_NEG: case G_ABS: case G_SQUARE: case G_SQRT: case G_SIN: case G_COS: case G_EXP: case G_LOG: case G_LENSQ:
        return {K_VALUE, K_VALUE, 0, 0, 0};
    case G_POWC: case G_TRIWAVE: return {K_VALUE, K_VALUE, 0, 0, 1};
    case G_CLAMP: return {K_VALUE, K_VALUE, 0, 0, 2};
    case G_FSMIN: case G_FSMAX: return {K_VALUE, K_VALUE, K_VALUE, 0, 1};
    default: return {0, 0, 0, 0, 0};
    }
}

// ------------------------------------------------------------------ primitives
//
// The kernel's formulas (evaluate.cpp), written the same way so that the
// reference interpreter rounds exactly as the kernel does.

inline double clamp01(double t) { return std::fmin(std::fmax(t, 0.0), 1.0); }

inline double slab(double d, double w) {
    const double od = std::fmax(d, 0.0), ow = std::fmax(w, 0.0);
    return std::fmin(std::fmax(d, w), 0.0) + std::sqrt(od * od + ow * ow);
}

inline double smin(double a, double b, double k) {
    const double h = std::fmax(k - std::fabs(a - b), 0.0) / k;
    return std::fmin(a, b) - h * h * k * 0.25;
}

inline double smax(double a, double b, double k) {
    const double h = std::fmax(k - std::fabs(a - b), 0.0) / k;
    return std::fmax(a, b) + h * h * k * 0.25;
}

inline double triwave(double a, double period) {
    const double twice = 2 * period;
    double rest = std::fmod(a, twice);
    if (rest < 0) rest += twice;
    return std::fabs(rest - period);
}

inline double field_smooth(double a, double b, double k, double sign) {
    const double u = sign * a, v = sign * b;
    const double h = k > 0 ? std::fmax(k - std::fabs(u - v), 0.0) / k : 0.0;
    return sign * (std::fmin(u, v) - h * h * k * 0.25);
}

// Leaves other than MESH and POLYGON at q (x, y, z; 2D leaves ignore z).
double leaf_value(uint8_t op, const double *p, const double *q) {
    const double x = q[0], y = q[1], z = q[2];
    switch (op) {
    case G_SPHERE: return std::sqrt(x * x + y * y + z * z) - p[0];
    case G_BOX: {
        const double r = p[3], bx = p[0], by = p[1], bz = p[2];
        const double qx = std::fabs(x) - bx, qy = std::fabs(y) - by, qz = std::fabs(z) - bz;
        const double ox = std::fmax(qx, 0.0), oy = std::fmax(qy, 0.0), oz = std::fmax(qz, 0.0);
        return std::sqrt(ox * ox + oy * oy + oz * oz) + std::fmin(std::fmax(qx, std::fmax(qy, qz)), 0.0) - r;
    }
    case G_CYLINDER: return slab(std::sqrt(x * x + y * y) - p[0], std::fabs(z) - p[1]);
    case G_CONE: {
        const double r1 = p[0], r2 = p[1], h = p[2];
        const double kx = p[3], ky = p[4], inv = p[5];
        const double qx = std::sqrt(x * x + y * y), qy = z;
        const double cax = qx - std::fmin(qx, qy < 0 ? r1 : r2), cay = std::fabs(qy) - h;
        const double t = clamp01(((r2 - qx) * kx + (h - qy) * ky) * inv);
        const double cbx = qx - r2 + kx * t, cby = qy - h + ky * t;
        const double sign = (cbx < 0 && cay < 0) ? -1.0 : 1.0;
        return sign * std::sqrt(std::fmin(cax * cax + cay * cay, cbx * cbx + cby * cby));
    }
    case G_CAPSULE: {
        const double bax = p[3], bay = p[4], baz = p[5], inv = p[6];
        const double px = x - p[0], py = y - p[1], pz = z - p[2];
        const double t = clamp01((px * bax + py * bay + pz * baz) * inv);
        const double dx = px - t * bax, dy = py - t * bay, dz = pz - t * baz;
        return std::sqrt(dx * dx + dy * dy + dz * dz) - p[7];
    }
    case G_TORUS: {
        const double qq = std::sqrt(x * x + y * y) - p[0];
        return std::sqrt(qq * qq + z * z) - p[1];
    }
    case G_ELLIPSOID: {
        const double ix = p[0], iy = p[1], iz = p[2];
        const double qx = x * ix, qy = y * iy, qz = z * iz;
        return (std::sqrt(qx * qx + qy * qy + qz * qz) - 1) * p[3];
    }
    case G_HALFSPACE: return p[0] * x + p[1] * y + p[2] * z - p[3];
    case G_CIRCLE: return std::sqrt(x * x + y * y) - p[0];
    case G_RECT: {
        const double r = p[2], bx = p[0], by = p[1];
        const double qx = std::fabs(x) - bx, qy = std::fabs(y) - by;
        const double ox = std::fmax(qx, 0.0), oy = std::fmax(qy, 0.0);
        return std::sqrt(ox * ox + oy * oy) + std::fmin(std::fmax(qx, qy), 0.0) - r;
    }
    case G_SEGMENT: {
        const double bax = p[2], bay = p[3], inv = p[4];
        const double px = x - p[0], py = y - p[1];
        const double t = clamp01((px * bax + py * bay) * inv);
        const double dx = px - t * bax, dy = py - t * bay;
        return std::sqrt(dx * dx + dy * dy) - p[5];
    }
    case G_ELLIPSE: {
        const double ix = p[0], iy = p[1];
        const double qx = x * ix, qy = y * iy;
        return (std::sqrt(qx * qx + qy * qy) - 1) * p[2];
    }
    case G_HALFPLANE: return p[0] * x + p[1] * y - p[2];
    default: return NAN;
    }
}

// Point maps (map_points in evaluate.cpp).
void map_point(uint8_t op, const double *p, const double *q, double *o) {
    const double x = q[0], y = q[1], z = q[2];
    switch (op) {
    case G_AFFINE3: {
        const double inv = p[12];
        const double dx = x - p[9], dy = y - p[10], dz = z - p[11];
        o[0] = (p[0] * dx + p[3] * dy + p[6] * dz) * inv;
        o[1] = (p[1] * dx + p[4] * dy + p[7] * dz) * inv;
        o[2] = (p[2] * dx + p[5] * dy + p[8] * dz) * inv;
        break;
    }
    case G_STRETCH3: {
        const double ix = p[0], iy = p[1], iz = p[2];
        o[0] = x * ix;
        o[1] = y * iy;
        o[2] = z * iz;
        break;
    }
    case G_MIRROR3: {
        const double m = 2 * (p[0] * x + p[1] * y + p[2] * z - p[3]);
        o[0] = x - m * p[0];
        o[1] = y - m * p[1];
        o[2] = z - m * p[2];
        break;
    }
    case G_TWIST: {
        const double angle = -p[0] * z;
        const double c = std::cos(angle), sn = std::sin(angle);
        o[0] = c * x - sn * y;
        o[1] = sn * x + c * y;
        o[2] = z;
        break;
    }
    case G_ELONGATE:
        o[0] = x - std::fmin(std::fmax(x, -p[0]), p[0]);
        o[1] = y - std::fmin(std::fmax(y, -p[1]), p[1]);
        o[2] = z - std::fmin(std::fmax(z, -p[2]), p[2]);
        break;
    case G_REVOLVE:
        o[0] = std::sqrt(x * x + y * y) - p[0];
        o[1] = z;
        o[2] = 0;
        break;
    case G_AFFINE2: {
        const double inv = p[6];
        const double dx = x - p[4], dy = y - p[5];
        o[0] = (p[0] * dx + p[2] * dy) * inv;
        o[1] = (p[1] * dx + p[3] * dy) * inv;
        o[2] = z;
        break;
    }
    case G_STRETCH2: {
        const double ix = p[0], iy = p[1];
        o[0] = x * ix;
        o[1] = y * iy;
        o[2] = z;
        break;
    }
    case G_MIRROR2: {
        const double m = 2 * (p[0] * x + p[1] * y - p[2]);
        o[0] = x - m * p[0];
        o[1] = y - m * p[1];
        o[2] = z;
        break;
    }
    case G_CLAMP3:
        for (int a = 0; a < 3; ++a) o[a] = std::fmin(std::fmax(q[a], p[a]), p[3 + a]);
        break;
    case G_TRANSLATE3:  // AFFINE3 with R = I and s = 1, which rounds the same
        o[0] = x - p[0];
        o[1] = y - p[1];
        o[2] = z - p[2];
        break;
    default: o[0] = o[1] = o[2] = NAN; break;
    }
}

// Value operations of one or two values (and a point for slabs and IMPLICIT).
double value_op(uint8_t op, const double *p, double a, double b, const double *q) {
    switch (op) {
    case G_SCALE: return a * p[0];
    case G_OFFSET: return a - p[0];
    case G_SHELL_IN: return std::fmax(a, -a - p[0]);
    case G_SHELL_OUT: return std::fmax(-a, a - p[0]);
    case G_SHELL_MID: case G_OUTLINE: return std::fabs(a) - p[0];
    case G_EXTRUDE: return slab(a, std::fabs(q[2] - p[0]) - p[1]);
    case G_TAPER: {
        const double z0 = p[0], inv = p[1], taper = p[2], c0 = p[3];
        const double d = (a - taper * clamp01((q[2] - z0) * inv)) * c0;
        return slab(d, std::fabs(q[2] - p[4]) - p[5]);
    }
    case G_LOFT: {
        const double t = clamp01((q[2] - p[0]) * p[1]);
        return slab(((1 - t) * a + t * b) * p[2], std::fabs(q[2] - p[3]) - p[4]);
    }
    case G_MIN: return std::fmin(a, b);
    case G_MAX: return std::fmax(a, b);
    case G_SMIN: return smin(a, b, p[0]);
    case G_SMAX: return smax(a, b, p[0]);
    case G_DIFF: return std::fmax(a, -b);
    case G_SDIFF: return smax(a, -b, p[0]);
    case G_IMPLICIT: {
        const double qx = std::fabs(q[0] - p[0]) - p[3];
        const double qy = std::fabs(q[1] - p[1]) - p[4];
        const double qz = std::fabs(q[2] - p[2]) - p[5];
        const double ox = std::fmax(qx, 0.0), oy = std::fmax(qy, 0.0), oz = std::fmax(qz, 0.0);
        const double box = std::sqrt(ox * ox + oy * oy + oz * oz) + std::fmin(std::fmax(qx, std::fmax(qy, qz)), 0.0);
        const double h = a / p[6];
        return h > 0 && box > 0 ? std::hypot(h, box) : std::fmax(h, box);
    }
    case G_CONST: return p[0];
    case G_ADD: return a + b;
    case G_SUB: return a - b;
    case G_MUL: return a * b;
    case G_DIV: return a / b;
    case G_NEG: return -a;
    case G_ABS: return std::fabs(a);
    case G_SQUARE: case G_LENSQ: return a * a;
    case G_SQRT: return std::sqrt(a);
    case G_SIN: return std::sin(a);
    case G_COS: return std::cos(a);
    case G_EXP: return std::exp(a);
    case G_LOG: return std::log(a);
    case G_POWC: return std::pow(a, p[0]);
    case G_ATAN2: return std::atan2(a, b);
    case G_CLAMP: return std::fmin(std::fmax(a, p[0]), p[1]);
    case G_LENACC: return a + b * b;
    case G_TRIWAVE: return triwave(a, p[0]);
    case G_FSMIN: return field_smooth(a, b, p[0], 1.0);
    case G_FSMAX: return field_smooth(a, b, p[0], -1.0);
    default: return NAN;
    }
}

// ------------------------------------------------------------------ polygons
//
// The kernel's edge search (polygon.cpp) over candidate edge ranges, or over
// the edge hierarchy per point; the sign rules are the kernel's.

inline double edge_distance2(const Polygon &polygon, uint32_t edge, double px, double py, double &t) {
    const double wx = px - polygon.x[edge], wy = py - polygon.y[edge];
    const double ex = polygon.ex[edge], ey = polygon.ey[edge];
    t = std::clamp((wx * ex + wy * ey) * polygon.inv_len2[edge], 0.0, 1.0);
    const double dx = wx - t * ex, dy = wy - t * ey;
    return dx * dx + dy * dy;
}

inline double point_box_distance2(double px, double py, const PolygonTree &node) {
    const double dx = std::fmax(std::fmax(node.lo[0] - px, px - node.hi[0]), 0.0);
    const double dy = std::fmax(std::fmax(node.lo[1] - py, py - node.hi[1]), 0.0);
    return dx * dx + dy * dy;
}

struct EdgeHit {
    double d2 = INFINITY, t = 0.5;
    uint32_t edge = 0;
};

EdgeHit nearest_edge(const Polygon &polygon, double px, double py) {
    uint32_t stack[128];
    int top = 0;
    stack[top++] = 0;
    EdgeHit best;
    while (top > 0) {
        const uint32_t index = stack[--top];
        const PolygonTree &node = polygon.tree[index];
        if (!(point_box_distance2(px, py, node) < best.d2)) continue;
        if (node.right == 0) {
            for (uint32_t edge = node.begin; edge < node.end; ++edge) {
                double t;
                const double d2 = edge_distance2(polygon, edge, px, py, t);
                if (d2 < best.d2) best = EdgeHit{d2, t, edge};
            }
            continue;
        }
        const uint32_t left = index + 1, right = node.right;
        const bool left_first =
            point_box_distance2(px, py, polygon.tree[left]) <= point_box_distance2(px, py, polygon.tree[right]);
        stack[top++] = left_first ? right : left;
        stack[top++] = left_first ? left : right;
    }
    return best;
}

double polygon_value(const Polygon &polygon, const uint32_t *ranges, uint32_t range_count, double px, double py) {
    const uint32_t count = static_cast<uint32_t>(polygon.x.size());
    EdgeHit best;
    if (range_count == kTreeSearch) {
        best = nearest_edge(polygon, px, py);
    } else {
        const double *vx = polygon.x.data(), *vy = polygon.y.data();
        const double *ex = polygon.ex.data(), *ey = polygon.ey.data(), *inv = polygon.inv_len2.data();
        for (uint32_t r = 0; r < range_count; ++r) {
            for (uint32_t e = ranges[2 * r]; e < ranges[2 * r + 1]; ++e) {
                const double wx = px - vx[e], wy = py - vy[e];
                const double t = std::clamp((wx * ex[e] + wy * ey[e]) * inv[e], 0.0, 1.0);
                const double dx = wx - t * ex[e], dy = wy - t * ey[e];
                const double d2 = dx * dx + dy * dy;
                if (d2 < best.d2) best = EdgeHit{d2, t, e};
            }
        }
    }
    if (!(best.d2 > 0)) return best.d2 == 0 ? 0.0 : best.d2;
    bool inside = false;
    const uint32_t edge = best.edge;
    const PolygonTree &box = polygon.tree.front();
    if (px < box.lo[0] || px > box.hi[0] || py < box.lo[1] || py > box.hi[1]) {
    } else if (best.t > 0 && best.t < 1) {
        inside = polygon.ex[edge] * (py - polygon.y[edge]) - polygon.ey[edge] * (px - polygon.x[edge]) > 0;
    } else {
        const uint32_t vertex = best.t <= 0 ? edge : (edge + 1 == count ? 0 : edge + 1);
        inside = (px - polygon.x[vertex]) * polygon.nx[vertex] + (py - polygon.y[vertex]) * polygon.ny[vertex] < 0;
    }
    return inside ? -std::sqrt(best.d2) : std::sqrt(best.d2);
}

// The edges that can be nearest to some point of the 2D box [lo, hi], as
// edge ranges (polygon.cpp's batch candidates with the box as the batch's
// box: the distance to the seed edge is convex, so its largest value over
// the box is at a corner). Returns false when more than `limit` edges stay.
bool polygon_candidates(const Polygon &polygon, const double *lo, const double *hi, uint32_t limit,
                        std::vector<uint32_t> &ranges) {
    const uint32_t seed = nearest_edge(polygon, 0.5 * (lo[0] + hi[0]), 0.5 * (lo[1] + hi[1])).edge;
    double bound2 = 0, t;
    for (int corner = 0; corner < 4; ++corner) {
        const double cx = corner & 1 ? hi[0] : lo[0], cy = corner & 2 ? hi[1] : lo[1];
        bound2 = std::fmax(bound2, edge_distance2(polygon, seed, cx, cy, t));
    }
    bound2 *= 1 + 1e-9;
    if (!std::isfinite(bound2)) return false;
    const size_t first = ranges.size();
    uint32_t stack[128];
    int top = 0;
    stack[top++] = 0;
    uint32_t candidates = 0;
    while (top > 0) {
        const uint32_t index = stack[--top];
        const PolygonTree &node = polygon.tree[index];
        const double dx = std::fmax(std::fmax(node.lo[0] - hi[0], lo[0] - node.hi[0]), 0.0);
        const double dy = std::fmax(std::fmax(node.lo[1] - hi[1], lo[1] - node.hi[1]), 0.0);
        if (!(dx * dx + dy * dy <= bound2)) continue;
        if (node.right != 0) {
            stack[top++] = node.right;
            stack[top++] = index + 1;
            continue;
        }
        candidates += node.end - node.begin;
        if (candidates > limit) {
            ranges.resize(first);
            return false;
        }
        if (ranges.size() > first && ranges.back() == node.begin) {
            ranges.back() = node.end;
        } else {
            ranges.push_back(node.begin);
            ranges.push_back(node.end);
        }
    }
    return true;
}

// ------------------------------------------------------------------ tapes

// One instruction. Operands are virtual registers until allocation, then
// physical ones; `aux` is the MESH or POLYGON index, or COORD's axis.
struct Ins {
    uint8_t op = 0;
    uint32_t dst = 0, a = 0, b = 0, c = 0;
    uint32_t aux = 0;
    uint32_t param_first = 0, param_count = 0;
    uint32_t range_first = 0, range_count = 0;  // POLYGON candidate ranges (pairs), or kTreeSearch
    bool fused = false;  // a point map whose only reader is the next instruction, a leaf: encoded inside it
    // A fold, scale or offset absorbed from the next instruction, which read
    // this one's value only: value = post(value, V[post_reg]) (the value is
    // the fold's second operand when post_swap), written to dst.
    uint8_t post = 0;
    bool post_swap = false;
    uint32_t post_reg = 0;
    double post_param = 0;
};

// Post operations (Ins::post), applied to an instruction's value.
enum Post : uint8_t { P_NONE = 0, P_MIN, P_MAX, P_SMIN, P_SMAX, P_DIFF, P_SDIFF, P_SCALE, P_OFFSET };

inline bool post_binary(uint8_t post) { return post >= P_MIN && post <= P_SDIFF; }

struct Tape {
    std::vector<Ins> code;
    std::vector<double> params;
    std::vector<uint32_t> ranges;
    uint32_t result = 0;
    uint32_t values = 0, points = 0;  // registers used
};

// Thrown when a part exceeds a program limit; the part becomes mesh only.
struct Limit : std::runtime_error {
    using std::runtime_error::runtime_error;
};

// The fold, scale or offset of an instruction's post operation, with the
// operands in the absorbed instruction's order (so values stay bitwise).
uint8_t post_op(uint8_t post) {
    static const uint8_t ops[] = {G_END, G_MIN, G_MAX, G_SMIN, G_SMAX, G_DIFF, G_SDIFF, G_SCALE, G_OFFSET};
    return ops[post];
}

double apply_post(const Ins &ins, double value, double other) {
    const double *p = &ins.post_param;
    if (!post_binary(ins.post)) return value_op(post_op(ins.post), p, value, 0.0, nullptr);
    return ins.post_swap ? value_op(post_op(ins.post), p, other, value, nullptr)
                         : value_op(post_op(ins.post), p, value, other, nullptr);
}

// The float64 reference interpreter.
double run_tape(const rsdf_graph &graph, const Tape &tape, const double *point) {
    double V[kValueRegisters];
    double P[kPointRegisters][3];
    P[0][0] = point[0];
    P[0][1] = point[1];
    P[0][2] = point[2];
    for (const Ins &ins : tape.code) {
        const double *p = tape.params.data() + ins.param_first;
        switch (ins.op) {
        case G_AFFINE3: case G_STRETCH3: case G_MIRROR3: case G_TWIST: case G_ELONGATE: case G_REVOLVE:
        case G_AFFINE2: case G_STRETCH2: case G_MIRROR2: case G_CLAMP3: case G_TRANSLATE3: {
            double out[3];
            map_point(ins.op, p, P[ins.a], out);
            std::copy(out, out + 3, P[ins.dst]);
            break;
        }
        case G_POINT3: {
            const double out[3] = {V[ins.a], V[ins.b], V[ins.c]};
            std::copy(out, out + 3, P[ins.dst]);
            break;
        }
        default: {
            double value;
            switch (ins.op) {
            case G_MESH: {
                const double *q = P[ins.a];
                eval_triangle_mesh(graph.meshes[ins.aux], 1, q, q + 1, q + 2, &value);
                break;
            }
            case G_POLYGON:
                value = polygon_value(graph.polygons[ins.aux], tape.ranges.data() + ins.range_first, ins.range_count,
                                      P[ins.a][0], P[ins.a][1]);
                break;
            case G_COORD: value = P[ins.a][ins.aux]; break;
            case G_SPHERE: case G_BOX: case G_CYLINDER: case G_CONE: case G_CAPSULE: case G_TORUS: case G_ELLIPSOID:
            case G_HALFSPACE: case G_CIRCLE: case G_RECT: case G_SEGMENT: case G_ELLIPSE: case G_HALFPLANE:
                value = leaf_value(ins.op, p, P[ins.a]);
                break;
            case G_EXTRUDE: case G_TAPER: case G_IMPLICIT: value = value_op(ins.op, p, V[ins.a], 0.0, P[ins.b]); break;
            case G_LOFT: value = value_op(ins.op, p, V[ins.a], V[ins.b], P[ins.c]); break;
            case G_CONST: value = p[0]; break;
            default: {
                const OpSpec s = spec(ins.op);
                value = value_op(ins.op, p, V[ins.a], s.b == K_VALUE ? V[ins.b] : 0.0, nullptr);
                break;
            }
            }
            if (ins.post) value = apply_post(ins, value, V[ins.post_reg]);
            V[ins.dst] = value;
            break;
        }
        }
    }
    return V[tape.result];
}

// ------------------------------------------------------------------ regions

// The set of points a node is evaluated at within a brick: a ball and a box
// of the node's own coordinates, each containing the set. An unbounded
// region (EVAL_AT shapes, full tapes) prunes nothing.
struct Region {
    bool bounded = false;
    int dim = 3;
    double c[3] = {0, 0, 0};
    double r = 0;
    double lo[3] = {0, 0, 0}, hi[3] = {0, 0, 0};
};

inline double pad(double value) { return value * (1 + 1e-12) + 1e-300; }

void finish_region(Region &region) {
    for (int a = 0; a < region.dim; ++a) {
        const double w = 1e-12 * (std::fabs(region.lo[a]) + std::fabs(region.hi[a])) + 1e-300;
        region.lo[a] = std::fmax(region.lo[a] - w, region.c[a] - region.r);
        region.hi[a] = std::fmin(region.hi[a] + w, region.c[a] + region.r);
        if (region.lo[a] > region.hi[a]) region.lo[a] = region.hi[a] = region.c[a];
    }
    for (int a = region.dim; a < 3; ++a) region.lo[a] = region.hi[a] = region.c[a] = 0;
}

// The box of { m (p - t) : p in [lo, hi] } (dim x dim, row-major m).
void affine_box(const double *m, const double *t, const double *lo, const double *hi, int dim, double *out_lo,
                double *out_hi) {
    for (int i = 0; i < dim; ++i) {
        double a = 0, b = 0;
        for (int j = 0; j < dim; ++j) {
            const double c = m[i * dim + j], l = lo[j] - t[j], h = hi[j] - t[j];
            if (c > 0) {
                a += c * l;
                b += c * h;
            } else if (c < 0) {
                a += c * h;
                b += c * l;
            }
        }
        out_lo[i] = a;
        out_hi[i] = b;
    }
}

// The child region of a point-mapping node (G params p of `op`).
Region map_region(uint8_t op, const double *p, const Region &in) {
    Region out;
    out.bounded = in.bounded;
    if (!in.bounded) return out;
    out.dim = (op == G_REVOLVE || op == G_AFFINE2 || op == G_STRETCH2 || op == G_MIRROR2) ? 2 : 3;
    double center[3];
    map_point(op, p, in.c, center);
    std::copy(center, center + 3, out.c);
    out.r = in.r;
    switch (op) {
    case G_AFFINE3: {
        double m[9];
        for (int i = 0; i < 3; ++i)
            for (int j = 0; j < 3; ++j) m[i * 3 + j] = p[j * 3 + i] * p[12];
        affine_box(m, p + 9, in.lo, in.hi, 3, out.lo, out.hi);
        out.r = in.r * p[12];
        break;
    }
    case G_AFFINE2: {
        const double m[4] = {p[0] * p[6], p[2] * p[6], p[1] * p[6], p[3] * p[6]};
        affine_box(m, p + 4, in.lo, in.hi, 2, out.lo, out.hi);
        out.r = in.r * p[6];
        break;
    }
    case G_STRETCH3: case G_STRETCH2: {
        const int dim = op == G_STRETCH3 ? 3 : 2;
        double largest = 0;
        for (int a = 0; a < dim; ++a) {
            out.lo[a] = in.lo[a] * p[a];
            out.hi[a] = in.hi[a] * p[a];
            largest = std::fmax(largest, p[a]);
        }
        out.r = in.r * largest;
        break;
    }
    case G_MIRROR3: case G_MIRROR2: {
        const int dim = op == G_MIRROR3 ? 3 : 2;
        double m[9], t[3] = {0, 0, 0};
        // x - 2 (n.x - d) n = (I - 2 n n^T) (x - d n)
        for (int i = 0; i < dim; ++i) {
            for (int j = 0; j < dim; ++j) m[i * dim + j] = (i == j ? 1.0 : 0.0) - 2 * p[i] * p[j];
            t[i] = p[dim] * p[i];
        }
        double lo[3], hi[3];
        affine_box(m, t, in.lo, in.hi, dim, lo, hi);
        for (int a = 0; a < dim; ++a) {
            out.lo[a] = lo[a] + t[a];
            out.hi[a] = hi[a] + t[a];
        }
        break;
    }
    case G_TWIST: {
        const double rx = std::fmax(std::fabs(in.lo[0]), std::fabs(in.hi[0]));
        const double ry = std::fmax(std::fabs(in.lo[1]), std::fabs(in.hi[1]));
        const double rho = std::hypot(rx, ry);
        out.lo[0] = out.lo[1] = -rho;
        out.hi[0] = out.hi[1] = rho;
        out.lo[2] = in.lo[2];
        out.hi[2] = in.hi[2];
        // The twist map stretches by at most sigma(rate rho), sigma(w) = (w + sqrt(w^2 + 4)) / 2.
        const double w = std::fabs(p[0]) * (std::hypot(in.c[0], in.c[1]) + in.r);
        out.r = in.r * 0.5 * (w + std::sqrt(w * w + 4));
        break;
    }
    case G_ELONGATE:
        for (int a = 0; a < 3; ++a) {
            out.lo[a] = in.lo[a] - std::fmin(std::fmax(in.lo[a], -p[a]), p[a]);
            out.hi[a] = in.hi[a] - std::fmin(std::fmax(in.hi[a], -p[a]), p[a]);
        }
        break;
    case G_REVOLVE: {
        const double gx = std::fmax(std::fmax(in.lo[0], -in.hi[0]), 0.0);
        const double gy = std::fmax(std::fmax(in.lo[1], -in.hi[1]), 0.0);
        const double far = std::hypot(std::fmax(std::fabs(in.lo[0]), std::fabs(in.hi[0])),
                                      std::fmax(std::fabs(in.lo[1]), std::fabs(in.hi[1])));
        out.lo[0] = std::hypot(gx, gy) - p[0];
        out.hi[0] = far - p[0];
        out.lo[1] = in.lo[2];
        out.hi[1] = in.hi[2];
        break;
    }
    case G_CLAMP3:
        for (int a = 0; a < 3; ++a) {
            out.lo[a] = std::fmin(std::fmax(in.lo[a], p[a]), p[3 + a]);
            out.hi[a] = std::fmin(std::fmax(in.hi[a], p[a]), p[3 + a]);
        }
        break;
    case G_TRANSLATE3:
        for (int a = 0; a < 3; ++a) {
            out.lo[a] = in.lo[a] - p[a];
            out.hi[a] = in.hi[a] - p[a];
        }
        break;
    default: break;
    }
    out.r = pad(out.r);
    finish_region(out);
    return out;
}

// The region of a 2D profile read at (x, y).
Region profile_region(const Region &in) {
    Region out = in;
    out.dim = 2;
    if (in.bounded) finish_region(out);
    return out;
}

// The distance from a region to a node's box (0 when they may meet).
double region_distance(const Region &region, const Node &node) {
    const double box = box_distance(region.lo, region.hi, node.lo, node.hi, region.dim);
    double sum = 0;
    for (int a = 0; a < region.dim; ++a) {
        const double gap = std::fmax(node.lo[a] - region.c[a], region.c[a] - node.hi[a]);
        if (gap > 0) sum += gap * gap;
    }
    return std::fmax(box, std::sqrt(sum) - region.r);
}

// ------------------------------------------------------------------ compiler

constexpr uint32_t kConstNode = 0xFFFFFFFFu;

// A node of the pruned tree. Children follow the evaluation order of the
// node (booleans: their kept operands in fold order; IMPLICIT: its shape
// reads in tape order) and precede it in Job::nodes.
struct PNode {
    uint32_t node = kConstNode;
    uint32_t first = 0, count = 0;  // children in Job::kids
    double constant = 0;            // kConstNode: the value
    uint32_t range_first = 0, range_count = 0;
    uint32_t need = 1;              // value registers its evaluation needs
};

struct Walked {
    uint32_t pnode = 0;
    double value = 0, lb = -INFINITY, ub = INFINITY;
};

struct Job {
    std::vector<PNode> nodes;
    std::vector<uint32_t> kids;
    std::vector<uint32_t> ranges;
    double scale = 1;  // coordinate magnitude of the brick, for decision slack
    uint32_t limit = 0;
    uint32_t root = 0;  // the pruned tree's root
};

class Compiler {
public:
    Compiler(const rsdf_graph &graph, uint32_t max_instructions) : g_(graph), max_instructions_(max_instructions) {
        // Per-node parameters in the form the tapes use, computed once.
        node_params_.resize(graph.nodes.size());
        node_ops_.resize(graph.nodes.size(), {0, 0});
        for (uint32_t id = 0; id < graph.nodes.size(); ++id) prepare(id);
    }

    // The pruned tree of `root` for the region: walked from the graph, or
    // from `source`, a pruned tree of a region holding this one. Throws Limit.
    Job prune(uint32_t root, const Region &region, double scale, const Job *source = nullptr) const {
        Job job;
        job.scale = scale;
        job.limit = max_instructions_;
        job.root = walk(job, source, source ? source->root : root, region, 0).pnode;
        return job;
    }

    // The tape of a pruned tree; throws Limit.
    Tape emit(Job &job) const { return emit_tape(job, job.root); }

    // The pruned tape of `root` for the region; throws Limit.
    Tape compile(uint32_t root, const Region &region, double scale) const {
        Job job = prune(root, region, scale);
        return emit(job);
    }

    // The value of `root` at the region's centre and its bounds over the
    // region (the pruning walk's own rules, used to classify bricks).
    void bounds(uint32_t root, const Region &region, double value, double &lb, double &ub, double &lipschitz) const {
        node_bounds(root, region, value, lb, ub, lipschitz);
    }

private:
    const rsdf_graph &g_;
    uint32_t max_instructions_;
    std::vector<std::vector<double>> node_params_;  // map (or leaf) parameters
    std::vector<std::array<uint8_t, 2>> node_ops_;   // (map or leaf op, finishing op)
    std::vector<std::vector<double>> finish_params_storage_;

    std::vector<double> &finish_params(uint32_t id) const {
        return const_cast<std::vector<std::vector<double>> &>(finish_params_storage_)[id];
    }

    void prepare(uint32_t id) {
        if (finish_params_storage_.size() < g_.nodes.size()) finish_params_storage_.resize(g_.nodes.size());
        const Node &node = g_.nodes[id];
        if (node.field) return;
        const double *p = g_.param_values(node);
        std::vector<double> &q = node_params_[id];
        std::vector<double> &f = finish_params_storage_[id];
        uint8_t op = 0, finish = 0;
        switch (node.op) {
        case OP_SPHERE: op = G_SPHERE; q = {p[0]}; break;
        case OP_BOX: op = G_BOX; q = {p[0] - p[3], p[1] - p[3], p[2] - p[3], p[3]}; break;
        case OP_CYLINDER: op = G_CYLINDER; q = {p[0], p[1]}; break;
        case OP_CONE: {
            op = G_CONE;
            const double r1 = p[0], r2 = p[1], h = p[2];
            const double kx = r2 - r1, ky = 2 * h, inv = 1 / (kx * kx + ky * ky);
            q = {r1, r2, h, kx, ky, inv};
            break;
        }
        case OP_CAPSULE: op = G_CAPSULE; q = {p[0], p[1], p[2], p[3] - p[0], p[4] - p[1], p[5] - p[2], node.c0, p[6]}; break;
        case OP_TORUS: op = G_TORUS; q = {p[0], p[1]}; break;
        case OP_ELLIPSOID: op = G_ELLIPSOID; q = {1 / p[0], 1 / p[1], 1 / p[2], node.c0}; break;
        case OP_HALFSPACE: op = G_HALFSPACE; q = {p[0], p[1], p[2], p[3]}; break;
        case OP_MESH: op = G_MESH; break;
        case OP_CIRCLE: op = G_CIRCLE; q = {p[0]}; break;
        case OP_RECT: op = G_RECT; q = {p[0] - p[2], p[1] - p[2], p[2]}; break;
        case OP_POLYGON: op = G_POLYGON; break;
        case OP_SEGMENT: op = G_SEGMENT; q = {p[0], p[1], p[2] - p[0], p[3] - p[1], node.c0, p[4]}; break;
        case OP_ELLIPSE: op = G_ELLIPSE; q = {1 / p[0], 1 / p[1], node.c0}; break;
        case OP_HALFPLANE: op = G_HALFPLANE; q = {p[0], p[1], p[2]}; break;
        case OP_TRANSFORM: {
            static const double kIdentity[9] = {1, 0, 0, 0, 1, 0, 0, 0, 1};
            finish = G_SCALE;
            f = {p[12]};
            if (p[12] == 1 && std::equal(p, p + 9, kIdentity)) {
                op = G_TRANSLATE3;
                q = {p[9], p[10], p[11]};
                break;
            }
            op = G_AFFINE3;
            q.assign(p, p + 12);
            q.push_back(1 / p[12]);
            break;
        }
        case OP_STRETCH: op = G_STRETCH3; q = {1 / p[0], 1 / p[1], 1 / p[2]}; finish = G_SCALE; f = {node.c0}; break;
        case OP_MIRROR: op = G_MIRROR3; q = {p[0], p[1], p[2], p[3]}; break;
        case OP_TWIST: op = G_TWIST; q = {p[0]}; finish = G_SCALE; f = {node.c0}; break;
        case OP_ELONGATE: op = G_ELONGATE; q = {p[0], p[1], p[2]}; break;
        case OP_REVOLVE: op = G_REVOLVE; q = {p[0]}; break;
        case OP_TRANSFORM2:
            op = G_AFFINE2;
            q = {p[0], p[1], p[2], p[3], p[4], p[5], 1 / p[6]};
            finish = G_SCALE;
            f = {p[6]};
            break;
        case OP_STRETCH2: op = G_STRETCH2; q = {1 / p[0], 1 / p[1]}; finish = G_SCALE; f = {node.c0}; break;
        case OP_MIRROR2: op = G_MIRROR2; q = {p[0], p[1], p[2]}; break;
        case OP_OFFSET: case OP_OFFSET2: finish = G_OFFSET; f = {p[0]}; break;
        case OP_SHELL:
            finish = p[1] == 0 ? G_SHELL_IN : p[1] == 1 ? G_SHELL_OUT : G_SHELL_MID;
            f = {p[1] == 2 ? 0.5 * p[0] : p[0]};
            break;
        case OP_OUTLINE2: finish = G_OUTLINE; f = {0.5 * p[0]}; break;
        case OP_EXTRUDE: finish = G_EXTRUDE; f = {0.5 * (p[0] + p[1]), 0.5 * (p[1] - p[0])}; break;
        case OP_TAPERED_EXTRUDE:
            finish = G_TAPER;
            f = {p[0], 1 / (p[1] - p[0]), p[2], node.c0, 0.5 * (p[0] + p[1]), 0.5 * (p[1] - p[0])};
            break;
        case OP_LOFT: finish = G_LOFT; f = {p[0], 1 / (p[1] - p[0]), node.c0, 0.5 * (p[0] + p[1]), 0.5 * (p[1] - p[0])}; break;
        case OP_UNION: case OP_UNION2: finish = p[0] > 0 ? G_SMIN : G_MIN; f = {p[0]}; break;
        case OP_INTERSECTION: case OP_INTERSECTION2: finish = p[0] > 0 ? G_SMAX : G_MAX; f = {p[0]}; break;
        case OP_DIFFERENCE: case OP_DIFFERENCE2: finish = p[0] > 0 ? G_SDIFF : G_DIFF; f = {p[0]}; break;
        case OP_IMPLICIT:
            op = G_CLAMP3;
            q.assign(p, p + 6);
            finish = G_IMPLICIT;
            f = {0.5 * (p[0] + p[3]), 0.5 * (p[1] + p[4]), 0.5 * (p[2] + p[5]),
                 0.5 * (p[3] - p[0]), 0.5 * (p[4] - p[1]), 0.5 * (p[5] - p[2]), node.c0};
            break;
        default: break;
        }
        node_ops_[id] = {op, finish};
    }

    // ---------------------------------------------------------- walking

    double slack(const Job &job, double a, double b, double k) const {
        return kSlack * (std::fabs(a) + std::fabs(b) + k + job.scale);
    }

    // low - high >= k everywhere, with rounding slack.
    bool dominates(const Job &job, double low, double high, double k) const {
        return low - high - k >= slack(job, low, high, k);
    }

    void node_bounds(uint32_t id, const Region &region, double value, double &lb, double &ub,
                     double &lipschitz) const {
        const Node &node = g_.nodes[id];
        lipschitz = node.regional ? shape_lipschitz(g_, id, region.lo, region.hi) : node.step_lipschitz;
        const double spread = lipschitz * region.r;
        lb = value - spread;
        ub = value + spread;
        const double distance = region_distance(region, node);
        if (distance > 0) lb = std::fmax(lb, node.ratio * distance);
    }

    uint32_t add_node(Job &job, uint32_t id, const uint32_t *children, uint32_t count) const {
        PNode pn;
        pn.node = id;
        pn.first = static_cast<uint32_t>(job.kids.size());
        pn.count = count;
        job.kids.insert(job.kids.end(), children, children + count);
        job.nodes.push_back(pn);
        if (job.nodes.size() > 4 * uint64_t(job.limit))
            throw Limit(format("a brick's tape exceeds %u instructions", job.limit));
        return static_cast<uint32_t>(job.nodes.size() - 1);
    }

    uint32_t add_constant(Job &job, double value) const {
        PNode pn;
        pn.constant = value;
        job.nodes.push_back(pn);
        return static_cast<uint32_t>(job.nodes.size() - 1);
    }

    struct Mark {
        size_t nodes, kids, ranges;
    };
    static Mark mark(const Job &job) { return {job.nodes.size(), job.kids.size(), job.ranges.size()}; }
    static void truncate(Job &job, const Mark &m) {
        job.nodes.resize(m.nodes);
        job.kids.resize(m.kids);
        job.ranges.resize(m.ranges);
    }

    // The children of a large boolean (after the first) that a threshold of
    // `threshold` (child value at least it everywhere culls) cannot cull, in
    // child order (collect_candidates in evaluate.cpp).
    void tree_candidates(const Job &job, const Node &node, const Region &region, double threshold,
                         std::vector<uint32_t> &out) const {
        const ChildTree &tree = g_.child_trees[node.child_tree - 1];
        const uint32_t *kids = g_.child_ids(node);
        out.insert(out.end(), tree.always.begin(), tree.always.end());
        const auto keep = [&](double ratio, const double *lo, const double *hi) {
            const double distance = box_distance(region.lo, region.hi, lo, hi, region.dim);
            return !(distance > 0 && ratio * distance - threshold >= slack(job, ratio * distance, threshold, 0));
        };
        uint32_t stack[64];
        int top = 0;
        if (!tree.branches.empty()) stack[top++] = 0;
        while (top > 0) {
            const uint32_t index = stack[--top];
            const ChildTree::Branch &branch = tree.branches[index];
            if (!keep(branch.ratio, branch.lo, branch.hi)) continue;
            if (branch.right != 0) {
                stack[top++] = branch.right;
                stack[top++] = index + 1;
                continue;
            }
            for (uint32_t i = branch.first; i < branch.first + branch.count; ++i) {
                const Node &c = g_.nodes[kids[tree.slots[i]]];
                if (keep(c.ratio, c.lo, c.hi)) out.push_back(tree.slots[i]);
            }
        }
        std::sort(out.begin(), out.end());
    }

    // Walks node `ref`: a graph node, or with a source a node of its pruned
    // tree, whose children are the operands it kept.
    Walked walk(Job &job, const Job *src, uint32_t ref, const Region &region, int depth) const {
        if (depth > kMaxDepth) throw Limit(format("nesting exceeds %d levels", kMaxDepth));
        if (src && src->nodes[ref].node == kConstNode) {
            Walked w;
            w.value = w.lb = w.ub = src->nodes[ref].constant;
            w.pnode = add_constant(job, w.value);
            return w;
        }
        const uint32_t id = src ? src->nodes[ref].node : ref;
        const Node &node = g_.nodes[id];
        const uint32_t child_count = src ? src->nodes[ref].count : node.child_count;
        const uint32_t *kids = src ? src->kids.data() + src->nodes[ref].first : g_.child_ids(node);
        const bool bounded = region.bounded;
        const auto [op, finish] = node_ops_[id];
        const double *q = node_params_[id].data();
        const double *f = finish_params(id).data();
        Walked w;
        double extra_lb = -INFINITY, extra_ub = INFINITY;  // bounds from the operands of booleans
        switch (node.op) {
        case OP_SPHERE: case OP_BOX: case OP_CYLINDER: case OP_CONE: case OP_CAPSULE: case OP_TORUS:
        case OP_ELLIPSOID: case OP_HALFSPACE: case OP_CIRCLE: case OP_RECT: case OP_SEGMENT: case OP_ELLIPSE:
        case OP_HALFPLANE:
            w.pnode = add_node(job, id, nullptr, 0);
            if (bounded) w.value = leaf_value(op, q, region.c);
            break;
        case OP_MESH:
            w.pnode = add_node(job, id, nullptr, 0);
            if (bounded) eval_triangle_mesh(g_.meshes[node.mesh], 1, region.c, region.c + 1, region.c + 2, &w.value);
            break;
        case OP_POLYGON: {
            const Polygon &polygon = g_.polygons[node.polygon];
            const uint32_t count = static_cast<uint32_t>(polygon.x.size());
            const uint32_t first = static_cast<uint32_t>(job.ranges.size());
            uint32_t ranges = kTreeSearch;
            if (count <= kPolygonAll) {
                job.ranges.push_back(0);
                job.ranges.push_back(count);
                ranges = 1;
            } else if (bounded && polygon_candidates(polygon, region.lo, region.hi, kPolygonCandidates, job.ranges)) {
                ranges = static_cast<uint32_t>(job.ranges.size() - first) / 2;
            }
            w.pnode = add_node(job, id, nullptr, 0);
            job.nodes[w.pnode].range_first = first;
            job.nodes[w.pnode].range_count = ranges;
            if (bounded)
                w.value = polygon_value(polygon, job.ranges.data() + first, ranges, region.c[0], region.c[1]);
            break;
        }
        case OP_TRANSFORM: case OP_STRETCH: case OP_MIRROR: case OP_TWIST: case OP_ELONGATE: case OP_REVOLVE:
        case OP_TRANSFORM2: case OP_STRETCH2: case OP_MIRROR2: {
            const Walked child = walk(job, src, kids[0], map_region(op, q, region), depth + 1);
            w.pnode = add_node(job, id, &child.pnode, 1);
            if (bounded) w.value = finish ? value_op(finish, f, child.value, 0, nullptr) : child.value;
            break;
        }
        case OP_OFFSET: case OP_OFFSET2: case OP_SHELL: case OP_OUTLINE2: {
            const Walked child = walk(job, src, kids[0], region, depth + 1);
            w.pnode = add_node(job, id, &child.pnode, 1);
            if (bounded) w.value = value_op(finish, f, child.value, 0, nullptr);
            break;
        }
        case OP_EXTRUDE: case OP_TAPERED_EXTRUDE: case OP_LOFT: {
            const Region profile = profile_region(region);
            const uint32_t profiles = node.op == OP_LOFT ? 2 : 1;
            Walked child[2];
            const Mark before = mark(job);
            for (uint32_t i = 0; i < profiles; ++i) child[i] = walk(job, src, kids[i], profile, depth + 1);
            uint32_t pnodes[2] = {child[0].pnode, child[1].pnode};
            if (bounded) {
                // The slab ignores the profile's exact value where the
                // profile (after the taper or blend) lies at or below
                // min(w, 0) everywhere: slab(d, w) = slab(c, w) bitwise.
                const double center = f[node.op == OP_EXTRUDE ? 0 : node.op == OP_LOFT ? 3 : 4];
                const double half = f[node.op == OP_EXTRUDE ? 1 : node.op == OP_LOFT ? 4 : 5];
                const double zlo = region.lo[2], zhi = region.hi[2];
                const double wlo = (zlo <= center && center <= zhi)
                                       ? -half
                                       : std::fmin(std::fabs(zlo - center), std::fabs(zhi - center)) - half;
                const double level = std::fmin(wlo, 0.0);
                double high;
                if (node.op == OP_EXTRUDE) {
                    high = child[0].ub;
                } else {
                    const double tlo = clamp01((zlo - f[0]) * f[1]), thi = clamp01((zhi - f[0]) * f[1]);
                    if (node.op == OP_TAPERED_EXTRUDE) {
                        high = std::fmax((child[0].ub - f[2] * tlo) * f[3], (child[0].ub - f[2] * thi) * f[3]);
                    } else {
                        high = std::fmax(((1 - tlo) * child[0].ub + tlo * child[1].ub) * f[2],
                                         ((1 - thi) * child[0].ub + thi * child[1].ub) * f[2]);
                    }
                }
                if (std::isfinite(high) && dominates(job, level, high, 0)) {
                    // Any constant at or below the profile's bound works; a
                    // fixed one lets bricks share tapes.
                    truncate(job, before);
                    for (uint32_t i = 0; i < profiles; ++i) pnodes[i] = add_constant(job, std::fmin(child[i].ub, kDeepInside));
                }
                const double z[3] = {region.c[0], region.c[1], region.c[2]};
                w.value = value_op(finish, f, child[0].value, node.op == OP_LOFT ? child[1].value : 0.0, z);
            }
            w.pnode = add_node(job, id, pnodes, profiles);
            break;
        }
        case OP_UNION: case OP_UNION2: case OP_INTERSECTION: case OP_INTERSECTION2: case OP_DIFFERENCE:
        case OP_DIFFERENCE2: {
            // The fold over the kept operands has these bounds: a union is
            // at most every kept operand's upper bound and at least their
            // smallest lower bound minus k x_(m-1) (smooth_union_reach in
            // graph.cpp); an intersection symmetrically; a difference at
            // least its base and every negated cutter.
            const double k = g_.param_values(node)[0];
            const bool is_union = node.op == OP_UNION || node.op == OP_UNION2;
            const bool is_difference = node.op == OP_DIFFERENCE || node.op == OP_DIFFERENCE2;
            const bool is_intersection = !is_union && !is_difference;
            std::vector<Walked> kept;
            double acc = 0, ub_acc = INFINITY, lb_acc = -INFINITY, extreme = 0, reach = 0;
            const auto restart = [&](const Walked &child) {
                kept.assign(1, child);
                acc = child.value;
                ub_acc = child.ub;
                lb_acc = child.lb;
                extreme = is_union ? child.lb : child.ub;
                reach = 0;
            };
            restart(walk(job, src, kids[0], region, depth + 1));
            std::vector<uint32_t> slots;
            if (bounded && !src && node.child_tree && !is_intersection) {
                // A child is culled where its value is at least the threshold.
                const double threshold = is_union ? ub_acc + k : k - lb_acc;
                if (std::isfinite(threshold)) tree_candidates(job, node, region, threshold, slots);
                else for (uint32_t i = 1; i < child_count; ++i) slots.push_back(i);
            } else {
                for (uint32_t i = 1; i < child_count; ++i) slots.push_back(i);
            }
            for (uint32_t slot : slots) {
                const uint32_t cid = kids[slot];
                const bool constant = src && src->nodes[cid].node == kConstNode;
                const Node &c = g_.nodes[constant ? id : src ? src->nodes[cid].node : cid];
                if (bounded && !is_intersection && !constant) {
                    const double distance = region_distance(region, c);
                    if (distance > 0) {
                        const double low = c.ratio * distance;
                        if (is_union ? dominates(job, low, ub_acc, k) : dominates(job, low, -lb_acc, k)) continue;
                    }
                }
                const Mark before = mark(job);
                const Walked child = walk(job, src, cid, region, depth + 1);
                if (!bounded) {
                    kept.push_back(child);
                    continue;
                }
                // Dropped: the operand leaves the fold unchanged (blend
                // weight 0, the fold's side of the min or max). Restarted:
                // the operand replaces the whole fold so far, whose blend
                // with it is then exactly the operand.
                bool drop = false, replace = false;
                if (is_union) {
                    drop = dominates(job, child.lb, ub_acc, k);
                    replace = !drop && dominates(job, lb_acc, child.ub, k);
                } else if (is_difference) {
                    drop = dominates(job, child.lb, -lb_acc, k);
                } else {
                    drop = dominates(job, lb_acc, child.ub, k);
                    replace = !drop && dominates(job, child.lb, ub_acc, k);
                }
                if (drop) {
                    truncate(job, before);
                    continue;
                }
                if (replace) {
                    restart(child);
                    continue;
                }
                kept.push_back(child);
                acc = value_op(finish, f, acc, child.value, nullptr);
                if (k > 0) reach += 0.25 * (1 - reach) * (1 - reach);
                if (is_union) {
                    ub_acc = std::fmin(ub_acc, child.ub);
                    extreme = std::fmin(extreme, child.lb);
                    lb_acc = extreme - k * reach;
                } else if (is_difference) {
                    lb_acc = std::fmax(lb_acc, -child.ub);
                } else {
                    lb_acc = std::fmax(lb_acc, child.lb);
                    extreme = std::fmax(extreme, child.ub);
                    ub_acc = extreme + k * reach;
                }
            }
            if (bounded && k == 0 && kept.size() > 1) {
                // Exact minima and maxima do not depend on the order of
                // their operands: an operand that some other operand
                // dominates everywhere can go, wherever it stands (the
                // dominating operand itself always stays).
                std::vector<Walked> remaining;
                if (is_union) {
                    double low = INFINITY;
                    for (const Walked &item : kept) low = std::fmin(low, item.ub);
                    for (const Walked &item : kept)
                        if (!dominates(job, item.lb, low, 0)) remaining.push_back(item);
                } else if (is_intersection) {
                    double high = -INFINITY;
                    for (const Walked &item : kept) high = std::fmax(high, item.lb);
                    for (const Walked &item : kept)
                        if (!dominates(job, high, item.ub, 0)) remaining.push_back(item);
                } else {
                    double high = kept[0].lb;
                    for (size_t i = 1; i < kept.size(); ++i) high = std::fmax(high, -kept[i].ub);
                    remaining.push_back(kept[0]);
                    for (size_t i = 1; i < kept.size(); ++i)
                        if (!dominates(job, kept[i].lb, -high, 0)) remaining.push_back(kept[i]);
                }
                kept.swap(remaining);
            }
            std::vector<uint32_t> pnodes(kept.size());
            for (size_t i = 0; i < kept.size(); ++i) pnodes[i] = kept[i].pnode;
            w.pnode = add_node(job, id, pnodes.data(), static_cast<uint32_t>(pnodes.size()));
            w.value = acc;
            if (is_union) extra_ub = ub_acc;
            else extra_lb = lb_acc;
            break;
        }
        case OP_IMPLICIT: {
            const FieldTape &tape = g_.tapes[node.tape - 1];
            const Region base = map_region(G_CLAMP3, q, region);
            const Region free_region;
            std::vector<uint32_t> pnodes;
            for (const FieldTape::Instr &ins : tape.code) {
                if (ins.op != FieldTape::kShape && ins.op != OP_EVAL_AT) continue;
                const uint32_t shape = src ? kids[pnodes.size()] : ins.op == OP_EVAL_AT ? ins.shape : ins.node;
                pnodes.push_back(walk(job, src, shape, ins.op == OP_EVAL_AT ? free_region : base, depth + 1).pnode);
            }
            w.pnode = add_node(job, id, pnodes.data(), static_cast<uint32_t>(pnodes.size()));
            if (bounded) eval_batch(g_, id, 1, region.c, region.c + 1, region.c + 2, &w.value);
            break;
        }
        default:
            throw Error(format("node %u (%s) cannot be compiled for the GPU", id, op_name(node.op)));
        }
        if (bounded) {
            double lipschitz;
            node_bounds(id, region, w.value, w.lb, w.ub, lipschitz);
            w.lb = std::fmax(w.lb, extra_lb);
            w.ub = std::fmin(w.ub, extra_ub);
        }
        return w;
    }

    // ---------------------------------------------------------- emitting

    struct Emitter {
        const Compiler &compiler;
        const Job &job;
        Tape tape;
        std::vector<uint8_t> kinds;  // per virtual register

        uint32_t fresh(Kind kind) {
            kinds.push_back(kind);
            return static_cast<uint32_t>(kinds.size() - 1);
        }

        uint32_t add(uint8_t op, uint32_t a, uint32_t b, uint32_t c, const double *params, uint32_t count,
                     uint32_t aux = 0) {
            const OpSpec s = spec(op);
            Ins ins;
            ins.op = op;
            ins.dst = fresh(static_cast<Kind>(s.dst));
            ins.a = a;
            ins.b = b;
            ins.c = c;
            ins.aux = aux;
            ins.param_first = static_cast<uint32_t>(tape.params.size());
            ins.param_count = s.params;
            if (count != s.params) throw Error("internal error: GPU instruction parameter count");
            tape.params.insert(tape.params.end(), params, params + count);
            tape.code.push_back(ins);
            if (tape.code.size() > job.limit)
                throw Limit(format("a brick's tape exceeds %u instructions", job.limit));
            return ins.dst;
        }

        uint32_t constant(double value) { return add(G_CONST, 0, 0, 0, &value, 1); }

        static uint32_t combine_need(uint32_t a, uint32_t b) { return a == b ? a + 1 : std::max(a, b); }

        // The fold of children [0, last] of a boolean, evaluating the
        // operand that needs more registers first.
        uint32_t fold(const PNode &pn, uint32_t last, uint32_t point, uint8_t op, const double *k,
                      const std::vector<uint32_t> &prefix) {
            const uint32_t *children = job.kids.data() + pn.first;
            if (last == 0) return emit(children[0], point);
            uint32_t acc, value;
            if (job.nodes[children[last]].need > prefix[last - 1]) {
                value = emit(children[last], point);
                acc = fold(pn, last - 1, point, op, k, prefix);
            } else {
                acc = fold(pn, last - 1, point, op, k, prefix);
                value = emit(children[last], point);
            }
            return add(op, acc, value, 0, k, spec(op).params);
        }

        uint32_t emit(uint32_t index, uint32_t point) {
            const PNode &pn = job.nodes[index];
            if (pn.node == kConstNode) return constant(pn.constant);
            const rsdf_graph &g = compiler.g_;
            const Node &node = g.nodes[pn.node];
            const auto [op, finish] = compiler.node_ops_[pn.node];
            const std::vector<double> &q = compiler.node_params_[pn.node];
            const std::vector<double> &f = compiler.finish_params(pn.node);
            const uint32_t *children = job.kids.data() + pn.first;
            switch (node.op) {
            case OP_MESH: return add(G_MESH, point, 0, 0, nullptr, 0, node.mesh);
            case OP_POLYGON: {
                const uint32_t dst = add(G_POLYGON, point, 0, 0, nullptr, 0, node.polygon);
                Ins &ins = tape.code.back();
                ins.range_first = static_cast<uint32_t>(tape.ranges.size());
                ins.range_count = pn.range_count;
                if (pn.range_count != kTreeSearch)
                    tape.ranges.insert(tape.ranges.end(), job.ranges.begin() + pn.range_first,
                                       job.ranges.begin() + pn.range_first + 2 * pn.range_count);
                return dst;
            }
            case OP_TRANSFORM: case OP_STRETCH: case OP_MIRROR: case OP_TWIST: case OP_ELONGATE: case OP_REVOLVE:
            case OP_TRANSFORM2: case OP_STRETCH2: case OP_MIRROR2: {
                const uint32_t mapped = add(op, point, 0, 0, q.data(), static_cast<uint32_t>(q.size()));
                const uint32_t value = emit(children[0], mapped);
                // Scaling by 1 is exact: rigid transforms need no instruction.
                return finish && f[0] != 1 ? add(finish, value, 0, 0, f.data(), 1) : value;
            }
            case OP_OFFSET: case OP_OFFSET2: case OP_SHELL: case OP_OUTLINE2:
                return add(finish, emit(children[0], point), 0, 0, f.data(), 1);
            case OP_EXTRUDE: case OP_TAPERED_EXTRUDE:
                return add(finish, emit(children[0], point), point, 0, f.data(), static_cast<uint32_t>(f.size()));
            case OP_LOFT: {
                uint32_t a, b;
                if (job.nodes[children[1]].need > job.nodes[children[0]].need) {
                    b = emit(children[1], point);
                    a = emit(children[0], point);
                } else {
                    a = emit(children[0], point);
                    b = emit(children[1], point);
                }
                return add(G_LOFT, a, b, point, f.data(), static_cast<uint32_t>(f.size()));
            }
            case OP_UNION: case OP_UNION2: case OP_INTERSECTION: case OP_INTERSECTION2: case OP_DIFFERENCE:
            case OP_DIFFERENCE2: {
                std::vector<uint32_t> prefix(pn.count);
                for (uint32_t i = 0; i < pn.count; ++i) {
                    const uint32_t need = job.nodes[children[i]].need;
                    prefix[i] = i == 0 ? need : combine_need(prefix[i - 1], need);
                }
                return fold(pn, pn.count - 1, point, finish, f.data(), prefix);
            }
            case OP_IMPLICIT: return emit_implicit(pn, node, q, f, point);
            default:
                if (op >= G_SPHERE && op <= G_HALFPLANE)
                    return add(op, point, 0, 0, q.data(), static_cast<uint32_t>(q.size()));
                throw Error(format("node %u (%s) cannot be compiled for the GPU", pn.node, op_name(node.op)));
            }
        }

        uint32_t emit_implicit(const PNode &pn, const Node &node, const std::vector<double> &q,
                               const std::vector<double> &f, uint32_t point) {
            const rsdf_graph &g = compiler.g_;
            const FieldTape &field = g.tapes[node.tape - 1];
            const uint32_t base = add(G_CLAMP3, point, 0, 0, q.data(), 6);
            uint32_t coords[3] = {UINT32_MAX, UINT32_MAX, UINT32_MAX};
            std::vector<uint32_t> slots(field.slots, UINT32_MAX);
            const auto operand = [&](uint32_t k) -> uint32_t {
                if (k < 3) {
                    if (coords[k] == UINT32_MAX) coords[k] = add(G_COORD, base, 0, 0, nullptr, 0, k);
                    return coords[k];
                }
                return slots[k - 3];
            };
            const uint32_t *children = job.kids.data() + pn.first;
            uint32_t shape = 0;
            for (const FieldTape::Instr &ins : field.code) {
                const uint32_t *args = field.args.data() + ins.first;
                uint32_t value;
                switch (ins.op) {
                case FieldTape::kShape: value = emit(children[shape++], base); break;
                case OP_EVAL_AT: {
                    const uint32_t at = add(G_POINT3, operand(args[0]), operand(args[1]), operand(args[2]), nullptr, 0);
                    value = emit(children[shape++], at);
                    break;
                }
                case OP_CONST: value = constant(ins.p0); break;
                case OP_MIN: case OP_MAX: {
                    value = operand(args[0]);
                    for (uint32_t i = 1; i < ins.count; ++i)
                        value = add(ins.op == OP_MIN ? G_MIN : G_MAX, value, operand(args[i]), 0, nullptr, 0);
                    if (ins.count == 1) value = add(ins.op == OP_MIN ? G_MIN : G_MAX, value, value, 0, nullptr, 0);
                    break;
                }
                case OP_LENGTH: {
                    value = add(G_LENSQ, operand(args[0]), 0, 0, nullptr, 0);
                    for (uint32_t i = 1; i < ins.count; ++i) value = add(G_LENACC, value, operand(args[i]), 0, nullptr, 0);
                    value = add(G_SQRT, value, 0, 0, nullptr, 0);
                    break;
                }
                case OP_POWC:
                    value = ins.p0 == 2 ? add(G_SQUARE, operand(args[0]), 0, 0, nullptr, 0)
                                        : add(G_POWC, operand(args[0]), 0, 0, &ins.p0, 1);
                    break;
                case OP_CLAMP: {
                    const double limits[2] = {ins.p0, ins.p1};
                    value = add(G_CLAMP, operand(args[0]), 0, 0, limits, 2);
                    break;
                }
                case OP_TRIWAVE: value = add(G_TRIWAVE, operand(args[0]), 0, 0, &ins.p0, 1); break;
                case OP_SMIN: case OP_SMAX:
                    value = add(ins.op == OP_SMIN ? G_FSMIN : G_FSMAX, operand(args[0]), operand(args[1]), 0, &ins.p0, 1);
                    break;
                default: {
                    uint8_t gop;
                    switch (ins.op) {
                    case OP_ADD: gop = G_ADD; break;
                    case OP_SUB: gop = G_SUB; break;
                    case OP_MUL: gop = G_MUL; break;
                    case OP_DIV: gop = G_DIV; break;
                    case OP_NEG: gop = G_NEG; break;
                    case OP_ABS: gop = G_ABS; break;
                    case OP_SQUARE: gop = G_SQUARE; break;
                    case OP_SQRT: gop = G_SQRT; break;
                    case OP_SIN: gop = G_SIN; break;
                    case OP_COS: gop = G_COS; break;
                    case OP_EXP: gop = G_EXP; break;
                    case OP_LOG: gop = G_LOG; break;
                    case OP_ATAN2: gop = G_ATAN2; break;
                    default: throw Error(format("field opcode %u cannot be compiled for the GPU", ins.op));
                    }
                    const uint32_t a = operand(args[0]);
                    const uint32_t b = spec(gop).b == K_VALUE ? operand(args[1]) : 0;
                    value = add(gop, a, b, 0, nullptr, 0);
                    break;
                }
                }
                slots[ins.dst - 3] = value;
            }
            const uint32_t result = operand(field.result);
            return add(G_IMPLICIT, result, point, 0, f.data(), 7);
        }
    };

    static void compute_needs(Job &job) {
        for (PNode &pn : job.nodes) {
            const uint32_t *children = job.kids.data() + pn.first;
            if (pn.count == 0) {
                pn.need = 1;
                continue;
            }
            uint32_t need = job.nodes[children[0]].need;
            for (uint32_t i = 1; i < pn.count; ++i) need = Emitter::combine_need(need, job.nodes[children[i]].need);
            pn.need = need;
        }
    }

    Tape emit_tape(Job &job, uint32_t root) const {
        compute_needs(job);
        Emitter emitter{*this, job, Tape(), {}};
        const uint32_t input = emitter.fresh(K_POINT);
        const uint32_t result = emitter.emit(root, input);
        Tape &tape = emitter.tape;
        const size_t count = emitter.kinds.size();
        std::vector<uint32_t> last(count, 0);
        const auto last_reads = [&] {
            std::fill(last.begin(), last.end(), 0);
            for (uint32_t i = 0; i < tape.code.size(); ++i) {
                const Ins &ins = tape.code[i];
                const OpSpec s = spec(ins.op);
                if (s.a) last[ins.a] = i;
                if (s.b) last[ins.b] = i;
                if (s.c) last[ins.c] = i;
                if (post_binary(ins.post)) last[ins.post_reg] = i;
            }
            last[result] = UINT32_MAX;
        };
        last_reads();
        // A fold, scale or offset whose operand the previous instruction
        // computes and nothing else reads becomes that instruction's post
        // operation (one instruction less; the same arithmetic in the same
        // operand order). The previous instruction must leave its record's
        // third-operand word free for the post parameter: not a point map,
        // POINT3 or LOFT.
        {
            std::vector<Ins> code;
            code.reserve(tape.code.size());
            for (uint32_t i = 0; i < tape.code.size(); ++i) {
                Ins ins = tape.code[i];
                const bool producer = spec(ins.op).dst == K_VALUE && ins.op != G_LOFT;
                if (producer && i + 1 < tape.code.size() && last[ins.dst] == i + 1) {
                    const Ins &next = tape.code[i + 1];
                    const OpSpec ns = spec(next.op);
                    uint8_t post = P_NONE;
                    switch (next.op) {
                    case G_MIN: post = P_MIN; break;
                    case G_MAX: post = P_MAX; break;
                    case G_SMIN: post = P_SMIN; break;
                    case G_SMAX: post = P_SMAX; break;
                    case G_DIFF: post = P_DIFF; break;
                    case G_SDIFF: post = P_SDIFF; break;
                    case G_SCALE: post = P_SCALE; break;
                    case G_OFFSET: post = P_OFFSET; break;
                    default: break;
                    }
                    const bool binary = post_binary(post);
                    const bool reads_a = next.a == ins.dst, reads_b = binary && ns.b && next.b == ins.dst;
                    if (post != P_NONE && (binary ? reads_a != reads_b : reads_a)) {
                        ins.post = post;
                        ins.post_swap = binary && reads_b;
                        ins.post_reg = binary ? (reads_b ? next.a : next.b) : 0;
                        ins.post_param = ns.params ? tape.params[next.param_first] : 0.0;
                        ins.dst = next.dst;
                        code.push_back(ins);
                        ++i;
                        continue;
                    }
                }
                code.push_back(ins);
            }
            tape.code.swap(code);
        }
        // Linear scan: an operand's register is free after its last read, so
        // an instruction may write a register it reads (the interpreter reads
        // every operand first).
        last_reads();
        // A point map read only by the next instruction, a leaf, is encoded
        // inside that leaf (no register write; the same arithmetic).
        for (uint32_t i = 0; i + 1 < tape.code.size(); ++i) {
            Ins &map = tape.code[i];
            const Ins &leaf = tape.code[i + 1];
            const bool is_map = (map.op >= G_AFFINE3 && map.op <= G_CLAMP3) || map.op == G_TRANSLATE3;
            const bool is_leaf = leaf.op >= G_SPHERE && leaf.op <= G_COORD;
            map.fused = is_map && is_leaf && leaf.a == map.dst && last[map.dst] == i + 1;
        }
        std::vector<uint32_t> physical(count, UINT32_MAX);
        uint64_t used[2] = {0, 0};  // value and point registers in use (bit sets)
        uint32_t limits[2] = {kValueRegisters, kPointRegisters};
        uint32_t highest[2] = {0, 0};
        const auto take = [&](uint32_t vreg) {
            const int file = emitter.kinds[vreg] == K_POINT ? 1 : 0;
            if (~used[file] == 0) throw Limit(format("a brick's tape needs more than %u %s registers", limits[file],
                                                     file ? "point" : "value"));
            const uint32_t reg = static_cast<uint32_t>(__builtin_ctzll(~used[file]));
            if (reg >= limits[file])
                throw Limit(format("a brick's tape needs more than %u %s registers", limits[file], file ? "point" : "value"));
            used[file] |= uint64_t(1) << reg;
            highest[file] = std::max(highest[file], reg + 1);
            physical[vreg] = reg;
        };
        const auto release = [&](uint32_t vreg, uint32_t at) {
            if (last[vreg] != at || physical[vreg] == UINT32_MAX) return;
            const int file = emitter.kinds[vreg] == K_POINT ? 1 : 0;
            used[file] &= ~(uint64_t(1) << physical[vreg]);
            last[vreg] = UINT32_MAX - 1;  // released (an operand may appear twice)
        };
        take(input);
        for (uint32_t i = 0; i < tape.code.size(); ++i) {
            Ins &ins = tape.code[i];
            const OpSpec s = spec(ins.op);
            const uint32_t a = ins.a, b = ins.b, c = ins.c, other = ins.post_reg;
            const bool binary = post_binary(ins.post);
            if (s.a) ins.a = physical[a];
            if (s.b) ins.b = physical[b];
            if (s.c) ins.c = physical[c];
            if (binary) ins.post_reg = physical[other];
            if (s.a) release(a, i);
            if (s.b) release(b, i);
            if (s.c) release(c, i);
            if (binary) release(other, i);
            take(ins.dst);
            ins.dst = physical[ins.dst];
        }
        tape.result = physical[result];
        tape.values = highest[0];
        tape.points = highest[1];
        return std::move(tape);
    }
};

// ------------------------------------------------------------------ encoding

inline uint32_t float_bits(double value) {
    const float f = static_cast<float>(value);
    uint32_t bits;
    std::memcpy(&bits, &f, 4);
    return bits;
}

constexpr uint32_t kFusedMap = 0x80;  // header flag: the record's w holds a point map applied to P[a] first

// Instructions a tape executes (a fused map and its leaf count once).
uint32_t instruction_count(const Tape &tape) {
    uint32_t count = 0;
    for (const Ins &ins : tape.code) count += ins.fused ? 0 : 1;
    return count;
}

// Tapes with equal content encode to equal words: the hash and equality
// that deduplicate a part's tapes before they are encoded.
uint64_t mix(uint64_t h, uint64_t v) {
    h ^= v + 0x9E3779B97F4A7C15ull + (h << 6) + (h >> 2);
    return h * 0xFF51AFD7ED558CCDull;
}
uint64_t bits64(double v) {
    uint64_t b;
    std::memcpy(&b, &v, 8);
    return b;
}
uint64_t tape_hash(const Tape &tape, double lipschitz) {
    uint64_t h = mix(bits64(lipschitz), tape.result);
    for (const Ins &ins : tape.code) {
        h = mix(h, uint64_t(ins.op) | uint64_t(ins.dst) << 8 | uint64_t(ins.a) << 24 | uint64_t(ins.b) << 40);
        h = mix(h, uint64_t(ins.c) | uint64_t(ins.aux) << 24 | uint64_t(ins.fused) << 48 | uint64_t(ins.post) << 52 |
                       uint64_t(ins.post_swap) << 60);
        h = mix(h, uint64_t(ins.post_reg) | uint64_t(ins.range_count) << 32);
        h = mix(h, bits64(ins.post_param));
    }
    for (const double v : tape.params) h = mix(h, bits64(v));
    for (const uint32_t v : tape.ranges) h = mix(h, v);
    return h;
}
bool same_tape(const Tape &x, const Tape &y) {
    if (x.result != y.result || x.code.size() != y.code.size() || x.params.size() != y.params.size() ||
        x.ranges != y.ranges)
        return false;
    for (size_t i = 0; i < x.code.size(); ++i) {
        const Ins &a = x.code[i], &b = y.code[i];
        if (a.op != b.op || a.dst != b.dst || a.a != b.a || a.b != b.b || a.c != b.c || a.aux != b.aux ||
            a.param_first != b.param_first || a.param_count != b.param_count || a.range_first != b.range_first ||
            a.range_count != b.range_count || a.fused != b.fused || a.post != b.post || a.post_swap != b.post_swap ||
            a.post_reg != b.post_reg || bits64(a.post_param) != bits64(b.post_param))
            return false;
    }
    for (size_t i = 0; i < x.params.size(); ++i)
        if (bits64(x.params[i]) != bits64(y.params[i])) return false;
    return true;
}

// Parameters shared by a part's tapes: word runs deduplicated by content,
// each padded to whole 16-byte records (the shader reads them as float4),
// at offsets relative to the pool's start.
struct Pool {
    std::vector<uint32_t> words;
    std::unordered_map<std::string, uint32_t> index;

    uint32_t add(std::vector<uint32_t> &run) {
        while (run.size() % 4) run.push_back(0);
        const std::string key(reinterpret_cast<const char *>(run.data()), 4 * run.size());
        const auto found = index.find(key);
        if (found != index.end()) return found->second;
        const uint32_t at = static_cast<uint32_t>(words.size());
        words.insert(words.end(), run.begin(), run.end());
        index.emplace(key, at);
        return at;
    }
};

constexpr uint32_t kPoolOffsetMask = 0x0FFFFFFFu;  // record w: map op << 28 | pool offset
constexpr uint32_t kPostSwap = 1u << 28;           // record x: the value is the post fold's second operand

// Whether an instruction's record names a parameter run in the pool (z).
bool has_parameters(uint8_t op) { return op == G_POLYGON || (op != G_MESH && spec(op).params > 0); }

// Appends the tape as 16-byte records, ending with an END record naming the
// result register in a (SDF-DESIGN.md, "Format"): x = op | fused map << 7 |
// dst << 8 | post << 12 | a << 16 | post register << 20 | b << 24 | swap
// << 28; y = third operand, or the post operation's parameter; z = pool
// offset of the parameters (a MESH: its index); w = fused map op << 28 |
// pool offset of its parameters. Pool offsets are relative.
void encode_tape(const Tape &tape, std::vector<uint32_t> &words, Pool &pool) {
    std::vector<uint32_t> run;
    const auto parameters = [&](const Ins &ins) {
        run.clear();
        const double *p = tape.params.data() + ins.param_first;
        if (ins.op == G_POLYGON) {
            run.push_back(ins.aux);
            run.push_back(ins.range_count);
            if (ins.range_count != kTreeSearch)
                run.insert(run.end(), tape.ranges.begin() + ins.range_first,
                           tape.ranges.begin() + ins.range_first + 2 * ins.range_count);
        } else if (ins.op == G_AFFINE3) {
            // t, then M = R^T / s row-major: local = M (q - t).
            for (int i = 0; i < 3; ++i) run.push_back(float_bits(p[9 + i]));
            for (int r = 0; r < 3; ++r)
                for (int c = 0; c < 3; ++c) run.push_back(float_bits(p[3 * c + r] * p[12]));
        } else {
            for (uint32_t i = 0; i < ins.param_count; ++i) run.push_back(float_bits(p[i]));
        }
        return pool.add(run);
    };
    for (size_t index = 0; index < tape.code.size(); ++index) {
        const Ins &ins = tape.code[index];
        if (ins.fused) continue;  // encoded with its leaf
        const OpSpec s = spec(ins.op);
        const uint32_t b = ins.op == G_COORD ? ins.aux : s.b ? ins.b : 0;
        const Ins *map = index > 0 && tape.code[index - 1].fused ? &tape.code[index - 1] : nullptr;
        const uint32_t a = map ? map->a : s.a ? ins.a : 0;
        const uint32_t other = post_binary(ins.post) ? ins.post_reg : 0;
        words.push_back(uint32_t(ins.op) | (map ? kFusedMap : 0) | ins.dst << 8 | uint32_t(ins.post) << 12 | a << 16 |
                        other << 20 | b << 24 | (ins.post_swap ? kPostSwap : 0));
        words.push_back(s.c ? ins.c : ins.post ? float_bits(ins.post_param) : 0);
        words.push_back(ins.op == G_MESH ? ins.aux : has_parameters(ins.op) ? parameters(ins) : 0);
        words.push_back(map ? uint32_t(map->op) << 28 | parameters(*map) : 0);
    }
    words.push_back(uint32_t(G_END) | tape.result << 16);
    words.push_back(0);
    words.push_back(0);
    words.push_back(0);
}

// Rounds outward to float32 (boxes stay conservative after conversion).
float float_down(double value) {
    float f = static_cast<float>(value);
    if (static_cast<double>(f) > value) f = std::nextafter(f, -INFINITY);
    return f;
}
float float_up(double value) {
    float f = static_cast<float>(value);
    if (static_cast<double>(f) < value) f = std::nextafter(f, INFINITY);
    return f;
}
uint32_t bits_of(float f) {
    uint32_t bits;
    std::memcpy(&bits, &f, 4);
    return bits;
}

// Mesh data for the GPU (see "GPU field programs" in SDF-DESIGN.md).
void encode_mesh(const TriangleMesh &m, std::vector<uint32_t> &words, uint32_t *record) {
    const auto put = [&](const std::vector<double> &values) {
        const uint32_t at = static_cast<uint32_t>(words.size());
        for (double v : values) words.push_back(float_bits(v));
        return at;
    };
    const auto put_u32 = [&](const std::vector<uint32_t> &values) {
        const uint32_t at = static_cast<uint32_t>(words.size());
        words.insert(words.end(), values.begin(), values.end());
        return at;
    };
    const auto put_tree = [&](const std::vector<TriangleMesh::Branch> &tree) {
        const uint32_t at = static_cast<uint32_t>(words.size());
        for (const TriangleMesh::Branch &b : tree) {
            for (int a = 0; a < 3; ++a) words.push_back(bits_of(float_down(b.lo[a])));
            for (int a = 0; a < 3; ++a) words.push_back(bits_of(float_up(b.hi[a])));
            words.push_back(b.right);
            words.push_back(b.first);
            words.push_back(b.count);
        }
        return at;
    };
    std::fill(record, record + kMeshWords, 0);
    record[0] = static_cast<uint32_t>(m.positions.size() / 3);
    record[1] = put(m.positions);
    record[2] = put(m.vertex_normals);
    record[3] = static_cast<uint32_t>(m.triangles.size() / 3);
    record[4] = put_u32(m.triangles);
    record[5] = put_u32(m.neighbours);
    record[6] = put(m.normals);
    record[7] = put_tree(m.tree);
    record[8] = static_cast<uint32_t>(m.tree.size());
    record[9] = static_cast<uint32_t>(m.bodies.size());
    record[10] = static_cast<uint32_t>(words.size());
    for (const TriangleMesh::Body &b : m.bodies) {
        for (int a = 0; a < 3; ++a) words.push_back(bits_of(float_down(b.lo[a])));
        for (int a = 0; a < 3; ++a) words.push_back(bits_of(float_up(b.hi[a])));
        words.push_back(b.first_root);
        words.push_back(b.root_count);
    }
    record[11] = put_u32(m.body_roots);
    if (m.bodies.size() > 1) {
        record[12] = put_tree(m.body_tree);
        record[13] = static_cast<uint32_t>(m.body_tree.size());
        record[14] = put_u32(m.body_order);
        record[15] = put_u32(m.triangle_body);
    }
    for (int a = 0; a < 3; ++a) {
        record[16 + a] = bits_of(float_down(m.lo[a]));
        record[19 + a] = bits_of(float_up(m.hi[a]));
    }
}

void encode_polygon(const Polygon &polygon, std::vector<uint32_t> &words, uint32_t *record) {
    std::fill(record, record + kPolygonWords, 0);
    const size_t count = polygon.x.size();
    record[0] = static_cast<uint32_t>(count);
    record[1] = static_cast<uint32_t>(words.size());
    for (size_t i = 0; i < count; ++i) {
        const double values[7] = {polygon.x[i], polygon.y[i], polygon.ex[i], polygon.ey[i], polygon.inv_len2[i],
                                  polygon.nx[i], polygon.ny[i]};
        for (double v : values) words.push_back(float_bits(v));
    }
    record[2] = static_cast<uint32_t>(words.size());
    for (const PolygonTree &node : polygon.tree) {
        words.push_back(bits_of(float_down(node.lo[0])));
        words.push_back(bits_of(float_down(node.lo[1])));
        words.push_back(bits_of(float_up(node.hi[0])));
        words.push_back(bits_of(float_up(node.hi[1])));
        words.push_back(node.begin);
        words.push_back(node.end);
        words.push_back(node.right);
    }
    record[3] = static_cast<uint32_t>(polygon.tree.size());
    const PolygonTree &root = polygon.tree.front();
    record[4] = bits_of(float_down(root.lo[0]));
    record[5] = bits_of(float_down(root.lo[1]));
    record[6] = bits_of(float_up(root.hi[0]));
    record[7] = bits_of(float_up(root.hi[1]));
}

// ------------------------------------------------------------------ threads

// Runs body(chunk) for every chunk on up to `threads` threads with large
// stacks (pruned walks recurse once per nesting level). The first exception
// is rethrown after every thread stops.
void parallel_deep(size_t chunks, int threads, const std::function<void(size_t)> &body) {
    struct Shared {
        const std::function<void(size_t)> *body;
        size_t chunks;
        std::atomic<size_t> next{0};
        std::atomic<bool> failed{false};
        std::mutex mutex;
        std::exception_ptr error;
    } shared;
    shared.body = &body;
    shared.chunks = chunks;
    const auto run = [](void *context) -> void * {
        Shared &s = *static_cast<Shared *>(context);
        while (!s.failed.load(std::memory_order_relaxed)) {
            const size_t chunk = s.next.fetch_add(1, std::memory_order_relaxed);
            if (chunk >= s.chunks) break;
            try {
                (*s.body)(chunk);
            } catch (...) {
                std::lock_guard<std::mutex> lock(s.mutex);
                if (!s.error) s.error = std::current_exception();
                s.failed.store(true, std::memory_order_relaxed);
            }
        }
        return nullptr;
    };
    const int count = std::max(1, std::min<int>(threads, static_cast<int>(std::min<size_t>(chunks, 256))));
    std::vector<pthread_t> handles;
    pthread_attr_t attributes;
    pthread_attr_init(&attributes);
    pthread_attr_setstacksize(&attributes, kStackBytes);
    for (int i = 0; i < count; ++i) {
        pthread_t handle;
        if (pthread_create(&handle, &attributes, run, &shared) == 0) handles.push_back(handle);
    }
    pthread_attr_destroy(&attributes);
    if (handles.empty()) {
        shared.failed.store(true);
        throw Error("could not start compiler threads");
    }
    for (pthread_t handle : handles) pthread_join(handle, nullptr);
    if (shared.error) std::rethrow_exception(shared.error);
}

}  // namespace

// ------------------------------------------------------------------ programs

// A part's cells for the reference interpreter: the top grid, then blocks of
// eight octant cells of refined cells.
struct PartReference {
    bool field = false;
    double origin[3] = {0, 0, 0};
    double brick = 1, margin = 0;
    uint32_t dims[3] = {0, 0, 0};
    double lo[3] = {0, 0, 0}, hi[3] = {0, 0, 0};
    double lipschitz = 1;
    uint32_t root = 0;
    std::vector<int32_t> cell_tape;   // index into Program::tapes, or -1
    std::vector<int32_t> cell_child;  // first octant cell of a refined cell, or -1
    std::vector<double> cell_value;   // band: Lipschitz bound; far: signed distance bound
};

}  // namespace gpu
}  // namespace rsdf

struct rsdf_gpu_program {
    const rsdf_graph *graph = nullptr;
    std::vector<uint32_t> words;
    std::vector<rsdf_gpu_part_info> parts;
    std::vector<rsdf::gpu::PartReference> references;
    std::vector<rsdf::gpu::Tape> tapes;  // distinct tapes, float64 parameters
    uint32_t max_instructions = 0;
    std::mutex full_mutex;
    std::unordered_map<uint32_t, std::unique_ptr<rsdf::gpu::Tape>> full;  // unpruned tapes by root
};

namespace rsdf {
namespace gpu {
namespace {

double box_gap(const double *p, const double *lo, const double *hi) {
    double sum = 0;
    for (int a = 0; a < 3; ++a) {
        const double gap = std::fmax(std::fmax(lo[a] - p[a], p[a] - hi[a]), 0.0);
        sum += gap * gap;
    }
    return std::sqrt(sum);
}

// One brick to classify or compile: its cell, region and level.
struct Item {
    uint32_t cell;
    int level;
    double centre[3];
    double size;   // edge
};

void build_part(rsdf_gpu_program &program, const Compiler &compiler, uint32_t root, uint32_t part_index,
                const rsdf_gpu_options &options, uint64_t max_words, std::vector<uint8_t> &mesh_done,
                std::vector<uint8_t> &polygon_done, uint32_t part_record_at, uint32_t mesh_table_at,
                uint32_t polygon_table_at) {
    const auto started = std::chrono::steady_clock::now();
    const rsdf_graph &g = *program.graph;
    const Node &node = g.nodes[root];
    rsdf_gpu_part_info &info = program.parts[part_index];
    PartReference &ref = program.references[part_index];
    ref.root = root;
    ref.brick = options.brick_size;
    ref.margin = options.margin;
    ref.lipschitz = node.step_lipschitz;
    for (int a = 0; a < 3; ++a) {
        ref.lo[a] = node.lo[a];
        ref.hi[a] = node.hi[a];
    }
    const int threads = resolve_threads(options.threads);
    const uint32_t refine = options.refine_instructions ? options.refine_instructions : kDefaultRefine;
    const double smallest = options.min_brick_size > 0 ? options.min_brick_size : options.brick_size / 8;
    int levels = 0;  // refinement levels allowed
    while (levels < kMaxLevels && options.brick_size / double(2 << levels) >= smallest * (1 - 1e-9)) ++levels;
    const size_t words_before = program.words.size();
    const size_t tapes_before = program.tapes.size();
    std::vector<size_t> added_meshes, added_polygons;  // encoded for this part (undone if it fails)
    uint32_t part_record[kPartWords] = {};
    try {
        for (int a = 0; a < 3; ++a)
            if (!std::isfinite(node.lo[a]) || !std::isfinite(node.hi[a]) || node.lo[a] > node.hi[a])
                throw Limit("the part's bounds are not finite and nonempty");
        uint64_t cells = 1;
        for (int a = 0; a < 3; ++a) {
            const double extent = node.hi[a] - node.lo[a] + 2 * options.margin;
            const double count = std::ceil(extent / options.brick_size) + 1;
            if (!(count <= 4096)) throw Limit("the part's brick grid is too large");
            ref.dims[a] = static_cast<uint32_t>(std::max(count, 1.0));
            // Centre the grid on the part's box.
            ref.origin[a] = 0.5 * (node.lo[a] + node.hi[a]) - 0.5 * ref.dims[a] * options.brick_size;
            cells *= ref.dims[a];
        }
        if (cells > kMaxGridCells)
            throw Limit(format("the part's brick grid of %llu bricks exceeds %llu", static_cast<unsigned long long>(cells),
                               static_cast<unsigned long long>(kMaxGridCells)));
        info.grid[0] = ref.dims[0];
        info.grid[1] = ref.dims[1];
        info.grid[2] = ref.dims[2];
        info.bricks = cells;
        ref.cell_tape.assign(cells, -1);
        ref.cell_child.assign(cells, -1);
        ref.cell_value.assign(cells, 0);
        const auto region_of = [&](const Item &item, Region &region, double &scale) {
            const double grown = 0.5 * item.size + options.margin;
            region.bounded = true;
            region.dim = 3;
            region.r = pad(std::sqrt(3.0) * grown);
            scale = 1;
            for (int a = 0; a < 3; ++a) {
                region.c[a] = item.centre[a];
                region.lo[a] = region.c[a] - grown;
                region.hi[a] = region.c[a] + grown;
                scale = std::fmax(scale, std::fabs(region.c[a]) + grown);
            }
            finish_region(region);
        };
        // Classifies items from the root's value at their centres: far
        // cells get their bound, band items are returned.
        const auto classify = [&](const std::vector<Item> &items) {
            std::vector<double> values(items.size());
            const size_t batches = (items.size() + kBatch - 1) / kBatch;
            parallel_deep(batches, threads, [&](size_t batch) {
                double x[kBatch], y[kBatch], z[kBatch];
                const size_t first = batch * kBatch;
                const int n = static_cast<int>(std::min<size_t>(kBatch, items.size() - first));
                for (int i = 0; i < n; ++i) {
                    x[i] = items[first + i].centre[0];
                    y[i] = items[first + i].centre[1];
                    z[i] = items[first + i].centre[2];
                }
                eval_batch(g, root, n, x, y, z, values.data() + first);
            });
            std::vector<Item> band;
            for (size_t i = 0; i < items.size(); ++i) {
                Region region;
                double scale, lb, ub, lipschitz;
                region_of(items[i], region, scale);
                compiler.bounds(root, region, values[i], lb, ub, lipschitz);
                const double slack = kSlack * (std::fabs(lb) + std::fabs(ub) + scale);
                if (lb > slack || ub < -slack) {
                    // A far brick: the distance to the surface from any point
                    // of its grown region is at least |bound| / L, and at
                    // least the distance to the part's box (which holds it).
                    double bound = lb > slack ? lb / lipschitz : ub / lipschitz;
                    if (lb > slack) bound = std::fmax(bound, region_distance(region, node));
                    ref.cell_value[items[i].cell] = bound;
                } else {
                    ref.cell_value[items[i].cell] = lipschitz;
                    band.push_back(items[i]);
                }
            }
            return band;
        };
        std::vector<Item> items(cells);
        for (uint64_t cell = 0; cell < cells; ++cell) {
            const uint64_t ijk[3] = {cell % ref.dims[0], (cell / ref.dims[0]) % ref.dims[1],
                                     cell / (uint64_t(ref.dims[0]) * ref.dims[1])};
            Item &item = items[cell];
            item.cell = static_cast<uint32_t>(cell);
            item.level = 0;
            item.size = options.brick_size;
            for (int a = 0; a < 3; ++a)
                item.centre[a] = ref.origin[a] + (static_cast<double>(ijk[a]) + 0.5) * options.brick_size;
        }
        const std::vector<Item> band = classify(items);
        // Each band brick compiles its tape; a tape longer than `refine`
        // tries its brick's octants and keeps them when their tapes are on
        // average at most kRefineGain of it (the octants may split again,
        // down to the minimum size). Cells are local to a brick until the
        // bricks are assembled in order.
        struct Local {
            Tape tape;
            std::shared_ptr<Job> pruned;  // the tape's pruned tree, while its octants are tried
            uint32_t count = 0;   // encoded instructions
            int32_t first = -1;   // octant cells (local index), when refined
            bool far = false;
            double value = 0;     // far: signed bound; band: Lipschitz bound
            int level = 0;
            uint64_t hash = 0;    // a tape's tape_hash (computed on the brick's thread)
        };
        constexpr double kRefineGain = 0.9;
        std::vector<std::vector<Local>> locals(band.size());
        std::vector<std::string> failures(band.size());
        const std::function<void(std::vector<Local> &, size_t, const Item &)> expand =
            [&](std::vector<Local> &out, size_t at, const Item &item) {
                if (out[at].count <= refine || item.level >= levels) return;
                Item children[8];
                double x[8], y[8], z[8], centre_values[8];
                for (uint32_t octant = 0; octant < 8; ++octant) {
                    Item &child = children[octant];
                    child.level = item.level + 1;
                    child.size = 0.5 * item.size;
                    for (int a = 0; a < 3; ++a)
                        child.centre[a] = item.centre[a] + (octant >> a & 1 ? 0.25 : -0.25) * item.size;
                    x[octant] = child.centre[0];
                    y[octant] = child.centre[1];
                    z[octant] = child.centre[2];
                }
                // The brick's tape is the field at the octants' centres (they lie
                // inside the brick), and much shorter than the root's graph.
                for (uint32_t octant = 0; octant < 8; ++octant) {
                    const double centre[3] = {x[octant], y[octant], z[octant]};
                    centre_values[octant] = run_tape(g, out[at].tape, centre);
                }
                Local made[8];
                double total = 0;
                for (uint32_t octant = 0; octant < 8; ++octant) {
                    Region region;
                    double scale, lb, ub, lipschitz;
                    region_of(children[octant], region, scale);
                    compiler.bounds(root, region, centre_values[octant], lb, ub, lipschitz);
                    const double slack = kSlack * (std::fabs(lb) + std::fabs(ub) + scale);
                    made[octant].level = children[octant].level;
                    if (lb > slack || ub < -slack) {
                        double bound = lb > slack ? lb / lipschitz : ub / lipschitz;
                        if (lb > slack) bound = std::fmax(bound, region_distance(region, node));
                        made[octant].far = true;
                        made[octant].value = bound;
                        total += 1;
                        continue;
                    }
                    made[octant].value = lipschitz;
                    try {
                        // Octants prune their brick's pruned tree further.
                        made[octant].pruned = std::make_shared<Job>(compiler.prune(root, region, scale, out[at].pruned.get()));
                        made[octant].tape = compiler.emit(*made[octant].pruned);
                    } catch (const Limit &) {
                        return;  // the octant is no better: keep this brick's tape
                    }
                    made[octant].count = instruction_count(made[octant].tape);
                    total += made[octant].count;
                }
                // Very long tapes always split: their gain may show a level deeper.
                if (total / 8 > kRefineGain * out[at].count && out[at].count <= 4 * refine) return;
                const size_t first = out.size();
                out[at].first = static_cast<int32_t>(first);
                out[at].tape = Tape();
                out[at].pruned.reset();
                for (Local &local : made) out.push_back(std::move(local));
                for (uint32_t octant = 0; octant < 8; ++octant) {
                    if (!out[first + octant].far) expand(out, first + octant, children[octant]);
                    out[first + octant].pruned.reset();
                }
            };
        parallel_deep(band.size(), threads, [&](size_t index) {
            Region region;
            double scale;
            region_of(band[index], region, scale);
            std::vector<Local> &out = locals[index];
            out.emplace_back();
            out[0].value = ref.cell_value[band[index].cell];
            try {
                out[0].pruned = std::make_shared<Job>(compiler.prune(root, region, scale));
                out[0].tape = compiler.emit(*out[0].pruned);
            } catch (const Limit &limit) {
                failures[index] = limit.what();
                return;
            }
            out[0].count = instruction_count(out[0].tape);
            expand(out, 0, band[index]);
            out[0].pruned.reset();
            for (Local &local : out)
                if (!local.far && local.first < 0) local.hash = tape_hash(local.tape, local.value);
        });
        for (size_t index = 0; index < band.size(); ++index)
            if (!failures[index].empty()) throw Limit(failures[index]);
        // Assemble: octant blocks get global cells in brick order, then
        // distinct tapes in cell order.
        std::vector<uint32_t> tape_words;                        // distinct tapes, relative offsets
        Pool pool;                                               // their parameters
        std::vector<uint32_t> cell_offsets(cells, 0);            // tape word offset (relative) per cell
        std::unordered_multimap<uint64_t, uint32_t> distinct;    // tape hash -> local tape index
        std::vector<uint32_t> distinct_offsets;
        std::vector<double> distinct_lipschitz;
        std::vector<uint32_t> encoded;
        uint32_t max_instructions = 0, values = 0, points = 0, depth = 0;
        uint64_t instructions = 0, band_cells = 0;
        std::vector<uint8_t> uses_mesh(g.meshes.size(), 0), uses_polygon(g.polygons.size(), 0);
        for (size_t index = 0; index < band.size(); ++index) {
            std::vector<Local> &out = locals[index];
            std::vector<uint32_t> global(out.size());
            global[0] = band[index].cell;
            for (size_t at = 0; at < out.size(); ++at) {
                Local &local = out[at];
                const uint32_t cell = global[at];
                if (local.first >= 0) {
                    const uint32_t first = static_cast<uint32_t>(ref.cell_tape.size());
                    ref.cell_child[cell] = static_cast<int32_t>(first);
                    ref.cell_tape.resize(first + 8, -1);
                    ref.cell_child.resize(first + 8, -1);
                    ref.cell_value.resize(first + 8, 0);
                    cell_offsets.resize(first + 8, 0);
                    if (uint64_t(ref.cell_tape.size()) > max_words) throw Limit("the program exceeds its size limit");
                    for (uint32_t octant = 0; octant < 8; ++octant) global[local.first + octant] = first + octant;
                    continue;
                }
                ref.cell_value[cell] = local.value;
                if (local.far) continue;
                const uint32_t count = instruction_count(local.tape);
                uint32_t found = UINT32_MAX;
                const auto range = distinct.equal_range(local.hash);
                for (auto it = range.first; it != range.second && found == UINT32_MAX; ++it)
                    if (distinct_lipschitz[it->second] == local.value &&
                        same_tape(program.tapes[tapes_before + it->second], local.tape))
                        found = it->second;
                if (found == UINT32_MAX) {
                    found = static_cast<uint32_t>(distinct_offsets.size());
                    distinct.emplace(local.hash, found);
                    distinct_offsets.push_back(static_cast<uint32_t>(tape_words.size()));
                    distinct_lipschitz.push_back(local.value);
                    // A header record (the cell's Lipschitz bound), then the tape.
                    encoded.assign({bits_of(float_up(local.value)), 0, 0, 0});
                    encode_tape(local.tape, encoded, pool);
                    tape_words.insert(tape_words.end(), encoded.begin(), encoded.end());
                    for (const Ins &ins : local.tape.code) {
                        if (ins.op == G_MESH) uses_mesh[ins.aux] = 1;
                        if (ins.op == G_POLYGON) uses_polygon[ins.aux] = 1;
                    }
                    program.tapes.push_back(std::move(local.tape));
                    if (tape_words.size() > max_words) throw Limit("the program exceeds its size limit");
                }
                const Tape &tape = program.tapes[tapes_before + found];
                cell_offsets[cell] = distinct_offsets[found];
                ref.cell_tape[cell] = static_cast<int32_t>(tapes_before + found);
                ++band_cells;
                instructions += count;
                max_instructions = std::max(max_instructions, count);
                values = std::max(values, tape.values);
                points = std::max(points, tape.points);
                depth = std::max<uint32_t>(depth, static_cast<uint32_t>(local.level));
            }
            out.clear();
            out.shrink_to_fit();
        }
        // Grid cells (the top grid padded to whole blocks of 8 words, then
        // octant blocks, 8-word aligned), the tapes (16-byte aligned
        // records), then their parameter pool.
        const uint64_t grid_at = (program.words.size() + 7) & ~uint64_t(7);
        const uint64_t pad = (8 - cells % 8) % 8;
        const auto cell_at = [&](uint64_t cell) { return grid_at + (cell < cells ? cell : cell + pad); };
        const uint64_t tapes_at = (grid_at + uint64_t(ref.cell_tape.size()) + pad + 3) & ~uint64_t(3);
        const uint64_t pool_at = tapes_at + tape_words.size();
        if (pool_at + pool.words.size() > max_words) throw Limit("the program exceeds its size limit");
        for (const uint32_t start : distinct_offsets) {
            for (size_t at = start + 4; (tape_words[at] & 0x7F) != G_END; at += 4) {
                if (has_parameters(tape_words[at] & 0x7F)) tape_words[at + 2] += static_cast<uint32_t>(pool_at);
                if (tape_words[at + 3] != 0) tape_words[at + 3] += static_cast<uint32_t>(pool_at);
            }
        }
        program.words.resize(tapes_at, 0);
        for (size_t cell = 0; cell < ref.cell_tape.size(); ++cell) {
            uint32_t word;
            const double v = ref.cell_value[cell];
            if (ref.cell_child[cell] >= 0) {
                word = kRefined | static_cast<uint32_t>(cell_at(uint64_t(ref.cell_child[cell])) >> 3);
            } else if (ref.cell_tape[cell] >= 0) {
                word = static_cast<uint32_t>((tapes_at + cell_offsets[cell]) >> 2);
            } else {
                // Far bounds round toward zero (they stay lower bounds of the distance).
                word = kFarCell | (v < 0 ? 0x40000000u : 0u) | bits_of(float_down(std::fabs(v))) >> 1;
            }
            program.words[cell_at(cell)] = word;
        }
        program.words.insert(program.words.end(), tape_words.begin(), tape_words.end());
        program.words.insert(program.words.end(), pool.words.begin(), pool.words.end());
        // Mesh and polygon data the tapes read (once per program).
        for (size_t m = 0; m < uses_mesh.size(); ++m)
            if (uses_mesh[m] && !mesh_done[m]) {
                uint32_t record[kMeshWords];
                encode_mesh(g.meshes[m], program.words, record);
                std::copy(record, record + kMeshWords, program.words.begin() + mesh_table_at + kMeshWords * m);
                mesh_done[m] = 1;
                added_meshes.push_back(m);
                if (program.words.size() > max_words) throw Limit("the program exceeds its size limit");
            }
        for (size_t p = 0; p < uses_polygon.size(); ++p)
            if (uses_polygon[p] && !polygon_done[p]) {
                uint32_t record[kPolygonWords];
                encode_polygon(g.polygons[p], program.words, record);
                std::copy(record, record + kPolygonWords, program.words.begin() + polygon_table_at + kPolygonWords * p);
                polygon_done[p] = 1;
                added_polygons.push_back(p);
                if (program.words.size() > max_words) throw Limit("the program exceeds its size limit");
            }
        ref.field = true;
        info.status = 0;
        info.band_bricks = band_cells;
        info.tapes = distinct_offsets.size();
        info.instructions = instructions;
        info.max_instructions = max_instructions;
        info.value_registers = values;
        info.point_registers = points;
        info.levels = depth;
        info.cells = ref.cell_tape.size();
        part_record[0] = 0;
        part_record[1] = static_cast<uint32_t>(grid_at);
        for (int a = 0; a < 3; ++a) part_record[2 + a] = ref.dims[a];
        part_record[5] = depth;
        for (int a = 0; a < 3; ++a) part_record[6 + a] = float_bits(ref.origin[a]);
        part_record[9] = float_bits(options.brick_size);
        part_record[10] = float_bits(1 / options.brick_size);
        part_record[11] = bits_of(float_down(options.margin));
        for (int a = 0; a < 3; ++a) {
            part_record[12 + a] = bits_of(float_down(ref.lo[a]));
            part_record[15 + a] = bits_of(float_up(ref.hi[a]));
        }
        part_record[18] = bits_of(float_up(ref.lipschitz));
        part_record[19] = static_cast<uint32_t>(distinct_offsets.size());
        part_record[20] = values;
        part_record[21] = points;
        part_record[22] = max_instructions;
        part_record[23] = static_cast<uint32_t>(ref.cell_tape.size());
    } catch (const Limit &limit) {
        // Mesh only: drop what this part added.
        for (size_t m : added_meshes) {
            mesh_done[m] = 0;
            std::fill_n(program.words.begin() + mesh_table_at + kMeshWords * m, kMeshWords, 0u);
        }
        for (size_t p : added_polygons) {
            polygon_done[p] = 0;
            std::fill_n(program.words.begin() + polygon_table_at + kPolygonWords * p, kPolygonWords, 0u);
        }
        program.words.resize(words_before);
        program.tapes.resize(tapes_before);
        ref = PartReference();
        ref.root = root;
        const uint32_t grid[3] = {info.grid[0], info.grid[1], info.grid[2]};
        const uint64_t bricks = info.bricks;
        info = rsdf_gpu_part_info();
        std::copy(grid, grid + 3, info.grid);
        info.bricks = bricks;
        info.status = 1;
        std::snprintf(info.message, sizeof info.message, "%s", limit.what());
        std::fill(part_record, part_record + kPartWords, 0);
        part_record[0] = 1;
    }
    std::copy(part_record, part_record + kPartWords, program.words.begin() + part_record_at);
    info.words = program.words.size() - words_before;
    info.seconds = std::chrono::duration<double>(std::chrono::steady_clock::now() - started).count();
}

// The reference value of a point (mode 0): the GPU's lookup in float64.
double reference_value(const rsdf_gpu_program &program, const PartReference &ref, const double *p) {
    double size = ref.brick, corner[3];
    int64_t cell[3];
    for (int a = 0; a < 3; ++a) {
        const double c = std::floor((p[a] - ref.origin[a]) / ref.brick);
        if (!(c >= 0 && c < ref.dims[a])) return box_gap(p, ref.lo, ref.hi);
        cell[a] = static_cast<int64_t>(c);
        corner[a] = ref.origin[a] + ref.brick * c;
    }
    size_t index = static_cast<size_t>(cell[0] + ref.dims[0] * (cell[1] + int64_t(ref.dims[1]) * cell[2]));
    while (ref.cell_child[index] >= 0) {
        size *= 0.5;
        uint32_t octant = 0;
        for (int a = 0; a < 3; ++a)
            if (p[a] >= corner[a] + size) {
                octant |= 1u << a;
                corner[a] += size;
            }
        index = static_cast<size_t>(ref.cell_child[index]) + octant;
    }
    if (ref.cell_tape[index] >= 0) return run_tape(*program.graph, program.tapes[ref.cell_tape[index]], p);
    // A far cell: at least its bound, and at least the margin plus the
    // distance to the cell's faces (the surface lies outside it grown by
    // the margin).
    const double bound = ref.cell_value[index];
    double inner = INFINITY;
    for (int a = 0; a < 3; ++a) inner = std::fmin(inner, std::fmin(p[a] - corner[a], corner[a] + size - p[a]));
    const double distance = std::fmax(std::fabs(bound), ref.margin + std::fmax(inner, 0.0));
    return bound < 0 ? -distance : distance;
}

}  // namespace
}  // namespace gpu
}  // namespace rsdf

using namespace rsdf;
using namespace rsdf::gpu;

extern "C" {

rsdf_gpu_program *rsdf_gpu_program_build(const rsdf_graph *graph, const uint32_t *nodes, uint32_t count,
                                         const rsdf_gpu_options *options) {
    rsdf_gpu_program *result = nullptr;
    guarded([&] {
        const rsdf_graph &g = checked_graph(graph);
        if (!options) throw Error("options must not be null");
        if (!nodes || count == 0) throw Error("a GPU program needs at least one part node");
        if (count > 65536) throw Error(format("%u part nodes exceed the limit of 65536", count));
        if (!(options->brick_size > 0) || !std::isfinite(options->brick_size))
            throw Error("brick_size must be positive and finite");
        if (!(options->margin >= 0) || !std::isfinite(options->margin))
            throw Error("margin must be finite and not negative");
        if (!(options->min_brick_size >= 0) || !std::isfinite(options->min_brick_size))
            throw Error("min_brick_size must be finite and not negative");
        for (uint32_t i = 0; i < count; ++i) {
            const Node &node = checked_node(g, nodes[i]);
            if (node.dim != 3 || node.field)
                throw Error(format("part node %u (%s) is not a 3D shape", nodes[i], op_name(node.op)));
        }
        // Offsets keep the cells' top bit and the records' map op bits free.
        const uint64_t max_words =
            std::min<uint64_t>(options->max_words ? options->max_words : kDefaultWords, kPoolOffsetMask);
        const uint32_t max_instructions =
            std::min<uint32_t>(options->max_tape_instructions ? options->max_tape_instructions : kDefaultInstructions,
                               1u << 20);
        auto program = std::make_unique<rsdf_gpu_program>();
        program->graph = &g;
        program->max_instructions = max_instructions;
        program->parts.resize(count);
        program->references.resize(count);
        const uint32_t meshes = static_cast<uint32_t>(g.meshes.size());
        const uint32_t polygons = static_cast<uint32_t>(g.polygons.size());
        const uint64_t table_words =
            kHeaderWords + uint64_t(kPartWords) * count + uint64_t(kMeshWords) * meshes + uint64_t(kPolygonWords) * polygons;
        if (table_words > max_words) throw Error("the GPU program's tables exceed its size limit");
        std::vector<uint32_t> &words = program->words;
        words.assign(table_words, 0);
        const uint32_t part_table = kHeaderWords;
        const uint32_t mesh_table = part_table + kPartWords * count;
        const uint32_t polygon_table = mesh_table + kMeshWords * meshes;
        words[0] = kMagic;
        words[1] = kVersion;
        words[2] = count;
        words[4] = part_table;
        words[5] = mesh_table;
        words[6] = meshes;
        words[7] = polygon_table;
        words[8] = polygons;
        words[9] = kValueRegisters;
        words[10] = kPointRegisters;
        words[11] = max_instructions;
        Compiler compiler(g, max_instructions);
        std::vector<uint8_t> mesh_done(meshes, 0), polygon_done(polygons, 0);
        for (uint32_t i = 0; i < count; ++i)
            build_part(*program, compiler, nodes[i], i, *options, max_words, mesh_done, polygon_done,
                       part_table + kPartWords * i, mesh_table, polygon_table);
        words[3] = static_cast<uint32_t>(words.size());
        result = program.release();
    });
    return result;
}

void rsdf_gpu_program_free(rsdf_gpu_program *program) { delete program; }

const uint32_t *rsdf_gpu_program_words(const rsdf_gpu_program *program, uint64_t *word_count) {
    if (word_count) *word_count = program ? program->words.size() : 0;
    return program ? program->words.data() : nullptr;
}

int rsdf_gpu_program_part(const rsdf_gpu_program *program, uint32_t part, rsdf_gpu_part_info *info) {
    return guarded([&] {
               if (!program) throw Error("program is null");
               if (!info) throw Error("info must not be null");
               if (part >= program->parts.size())
                   throw Error(format("part %u is not one of the program's %zu parts", part, program->parts.size()));
               *info = program->parts[part];
           })
               ? 0
               : -1;
}

int rsdf_gpu_program_evaluate(const rsdf_gpu_program *program, uint32_t part, const double *points, uint64_t count,
                              int32_t mode, double *values) {
    return guarded([&] {
               if (!program) throw Error("program is null");
               if (part >= program->parts.size())
                   throw Error(format("part %u is not one of the program's %zu parts", part, program->parts.size()));
               if (mode != 0 && mode != 1) throw Error("mode must be 0 (bricks) or 1 (unpruned)");
               if (count == 0) return;
               if (!points || !values) throw Error("input and output buffers must not be null");
               const PartReference &ref = program->references[part];
               if (!ref.field) throw Error(format("part %u has no GPU program (mesh only)", part));
               const rsdf_graph &g = *program->graph;
               const Tape *full = nullptr;
               if (mode == 1) {
                   rsdf_gpu_program &shared = const_cast<rsdf_gpu_program &>(*program);
                   std::lock_guard<std::mutex> lock(shared.full_mutex);
                   auto found = shared.full.find(ref.root);
                   if (found == shared.full.end()) {
                       std::unique_ptr<Tape> tape;
                       parallel_deep(1, 1, [&](size_t) {
                           try {
                               tape = std::make_unique<Tape>(Compiler(g, 1u << 22).compile(ref.root, Region(), 1));
                           } catch (const Limit &limit) {
                               throw Error(std::string("the unpruned tape exceeds a limit: ") + limit.what());
                           }
                       });
                       found = shared.full.emplace(ref.root, std::move(tape)).first;
                   }
                   full = found->second.get();
               }
               for (uint64_t i = 0; i < count; ++i) {
                   const double *p = points + 3 * i;
                   for (int a = 0; a < 3; ++a)
                       if (!(std::fabs(p[a]) <= 1e150))
                           throw Error(format("point %llu is not finite", static_cast<unsigned long long>(i)));
                   values[i] = full ? run_tape(g, *full, p) : reference_value(*program, ref, p);
               }
           })
               ? 0
               : -1;
}

}  // extern "C"
