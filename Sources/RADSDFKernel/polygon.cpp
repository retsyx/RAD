// Exact signed distance to a simple polygon.
//
// Each batch selects candidate edges by branch and bound over a bounding
// volume hierarchy of contiguous edge ranges: the distance from every batch
// point to one seed edge bounds every point's nearest-edge distance, so only
// hierarchy nodes whose box lies within that bound of the batch box can hold a
// nearest edge. When that still leaves many edges, each point searches the
// hierarchy on its own. The sign comes from the nearest feature: the side of
// an edge interior, or for a vertex the side of its pseudonormal (the sum of
// its two edges' unit outward normals). Points nearest to a convex vertex lie
// on its outer side of that normal and points nearest to a reflex vertex on
// its inner side, and near-straight vertices take the side of their edges, so
// distance ties between a vertex and an adjacent edge interior (which
// rounding decides arbitrarily) cannot flip the sign.
#include "kernel.h"

#include <algorithm>

namespace rsdf {
namespace {

constexpr uint32_t kLeafEdges = 8;
constexpr size_t kDirectEdges = 32;      // smaller polygons test every edge
constexpr size_t kPerPointEdges = 256;   // more candidates than this: search per point

uint32_t build_tree(Polygon &polygon, uint32_t begin, uint32_t end) {
    const uint32_t index = static_cast<uint32_t>(polygon.tree.size());
    polygon.tree.push_back(PolygonTree{});
    const size_t count = polygon.x.size();
    double lo[2] = {INFINITY, INFINITY}, hi[2] = {-INFINITY, -INFINITY};
    for (uint32_t v = begin; v <= end; ++v) {  // edge vertices, including the end of the last edge
        const size_t i = v % count;
        lo[0] = std::min(lo[0], polygon.x[i]);
        hi[0] = std::max(hi[0], polygon.x[i]);
        lo[1] = std::min(lo[1], polygon.y[i]);
        hi[1] = std::max(hi[1], polygon.y[i]);
    }
    uint32_t right = 0;
    if (end - begin > kLeafEdges) {
        const uint32_t middle = begin + (end - begin) / 2;
        build_tree(polygon, begin, middle);
        right = build_tree(polygon, middle, end);
    }
    polygon.tree[index] = PolygonTree{{lo[0], lo[1]}, {hi[0], hi[1]}, begin, end, right};
    return index;
}

inline double point_box_distance2(double px, double py, const PolygonTree &node) {
    const double dx = std::fmax(std::fmax(node.lo[0] - px, px - node.hi[0]), 0.0);
    const double dy = std::fmax(std::fmax(node.lo[1] - py, py - node.hi[1]), 0.0);
    return dx * dx + dy * dy;
}

// Squared distance from (px, py) to edge `edge`; t receives the clamped
// parameter of the nearest point.
inline double edge_distance2(const Polygon &polygon, uint32_t edge, double px, double py, double &t) {
    const double wx = px - polygon.x[edge], wy = py - polygon.y[edge];
    const double ex = polygon.ex[edge], ey = polygon.ey[edge];
    t = std::clamp((wx * ex + wy * ey) * polygon.inv_len2[edge], 0.0, 1.0);
    const double dx = wx - t * ex, dy = wy - t * ey;
    return dx * dx + dy * dy;
}

struct Nearest {
    double d2 = INFINITY, t = 0.5;
    uint32_t edge = 0;
};

// The edge nearest to (px, py), by depth-first branch and bound.
Nearest nearest_edge(const Polygon &polygon, double px, double py) {
    uint32_t stack[128];
    int top = 0;
    stack[top++] = 0;
    Nearest best;
    while (top > 0) {
        const uint32_t index = stack[--top];
        const PolygonTree &node = polygon.tree[index];
        if (!(point_box_distance2(px, py, node) < best.d2)) continue;
        if (node.right == 0) {
            for (uint32_t edge = node.begin; edge < node.end; ++edge) {
                double t;
                const double d2 = edge_distance2(polygon, edge, px, py, t);
                if (d2 < best.d2) best = Nearest{d2, t, edge};
            }
            continue;
        }
        const uint32_t left = index + 1, right = node.right;
        const bool left_first =
            point_box_distance2(px, py, polygon.tree[left]) <= point_box_distance2(px, py, polygon.tree[right]);
        stack[top++] = left_first ? right : left;  // visited second
        stack[top++] = left_first ? left : right;
    }
    return best;
}

// Removes zero-width spikes: a vertex whose two edges run back along each
// other has no area on either side, and the opposite sides of its overlapping
// edges would make the sign beside it arbitrary. Removing a tip (and a point
// repeated as a result) can expose another; the work list repeats until none
// is left. Returns the number of remaining points.
size_t remove_spikes(std::vector<double> &x, std::vector<double> &y) {
    const size_t n = x.size();
    std::vector<size_t> prev(n), next(n), work(n);
    std::vector<uint8_t> removed(n, 0);
    for (size_t i = 0; i < n; ++i) {
        prev[i] = (i + n - 1) % n;
        next[i] = (i + 1) % n;
        work[i] = n - 1 - i;
    }
    size_t alive = n;
    while (!work.empty() && alive >= 3) {
        const size_t i = work.back();
        work.pop_back();
        if (removed[i]) continue;
        const size_t a = prev[i], b = next[i];
        const double e1x = x[i] - x[a], e1y = y[i] - y[a], e2x = x[b] - x[i], e2y = y[b] - y[i];
        const bool repeated = e2x == 0 && e2y == 0;
        const bool spike = e1x * e2y - e1y * e2x == 0 && e1x * e2x + e1y * e2y < 0;
        if (!repeated && !spike) continue;
        removed[i] = 1;
        --alive;
        next[a] = b;
        prev[b] = a;
        work.push_back(b);
        work.push_back(a);
    }
    if (alive == n) return n;
    size_t kept = 0;
    for (size_t i = 0; i < n; ++i) {
        if (removed[i]) continue;
        x[kept] = x[i];
        y[kept] = y[i];
        ++kept;
    }
    x.resize(kept);
    y.resize(kept);
    return kept;
}

}  // namespace

void build_polygon(Polygon &polygon, const double *coords, size_t point_count) {
    std::vector<double> &x = polygon.x, &y = polygon.y;
    for (size_t i = 0; i < point_count; ++i) {
        const double px = coords[2 * i], py = coords[2 * i + 1];
        if (!x.empty() && px == x.back() && py == y.back()) continue;  // repeated point
        x.push_back(px);
        y.push_back(py);
    }
    while (x.size() > 1 && x.front() == x.back() && y.front() == y.back()) {  // explicit closing point
        x.pop_back();
        y.pop_back();
    }
    if (x.size() < 3) throw Error("polygon needs at least three distinct points");
    if (remove_spikes(x, y) < 3) throw Error("polygon has zero area");
    const size_t count = x.size();
    if (count > UINT32_MAX / 2) throw Error("polygon has too many points");
    double area2 = 0;
    for (size_t i = 0; i < count; ++i) {
        const size_t j = (i + 1) % count;
        area2 += (x[i] - x[0]) * (y[j] - y[0]) - (x[j] - x[0]) * (y[i] - y[0]);
    }
    if (!std::isfinite(area2)) throw Error("polygon coordinates are too large");
    if (area2 == 0) throw Error("polygon has zero area");
    if (area2 < 0) {
        std::reverse(x.begin(), x.end());
        std::reverse(y.begin(), y.end());
    }
    polygon.ex.resize(count);
    polygon.ey.resize(count);
    polygon.inv_len2.resize(count);
    for (size_t i = 0; i < count; ++i) {
        const size_t j = (i + 1) % count;
        polygon.ex[i] = x[j] - x[i];
        polygon.ey[i] = y[j] - y[i];
        const double inv = 1 / (polygon.ex[i] * polygon.ex[i] + polygon.ey[i] * polygon.ey[i]);
        if (!std::isfinite(inv)) throw Error(format("polygon edge %zu is too short", i));
        polygon.inv_len2[i] = inv;
    }
    // Counter-clockwise: the outward normal of edge e is (ey, -ex) / |e|.
    polygon.nx.resize(count);
    polygon.ny.resize(count);
    for (size_t i = 0; i < count; ++i) {
        const size_t prev = (i + count - 1) % count;
        const double a = std::sqrt(polygon.inv_len2[prev]), b = std::sqrt(polygon.inv_len2[i]);
        polygon.nx[i] = polygon.ey[prev] * a + polygon.ey[i] * b;
        polygon.ny[i] = -polygon.ex[prev] * a - polygon.ex[i] * b;
    }
    build_tree(polygon, 0, static_cast<uint32_t>(count));
}

void eval_polygon(const Polygon &polygon, int n, const double *x, const double *y, double *out) {
    // Candidate edges as [begin, end) ranges; per-thread to avoid allocation.
    thread_local std::vector<uint32_t> ranges;
    ranges.clear();
    const uint32_t count = static_cast<uint32_t>(polygon.x.size());
    size_t candidates = count;
    if (count > kDirectEdges && n > 0) {
        double lo[2] = {x[0], y[0]}, hi[2] = {x[0], y[0]};
        for (int i = 1; i < n; ++i) {
            lo[0] = std::fmin(lo[0], x[i]);
            hi[0] = std::fmax(hi[0], x[i]);
            lo[1] = std::fmin(lo[1], y[i]);
            hi[1] = std::fmax(hi[1], y[i]);
        }
        const uint32_t seed = nearest_edge(polygon, 0.5 * (lo[0] + hi[0]), 0.5 * (lo[1] + hi[1])).edge;
        double bound2 = 0, t;
        for (int i = 0; i < n; ++i) bound2 = std::fmax(bound2, edge_distance2(polygon, seed, x[i], y[i], t));
        bound2 *= 1 + 1e-9;  // rounding slack: never prune an edge tied with the bound
        uint32_t stack[128];
        int top = 0;
        stack[top++] = 0;
        candidates = 0;
        while (top > 0) {
            const uint32_t index = stack[--top];
            const PolygonTree &node = polygon.tree[index];
            const double dx = std::fmax(std::fmax(node.lo[0] - hi[0], lo[0] - node.hi[0]), 0.0);
            const double dy = std::fmax(std::fmax(node.lo[1] - hi[1], lo[1] - node.hi[1]), 0.0);
            if (!(dx * dx + dy * dy <= bound2)) continue;
            if (node.right != 0) {
                stack[top++] = node.right;
                stack[top++] = index + 1;  // left first: ranges stay in edge order
                continue;
            }
            candidates += node.end - node.begin;
            if (!ranges.empty() && ranges.back() == node.begin) {
                ranges.back() = node.end;
            } else {
                ranges.push_back(node.begin);
                ranges.push_back(node.end);
            }
        }
    }
    if (ranges.empty()) {  // small polygon (or a non-finite batch box)
        ranges.push_back(0);
        ranges.push_back(count);
    }
    // A scattered batch keeps many candidates: searching per point is cheaper.
    const bool per_point = count > kDirectEdges && candidates > kPerPointEdges && candidates > 8 * size_t(n);

    const double *vx = polygon.x.data(), *vy = polygon.y.data();
    const double *ex = polygon.ex.data(), *ey = polygon.ey.data(), *inv = polygon.inv_len2.data();
    const PolygonTree &box = polygon.tree.front();  // the root covers every edge
    for (int i = 0; i < n; ++i) {
        const double px = x[i], py = y[i];
        Nearest best;
        if (per_point) {
            best = nearest_edge(polygon, px, py);
        } else {
            for (size_t r = 0; r < ranges.size(); r += 2) {
                for (uint32_t e = ranges[r]; e < ranges[r + 1]; ++e) {
                    const double wx = px - vx[e], wy = py - vy[e];
                    const double t = std::clamp((wx * ex[e] + wy * ey[e]) * inv[e], 0.0, 1.0);
                    const double dx = wx - t * ex[e], dy = wy - t * ey[e];
                    const double d2 = dx * dx + dy * dy;
                    if (d2 < best.d2) best = Nearest{d2, t, e};
                }
            }
        }
        if (!(best.d2 > 0)) {
            out[i] = best.d2 == 0 ? 0.0 : best.d2;  // on the boundary (or NaN input)
            continue;
        }
        bool inside = false;
        const uint32_t edge = best.edge;
        if (px < box.lo[0] || px > box.hi[0] || py < box.lo[1] || py > box.hi[1]) {
            // outside the vertices' box: outside, however rounding ties the
            // features of a polygon far smaller than |p|
        } else if (best.t > 0 && best.t < 1) {
            inside = ex[edge] * (py - vy[edge]) - ey[edge] * (px - vx[edge]) > 0;
        } else {
            const uint32_t vertex = best.t <= 0 ? edge : (edge + 1 == count ? 0 : edge + 1);
            inside = (px - vx[vertex]) * polygon.nx[vertex] + (py - vy[vertex]) * polygon.ny[vertex] < 0;
        }
        out[i] = inside ? -std::sqrt(best.d2) : std::sqrt(best.d2);
    }
}

}  // namespace rsdf
