// Exact signed distance to a simple polygon.
//
// Decoding normalises the points (repeated points, a closing copy and the
// tips of zero-width spikes are dropped) and then requires a simple polygon:
// no two edges that are not neighbours share a point, and neighbours share
// only their common vertex. A sweep over the vertices in lexicographic order
// (Shamos and Hoey) keeps the edges crossing the sweep line in order and tests
// every pair that becomes adjacent there, so it finds a defect, if there is
// one, in O(n log n). Every decision uses exact orientation signs (a
// floating-point filter, then exact integer arithmetic on the doubles'
// significands), so no tolerance rejects a thin valid slot or accepts a
// crossing.
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
#include <climits>
#include <numeric>
#include <set>

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

// ---------------------------------------------------------------- exact signs

// The sign of sum_k a[k] * b[k] computed exactly for any finite doubles: each
// double is m * 2^e with an integer |m| < 2^53, so each product is a 106-bit
// integer times a power of two, and the terms add without rounding in a
// two's-complement integer wide enough for the whole exponent range.
int exact_dot_sign(const double *a, const double *b, int count) {
    constexpr int kLimbs = 70;  // 4,480 bits: product exponents span 4,194 bits, products 106
    unsigned __int128 magnitude[6];
    int exponent[6];
    bool negative[6];
    int lowest = INT_MAX;
    for (int k = 0; k < count; ++k) {
        magnitude[k] = 0;
        if (a[k] == 0 || b[k] == 0) continue;
        int ea, eb;
        const auto ma = static_cast<int64_t>(std::ldexp(std::frexp(a[k], &ea), 53));
        const auto mb = static_cast<int64_t>(std::ldexp(std::frexp(b[k], &eb), 53));
        const __int128 product = static_cast<__int128>(ma) * mb;
        negative[k] = product < 0;
        magnitude[k] = static_cast<unsigned __int128>(product < 0 ? -product : product);
        exponent[k] = ea + eb - 106;
        lowest = std::min(lowest, exponent[k]);
    }
    if (lowest == INT_MAX) return 0;
    uint64_t sum[kLimbs] = {};
    for (int k = 0; k < count; ++k) {
        if (magnitude[k] == 0) continue;
        const int shift = exponent[k] - lowest, word = shift / 64, bit = shift % 64;
        const auto lo = static_cast<uint64_t>(magnitude[k]), hi = static_cast<uint64_t>(magnitude[k] >> 64);
        const uint64_t part[3] = {lo << bit, bit ? (hi << bit) | (lo >> (64 - bit)) : hi, bit ? hi >> (64 - bit) : 0};
        uint64_t carry = 0;
        for (int w = word; w < kLimbs; ++w) {
            const uint64_t value = w - word < 3 ? part[w - word] : 0;
            if (w - word >= 3 && carry == 0) break;
            if (!negative[k]) {
                const uint64_t total = sum[w] + value;
                const uint64_t next = (total < value) + (total + carry < total);
                sum[w] = total + carry;
                carry = next;
            } else {
                const uint64_t total = sum[w] - value;
                const uint64_t next = (sum[w] < value) + (total < carry);
                sum[w] = total - carry;
                carry = next;
            }
        }
    }
    if (sum[kLimbs - 1] >> 63) return -1;
    for (int w = 0; w < kLimbs; ++w)
        if (sum[w] != 0) return 1;
    return 0;
}

inline int sign_of(double value) { return (value > 0) - (value < 0); }

// The exact sign of (b - a) x (c - a): positive when c lies to the left of the
// line from a to b, zero when the three points are collinear.
int orient(double ax, double ay, double bx, double by, double cx, double cy) {
#pragma clang fp contract(off)
    const double abx = bx - ax, aby = by - ay, acx = cx - ax, acy = cy - ay;
    const double l = abx * acy, r = aby * acx, det = l - r, sum = std::fabs(l) + std::fabs(r);
    // Shewchuk's bound for this expression, valid while nothing overflows and
    // the products are far above the subnormal range.
    constexpr double kBound = (3.0 + 16.0 * 0x1p-53) * 0x1p-53;
    if (std::fabs(det) > kBound * sum && sum >= 0x1p-960 && sum <= 0x1p+1000) return det > 0 ? 1 : -1;
    // A rounded difference keeps the sign of the exact one (subnormal
    // differences are exact), so the products' signs are exact.
    const int left = sign_of(abx) * sign_of(acy), right = sign_of(aby) * sign_of(acx);
    if (left != right) return left > right ? 1 : -1;
    if (left == 0) return 0;
    // bx*cy - bx*ay - ax*cy - by*cx + ax*by + ay*cx
    const double first[6] = {bx, -bx, -ax, -by, ax, ay}, second[6] = {cy, ay, cy, cx, by, cx};
    return exact_dot_sign(first, second, 6);
}

// Lexicographic order of points, (x, then y): the sweep order.
inline int compare_points(double ax, double ay, double bx, double by) {
    if (ax != bx) return ax < bx ? -1 : 1;
    if (ay != by) return ay < by ? -1 : 1;
    return 0;
}

// ------------------------------------------------------------- simplicity

// The first defect that makes a normalised polygon not simple.
struct Defect {
    enum Kind { kNone, kCoincide, kTouch, kOverlap, kCross } kind = kNone;
    uint32_t a = 0, b = 0;   // kCoincide: two vertices; kTouch: vertex a on edge b; else edges a and b
    double x0 = 0, y0 = 0;   // where (kOverlap: one end of the shared part)
    double x1 = 0, y1 = 0;   // kOverlap: the other end
};

// A Shamos-Hoey sweep over the vertices in lexicographic order. The status
// holds the edges crossing the sweep line, ordered from below; each pair of
// edges that becomes adjacent there is tested for any common point. If the
// polygon is not simple, let q be the first point (in sweep order) where two
// edges meet improperly: just before q the status order is exact, the edges
// through q form a contiguous run in it and two of them are adjacent, so they
// were tested when they became adjacent, or q is a vertex whose new edges are
// placed beside that run, which tests q against an edge through it.
class SimplicityCheck {
public:
    SimplicityCheck(const std::vector<double> &x, const std::vector<double> &y)
        : x_(x), y_(y), n_(static_cast<uint32_t>(x.size())), status_(Below{this}) {}

    // The first defect, or kNone; `orientation` receives the polygon's
    // orientation (the sign of its area) when it is simple.
    Defect run(int &orientation) {
        std::vector<uint32_t> order(n_);
        std::iota(order.begin(), order.end(), 0u);
        std::sort(order.begin(), order.end(), [&](uint32_t a, uint32_t b) {
            const int c = compare_points(x_[a], y_[a], x_[b], y_[b]);
            return c != 0 ? c < 0 : a < b;
        });
        rank_.resize(n_);
        for (uint32_t k = 0; k < n_; ++k) {
            rank_[order[k]] = k;
            if (k > 0 && x_[order[k]] == x_[order[k - 1]] && y_[order[k]] == y_[order[k - 1]]) {
                Defect defect{Defect::kCoincide, order[k - 1], order[k]};
                defect.x0 = x_[order[k]];
                defect.y0 = y_[order[k]];
                return defect;
            }
        }
        segment_.resize(n_);
        for (uint32_t e = 0; e < n_; ++e) {
            const uint32_t l = left(e), r = right(e);
            segment_[e] = Segment{x_[l], y_[l], x_[r], y_[r]};
        }
        where_.resize(n_);
        for (uint32_t k = 0; k < n_; ++k) {
            const Defect defect = visit(order[k]);
            if (defect.kind != Defect::kNone) return defect;
        }
        // The lowest vertex is convex: its turn is the polygon's orientation.
        const uint32_t v = order[0];
        orientation = orient(x_[prev(v)], y_[prev(v)], x_[v], y_[v], x_[next(v)], y_[next(v)]);
        return Defect{};
    }

private:
    struct Segment {
        double lx, ly, rx, ry;  // left and right end in sweep order
    };
    // An edge in the status; a vertex where the outline passes through
    // replaces its edge in place, which keeps the order.
    struct Slot {
        mutable uint32_t edge;
    };
    struct Below {
        SimplicityCheck *check;
        bool operator()(const Slot &a, const Slot &b) const { return check->below(a.edge, b.edge); }
    };
    using Status = std::set<Slot, Below>;

    uint32_t next(uint32_t v) const { return v + 1 == n_ ? 0 : v + 1; }
    uint32_t prev(uint32_t v) const { return v == 0 ? n_ - 1 : v - 1; }
    // Edge e joins vertex e to vertex e + 1; left and right in sweep order.
    uint32_t left(uint32_t e) const { return rank_[e] < rank_[next(e)] ? e : next(e); }
    uint32_t right(uint32_t e) const { return rank_[e] < rank_[next(e)] ? next(e) : e; }
    int turn(uint32_t a, uint32_t b, uint32_t c) const { return orient(x_[a], y_[a], x_[b], y_[b], x_[c], y_[c]); }
    int compare(uint32_t a, uint32_t b) const { return compare_points(x_[a], y_[a], x_[b], y_[b]); }

    // Order of the status: an edge starting at the event vertex against the
    // edges already there, by the vertex's side of them (and against the
    // other edge starting there by direction). A vertex on an edge is
    // recorded as a defect.
    bool below(uint32_t a, uint32_t b) {
        if (a == b) return false;
        const Segment &p = segment_[a], &q = segment_[b];
        const double ex = x_[event_], ey = y_[event_];
        const bool new_a = p.lx == ex && p.ly == ey, new_b = q.lx == ex && q.ly == ey;
        if (new_a && new_b) return orient(ex, ey, p.rx, p.ry, q.rx, q.ry) > 0;
        if (new_a || new_b) {
            const Segment &old = new_a ? q : p;
            const int side = orient(old.lx, old.ly, old.rx, old.ry, ex, ey);
            if (side == 0 && touch_.kind == Defect::kNone) touch_ = touch(event_, new_a ? b : a);
            return new_a ? side < 0 : side > 0;
        }
        return a < b;  // never asked: insertions compare the new edge only
    }

    Defect touch(uint32_t vertex, uint32_t edge) const {
        Defect defect{Defect::kTouch, vertex, edge};
        defect.x0 = x_[vertex];
        defect.y0 = y_[vertex];
        return defect;
    }

    // Whether (px, py), collinear with segment s, lies on it.
    static bool within(const Segment &s, double px, double py) {
        return compare_points(s.lx, s.ly, px, py) <= 0 && compare_points(px, py, s.rx, s.ry) <= 0;
    }

    // Any common point of edges s and t other than a shared vertex.
    Defect classify(uint32_t s, uint32_t t) const {
        if (next(s) == t || next(t) == s) {
            // Neighbours meet at their common vertex c; they overlap when
            // they leave it in the same direction.
            const bool s_first = next(s) == t;
            const uint32_t c = s_first ? t : s, u = s_first ? s : next(s), w = s_first ? next(t) : t;
            if (turn(u, c, w) != 0 || compare(u, c) != compare(w, c)) return Defect{};
            const bool after = compare(u, c) > 0;
            const uint32_t near = (compare(u, w) < 0) == after ? u : w;
            return Defect{Defect::kOverlap, s, t, x_[c], y_[c], x_[near], y_[near]};
        }
        const Segment &a = segment_[s], &b = segment_[t];
        if (a.rx < b.lx || b.rx < a.lx || std::max(a.ly, a.ry) < std::min(b.ly, b.ry) ||
            std::max(b.ly, b.ry) < std::min(a.ly, a.ry))
            return Defect{};  // their boxes are apart
        const int d1 = orient(a.lx, a.ly, a.rx, a.ry, b.lx, b.ly), d2 = orient(a.lx, a.ly, a.rx, a.ry, b.rx, b.ry);
        if (d1 * d2 > 0) return Defect{};
        const int d3 = orient(b.lx, b.ly, b.rx, b.ry, a.lx, a.ly), d4 = orient(b.lx, b.ly, b.rx, b.ry, a.rx, a.ry);
        if (d3 * d4 > 0) return Defect{};
        if (d1 == 0 && d2 == 0) {
            const uint32_t lo = compare(left(s), left(t)) > 0 ? left(s) : left(t);
            const uint32_t hi = compare(right(s), right(t)) < 0 ? right(s) : right(t);
            if (compare(lo, hi) >= 0) return Defect{};  // apart along their common line
            return Defect{Defect::kOverlap, s, t, x_[lo], y_[lo], x_[hi], y_[hi]};
        }
        if (d1 == 0 && within(a, b.lx, b.ly)) return touch(left(t), s);
        if (d2 == 0 && within(a, b.rx, b.ry)) return touch(right(t), s);
        if (d3 == 0 && within(b, a.lx, a.ly)) return touch(left(s), t);
        if (d4 == 0 && within(b, a.rx, a.ry)) return touch(right(s), t);
        if (d1 * d2 < 0 && d3 * d4 < 0) {
            // Proper crossing; its location is only reported, so rounding is fine.
            const double ex = a.rx - a.lx, ey = a.ry - a.ly, fx = b.rx - b.lx, fy = b.ry - b.ly;
            double f = ((b.lx - a.lx) * fy - (b.ly - a.ly) * fx) / (ex * fy - ey * fx);
            f = std::isfinite(f) ? std::clamp(f, 0.0, 1.0) : 0.5;
            return Defect{Defect::kCross, s, t, a.lx + f * ex, a.ly + f * ey};
        }
        return Defect{};
    }

    // Processes vertex v: replaces, removes or inserts its two edges and
    // tests the pairs that become adjacent.
    Defect visit(uint32_t v) {
        const uint32_t into = prev(v), out = v;  // edges (v - 1, v) and (v, v + 1)
        const bool into_ends = rank_[prev(v)] < rank_[v], out_ends = rank_[next(v)] < rank_[v];
        if (into_ends != out_ends) {
            // The outline passes through v: the edge leaving v takes the place
            // of the edge arriving there, between the same neighbours (a
            // neighbour through v meets the new edge there).
            const Status::iterator it = where_[into_ends ? into : out];
            const uint32_t starting = into_ends ? out : into;
            it->edge = starting;
            where_[starting] = it;
            return test_neighbours(it, it);
        }
        if (into_ends) {
            // Both end at v, `upper` arriving above `lower`; nothing may lie
            // between them.
            const Defect folded = classify(into, out);
            if (folded.kind != Defect::kNone) return folded;
            const bool into_lower = turn(prev(v), v, next(v)) > 0;
            const uint32_t lower = into_lower ? into : out, upper = into_lower ? out : into;
            const Status::iterator low = where_[lower], high = where_[upper];
            if (std::next(low) != high) {
                // An edge between them passes through v (or meets one of them).
                if (std::next(low) == status_.end()) throw Error("polygon sweep lost its order");  // unreachable
                const uint32_t between = std::next(low)->edge;
                Defect defect = classify(between, lower);
                if (defect.kind == Defect::kNone) defect = classify(between, upper);
                return defect.kind != Defect::kNone ? defect : touch(v, between);
            }
            const Status::iterator above = status_.erase(status_.erase(low));
            if (above == status_.end() || above == status_.begin()) return Defect{};
            return classify(std::prev(above)->edge, above->edge);
        }
        // Both start at v.
        const int side = turn(v, prev(v), next(v));  // positive: the edge to v + 1 lies above
        if (side == 0) return classify(into, out);   // they leave v in the same direction
        const uint32_t lower = side > 0 ? into : out, upper = side > 0 ? out : into;
        event_ = v;
        const Status::iterator first = status_.insert(Slot{lower}).first;
        if (touch_.kind != Defect::kNone) return touch_;
        const Status::iterator last = status_.insert(std::next(first), Slot{upper});
        if (touch_.kind != Defect::kNone) return touch_;
        where_[lower] = first;
        where_[upper] = last;
        return test_neighbours(first, last);
    }

    // Tests the run [first, last] of new edges against its neighbours.
    Defect test_neighbours(Status::iterator first, Status::iterator last) const {
        if (first != status_.begin()) {
            const Defect defect = classify(std::prev(first)->edge, first->edge);
            if (defect.kind != Defect::kNone) return defect;
        }
        const Status::iterator above = std::next(last);
        return above == status_.end() ? Defect{} : classify(last->edge, above->edge);
    }

    const std::vector<double> &x_, &y_;
    const uint32_t n_;
    std::vector<uint32_t> rank_;
    std::vector<Segment> segment_;  // each edge's ends in sweep order
    std::vector<Status::iterator> where_;
    Status status_;
    uint32_t event_ = UINT32_MAX;
    Defect touch_;
};

// Removes zero-width spikes: a vertex whose two edges run back along each
// other has no area on either side, and the opposite sides of its overlapping
// edges would make the sign beside it arbitrary. Removing a tip (and a point
// repeated as a result) can expose another; the work list repeats until none
// is left. Both tests are exact. Returns the number of remaining points.
size_t remove_spikes(std::vector<double> &x, std::vector<double> &y, std::vector<uint32_t> &source) {
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
        const bool repeated = x[b] == x[i] && y[b] == y[i];
        // Collinear, with a and b on the same side of i (strictly).
        const int side = compare_points(x[a], y[a], x[i], y[i]);
        const bool spike = side != 0 && side == compare_points(x[b], y[b], x[i], y[i]) &&
                           orient(x[a], y[a], x[i], y[i], x[b], y[b]) == 0;
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
        source[kept] = source[i];
        ++kept;
    }
    x.resize(kept);
    y.resize(kept);
    source.resize(kept);
    return kept;
}

// "point 3 (x, y)": an input point, numbered from 0 as given.
std::string describe_point(const std::vector<double> &x, const std::vector<double> &y,
                           const std::vector<uint32_t> &source, uint32_t v) {
    return format("point %u (%.9g, %.9g)", source[v], x[v], y[v]);
}

std::string describe_edge(const std::vector<double> &x, const std::vector<double> &y,
                          const std::vector<uint32_t> &source, uint32_t e) {
    const uint32_t f = e + 1 == x.size() ? 0 : e + 1;
    return "the edge from " + describe_point(x, y, source, e) + " to " + describe_point(x, y, source, f);
}

}  // namespace

void build_polygon(Polygon &polygon, const double *coords, size_t point_count) {
    std::vector<double> &x = polygon.x, &y = polygon.y;
    std::vector<uint32_t> source;  // the input point of each vertex, for messages
    for (size_t i = 0; i < point_count; ++i) {
        const double px = coords[2 * i], py = coords[2 * i + 1];
        if (!x.empty() && px == x.back() && py == y.back()) continue;  // repeated point
        x.push_back(px);
        y.push_back(py);
        source.push_back(static_cast<uint32_t>(i));
    }
    while (x.size() > 1 && x.front() == x.back() && y.front() == y.back()) {  // explicit closing point
        x.pop_back();
        y.pop_back();
        source.pop_back();
    }
    if (x.size() < 3) throw Error("polygon needs at least three distinct points");
    if (remove_spikes(x, y, source) < 3) throw Error("polygon has zero area");
    const size_t count = x.size();
    if (count > UINT32_MAX / 2) throw Error("polygon has too many points");
    int orientation = 0;
    Defect defect = SimplicityCheck(x, y).run(orientation);
    if (defect.kind != Defect::kNone) {
        if (defect.kind != Defect::kTouch && defect.a > defect.b) std::swap(defect.a, defect.b);  // in input order
        std::string detail;
        switch (defect.kind) {
        case Defect::kCoincide:
            detail = format("point %u and point %u are both at (%.9g, %.9g)", source[defect.a], source[defect.b],
                            defect.x0, defect.y0);
            break;
        case Defect::kTouch:
            detail = describe_point(x, y, source, defect.a) + " lies on " + describe_edge(x, y, source, defect.b);
            break;
        case Defect::kOverlap:
            detail = describe_edge(x, y, source, defect.a) + " overlaps " + describe_edge(x, y, source, defect.b) +
                     format(" between (%.9g, %.9g) and (%.9g, %.9g)", defect.x0, defect.y0, defect.x1, defect.y1);
            break;
        default:
            detail = describe_edge(x, y, source, defect.a) + " crosses " + describe_edge(x, y, source, defect.b) +
                     format(" at (%.9g, %.9g)", defect.x0, defect.y0);
            break;
        }
        throw Error("polygon is not simple (its outline must not cross, touch or overlap itself): " + detail);
    }
    double area2 = 0;
    for (size_t i = 0; i < count; ++i) {
        const size_t j = (i + 1) % count;
        area2 += (x[i] - x[0]) * (y[j] - y[0]) - (x[j] - x[0]) * (y[i] - y[0]);
    }
    if (!std::isfinite(area2)) throw Error("polygon coordinates are too large");
    if (area2 == 0) throw Error("polygon has zero area");
    if (orientation < 0) {
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
