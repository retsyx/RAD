// Feature-preserving simplification of a finished mesh against its signed
// distance field (SDF-DESIGN.md, "Mesh simplification").
//
// Edge collapses ordered by quadric error (Garland-Heckbert) run in rounds.
// Each round proposes one collapse per eligible edge, picks collapses in
// order of error whose neighbourhoods cannot interact, validates them
// against the mesh at the start of the round (link condition, valences,
// creases, triangle shape and orientation, folded edges, distance of removed
// vertices and crease points, and |f| at samples of every changed triangle)
// and applies the valid ones in parallel. A collapse reads only its own
// neighbourhood, where no other collapse of its round writes, but for the
// triangles across its fan's outer edges, whose far corner another collapse
// may move (settle() checks such pairs together), so nothing depends on the
// thread count.
#include "mesh.h"

#include <algorithm>
#include <cstring>
#include <utility>

namespace rsdf {
namespace {

constexpr uint32_t kNone = UINT32_MAX;
constexpr uint32_t kExact = 1u << 31;  // a sample owner flag: a vertex, held to the whole tolerance
constexpr size_t kChunk = 4096;
constexpr int kMaxCrease = 8;          // crease edges of a vertex that may take part in collapses
constexpr size_t kMaxFan = 512;        // triangles around a vertex a collapse may change
constexpr size_t kMaxAbsorbed = 4096;  // removed crease points remembered per crease edge
constexpr double kFacing = 0.5;        // changed triangles face within 60 degrees of their reference normal,
constexpr double kTurn = 0.25;         // turn by less than about 75 degrees,
constexpr double kNeedle = 0.05;       // and keep a quality of 0.05 (an angle of about 1.7 degrees),
                                       // unless they were already worse
constexpr double kFold = -0.5;         // the triangles on an edge of a changed triangle turn by at most 120 degrees
constexpr double kFoldSlack = 1e-3;    // or (normal cosines) by no more than the turns they replace there
constexpr double kRegular = 1e-3;      // Tikhonov weight of the quadric minimum, relative to the mean eigenvalue
constexpr double kLength = 1e-6;       // squared edge length weight of the order: shorter edges first on flats
constexpr int kMaxSpan = 16;           // most intervals per spoke, and per triangle side for interior samples
constexpr double kMargin = 0.88;       // field samples between vertices stay within this part of the tolerance
                                       // (a quadratic profile along an edge with one midpoint sample and both
                                       // ends within the tolerance peaks at most 16% above it)
constexpr size_t kGroup = 32;          // collapses validated per task
constexpr uint32_t kMaxRounds = 1u << 20;

enum Kind : uint8_t { kSmooth, kCrease, kCorner, kFrozen, kDead };

inline uint32_t next3(uint32_t c) { return c % 3 == 2 ? c - 2 : c + 1; }
inline uint32_t prev3(uint32_t c) { return c % 3 == 0 ? c + 2 : c - 1; }

inline double dot3(const double *a, const double *b) { return a[0] * b[0] + a[1] * b[1] + a[2] * b[2]; }

inline void sub3(const double *a, const double *b, double *out) {
    for (int i = 0; i < 3; ++i) out[i] = a[i] - b[i];
}

inline void cross3(const double *a, const double *b, double *out) {
    out[0] = a[1] * b[2] - a[2] * b[1];
    out[1] = a[2] * b[0] - a[0] * b[2];
    out[2] = a[0] * b[1] - a[1] * b[0];
}

inline double distance2(const double *a, const double *b) {
    double d[3];
    sub3(a, b, d);
    return dot3(d, d);
}

// Unnormalised normal (twice the area vector) of triangle a, b, c.
inline void facet(const double *a, const double *b, const double *c, double *out) {
    double u[3], v[3];
    sub3(b, a, u);
    sub3(c, a, v);
    cross3(u, v, out);
}

// Squared distance from p to the segment a-b.
double segment_distance2(const double *p, const double *a, const double *b) {
    double ab[3], ap[3];
    sub3(b, a, ab);
    sub3(p, a, ap);
    const double length2 = dot3(ab, ab);
    double t = length2 > 0 ? dot3(ap, ab) / length2 : 0;
    t = std::fmin(std::fmax(t, 0.0), 1.0);
    double q[3];
    for (int i = 0; i < 3; ++i) q[i] = a[i] + t * ab[i];
    return distance2(p, q);
}

// Squared distance from p to triangle a, b, c (Ericson, Real-Time Collision
// Detection, 5.1.5).
double triangle_distance2(const double *p, const double *a, const double *b, const double *c) {
    double ab[3], ac[3], ap[3];
    sub3(b, a, ab);
    sub3(c, a, ac);
    sub3(p, a, ap);
    const double d1 = dot3(ab, ap), d2 = dot3(ac, ap);
    if (d1 <= 0 && d2 <= 0) return distance2(p, a);
    double bp[3];
    sub3(p, b, bp);
    const double d3 = dot3(ab, bp), d4 = dot3(ac, bp);
    if (d3 >= 0 && d4 <= d3) return distance2(p, b);
    const double vc = d1 * d4 - d3 * d2;
    if (vc <= 0 && d1 >= 0 && d3 <= 0) return segment_distance2(p, a, b);
    double cp[3];
    sub3(p, c, cp);
    const double d5 = dot3(ab, cp), d6 = dot3(ac, cp);
    if (d6 >= 0 && d5 <= d6) return distance2(p, c);
    const double vb = d5 * d2 - d1 * d6;
    if (vb <= 0 && d2 >= 0 && d6 <= 0) return segment_distance2(p, a, c);
    const double va = d3 * d6 - d5 * d4;
    if (va <= 0 && d4 - d3 >= 0 && d5 - d6 >= 0) return segment_distance2(p, b, c);
    const double sum = va + vb + vc;
    if (!(sum > 0)) return std::fmin(segment_distance2(p, a, b), std::fmin(segment_distance2(p, a, c),
                                                                           segment_distance2(p, b, c)));
    const double v = vb / sum, w = vc / sum;
    double q[3];
    for (int i = 0; i < 3; ++i) q[i] = a[i] + v * ab[i] + w * ac[i];
    return distance2(p, q);
}

// Rounds a position to the single precision of the finished mesh, so the
// validated geometry is exactly the returned geometry.
inline void round_to_float(double *p) {
    for (int i = 0; i < 3; ++i) p[i] = static_cast<double>(static_cast<float>(p[i]));
}

// Weighted squared distances to planes, Q(p) = p'Ap - 2b'p + c, in
// coordinates relative to the mesh centre; w sums the weights (areas).
struct Quadric {
    double a[6] = {0, 0, 0, 0, 0, 0};  // a00 a01 a02 a11 a12 a22
    double b[3] = {0, 0, 0};
    double c = 0, w = 0;

    void add_plane(const double n[3], double d, double weight) {
        a[0] += weight * n[0] * n[0];
        a[1] += weight * n[0] * n[1];
        a[2] += weight * n[0] * n[2];
        a[3] += weight * n[1] * n[1];
        a[4] += weight * n[1] * n[2];
        a[5] += weight * n[2] * n[2];
        for (int i = 0; i < 3; ++i) b[i] += weight * d * n[i];
        c += weight * d * d;
        w += weight;
    }

    void add(const Quadric &q) {
        for (int i = 0; i < 6; ++i) a[i] += q.a[i];
        for (int i = 0; i < 3; ++i) b[i] += q.b[i];
        c += q.c;
        w += q.w;
    }

    double error(const double *p) const {
        const double x = p[0], y = p[1], z = p[2];
        const double pap = a[0] * x * x + a[3] * y * y + a[5] * z * z + 2 * (a[1] * x * y + a[2] * x * z + a[4] * y * z);
        return std::fmax(pap - 2 * (b[0] * x + b[1] * y + b[2] * z) + c, 0.0);
    }

    // The minimum about m: m + y with (A + eps I) y = b - A m, eps a small
    // fraction of the mean eigenvalue, so directions the planes do not
    // constrain stay at m. Returns false when the system is singular.
    bool minimum(const double *m, double *out) const {
        const double eps = kRegular * (a[0] + a[3] + a[5]) / 3;
        if (!(eps > 0)) return false;
        const double r[3] = {b[0] - (a[0] * m[0] + a[1] * m[1] + a[2] * m[2]),
                             b[1] - (a[1] * m[0] + a[3] * m[1] + a[4] * m[2]),
                             b[2] - (a[2] * m[0] + a[4] * m[1] + a[5] * m[2])};
        // Cholesky factorisation of A + eps I.
        const double m00 = a[0] + eps, m11 = a[3] + eps, m22 = a[5] + eps;
        if (!(m00 > 0)) return false;
        const double l00 = std::sqrt(m00), l10 = a[1] / l00, l20 = a[2] / l00;
        const double s11 = m11 - l10 * l10;
        if (!(s11 > 0)) return false;
        const double l11 = std::sqrt(s11), l21 = (a[4] - l20 * l10) / l11;
        const double s22 = m22 - l20 * l20 - l21 * l21;
        if (!(s22 > 0)) return false;
        const double l22 = std::sqrt(s22);
        const double z0 = r[0] / l00, z1 = (r[1] - l10 * z0) / l11, z2 = (r[2] - l20 * z0 - l21 * z1) / l22;
        const double y2 = z2 / l22, y1 = (z1 - l21 * y2) / l11, y0 = (z0 - l10 * y1 - l20 * y2) / l00;
        out[0] = m[0] + y0;
        out[1] = m[1] + y1;
        out[2] = m[2] + y2;
        return std::isfinite(out[0]) && std::isfinite(out[1]) && std::isfinite(out[2]);
    }
};

// Crease edges of a crease or corner vertex. Each crease edge owns a list of
// the crease points its collapses removed (absorbed), shared by the slots of
// both its endpoints.
struct CreaseNode {
    uint32_t count = 0;
    uint32_t neighbor[kMaxCrease];
    uint32_t list[kMaxCrease];
};

// A proposed collapse of the edge (tri[corner], tri[next3(corner)]): drop
// merges into keep, which moves to the quadric minimum when `optimal`.
struct Candidate {
    float key;
    uint32_t corner;
    uint8_t attempt;
    bool drop_first;
    bool optimal;
};

struct Selected {
    uint32_t candidate;
    uint32_t fan;          // corners around drop, then around keep, in fans_
    uint32_t drop_count, keep_count;
};

// A triangle around the merged vertex after a collapse: its corner there
// before the collapse and its unnormalised normal before and after.
struct RingTriangle {
    uint32_t corner;
    bool changed;
    double before[3], after[3];
};

// Per task scratch of validation.
struct Scratch {
    std::vector<double> x, y, z, values;
    std::vector<uint32_t> owner;
    std::vector<uint32_t> rim;
    std::vector<double> fan;  // 9 per triangle of the new fan
    std::vector<RingTriangle> ring;
};

class Simplifier {
public:
    // Takes over the input's positions and triangles.
    Simplifier(const MeshField &field, SimplifyInput &input, const SimplifySettings &settings)
        : field_(field), s_(settings), tolerance_(settings.tolerance), spacing_(2 * settings.cell_size),
          cos_crease_(std::cos(settings.crease_angle * M_PI / 180)), nv_(input.surface.positions.size() / 3),
          nt_(input.surface.triangles.size() / 3), pos_(std::move(input.surface.positions)),
          tri_(std::move(input.surface.triangles)) {
        if (nt_ >= UINT32_MAX / 3 || nv_ >= UINT32_MAX) throw Error("the mesh is too large to simplify");
        for (uint32_t v : tri_)
            if (v >= nv_) throw Error("the mesh references a missing vertex");
        double lo[3] = {INFINITY, INFINITY, INFINITY}, hi[3] = {-INFINITY, -INFINITY, -INFINITY};
        for (size_t v = 0; v < nv_; ++v)
            for (int a = 0; a < 3; ++a) {
                lo[a] = std::fmin(lo[a], pos_[3 * v + a]);
                hi[a] = std::fmax(hi[a], pos_[3 * v + a]);
            }
        for (int a = 0; a < 3; ++a) origin_[a] = nv_ ? 0.5 * (lo[a] + hi[a]) : 0.0;
        reference_normals(input.interior_normals);
        std::vector<uint32_t> first, corners;
        build_corners(first, corners);
        classify(first, corners, input.creases);
        build_quadrics(first, corners);
        memo_.assign(tri_.size(), 0);
        cached_round_.assign(tri_.size(), 0);
        cached_key_.assign(tri_.size(), 0.0f);
        cached_.assign(tri_.size(), 0);
        for (auto *stamps : {&changed_, &moved_, &region_, &opposite_}) stamps->assign(nv_, 0);
        mover_.assign(nv_, 0);
        alive_ = nt_;
    }

    Surface run();

private:
    // ------------------------------------------------------------ setup

    // Interior normals of the finished mesh, or facet normals without them.
    void reference_normals(const std::vector<float> &normals) {
        ref_.assign(3 * nt_, 0.0f);
        const bool given = normals.size() == 3 * nt_;
        parallel_chunks(chunks_of(nt_, kChunk), s_.threads, [&](size_t chunk) {
            const size_t end = std::min(nt_, (chunk + 1) * kChunk);
            for (size_t t = chunk * kChunk; t < end; ++t) {
                double n[3];
                if (given) {
                    for (int a = 0; a < 3; ++a) n[a] = normals[3 * t + a];
                } else {
                    facet(&pos_[3 * size_t(tri_[3 * t])], &pos_[3 * size_t(tri_[3 * t + 1])],
                          &pos_[3 * size_t(tri_[3 * t + 2])], n);
                }
                const double length = std::sqrt(dot3(n, n));
                if (!(length > 0) || !std::isfinite(length)) continue;
                for (int a = 0; a < 3; ++a) ref_[3 * t + a] = static_cast<float>(n[a] / length);
            }
        });
    }

    // Corners around each vertex (CSR) and the opposite-corner table: opp_[c]
    // is the corner across the edge opposite c, or kNone where that edge is
    // a boundary, non-manifold or inconsistently oriented.
    void build_corners(std::vector<uint32_t> &first, std::vector<uint32_t> &corners) {
        first.assign(nv_ + 1, 0);
        for (uint32_t v : tri_) ++first[v + 1];
        for (size_t v = 0; v < nv_; ++v) first[v + 1] += first[v];
        corners.resize(tri_.size());
        std::vector<uint32_t> fill(first.begin(), first.end() - 1);
        for (size_t c = 0; c < tri_.size(); ++c) corners[fill[tri_[c]]++] = static_cast<uint32_t>(c);
        opp_.assign(tri_.size(), kNone);
        parallel_chunks(chunks_of(nv_, kChunk), s_.threads, [&](size_t chunk) {
            const size_t end = std::min(nv_, (chunk + 1) * kChunk);
            for (size_t v = chunk * kChunk; v < end; ++v)
                for (uint32_t i = first[v]; i < first[v + 1]; ++i) {
                    // The directed edge v -> b, opposite corner prev3(c), and its twin b -> v.
                    const uint32_t c = corners[i], b = tri_[next3(c)];
                    bool unique = b != v;
                    for (uint32_t j = first[v]; j < first[v + 1] && unique; ++j)
                        unique = j == i || tri_[next3(corners[j])] != b;
                    uint32_t twin = kNone, twins = 0;
                    for (uint32_t j = first[b]; j < first[b + 1]; ++j)
                        if (tri_[next3(corners[j])] == v) {
                            twin = corners[j];
                            ++twins;
                        }
                    if (unique && twins == 1) opp_[prev3(c)] = prev3(twin);
                }
        });
    }

    // Walks the triangles around the vertex of corner `start` (counter-
    // clockwise seen from outside), appending their corners at that vertex
    // to `out` when given. Returns the triangle count, or 0 when the walk
    // meets a boundary or exceeds `limit` triangles.
    size_t walk(uint32_t start, std::vector<uint32_t> *out, size_t limit) const {
        uint32_t c = start;
        size_t n = 0;
        do {
            if (++n > limit) return 0;
            if (out) out->push_back(c);
            const uint32_t o = opp_[next3(c)];
            if (o == kNone) return 0;
            c = next3(o);
        } while (c != start);
        return n;
    }

    // Vertex kinds and crease nodes. A vertex whose triangles do not form one
    // closed fan is frozen, as is one with more than kMaxCrease crease edges.
    // Crease vertices have exactly two crease edges turning by at most the
    // crease angle; other vertices with crease edges are corners.
    void classify(const std::vector<uint32_t> &first, const std::vector<uint32_t> &corners,
                  const std::vector<uint32_t> &creases) {
        kind_.assign(nv_, kDead);
        parallel_chunks(chunks_of(nv_, kChunk), s_.threads, [&](size_t chunk) {
            const size_t end = std::min(nv_, (chunk + 1) * kChunk);
            for (size_t v = chunk * kChunk; v < end; ++v) {
                const size_t count = first[v + 1] - first[v];
                if (count == 0) continue;
                kind_[v] = walk(corners[first[v]], nullptr, count) == count ? kSmooth : kFrozen;
            }
        });
        crease_of_.assign(nv_, kNone);
        std::vector<uint8_t> overflow(nv_, 0);
        // Whether v can list one more crease edge toward `other` (creating its node).
        auto open = [&](uint32_t v, uint32_t other) {
            if (crease_of_[v] == kNone) {
                crease_of_[v] = static_cast<uint32_t>(creases_.size());
                creases_.emplace_back();
            }
            const CreaseNode &node = creases_[crease_of_[v]];
            for (uint32_t i = 0; i < node.count; ++i)
                if (node.neighbor[i] == other) return false;  // listed already
            if (node.count < kMaxCrease) return true;
            overflow[v] = 1;
            return false;
        };
        for (size_t i = 0; i + 1 < creases.size(); i += 2) {
            const uint32_t a = creases[i], b = creases[i + 1];
            if (a >= nv_ || b >= nv_ || a == b) continue;
            const bool open_a = open(a, b), open_b = open(b, a);
            if (!open_a || !open_b) continue;
            const uint32_t list = static_cast<uint32_t>(lists_.size());
            lists_.emplace_back();
            CreaseNode &na = creases_[crease_of_[a]], &nb = creases_[crease_of_[b]];
            na.neighbor[na.count] = b;
            na.list[na.count++] = list;
            nb.neighbor[nb.count] = a;
            nb.list[nb.count++] = list;
        }
        for (size_t v = 0; v < nv_; ++v) {
            if (kind_[v] != kSmooth || crease_of_[v] == kNone) continue;
            const CreaseNode &node = creases_[crease_of_[v]];
            if (overflow[v]) {
                kind_[v] = kFrozen;
            } else if (node.count == 2) {
                double d1[3], d2[3];
                sub3(&pos_[3 * v], &pos_[3 * size_t(node.neighbor[0])], d1);
                sub3(&pos_[3 * size_t(node.neighbor[1])], &pos_[3 * v], d2);
                const double scale = std::sqrt(dot3(d1, d1) * dot3(d2, d2));
                kind_[v] = scale > 0 && dot3(d1, d2) >= cos_crease_ * scale ? kCrease : kCorner;
            } else if (node.count > 0) {
                kind_[v] = kCorner;
            }
        }
    }

    // Area-weighted plane quadrics of the triangles around each vertex.
    void build_quadrics(const std::vector<uint32_t> &first, const std::vector<uint32_t> &corners) {
        quadric_.assign(nv_, Quadric());
        parallel_chunks(chunks_of(nv_, kChunk), s_.threads, [&](size_t chunk) {
            const size_t end = std::min(nv_, (chunk + 1) * kChunk);
            for (size_t v = chunk * kChunk; v < end; ++v)
                for (uint32_t i = first[v]; i < first[v + 1]; ++i) {
                    const size_t t = corners[i] / 3;
                    double p[3][3], n[3];
                    for (int k = 0; k < 3; ++k) sub3(&pos_[3 * size_t(tri_[3 * t + k])], origin_, p[k]);
                    facet(p[0], p[1], p[2], n);
                    const double length = std::sqrt(dot3(n, n));
                    if (!(length > 0) || !std::isfinite(length)) continue;
                    for (int a = 0; a < 3; ++a) n[a] /= length;
                    quadric_[v].add_plane(n, dot3(n, p[0]), 0.5 * length);
                }
        });
    }

    // ------------------------------------------------------------ proposals

    bool crease_edge(uint32_t a, uint32_t b) const {
        if (crease_of_[a] == kNone) return false;
        const CreaseNode &node = creases_[crease_of_[a]];
        for (uint32_t i = 0; i < node.count; ++i)
            if (node.neighbor[i] == b) return true;
        return false;
    }

    // Where keep goes when drop merges into it: the minimum of their summed
    // quadrics near the edge midpoint (optimal), or keep's own position.
    void placement(uint32_t drop, uint32_t keep, bool optimal, double *p) const {
        const double *pd = &pos_[3 * size_t(drop)], *pk = &pos_[3 * size_t(keep)];
        if (!optimal) {
            std::copy(pk, pk + 3, p);
            return;
        }
        Quadric q = quadric_[drop];
        q.add(quadric_[keep]);
        double m[3], x[3];
        for (int a = 0; a < 3; ++a) m[a] = 0.5 * (pd[a] + pk[a]) - origin_[a];
        if (!q.minimum(m, x) || distance2(x, m) > distance2(pd, pk)) std::copy(m, m + 3, x);
        for (int a = 0; a < 3; ++a) p[a] = x[a] + origin_[a];
        round_to_float(p);
    }

    // The order key of a collapse: the mean squared distance of the new
    // position to the planes merged into both quadrics, plus a small
    // length term so that flat regions collapse their shortest edges first.
    double key(uint32_t drop, uint32_t keep, bool optimal) const {
        double p[3];
        placement(drop, keep, optimal, p);
        Quadric q = quadric_[drop];
        q.add(quadric_[keep]);
        double r[3];
        sub3(p, origin_, r);
        const double error = q.w > 0 ? q.error(r) / q.w : 0.0;
        return error + kLength * distance2(&pos_[3 * size_t(drop)], &pos_[3 * size_t(keep)]);
    }

    struct Option {
        bool drop_first, optimal;
        double key;
    };

    // The collapse edge (a, b) tries at `attempt` (0, 1, ...), or false when
    // it has no more. Smooth vertices merge at the quadric minimum, else into
    // either end (the cheaper first); a smooth vertex merges into a crease or
    // corner vertex, which stays; crease vertices merge only along their
    // crease edges, into crease or corner vertices, which stay, so crease
    // vertices remain vertices of the original crease lines; corners never
    // move.
    bool option(uint32_t a, uint32_t b, uint32_t attempt, Option &out) const {
        const uint8_t ka = kind_[a], kb = kind_[b];
        auto half = [&](bool drop_first) {
            out = {drop_first, false, drop_first ? key(a, b, false) : key(b, a, false)};
            return true;
        };
        // Both ends in turn, the cheaper first.
        auto halves = [&](uint32_t which) {
            const double first = key(a, b, false), second = key(b, a, false);
            const bool drop_a = (second < first) == (which == 1);
            out = {drop_a, false, drop_a ? first : second};
            return true;
        };
        if (ka == kSmooth && kb == kSmooth) {
            if (attempt == 0) {
                out = {false, true, key(b, a, true)};
                return true;
            }
            return attempt <= 2 && halves(attempt - 1);
        }
        if (attempt == 0 && ka == kSmooth) return half(true);
        if (attempt == 0 && kb == kSmooth) return half(false);
        if (ka == kSmooth || kb == kSmooth || !crease_edge(a, b)) return false;
        if (ka == kCrease && kb == kCrease) return attempt <= 1 && halves(attempt);
        if (attempt == 0 && ka == kCrease) return half(true);
        if (attempt == 0 && kb == kCrease) return half(false);
        return false;
    }

    // One candidate per eligible edge, visited from the triangle where it
    // runs from its lower to its higher vertex. A rejected collapse tries
    // the edge's next option in a later round, and the first again once its
    // neighbourhood has changed. Only candidates that are the best (lowest
    // key, then first corner) at one of their ends are kept: a vertex ends at
    // most one collapse per round, so the others would mostly lose to them.
    // The minimum of unique (key, corner) pairs does not depend on the order
    // threads visit them.
    void gather(std::vector<Candidate> &out) {
        const size_t chunks = chunks_of(nt_, kChunk);
        proposed_.resize(tri_.size());
        best_.resize(nv_);
        auto packed = [&](uint32_t c) {
            uint32_t bits;
            std::memcpy(&bits, &cached_key_[c], sizeof bits);
            return uint64_t(bits) << 32 | c;
        };
        // Proposals (with cached options) and a reset of their ends' minima.
        parallel_chunks(chunks, s_.threads, [&](size_t chunk) {
            Option chosen;
            const size_t end = std::min(nt_, (chunk + 1) * kChunk);
            for (size_t t = chunk * kChunk; t < end; ++t) {
                proposed_[3 * t] = proposed_[3 * t + 1] = proposed_[3 * t + 2] = 0;
                if (tri_[3 * t] == kNone) continue;
                for (uint32_t c = static_cast<uint32_t>(3 * t); c < 3 * t + 3; ++c) {
                    const uint32_t a = tri_[c], b = tri_[next3(c)];
                    if (a > b || kind_[a] >= kFrozen || kind_[b] >= kFrozen || opp_[prev3(c)] == kNone) continue;
                    uint32_t attempt = 0;
                    if (memo_[c] && (memo_[c] >> 8) > std::max(changed_[a], changed_[b])) attempt = (memo_[c] & 255) + 1;
                    // The cached option stays valid while neither end moves.
                    if (cached_round_[c] <= std::max(moved_[a], moved_[b]) || (cached_[c] & 63) != attempt) {
                        if (!option(a, b, attempt, chosen)) continue;
                        float key_value = static_cast<float>(chosen.key);
                        if (!(key_value >= 0) || !std::isfinite(key_value)) key_value = 3.0e38f;
                        cached_key_[c] = key_value;
                        cached_[c] = static_cast<uint8_t>(attempt | chosen.drop_first << 6 | chosen.optimal << 7);
                        cached_round_[c] = round_;
                    }
                    proposed_[c] = 1;
                    __atomic_store_n(&best_[a], ~uint64_t(0), __ATOMIC_RELAXED);
                    __atomic_store_n(&best_[b], ~uint64_t(0), __ATOMIC_RELAXED);
                }
            }
        });
        parallel_chunks(chunks, s_.threads, [&](size_t chunk) {
            const size_t end = std::min(3 * nt_, 3 * (chunk + 1) * kChunk);
            for (size_t c = 3 * chunk * kChunk; c < end; ++c) {
                if (!proposed_[c]) continue;
                const uint64_t mine = packed(static_cast<uint32_t>(c));
                for (uint32_t v : {tri_[c], tri_[next3(static_cast<uint32_t>(c))]}) {
                    uint64_t current = __atomic_load_n(&best_[v], __ATOMIC_RELAXED);
                    while (mine < current && !__atomic_compare_exchange_n(&best_[v], &current, mine, true,
                                                                          __ATOMIC_RELAXED, __ATOMIC_RELAXED)) {
                    }
                }
            }
        });
        auto kept = [&](size_t c) {
            if (!proposed_[c]) return false;
            const uint64_t mine = packed(static_cast<uint32_t>(c));
            return __atomic_load_n(&best_[tri_[c]], __ATOMIC_RELAXED) == mine ||
                   __atomic_load_n(&best_[tri_[next3(static_cast<uint32_t>(c))]], __ATOMIC_RELAXED) == mine;
        };
        std::vector<size_t> offsets(chunks + 1, 0);
        parallel_chunks(chunks, s_.threads, [&](size_t chunk) {
            const size_t end = std::min(3 * nt_, 3 * (chunk + 1) * kChunk);
            size_t count = 0;
            for (size_t c = 3 * chunk * kChunk; c < end; ++c) count += kept(c);
            offsets[chunk + 1] = count;
        });
        for (size_t chunk = 0; chunk < chunks; ++chunk) offsets[chunk + 1] += offsets[chunk];
        out.resize(offsets[chunks]);
        parallel_chunks(chunks, s_.threads, [&](size_t chunk) {
            const size_t end = std::min(3 * nt_, 3 * (chunk + 1) * kChunk);
            size_t next = offsets[chunk];
            for (size_t c = 3 * chunk * kChunk; c < end; ++c) {
                if (!kept(c)) continue;
                const uint8_t meta = cached_[c];
                out[next++] = {cached_key_[c], static_cast<uint32_t>(c), static_cast<uint8_t>(meta & 63),
                               static_cast<bool>(meta & 64), static_cast<bool>(meta & 128)};
            }
        });
    }

    // ------------------------------------------------------------ selection

    // Picks collapses in key order (ties in edge order) so that no collapse's
    // ends lie in another's neighbourhood (both ends and the vertices around
    // them) and no two collapses remove triangles at the same opposite
    // vertex. Checking a candidate's ends suffices: had its neighbourhood
    // held another collapse's end, its own end would lie in that collapse's
    // neighbourhood. Fans of the selected collapses are kept in fans_.
    void select(const std::vector<Candidate> &candidates, const std::vector<uint64_t> &order,
                std::vector<Selected> &selected) {
        selected.clear();
        fans_.clear();
        for (uint64_t item : order) {
            const uint32_t index = static_cast<uint32_t>(item);
            const Candidate &cand = candidates[index];
            const uint32_t c = cand.corner, a = tri_[c], b = tri_[next3(c)];
            const uint32_t drop = cand.drop_first ? a : b, keep = cand.drop_first ? b : a;
            if (region_[drop] == round_ || region_[keep] == round_) continue;
            const uint32_t x = tri_[prev3(c)], y = tri_[opp_[prev3(c)]];
            if (opposite_[x] == round_ || opposite_[y] == round_) continue;
            const size_t start = fans_.size();
            const size_t nd = walk(cand.drop_first ? c : next3(c), &fans_, kMaxFan);
            const size_t nk = nd ? walk(cand.drop_first ? next3(c) : c, &fans_, kMaxFan) : 0;
            if (!nd || !nk) {
                fans_.resize(start);
                memo_[c] = round_ << 8 | 255;  // until its neighbourhood changes
                continue;
            }
            for (size_t i = start; i < fans_.size(); ++i) region_[tri_[next3(fans_[i])]] = round_;
            region_[drop] = region_[keep] = round_;
            opposite_[x] = opposite_[y] = round_;
            selected.push_back({index, static_cast<uint32_t>(start), static_cast<uint32_t>(nd),
                                static_cast<uint32_t>(nk)});
        }
    }

    // ------------------------------------------------------------ validation

    // Whether moving a triangle's corner from `from` to `to` keeps the
    // triangle sound: not degenerate, no new needle, facing its reference
    // normal (the interior normal it started with) and not turning sharply,
    // unless the change brings it closer to its reference.
    static bool triangle_ok(const double *from, const double *w1, const double *w2, const double *to,
                            const float *reference) {
        double n0[3], n1[3];
        facet(from, w1, w2, n0);
        facet(to, w1, w2, n1);
        const double l0 = std::sqrt(dot3(n0, n0)), l1 = std::sqrt(dot3(n1, n1));
        const double e0 = distance2(from, w1) + distance2(w1, w2) + distance2(w2, from);
        const double e1 = distance2(to, w1) + distance2(w1, w2) + distance2(w2, to);
        if (!(l1 > 1e-12 * e1)) return false;
        const double root12 = 3.4641016151377544;  // 2 sqrt(3): quality 4 sqrt(3) area / sum of squared sides
        const double q0 = e0 > 0 ? root12 * l0 / e0 : 0.0, q1 = root12 * l1 / e1;
        if (q1 < kNeedle && q1 < q0) return false;
        double r[3] = {reference[0], reference[1], reference[2]};
        if (dot3(r, r) == 0) {
            if (!(l0 > 0)) return true;
            for (int a = 0; a < 3; ++a) r[a] = n0[a] / l0;
        }
        const double f1 = dot3(n1, r) / l1, f0 = l0 > 0 ? dot3(n0, r) / l0 : -1.0;
        if (f1 < kFacing && f1 < f0) return false;
        if (l0 > 0 && dot3(n0, n1) < kTurn * l0 * l1 && f1 < f0) return false;
        return true;
    }

    void sample(const double *p, uint32_t owner, Scratch &s) const {
        s.x.push_back(p[0]);
        s.y.push_back(p[1]);
        s.z.push_back(p[2]);
        s.owner.push_back(owner);
    }

    // Intervals of `length` at most `spacing_` long: a multiple of `step`, at
    // least `least`, at most kMaxSpan.
    int span(double length, int least, int step) const {
        const double cells = std::ceil(length / spacing_);
        const int n = cells >= kMaxSpan ? kMaxSpan : std::max(least, static_cast<int>(cells));
        return std::min(kMaxSpan / step, (n + step - 1) / step) * step;
    }

    // Points inside segment a-b in an even number of intervals at most
    // `spacing_` long where possible, so they include the midpoint.
    void sample_segment(const double *a, const double *b, uint32_t owner, Scratch &s) const {
        const int m = span(std::sqrt(distance2(a, b)), 2, 2) - 1;
        for (int j = 1; j <= m; ++j) {
            const double t = static_cast<double>(j) / (m + 1);
            const double p[3] = {a[0] + t * (b[0] - a[0]), a[1] + t * (b[1] - a[1]), a[2] + t * (b[2] - a[2])};
            sample(p, owner, s);
        }
    }

    // Interior points of a barycentric grid whose divisions (a multiple of
    // three, so it holds the centroid) are at most `spacing_` long where
    // possible.
    void sample_triangle(const double *a, const double *b, const double *c, uint32_t owner, Scratch &s) const {
        const double longest = std::sqrt(std::max({distance2(a, b), distance2(b, c), distance2(c, a)}));
        const int n = span(longest, 3, 3);
        for (int i = 1; i <= n - 2; ++i)
            for (int j = 1; i + j <= n - 1; ++j) {
                const double wa = static_cast<double>(i) / n, wb = static_cast<double>(j) / n, wc = 1 - wa - wb;
                const double p[3] = {wa * a[0] + wb * b[0] + wc * c[0], wa * a[1] + wb * b[1] + wc * c[1],
                                     wa * a[2] + wb * b[2] + wc * c[2]};
                sample(p, owner, s);
            }
    }

    // Whether drop, keep, their neighbours and the new position p lie on one
    // plane (the area-weighted normal of the triangles around drop and keep,
    // through keep) within a thousandth of the tolerance plus four float
    // steps of the coordinates, with every old triangle and every changed
    // triangle (the first `changed` values of `fan`, 9 per triangle) facing
    // the same side: then the old and new triangles cover the same polygon.
    bool planar(const uint32_t *fd, size_t nd, const uint32_t *fk, size_t nk, const double *p,
                const std::vector<double> &fan, size_t changed) const {
        const uint32_t keep = tri_[fk[0]];
        const double *pk = &pos_[3 * size_t(keep)];
        double normal[3] = {0, 0, 0}, extent = std::fmax(std::fabs(p[0]), std::fmax(std::fabs(p[1]), std::fabs(p[2])));
        auto old_facet = [&](uint32_t c, double *f) {
            facet(&pos_[3 * size_t(tri_[c])], &pos_[3 * size_t(tri_[next3(c)])], &pos_[3 * size_t(tri_[prev3(c)])], f);
        };
        for (const uint32_t *corners : {fd, fk})
            for (size_t i = 0, n = corners == fd ? nd : nk; i < n; ++i) {
                double f[3];
                old_facet(corners[i], f);
                for (int a = 0; a < 3; ++a) normal[a] += f[a];
                const double *w = &pos_[3 * size_t(tri_[next3(corners[i])])];
                extent = std::fmax(extent, std::fmax(std::fabs(w[0]), std::fmax(std::fabs(w[1]), std::fabs(w[2]))));
            }
        const double length = std::sqrt(dot3(normal, normal));
        if (!(length > 0)) return false;
        const double limit = (1e-3 * tolerance_ + 0x1p-22 * extent) * length;
        auto off = [&](const double *q) {
            double d[3];
            sub3(q, pk, d);
            return !(std::fabs(dot3(d, normal)) <= limit);
        };
        if (off(p) || off(&pos_[3 * size_t(tri_[fd[0]])])) return false;
        for (const uint32_t *corners : {fd, fk})
            for (size_t i = 0, n = corners == fd ? nd : nk; i < n; ++i) {
                double f[3];
                old_facet(corners[i], f);
                if (off(&pos_[3 * size_t(tri_[next3(corners[i])])]) || !(dot3(f, normal) > 0)) return false;
            }
        for (size_t i = 0; i < changed; i += 9) {
            double f[3];
            facet(&fan[i], &fan[i + 3], &fan[i + 6], f);
            if (!(dot3(f, normal) > 0)) return false;
        }
        return true;
    }

    // Cosine between two unnormalised normals, or `undefined` when either vanishes.
    static double turn(const double *n1, const double *n2, double undefined) {
        const double scale = std::sqrt(dot3(n1, n1) * dot3(n2, n2));
        return scale > 0 && std::isfinite(scale) ? dot3(n1, n2) / scale : undefined;
    }

    // Whether an edge's two triangles turning by `after` (normal cosine) do
    // not fold where they turned by `before`.
    static bool keeps_shape(double after, double before) { return after >= kFold || after >= before - kFoldSlack; }

    // Whether the collapse folds no edge. The triangles around keep afterwards
    // are drop's (but ta and tb) in their order around drop, then keep's in
    // theirs; on every edge of a changed triangle (the spokes between them and
    // the outer edges, whose triangles across stay as they are) the two
    // triangles must turn by at most 120 degrees afterwards (kFold), or by no
    // more than the sharpest turn of the triangles they replace there: the
    // same two before the collapse, or the two that ta or tb separated, and
    // that removed triangle. A vanishing normal before the collapse sets no
    // turn. Another collapse of the round may move the corner across an
    // outer edge: settle() checks such pairs together, given the sharpest
    // outer edge turn afterwards against the old triangles across (`rim`,
    // -1 where such a triangle has no normal) and the largest turn of a
    // changed triangle (`rotation`), as normal cosines.
    bool unfolded(const uint32_t *fd, size_t nd, const uint32_t *fk, size_t nk, uint32_t ta, uint32_t tb,
                  const double *p, bool moved, Scratch &s, float *rim, float *rotation) const {
        auto corner_facet = [&](uint32_t corner, const double *at, double *out) {
            facet(at ? at : &pos_[3 * size_t(tri_[corner])], &pos_[3 * size_t(tri_[next3(corner)])],
                  &pos_[3 * size_t(tri_[prev3(corner)])], out);
        };
        // Each fan without ta and tb, in walk order (both walks start at ta,
        // and tb follows or precedes it), drop's then keep's.
        s.ring.clear();
        size_t drop_first = 0, drop_last = 0;  // ring positions of drop's ends
        for (int side = 0; side < 2; ++side) {
            const uint32_t *fan = side == 0 ? fd : fk;
            const size_t n = side == 0 ? nd : nk;
            const size_t begin = fan[1] / 3 == tb ? 2 : 1, end = fan[1] / 3 == tb ? n : n - 1;
            for (size_t i = begin; i < end; ++i) {
                RingTriangle entry;
                entry.corner = fan[i];
                entry.changed = side == 0 || moved;
                corner_facet(fan[i], nullptr, entry.before);
                corner_facet(fan[i], entry.changed ? p : nullptr, entry.after);
                s.ring.push_back(entry);
            }
            if (side == 0) drop_last = s.ring.size() - 1;
        }
        // The triangle a fan walk passes between its ring ends.
        double next_removed[3], prev_removed[3];
        const size_t last = fd[1] / 3 == tb ? nd - 1 : nd - 2;
        corner_facet(fd[(last + 1) % nd], nullptr, next_removed);
        corner_facet(fd[fd[1] / 3 == tb ? 1 : 0], nullptr, prev_removed);
        const size_t count = s.ring.size();
        double sharpest = 1, turned = 1;
        for (size_t i = 0; i < count; ++i) {
            const RingTriangle &one = s.ring[i], &two = s.ring[(i + 1) % count];
            if (one.changed) {  // the outer edge
                turned = std::fmin(turned, turn(one.before, one.after, 1.0));
                const uint32_t across = opp_[one.corner];
                if (across != kNone) {
                    double outer[3];
                    corner_facet(across, nullptr, outer);
                    if (!keeps_shape(turn(one.after, outer, 1.0), turn(one.before, outer, 1.0))) return false;
                    sharpest = std::fmin(sharpest, turn(one.after, outer, -1.0));
                }
            }
            if (!one.changed && !two.changed) continue;
            // The spoke between one and two: shared before, or ta or tb lay between them.
            double before = turn(one.before, two.before, 1.0);
            if (i == drop_last || (i + 1) % count == drop_first) {
                const double *removed = i == drop_last ? next_removed : prev_removed;
                before = std::fmin(before, std::fmin(turn(one.before, removed, 1.0), turn(removed, two.before, 1.0)));
            }
            if (!keeps_shape(turn(one.after, two.after, 1.0), before)) return false;
        }
        *rim = static_cast<float>(sharpest);
        *rotation = static_cast<float>(turned);
        return true;
    }

    // Everything about a collapse but the field samples, which it appends to
    // `s` (owner `owner`), keep's position afterwards (`placed`) and the
    // turns settle() needs (`shape`: rim, rotation). Reads only the
    // collapse's neighbourhood.
    bool check(const Candidate &cand, const Selected &sel, uint32_t owner, Scratch &s, double *placed,
               float *shape) const {
        const uint32_t c = cand.corner, a = tri_[c], b = tri_[next3(c)];
        const uint32_t drop = cand.drop_first ? a : b, keep = cand.drop_first ? b : a;
        const uint32_t cx = prev3(c), oc = opp_[cx], x = tri_[cx], y = tri_[oc];
        const uint32_t ta = c / 3, tb = oc / 3;
        const uint32_t *fd = &fans_[sel.fan], *fk = fd + sel.drop_count;
        const size_t nd = sel.drop_count, nk = sel.keep_count;
        const double tolerance2 = tolerance_ * tolerance_;
        if (x == y || kind_[x] == kFrozen || kind_[y] == kFrozen) return false;
        // The link condition: drop and keep share exactly the neighbours x and y.
        size_t common = 0;
        for (size_t i = 0; i < nd; ++i) {
            const uint32_t w = tri_[next3(fd[i])];
            for (size_t j = 0; j < nk; ++j)
                if (tri_[next3(fk[j])] == w) {
                    if (w != x && w != y) return false;
                    ++common;
                }
        }
        // keep, x and y keep at least three triangles each.
        if (common != 2 || nd + nk < 7 || walk(cx, nullptr, 3) || walk(oc, nullptr, 3)) return false;
        double p[3];
        placement(drop, keep, cand.optimal, p);
        std::copy(p, p + 3, placed);
        const double *pd = &pos_[3 * size_t(drop)], *pk = &pos_[3 * size_t(keep)];
        const bool moved = p[0] != pk[0] || p[1] != pk[1] || p[2] != pk[2];
        // A crease vertex merges along its crease into a vertex that stays: the
        // new crease edge keep-other stays within the tolerance of every crease
        // point it replaces (drop and the points earlier collapses absorbed
        // into drop's two crease edges).
        if (kind_[drop] == kCrease) {
            const CreaseNode &node = creases_[crease_of_[drop]];
            const int along = node.neighbor[0] == keep ? 0 : node.neighbor[1] == keep ? 1 : -1;
            if (along < 0 || moved) return false;
            const uint32_t other = node.neighbor[1 - along];
            if (crease_edge(keep, other)) return false;  // a three-edge crease loop
            const double *po = &pos_[3 * size_t(other)];
            if (segment_distance2(pd, pk, po) > tolerance2) return false;
            for (uint32_t list : {node.list[0], node.list[1]}) {
                const std::vector<double> &points = lists_[list];
                for (size_t i = 0; i < points.size(); i += 3)
                    if (segment_distance2(&points[i], pk, po) > tolerance2) return false;
            }
        } else if (kind_[drop] != kSmooth) {
            return false;
        }
        // Changed triangles (drop's, and keep's when it moves) stay sound;
        // their new corners, then keep's unchanged triangles, go to s.fan.
        s.fan.clear();
        auto add = [&](const double *u, const double *v, const double *w) {
            s.fan.insert(s.fan.end(), {u[0], u[1], u[2], v[0], v[1], v[2], w[0], w[1], w[2]});
        };
        for (size_t i = 0; i < nd; ++i) {
            const uint32_t corner = fd[i], t = corner / 3;
            if (t == ta || t == tb) continue;
            const double *w1 = &pos_[3 * size_t(tri_[next3(corner)])], *w2 = &pos_[3 * size_t(tri_[prev3(corner)])];
            if (!triangle_ok(pd, w1, w2, p, &ref_[3 * size_t(t)])) return false;
            add(p, w1, w2);
        }
        for (size_t i = 0; i < nk && moved; ++i) {
            const uint32_t corner = fk[i], t = corner / 3;
            if (t == ta || t == tb) continue;
            const double *w1 = &pos_[3 * size_t(tri_[next3(corner)])], *w2 = &pos_[3 * size_t(tri_[prev3(corner)])];
            if (!triangle_ok(pk, w1, w2, p, &ref_[3 * size_t(t)])) return false;
            add(p, w1, w2);
        }
        if (!unfolded(fd, nd, fk, nk, ta, tb, p, moved, s, &shape[0], &shape[1])) return false;
        const size_t changed = s.fan.size();
        for (size_t i = 0; i < nk && !moved; ++i) {
            const uint32_t corner = fk[i], t = corner / 3;
            if (t == ta || t == tb) continue;
            add(pk, &pos_[3 * size_t(tri_[next3(corner)])], &pos_[3 * size_t(tri_[prev3(corner)])]);
        }
        // Removed vertices stay within the tolerance of the new fan.
        auto near_fan = [&](const double *q) {
            for (size_t i = 0; i < s.fan.size(); i += 9)
                if (triangle_distance2(q, &s.fan[i], &s.fan[i + 3], &s.fan[i + 6]) <= tolerance2) return true;
            return false;
        };
        if (!near_fan(pd) || (moved && !near_fan(pk))) return false;
        // A planar fan whose triangles keep their orientation covers the same
        // polygon before and after: the surface is unchanged, so the field
        // needs no samples. Planarity allows for the single-precision
        // rounding of the coordinates.
        if (planar(fd, nd, fk, nk, p, s.fan, changed)) return true;
        // Field samples: the moved vertex, new or moved spokes, changed triangles.
        if (moved) sample(p, owner | kExact, s);
        s.rim.clear();
        for (size_t i = 0; i < nd; ++i) {
            const uint32_t w = tri_[next3(fd[i])];
            if (w != keep && (moved || (w != x && w != y))) s.rim.push_back(w);
        }
        for (size_t i = 0; i < nk && moved; ++i) {
            const uint32_t w = tri_[next3(fk[i])];
            if (w != drop && w != x && w != y) s.rim.push_back(w);
        }
        for (uint32_t w : s.rim) sample_segment(p, &pos_[3 * size_t(w)], owner, s);
        for (size_t i = 0; i < changed; i += 9) sample_triangle(&s.fan[i], &s.fan[i + 3], &s.fan[i + 6], owner, s);
        return true;
    }

    // Validates the selected collapses in groups (in Morton order of the
    // dropped vertex for costly fields, so field batches stay compact).
    void validate(const std::vector<Candidate> &candidates, const std::vector<Selected> &selected,
                  std::vector<uint8_t> &valid) {
        valid.assign(selected.size(), 0);
        placed_.resize(3 * selected.size());
        shape_.resize(2 * selected.size());
        std::vector<uint32_t> order;
        if (s_.spatial_order) {
            order = morton_order(selected.size(), s_.threads, [&](size_t i, double *xyz) {
                const Candidate &cand = candidates[selected[i].candidate];
                const uint32_t drop = tri_[cand.drop_first ? cand.corner : next3(cand.corner)];
                std::copy(&pos_[3 * size_t(drop)], &pos_[3 * size_t(drop)] + 3, xyz);
            });
        } else {
            order.resize(selected.size());
            for (size_t i = 0; i < order.size(); ++i) order[i] = static_cast<uint32_t>(i);
        }
        parallel_chunks(chunks_of(selected.size(), kGroup), s_.threads, [&](size_t group) {
            thread_local Scratch s;
            s.x.clear();
            s.y.clear();
            s.z.clear();
            s.owner.clear();
            const size_t begin = group * kGroup, end = std::min(selected.size(), begin + kGroup);
            for (size_t i = begin; i < end; ++i) {
                const Selected &sel = selected[order[i]];
                valid[order[i]] = check(candidates[sel.candidate], sel, static_cast<uint32_t>(i - begin), s,
                                        &placed_[3 * size_t(order[i])], &shape_[2 * size_t(order[i])]);
            }
            s.values.resize(s.x.size());
            for (size_t i = 0; i < s.x.size(); i += kBatch) {
                const int n = static_cast<int>(std::min<size_t>(kBatch, s.x.size() - i));
                field_.evaluate(n, &s.x[i], &s.y[i], &s.z[i], &s.values[i]);
            }
            for (size_t i = 0; i < s.values.size(); ++i) {
                const uint32_t owner = s.owner[i] & ~kExact;
                const double limit = s.owner[i] & kExact ? tolerance_ : kMargin * tolerance_;
                if (!(std::fabs(s.values[i]) <= limit)) valid[order[begin + owner]] = 0;
            }
            // The vertices valid collapses move, for settle().
            for (size_t i = begin; i < end; ++i) {
                const size_t index = order[i];
                if (!valid[index]) continue;
                uint32_t drop, keep;
                const bool moved = ends(candidates[selected[index].candidate], index, drop, keep);
                const uint64_t stamp = uint64_t(round_) << 32 | index;
                mover_[drop] = stamp;
                if (moved) mover_[keep] = stamp;
            }
        });
    }

    // A collapse's ends and whether keep moves (after validate()).
    bool ends(const Candidate &cand, size_t index, uint32_t &drop, uint32_t &keep) const {
        const uint32_t a = tri_[cand.corner], b = tri_[next3(cand.corner)];
        drop = cand.drop_first ? a : b;
        keep = cand.drop_first ? b : a;
        const double *pk = &pos_[3 * size_t(keep)], *p = &placed_[3 * index];
        return p[0] != pk[0] || p[1] != pk[1] || p[2] != pk[2];
    }

    // Collapses whose changed triangles share an outer edge (one moves the
    // corner across the other's outer edge) were each validated against the
    // other's old triangle. Where their combined result folds that edge
    // (keeps_shape() against the start of the round), the later of the two in
    // selection order waits for a later round (valid 2: no attempt used), so
    // the first valid collapse of every round goes ahead. Two triangles turn
    // by at most the sum of their turns against a third, so only a collapse
    // whose sharpest outer edge turn plus the round's largest triangle turn
    // exceeds 120 degrees needs this.
    void settle(const std::vector<Candidate> &candidates, const std::vector<Selected> &selected,
                std::vector<uint8_t> &valid) {
        double largest = 0;  // the round's largest turn of a changed triangle (radians)
        for (size_t i = 0; i < selected.size(); ++i)
            if (valid[i]) largest = std::fmax(largest, std::acos(std::fmax(-1.0, std::fmin(1.0, double(shape_[2 * i + 1])))));
        const double limit = std::acos(kFold) - largest;
        std::vector<uint8_t> waits(selected.size(), 0);
        parallel_chunks(chunks_of(selected.size(), kGroup), s_.threads, [&](size_t group) {
            const size_t end = std::min(selected.size(), (group + 1) * kGroup);
            for (size_t i = group * kGroup; i < end; ++i) {
                if (!valid[i] || std::acos(std::fmax(-1.0, std::fmin(1.0, double(shape_[2 * i])))) <= limit) continue;
                const Selected &sel = selected[i];
                const Candidate &cand = candidates[sel.candidate];
                const uint32_t ta = cand.corner / 3, tb = opp_[prev3(cand.corner)] / 3;
                uint32_t drop, keep;
                const bool moved = ends(cand, i, drop, keep);
                const double *p = &placed_[3 * i];
                for (size_t k = 0; k < size_t(sel.drop_count) + (moved ? sel.keep_count : 0) && !waits[i]; ++k) {
                    const uint32_t corner = fans_[sel.fan + k], t = corner / 3, across = opp_[corner];
                    if (t == ta || t == tb || across == kNone) continue;
                    const uint64_t stamp = mover_[tri_[across]];
                    if (stamp >> 32 != round_ || uint32_t(stamp) >= i) continue;
                    const double *w1 = &pos_[3 * size_t(tri_[next3(corner)])], *w2 = &pos_[3 * size_t(tri_[prev3(corner)])];
                    const double *u1 = &pos_[3 * size_t(tri_[next3(across)])], *u2 = &pos_[3 * size_t(tri_[prev3(across)])];
                    double before[3], after[3], outer_before[3], outer_after[3];
                    facet(&pos_[3 * size_t(tri_[corner])], w1, w2, before);
                    facet(p, w1, w2, after);
                    facet(&pos_[3 * size_t(tri_[across])], u1, u2, outer_before);
                    facet(&placed_[3 * size_t(uint32_t(stamp))], u1, u2, outer_after);
                    if (!keeps_shape(turn(after, outer_after, 1.0), turn(before, outer_before, 1.0))) waits[i] = 1;
                }
            }
        });
        for (size_t i = 0; i < selected.size(); ++i)
            if (waits[i]) valid[i] = 2;
    }

    // ------------------------------------------------------------ application

    // Records the round's outcome before the mesh changes: the neighbourhoods
    // of valid collapses change, rejected candidates remember their attempt.
    void record(const std::vector<Candidate> &candidates, const std::vector<Selected> &selected,
                const std::vector<uint8_t> &valid) {
        for (size_t i = 0; i < selected.size(); ++i) {
            const Selected &sel = selected[i];
            const Candidate &cand = candidates[sel.candidate];
            if (valid[i] != 1) {
                if (!valid[i]) memo_[cand.corner] = round_ << 8 | cand.attempt;
                continue;
            }
            const uint32_t a = tri_[cand.corner], b = tri_[next3(cand.corner)];
            changed_[a] = changed_[b] = round_;
            moved_[cand.drop_first ? b : a] = round_;
            for (size_t j = 0; j < size_t(sel.drop_count) + sel.keep_count; ++j)
                changed_[tri_[next3(fans_[sel.fan + j])]] = round_;
            alive_ -= 2;
        }
    }

    // Applies one validated collapse. Collapses of a round write disjoint
    // data: their own ends, the triangles around drop, the two triangles
    // across each removed triangle and the crease slots toward drop (a
    // crease neighbour that two of them share has its slots read atomically).
    void apply(const Candidate &cand, const Selected &sel) {
        const uint32_t c = cand.corner, a = tri_[c], b = tri_[next3(c)];
        const uint32_t drop = cand.drop_first ? a : b, keep = cand.drop_first ? b : a;
        const uint32_t cd = cand.drop_first ? c : next3(c), ck = cand.drop_first ? next3(c) : c;
        const uint32_t oc = opp_[prev3(c)], ta = c / 3, tb = oc / 3;
        const uint32_t cdb = tri_[next3(oc)] == drop ? next3(oc) : prev3(oc);
        const uint32_t ckb = cdb == next3(oc) ? prev3(oc) : next3(oc);
        double p[3];
        placement(drop, keep, cand.optimal, p);
        // The triangles across each removed triangle's two other sides become neighbours.
        auto join = [&](uint32_t u, uint32_t v) {
            const uint32_t ou = opp_[u], ov = opp_[v];
            opp_[ou] = ov;
            opp_[ov] = ou;
        };
        join(cd, ck);
        join(cdb, ckb);
        const uint32_t *fd = &fans_[sel.fan];
        for (size_t i = 0; i < sel.drop_count; ++i) {
            const uint32_t t = fd[i] / 3;
            if (t != ta && t != tb) tri_[fd[i]] = keep;
        }
        for (uint32_t t : {ta, tb}) tri_[3 * t] = tri_[3 * t + 1] = tri_[3 * t + 2] = kNone;
        std::copy(p, p + 3, &pos_[3 * size_t(keep)]);
        quadric_[keep].add(quadric_[drop]);
        if (kind_[drop] == kCrease) {
            CreaseNode &node = creases_[crease_of_[drop]];
            const int along = node.neighbor[0] == keep ? 0 : 1;
            const uint32_t other = node.neighbor[1 - along], list = node.list[along], gone = node.list[1 - along];
            std::vector<double> &points = lists_[list];
            points.insert(points.end(), &pos_[3 * size_t(drop)], &pos_[3 * size_t(drop)] + 3);
            points.insert(points.end(), lists_[gone].begin(), lists_[gone].end());
            std::vector<double>().swap(lists_[gone]);
            if (points.size() > 3 * kMaxAbsorbed) {  // keep every other point
                size_t kept = 0;
                for (size_t i = 0; i < points.size(); i += 6, kept += 3) std::copy(&points[i], &points[i] + 3, &points[kept]);
                points.resize(kept);
            }
            CreaseNode &kept_node = creases_[crease_of_[keep]];
            for (uint32_t i = 0; i < kept_node.count; ++i)
                if (kept_node.neighbor[i] == drop) kept_node.neighbor[i] = other;
            // Two crease collapses of a round may share `other`: each rewrites
            // only the slot toward its own drop, but reads the others' neighbours.
            CreaseNode &other_node = creases_[crease_of_[other]];
            for (uint32_t i = 0; i < other_node.count; ++i)
                if (__atomic_load_n(&other_node.neighbor[i], __ATOMIC_RELAXED) == drop) {
                    __atomic_store_n(&other_node.neighbor[i], keep, __ATOMIC_RELAXED);
                    other_node.list[i] = list;
                }
            node.count = 0;
        }
        kind_[drop] = kDead;
    }

    // Drops removed triangles from the triangle arrays.
    void compact() {
        std::vector<uint32_t> remap(nt_, kNone);
        size_t n = 0;
        for (size_t t = 0; t < nt_; ++t)
            if (tri_[3 * t] != kNone) remap[t] = static_cast<uint32_t>(n++);
        std::vector<uint32_t> tri(3 * n), opp(3 * n), memo(3 * n), cached_round(3 * n);
        std::vector<float> ref(3 * n), cached_key(3 * n);
        std::vector<uint8_t> cached(3 * n);
        parallel_chunks(chunks_of(nt_, kChunk), s_.threads, [&](size_t chunk) {
            const size_t end = std::min(nt_, (chunk + 1) * kChunk);
            for (size_t t = chunk * kChunk; t < end; ++t) {
                if (remap[t] == kNone) continue;
                const size_t to = 3 * size_t(remap[t]);
                for (int k = 0; k < 3; ++k) {
                    const uint32_t o = opp_[3 * t + k];
                    tri[to + k] = tri_[3 * t + k];
                    opp[to + k] = o == kNone ? kNone : 3 * remap[o / 3] + o % 3;
                    memo[to + k] = memo_[3 * t + k];
                    ref[to + k] = ref_[3 * t + k];
                    cached_round[to + k] = cached_round_[3 * t + k];
                    cached_key[to + k] = cached_key_[3 * t + k];
                    cached[to + k] = cached_[3 * t + k];
                }
            }
        });
        tri_.swap(tri);
        opp_.swap(opp);
        memo_.swap(memo);
        ref_.swap(ref);
        cached_round_.swap(cached_round);
        cached_key_.swap(cached_key);
        cached_.swap(cached);
        nt_ = n;
    }

    // The remaining triangles and the vertices they use, in index order.
    Surface output() const {
        Surface out;
        std::vector<uint32_t> index(nv_, kNone);
        for (size_t c = 0; c < 3 * nt_; ++c)
            if (tri_[c] != kNone) index[tri_[c]] = 0;
        uint32_t next = 0;
        for (size_t v = 0; v < nv_; ++v) {
            if (index[v] == kNone) continue;
            index[v] = next++;
            out.positions.insert(out.positions.end(), &pos_[3 * v], &pos_[3 * v] + 3);
        }
        out.triangles.reserve(3 * alive_);
        for (size_t c = 0; c < 3 * nt_; ++c)
            if (tri_[c] != kNone) out.triangles.push_back(index[tri_[c]]);
        return out;
    }

    const MeshField &field_;
    const SimplifySettings &s_;
    const double tolerance_, spacing_, cos_crease_;
    size_t nv_, nt_, alive_ = 0;
    double origin_[3] = {0, 0, 0};
    std::vector<double> pos_;
    std::vector<uint32_t> tri_, opp_, memo_;
    // Per edge (at its gathering corner): the option last computed, its key
    // and the round it was computed in (attempt | drop_first << 6 | optimal << 7).
    std::vector<uint32_t> cached_round_;
    std::vector<float> cached_key_;
    std::vector<uint8_t> cached_;
    std::vector<float> ref_;
    std::vector<Quadric> quadric_;
    std::vector<uint8_t> kind_;
    std::vector<uint32_t> changed_, moved_, region_, opposite_;
    std::vector<uint32_t> crease_of_;
    std::vector<CreaseNode> creases_;
    std::vector<std::vector<double>> lists_;
    std::vector<uint32_t> fans_;
    std::vector<uint64_t> best_;     // per vertex: its best candidate this round (gather)
    std::vector<uint8_t> proposed_;  // per corner: proposes its edge this round (gather)
    std::vector<double> placed_;     // per selected collapse: keep's position afterwards (validate)
    std::vector<float> shape_;       // per selected collapse: sharpest outer edge turn and largest triangle turn
    std::vector<uint64_t> mover_;    // per vertex: round << 32 | the valid collapse moving it (settle)
    uint32_t round_ = 0;
};

Surface Simplifier::run() {
    std::vector<Candidate> candidates;
    std::vector<uint64_t> order;
    std::vector<Selected> selected;
    std::vector<uint8_t> valid;
    while (round_ + 1 < kMaxRounds) {
        ++round_;
        gather(candidates);
        if (candidates.empty()) break;
        order.resize(candidates.size());
        for (size_t i = 0; i < candidates.size(); ++i) {
            uint32_t bits;
            std::memcpy(&bits, &candidates[i].key, sizeof bits);
            order[i] = uint64_t(bits) << 32 | i;
        }
        // Twelve bits of the key (exponent and three mantissa bits, steps of
        // at most 1/8) order the collapses; equal keys keep the mesh order,
        // which keeps selection's memory accesses local.
        radix_sort_items(order, 52, 12, s_.threads);
        select(candidates, order, selected);
        if (selected.empty()) break;
        validate(candidates, selected, valid);
        settle(candidates, selected, valid);
        record(candidates, selected, valid);
        parallel_chunks(chunks_of(selected.size(), kGroup), s_.threads, [&](size_t group) {
            const size_t end = std::min(selected.size(), (group + 1) * kGroup);
            for (size_t i = group * kGroup; i < end; ++i)
                if (valid[i] == 1) apply(candidates[selected[i].candidate], selected[i]);
        });
        if (2 * alive_ < nt_) compact();
    }
    return output();
}

}  // namespace

Surface simplify_surface(const MeshField &field, SimplifyInput input, const SimplifySettings &settings) {
    if (input.surface.triangles.empty()) return Surface();
    return Simplifier(field, input, settings).run();
}

}  // namespace rsdf
