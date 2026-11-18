// Moving points onto the surface and refining smooth features narrower than
// a cell (SDF-DESIGN.md, "Meshing" steps 5 and 7).
#include "mesh.h"

#include <algorithm>

namespace rsdf {
namespace {

constexpr size_t kChunk = 4096;
constexpr int kRefineLevels = 2;           // each edge splits into at most four
constexpr double kRefineTurn = 15;         // degrees the surface may turn along an edge
constexpr double kShortestEdge = 1.0 / 8;  // cells; shorter edges never split
constexpr double kSide = 0.05;             // cells either side of a point for side_cosines
constexpr double kBulge = 0.01;            // cells the surface may bulge from a triangle's centroid
static const double kTetra[4][3] = {{1, -1, -1}, {-1, -1, 1}, {-1, 1, -1}, {1, 1, 1}};

inline double dot3(const double *a, const double *b) { return a[0] * b[0] + a[1] * b[1] + a[2] * b[2]; }

// The order to visit `count` points in: Morton order for costly fields.
template <class PointAt>
std::vector<uint32_t> visiting_order(size_t count, const MeshSettings &s, const PointAt &point) {
    if (s.spatial_order) return morton_order(count, s.threads, point);
    std::vector<uint32_t> order(count);
    for (size_t i = 0; i < count; ++i) order[i] = static_cast<uint32_t>(i);
    return order;
}

inline uint64_t edge_key(uint32_t a, uint32_t b) {
    return a < b ? uint64_t(a) << 32 | b : uint64_t(b) << 32 | a;
}

// Field values at points (3 per point).
std::vector<double> values(const MeshField &field, const MeshSettings &s, const std::vector<double> &points) {
    const size_t count = points.size() / 3;
    std::vector<double> out(count);
    const std::vector<uint32_t> order = visiting_order(count, s, [&](size_t i, double *xyz) {
        std::copy(&points[3 * i], &points[3 * i] + 3, xyz);
    });
    parallel_chunks(chunks_of(count, kChunk), s.threads, [&](size_t chunk) {
        double x[kBatch], y[kBatch], z[kBatch], f[kBatch];
        const size_t end = std::min(count, (chunk + 1) * kChunk);
        for (size_t first = chunk * kChunk; first < end; first += kBatch) {
            const int n = static_cast<int>(std::min<size_t>(kBatch, end - first));
            for (int i = 0; i < n; ++i) {
                x[i] = points[3 * size_t(order[first + i])];
                y[i] = points[3 * size_t(order[first + i]) + 1];
                z[i] = points[3 * size_t(order[first + i]) + 2];
            }
            field.evaluate(n, x, y, z, f);
            for (int i = 0; i < n; ++i) out[order[first + i]] = f[i];
        }
    });
    return out;
}

}  // namespace

void project_points(const MeshField &field, const MeshSettings &s, double settled, std::vector<double> &positions,
                    const std::vector<double> &guides, const std::vector<double> &boxes,
                    const std::vector<uint8_t> &fixed) {
    constexpr int kSamples = 4;  // field samples per point: at most three steps
    constexpr int kPoints = kBatch / 5;
    const size_t count = positions.size() / 3;
    const double eps = s.gradient_step;
    const std::vector<uint32_t> order = visiting_order(count, s, [&](size_t i, double *xyz) {
        std::copy(&positions[3 * i], &positions[3 * i] + 3, xyz);
    });
    parallel_chunks(chunks_of(count, kChunk), s.threads, [&](size_t chunk) {
        double x[kPoints][3], best[kPoints][3], best_f[kPoints];
        double sx[kBatch], sy[kBatch], sz[kBatch], values[kBatch];
        int active[kPoints];
        const size_t end = std::min(count, (chunk + 1) * kChunk);
        for (size_t first = chunk * kChunk; first < end; first += kPoints) {
            const int n = static_cast<int>(std::min<size_t>(kPoints, end - first));
            int live = 0;
            for (int i = 0; i < n; ++i) {
                const size_t p = order[first + i];
                std::copy(&positions[3 * p], &positions[3 * p] + 3, x[i]);
                std::copy(x[i], x[i] + 3, best[i]);
                best_f[i] = INFINITY;
                if (fixed.empty() || !fixed[p]) active[live++] = i;
            }
            for (int sample = 0; sample < kSamples && live > 0; ++sample) {
                // The value at the point and the gradient by tetrahedral
                // differences. The mean of those differences is no value: at a
                // sharp corner it lies about eps outside.
                for (int j = 0; j < live; ++j) {
                    for (int k = 0; k < 4; ++k) {
                        sx[5 * j + k] = x[active[j]][0] + eps * kTetra[k][0];
                        sy[5 * j + k] = x[active[j]][1] + eps * kTetra[k][1];
                        sz[5 * j + k] = x[active[j]][2] + eps * kTetra[k][2];
                    }
                    sx[5 * j + 4] = x[active[j]][0];
                    sy[5 * j + 4] = x[active[j]][1];
                    sz[5 * j + 4] = x[active[j]][2];
                }
                field.evaluate(5 * live, sx, sy, sz, values);
                int next = 0;
                for (int j = 0; j < live; ++j) {
                    const int i = active[j];
                    const size_t p = order[first + i];
                    const double f = values[5 * j + 4];
                    double g[3] = {0, 0, 0};
                    for (int k = 0; k < 4; ++k)
                        for (int a = 0; a < 3; ++a) g[a] += values[5 * j + k] * kTetra[k][a];
                    if (!(std::fabs(f) < best_f[i])) {  // the last step overshot: try half of it
                        for (int a = 0; a < 3; ++a) x[i][a] = 0.5 * (x[i][a] + best[i][a]);
                        if (sample + 1 < kSamples) active[next++] = i;
                        continue;
                    }
                    std::copy(x[i], x[i] + 3, best[i]);
                    best_f[i] = std::fabs(f);
                    const double gg = dot3(g, g);
                    if (sample + 1 == kSamples || best_f[i] <= settled || !(gg > 0) || !std::isfinite(gg) ||
                        dot3(g, &guides[3 * p]) <= 0)
                        continue;
                    const double step = f * 4 * eps / gg;  // the gradient is g / (4 eps)
                    const double *box = &boxes[6 * p];
                    for (int a = 0; a < 3; ++a) x[i][a] = std::fmin(std::fmax(x[i][a] - step * g[a], box[a]), box[3 + a]);
                    active[next++] = i;
                }
                live = next;
            }
            for (int i = 0; i < n; ++i) std::copy(best[i], best[i] + 3, &positions[3 * size_t(order[first + i])]);
        }
    });
}

std::vector<double> side_cosines(const MeshField &field, const MeshSettings &s, const std::vector<double> &points,
                                 const std::vector<double> &directions, double side) {
    constexpr int kPairs = kBatch / 8;  // two gradients of four samples each
    const size_t count = points.size() / 3;
    std::vector<double> cosines(count, 1.0);
    const std::vector<uint32_t> order = visiting_order(count, s, [&](size_t i, double *xyz) {
        std::copy(&points[3 * i], &points[3 * i] + 3, xyz);
    });
    const BatchField batch = field.batch();
    parallel_chunks(chunks_of(count, kChunk), s.threads, [&](size_t chunk) {
        double x[2 * kPairs], y[2 * kPairs], z[2 * kPairs], n[3][2 * kPairs];
        size_t ids[kPairs];
        const size_t end = std::min(count, (chunk + 1) * kChunk);
        for (size_t first = chunk * kChunk; first < end;) {
            int m = 0;
            for (; first < end && m < kPairs; ++first) {
                const size_t p = order[first];
                const double *d = &directions[3 * p];
                const double length = std::sqrt(dot3(d, d));
                if (!(length > 0) || !std::isfinite(length)) continue;
                for (int k = 0; k < 2; ++k) {
                    const double offset = (k ? -side : side) / length;
                    x[2 * m + k] = points[3 * p] + offset * d[0];
                    y[2 * m + k] = points[3 * p + 1] + offset * d[1];
                    z[2 * m + k] = points[3 * p + 2] + offset * d[2];
                }
                ids[m++] = p;
            }
            if (m == 0) continue;
            batch_gradients(batch, 2 * m, x, y, z, s.gradient_step, n[0], n[1], n[2]);
            for (int i = 0; i < m; ++i)
                cosines[ids[i]] = n[0][2 * i] * n[0][2 * i + 1] + n[1][2 * i] * n[1][2 * i + 1] + n[2][2 * i] * n[2][2 * i + 1];
        }
    });
    return cosines;
}

// Splits edges along which the surface turns by more than kRefineTurn: the
// angle from one end's gradient to the gradient at the midpoint and on to
// the other end's (the ends alone miss a bead whose flank tilts between
// them). The first level tests the edges of triangles with a kSpreadVertex
// or whose centroid lies more than kBulge cells off the surface; a blend can
// pass through cells without crossing their grid edges, which then see only
// the faces beside it, but the triangles between their vertices span it.
// Later levels test the triangles with a vertex the previous level added.
// Triangles with a kSharpVertex, edges shorter than kShortestEdge cells and
// edges whose midpoint lies on a sharp edge (the gradients kSide cells either
// side along the edge differ by more than kRefineTurn and half its turn)
// stay. New vertices are the midpoints moved onto the surface. A triangle
// with one, two or three split edges becomes two, three or four (two split
// edges leave a quad, cut along its shorter diagonal), so the mesh stays
// conforming and keeps its orientation. Deterministic for any thread count.
void refine_surface(const MeshField &field, Surface &surface, const MeshSettings &s) {
    if (surface.marks.size() != surface.positions.size() / 3) return;
    const double h = s.cell_size, cos_turn = std::cos(kRefineTurn * M_PI / 180);
    std::vector<double> centroids(surface.triangles.size());
    for (size_t t = 0; t < surface.triangles.size() / 3; ++t)
        for (int a = 0; a < 3; ++a) {
            double sum = 0;
            for (int k = 0; k < 3; ++k) sum += surface.positions[3 * size_t(surface.triangles[3 * t + k]) + a];
            centroids[3 * t + a] = sum / 3;
        }
    const std::vector<double> bulges = values(field, s, centroids);
    size_t fresh = 0;  // the first vertex the last level added: later levels test only triangles with one
    for (int level = 0; level < kRefineLevels; ++level) {
        const size_t vertices = surface.positions.size() / 3, triangles = surface.triangles.size() / 3;
        const uint32_t *tri = surface.triangles.data();
        const double *pos = surface.positions.data();
        std::vector<uint64_t> keys;
        for (size_t t = 0; t < triangles; ++t) {
            const uint32_t *v = &tri[3 * t];
            const uint8_t marks = surface.marks[v[0]] | surface.marks[v[1]] | surface.marks[v[2]];
            if (marks & kSharpVertex) continue;
            if (level == 0 ? !(marks & kSpreadVertex) && !(std::fabs(bulges[t]) > kBulge * h)
                           : std::max({v[0], v[1], v[2]}) < fresh)
                continue;
            for (int k = 0; k < 3; ++k) keys.push_back(edge_key(v[k], v[(k + 1) % 3]));
        }
        std::sort(keys.begin(), keys.end());
        keys.erase(std::unique(keys.begin(), keys.end()), keys.end());
        // Long enough edges, their ends and midpoints.
        keys.erase(std::remove_if(keys.begin(), keys.end(), [&](uint64_t key) {
                       const double *a = &pos[3 * size_t(key >> 32)], *b = &pos[3 * size_t(uint32_t(key))];
                       const double d[3] = {b[0] - a[0], b[1] - a[1], b[2] - a[2]};
                       return !(dot3(d, d) > kShortestEdge * kShortestEdge * h * h);
                   }),
                   keys.end());
        if (keys.empty()) break;
        std::vector<uint32_t> ends;
        ends.reserve(2 * keys.size());
        for (uint64_t key : keys) ends.insert(ends.end(), {uint32_t(key >> 32), uint32_t(key)});
        std::sort(ends.begin(), ends.end());
        ends.erase(std::unique(ends.begin(), ends.end()), ends.end());
        const std::vector<double> end_normals = gradients(field, ends.size(), s.gradient_step, s.threads,
                                                          s.spatial_order, [&](size_t i, double *xyz) {
                                                              std::copy(&pos[3 * size_t(ends[i])], &pos[3 * size_t(ends[i])] + 3, xyz);
                                                          });
        auto end_normal = [&](uint32_t v) {
            return &end_normals[3 * size_t(std::lower_bound(ends.begin(), ends.end(), v) - ends.begin())];
        };
        std::vector<double> mids(3 * keys.size()), along(3 * keys.size());
        for (size_t i = 0; i < keys.size(); ++i) {
            const double *a = &pos[3 * size_t(keys[i] >> 32)], *b = &pos[3 * size_t(uint32_t(keys[i]))];
            for (int c = 0; c < 3; ++c) {
                mids[3 * i + c] = 0.5 * (a[c] + b[c]);
                along[3 * i + c] = b[c] - a[c];
            }
        }
        const std::vector<double> mid_normals = gradients(field, keys.size(), s.gradient_step, s.threads,
                                                          s.spatial_order, [&](size_t i, double *xyz) {
                                                              std::copy(&mids[3 * i], &mids[3 * i] + 3, xyz);
                                                          });
        // Edges along which the surface turns: their turn's half-angle cosine.
        std::vector<uint8_t> turning(keys.size(), 0);
        std::vector<double> half(keys.size(), 1.0);
        for (size_t i = 0; i < keys.size(); ++i) {
            const double *na = end_normal(uint32_t(keys[i] >> 32)), *nb = end_normal(uint32_t(keys[i]));
            const double *nm = &mid_normals[3 * i];
            const double ca = dot3(na, nm), cb = dot3(nm, nb);
            if (dot3(na, na) == 0 || dot3(nb, nb) == 0 || dot3(nm, nm) == 0) continue;
            const double turn = std::acos(std::fmax(-1.0, std::fmin(1.0, ca))) + std::acos(std::fmax(-1.0, std::fmin(1.0, cb)));
            if (!(std::cos(std::fmin(turn, M_PI)) < cos_turn)) continue;
            turning[i] = 1;
            half[i] = std::cos(0.5 * std::fmin(turn, M_PI));
        }
        std::vector<double> probe_points, probe_along;
        std::vector<size_t> probed;
        for (size_t i = 0; i < keys.size(); ++i) {
            if (!turning[i]) continue;
            probed.push_back(i);
            probe_points.insert(probe_points.end(), &mids[3 * i], &mids[3 * i] + 3);
            probe_along.insert(probe_along.end(), &along[3 * i], &along[3 * i] + 3);
        }
        const std::vector<double> gaps = side_cosines(field, s, probe_points, probe_along, kSide * h);
        std::vector<uint64_t> split;
        std::vector<double> points, guides, boxes;
        for (size_t j = 0; j < probed.size(); ++j) {
            const size_t i = probed[j];
            if (gaps[j] < std::fmin(half[i], cos_turn)) continue;  // a sharp edge between the samples
            split.push_back(keys[i]);
            const double *na = end_normal(uint32_t(keys[i] >> 32)), *nb = end_normal(uint32_t(keys[i]));
            const double reach = 0.5 * std::sqrt(dot3(&along[3 * i], &along[3 * i]));
            points.insert(points.end(), &mids[3 * i], &mids[3 * i] + 3);
            guides.insert(guides.end(), {na[0] + nb[0], na[1] + nb[1], na[2] + nb[2]});
            for (int side = 0; side < 2; ++side)
                for (int c = 0; c < 3; ++c) boxes.push_back(mids[3 * i + c] + (side ? reach : -reach));
        }
        if (split.empty()) break;
        fresh = vertices;
        project_points(field, s, 1e-6 * h, points, guides, boxes, {});
        if (vertices + split.size() >= UINT32_MAX) throw Error("the mesh has too many vertices");
        surface.positions.insert(surface.positions.end(), points.begin(), points.end());
        surface.marks.resize(vertices + split.size(), kSpreadVertex);

        // Conforming subdivision: new vertex of edge (a, b), or kNone.
        constexpr uint32_t kNone = UINT32_MAX;
        std::vector<uint8_t> touched(vertices, 0);
        for (uint64_t key : split) touched[key >> 32] = touched[uint32_t(key)] = 1;
        auto middle = [&](uint32_t a, uint32_t b) -> uint32_t {
            if (!touched[a] || !touched[b]) return kNone;
            const auto it = std::lower_bound(split.begin(), split.end(), edge_key(a, b));
            return it != split.end() && *it == edge_key(a, b) ? uint32_t(vertices + (it - split.begin())) : kNone;
        };
        auto distance2 = [&](uint32_t a, uint32_t b) {
            const double *p = &surface.positions[3 * size_t(a)], *q = &surface.positions[3 * size_t(b)];
            const double d[3] = {q[0] - p[0], q[1] - p[1], q[2] - p[2]};
            return dot3(d, d);
        };
        std::vector<uint32_t> out;
        out.reserve(surface.triangles.size() + 9 * split.size());
        for (size_t t = 0; t < triangles; ++t) {
            const uint32_t v[3] = {tri[3 * t], tri[3 * t + 1], tri[3 * t + 2]};
            const uint32_t m[3] = {middle(v[0], v[1]), middle(v[1], v[2]), middle(v[2], v[0])};  // m[k] on edge k
            const int count = (m[0] != kNone) + (m[1] != kNone) + (m[2] != kNone);
            if (count == 0) {
                out.insert(out.end(), v, v + 3);
            } else if (count == 3) {
                out.insert(out.end(), {v[0], m[0], m[2], m[0], v[1], m[1], m[2], m[1], v[2], m[0], m[1], m[2]});
            } else if (count == 1) {
                const int k = m[0] != kNone ? 0 : m[1] != kNone ? 1 : 2;
                const uint32_t a = v[k], b = v[(k + 1) % 3], c = v[(k + 2) % 3];
                out.insert(out.end(), {a, m[k], c, m[k], b, c});
            } else {
                const int k = m[0] == kNone ? 0 : m[1] == kNone ? 1 : 2;  // the unsplit edge
                const uint32_t a = v[k], b = v[(k + 1) % 3], c = v[(k + 2) % 3];
                const uint32_t p = m[(k + 1) % 3], q = m[(k + 2) % 3];   // on b-c and c-a
                out.insert(out.end(), {p, c, q});
                if (distance2(a, p) <= distance2(b, q)) {
                    out.insert(out.end(), {a, b, p, a, p, q});
                } else {
                    out.insert(out.end(), {a, b, q, b, p, q});
                }
            }
        }
        surface.triangles.swap(out);
    }
}

}  // namespace rsdf
