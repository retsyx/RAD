// Moving points onto the surface and measuring how far triangles stray from
// it (SDF-DESIGN.md, "Meshing").
#include "mesh.h"

#include <algorithm>

namespace rsdf {
namespace {

constexpr size_t kChunk = 4096;
constexpr double kMinimumSlope = 0.25;  // |grad f| floor when estimating distances from |f|
static const double kTetra[4][3] = {{1, -1, -1}, {-1, -1, 1}, {-1, 1, -1}, {1, 1, 1}};

inline double dot3(const double *a, const double *b) { return a[0] * b[0] + a[1] * b[1] + a[2] * b[2]; }

// The order to visit `count` points in: Morton order for costly fields.
template <class PointAt>
std::vector<uint32_t> visiting_order(size_t count, int threads, bool spatial, const PointAt &point) {
    if (spatial) return morton_order(count, threads, point);
    std::vector<uint32_t> order(count);
    for (size_t i = 0; i < count; ++i) order[i] = static_cast<uint32_t>(i);
    return order;
}

// The largest |f| on a triangle between its samples: the quadratic through
// f at its corners (values[0..2] at corners a, b, c) and edge midpoints
// (samples[0..2]: ab, bc, ca) peaks inside it or along an edge (a chord's
// sagitta is quadratic), plus the most the quadratic misses f by at the
// centroid and the three interior points (samples[3..6]), which bounds
// what it cannot follow (a crease or blend inside the triangle). Writes the
// weights of b and c where the quadratic peaks to peak_at.
double interpolated_bound(const double corner[3], const double *samples, double *peak_at) {
    // P(s, t) = A s^2 + B t^2 + C s t + D s + E t + F, with s, t the weights of b and c.
    const double v0 = corner[0], v1 = corner[1], v2 = corner[2];
    const double m01 = samples[0], m12 = samples[1], m20 = samples[2];
    const double F = v0, A = 2 * v1 + 2 * v0 - 4 * m01, D = 4 * m01 - 3 * v0 - v1;
    const double B = 2 * v2 + 2 * v0 - 4 * m20, E = 4 * m20 - 3 * v0 - v2;
    const double C = 4 * (m12 - F - 0.5 * D - 0.5 * E) - A - B;
    auto P = [&](double u, double w) { return A * u * u + B * w * w + C * u * w + D * u + E * w + F; };
    double peak = -1;
    auto consider = [&](double u, double w) {
        const double value = std::fabs(P(u, w));
        if (!(value > peak)) return;
        peak = value;
        peak_at[0] = u;
        peak_at[1] = w;
    };
    consider(0, 0);
    consider(1, 0);
    consider(0, 1);
    // Edges: P(start + x (end - start)) = a x^2 + b x + c on [0, 1].
    const double ends[3][4] = {{0, 0, 1, 0}, {1, 0, 0, 1}, {0, 1, 0, 0}};
    for (const auto &e : ends) {
        const double c0 = P(e[0], e[1]), half = P(0.5 * (e[0] + e[2]), 0.5 * (e[1] + e[3])), c1 = P(e[2], e[3]);
        const double a = 2 * c0 + 2 * c1 - 4 * half, b = 4 * half - 3 * c0 - c1;
        if (a != 0) {
            const double x = -b / (2 * a);
            if (x > 0 && x < 1) consider(e[0] + x * (e[2] - e[0]), e[1] + x * (e[3] - e[1]));
        }
    }
    const double det = 4 * A * B - C * C;
    if (det != 0) {
        const double u = (C * E - 2 * B * D) / det, w = (C * D - 2 * A * E) / det;
        if (u >= 0 && w >= 0 && u + w <= 1) consider(u, w);
    }
    static const double kInterior[4][2] = {{1 / 3.0, 1 / 3.0}, {1 / 6.0, 1 / 6.0}, {2 / 3.0, 1 / 6.0}, {1 / 6.0, 2 / 3.0}};
    double miss = 0;
    for (int k = 0; k < 4; ++k) miss = std::fmax(miss, std::fabs(samples[3 + k] - P(kInterior[k][0], kInterior[k][1])));
    const double bound = peak + miss;
    return std::isfinite(bound) ? bound : INFINITY;
}

}  // namespace

void project_points(const MeshField &field, const MeshSettings &s, double settled, std::vector<double> &positions,
                    const std::vector<double> &guides, const std::vector<double> &boxes,
                    const std::vector<uint8_t> &fixed) {
    constexpr int kSamples = 4;  // field samples per point: at most three steps
    constexpr int kPoints = kBatch / 5;
    const size_t count = positions.size() / 3;
    const double eps = s.gradient_step;
    const std::vector<uint32_t> order = visiting_order(count, s.threads, s.spatial_order, [&](size_t i, double *xyz) {
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
                                 const std::vector<double> &directions, const std::vector<double> &sides) {
    constexpr int kPairs = kBatch / 8;  // two gradients of four samples each
    const size_t count = points.size() / 3;
    std::vector<double> cosines(count, 1.0);
    const std::vector<uint32_t> order = visiting_order(count, s.threads, s.spatial_order, [&](size_t i, double *xyz) {
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
                    const double offset = (k ? -sides[p] : sides[p]) / length;
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

std::vector<double> field_values(const MeshField &field, int threads, bool spatial, const std::vector<double> &points) {
    const size_t count = points.size() / 3;
    std::vector<double> out(count);
    const std::vector<uint32_t> order = visiting_order(count, threads, spatial, [&](size_t i, double *xyz) {
        std::copy_n(&points[3 * i], 3, xyz);
    });
    parallel_chunks(chunks_of(count, kChunk), threads, [&](size_t chunk) {
        double x[kBatch], y[kBatch], z[kBatch], f[kBatch];
        const size_t end = std::min(count, (chunk + 1) * kChunk);
        for (size_t first = chunk * kChunk; first < end; first += kBatch) {
            const int n = static_cast<int>(std::min<size_t>(kBatch, end - first));
            for (int i = 0; i < n; ++i) {
                const size_t p = order[first + i];
                x[i] = points[3 * p];
                y[i] = points[3 * p + 1];
                z[i] = points[3 * p + 2];
            }
            field.evaluate(n, x, y, z, f);
            for (int i = 0; i < n; ++i) out[order[first + i]] = f[i];
        }
    });
    return out;
}

void measure_triangles(const MeshField &field, int threads, bool spatial, double eps, const double *positions,
                       const uint32_t *triangles, const double *corner_values, const double *corner_slopes,
                       const std::vector<uint32_t> &which, std::vector<TriangleResult> &results) {
    // Samples per triangle: three edge midpoints, the centroid and three
    // interior points, then the four points of the centroid's gradient
    // stencil; then the point where the quadratic through the corner and
    // midpoint values peaks, with its stencil.
    constexpr int kSamples = 7, kStencil = 4, kPerTriangle = kSamples + kStencil;
    constexpr int kTriangles = kBatch / kPerTriangle;
    static const double kWeights[kSamples][3] = {{0.5, 0.5, 0}, {0, 0.5, 0.5}, {0.5, 0, 0.5}, {1 / 3.0, 1 / 3.0, 1 / 3.0},
                                                {2 / 3.0, 1 / 6.0, 1 / 6.0}, {1 / 6.0, 2 / 3.0, 1 / 6.0},
                                                {1 / 6.0, 1 / 6.0, 2 / 3.0}};
    const size_t count = which.size();
    auto corner = [&](uint32_t t, int k) { return &positions[3 * size_t(triangles[3 * size_t(t) + k])]; };
    auto centroid = [&](size_t i, double *xyz) {
        for (int a = 0; a < 3; ++a) xyz[a] = (corner(which[i], 0)[a] + corner(which[i], 1)[a] + corner(which[i], 2)[a]) / 3;
    };
    const std::vector<uint32_t> order = visiting_order(count, threads, spatial, centroid);
    parallel_chunks(chunks_of(count, kChunk), threads, [&](size_t chunk) {
        double x[kBatch], y[kBatch], z[kBatch], f[kBatch];
        const size_t end = std::min(count, (chunk + 1) * kChunk);
        for (size_t first = chunk * kChunk; first < end; first += kTriangles) {
            const int n = static_cast<int>(std::min<size_t>(kTriangles, end - first));
            for (int i = 0; i < n; ++i) {
                const uint32_t t = which[order[first + i]];
                const double *a = corner(t, 0), *b = corner(t, 1), *c = corner(t, 2);
                double centre[3];
                for (int k = 0; k < kSamples; ++k) {
                    const double *w = kWeights[k];
                    const int j = kPerTriangle * i + k;
                    x[j] = w[0] * a[0] + w[1] * b[0] + w[2] * c[0];
                    y[j] = w[0] * a[1] + w[1] * b[1] + w[2] * c[1];
                    z[j] = w[0] * a[2] + w[1] * b[2] + w[2] * c[2];
                }
                for (int axis = 0; axis < 3; ++axis) centre[axis] = (a[axis] + b[axis] + c[axis]) / 3;
                for (int k = 0; k < kStencil; ++k) {
                    const int j = kPerTriangle * i + kSamples + k;
                    x[j] = centre[0] + eps * kTetra[k][0];
                    y[j] = centre[1] + eps * kTetra[k][1];
                    z[j] = centre[2] + eps * kTetra[k][2];
                }
            }
            field.evaluate(kPerTriangle * n, x, y, z, f);
            double bound[kTriangles], peak[kTriangles][2], px[kBatch], py[kBatch], pz[kBatch], pf[kBatch];
            for (int i = 0; i < n; ++i) {
                const uint32_t t = which[order[first + i]];
                const double corner_value[3] = {corner_values[triangles[3 * size_t(t)]],
                                                corner_values[triangles[3 * size_t(t) + 1]],
                                                corner_values[triangles[3 * size_t(t) + 2]]};
                bound[i] = interpolated_bound(corner_value, &f[kPerTriangle * i], peak[i]);
                const double *a = corner(t, 0), *b = corner(t, 1), *c = corner(t, 2);
                const double wa = 1 - peak[i][0] - peak[i][1], wb = peak[i][0], wc = peak[i][1];
                const double p[3] = {wa * a[0] + wb * b[0] + wc * c[0], wa * a[1] + wb * b[1] + wc * c[1],
                                     wa * a[2] + wb * b[2] + wc * c[2]};
                for (int k = 0; k <= kStencil; ++k) {
                    const int j = (kStencil + 1) * i + k;
                    const double step = k < kStencil ? eps : 0;
                    px[j] = p[0] + step * kTetra[k % kStencil][0];
                    py[j] = p[1] + step * kTetra[k % kStencil][1];
                    pz[j] = p[2] + step * kTetra[k % kStencil][2];
                }
            }
            field.evaluate((kStencil + 1) * n, px, py, pz, pf);
            for (int i = 0; i < n; ++i) {
                const uint32_t t = which[order[first + i]];
                const double *values = &f[kPerTriangle * i];
                double g[3] = {0, 0, 0}, h[3] = {0, 0, 0};
                for (int k = 0; k < kStencil; ++k)
                    for (int axis = 0; axis < 3; ++axis) {
                        g[axis] += values[kSamples + k] * kTetra[k][axis];
                        h[axis] += pf[(kStencil + 1) * i + k] * kTetra[k][axis];
                    }
                const double length = std::sqrt(dot3(g, g)), peak_length = std::sqrt(dot3(h, h));
                // The smallest slope at the centroid, the corners and the peak.
                double slope = std::isfinite(length) ? length / (4 * eps) : 0;
                slope = std::fmin(slope, std::isfinite(peak_length) ? peak_length / (4 * eps) : 0);
                for (int k = 0; k < 3; ++k) slope = std::fmin(slope, corner_slopes[triangles[3 * size_t(t) + k]]);
                slope = std::fmax(slope, kMinimumSlope);
                double deviation = std::fmax(bound[i], std::fabs(pf[(kStencil + 1) * i + kStencil]));
                for (int k = 0; k < 3; ++k)
                    deviation = std::fmax(deviation, std::fabs(corner_values[triangles[3 * size_t(t) + k]]));
                for (int k = 0; k < kSamples; ++k) deviation = std::fmax(deviation, std::fabs(values[k]));
                TriangleResult &r = results[t];
                r.deviation = deviation / slope;
                const double *a = corner(t, 0), *b = corner(t, 1), *c = corner(t, 2);
                const double u[3] = {b[0] - a[0], b[1] - a[1], b[2] - a[2]}, v[3] = {c[0] - a[0], c[1] - a[1], c[2] - a[2]};
                const double facet[3] = {u[1] * v[2] - u[2] * v[1], u[2] * v[0] - u[0] * v[2], u[0] * v[1] - u[1] * v[0]};
                const double area = std::sqrt(dot3(facet, facet));
                if (length > 0 && std::isfinite(length)) {
                    for (int axis = 0; axis < 3; ++axis) r.normal[axis] = g[axis] / length;
                } else {
                    r.normal[0] = r.normal[1] = r.normal[2] = 0;
                }
                r.cosine = area > 0 && length > 0 && std::isfinite(area * length) ? dot3(facet, r.normal) / area : 1;
            }
        }
    });
}

}  // namespace rsdf
