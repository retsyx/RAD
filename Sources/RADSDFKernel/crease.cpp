// Align lighting discontinuities with shared mesh edges.
#include "mesh.h"

#include <algorithm>

namespace rsdf {
namespace {

constexpr size_t kChunk = 1024;
constexpr int kSearchSteps = 10;
const double kCosCrease = std::cos(7 * M_PI / 180);
const double kCosSteepCrease = std::cos(35 * M_PI / 180);

using Vector = std::array<double, 3>;

double dot(const Vector &a, const Vector &b) {
    return a[0] * b[0] + a[1] * b[1] + a[2] * b[2];
}

Vector point_at(const std::vector<double> &points, size_t i) {
    return {points[3 * i], points[3 * i + 1], points[3 * i + 2]};
}

struct Sample {
    Vector point, normal;
};

struct Span {
    Sample left, right;
    Vector guide;
    double lo = 0, hi = 1;
    double length;
};

Sample sample_surface(const MeshField &field, const Span &span, const Vector &a, const Vector &b,
                      double t, double eps) {
    Vector p;
    for (int axis = 0; axis < 3; ++axis) p[axis] = a[axis] + t * (b[axis] - a[axis]);
    Vector normal{};
    double offset = 0;
    // Project along the averaged outward direction. Keeping the direction
    // fixed preserves the parameter across the two faces of a crease.
    for (int step = 0; step < 5; ++step) {
        double x[5], y[5], z[5], f[5];
        constexpr double stencil[5][3] = {
            {1, -1, -1}, {-1, -1, 1}, {-1, 1, -1}, {1, 1, 1}, {0, 0, 0}
        };
        for (int i = 0; i < 5; ++i) {
            x[i] = p[0] + eps * stencil[i][0];
            y[i] = p[1] + eps * stencil[i][1];
            z[i] = p[2] + eps * stencil[i][2];
        }
        field.evaluate(5, x, y, z, f);
        Vector gradient{};
        for (int i = 0; i < 4; ++i)
            for (int axis = 0; axis < 3; ++axis)
                gradient[axis] += stencil[i][axis] * f[i] / (4 * eps);
        const double length = std::sqrt(dot(gradient, gradient));
        if (!(length > 0) || !std::isfinite(length)) break;
        const double slope = dot(gradient, span.guide);
        if (!(slope > 0.3 * length)) break;
        const double correction = f[4] / slope;
        if (std::fabs(correction) < eps) {
            for (int axis = 0; axis < 3; ++axis) normal[axis] = gradient[axis] / length;
            break;
        }
        offset -= correction;
        if (std::fabs(offset) > 0.25 * span.length) break;
        for (int axis = 0; axis < 3; ++axis) p[axis] -= correction * span.guide[axis];
    }
    return {p, normal};
}

struct Cut {
    Vector chord_point;
    CreaseSample sample;
    bool valid = false;
};

Cut trace(const MeshField &field, const MeshSettings &settings, const Vector &a, const Vector &b,
          const Vector &na, const Vector &nb) {
    Vector along, guide;
    for (int axis = 0; axis < 3; ++axis) {
        along[axis] = b[axis] - a[axis];
        guide[axis] = na[axis] + nb[axis];
    }
    const double length = std::sqrt(dot(along, along));
    const double guide_length = std::sqrt(dot(guide, guide));
    if (!(length > settings.cell_size / 100) || !(guide_length > 1)) return {};
    for (double &component : guide) component /= guide_length;
    Span span{{a, na}, {b, nb}, guide, 0, 1, length};
    const double eps = settings.gradient_step / 16;
    // Keep the half with the larger normal change. For a smooth surface the
    // change vanishes as the bracket shrinks, even across a narrow fillet.
    for (int step = 0; step < kSearchSteps; ++step) {
        const double t = 0.5 * (span.lo + span.hi);
        const Sample mid = sample_surface(field, span, a, b, t, eps);
        if (dot(mid.normal, mid.normal) == 0) return {};
        if (dot(span.left.normal, mid.normal) < dot(mid.normal, span.right.normal)) {
            span.hi = t;
            span.right = mid;
        } else {
            span.lo = t;
            span.left = mid;
        }
        if (dot(span.left.normal, span.right.normal) >= kCosCrease) return {};
    }
    const double t = 0.5 * (span.lo + span.hi);
    if (std::min(t, 1 - t) * length < 2 * settings.gradient_step) return {};
    // Move the normal stencils clear of the jump before testing its angle.
    // Strong creases already belong to the extraction and finishing passes.
    const double probe = std::max(0.55 * (span.hi - span.lo), 0.002 * settings.cell_size / length);
    const auto left = sample_surface(field, span, a, b, t - probe, eps);
    const auto right = sample_surface(field, span, a, b, t + probe, eps);
    const double agreement = dot(left.normal, right.normal);
    if (!(agreement < kCosCrease && agreement > kCosSteepCrease)) return {};
    Vector chord;
    for (int axis = 0; axis < 3; ++axis) chord[axis] = a[axis] + t * along[axis];
    if (same_preview_position(chord.data(), a.data()) || same_preview_position(chord.data(), b.data())) return {};
    const Sample hit = sample_surface(field, span, a, b, t, eps);
    if (dot(hit.normal, hit.normal) == 0) return {};
    return {chord, {hit.point, {left.normal, right.normal}}, true};
}

}  // namespace

void align_surface_creases(const MeshField &field, Surface &surface, const MeshSettings &settings,
                           const std::vector<double> &normals) {
    std::vector<uint64_t> candidates;
    for (size_t i = 0; i < surface.triangles.size(); i += 3) {
        const uint32_t *v = &surface.triangles[i];
        for (int k = 0; k < 3; ++k) {
            const uint32_t a = v[k], b = v[(k + 1) % 3];
            if ((surface.marks[a] & surface.marks[b]) & kSharpVertex) continue;
            if (dot(point_at(normals, a), point_at(normals, b)) < kCosCrease)
                candidates.push_back(surface_edge_key(a, b));
        }
    }
    if (candidates.empty()) return;
    std::sort(candidates.begin(), candidates.end());
    candidates.erase(std::unique(candidates.begin(), candidates.end()), candidates.end());
    std::vector<Cut> cuts(candidates.size());
    parallel_chunks(chunks_of(candidates.size(), kChunk), settings.threads, [&](size_t chunk) {
        const size_t end = std::min(candidates.size(), (chunk + 1) * kChunk);
        for (size_t index = chunk * kChunk; index < end; ++index) {
            const uint32_t a = candidates[index] >> 32, b = static_cast<uint32_t>(candidates[index]);
            cuts[index] = trace(field, settings, point_at(surface.positions, a), point_at(surface.positions, b),
                                point_at(normals, a), point_at(normals, b));
        }
    });

    std::vector<uint64_t> cut_edges;
    std::vector<Vector> chord_points;
    for (size_t i = 0; i < cuts.size(); ++i) {
        const Cut &cut = cuts[i];
        if (!cut.valid) continue;
        cut_edges.push_back(candidates[i]);
        const uint32_t a = candidates[i] >> 32, b = static_cast<uint32_t>(candidates[i]);
        const bool rounded_to_endpoint = same_preview_position(cut.sample.point.data(), &surface.positions[3 * size_t(a)]) ||
                                         same_preview_position(cut.sample.point.data(), &surface.positions[3 * size_t(b)]);
        const auto &position = rounded_to_endpoint ? cut.chord_point : cut.sample.point;
        surface.positions.insert(surface.positions.end(), position.begin(), position.end());
        surface.crease_samples.push_back(cut.sample);
        chord_points.push_back(cut.chord_point);
    }
    if (cut_edges.empty()) return;
    const size_t vertices = surface.positions.size() / 3;
    if (vertices >= UINT32_MAX) throw Error("the mesh has too many vertices");
    surface.first_crease_vertex = static_cast<uint32_t>(vertices - cut_edges.size());
    subdivide_surface(surface, cut_edges, surface.first_crease_vertex, chord_points);
}

}  // namespace rsdf
