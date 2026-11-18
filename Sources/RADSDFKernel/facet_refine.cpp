// Locally subdivide preview triangles that interpolate across small panels of
// a triangle-mesh field. The new vertices sample the same surface.
#include "mesh.h"

#include <algorithm>

namespace rsdf {
namespace {

constexpr int kLevels = 5;
constexpr double kMinimumEdge = 1.0 / 32;
const double kSteep = std::cos(35 * M_PI / 180);
const double kCandidateTurn = std::cos(3 * M_PI / 180);
const double kNormalError = std::cos(5 * M_PI / 180);

double dot(const double *a, const double *b) {
    return a[0] * b[0] + a[1] * b[1] + a[2] * b[2];
}

struct Edge {
    uint64_t key;
    double error;
};

}  // namespace

void refine_mesh_facets(const MeshField &field, Surface &surface, const MeshSettings &s,
                        std::vector<double> &normals) {
    const double h = s.cell_size;
    const size_t original = surface.positions.size() / 3;
    // Bound the extra work independently of the preview triangle limit.
    const size_t max_extra = original * 3 / 2;
    size_t fresh = 0;
    for (int level = 0; level < kLevels; ++level) {
        const size_t vertices = surface.positions.size() / 3;
        const size_t room = std::min(original + max_extra - vertices, max_extra * (level + 1) / kLevels);
        if (room == 0) break;
        const auto point = [&](uint32_t v) { return &surface.positions[3 * size_t(v)]; };
        const auto normal = [&](uint32_t v) { return &normals[3 * size_t(v)]; };
        std::vector<size_t> candidates;
        for (size_t t = 0; t < surface.triangles.size(); t += 3) {
            const uint32_t *v = &surface.triangles[t];
            if (std::max({v[0], v[1], v[2]}) < fresh) continue;
            const double *a = normal(v[0]), *b = normal(v[1]), *c = normal(v[2]);
            const double turn = std::min({dot(a, b), dot(b, c), dot(c, a)});
            if (turn > kSteep && turn < kCandidateTurn) candidates.push_back(t);
        }
        if (candidates.empty()) break;
        // Compare the linear lighting model against three points biased
        // toward different corners. A thin panel can miss the centroid.
        auto probe = [&](size_t i, double *p) {
            const uint32_t *v = &surface.triangles[candidates[i / 3]];
            const double *a = point(v[i % 3]), *b = point(v[(i + 1) % 3]), *c = point(v[(i + 2) % 3]);
            for (int axis = 0; axis < 3; ++axis) p[axis] = (2 * a[axis] + b[axis] + c[axis]) / 4;
        };
        const auto sampled = gradients(field, 3 * candidates.size(), s.gradient_step, s.threads, s.spatial_order, probe);
        std::vector<double> widths(candidates.size());
        constexpr size_t kChunk = 4 * kBatch;
        parallel_chunks(chunks_of(candidates.size(), kChunk), s.threads, [&](size_t chunk) {
            double point[3][kBatch], value[kBatch];
            const size_t end = std::min(candidates.size(), (chunk + 1) * kChunk);
            for (size_t first = chunk * kChunk; first < end; first += kBatch) {
                const int count = static_cast<int>(std::min<size_t>(kBatch, end - first));
                for (int i = 0; i < count; ++i) {
                    double coordinates[3];
                    probe(3 * (first + i), coordinates);
                    for (int axis = 0; axis < 3; ++axis) point[axis][i] = coordinates[axis];
                }
                eval_batch(*field.graph, field.node, count, point[0], point[1], point[2], value, &widths[first], 0.3 * h);
            }
        });
        std::vector<Edge> selected;
        for (size_t i = 0; i < candidates.size(); ++i) {
            const uint32_t *v = &surface.triangles[candidates[i]];
            double agreement = 1;
            for (int k = 0; k < 3; ++k) {
                const double *a = normal(v[k]), *b = normal(v[(k + 1) % 3]), *c = normal(v[(k + 2) % 3]);
                const double blended[] = {2 * a[0] + b[0] + c[0], 2 * a[1] + b[1] + c[1], 2 * a[2] + b[2] + c[2]};
                const double *actual = &sampled[3 * (3 * i + k)];
                agreement = std::min(agreement, dot(actual, blended) / std::sqrt(dot(blended, blended)));
            }
            if (agreement >= kNormalError) continue;
            const double detail = std::clamp(0.4 * h / widths[i], 0.15, 10.0);
            double score[3]{};
            for (int k = 0; k < 3; ++k) {
                const uint32_t a = v[k], b = v[(k + 1) % 3];
                if ((surface.marks[a] & surface.marks[b]) & kSharpVertex) continue;
                const double *p = point(a), *q = point(b);
                const double d[] = {p[0] - q[0], p[1] - q[1], p[2] - q[2]};
                const double length2 = dot(d, d);
                if (length2 > kMinimumEdge * kMinimumEdge * h * h)
                    score[k] = length2 * (1 - dot(normal(a), normal(b)) + 0.0001);
            }
            const double strongest = std::max({score[0], score[1], score[2]});
            if (strongest == 0) continue;
            for (int k = 0; k < 3; ++k)
                if (score[k] >= 0.12 * strongest)
                    selected.push_back({surface_edge_key(v[k], v[(k + 1) % 3]), detail * detail * (1 - agreement) * score[k]});
        }
        if (selected.empty()) break;
        std::sort(selected.begin(), selected.end(), [](const Edge &a, const Edge &b) {
            return a.key == b.key ? a.error > b.error : a.key < b.key;
        });
        selected.erase(std::unique(selected.begin(), selected.end(), [](const Edge &a, const Edge &b) {
            return a.key == b.key;
        }), selected.end());
        if (selected.size() > room) {
            std::nth_element(selected.begin(), selected.begin() + room, selected.end(), [](const Edge &a, const Edge &b) {
                return a.error == b.error ? a.key < b.key : a.error > b.error;
            });
            selected.resize(room);
            std::sort(selected.begin(), selected.end(), [](const Edge &a, const Edge &b) { return a.key < b.key; });
        }

        if (vertices + selected.size() >= UINT32_MAX) throw Error("the mesh has too many vertices");
        std::vector<uint64_t> edges;
        std::vector<double> positions, guides, boxes;
        std::vector<std::array<double, 3>> chords;
        edges.reserve(selected.size());
        positions.reserve(3 * selected.size());
        guides.reserve(3 * selected.size());
        boxes.reserve(6 * selected.size());
        chords.reserve(selected.size());
        for (const Edge &edge : selected) {
            const uint32_t a = edge.key >> 32, b = static_cast<uint32_t>(edge.key);
            const double *p = point(a), *q = point(b);
            const double delta[] = {q[0] - p[0], q[1] - p[1], q[2] - p[2]};
            const double reach = 0.5 * std::sqrt(dot(delta, delta));
            const std::array<double, 3> mid{(p[0] + q[0]) * 0.5, (p[1] + q[1]) * 0.5, (p[2] + q[2]) * 0.5};
            if (same_preview_position(mid.data(), p) || same_preview_position(mid.data(), q)) continue;
            chords.push_back(mid);
            edges.push_back(edge.key);
            positions.insert(positions.end(), mid.begin(), mid.end());
            for (int axis = 0; axis < 3; ++axis) guides.push_back(normal(a)[axis] + normal(b)[axis]);
            for (double coordinate : mid) boxes.push_back(coordinate - reach);
            for (double coordinate : mid) boxes.push_back(coordinate + reach);
        }
        if (edges.empty()) break;
        project_points(field, s, 1e-6 * h, positions, guides, boxes, {});
        for (size_t i = 0; i < edges.size(); ++i) {
            const uint32_t a = edges[i] >> 32, b = static_cast<uint32_t>(edges[i]);
            double *projected = &positions[3 * i];
            if (same_preview_position(projected, point(a)) || same_preview_position(projected, point(b)))
                std::copy(chords[i].begin(), chords[i].end(), projected);
        }
        surface.positions.insert(surface.positions.end(), positions.begin(), positions.end());
        surface.marks.resize(vertices + edges.size(), kSpreadVertex);
        subdivide_surface(surface, edges, static_cast<uint32_t>(vertices), chords);
        const auto added = gradients(field, edges.size(), s.gradient_step, s.threads, s.spatial_order,
            [&](size_t i, double *p) { std::copy_n(point(static_cast<uint32_t>(vertices + i)), 3, p); });
        normals.insert(normals.end(), added.begin(), added.end());
        fresh = vertices;
    }
}

}  // namespace rsdf
