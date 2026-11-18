// Mesh finishing (SDF-DESIGN.md, "Meshing" steps 8-10): split shading normals,
// crease segments and statistics, plus the mesh C API.
#include "mesh.h"

#include <algorithm>
#include <chrono>
#include <cstring>
#include <memory>
#include <numeric>

struct rsdf_mesh {
    std::vector<float> positions;
    std::vector<float> normals;
    std::vector<uint32_t> triangles;
    std::vector<float> creases;
    rsdf_mesh_stats stats;
    // What rsdf_mesh_simplify needs: the meshing setup, the surface before
    // vertex splitting (split copy base_vertices + i came from vertex
    // split_origin[i]), crease edges as pairs of those vertices and each
    // triangle's interior normal.
    uint64_t graph = 0;  // rsdf_graph::identity
    uint32_t node = 0;
    double box_lo[3] = {0, 0, 0}, box_hi[3] = {0, 0, 0};
    double crease_angle = 35, gradient_step = 0;
    bool spatial_order = false;
    uint64_t base_vertices = 0;
    std::vector<uint32_t> split_origin, crease_edges;
    std::vector<float> interior_normals;
};

namespace rsdf {
namespace {

constexpr size_t kChunk = 4096;
const double kNormalAgreement = std::cos(20 * M_PI / 180);
// Meshing needs about 330 bytes per narrow-band cell, so the default limit
// keeps peak memory near 5 GB.
constexpr uint64_t kDefaultMaxCells = 16000000;

inline double dot(const double *a, const double *b) { return a[0] * b[0] + a[1] * b[1] + a[2] * b[2]; }

inline bool is_zero(const double *v) { return v[0] == 0 && v[1] == 0 && v[2] == 0; }

inline void normalize(double *v) {
    const double length = std::sqrt(dot(v, v));
    if (length > 0 && std::isfinite(length)) {
        for (int a = 0; a < 3; ++a) v[a] /= length;
    } else {
        v[0] = v[1] = v[2] = 0;
    }
}

// Unnormalised geometric normal (twice the area vector) of triangle t.
void face_vector(const Surface &surface, size_t t, double *out) {
    const double *a = &surface.positions[3 * size_t(surface.triangles[3 * t])];
    const double *b = &surface.positions[3 * size_t(surface.triangles[3 * t + 1])];
    const double *c = &surface.positions[3 * size_t(surface.triangles[3 * t + 2])];
    const double u[3] = {b[0] - a[0], b[1] - a[1], b[2] - a[2]}, v[3] = {c[0] - a[0], c[1] - a[1], c[2] - a[2]};
    out[0] = u[1] * v[2] - u[2] * v[1];
    out[1] = u[2] * v[0] - u[0] * v[2];
    out[2] = u[0] * v[1] - u[1] * v[0];
}

class Finisher {
public:
    Finisher(const Surface &surface, const MeshField &field, double eps, double crease_degrees, int threads,
             bool spatial)
        : surface_(surface), field_(field), eps_(eps), cos_crease_(std::cos(crease_degrees * M_PI / 180)),
          cos_half_crease_(std::cos(crease_degrees * M_PI / 360)), threads_(threads), spatial_(spatial), vertex_count_(surface.positions.size() / 3),
          triangle_count_(surface.triangles.size() / 3) {}

    void run(rsdf_mesh &mesh, std::vector<double> vertex_normals = {}) {
        vertex_normals_ = std::move(vertex_normals);
        const size_t existing = vertex_normals_.size() / 3;
        if (existing == 0) {
            vertex_normals_ = gradients(field_, vertex_count_, eps_, threads_, spatial_, [&](size_t v, double *p) {
                std::copy_n(&surface_.positions[3 * v], 3, p);
            });
        } else if (existing < vertex_count_) {
            const auto extra = gradients(field_, vertex_count_ - existing, eps_, threads_, spatial_, [&](size_t v, double *p) {
                std::copy_n(&surface_.positions[3 * (existing + v)], 3, p);
            });
            vertex_normals_.insert(vertex_normals_.end(), extra.begin(), extra.end());
        }
        triangle_normals(mesh.stats.cell_size);
        build_adjacency();
        assign_crease_sides(mesh.stats.cell_size);
        split_vertices(mesh);
        analyse_edges(mesh);
        prune_spurs(mesh);
        statistics(mesh);
        mesh.base_vertices = vertex_count_;
        mesh.interior_normals.assign(tri_normals_.begin(), tri_normals_.end());
    }

private:
    // Find the nearest sampled sign change along the outward guide and
    // return the exterior end of its bracket, within one gradient step.
    bool surface_offset(const double *point, const double *guide, double cell, double &outside) const {
        double value;
        field_.evaluate(1, point, point + 1, point + 2, &value);
        outside = 0;
        if (value == 0) return true;
        // Probe from the sample towards the expected side, nearest first.
        // A local sign bracket also works for fields scaled away from a
        // true distance. Search from 1/1024 to 1/4 of a cell.
        constexpr int probes = 9;
        double offsets[probes], xyz[3][probes], values[probes];
        double step = std::copysign(cell / 1024, -value);
        for (int i = 0; i < probes; ++i, step *= 2) {
            offsets[i] = step;
            for (int a = 0; a < 3; ++a) xyz[a][i] = point[a] + step * guide[a];
        }
        field_.evaluate(probes, xyz[0], xyz[1], xyz[2], values);
        for (int i = 0; i < probes; ++i) {
            if ((value < 0 && values[i] < 0) || (value > 0 && values[i] > 0)) continue;
            double near = i ? offsets[i - 1] : 0, far = offsets[i], sample[3];
            // A buried cap can make the field nearly constant across part of
            // the bracket. Resolve the crossing to the gradient stencil size.
            while (std::fabs(far - near) > eps_) {
                const double mid = 0.5 * (near + far);
                for (int a = 0; a < 3; ++a) sample[a] = point[a] + mid * guide[a];
                double f;
                field_.evaluate(1, sample, sample + 1, sample + 2, &f);
                if ((f < 0) == (value < 0)) near = mid;
                else far = mid;
            }
            outside = value < 0 ? far : near;
            return true;
        }
        return false;
    }

    // A triangle centroid or refined vertex can sit inside a union where the
    // field gradient points to a buried cap. If the gradient disagrees with
    // the local mesh, resample at the surface. The mesh guides the search
    // while field normals continue to define lighting and creases.
    void surface_normal(const double *point, double *guide, double cell, double *normal) const {
        normalize(guide);
        if (is_zero(guide)) return;
        const double agreement = dot(guide, normal);
        if (agreement >= kNormalAgreement) return;
        if (is_zero(normal)) std::copy_n(guide, 3, normal);

        double outside;
        if (!surface_offset(point, guide, cell, outside)) return;
        // Keep the tetrahedral gradient stencil on the exterior side of a seam.
        double sample[3], corrected[3];
        for (int a = 0; a < 3; ++a) sample[a] = point[a] + (outside + 2 * eps_) * guide[a];
        batch_gradients(field_.batch(), 1, sample, sample + 1, sample + 2, eps_,
                        corrected, corrected + 1, corrected + 2);
        if (!is_zero(corrected) && dot(corrected, guide) > agreement)
            std::copy_n(corrected, 3, normal);
    }

    void triangle_normals(double cell) {
        auto centroid = [&](size_t t, double *point) {
            for (int a = 0; a < 3; ++a) {
                double sum = 0;
                for (int k = 0; k < 3; ++k) sum += surface_.positions[3 * size_t(surface_.triangles[3 * t + k]) + a];
                point[a] = sum / 3;
            }
        };
        tri_normals_ = gradients(field_, triangle_count_, eps_, threads_, spatial_, centroid);
        parallel_chunks(chunks_of(triangle_count_, kChunk), threads_, [&](size_t chunk) {
            const size_t end = std::min(triangle_count_, (chunk + 1) * kChunk);
            for (size_t t = chunk * kChunk; t < end; ++t) {
                // The surface samples at the corners take precedence over a
                // poorly conditioned facet of a sliver or thin wall.
                double *normal = &tri_normals_[3 * t];
                const uint32_t *corners = &surface_.triangles[3 * t];
                if (std::all_of(corners, corners + 3, [&](uint32_t v) {
                        return dot(normal, &vertex_normals_[3 * size_t(v)]) >= kNormalAgreement;
                    })) continue;
                double facet[3], point[3];
                face_vector(surface_, t, facet);
                centroid(t, point);
                surface_normal(point, facet, cell, normal);
            }
        });
    }

    // Triangle corners (3 t + k) around each vertex, in triangle order.
    void build_adjacency() {
        first_corner_.assign(vertex_count_ + 1, 0);
        for (uint32_t v : surface_.triangles) ++first_corner_[v + 1];
        for (size_t v = 0; v < vertex_count_; ++v) first_corner_[v + 1] += first_corner_[v];
        corners_.resize(surface_.triangles.size());
        std::vector<uint32_t> fill(first_corner_.begin(), first_corner_.end() - 1);
        for (size_t c = 0; c < surface_.triangles.size(); ++c) corners_[fill[surface_.triangles[c]]++] = uint32_t(c);
    }

    // Probe just inside each incident triangle at the located surface crease.
    // Its two one-sided normals are shared by every corner on each face.
    void assign_crease_sides(double cell) {
        if (surface_.crease_samples.empty()) return;
        const size_t start = first_corner_[surface_.first_crease_vertex];
        const auto side_normals = gradients(field_, corners_.size() - start, eps_ / 16, threads_, spatial_,
            [&](size_t i, double *sample) {
                const uint32_t corner = corners_[start + i];
                const uint32_t *v = &surface_.triangles[3 * (corner / 3)];
                const uint32_t at = corner % 3;
                const double *p = &surface_.positions[3 * size_t(v[at])];
                const double *a = &surface_.positions[3 * size_t(v[(at + 1) % 3])];
                const double *b = &surface_.positions[3 * size_t(v[(at + 2) % 3])];
                double direction[3];
                for (int axis = 0; axis < 3; ++axis) direction[axis] = a[axis] + b[axis] - 2 * p[axis];
                normalize(direction);
                const auto &origin = surface_.crease_samples[v[at] - surface_.first_crease_vertex].point;
                for (int axis = 0; axis < 3; ++axis) sample[axis] = origin[axis] + 0.006 * cell * direction[axis];
            });
        crease_sides_.resize(side_normals.size() / 3);
        for (size_t i = 0; i < crease_sides_.size(); ++i) {
            const uint32_t v = surface_.triangles[corners_[start + i]];
            const auto &sides = surface_.crease_samples[v - surface_.first_crease_vertex].normals;
            const double *normal = &side_normals[3 * i];
            crease_sides_[i] = dot(normal, sides[1].data()) > dot(normal, sides[0].data());
        }
    }

    // Most surface vertex gradients already agree with their incident
    // triangles. Use the area vector of the fan to guide ambiguous samples.
    void vertex_normal(size_t v, double cell, double *normal) const {
        std::copy_n(&vertex_normals_[3 * v], 3, normal);
        for (uint32_t i = first_corner_[v]; i < first_corner_[v + 1]; ++i)
            if (dot(normal, &tri_normals_[3 * size_t(corners_[i] / 3)]) >= kNormalAgreement) return;
        double guide[3] = {};
        for (uint32_t i = first_corner_[v]; i < first_corner_[v + 1]; ++i) {
            double facet[3];
            face_vector(surface_, corners_[i] / 3, facet);
            for (int a = 0; a < 3; ++a) guide[a] += facet[a];
        }
        surface_normal(&surface_.positions[3 * v], guide, cell, normal);
        if (is_zero(normal)) normal[2] = 1;
    }

    // Whether the edge two triangles share is a crease (step 9): both have
    // interior normals and they differ by more than the crease angle.
    bool crease_between(size_t t0, size_t t1) const {
        const double *a = &tri_normals_[3 * t0], *b = &tri_normals_[3 * t1];
        return !is_zero(a) && !is_zero(b) && dot(a, b) < cos_crease_;
    }

    // Smoothing groups of the corners of vertex v: two corners share a group
    // when their triangles share an edge at v that is not a crease, so the
    // groups are the parts of the fan that crease edges separate. Writes each
    // corner's group (numbered from 0 in corner order) to `groups`; returns
    // the group count.
    uint32_t group_corners(size_t v, std::vector<uint32_t> &groups, std::vector<uint32_t> &parent) const {
        const uint32_t begin = first_corner_[v], count = first_corner_[v + 1] - begin;
        groups.assign(count, 0);
        if (v >= surface_.first_crease_vertex) {
            const size_t start = begin - first_corner_[surface_.first_crease_vertex];
            const uint8_t first_side = crease_sides_[start];
            uint32_t last_group = 0;
            for (uint32_t i = 1; i < count; ++i) {
                groups[i] = crease_sides_[start + i] != first_side;
                last_group |= groups[i];
            }
            return last_group + 1;
        }
        // Normals within half the crease angle of one of them pairwise differ
        // by less than the crease angle: no crease edge, one group.
        const double *first = count ? &tri_normals_[3 * size_t(corners_[begin] / 3)] : nullptr;
        bool smooth = true;
        for (uint32_t i = 1; i < count && smooth; ++i)
            smooth = dot(first, &tri_normals_[3 * size_t(corners_[begin + i] / 3)]) >= cos_half_crease_;
        if (smooth) return count ? 1 : 0;
        parent.resize(count);
        for (uint32_t i = 0; i < count; ++i) parent[i] = i;
        auto root = [&](uint32_t i) {
            while (parent[i] != i) i = parent[i] = parent[parent[i]];
            return i;
        };
        // The two other vertices of the triangle of corner c.
        auto ends = [&](uint32_t c, uint32_t out[2]) {
            const uint32_t t = c / 3, k = c % 3;
            out[0] = surface_.triangles[3 * t + (k + 1) % 3];
            out[1] = surface_.triangles[3 * t + (k + 2) % 3];
        };
        for (uint32_t i = 0; i < count; ++i) {
            uint32_t a[2];
            ends(corners_[begin + i], a);
            for (uint32_t j = i + 1; j < count; ++j) {
                uint32_t b[2];
                ends(corners_[begin + j], b);
                const bool adjacent = a[0] == b[0] || a[0] == b[1] || a[1] == b[0] || a[1] == b[1];
                if (!adjacent || crease_between(corners_[begin + i] / 3, corners_[begin + j] / 3)) continue;
                const uint32_t ri = root(i), rj = root(j);
                if (ri != rj) parent[std::max(ri, rj)] = std::min(ri, rj);
            }
        }
        uint32_t next = 0;
        for (uint32_t i = 0; i < count; ++i) {
            const uint32_t r = root(i);
            groups[i] = r == i ? next++ : groups[r];  // a root precedes its members
        }
        return next;
    }

    // Angle of the triangle corner c at its vertex (radians).
    double corner_angle(uint32_t c) const {
        const uint32_t t = c / 3, k = c % 3;
        const double *p = &surface_.positions[3 * size_t(surface_.triangles[3 * t + k])];
        const double *a = &surface_.positions[3 * size_t(surface_.triangles[3 * t + (k + 1) % 3])];
        const double *b = &surface_.positions[3 * size_t(surface_.triangles[3 * t + (k + 2) % 3])];
        const double u[3] = {a[0] - p[0], a[1] - p[1], a[2] - p[2]}, w[3] = {b[0] - p[0], b[1] - p[1], b[2] - p[2]};
        const double lengths = std::sqrt(dot(u, u) * dot(w, w));
        if (!(lengths > 0) || !std::isfinite(lengths)) return 0;
        return std::acos(std::fmax(-1.0, std::fmin(1.0, dot(u, w) / lengths)));
    }

    // Splits vertices at creases so each side shades with its own normal. A
    // vertex whose fan is one smoothing group uses its surface gradient.
    // A crease vertex lies where faces meet and its gradient blends them,
    // so each group takes the angle-weighted mean of its interior normals.
    void split_vertices(rsdf_mesh &mesh) {
        std::vector<uint32_t> extra(vertex_count_ + 1, 0);
        parallel_chunks(chunks_of(vertex_count_, kChunk), threads_, [&](size_t chunk) {
            std::vector<uint32_t> groups, parent;
            const size_t end = std::min(vertex_count_, (chunk + 1) * kChunk);
            for (size_t v = chunk * kChunk; v < end; ++v) {
                const uint32_t count = group_corners(v, groups, parent);
                extra[v + 1] = count > 1 ? count - 1 : 0;
            }
        });
        for (size_t v = 0; v < vertex_count_; ++v) extra[v + 1] += extra[v];
        const size_t total = vertex_count_ + extra[vertex_count_];
        if (total >= UINT32_MAX) throw Error("the mesh has too many vertices");
        mesh.positions.resize(3 * total);
        mesh.normals.resize(3 * total);
        mesh.triangles.resize(surface_.triangles.size());
        mesh.split_origin.resize(total - vertex_count_);
        parallel_chunks(chunks_of(vertex_count_, kChunk), threads_, [&](size_t chunk) {
            std::vector<uint32_t> groups, parent;
            std::vector<double> weighted, plain;
            const size_t end = std::min(vertex_count_, (chunk + 1) * kChunk);
            for (size_t v = chunk * kChunk; v < end; ++v) {
                const uint32_t count = std::max(group_corners(v, groups, parent), 1u);
                // Output vertex of each group: the original index first, then new ones.
                auto output = [&](uint32_t group) -> size_t {
                    return group == 0 ? v : vertex_count_ + extra[v] + group - 1;
                };
                weighted.assign(3 * count, 0.0);
                plain.assign(3 * count, 0.0);
                const uint32_t begin = first_corner_[v];
                for (size_t i = 0; i < groups.size(); ++i) {
                    const uint32_t corner = corners_[begin + i];
                    mesh.triangles[corner] = static_cast<uint32_t>(output(groups[i]));
                    if (count == 1 || v >= surface_.first_crease_vertex) continue;
                    const double *nt = &tri_normals_[3 * size_t(corner / 3)];
                    const double angle = corner_angle(corner);
                    for (int a = 0; a < 3; ++a) {
                        weighted[3 * groups[i] + a] += angle * nt[a];
                        plain[3 * groups[i] + a] += nt[a];
                    }
                }
                for (uint32_t group = 0; group < count; ++group) {
                    double normal[3];
                    if (v >= surface_.first_crease_vertex) {
                        const size_t start = begin - first_corner_[surface_.first_crease_vertex];
                        const auto &sides = surface_.crease_samples[v - surface_.first_crease_vertex].normals;
                        const auto &side = sides[crease_sides_[start] ^ group];
                        std::copy(side.begin(), side.end(), normal);
                    } else if (count == 1) {
                        vertex_normal(v, mesh.stats.cell_size, normal);
                    } else {
                        std::copy(&weighted[3 * group], &weighted[3 * group] + 3, normal);
                        normalize(normal);
                        if (is_zero(normal)) {  // only degenerate corners
                            std::copy(&plain[3 * group], &plain[3 * group] + 3, normal);
                            normalize(normal);
                        }
                        if (is_zero(normal)) vertex_normal(v, mesh.stats.cell_size, normal);
                    }
                    const size_t out = output(group);
                    if (out >= vertex_count_) mesh.split_origin[out - vertex_count_] = static_cast<uint32_t>(v);
                    for (int a = 0; a < 3; ++a) {
                        mesh.positions[3 * out + a] = static_cast<float>(surface_.positions[3 * v + a]);
                        mesh.normals[3 * out + a] = static_cast<float>(normal[a]);
                    }
                }
            }
        });
    }

    // Counts boundary and non-manifold edges and collects crease segments,
    // visiting each undirected edge from its lower vertex.
    void analyse_edges(rsdf_mesh &mesh) {
        const size_t chunks = chunks_of(vertex_count_, kChunk);
        std::vector<std::vector<float>> segments(chunks);
        std::vector<std::vector<uint32_t>> pairs(chunks);
        std::vector<uint64_t> boundary(chunks, 0), nonmanifold(chunks, 0);
        parallel_chunks(chunks, threads_, [&](size_t chunk) {
            std::vector<std::pair<uint32_t, uint32_t>> edges;  // (other vertex, triangle)
            const size_t end = std::min(vertex_count_, (chunk + 1) * kChunk);
            for (size_t v = chunk * kChunk; v < end; ++v) {
                edges.clear();
                for (uint32_t i = first_corner_[v]; i < first_corner_[v + 1]; ++i) {
                    const uint32_t corner = corners_[i], t = corner / 3, k = corner % 3;
                    for (uint32_t step : {1u, 2u}) {
                        const uint32_t other = surface_.triangles[3 * t + (k + step) % 3];
                        if (other > v) edges.emplace_back(other, t);
                    }
                }
                std::sort(edges.begin(), edges.end());
                for (size_t i = 0; i < edges.size();) {
                    size_t j = i + 1;
                    while (j < edges.size() && edges[j].first == edges[i].first) ++j;
                    if (j - i == 1) {
                        ++boundary[chunk];
                    } else if (j - i > 2) {
                        ++nonmanifold[chunk];
                    } else {
                        if (crease_between(edges[i].second, edges[i + 1].second)) {
                            const double *p = &surface_.positions[3 * v];
                            const double *q = &surface_.positions[3 * size_t(edges[i].first)];
                            segments[chunk].insert(segments[chunk].end(),
                                                   {float(p[0]), float(p[1]), float(p[2]), float(q[0]), float(q[1]),
                                                    float(q[2])});
                            pairs[chunk].insert(pairs[chunk].end(), {static_cast<uint32_t>(v), edges[i].first});
                        }
                    }
                    i = j;
                }
            }
        });
        for (size_t chunk = 0; chunk < chunks; ++chunk) {
            mesh.creases.insert(mesh.creases.end(), segments[chunk].begin(), segments[chunk].end());
            mesh.crease_edges.insert(mesh.crease_edges.end(), pairs[chunk].begin(), pairs[chunk].end());
            mesh.stats.boundary_edges += boundary[chunk];
            mesh.stats.nonmanifold_edges += nonmanifold[chunk];
        }
    }

    // Drops crease noise from the displayed lines: beside a sharp edge whose
    // vertices sit on either side of it (as along twisted edges) a stray mesh
    // edge or two crease off the line. Chains of crease edges run between
    // vertices whose crease degree is not 2; a chain shorter than a cell is
    // dropped when it hangs from a free end to a junction where the junction's
    // two other lines continue within 30 degrees of straight (a spur beside a
    // line; short real edges, such as those meeting at a chamfered corner,
    // meet other lines at a turn) or closes on its own junction, and one
    // shorter than two cells when it joins two junctions that a shorter chain
    // also joins (a detour beside the line). The crease edges simplification
    // reads keep them all.
    void prune_spurs(rsdf_mesh &mesh) {
        const size_t count = mesh.crease_edges.size() / 2;
        const double h = mesh.stats.cell_size;
        if (count == 0 || !(h > 0)) return;
        const uint32_t *ends = mesh.crease_edges.data();
        std::vector<uint32_t> start(vertex_count_ + 1, 0);
        for (size_t i = 0; i < 2 * count; ++i) ++start[ends[i] + 1];
        for (size_t v = 0; v < vertex_count_; ++v) start[v + 1] += start[v];
        std::vector<uint32_t> fill(start.begin(), start.end() - 1), incident(2 * count);
        for (size_t i = 0; i < 2 * count; ++i) incident[fill[ends[i]]++] = static_cast<uint32_t>(i / 2);
        auto degree = [&](uint32_t v) { return start[v + 1] - start[v]; };
        struct Chain {
            uint32_t from, to, first, last;  // end vertices (from <= to) and end segments
            double length;
            std::vector<uint32_t> segments;
        };
        std::vector<Chain> chains;  // short chains between vertices of degree other than 2
        for (size_t v = 0; v < vertex_count_; ++v) {
            const uint32_t d = degree(static_cast<uint32_t>(v));
            if (d == 0 || d == 2) continue;
            for (uint32_t k = start[v]; k < start[v + 1]; ++k) {
                Chain chain{static_cast<uint32_t>(v), 0, incident[k], 0, 0, {}};
                uint32_t at = chain.from, segment = chain.first;
                while (chain.length < 2 * h) {
                    chain.segments.push_back(segment);
                    const uint32_t next = ends[2 * segment] == at ? ends[2 * segment + 1] : ends[2 * segment];
                    const double *p = &surface_.positions[3 * size_t(at)], *q = &surface_.positions[3 * size_t(next)];
                    chain.length += std::sqrt((p[0] - q[0]) * (p[0] - q[0]) + (p[1] - q[1]) * (p[1] - q[1]) +
                                              (p[2] - q[2]) * (p[2] - q[2]));
                    at = next;
                    if (degree(next) != 2) break;
                    const uint32_t s0 = incident[start[next]], s1 = incident[start[next] + 1];
                    segment = s0 == segment ? s1 : s0;
                }
                if (!(chain.length < 2 * h)) continue;
                chain.to = at;
                chain.last = chain.segments.back();
                // Each chain is walked from both ends: keep one walk.
                if (chain.to < chain.from || (chain.to == chain.from && chain.last < chain.first)) continue;
                if (chain.to == chain.from && chain.last == chain.first && chain.segments.size() > 1) continue;
                chains.push_back(std::move(chain));
            }
        }
        std::vector<uint8_t> removed(count, 0);
        auto drop = [&](const Chain &chain) {
            for (uint32_t segment : chain.segments) removed[segment] = 1;
        };
        // Detours: chains joining the same two junctions, all but the shortest.
        std::sort(chains.begin(), chains.end(), [](const Chain &a, const Chain &b) {
            if (a.from != b.from) return a.from < b.from;
            if (a.to != b.to) return a.to < b.to;
            if (a.length != b.length) return a.length < b.length;
            return a.first < b.first;
        });
        // Spurs: (junction, its segment) of chains shorter than a cell from a
        // free end to a junction.
        std::vector<std::pair<uint32_t, uint32_t>> spurs;
        for (const Chain &chain : chains) {
            const uint32_t df = degree(chain.from), dt = degree(chain.to);
            if (chain.from != chain.to && (df == 1 || dt == 1) && (df >= 3 || dt >= 3) && chain.length < h)
                spurs.emplace_back(df >= 3 ? chain.from : chain.to, df >= 3 ? chain.first : chain.last);
        }
        std::sort(spurs.begin(), spurs.end());
        // The unit direction of the crease line leaving `v` along `segment`,
        // to the point about a cell along it (or its next junction).
        auto direction = [&](uint32_t v, uint32_t segment, double *out) {
            uint32_t at = v;
            double length = 0;
            for (;;) {
                const uint32_t next = ends[2 * segment] == at ? ends[2 * segment + 1] : ends[2 * segment];
                const double *p = &surface_.positions[3 * size_t(at)], *q = &surface_.positions[3 * size_t(next)];
                length += std::sqrt((p[0] - q[0]) * (p[0] - q[0]) + (p[1] - q[1]) * (p[1] - q[1]) +
                                    (p[2] - q[2]) * (p[2] - q[2]));
                at = next;
                if (length >= h || degree(next) != 2 || next == v) break;
                const uint32_t s0 = incident[start[next]], s1 = incident[start[next] + 1];
                segment = s0 == segment ? s1 : s0;
            }
            double norm = 0;
            for (int a = 0; a < 3; ++a) {
                out[a] = surface_.positions[3 * size_t(at) + a] - surface_.positions[3 * size_t(v) + a];
                norm += out[a] * out[a];
            }
            norm = std::sqrt(norm);
            for (int a = 0; a < 3; ++a) out[a] = norm > 0 ? out[a] / norm : 0.0;
        };
        // Whether the lines at junction v other than its spurs are two that
        // continue within 30 degrees of straight.
        auto spurs_beside_line = [&](uint32_t v) {
            uint32_t other[2], found = 0;
            for (uint32_t k = start[v]; k < start[v + 1]; ++k) {
                const uint32_t segment = incident[k];
                if (std::binary_search(spurs.begin(), spurs.end(), std::make_pair(v, segment))) continue;
                if (found == 2) return false;
                other[found++] = segment;
            }
            if (found != 2) return false;
            double a[3], b[3];
            direction(v, other[0], a);
            direction(v, other[1], b);
            return a[0] * b[0] + a[1] * b[1] + a[2] * b[2] <= -0.8660254037844386;  // cos 150 degrees
        };
        for (size_t i = 0; i < chains.size(); ++i) {
            const Chain &chain = chains[i];
            const uint32_t df = degree(chain.from), dt = degree(chain.to);
            if (chain.from == chain.to) {
                if (df >= 3 && chain.length < h) drop(chain);  // a short loop on a junction
            } else if (df == 1 || dt == 1) {
                if ((df >= 3 || dt >= 3) && chain.length < h && spurs_beside_line(df >= 3 ? chain.from : chain.to))
                    drop(chain);  // a spur beside a line
            } else if (i > 0 && chains[i - 1].from == chain.from && chains[i - 1].to == chain.to) {
                drop(chain);  // a longer detour between the junctions of a shorter chain
            }
        }
        size_t kept = 0;
        for (size_t i = 0; i < count; ++i)
            if (!removed[i]) {
                if (kept != i) std::copy_n(&mesh.creases[6 * i], 6, &mesh.creases[6 * kept]);
                ++kept;
            }
        mesh.creases.resize(6 * kept);
    }

    void statistics(rsdf_mesh &mesh) {
        rsdf_mesh_stats &stats = mesh.stats;
        stats.vertex_count = mesh.positions.size() / 3;
        stats.triangle_count = triangle_count_;
        stats.crease_segment_count = mesh.creases.size() / 6;
        stats.active_cells = surface_.active_cells;
        if (vertex_count_ == 0) return;

        const double *p = surface_.positions.data();
        double lo[3] = {p[0], p[1], p[2]}, hi[3] = {p[0], p[1], p[2]};
        for (size_t v = 1; v < vertex_count_; ++v)
            for (int a = 0; a < 3; ++a) {
                lo[a] = std::min(lo[a], p[3 * v + a]);
                hi[a] = std::max(hi[a], p[3 * v + a]);
            }
        const double reference[3] = {0.5 * (lo[0] + hi[0]), 0.5 * (lo[1] + hi[1]), 0.5 * (lo[2] + hi[2])};

        // Divergence theorem about the bounds centre, summed in fixed chunks for determinism.
        const size_t chunks = chunks_of(triangle_count_, kChunk);
        std::vector<double> partial(8 * chunks, 0.0);  // volume, area, 3 x volume moment, 3 x area moment
        parallel_chunks(chunks, threads_, [&](size_t chunk) {
            double *sum = &partial[8 * chunk];
            const size_t end = std::min(triangle_count_, (chunk + 1) * kChunk);
            for (size_t t = chunk * kChunk; t < end; ++t) {
                double q[3][3];
                for (int k = 0; k < 3; ++k)
                    for (int a = 0; a < 3; ++a) q[k][a] = p[3 * size_t(surface_.triangles[3 * t + k]) + a] - reference[a];
                double facet[3];
                face_vector(surface_, t, facet);
                const double volume = dot(q[0], facet) / 6;
                const double area = 0.5 * std::sqrt(dot(facet, facet));
                sum[0] += volume;
                sum[1] += area;
                for (int a = 0; a < 3; ++a) {
                    const double corner_sum = q[0][a] + q[1][a] + q[2][a];
                    sum[2 + a] += volume * corner_sum / 4;
                    sum[5 + a] += area * corner_sum / 3;
                }
            }
        });
        double total[8] = {};
        for (size_t chunk = 0; chunk < chunks; ++chunk)
            for (int i = 0; i < 8; ++i) total[i] += partial[8 * chunk + i];
        stats.volume = total[0];
        stats.area = total[1];
        for (int a = 0; a < 3; ++a) {
            const double offset = total[0] != 0 ? total[2 + a] / total[0] : total[1] > 0 ? total[5 + a] / total[1] : 0;
            stats.centroid[a] = reference[a] + offset;
            stats.bounds_lo[a] = lo[a];
            stats.bounds_hi[a] = hi[a];
        }

        // Connected components (union-find over triangle edges).
        std::vector<uint32_t> parent(vertex_count_);
        std::iota(parent.begin(), parent.end(), 0u);
        auto root = [&](uint32_t v) {
            while (parent[v] != v) v = parent[v] = parent[parent[v]];
            return v;
        };
        for (size_t t = 0; t < triangle_count_; ++t)
            for (int k = 0; k < 2; ++k) {
                const uint32_t a = root(surface_.triangles[3 * t + k]), b = root(surface_.triangles[3 * t + k + 1]);
                if (a != b) parent[std::max(a, b)] = std::min(a, b);
            }
        for (size_t v = 0; v < vertex_count_; ++v) stats.components += root(static_cast<uint32_t>(v)) == v;

        // Sealed voids: components enclosing negative volume (surfaces facing
        // into a void), summed per component in triangle order. A speck or a
        // void below a cell can mesh as a folded sliver enclosing a tiny
        // volume of either sign, so a component enclosing less than half a
        // cell cubed either way is classified by what surrounds it: a void
        // when most of 26 field samples around it (the corners, edge
        // midpoints and face centres of its bounding box grown by half a
        // cell) are inside material, a piece otherwise.
        const double h = stats.cell_size, small = 0.5 * h * h * h;
        std::vector<double> enclosed(vertex_count_, 0.0);
        for (size_t t = 0; t < triangle_count_; ++t) {
            const uint32_t first = surface_.triangles[3 * t];
            double q[3], facet[3];
            for (int a = 0; a < 3; ++a) q[a] = p[3 * size_t(first) + a] - reference[a];
            face_vector(surface_, t, facet);
            enclosed[root(first)] += dot(q, facet) / 6;
        }
        std::vector<uint32_t> slot(vertex_count_, UINT32_MAX);  // a small component's index, at its root
        std::vector<double> boxes;                               // lo[3], hi[3] per small component
        for (size_t v = 0; v < vertex_count_; ++v)
            if (root(static_cast<uint32_t>(v)) == v && std::fabs(enclosed[v]) < small) {
                slot[v] = static_cast<uint32_t>(boxes.size() / 6);
                boxes.insert(boxes.end(), {INFINITY, INFINITY, INFINITY, -INFINITY, -INFINITY, -INFINITY});
            }
        for (size_t v = 0; v < vertex_count_; ++v) {
            const uint32_t s = slot[root(static_cast<uint32_t>(v))];
            if (s == UINT32_MAX) continue;
            for (int a = 0; a < 3; ++a) {
                boxes[6 * size_t(s) + a] = std::min(boxes[6 * size_t(s) + a], p[3 * v + a]);
                boxes[6 * size_t(s) + 3 + a] = std::max(boxes[6 * size_t(s) + 3 + a], p[3 * v + a]);
            }
        }
        const size_t samples = 26 * (boxes.size() / 6);
        std::vector<double> sample(4 * samples);  // x, y, z and the field value, in blocks
        double *sx = sample.data(), *sy = sx + samples, *sz = sy + samples, *value = sz + samples;
        for (size_t i = 0; i < samples; ++i) {
            const size_t s = i / 26, k = i % 26;
            const int cell = static_cast<int>(k < 13 ? k : k + 1);  // of the 3 x 3 x 3 around the box, not its centre
            const int step[3] = {cell % 3, cell / 3 % 3, cell / 9};
            double *out[3] = {sx + i, sy + i, sz + i};
            for (int a = 0; a < 3; ++a) {
                const double lo = boxes[6 * s + a] - 0.5 * h, hi = boxes[6 * s + 3 + a] + 0.5 * h;
                *out[a] = step[a] == 0 ? lo : step[a] == 2 ? hi : 0.5 * (lo + hi);
            }
        }
        parallel_chunks(chunks_of(samples, kChunk), threads_, [&](size_t chunk) {
            const size_t begin = chunk * kChunk, end = std::min(samples, begin + kChunk);
            field_.evaluate(static_cast<int>(end - begin), sx + begin, sy + begin, sz + begin, value + begin);
        });
        for (size_t v = 0; v < vertex_count_; ++v) {
            if (root(static_cast<uint32_t>(v)) != v) continue;
            bool cavity = enclosed[v] <= -small;
            if (slot[v] != UINT32_MAX) {
                int inside = 0;
                for (size_t k = 0; k < 26; ++k) inside += value[26 * size_t(slot[v]) + k] < 0;
                cavity = inside > 13;
            }
            if (cavity) {
                ++stats.cavities;
                stats.cavity_volume += std::fabs(enclosed[v]);
            }
        }
    }

    const Surface &surface_;
    const MeshField &field_;
    const double eps_;
    const double cos_crease_, cos_half_crease_;
    const int threads_;
    const bool spatial_;
    const size_t vertex_count_, triangle_count_;
    std::vector<double> tri_normals_, vertex_normals_;
    std::vector<uint32_t> first_corner_, corners_;
    std::vector<uint8_t> crease_sides_;
};

std::unique_ptr<rsdf_mesh> build_mesh(const rsdf_graph &graph, uint32_t node, const rsdf_mesh_options *options) {
    const auto start = std::chrono::steady_clock::now();
    if (!options) throw Error("mesh options are null");
    const Node &n = checked_node(graph, node);
    if (n.dim != 3) throw Error(format("meshing needs a 3D node; node %u is 2D", node));
    if (n.field)
        throw Error(format("meshing needs a shape; node %u (%s) is a field (IMPLICIT makes it a shape)", node,
                           op_name(n.op)));
    const double h = options->cell_size;
    if (!(h > 0) || !std::isfinite(h)) throw Error("cell_size must be positive and finite");
    const double crease = std::isnan(options->crease_angle) || options->crease_angle <= 0 ? 35.0
                          : std::min(options->crease_angle, 180.0);

    MeshField field{&graph, node, {}, {}};
    const double *lo = options->bounds_valid ? options->bounds_lo : n.lo;
    const double *hi = options->bounds_valid ? options->bounds_hi : n.hi;
    bool empty = false;
    for (int a = 0; a < 3; ++a) {
        if (!std::isfinite(lo[a]) || !std::isfinite(hi[a])) {
            if (options->bounds_valid) throw Error("explicit meshing bounds must be finite");
            if (lo[a] > hi[a]) {  // the node's solid is empty
                empty = true;
                continue;
            }
            throw Error(format("node %u is unbounded along %c; pass explicit meshing bounds", node, "xyz"[a]));
        }
        if (lo[a] > hi[a]) {
            if (options->bounds_valid) throw Error("explicit meshing bounds must have lo <= hi");
            empty = true;
        }
        field.lo[a] = lo[a];
        field.hi[a] = hi[a];
    }

    if (!empty) {
        // Positions are returned in single precision, which rounds by up to
        // 2^-24 of a coordinate; beyond a sixteenth of a cell facets degenerate.
        double extent = 0;
        for (int a = 0; a < 3; ++a) extent = std::max({extent, std::fabs(lo[a]), std::fabs(hi[a])});
        extent += 4 * h;  // the grid's padding
        if (extent * 0x1p-24 > h / 16)
            throw Error(format("single-precision vertex positions cannot resolve cell size %g mm at coordinates up to "
                               "%g mm; use a cell size of at least %g mm or model closer to the origin",
                               h, extent, extent * 0x1p-20));
    }

    auto mesh = std::make_unique<rsdf_mesh>();
    std::memset(&mesh->stats, 0, sizeof mesh->stats);
    mesh->stats.cell_size = h;
    mesh->graph = graph.identity;
    mesh->node = node;
    std::copy(field.lo, field.lo + 3, mesh->box_lo);
    std::copy(field.hi, field.hi + 3, mesh->box_hi);
    mesh->crease_angle = crease;
    mesh->gradient_step = 1e-4 * h;
    mesh->spatial_order = n.sorts_queries;
    if (!empty) {
        MeshSettings settings;
        settings.cell_size = h;
        settings.lipschitz = std::max(n.step_lipschitz, 1.0);  // the box clip is 1-Lipschitz
        settings.gradient_step = 1e-4 * h;
        settings.max_cells = options->max_cells ? options->max_cells : kDefaultMaxCells;
        settings.threads = evaluation_threads(n, options->threads);
        settings.refine_steps = options->refine_steps > 0 ? std::min(options->refine_steps, 64) : 6;
        settings.spatial_order = n.sorts_queries;
        Surface surface = extract_surface(field, settings);
        refine_surface(field, surface, settings);
        std::vector<double> normals;
        if (options->flags & RSDF_MESH_PREVIEW_CREASES) {
            normals = gradients(field, surface.positions.size() / 3, settings.gradient_step, settings.threads,
                                settings.spatial_order, [&](size_t v, double *p) {
                                    std::copy_n(&surface.positions[3 * v], 3, p);
                                });
            if (n.has_mesh && (options->flags & RSDF_MESH_PREVIEW_FACETS))
                refine_mesh_facets(field, surface, settings, normals);
            align_surface_creases(field, surface, settings, normals);
        }
        Finisher(surface, field, settings.gradient_step, crease, settings.threads, settings.spatial_order)
            .run(*mesh, std::move(normals));
    }
    mesh->stats.seconds = std::chrono::duration<double>(std::chrono::steady_clock::now() - start).count();
    return mesh;
}

// A simplified copy of a finished mesh (simplify.cpp), finished again so its
// normals, creases and statistics describe the simplified surface.
std::unique_ptr<rsdf_mesh> simplify_mesh(const rsdf_graph &graph, uint32_t node, const rsdf_mesh *source,
                                         const rsdf_simplify_options *options) {
    const auto start = std::chrono::steady_clock::now();
    if (!source) throw Error("mesh is null");
    if (!options) throw Error("simplify options are null");
    const Node &n = checked_node(graph, node);
    if (source->graph != graph.identity) throw Error("the mesh was built from another graph");
    if (node != source->node || n.dim != 3)
        throw Error(format("the mesh was built from node %u, not node %u", source->node, node));
    const double tolerance = options->tolerance;
    if (!(tolerance > 0) || !std::isfinite(tolerance)) throw Error("tolerance must be positive and finite");

    auto mesh = std::make_unique<rsdf_mesh>();
    std::memset(&mesh->stats, 0, sizeof mesh->stats);
    mesh->stats.cell_size = source->stats.cell_size;
    mesh->graph = graph.identity;
    mesh->node = node;
    std::copy(source->box_lo, source->box_lo + 3, mesh->box_lo);
    std::copy(source->box_hi, source->box_hi + 3, mesh->box_hi);
    mesh->crease_angle = source->crease_angle;
    mesh->gradient_step = source->gradient_step;
    mesh->spatial_order = source->spatial_order;
    if (!source->triangles.empty()) {
        MeshField field{&graph, node, {}, {}};
        std::copy(source->box_lo, source->box_lo + 3, field.lo);
        std::copy(source->box_hi, source->box_hi + 3, field.hi);
        SimplifyInput input;
        const size_t base = source->base_vertices;
        input.surface.positions.assign(source->positions.begin(), source->positions.begin() + 3 * base);
        input.surface.triangles.resize(source->triangles.size());
        for (size_t i = 0; i < source->triangles.size(); ++i) {
            const uint32_t v = source->triangles[i];
            input.surface.triangles[i] = v < base ? v : source->split_origin[v - base];
        }
        input.creases = source->crease_edges;
        input.interior_normals = source->interior_normals;
        const int threads = evaluation_threads(n, options->threads);
        const SimplifySettings settings{tolerance, source->stats.cell_size, source->crease_angle, threads,
                                        source->spatial_order};
        Surface surface = simplify_surface(field, std::move(input), settings);
        surface.active_cells = source->stats.active_cells;
        Finisher(surface, field, source->gradient_step, source->crease_angle, threads, source->spatial_order)
            .run(*mesh);
    }
    mesh->stats.seconds = std::chrono::duration<double>(std::chrono::steady_clock::now() - start).count();
    return mesh;
}

}  // namespace
}  // namespace rsdf

using namespace rsdf;

extern "C" {

rsdf_mesh *rsdf_mesh_build(const rsdf_graph *graph, uint32_t node, const rsdf_mesh_options *options) {
    rsdf_mesh *mesh = nullptr;
    guarded([&] { mesh = build_mesh(checked_graph(graph), node, options).release(); });
    return mesh;
}

rsdf_mesh *rsdf_mesh_simplify(const rsdf_graph *graph, uint32_t node, const rsdf_mesh *mesh,
                              const rsdf_simplify_options *options) {
    rsdf_mesh *simplified = nullptr;
    guarded([&] { simplified = simplify_mesh(checked_graph(graph), node, mesh, options).release(); });
    return simplified;
}

void rsdf_mesh_free(rsdf_mesh *mesh) { delete mesh; }

void rsdf_mesh_get_stats(const rsdf_mesh *mesh, rsdf_mesh_stats *stats) {
    if (!stats) return;
    if (mesh) *stats = mesh->stats;
    else std::memset(stats, 0, sizeof *stats);
}

const float *rsdf_mesh_positions(const rsdf_mesh *mesh) { return mesh ? mesh->positions.data() : nullptr; }

const float *rsdf_mesh_normals(const rsdf_mesh *mesh) { return mesh ? mesh->normals.data() : nullptr; }

const uint32_t *rsdf_mesh_triangles(const rsdf_mesh *mesh) { return mesh ? mesh->triangles.data() : nullptr; }

const float *rsdf_mesh_creases(const rsdf_mesh *mesh) { return mesh ? mesh->creases.data() : nullptr; }

}  // extern "C"
