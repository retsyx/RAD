// Internal interface between surface extraction (mesh_extract.cpp), surface
// projection and measurement (refine.cpp) and mesh finishing (mesh.cpp).
#pragma once

#include "kernel.h"

#include <array>

namespace rsdf {

// The field being meshed: the node intersected with the meshing box, so the
// mesh stays closed even when explicit bounds cut the shape. For conservative
// node bounds the intersection changes nothing.
struct MeshField {
    const rsdf_graph *graph;
    uint32_t node;
    double lo[3], hi[3];

    void evaluate(int n, const double *x, const double *y, const double *z, double *out) const;
    BatchField batch() const;
};

struct MeshSettings {
    double cell_size;     // base cell (octree level 0)
    double lipschitz;     // stepping Lipschitz bound of the meshed field (>= 1)
    double gradient_step; // finite-difference step for normals
    double tolerance;     // largest estimated deviation refinement aims for (model units)
    double crease_angle;  // degrees; display creases (mesh.cpp) away from sharp features are refined away
    uint64_t max_cells;   // narrow-band base cells (fails beyond) and octree leaves (refinement stops)
    uint64_t max_triangles;  // refinement stops before its estimate exceeds this (0: no limit)
    int max_depth;        // octree levels below cell_size (0: a uniform grid)
    int threads;
    int refine_steps;
    bool spatial_order;   // evaluate crossings and normals in Morton order (costly fields, see Node::sorts_queries)
};

// Vertex marks from extraction.
enum : uint8_t {
    kSharpVertex = 1,   // a QEF point on a sharp edge or corner
    kSpreadVertex = 2,  // not sharp, but its cell's crossing normals spread (a curved or blended region)
    kJumpVertex = 4,    // a crossing of its cell is a discontinuity of the field (never refined)
};

// Dual contouring output before shading: one vertex per surface component
// of each active octree leaf.
struct Surface {
    std::vector<double> positions;    // 3 per vertex
    std::vector<uint32_t> triangles;  // 3 per triangle, outward counter-clockwise
    std::vector<uint8_t> marks;       // 1 per vertex from extraction (kSharpVertex, ...); empty for others
    // 3 per triangle: the unit field normal at its centroid, and 3 per
    // vertex: the unit field normal there, as measured by extraction; empty
    // when the finisher computes them.
    std::vector<double> interior_normals, vertex_normals;
    uint64_t active_cells = 0;
    // Refinement report (extraction only).
    double max_deviation = 0;        // largest sampled estimated deviation (|f| / |grad f|)
    uint64_t flagged_triangles = 0;  // triangles still outside the tolerance or the normal angle
    uint64_t refined_cells = 0;      // leaves split by the criteria
    uint64_t balanced_cells = 0;     // leaves split to keep the octree balanced
    uint32_t max_level = 0;          // deepest leaf level
    uint32_t passes = 0;             // extraction passes
    bool budget_limited = false;     // the triangle budget or max_cells stopped refinement
};

inline uint64_t surface_edge_key(uint32_t a, uint32_t b) {
    return a < b ? uint64_t(a) << 32 | b : uint64_t(b) << 32 | a;
}

Surface extract_surface(const MeshField &field, const MeshSettings &settings);

// Fixed-size chunking helper: chunk count for `count` items of `size`.
inline size_t chunks_of(size_t count, size_t size) { return (count + size - 1) / size; }

// Normalised field gradients at `count` points, the i-th written by
// point(i, xyz); zero where the gradient vanishes. Costly fields visit the
// points in Morton order (vertices and triangles come in grid-key order, whose
// batches are long strips that defeat culling).
template <class PointAt>
std::vector<double> gradients(const MeshField &field, size_t count, double eps, int threads, bool spatial,
                              const PointAt &point, std::vector<double> *magnitudes = nullptr) {
    if (magnitudes) magnitudes->assign(count, 0.0);
    constexpr size_t kChunk = 4096;
    std::vector<double> normals(3 * count);
    const BatchField batch = field.batch();
    std::vector<uint32_t> order;
    if (spatial) {
        order = morton_order(count, threads, point);
    } else {
        order.resize(count);
        for (size_t i = 0; i < count; ++i) order[i] = static_cast<uint32_t>(i);
    }
    constexpr int kPoints = kBatch / 4;
    parallel_chunks(chunks_of(count, kChunk), threads, [&](size_t chunk) {
        double x[kPoints], y[kPoints], z[kPoints], nx[kPoints], ny[kPoints], nz[kPoints], m[kPoints];
        const size_t end = std::min(count, (chunk + 1) * kChunk);
        for (size_t first = chunk * kChunk; first < end; first += kPoints) {
            const int n = static_cast<int>(std::min<size_t>(kPoints, end - first));
            for (int i = 0; i < n; ++i) {
                double xyz[3];
                point(order[first + i], xyz);
                x[i] = xyz[0];
                y[i] = xyz[1];
                z[i] = xyz[2];
            }
            batch_gradients(batch, n, x, y, z, eps, nx, ny, nz, magnitudes ? m : nullptr);
            for (int i = 0; i < n; ++i) {
                double *normal = &normals[3 * size_t(order[first + i])];
                normal[0] = nx[i];
                normal[1] = ny[i];
                normal[2] = nz[i];
                if (magnitudes) (*magnitudes)[order[first + i]] = m[i];
            }
        }
    });
    return normals;
}

// Moves points onto the surface by Newton steps along the field gradient
// (SDF-DESIGN.md, "Meshing" step 5). Point i stays within its box (6 i: lo,
// then hi), steps only while the gradient faces guides[3 i ...] (so it never
// crosses to the other side of a thin wall), keeps a step only if it reduces
// |f| (halving one that does not), and stops once |f| <= settled. Points with
// fixed[i] != 0 stay; `fixed` may be empty.
void project_points(const MeshField &field, const MeshSettings &settings, double settled,
                    std::vector<double> &positions, const std::vector<double> &guides,
                    const std::vector<double> &boxes, const std::vector<uint8_t> &fixed);

// Cosines of the angle between the gradients a distance sides[i] either side
// of each point, along directions[3 i ...] (any length; a zero direction
// gives 1). Near 1 where the field is smooth; on a sharp edge between the
// two samples, the cosine of the angle between its faces.
std::vector<double> side_cosines(const MeshField &field, const MeshSettings &settings,
                                 const std::vector<double> &points, const std::vector<double> &directions,
                                 const std::vector<double> &sides);

// Field values at points (3 per point), visited in Morton order when `spatial`.
std::vector<double> field_values(const MeshField &field, int threads, bool spatial, const std::vector<double> &points);

// How far a triangle strays from the surface (SDF-DESIGN.md, "Meshing").
struct TriangleResult {
    double deviation;  // largest estimated distance |f| / |grad f| at its samples
    double cosine;     // its facet normal . the unit field normal at its centroid
    double normal[3];  // the unit field normal at its centroid (zero where the gradient vanishes)
};

// Measures triangles `which` (indices into triangles, 3 vertex indices each):
// the largest |f| at their corners (corner_values, per vertex), edge
// midpoints, centroid and three interior points (barycentric 2/3, 1/6, 1/6
// and permutations), or the peak of the quadratic through the corner and
// midpoint values plus its largest miss at the interior points if larger,
// divided by the smallest |grad f| at the centroid and the corners
// (corner_slopes, per vertex; at least 1/4); and the field normal at the
// centroid (tetrahedral differences, step eps). Writes results[t].
void measure_triangles(const MeshField &field, int threads, bool spatial, double eps, const double *positions,
                       const uint32_t *triangles, const double *corner_values, const double *corner_slopes,
                       const std::vector<uint32_t> &which, std::vector<TriangleResult> &results);

// What a finished mesh keeps for simplification (simplify.cpp): its surface
// before vertex splitting, its crease edges and the interior shading normal
// of every triangle.
struct SimplifyInput {
    Surface surface;
    std::vector<uint32_t> creases;        // pairs of surface vertex indices
    std::vector<float> interior_normals;  // 3 per triangle, unit
};

struct SimplifySettings {
    double tolerance;     // largest |f| on the simplified surface (model units)
    double cell_size;     // the mesh's grid spacing; field samples are two cells apart
    double crease_angle;  // degrees; a crease turning more at a vertex makes it a corner
    int threads;
    bool spatial_order;   // validate in Morton order (costly fields)
};

// Feature-preserving edge-collapse simplification (SDF-DESIGN.md, "Mesh
// simplification"). Deterministic for any thread count.
Surface simplify_surface(const MeshField &field, SimplifyInput input, const SimplifySettings &settings);

}  // namespace rsdf
