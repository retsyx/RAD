// Internal interface between surface extraction (mesh_extract.cpp), surface
// projection and refinement (refine.cpp) and mesh finishing (mesh.cpp).
#pragma once

#include "kernel.h"

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
    double cell_size;
    double lipschitz;     // stepping Lipschitz bound of the meshed field (>= 1)
    double gradient_step; // finite-difference step for normals
    uint64_t max_cells;
    int threads;
    int refine_steps;
    bool spatial_order;   // evaluate crossings and normals in Morton order (costly fields, see Node::sorts_queries)
};

// Vertex marks from extraction (Surface::marks).
enum : uint8_t {
    kSharpVertex = 1,   // a QEF point on a sharp edge or corner
    kSpreadVertex = 2,  // not sharp, but its cell's crossing normals spread (a curved or blended region)
};

// Dual contouring output before shading: one vertex per referenced active
// cell, plus the vertices refinement adds.
struct Surface {
    std::vector<double> positions;    // 3 per vertex
    std::vector<uint32_t> triangles;  // 3 per triangle, outward counter-clockwise
    std::vector<uint8_t> marks;       // 1 per vertex from extraction; empty for other surfaces
    uint64_t active_cells = 0;
};

Surface extract_surface(const MeshField &field, const MeshSettings &settings);

// Fixed-size chunking helper: chunk count for `count` items of `size`.
inline size_t chunks_of(size_t count, size_t size) { return (count + size - 1) / size; }

// Normalised field gradients at `count` points, the i-th written by
// point(i, xyz); zero where the gradient vanishes. Costly fields visit the
// points in Morton order (vertices and triangles come in grid-key order, whose
// batches are long strips that defeat culling).
template <class PointAt>
std::vector<double> gradients(const MeshField &field, size_t count, double eps, int threads, bool spatial,
                              const PointAt &point) {
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
        double x[kPoints], y[kPoints], z[kPoints], nx[kPoints], ny[kPoints], nz[kPoints];
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
            batch_gradients(batch, n, x, y, z, eps, nx, ny, nz);
            for (int i = 0; i < n; ++i) {
                double *normal = &normals[3 * size_t(order[first + i])];
                normal[0] = nx[i];
                normal[1] = ny[i];
                normal[2] = nz[i];
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

// Cosines of the angle between the gradients a distance `side` either side of
// each point, along directions[3 i ...] (any length; a zero direction gives
// 1). Near 1 where the field is smooth; on a sharp edge between the two
// samples, the cosine of the angle between its faces.
std::vector<double> side_cosines(const MeshField &field, const MeshSettings &settings,
                                 const std::vector<double> &points, const std::vector<double> &directions,
                                 double side);

// Refines the surface where a smooth feature is narrower than a cell (step 7):
// splits edges along which the surface turns by more than 20 degrees, away
// from sharp features, with new vertices on the surface. Needs marks.
void refine_surface(const MeshField &field, Surface &surface, const MeshSettings &settings);

// What a finished mesh keeps for simplification (simplify.cpp): its surface
// before vertex splitting, its crease edges and the interior normal (field
// gradient at the centroid) of every triangle.
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
