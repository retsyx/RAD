// Internal interface between surface extraction (mesh_extract.cpp) and mesh
// finishing (mesh.cpp).
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

// Dual contouring output before shading: one vertex per referenced active cell.
struct Surface {
    std::vector<double> positions;    // 3 per vertex
    std::vector<uint32_t> triangles;  // 3 per triangle, outward counter-clockwise
    uint64_t active_cells = 0;
};

Surface extract_surface(const MeshField &field, const MeshSettings &settings);

// Fixed-size chunking helper: chunk count for `count` items of `size`.
inline size_t chunks_of(size_t count, size_t size) { return (count + size - 1) / size; }

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
