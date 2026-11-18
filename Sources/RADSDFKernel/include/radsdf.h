/* RAD signed distance field kernel: C API.
 *
 * A graph is an immutable, topologically ordered expression DAG decoded from
 * the binary "RSDF" format written by the Python bindings (Engine/radsdf.py).
 * Every function is thread-safe for concurrent use of the same graph.
 *
 * Functions returning int report 0 on success and a negative value on
 * failure; rsdf_last_error() then describes the failure for this thread.
 */
#ifndef RADSDF_H
#define RADSDF_H

#include <stddef.h>
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

#define RSDF_ABI_VERSION 2

typedef struct rsdf_graph rsdf_graph;
typedef struct rsdf_mesh rsdf_mesh;

typedef struct {
    double cell_size;       /* grid spacing in model units (mm), > 0 */
    double crease_angle;    /* degrees; sharper creases keep split normals (default 35) */
    uint64_t max_cells;     /* fail beyond this many narrow-band cells (0 = 16 million) */
    int32_t threads;        /* worker threads (0 = all cores) */
    int32_t refine_steps;   /* surface crossing refinement iterations (default 6) */
    double bounds_lo[3];    /* explicit meshing bounds, used when bounds_valid != 0 */
    double bounds_hi[3];
    int32_t bounds_valid;
    int32_t reserved;
} rsdf_mesh_options;

typedef struct {
    uint64_t vertex_count;
    uint64_t triangle_count;
    uint64_t crease_segment_count;
    uint64_t active_cells;
    uint64_t boundary_edges;     /* edges with a single adjacent triangle */
    uint64_t nonmanifold_edges;  /* edges shared by more than two triangles */
    uint64_t components;         /* connected surface components */
    uint64_t cavities;           /* sealed voids: components enclosing at least half a cell cubed of negative
                                    volume, or less volume of either sign and surrounded by material */
    double volume;
    double area;
    double cavity_volume;        /* volume enclosed by the cavity surfaces (islands inside them included) */
    double centroid[3];
    double bounds_lo[3];
    double bounds_hi[3];
    double cell_size;
    double seconds;
} rsdf_mesh_stats;

uint32_t rsdf_abi_version(void);
const char *rsdf_kernel_version(void);
const char *rsdf_last_error(void);

rsdf_graph *rsdf_graph_create(const uint8_t *bytes, size_t length);
void rsdf_graph_free(rsdf_graph *graph);
uint32_t rsdf_graph_node_count(const rsdf_graph *graph);
int rsdf_node_dimension(const rsdf_graph *graph, uint32_t node);
/* Conservative bounds (may be infinite) and a Lipschitz bound of the field.
 * lo and hi receive one value per node dimension; an empty solid has lo > hi.
 * Any output pointer may be NULL. */
int rsdf_bounds(const rsdf_graph *graph, uint32_t node, double *lo, double *hi, double *lipschitz);

/* points: count * dimension doubles (x, y[, z]) for 2D or 3D nodes; every
 * coordinate must be finite with magnitude at most 1e150. */
int rsdf_evaluate(const rsdf_graph *graph, uint32_t node, const double *points, uint64_t count,
                  double *distances, int32_t threads);
/* Unit gradients (outward normals) of a 3D node; eps <= 0 picks a default
 * (1e-5). A vanishing gradient reports (0, 0, 1). */
int rsdf_gradient(const rsdf_graph *graph, uint32_t node, const double *points, uint64_t count,
                  double eps, double *normals, int32_t threads);
/* Sphere tracing. hits[i] is 1 when the ray met the surface within max_distance.
 * Directions need not be unit length; distances are measured along the
 * normalised direction. A ray starting inside reports its exit point; a miss
 * reports the traced distance capped at max_distance. tolerance <= 0 picks
 * 1e-4 and max_steps <= 0 picks 512. */
int rsdf_raycast(const rsdf_graph *graph, uint32_t node, const double *origins, const double *directions,
                 uint64_t count, double max_distance, double tolerance, int32_t max_steps,
                 uint8_t *hits, double *distances, int32_t threads);

/* Watertight surface extraction (sharp-feature dual contouring) of the node
 * intersected with the meshing bounds. Returns NULL on failure; an empty solid
 * gives an empty mesh. Array accessors may return NULL when their count is 0. */
rsdf_mesh *rsdf_mesh_build(const rsdf_graph *graph, uint32_t node, const rsdf_mesh_options *options);
void rsdf_mesh_free(rsdf_mesh *mesh);
void rsdf_mesh_get_stats(const rsdf_mesh *mesh, rsdf_mesh_stats *stats);
const float *rsdf_mesh_positions(const rsdf_mesh *mesh);    /* vertex_count * 3 */
const float *rsdf_mesh_normals(const rsdf_mesh *mesh);      /* vertex_count * 3, unit */
const uint32_t *rsdf_mesh_triangles(const rsdf_mesh *mesh); /* triangle_count * 3, outward CCW */
const float *rsdf_mesh_creases(const rsdf_mesh *mesh);      /* crease_segment_count * 6 */

typedef struct {
    double tolerance;   /* largest |f| on the simplified surface, in model units (> 0) */
    int32_t threads;    /* worker threads (0 = all cores) */
    int32_t reserved;
} rsdf_simplify_options;

/* Feature-preserving simplification of a mesh that rsdf_mesh_build or
 * rsdf_mesh_simplify made from this graph and node (a mesh of another graph
 * or node is an error): edge collapses in order of quadric error that keep
 * |f| <= tolerance at the moved vertices and within a margin of it at
 * samples of every changed triangle (between samples |f| can exceed it a
 * little), keep removed vertices and crease points within the tolerance,
 * keep crease edges (crease vertices move only along their creases; corners
 * stay), never flip or degenerate triangles, never fold an edge (turn its
 * triangles by more than 120 degrees where the triangles they replace
 * turned less) and keep closed manifold surfaces closed and manifold (link
 * condition). Deterministic for any thread count. Returns a new mesh with
 * normals, creases and statistics of the simplified surface (free both
 * meshes), or NULL on failure. */
rsdf_mesh *rsdf_mesh_simplify(const rsdf_graph *graph, uint32_t node, const rsdf_mesh *mesh,
                              const rsdf_simplify_options *options);

/* Custom fields. A field (a traced expression, opcodes 200-223) becomes a
 * solid through an IMPLICIT node, whose load proves an upper bound L of
 * |grad f| over its bounds. rsdf_field_analyze repeats that proof for an
 * IMPLICIT node's field or a field node that no other field reads (a field
 * root) over the box [lo, hi] (both NULL: an IMPLICIT node's bounds grown by
 * 1e-6 of their largest extent, as at load), with up to max_boxes interval
 * evaluations (0 = the default budget: at most 32768, fewer for large
 * fields), and, when a bound exists, samples |grad f| at a fixed
 * low-discrepancy set of points of the box. Returns 0 when the report was
 * written, whether or not a finite bound exists (report->ok), and a negative
 * value for invalid arguments. */
typedef struct {
    double bound;            /* proven upper bound of |grad f| over the box; INFINITY when none exists */
    double value_lo;         /* proven range of f over the box */
    double value_hi;
    double gradient_min;     /* |grad f| at the sample points (finite samples only) */
    double gradient_median;
    double gradient_max;
    double quality;          /* gradient_median / bound: 1 for a true distance; 0 without a bound */
    uint64_t boxes;          /* boxes evaluated */
    uint64_t samples;        /* sample points with a finite gradient */
    int32_t ok;              /* 1 when a finite bound was proven */
    uint32_t failed_node;    /* without a bound: the node whose slope is unbounded */
    double failed_lo[3];     /* without a bound: the region where it fails */
    double failed_hi[3];
    char message[512];       /* without a bound: what failed, naming the node and region; else "" */
} rsdf_field_report;

int rsdf_field_analyze(const rsdf_graph *graph, uint32_t node, const double *lo, const double *hi,
                       uint32_t max_boxes, rsdf_field_report *report);

#ifdef __cplusplus
}
#endif

#endif
