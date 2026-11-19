/* RAD signed distance field kernel: GPU field programs.
 *
 * rsdf_gpu_program_build compiles 3D part nodes into one binary program that
 * a GPU interpreter evaluates per point (SDF-DESIGN.md, "GPU field
 * programs"): every part gets a sparse grid of bricks, refined into octants
 * where that shortens tapes, and each brick near the surface holds a
 * straight-line register tape pruned to the nodes that can change values
 * within the brick grown by a margin. The program is a sequence of
 * little-endian 32-bit words (field.bin); the same words are what the Metal
 * evaluator reads. Conventions follow radsdf.h: functions
 * returning int report 0 on success and a negative value on failure, and
 * rsdf_last_error() describes the failure.
 */
#ifndef RADSDF_GPU_H
#define RADSDF_GPU_H

#include "radsdf.h"

#ifdef __cplusplus
extern "C" {
#endif

typedef struct rsdf_gpu_program rsdf_gpu_program;

typedef struct {
    double brick_size;               /* top-level brick edge in model units (> 0) */
    double margin;                   /* tapes are exact within each brick grown by this much (>= 0) */
    double min_brick_size;           /* smallest refined brick (0 = brick_size / 8) */
    uint64_t max_words;              /* program size limit in 32-bit words (0 = 2^26, 256 MB) */
    uint32_t max_tape_instructions;  /* per brick tape (0 = 4096) */
    uint32_t refine_instructions;    /* longer tapes try their brick's octants (0 = 4) */
    int32_t threads;                 /* worker threads (0 = all cores) */
    int32_t reserved;
} rsdf_gpu_options;

typedef struct {
    int32_t status;                  /* 0: field program; 1: mesh only (limits exceeded, see message) */
    uint32_t grid[3];                /* bricks along x, y, z */
    uint64_t bricks;                 /* grid cells */
    uint64_t band_bricks;            /* cells with a tape (any level) */
    uint64_t tapes;                  /* distinct tapes */
    uint64_t words;                  /* words this part added (tapes, grid, meshes) */
    uint64_t instructions;           /* instructions summed over band bricks */
    uint32_t max_instructions;       /* largest tape */
    uint32_t value_registers;        /* largest register use of a tape */
    uint32_t point_registers;
    uint32_t levels;                 /* deepest refinement level used */
    uint64_t cells;                  /* grid cells including octant cells of refined bricks */
    double seconds;
    char message[256];               /* mesh only: why */
} rsdf_gpu_part_info;

/* Compiles `count` 3D part nodes (each with finite bounds) into one program.
 * A part that exceeds a limit (tape length, registers, nesting, grid size or
 * the program's size) becomes mesh only; that is not a failure. Returns NULL
 * for invalid arguments. Deterministic for any thread count. */
rsdf_gpu_program *rsdf_gpu_program_build(const rsdf_graph *graph, const uint32_t *nodes, uint32_t count,
                                         const rsdf_gpu_options *options);
void rsdf_gpu_program_free(rsdf_gpu_program *program);
/* The program words; *word_count receives their number. */
const uint32_t *rsdf_gpu_program_words(const rsdf_gpu_program *program, uint64_t *word_count);
int rsdf_gpu_program_part(const rsdf_gpu_program *program, uint32_t part, rsdf_gpu_part_info *info);

/* The float64 reference interpreter of the program's tapes, for tests (the
 * graph the program was built from must still be alive):
 * mode 0 evaluates as the GPU does (brick lookup, pruned tapes, far values
 * outside the band), mode 1 runs the part's unpruned tape everywhere. Values
 * are bitwise those of rsdf_evaluate inside the band (both modes) up to
 * exact ties. Points are x y z triples. Fails for a mesh-only part. */
int rsdf_gpu_program_evaluate(const rsdf_gpu_program *program, uint32_t part, const double *points, uint64_t count,
                              int32_t mode, double *values);

#ifdef __cplusplus
}
#endif

#endif
