// Exercises GPU field programs through the C API for a sanitizer build of
// the kernel (test_gpu_program.GPUProgramSanitizerTests): every RSDF file
// given is loaded (failures are fine), then up to 24 bounded 3D nodes are
// compiled together on several threads (small bricks, tight limits, so
// refinement, mesh-only parts and the size limit all occur) and evaluated by
// the reference interpreter in both modes.
#include "radsdf.h"
#include "radsdf_gpu.h"

#include <cmath>
#include <cstdio>
#include <fstream>
#include <iterator>
#include <random>
#include <vector>

int main(int argc, char **argv) {
    std::mt19937_64 random(11);
    std::uniform_real_distribution<double> unit(-1.0, 1.0);
    int loaded = 0, programs = 0, mesh_only = 0;
    for (int file = 1; file < argc; ++file) {
        std::ifstream in(argv[file], std::ios::binary);
        std::vector<uint8_t> bytes((std::istreambuf_iterator<char>(in)), std::istreambuf_iterator<char>());
        rsdf_graph *graph = rsdf_graph_create(bytes.data(), bytes.size());
        if (!graph) continue;
        ++loaded;
        std::vector<uint32_t> nodes;
        double extent = 0;
        for (uint32_t node = 0; node < rsdf_graph_node_count(graph) && nodes.size() < 24; ++node) {
            double lo[3], hi[3];
            if (rsdf_node_dimension(graph, node) != 3 || rsdf_bounds(graph, node, lo, hi, nullptr) != 0) continue;
            bool usable = true;
            for (int a = 0; a < 3; ++a) {
                usable = usable && std::isfinite(lo[a]) && std::isfinite(hi[a]) && lo[a] <= hi[a] && hi[a] - lo[a] < 200;
                if (usable) extent = std::fmax(extent, hi[a] - lo[a]);
            }
            if (usable) nodes.push_back(node);
        }
        if (nodes.empty()) {
            rsdf_graph_free(graph);
            continue;
        }
        for (int variant = 0; variant < 2; ++variant) {
            rsdf_gpu_options options{};
            options.brick_size = std::fmax(extent, 1e-3) / (variant ? 4 : 10);
            options.margin = options.brick_size / 40;
            options.min_brick_size = options.brick_size / 4;
            options.max_tape_instructions = variant ? 24 : 0;
            options.max_words = variant ? 4000 : 0;
            options.refine_instructions = variant ? 4 : 0;
            options.threads = 3;
            rsdf_gpu_program *program = rsdf_gpu_program_build(graph, nodes.data(), uint32_t(nodes.size()), &options);
            if (!program) {
                std::fprintf(stderr, "build failed: %s\n", rsdf_last_error());
                return 1;
            }
            ++programs;
            uint64_t words = 0;
            if (!rsdf_gpu_program_words(program, &words) || words < 16) return 1;
            std::vector<double> points(3 * 400), values(400);
            for (double &p : points) p = 0.6 * extent * unit(random);
            for (uint32_t part = 0; part < nodes.size(); ++part) {
                rsdf_gpu_part_info info;
                if (rsdf_gpu_program_part(program, part, &info) != 0) return 1;
                if (info.status != 0) {
                    ++mesh_only;
                    continue;
                }
                rsdf_gpu_program_evaluate(program, part, points.data(), 400, 0, values.data());
                rsdf_gpu_program_evaluate(program, part, points.data(), 100, 1, values.data());
            }
            rsdf_gpu_program_free(program);
        }
        rsdf_graph_free(graph);
    }
    std::printf("loaded %d programs %d mesh-only parts %d\n", loaded, programs, mesh_only);
    return loaded > 0 && programs > 0 ? 0 : 1;
}
