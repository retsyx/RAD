// Exercises custom fields (traced expressions) through the C API for a
// sanitizer build of the kernel (test_sdf_fields_native.FieldSanitizerTests):
// every RSDF file given is loaded (failures are fine), then every node is
// evaluated, field roots and IMPLICIT nodes are analysed, and IMPLICIT nodes
// are meshed coarsely, on several threads.
#include "radsdf.h"

#include <cmath>
#include <cstdio>
#include <fstream>
#include <iterator>
#include <random>
#include <vector>

int main(int argc, char **argv) {
    std::mt19937_64 random(7);
    std::uniform_real_distribution<double> unit(-1.0, 1.0);
    int loaded = 0, refused = 0;
    for (int file = 1; file < argc; ++file) {
        std::ifstream in(argv[file], std::ios::binary);
        std::vector<uint8_t> bytes((std::istreambuf_iterator<char>(in)), std::istreambuf_iterator<char>());
        rsdf_graph *graph = rsdf_graph_create(bytes.data(), bytes.size());
        if (!graph) {
            ++refused;
            continue;
        }
        ++loaded;
        const uint32_t count = rsdf_graph_node_count(graph);
        std::vector<double> points(3 * 1500), values(1500), normals(3 * 1500);
        for (double &p : points) p = 25 * unit(random);
        for (uint32_t node = 0; node < count; ++node) {
            const int dimension = rsdf_node_dimension(graph, node);
            if (dimension == 3) {
                rsdf_evaluate(graph, node, points.data(), 1500, values.data(), 3);
                rsdf_gradient(graph, node, points.data(), 500, 0, normals.data(), 2);
            } else if (dimension == 2) {
                rsdf_evaluate(graph, node, points.data(), 1000, values.data(), 2);
            }
            rsdf_field_report report;
            const double lo[3] = {-6, -5, -4}, hi[3] = {5, 6, 7};
            if (rsdf_field_analyze(graph, node, nullptr, nullptr, 2000, &report) != 0)
                rsdf_field_analyze(graph, node, lo, hi, 2000, &report);
            double box_lo[3], box_hi[3], lipschitz;
            if (rsdf_bounds(graph, node, box_lo, box_hi, &lipschitz) == 0 && dimension == 3) {
                bool finite = true;
                double extent = 0;
                for (int a = 0; a < 3; ++a) {
                    finite = finite && std::isfinite(box_lo[a]) && std::isfinite(box_hi[a]) && box_lo[a] < box_hi[a];
                    extent = std::fmax(extent, box_hi[a] - box_lo[a]);
                }
                if (finite && extent < 1e4) {
                    rsdf_mesh_options options = {};
                    options.cell_size = extent / 24;
                    options.threads = 3;
                    if (rsdf_mesh *mesh = rsdf_mesh_build(graph, node, &options)) rsdf_mesh_free(mesh);
                }
            }
        }
        rsdf_graph_free(graph);
    }
    std::printf("loaded %d graphs, refused %d\n", loaded, refused);
    return 0;
}
