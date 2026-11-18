// Sanitizer driver for the MESH decoder (test_mesh_import.MeshSanitizerFuzzTests):
// decodes every RSDF file named on the command line and, when node 0 loads,
// queries it like the app does: bounds, values near and far from the mesh,
// gradients, rays and a coarse mesh. Built with AddressSanitizer and
// UndefinedBehaviorSanitizer, any memory or arithmetic fault aborts the run.
#include <cmath>
#include <cstdio>
#include <cstring>
#include <random>
#include <vector>

#include "radsdf.h"

static std::vector<uint8_t> read_file(const char *path) {
    std::vector<uint8_t> data;
    FILE *file = std::fopen(path, "rb");
    if (!file) return data;
    std::fseek(file, 0, SEEK_END);
    const long size = std::ftell(file);
    std::fseek(file, 0, SEEK_SET);
    data.resize(size > 0 ? size_t(size) : 0);
    if (!data.empty() && std::fread(data.data(), 1, data.size(), file) != data.size()) data.clear();
    std::fclose(file);
    return data;
}

int main(int argc, char **argv) {
    int loaded = 0, rejected = 0, failures = 0;
    for (int i = 1; i < argc; ++i) {
        const std::vector<uint8_t> data = read_file(argv[i]);
        rsdf_graph *graph = rsdf_graph_create(data.data(), data.size());
        if (!graph) {
            ++rejected;
            continue;
        }
        ++loaded;
        double lo[3], hi[3], lipschitz = 0;
        rsdf_bounds(graph, 0, lo, hi, &lipschitz);
        double span = 0;
        for (int a = 0; a < 3; ++a) span = std::fmax(span, hi[a] - lo[a]);
        if (!(std::isfinite(span) && span > 0)) span = 1;
        std::mt19937_64 random(static_cast<uint64_t>(i));
        std::uniform_real_distribution<double> uniform(-0.5, 1.5);
        std::normal_distribution<double> normal;
        const int count = 2000;
        std::vector<double> points(3 * count), values(count), gradients(3 * count);
        for (int k = 0; k < count; ++k) {
            const double far = k % 10 == 0 ? std::pow(10.0, 3 + k % 13) : 0;  // up to 1e15 spans away
            for (int a = 0; a < 3; ++a) {
                const double extent = hi[a] - lo[a] > 0 ? hi[a] - lo[a] : span;
                points[3 * k + a] = lo[a] + uniform(random) * extent + far * span * normal(random);
            }
        }
        if (rsdf_evaluate(graph, 0, points.data(), count, values.data(), 0) != 0) {
            std::printf("%s: evaluate failed: %s\n", argv[i], rsdf_last_error());
            ++failures;
        }
        for (int k = 0; k < count; ++k) {
            bool outside = false;
            for (int a = 0; a < 3; ++a) outside = outside || points[3 * k + a] < lo[a] || points[3 * k + a] > hi[a];
            if (!std::isfinite(values[k]) || (outside && !(values[k] > 0))) {
                std::printf("%s: value %g at point %d (outside the box: %d)\n", argv[i], values[k], k, outside);
                ++failures;
                break;
            }
        }
        rsdf_gradient(graph, 0, points.data(), 200, 0.0, gradients.data(), 0);
        std::vector<double> directions(300), distances(100);
        std::vector<uint8_t> hits(100);
        for (double &value : directions) value = normal(random);
        rsdf_raycast(graph, 0, points.data(), directions.data(), 100, 4 * span + 1, 0, 0, hits.data(),
                     distances.data(), 0);
        rsdf_mesh_options options;
        std::memset(&options, 0, sizeof options);
        options.cell_size = span / 16;
        options.max_cells = 2000000;
        rsdf_mesh *mesh = rsdf_mesh_build(graph, 0, &options);
        if (mesh) rsdf_mesh_free(mesh);
        rsdf_graph_free(graph);
    }
    std::printf("loaded %d, rejected %d, failures %d\n", loaded, rejected, failures);
    return failures ? 1 : 0;
}
