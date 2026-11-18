// Sanitizer driver for mesh simplification (test_mesh_simplify.SimplifySanitizerTests):
// for every "path root cell" triple on the command line, meshes the RSDF
// graph's root with eight threads, simplifies it with eight threads and with
// one (the results must match), simplifies the result again, and checks that
// its mesh is refused against another graph decoded from the same bytes. Built with
// ThreadSanitizer, or with AddressSanitizer and UndefinedBehaviorSanitizer,
// any race or memory fault aborts the run or fails it with a report.
#include <cstdio>
#include <cstdlib>
#include <cstring>
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

static bool same_mesh(const rsdf_mesh *a, const rsdf_mesh *b) {
    rsdf_mesh_stats sa, sb;
    rsdf_mesh_get_stats(a, &sa);
    rsdf_mesh_get_stats(b, &sb);
    if (sa.vertex_count != sb.vertex_count || sa.triangle_count != sb.triangle_count) return false;
    return (sa.vertex_count == 0 ||
            std::memcmp(rsdf_mesh_positions(a), rsdf_mesh_positions(b), sa.vertex_count * 3 * sizeof(float)) == 0) &&
           (sa.triangle_count == 0 || std::memcmp(rsdf_mesh_triangles(a), rsdf_mesh_triangles(b),
                                                  sa.triangle_count * 3 * sizeof(uint32_t)) == 0);
}

int main(int argc, char **argv) {
    int failures = 0;
    for (int i = 1; i + 2 < argc; i += 3) {
        const std::vector<uint8_t> data = read_file(argv[i]);
        const uint32_t root = static_cast<uint32_t>(std::strtoul(argv[i + 1], nullptr, 10));
        rsdf_graph *graph = rsdf_graph_create(data.data(), data.size());
        if (!graph) {
            std::printf("%s: graph refused: %s\n", argv[i], rsdf_last_error());
            ++failures;
            continue;
        }
        rsdf_mesh_options options = {};
        options.cell_size = std::atof(argv[i + 2]);
        options.crease_angle = 35;
        options.threads = 8;
        rsdf_mesh *mesh = rsdf_mesh_build(graph, root, &options);
        rsdf_simplify_options many = {0.01, 8, 0}, one = {0.01, 1, 0};
        rsdf_mesh *simplified = mesh ? rsdf_mesh_simplify(graph, root, mesh, &many) : nullptr;
        rsdf_mesh *serial = mesh ? rsdf_mesh_simplify(graph, root, mesh, &one) : nullptr;
        rsdf_mesh *again = simplified ? rsdf_mesh_simplify(graph, root, simplified, &many) : nullptr;
        if (!mesh || !simplified || !serial || !again) {
            std::printf("%s: failed: %s\n", argv[i], rsdf_last_error());
            ++failures;
        } else if (!same_mesh(simplified, serial)) {
            std::printf("%s: eight threads and one thread differ\n", argv[i]);
            ++failures;
        }
        rsdf_graph *twin = rsdf_graph_create(data.data(), data.size());
        if (mesh && twin) {
            rsdf_mesh *foreign = rsdf_mesh_simplify(twin, root, mesh, &many);
            if (foreign || !std::strstr(rsdf_last_error(), "another graph")) {
                std::printf("%s: a mesh of another graph was not refused\n", argv[i]);
                ++failures;
            }
            rsdf_mesh_free(foreign);
        }
        rsdf_graph_free(twin);
        rsdf_mesh_stats stats = {};
        rsdf_mesh_get_stats(simplified, &stats);
        std::printf("%s: %llu triangles\n", argv[i], static_cast<unsigned long long>(stats.triangle_count));
        rsdf_mesh_free(again);
        rsdf_mesh_free(serial);
        rsdf_mesh_free(simplified);
        rsdf_mesh_free(mesh);
        rsdf_graph_free(graph);
    }
    return failures ? 1 : 0;
}
