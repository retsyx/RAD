// Error reporting, threading and small C API entry points.
#include "kernel.h"

#include <dispatch/dispatch.h>
#include <unistd.h>

#include <algorithm>
#include <atomic>
#include <cmath>
#include <cstdarg>
#include <cstdio>
#include <exception>
#include <mutex>
#include <new>

namespace rsdf {
namespace {

thread_local std::string last_error;

struct ChunkJob {
    const std::function<void(size_t)> *body;
    size_t chunks;
    std::atomic<size_t> next{0};
    std::atomic<bool> failed{false};
    std::mutex mutex;
    std::exception_ptr error;
};

// One libdispatch worker: claims chunks until none are left or a chunk failed.
void run_chunks(void *context, size_t) {
    auto *job = static_cast<ChunkJob *>(context);
    while (!job->failed.load(std::memory_order_relaxed)) {
        size_t chunk = job->next.fetch_add(1, std::memory_order_relaxed);
        if (chunk >= job->chunks) return;
        try {
            (*job->body)(chunk);
        } catch (...) {
            std::lock_guard<std::mutex> lock(job->mutex);
            if (!job->error) job->error = std::current_exception();
            job->failed.store(true, std::memory_order_relaxed);
        }
    }
}

}  // namespace

std::string format(const char *fmt, ...) {
    char buffer[512];
    va_list args;
    va_start(args, fmt);
    vsnprintf(buffer, sizeof buffer, fmt, args);
    va_end(args);
    return buffer;
}

void set_error(const std::string &message) { last_error = message; }

void clear_error() { last_error.clear(); }

bool guarded(const std::function<void()> &body) {
    clear_error();
    try {
        body();
        return true;
    } catch (const Error &error) {
        set_error(error.what());
    } catch (const std::bad_alloc &) {
        set_error("out of memory");
    } catch (const std::exception &error) {
        set_error(std::string("internal error: ") + error.what());
    } catch (...) {
        set_error("internal error");
    }
    return false;
}

const rsdf_graph &checked_graph(const rsdf_graph *graph) {
    if (!graph) throw Error("graph is null");
    return *graph;
}

const Node &checked_node(const rsdf_graph &graph, uint32_t node) {
    if (node >= graph.nodes.size())
        throw Error(format("node %u does not exist (the graph has %zu nodes)", node, graph.nodes.size()));
    return graph.nodes[node];
}

int resolve_threads(int32_t requested) {
    long cores = sysconf(_SC_NPROCESSORS_ONLN);
    if (cores < 1) cores = 1;
    if (requested <= 0) return static_cast<int>(cores);
    return requested > 1024 ? 1024 : requested;
}

int evaluation_threads(const Node &node, int32_t requested) {
    constexpr double kBudget = 1 << 30;
    const double per_thread = (node.live_buffers + 16.0) * kBatch * sizeof(double) + (node.depth + 16.0) * 256;
    const int affordable = static_cast<int>(std::max(1.0, std::floor(kBudget / per_thread)));
    return std::min(resolve_threads(requested), affordable);
}

void parallel_chunks(size_t chunks, int threads, const std::function<void(size_t)> &body) {
    if (chunks == 0) return;
    size_t workers = static_cast<size_t>(threads < 1 ? 1 : threads);
    if (workers > chunks) workers = chunks;
    if (workers == 1) {
        for (size_t chunk = 0; chunk < chunks; ++chunk) body(chunk);
        return;
    }
    ChunkJob job;
    job.body = &body;
    job.chunks = chunks;
    dispatch_apply_f(workers, dispatch_get_global_queue(QOS_CLASS_USER_INITIATED, 0), &job, run_chunks);
    if (job.error) std::rethrow_exception(job.error);
}

void radix_sort(std::vector<uint64_t> &keys, std::vector<uint32_t> *values, int bits) {
    constexpr int kDigit = 11;
    const size_t n = keys.size();
    std::vector<uint64_t> key_buffer(n);
    std::vector<uint32_t> value_buffer(values ? n : 0);
    for (int shift = 0; shift < bits; shift += kDigit) {
        size_t count[(1 << kDigit) + 1] = {};
        for (size_t i = 0; i < n; ++i) ++count[((keys[i] >> shift) & ((1 << kDigit) - 1)) + 1];
        for (int d = 0; d < (1 << kDigit); ++d) count[d + 1] += count[d];
        for (size_t i = 0; i < n; ++i) {
            const size_t to = count[(keys[i] >> shift) & ((1 << kDigit) - 1)]++;
            key_buffer[to] = keys[i];
            if (values) value_buffer[to] = (*values)[i];
        }
        keys.swap(key_buffer);
        if (values) values->swap(value_buffer);
    }
}

void radix_sort_items(std::vector<uint64_t> &items, int shift, int bits, int threads) {
    constexpr int kDigit = 12;
    constexpr size_t kBuckets = size_t(1) << kDigit, kPart = 65536;
    const size_t n = items.size(), parts = (n + kPart - 1) / kPart;
    std::vector<uint64_t> buffer(n);
    std::vector<size_t> offsets(parts * kBuckets);
    for (int pass = shift; pass < shift + bits; pass += kDigit) {
        const int width = std::min(kDigit, shift + bits - pass);
        const uint64_t mask = (uint64_t(1) << width) - 1;
        parallel_chunks(parts, threads, [&](size_t part) {
            size_t *count = &offsets[part * kBuckets];
            std::fill(count, count + kBuckets, size_t(0));
            const size_t end = std::min(n, (part + 1) * kPart);
            for (size_t i = part * kPart; i < end; ++i) ++count[(items[i] >> pass) & mask];
        });
        // Bucket-major, part-minor offsets keep the sort stable.
        size_t total = 0;
        for (size_t bucket = 0; bucket <= mask; ++bucket)
            for (size_t part = 0; part < parts; ++part) {
                size_t &slot = offsets[part * kBuckets + bucket];
                const size_t start = total;
                total += slot;
                slot = start;
            }
        parallel_chunks(parts, threads, [&](size_t part) {
            size_t *next = &offsets[part * kBuckets];
            const size_t end = std::min(n, (part + 1) * kPart);
            for (size_t i = part * kPart; i < end; ++i) buffer[next[(items[i] >> pass) & mask]++] = items[i];
        });
        items.swap(buffer);
    }
}

}  // namespace rsdf

extern "C" {

uint32_t rsdf_abi_version(void) { return RSDF_ABI_VERSION; }

const char *rsdf_kernel_version(void) { return "1.2.1"; }

const char *rsdf_last_error(void) { return rsdf::last_error.c_str(); }

}  // extern "C"
