// Point queries: distances, gradients and ray casts.
#include "kernel.h"

#include <algorithm>

namespace rsdf {
namespace {

constexpr double kMaxCoordinate = 1e150;  // keeps squared distances finite
constexpr uint64_t kChunkPoints = 16 * kBatch;
constexpr uint64_t kChunkRays = 4 * kBatch;
constexpr int kBisections = 60;

void require_buffers(bool ok) {
    if (!ok) throw Error("input and output buffers must not be null");
}

void check_coordinate(double value, const char *what, uint64_t index) {
    if (!(std::fabs(value) <= kMaxCoordinate))
        throw Error(format("%s %llu has a coordinate that is not finite or exceeds 1e150 in magnitude", what,
                           static_cast<unsigned long long>(index)));
}

size_t chunk_count(uint64_t count, uint64_t chunk) {
    if (count > (uint64_t(1) << 56)) throw Error("too many points");
    return static_cast<size_t>((count + chunk - 1) / chunk);
}

const Node &checked_3d(const rsdf_graph &graph, uint32_t node, const char *what) {
    const Node &n = checked_node(graph, node);
    if (n.dim != 3) throw Error(format("%s needs a 3D node; node %u is 2D", what, node));
    return n;
}

// Fields evaluate only from their root (read by no field) or IMPLICIT, whose
// compiled tapes run each node once per batch.
const Node &checked_queryable(const rsdf_graph &graph, uint32_t node) {
    const Node &n = checked_node(graph, node);
    if (n.field && !n.tape)
        throw Error(format("node %u (%s) is a field read by other fields; query its root field or its IMPLICIT node",
                           node, op_name(n.op)));
    return n;
}

// Requests of at least this many points are visited in Morton order, so
// batches of scattered points still stay compact (outputs keep their order).
constexpr uint64_t kSortMinimum = 4 * kBatch;

// The visiting order of a request (empty: the given order).
template <class PointAt>
std::vector<uint32_t> visiting_order(const Node &node, uint64_t count, int threads, const PointAt &point) {
    if (!node.sorts_queries || count < kSortMinimum || count > UINT32_MAX) return {};
    return morton_order(static_cast<size_t>(count), threads, point);
}

inline uint64_t visit(const std::vector<uint32_t> &order, uint64_t position) {
    return order.empty() ? position : order[position];
}

}  // namespace
}  // namespace rsdf

using namespace rsdf;

extern "C" {

int rsdf_evaluate(const rsdf_graph *graph, uint32_t node, const double *points, uint64_t count, double *distances,
                  int32_t threads) {
    return guarded([&] {
               const rsdf_graph &g = checked_graph(graph);
               const Node &target = checked_queryable(g, node);
               const int dim = static_cast<int>(target.dim);
               if (count == 0) return;
               require_buffers(points && distances);
               const int workers = evaluation_threads(target, threads);
               const std::vector<uint32_t> order = visiting_order(target, count, workers, [&](size_t i, double *p) {
                   for (int a = 0; a < 3; ++a) p[a] = a < dim ? points[i * dim + a] : 0.0;
               });
               parallel_chunks(chunk_count(count, kChunkPoints), workers, [&](size_t chunk) {
                   double x[kBatch], y[kBatch], z[kBatch], out[kBatch];
                   const uint64_t end = std::min<uint64_t>(count, (chunk + 1) * kChunkPoints);
                   for (uint64_t first = chunk * kChunkPoints; first < end; first += kBatch) {
                       const int n = static_cast<int>(std::min<uint64_t>(kBatch, end - first));
                       for (int i = 0; i < n; ++i) {
                           const uint64_t index = visit(order, first + i);
                           const double *point = points + index * dim;
                           for (int a = 0; a < dim; ++a) check_coordinate(point[a], "point", index);
                           x[i] = point[0];
                           y[i] = point[1];
                           z[i] = dim == 3 ? point[2] : 0.0;
                       }
                       eval_batch(g, node, n, x, y, z, out);
                       for (int i = 0; i < n; ++i) distances[visit(order, first + i)] = out[i];
                   }
               });
           })
               ? 0
               : -1;
}

int rsdf_gradient(const rsdf_graph *graph, uint32_t node, const double *points, uint64_t count, double eps,
                  double *normals, int32_t threads) {
    return guarded([&] {
               const rsdf_graph &g = checked_graph(graph);
               const Node &target = checked_3d(g, node, "gradient");
               checked_queryable(g, node);
               if (!std::isfinite(eps)) throw Error("gradient step eps must be finite");
               const double step = eps > 0 ? eps : 1e-5;
               if (count == 0) return;
               require_buffers(points && normals);
               const BatchField field = [&](int n, const double *x, const double *y, const double *z, double *out) {
                   eval_batch(g, node, n, x, y, z, out);
               };
               constexpr int kPoints = kBatch / 4;
               const int workers = evaluation_threads(target, threads);
               const std::vector<uint32_t> order = visiting_order(target, count, workers, [&](size_t i, double *p) {
                   std::copy(points + 3 * i, points + 3 * i + 3, p);
               });
               parallel_chunks(chunk_count(count, kChunkPoints), workers, [&](size_t chunk) {
                   double x[kPoints], y[kPoints], z[kPoints], nx[kPoints], ny[kPoints], nz[kPoints];
                   const uint64_t end = std::min<uint64_t>(count, (chunk + 1) * kChunkPoints);
                   for (uint64_t first = chunk * kChunkPoints; first < end; first += kPoints) {
                       const int n = static_cast<int>(std::min<uint64_t>(kPoints, end - first));
                       for (int i = 0; i < n; ++i) {
                           const uint64_t index = visit(order, first + i);
                           const double *point = points + index * 3;
                           for (int a = 0; a < 3; ++a) check_coordinate(point[a], "point", index);
                           x[i] = point[0];
                           y[i] = point[1];
                           z[i] = point[2];
                       }
                       batch_gradients(field, n, x, y, z, step, nx, ny, nz);
                       for (int i = 0; i < n; ++i) {  // a vanishing gradient reports +Z
                           double *normal = normals + visit(order, first + i) * 3;
                           const bool flat = nx[i] == 0 && ny[i] == 0 && nz[i] == 0;
                           normal[0] = nx[i];
                           normal[1] = ny[i];
                           normal[2] = flat ? 1.0 : nz[i];
                       }
                   }
               });
           })
               ? 0
               : -1;
}

// Sphere tracing: each ray advances by |f| / L (L the node's stepping
// Lipschitz bound) until |f| < tolerance. A sign change between two samples
// without a tolerance hit is resolved by bisection, so rays starting inside
// report the exit point and overshoots never skip a surface. Rays are visited
// in the Morton order of the points where they enter the node's box.
int rsdf_raycast(const rsdf_graph *graph, uint32_t node, const double *origins, const double *directions,
                 uint64_t count, double max_distance, double tolerance, int32_t max_steps, uint8_t *hits,
                 double *distances, int32_t threads) {
    return guarded([&] {
               const rsdf_graph &g = checked_graph(graph);
               const Node &n = checked_3d(g, node, "raycast");
               if (n.field)
                   throw Error(format("raycast needs a shape; node %u (%s) is a field, which has no slope bound "
                                      "(IMPLICIT makes it a shape)", node, op_name(n.op)));
               if (!(max_distance > 0)) throw Error("raycast max_distance must be positive");
               if (std::isnan(tolerance)) throw Error("raycast tolerance must not be NaN");
               const double tol = tolerance > 0 ? tolerance : 1e-4;
               const int64_t steps = max_steps > 0 ? max_steps : 512;
               const double inv_lipschitz = 1 / n.step_lipschitz;
               if (count == 0) return;
               require_buffers(origins && directions && hits && distances);
               const int workers = evaluation_threads(n, threads);
               const std::vector<uint32_t> order = visiting_order(n, count, workers, [&](size_t ray, double *p) {
                   const double *o = origins + 3 * ray, *d = directions + 3 * ray;
                   double enter = 0, leave = INFINITY;
                   for (int a = 0; a < 3; ++a) {
                       if (d[a] == 0) {
                           if (!(o[a] >= n.lo[a] && o[a] <= n.hi[a])) leave = -INFINITY;
                           continue;
                       }
                       const double t1 = (n.lo[a] - o[a]) / d[a], t2 = (n.hi[a] - o[a]) / d[a];
                       enter = std::fmax(enter, std::fmin(t1, t2));
                       leave = std::fmin(leave, std::fmax(t1, t2));
                   }
                   const double along = enter <= leave && std::isfinite(enter) ? enter : 0.0;
                   for (int a = 0; a < 3; ++a) p[a] = o[a] + along * d[a];
               });
               parallel_chunks(chunk_count(count, kChunkRays), workers, [&](size_t chunk) {
                   const uint64_t end = std::min<uint64_t>(count, (chunk + 1) * kChunkRays);
                   for (uint64_t first = chunk * kChunkRays; first < end; first += kBatch) {
                       const int m = static_cast<int>(std::min<uint64_t>(kBatch, end - first));
                       double ox[kBatch], oy[kBatch], oz[kBatch], dx[kBatch], dy[kBatch], dz[kBatch];
                       double t[kBatch], lo[kBatch], hi[kBatch];
                       bool inside[kBatch], bracketed[kBatch], started[kBatch];
                       uint64_t rays[kBatch];
                       int active[kBatch];
                       int active_count = 0;
                       for (int i = 0; i < m; ++i) {
                           const uint64_t ray = rays[i] = visit(order, first + i);
                           const double *o = origins + ray * 3, *d = directions + ray * 3;
                           for (int a = 0; a < 3; ++a) {
                               check_coordinate(o[a], "ray origin", ray);
                               check_coordinate(d[a], "ray direction", ray);
                           }
                           // Scaled first: squares of components near 1e150 would overflow.
                           const double scale = std::fmax(std::fabs(d[0]), std::fmax(std::fabs(d[1]), std::fabs(d[2])));
                           if (!(scale > 0)) throw Error(format("ray direction %llu is zero",
                                                                static_cast<unsigned long long>(ray)));
                           const double ux = d[0] / scale, uy = d[1] / scale, uz = d[2] / scale;
                           const double length = std::sqrt(ux * ux + uy * uy + uz * uz);
                           ox[i] = o[0];
                           oy[i] = o[1];
                           oz[i] = o[2];
                           dx[i] = ux / length;
                           dy[i] = uy / length;
                           dz[i] = uz / length;
                           t[i] = 0;
                           bracketed[i] = false;
                           started[i] = false;
                           inside[i] = false;
                           hits[ray] = 0;
                           distances[ray] = max_distance;
                           active[active_count++] = i;
                       }
                       double px[kBatch], py[kBatch], pz[kBatch], values[kBatch];
                       for (int64_t iteration = 0; iteration < steps + kBisections && active_count > 0; ++iteration) {
                           for (int j = 0; j < active_count; ++j) {
                               const int i = active[j];
                               const double along = bracketed[i] ? 0.5 * (lo[i] + hi[i]) : t[i];
                               px[j] = ox[i] + along * dx[i];
                               py[j] = oy[i] + along * dy[i];
                               pz[j] = oz[i] + along * dz[i];
                           }
                           eval_batch(g, node, active_count, px, py, pz, values);
                           int kept = 0;
                           for (int j = 0; j < active_count; ++j) {
                               const int i = active[j];
                               const uint64_t ray = rays[i];
                               const double v = values[j];
                               if (!std::isfinite(v)) {  // only far beyond any sane model: a miss
                                   if (!bracketed[i]) distances[ray] = std::min(t[i], max_distance);
                                   continue;
                               }
                               if (bracketed[i]) {
                                   const double mid = 0.5 * (lo[i] + hi[i]);
                                   if (std::fabs(v) < tol || hi[i] - lo[i] < tol) {
                                       hits[ray] = 1;
                                       distances[ray] = mid;
                                       continue;
                                   }
                                   if ((v < 0) == inside[i]) lo[i] = mid;
                                   else hi[i] = mid;
                                   active[kept++] = i;
                                   continue;
                               }
                               if (std::fabs(v) < tol) {
                                   hits[ray] = 1;
                                   distances[ray] = t[i];
                                   continue;
                               }
                               if (!started[i]) {
                                   started[i] = true;
                                   inside[i] = v < 0;
                               } else if ((v < 0) != inside[i]) {
                                   // Crossed the surface: the root lies between the previous
                                   // sample (stored in lo) and this one.
                                   hi[i] = t[i];
                                   bracketed[i] = true;
                                   active[kept++] = i;
                                   continue;
                               }
                               if (iteration >= steps - 1) {  // out of marching steps: a miss
                                   distances[ray] = std::min(t[i], max_distance);
                                   continue;
                               }
                               lo[i] = t[i];
                               t[i] += std::fabs(v) * inv_lipschitz;
                               if (t[i] > max_distance) continue;  // a miss
                               active[kept++] = i;
                           }
                           active_count = kept;
                       }
                       for (int j = 0; j < active_count; ++j) {  // bisection budget spent: report the bracket
                           const int i = active[j];
                           if (!bracketed[i]) continue;
                           hits[rays[i]] = 1;
                           distances[rays[i]] = 0.5 * (lo[i] + hi[i]);
                       }
                   }
               });
           })
               ? 0
               : -1;
}

}  // extern "C"
