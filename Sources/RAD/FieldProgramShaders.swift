/// Metal source of the GPU field program evaluator (SDF-DESIGN.md, "GPU field
/// programs"). `library` is self-contained: device functions that read a
/// program's words (field.bin, one buffer) plus the compute kernels used to
/// validate and benchmark them. Shaders that render with the field prepend
/// `functions` to their own source, compile it with the program's
/// `FieldProgram.shaderMacros`, and call `rad_field_distance`,
/// `rad_field_gradient` and `rad_field_trace` (or the cached forms that
/// reuse a part's grid and cell) from fragment or compute functions. Every
/// function inlines; `rad_field_trace` calls the interpreter at one site, so
/// a shader that traces once holds one copy of it (a non-inlined call is
/// slower in fragment functions). Everything is float32; the code avoids
/// infinities and NaN tests, so it also compiles with fast math, but
/// `MTLMathMode.safe` keeps transcendental functions and the min/max of
/// signed zeros exact.
enum FieldProgramShaders {
    static let functions = #"""
#include <metal_stdlib>
using namespace metal;

// Register files and optional code, sized per program (FieldProgram.shaderMacros;
// powers of two, at most the format's 16 values and 8 points): smaller
// thread-private arrays keep more threads in flight.
#ifndef RAD_FIELD_VALUES
#define RAD_FIELD_VALUES 16
#endif
#ifndef RAD_FIELD_POINTS
#define RAD_FIELD_POINTS 8
#endif
#ifndef RAD_FIELD_MESHES
#define RAD_FIELD_MESHES 1
#endif
#ifndef RAD_FIELD_POLYGONS
#define RAD_FIELD_POLYGONS 1
#endif
// RAD_FIELD_GRADIENT_PASS 1: a gradient's four samples run the tape once
// (rad_field_run4), which pays where register files are small; with
// larger ones the four-wide registers cost fragment functions more than
// the shared decoding saves (FieldProgram.shaderMacros decides).
#ifndef RAD_FIELD_GRADIENT_PASS
#define RAD_FIELD_GRADIENT_PASS 0
#endif
// The opcodes the program's tapes use (FieldProgram.shaderMacros): bit n of
// RAD_FIELD_OPS_A for opcode n < 32, of RAD_FIELD_OPS_B for 32 + n, of
// RAD_FIELD_OPS_C for 64 + n. The interpreter compiles only those cases:
// a small interpreter stays in the GPU's instruction cache.
#ifndef RAD_FIELD_OPS_A
#define RAD_FIELD_OPS_A 0xFFFFFFFF
#endif
#ifndef RAD_FIELD_OPS_B
#define RAD_FIELD_OPS_B 0xFFFFFFFF
#endif
#ifndef RAD_FIELD_OPS_C
#define RAD_FIELD_OPS_C 0xFFFFFFFF
#endif
#define RAD_FIELD_USES(n) \
    ((((n) < 32 ? uint(RAD_FIELD_OPS_A) : (n) < 64 ? uint(RAD_FIELD_OPS_B) : uint(RAD_FIELD_OPS_C)) >> ((n) & 31)) & 1u)
// A case label no opcode reaches unless the program uses the opcode.
#define RAD_CASE(n) case (RAD_FIELD_USES(n) ? (n) : 256 + (n))
#define RAD_FIELD_STACK 64
#define RAD_FIELD_MAX_STEPS 65536
#define RAD_FIELD_FAR 0xFFFFFFFFu      // RADFieldCell.word of a far cell
#define RAD_FIELD_FAR_CELL 0x80000000u // cell word flags (field.bin)
#define RAD_FIELD_REFINED 0x40000000u
#define RAD_FIELD_TREE_SEARCH 0xFFFFFFFFu
#define RAD_FIELD_BIG 3.0e38f

inline float rad_f(device const uint *w, uint at) { return as_type<float>(w[at]); }
inline float3 rad_f3(device const uint *w, uint at) { return float3(rad_f(w, at), rad_f(w, at + 1), rad_f(w, at + 2)); }
inline float rad_clamp01(float t) { return min(max(t, 0.0f), 1.0f); }
inline float rad_slab(float d, float w) {
    const float od = max(d, 0.0f), ow = max(w, 0.0f);
    return min(max(d, w), 0.0f) + sqrt(od * od + ow * ow);
}
inline float rad_smin(float a, float b, float k) {
    const float h = max(k - fabs(a - b), 0.0f) / k;
    return min(a, b) - h * h * k * 0.25f;
}
inline float rad_smax(float a, float b, float k) {
    const float h = max(k - fabs(a - b), 0.0f) / k;
    return max(a, b) + h * h * k * 0.25f;
}
inline float rad_field_smooth(float a, float b, float k, float sign) {
    const float u = sign * a, v = sign * b;
    const float h = k > 0 ? max(k - fabs(u - v), 0.0f) / k : 0.0f;
    return sign * (min(u, v) - h * h * k * 0.25f);
}
inline float rad_triwave(float a, float period) {
    const float twice = 2 * period;
    float rest = precise::fmod(a, twice);
    if (rest < 0) rest += twice;
    return fabs(rest - period);
}

// ---------------------------------------------------------------- point maps

// Parameters are 16-byte aligned runs read as float4 (k[0], k[1], ...).
inline float3 rad_field_map(uint op, device const float4 *k, float3 q) {
    switch (op) {
    RAD_CASE(1): {  // AFFINE3: t, M = R^T / s row-major; local = M (q - t)
        const float4 k0 = k[0], k1 = k[1], k2 = k[2];
        const float3 d = q - k0.xyz;
        return float3(k0.w * d.x + k1.x * d.y + k1.y * d.z, k1.z * d.x + k1.w * d.y + k2.x * d.z,
                      k2.y * d.x + k2.z * d.y + k2.w * d.z);
    }
    RAD_CASE(2): return q * k[0].xyz;  // STRETCH3: 1 / s
    RAD_CASE(3): {  // MIRROR3: n d
        const float4 n = k[0];
        return q - 2 * (dot(n.xyz, q) - n.w) * n.xyz;
    }
    RAD_CASE(4): {  // TWIST: rate
        const float angle = -k[0].x * q.z;
        const float c = precise::cos(angle), s = precise::sin(angle);
        return float3(c * q.x - s * q.y, s * q.x + c * q.y, q.z);
    }
    RAD_CASE(5): {  // ELONGATE: e
        const float3 e = k[0].xyz;
        return q - clamp(q, -e, e);
    }
    RAD_CASE(6): return float3(length(q.xy) - k[0].x, q.z, 0);  // REVOLVE: offset
    RAD_CASE(7): {  // AFFINE2: R t inv
        const float4 r = k[0], t = k[1];
        const float2 d = q.xy - t.xy;
        return float3((r.x * d.x + r.z * d.y) * t.z, (r.y * d.x + r.w * d.y) * t.z, q.z);
    }
    RAD_CASE(8): return float3(q.xy * k[0].xy, q.z);  // STRETCH2: 1 / s
    RAD_CASE(9): {  // MIRROR2: n d
        const float4 n = k[0];
        return float3(q.xy - 2 * (dot(n.xy, q.xy) - n.z) * n.xy, q.z);
    }
    RAD_CASE(10): {  // CLAMP3: lo hi
        const float4 k0 = k[0], k1 = k[1];
        return clamp(q, k0.xyz, float3(k0.w, k1.x, k1.y));
    }
    RAD_CASE(12): return q - k[0].xyz;  // TRANSLATE3: t
    default: return q;
    }
}

// ---------------------------------------------------------------- polygons

inline float rad_edge_distance2(device const uint *w, uint data, uint e, float2 p, thread float &t) {
    const uint at = data + 7 * e;
    const float2 v = float2(rad_f(w, at), rad_f(w, at + 1)), ed = float2(rad_f(w, at + 2), rad_f(w, at + 3));
    const float2 d = p - v;
    t = rad_clamp01(dot(d, ed) * rad_f(w, at + 4));
    const float2 r = d - t * ed;
    return dot(r, r);
}

inline float rad_point_box2(float2 p, float2 lo, float2 hi) {
    const float2 g = max(max(lo - p, p - hi), 0.0f);
    return dot(g, g);
}

float rad_field_polygon(device const uint *w, uint polygon, uint ranges, uint range_count, float2 p, thread uint *stack) {
    const uint record = w[7] + 8 * polygon;
    const uint count = w[record], data = w[record + 1], tree = w[record + 2];
    float best = RAD_FIELD_BIG, best_t = 0.5f;
    uint best_edge = 0;
    if (range_count == RAD_FIELD_TREE_SEARCH) {
        int top = 0;
        stack[top++] = 0;
        for (uint guard = 0; top > 0 && guard < RAD_FIELD_MAX_STEPS; ++guard) {
            const uint index = stack[--top];
            const uint at = tree + 7 * index;
            const float2 lo = float2(rad_f(w, at), rad_f(w, at + 1)), hi = float2(rad_f(w, at + 2), rad_f(w, at + 3));
            if (!(rad_point_box2(p, lo, hi) < best)) continue;
            const uint right = w[at + 6];
            if (right == 0) {
                for (uint e = w[at + 4]; e < w[at + 5]; ++e) {
                    float t;
                    const float d2 = rad_edge_distance2(w, data, e, p, t);
                    if (d2 < best) { best = d2; best_t = t; best_edge = e; }
                }
                continue;
            }
            const uint l_at = tree + 7 * (index + 1), r_at = tree + 7 * right;
            const float dl = rad_point_box2(p, float2(rad_f(w, l_at), rad_f(w, l_at + 1)), float2(rad_f(w, l_at + 2), rad_f(w, l_at + 3)));
            const float dr = rad_point_box2(p, float2(rad_f(w, r_at), rad_f(w, r_at + 1)), float2(rad_f(w, r_at + 2), rad_f(w, r_at + 3)));
            if (top + 2 > RAD_FIELD_STACK) break;
            const bool left_first = dl <= dr;
            stack[top++] = left_first ? right : index + 1;
            stack[top++] = left_first ? index + 1 : right;
        }
    } else {
        for (uint r = 0; r < range_count; ++r) {
            const uint end = w[ranges + 2 * r + 1];
            for (uint e = w[ranges + 2 * r]; e < end; ++e) {
                float t;
                const float d2 = rad_edge_distance2(w, data, e, p, t);
                if (d2 < best) { best = d2; best_t = t; best_edge = e; }
            }
        }
    }
    if (!(best > 0)) return 0;
    const float2 lo = float2(rad_f(w, record + 4), rad_f(w, record + 5)), hi = float2(rad_f(w, record + 6), rad_f(w, record + 7));
    bool inside = false;
    if (p.x < lo.x || p.x > hi.x || p.y < lo.y || p.y > hi.y) {
        inside = false;  // outside the vertices' box: outside
    } else if (best_t > 0 && best_t < 1) {
        const uint at = data + 7 * best_edge;
        inside = rad_f(w, at + 2) * (p.y - rad_f(w, at + 1)) - rad_f(w, at + 3) * (p.x - rad_f(w, at)) > 0;
    } else {
        const uint corner = best_t <= 0 ? best_edge : (best_edge + 1 == count ? 0 : best_edge + 1);
        const uint at = data + 7 * corner;
        inside = (p.x - rad_f(w, at)) * rad_f(w, at + 5) + (p.y - rad_f(w, at + 1)) * rad_f(w, at + 6) < 0;
    }
    const float d = sqrt(best);
    return inside ? -d : d;
}

// ---------------------------------------------------------------- triangle meshes

struct RADMeshNearest {
    float d2;
    uint triangle;
    int feature;  // 0 face, 1 + k edge k (corner k to k + 1), 4 + k corner k
};

inline float rad_box_distance2(device const uint *w, uint at, float3 p) {
    const float3 g = max(max(rad_f3(w, at) - p, p - rad_f3(w, at + 3)), 0.0f);
    return dot(g, g);
}

inline void rad_mesh_triangle(device const uint *w, uint mesh, uint t, float3 p, thread RADMeshNearest &best) {
    const uint positions = w[mesh + 1], triangles = w[mesh + 4], normals = w[mesh + 6];
    const uint3 v = uint3(w[triangles + 3 * t], w[triangles + 3 * t + 1], w[triangles + 3 * t + 2]);
    const float3 a = rad_f3(w, positions + 3 * v.x), b = rad_f3(w, positions + 3 * v.y), c = rad_f3(w, positions + 3 * v.z);
    const float3 ap = p - a;
    const float plane = dot(ap, rad_f3(w, normals + 3 * t));
    const float plane2 = plane * plane;
    if (!(plane2 < best.d2)) return;
    const float3 ab = b - a, ac = c - a, bp = p - b, cp = p - c;
    const float s1 = dot(ab, ap), s2 = dot(ac, ap), s3 = dot(ab, bp), s4 = dot(ac, bp), s5 = dot(ab, cp), s6 = dot(ac, cp);
    const float vc = s1 * s4 - s3 * s2, vb = s5 * s2 - s1 * s6, va = s3 * s6 - s5 * s4;
    float d2;
    int feature;
    if (s1 <= 0 && s2 <= 0) { d2 = dot(ap, ap); feature = 4; }
    else if (s3 >= 0 && s4 <= s3) { d2 = dot(bp, bp); feature = 5; }
    else if (vc <= 0 && s1 >= 0 && s3 <= 0) { const float3 e = ap - s1 / (s1 - s3) * ab; d2 = dot(e, e); feature = 1; }
    else if (s6 >= 0 && s5 <= s6) { d2 = dot(cp, cp); feature = 6; }
    else if (vb <= 0 && s2 >= 0 && s6 <= 0) { const float3 e = ap - s2 / (s2 - s6) * ac; d2 = dot(e, e); feature = 3; }
    else if (va <= 0 && s4 - s3 >= 0 && s5 - s6 >= 0) {
        const float u = (s4 - s3) / ((s4 - s3) + (s5 - s6));
        const float3 e = bp - u * (c - b);
        d2 = dot(e, e);
        feature = 2;
    } else { d2 = plane2; feature = 0; }
    if (d2 < best.d2) { best.d2 = d2; best.triangle = t; best.feature = feature; }
}

// Nearest-first branch and bound over the subtree at `root` of the hierarchy at `tree`.
void rad_mesh_nearest(device const uint *w, uint mesh, uint root, float3 p, thread RADMeshNearest &best,
                      thread uint *nodes, thread float *keys) {
    const uint tree = w[mesh + 7];
    int top = 0;
    nodes[top] = root;
    keys[top++] = rad_box_distance2(w, tree + 9 * root, p);
    for (uint guard = 0; top > 0 && guard < RAD_FIELD_MAX_STEPS; ++guard) {
        --top;
        const uint index = nodes[top];
        if (!(keys[top] < best.d2)) continue;
        const uint at = tree + 9 * index;
        const uint right = w[at + 6];
        if (right == 0) {
            const uint first = w[at + 7], end = first + w[at + 8];
            for (uint i = first; i < end; ++i) rad_mesh_triangle(w, mesh, i, p, best);
            continue;
        }
        const uint left = index + 1;
        const float dl = rad_box_distance2(w, tree + 9 * left, p), dr = rad_box_distance2(w, tree + 9 * right, p);
        const bool left_nearer = dl <= dr;
        const uint near_node = left_nearer ? left : right, far_node = left_nearer ? right : left;
        const float near_key = left_nearer ? dl : dr, far_key = left_nearer ? dr : dl;
        if (top + 2 > RAD_FIELD_STACK) break;
        if (far_key < best.d2) { nodes[top] = far_node; keys[top++] = far_key; }
        if (near_key < best.d2) { nodes[top] = near_node; keys[top++] = near_key; }
    }
}

inline float rad_mesh_signed(device const uint *w, uint mesh, float3 p, RADMeshNearest best) {
    if (!(best.d2 > 0)) return 0;
    const uint positions = w[mesh + 1], vertex_normals = w[mesh + 2], triangles = w[mesh + 4];
    const uint neighbours = w[mesh + 5], normals = w[mesh + 6];
    const uint t = best.triangle;
    float3 normal, q;
    if (best.feature == 0) {
        normal = rad_f3(w, normals + 3 * t);
        q = rad_f3(w, positions + 3 * w[triangles + 3 * t]);
    } else if (best.feature <= 3) {
        const uint k = uint(best.feature - 1);
        normal = rad_f3(w, normals + 3 * t) + rad_f3(w, normals + 3 * w[neighbours + 3 * t + k]);
        q = rad_f3(w, positions + 3 * w[triangles + 3 * t + k]);
    } else {
        const uint corner = w[triangles + 3 * t + uint(best.feature - 4)];
        normal = rad_f3(w, vertex_normals + 3 * corner);
        q = rad_f3(w, positions + 3 * corner);
    }
    const float d = sqrt(best.d2);
    return dot(p - q, normal) < 0 ? -d : d;
}

inline bool rad_box_contains(device const uint *w, uint at, float3 p) {
    return all(p >= rad_f3(w, at)) && all(p <= rad_f3(w, at + 3));
}

float rad_field_mesh(device const uint *w, uint index, float3 p, thread uint *nodes, thread float *keys) {
    const uint mesh = w[5] + 24 * index;
    RADMeshNearest best;
    best.d2 = RAD_FIELD_BIG;
    best.triangle = 0;
    best.feature = 0;
    rad_mesh_nearest(w, mesh, 0, p, best, nodes, keys);
    if (!rad_box_contains(w, mesh + 16, p)) return sqrt(best.d2);  // the solid lies within the vertices' box
    float value = rad_mesh_signed(w, mesh, p, best);
    const uint bodies = w[mesh + 9];
    if (bodies <= 1) return value;
    // Only a body holding p can lower the value: those whose boxes hold it.
    const uint nearest_body = w[w[mesh + 15] + best.triangle];
    const uint body_tree = w[mesh + 12], body_order = w[mesh + 14], body_records = w[mesh + 10], roots = w[mesh + 11];
    uint stack[32];
    int top = 0;
    stack[top++] = 0;
    for (uint guard = 0; top > 0 && guard < RAD_FIELD_MAX_STEPS; ++guard) {
        const uint index_node = stack[--top];
        const uint at = body_tree + 9 * index_node;
        if (!rad_box_contains(w, at, p)) continue;
        const uint right = w[at + 6];
        if (right != 0) {
            if (top + 2 > 32) break;
            stack[top++] = right;
            stack[top++] = index_node + 1;
            continue;
        }
        const uint first = w[at + 7], end = first + w[at + 8];
        for (uint i = first; i < end; ++i) {
            const uint body = w[body_order + i];
            const uint record = body_records + 8 * body;
            if (body == nearest_body || !rad_box_contains(w, record, p)) continue;
            RADMeshNearest own;
            own.d2 = RAD_FIELD_BIG;
            own.triangle = 0;
            own.feature = 0;
            const uint root_first = w[record + 6], root_end = root_first + w[record + 7];
            for (uint r = root_first; r < root_end; ++r) rad_mesh_nearest(w, mesh, w[roots + r], p, own, nodes, keys);
            value = min(value, rad_mesh_signed(w, mesh, p, own));
        }
    }
    return value;
}

// ---------------------------------------------------------------- tapes

inline float rad_field_leaf(uint op, device const float4 *k, float3 q) {
    switch (op) {
    RAD_CASE(20): return length(q) - k[0].x;  // SPHERE
    RAD_CASE(21): {  // BOX: h - r, r
        const float4 k0 = k[0];
        const float3 d = fabs(q) - k0.xyz;
        return length(max(d, 0.0f)) + min(max(d.x, max(d.y, d.z)), 0.0f) - k0.w;
    }
    RAD_CASE(22): {  // CYLINDER
        const float4 k0 = k[0];
        return rad_slab(length(q.xy) - k0.x, fabs(q.z) - k0.y);
    }
    RAD_CASE(23): {  // CONE: r1 r2 h kx ky inv
        const float4 k0 = k[0], k1 = k[1];
        const float r1 = k0.x, r2 = k0.y, h = k0.z, kx = k0.w, ky = k1.x, inv = k1.y;
        const float qx = length(q.xy), qy = q.z;
        const float cax = qx - min(qx, qy < 0 ? r1 : r2), cay = fabs(qy) - h;
        const float t = rad_clamp01(((r2 - qx) * kx + (h - qy) * ky) * inv);
        const float cbx = qx - r2 + kx * t, cby = qy - h + ky * t;
        const float sign = (cbx < 0 && cay < 0) ? -1.0f : 1.0f;
        return sign * sqrt(min(cax * cax + cay * cay, cbx * cbx + cby * cby));
    }
    RAD_CASE(24): {  // CAPSULE: a b-a inv r
        const float4 k0 = k[0], k1 = k[1];
        const float3 pa = q - k0.xyz, ba = float3(k0.w, k1.x, k1.y);
        const float t = rad_clamp01(dot(pa, ba) * k1.z);
        return length(pa - t * ba) - k1.w;
    }
    RAD_CASE(25): {  // TORUS
        const float4 k0 = k[0];
        return length(float2(length(q.xy) - k0.x, q.z)) - k0.y;
    }
    RAD_CASE(26): {  // ELLIPSOID: 1/r c0
        const float4 k0 = k[0];
        return (length(q * k0.xyz) - 1) * k0.w;
    }
    RAD_CASE(27): {  // HALFSPACE
        const float4 k0 = k[0];
        return dot(k0.xyz, q) - k0.w;
    }
    RAD_CASE(29): return length(q.xy) - k[0].x;  // CIRCLE
    RAD_CASE(30): {  // RECT: h - r, r
        const float4 k0 = k[0];
        const float2 d = fabs(q.xy) - k0.xy;
        return length(max(d, 0.0f)) + min(max(d.x, d.y), 0.0f) - k0.z;
    }
    RAD_CASE(32): {  // SEGMENT: a b-a inv r
        const float4 k0 = k[0], k1 = k[1];
        const float2 pa = q.xy - k0.xy, ba = k0.zw;
        const float t = rad_clamp01(dot(pa, ba) * k1.x);
        return length(pa - t * ba) - k1.y;
    }
    RAD_CASE(33): {  // ELLIPSE
        const float4 k0 = k[0];
        return (length(q.xy * k0.xy) - 1) * k0.z;
    }
    RAD_CASE(34): {  // HALFPLANE
        const float4 k0 = k[0];
        return k0.x * q.x + k0.y * q.y - k0.z;
    }
    default: return 0;
    }
}

// Runs the tape at word `tape` for point p. Records (16 bytes): x = op |
// fused map << 7 | dst << 8 | post << 12 | a << 16 | post register << 20 |
// b << 24 | swap << 28; y = third operand or the post parameter; z =
// parameters (a MESH: its index); w = fused map op << 28 | its parameters.
float rad_field_run(device const uint *w, uint tape, float3 point) {
    device const uint4 *code = (device const uint4 *)(w + tape);
    float V[RAD_FIELD_VALUES];
    float3 P[RAD_FIELD_POINTS];
#if RAD_FIELD_MESHES
    uint nodes[RAD_FIELD_STACK];   // mesh and polygon hierarchy searches
    float keys[RAD_FIELD_STACK];
#elif RAD_FIELD_POLYGONS
    uint nodes[RAD_FIELD_STACK];
#endif
    P[0] = point;
    for (uint i = 0; i < RAD_FIELD_MAX_STEPS; ++i) {
        const uint4 ins = code[i];
        const uint op = ins.x & 0x7Fu, d = (ins.x >> 8) & 0xFu, a = (ins.x >> 16) & 0xFu, b = (ins.x >> 24) & 0xFu;
        if (op == 0) return V[a & (RAD_FIELD_VALUES - 1)];
        device const float4 *k = (device const float4 *)(w + ins.z);
        if (op < 20) {  // point maps
            const uint dst = d & (RAD_FIELD_POINTS - 1);
            if (op == 11) {  // POINT3: three values
                P[dst] = float3(V[a & (RAD_FIELD_VALUES - 1)], V[b & (RAD_FIELD_VALUES - 1)], V[ins.y & (RAD_FIELD_VALUES - 1)]);
            } else {
                P[dst] = rad_field_map(op, k, P[a & (RAD_FIELD_POINTS - 1)]);
            }
            continue;
        }
        float result;
        if (op < 40) {  // leaves, after a fused point map
            float3 q = P[a & (RAD_FIELD_POINTS - 1)];
            if (ins.x & 0x80u) q = rad_field_map(ins.w >> 28, (device const float4 *)(w + (ins.w & 0x0FFFFFFFu)), q);
            switch (op) {
#if RAD_FIELD_MESHES
            RAD_CASE(28): result = rad_field_mesh(w, ins.z, q, nodes, keys); break;
#endif
#if RAD_FIELD_POLYGONS
            RAD_CASE(31): result = rad_field_polygon(w, w[ins.z], ins.z + 2, w[ins.z + 1], q.xy, nodes); break;
#endif
            RAD_CASE(35): result = q[min(b, 2u)]; break;  // COORD
            default: result = rad_field_leaf(op, k, q); break;
            }
        } else {
            const float va = V[a & (RAD_FIELD_VALUES - 1)], vb = V[b & (RAD_FIELD_VALUES - 1)];
            switch (op) {
            RAD_CASE(40): result = va * k[0].x; break;
            RAD_CASE(41): result = va - k[0].x; break;
            RAD_CASE(42): result = max(va, -va - k[0].x); break;
            RAD_CASE(43): result = max(-va, va - k[0].x); break;
            RAD_CASE(44): RAD_CASE(45): result = fabs(va) - k[0].x; break;
            RAD_CASE(46): {  // EXTRUDE: centre half (b: the point)
                const float4 k0 = k[0];
                const float z = P[b & (RAD_FIELD_POINTS - 1)].z;
                result = rad_slab(va, fabs(z - k0.x) - k0.y);
                break;
            }
            RAD_CASE(47): {  // TAPER: z0 inv taper c0 centre half
                const float4 k0 = k[0], k1 = k[1];
                const float z = P[b & (RAD_FIELD_POINTS - 1)].z;
                const float dd = (va - k0.z * rad_clamp01((z - k0.x) * k0.y)) * k0.w;
                result = rad_slab(dd, fabs(z - k1.x) - k1.y);
                break;
            }
            RAD_CASE(48): {  // LOFT: z0 inv c0 centre half (y: the point)
                const float4 k0 = k[0], k1 = k[1];
                const float z = P[ins.y & (RAD_FIELD_POINTS - 1)].z;
                const float t = rad_clamp01((z - k0.x) * k0.y);
                result = rad_slab(((1 - t) * va + t * vb) * k0.z, fabs(z - k0.w) - k1.x);
                break;
            }
            RAD_CASE(49): result = min(va, vb); break;
            RAD_CASE(50): result = max(va, vb); break;
            RAD_CASE(51): result = rad_smin(va, vb, k[0].x); break;
            RAD_CASE(52): result = rad_smax(va, vb, k[0].x); break;
            RAD_CASE(53): result = max(va, -vb); break;
            RAD_CASE(54): result = rad_smax(va, -vb, k[0].x); break;
            RAD_CASE(55): {  // IMPLICIT: centre half L (a: field value, b: the point)
                const float4 k0 = k[0], k1 = k[1];
                const float3 qq = fabs(P[b & (RAD_FIELD_POINTS - 1)] - k0.xyz) - float3(k0.w, k1.x, k1.y);
                const float box = length(max(qq, 0.0f)) + min(max(qq.x, max(qq.y, qq.z)), 0.0f);
                const float h = va / k1.z;
                result = h > 0 && box > 0 ? sqrt(h * h + box * box) : max(h, box);
                break;
            }
            RAD_CASE(60): result = k[0].x; break;
            RAD_CASE(61): result = va + vb; break;
            RAD_CASE(62): result = va - vb; break;
            RAD_CASE(63): result = va * vb; break;
            RAD_CASE(64): result = va / vb; break;
            RAD_CASE(65): result = -va; break;
            RAD_CASE(66): result = fabs(va); break;
            RAD_CASE(67): RAD_CASE(76): result = va * va; break;
            RAD_CASE(68): result = sqrt(va); break;
            RAD_CASE(69): result = precise::sin(va); break;
            RAD_CASE(70): result = precise::cos(va); break;
            RAD_CASE(71): result = precise::exp(va); break;
            RAD_CASE(72): result = precise::log(va); break;
            RAD_CASE(73): result = precise::pow(va, k[0].x); break;
            RAD_CASE(74): result = precise::atan2(va, vb); break;
            RAD_CASE(75): { const float4 k0 = k[0]; result = min(max(va, k0.x), k0.y); break; }
            RAD_CASE(77): result = va + vb * vb; break;
            RAD_CASE(78): result = rad_triwave(va, k[0].x); break;
            RAD_CASE(79): result = rad_field_smooth(va, vb, k[0].x, 1.0f); break;
            RAD_CASE(80): result = rad_field_smooth(va, vb, k[0].x, -1.0f); break;
            default: result = 0; break;
            }
        }
        // The post operation absorbed from the next instruction, its
        // operands in that instruction's order.
        const uint post = (ins.x >> 12) & 0xFu;
        if (post != 0) {
            const float other = V[(ins.x >> 20) & (RAD_FIELD_VALUES - 1)], kk = as_type<float>(ins.y);
            const bool swap = (ins.x >> 28) & 1u;
            const float l = swap ? other : result, r = swap ? result : other;
            switch (post) {
            case 1: result = min(l, r); break;
            case 2: result = max(l, r); break;
            case 3: result = rad_smin(l, r, kk); break;
            case 4: result = rad_smax(l, r, kk); break;
            case 5: result = max(l, -r); break;
            case 6: result = rad_smax(l, -r, kk); break;
            case 7: result = result * kk; break;
            default: result = result - kk; break;
            }
        }
        V[d & (RAD_FIELD_VALUES - 1)] = result;
    }
    return 0;
}

// ---------------------------------------------------------------- four points at once
#if RAD_FIELD_GRADIENT_PASS

// A gradient's four samples in one pass of the tape: decoding, dispatch and
// parameter loads, most of an instruction's cost, are shared. The points
// of the stencil, by coordinate.
struct RADFieldPoints {
    float4 x, y, z;
};

inline float4 rad_len2(float4 x, float4 y) { return sqrt(x * x + y * y); }
inline float4 rad_len3(float4 x, float4 y, float4 z) { return sqrt(x * x + y * y + z * z); }
inline float4 rad_clamp01(float4 t) { return min(max(t, 0.0f), 1.0f); }
inline float4 rad_slab(float4 d, float4 w) {
    const float4 od = max(d, 0.0f), ow = max(w, 0.0f);
    return min(max(d, w), 0.0f) + sqrt(od * od + ow * ow);
}
inline float4 rad_smin(float4 a, float4 b, float k) {
    const float4 h = max(k - fabs(a - b), 0.0f) / k;
    return min(a, b) - h * h * k * 0.25f;
}
inline float4 rad_smax(float4 a, float4 b, float k) {
    const float4 h = max(k - fabs(a - b), 0.0f) / k;
    return max(a, b) + h * h * k * 0.25f;
}
inline float4 rad_field_smooth(float4 a, float4 b, float k, float sign) {
    const float4 u = sign * a, v = sign * b;
    const float4 h = k > 0 ? max(k - fabs(u - v), 0.0f) / k : float4(0.0f);
    return sign * (min(u, v) - h * h * k * 0.25f);
}
inline float4 rad_triwave(float4 a, float period) {
    const float twice = 2 * period;
    float4 rest = precise::fmod(a, float4(twice));
    rest = select(rest, rest + twice, rest < 0);
    return fabs(rest - period);
}

inline RADFieldPoints rad_field_map4(uint op, device const float4 *k, RADFieldPoints q) {
    RADFieldPoints r;
    switch (op) {
    RAD_CASE(1): {  // AFFINE3
        const float4 k0 = k[0], k1 = k[1], k2 = k[2];
        const float4 dx = q.x - k0.x, dy = q.y - k0.y, dz = q.z - k0.z;
        r.x = k0.w * dx + k1.x * dy + k1.y * dz;
        r.y = k1.z * dx + k1.w * dy + k2.x * dz;
        r.z = k2.y * dx + k2.z * dy + k2.w * dz;
        return r;
    }
    RAD_CASE(2): { const float4 s = k[0]; r.x = q.x * s.x; r.y = q.y * s.y; r.z = q.z * s.z; return r; }
    RAD_CASE(3): {
        const float4 n = k[0];
        const float4 t = 2 * ((n.x * q.x + n.y * q.y + n.z * q.z) - n.w);
        r.x = q.x - t * n.x; r.y = q.y - t * n.y; r.z = q.z - t * n.z;
        return r;
    }
    RAD_CASE(4): {
        const float4 angle = -k[0].x * q.z;
        const float4 c = precise::cos(angle), s = precise::sin(angle);
        r.x = c * q.x - s * q.y; r.y = s * q.x + c * q.y; r.z = q.z;
        return r;
    }
    RAD_CASE(5): {
        const float4 e = k[0];
        r.x = q.x - clamp(q.x, -e.x, e.x); r.y = q.y - clamp(q.y, -e.y, e.y); r.z = q.z - clamp(q.z, -e.z, e.z);
        return r;
    }
    RAD_CASE(6): r.x = rad_len2(q.x, q.y) - k[0].x; r.y = q.z; r.z = 0; return r;
    RAD_CASE(7): {
        const float4 m = k[0], t = k[1];
        const float4 dx = q.x - t.x, dy = q.y - t.y;
        r.x = (m.x * dx + m.z * dy) * t.z; r.y = (m.y * dx + m.w * dy) * t.z; r.z = q.z;
        return r;
    }
    RAD_CASE(8): { const float4 s = k[0]; r.x = q.x * s.x; r.y = q.y * s.y; r.z = q.z; return r; }
    RAD_CASE(9): {
        const float4 n = k[0];
        const float4 t = 2 * ((n.x * q.x + n.y * q.y) - n.z);
        r.x = q.x - t * n.x; r.y = q.y - t * n.y; r.z = q.z;
        return r;
    }
    RAD_CASE(10): {
        const float4 k0 = k[0], k1 = k[1];
        r.x = clamp(q.x, k0.x, k0.w); r.y = clamp(q.y, k0.y, k1.x); r.z = clamp(q.z, k0.z, k1.y);
        return r;
    }
    RAD_CASE(12): { const float4 t = k[0]; r.x = q.x - t.x; r.y = q.y - t.y; r.z = q.z - t.z; return r; }
    default: return q;
    }
}

inline float4 rad_field_leaf4(uint op, device const float4 *k, RADFieldPoints q) {
    switch (op) {
    RAD_CASE(20): return rad_len3(q.x, q.y, q.z) - k[0].x;
    RAD_CASE(21): {
        const float4 k0 = k[0];
        const float4 dx = fabs(q.x) - k0.x, dy = fabs(q.y) - k0.y, dz = fabs(q.z) - k0.z;
        return rad_len3(max(dx, 0.0f), max(dy, 0.0f), max(dz, 0.0f)) + min(max(dx, max(dy, dz)), 0.0f) - k0.w;
    }
    RAD_CASE(22): { const float4 k0 = k[0]; return rad_slab(rad_len2(q.x, q.y) - k0.x, fabs(q.z) - k0.y); }
    RAD_CASE(23): {
        const float4 k0 = k[0], k1 = k[1];
        const float r1 = k0.x, r2 = k0.y, h = k0.z, kx = k0.w, ky = k1.x, inv = k1.y;
        const float4 qx = rad_len2(q.x, q.y), qy = q.z;
        const float4 cax = qx - min(qx, select(float4(r2), float4(r1), qy < 0)), cay = fabs(qy) - h;
        const float4 t = rad_clamp01(((r2 - qx) * kx + (h - qy) * ky) * inv);
        const float4 cbx = qx - r2 + kx * t, cby = qy - h + ky * t;
        const float4 sign = select(float4(1.0f), float4(-1.0f), (cbx < 0) & (cay < 0));
        return sign * sqrt(min(cax * cax + cay * cay, cbx * cbx + cby * cby));
    }
    RAD_CASE(24): {
        const float4 k0 = k[0], k1 = k[1];
        const float4 px = q.x - k0.x, py = q.y - k0.y, pz = q.z - k0.z;
        const float4 t = rad_clamp01((px * k0.w + py * k1.x + pz * k1.y) * k1.z);
        return rad_len3(px - t * k0.w, py - t * k1.x, pz - t * k1.y) - k1.w;
    }
    RAD_CASE(25): { const float4 k0 = k[0]; return rad_len2(rad_len2(q.x, q.y) - k0.x, q.z) - k0.y; }
    RAD_CASE(26): { const float4 k0 = k[0]; return (rad_len3(q.x * k0.x, q.y * k0.y, q.z * k0.z) - 1) * k0.w; }
    RAD_CASE(27): { const float4 k0 = k[0]; return k0.x * q.x + k0.y * q.y + k0.z * q.z - k0.w; }
    RAD_CASE(29): return rad_len2(q.x, q.y) - k[0].x;
    RAD_CASE(30): {
        const float4 k0 = k[0];
        const float4 dx = fabs(q.x) - k0.x, dy = fabs(q.y) - k0.y;
        return rad_len2(max(dx, 0.0f), max(dy, 0.0f)) + min(max(dx, dy), 0.0f) - k0.z;
    }
    RAD_CASE(32): {
        const float4 k0 = k[0], k1 = k[1];
        const float4 px = q.x - k0.x, py = q.y - k0.y;
        const float4 t = rad_clamp01((px * k0.z + py * k0.w) * k1.x);
        return rad_len2(px - t * k0.z, py - t * k0.w) - k1.y;
    }
    RAD_CASE(33): { const float4 k0 = k[0]; return (rad_len2(q.x * k0.x, q.y * k0.y) - 1) * k0.z; }
    RAD_CASE(34): { const float4 k0 = k[0]; return k0.x * q.x + k0.y * q.y - k0.z; }
    default: return 0;
    }
}

// The tape's values at the four points of the gradient stencil around p
// (p + eps * rad_field_stencil[i]), as rad_field_run computes each.
float4 rad_field_run4(device const uint *w, uint tape, float3 p, float eps) {
    device const uint4 *code = (device const uint4 *)(w + tape);
    float4 V[RAD_FIELD_VALUES];
    RADFieldPoints P[RAD_FIELD_POINTS];
#if RAD_FIELD_MESHES
    uint nodes[RAD_FIELD_STACK];
    float keys[RAD_FIELD_STACK];
#elif RAD_FIELD_POLYGONS
    uint nodes[RAD_FIELD_STACK];
#endif
    P[0].x = p.x + eps * float4(1, -1, -1, 1);
    P[0].y = p.y + eps * float4(-1, -1, 1, 1);
    P[0].z = p.z + eps * float4(-1, 1, -1, 1);
    for (uint i = 0; i < RAD_FIELD_MAX_STEPS; ++i) {
        const uint4 ins = code[i];
        const uint op = ins.x & 0x7Fu, d = (ins.x >> 8) & 0xFu, a = (ins.x >> 16) & 0xFu, b = (ins.x >> 24) & 0xFu;
        if (op == 0) return V[a & (RAD_FIELD_VALUES - 1)];
        device const float4 *k = (device const float4 *)(w + ins.z);
        if (op < 20) {
            const uint dst = d & (RAD_FIELD_POINTS - 1);
            if (op == 11) {
                RADFieldPoints r;
                r.x = V[a & (RAD_FIELD_VALUES - 1)]; r.y = V[b & (RAD_FIELD_VALUES - 1)]; r.z = V[ins.y & (RAD_FIELD_VALUES - 1)];
                P[dst] = r;
            } else {
                P[dst] = rad_field_map4(op, k, P[a & (RAD_FIELD_POINTS - 1)]);
            }
            continue;
        }
        float4 result;
        if (op < 40) {
            RADFieldPoints q = P[a & (RAD_FIELD_POINTS - 1)];
            if (ins.x & 0x80u) q = rad_field_map4(ins.w >> 28, (device const float4 *)(w + (ins.w & 0x0FFFFFFFu)), q);
            switch (op) {
#if RAD_FIELD_MESHES
            RAD_CASE(28):
                for (uint l = 0; l < 4; ++l) result[l] = rad_field_mesh(w, ins.z, float3(q.x[l], q.y[l], q.z[l]), nodes, keys);
                break;
#endif
#if RAD_FIELD_POLYGONS
            RAD_CASE(31):
                for (uint l = 0; l < 4; ++l) result[l] = rad_field_polygon(w, w[ins.z], ins.z + 2, w[ins.z + 1], float2(q.x[l], q.y[l]), nodes);
                break;
#endif
            RAD_CASE(35): result = b == 0 ? q.x : b == 1 ? q.y : q.z; break;
            default: result = rad_field_leaf4(op, k, q); break;
            }
        } else {
            const float4 va = V[a & (RAD_FIELD_VALUES - 1)], vb = V[b & (RAD_FIELD_VALUES - 1)];
            switch (op) {
            RAD_CASE(40): result = va * k[0].x; break;
            RAD_CASE(41): result = va - k[0].x; break;
            RAD_CASE(42): result = max(va, -va - k[0].x); break;
            RAD_CASE(43): result = max(-va, va - k[0].x); break;
            RAD_CASE(44): RAD_CASE(45): result = fabs(va) - k[0].x; break;
            RAD_CASE(46): { const float4 k0 = k[0]; result = rad_slab(va, fabs(P[b & (RAD_FIELD_POINTS - 1)].z - k0.x) - k0.y); break; }
            RAD_CASE(47): {
                const float4 k0 = k[0], k1 = k[1];
                const float4 z = P[b & (RAD_FIELD_POINTS - 1)].z;
                result = rad_slab((va - k0.z * rad_clamp01((z - k0.x) * k0.y)) * k0.w, fabs(z - k1.x) - k1.y);
                break;
            }
            RAD_CASE(48): {
                const float4 k0 = k[0], k1 = k[1];
                const float4 z = P[ins.y & (RAD_FIELD_POINTS - 1)].z;
                const float4 t = rad_clamp01((z - k0.x) * k0.y);
                result = rad_slab(((1 - t) * va + t * vb) * k0.z, fabs(z - k0.w) - k1.x);
                break;
            }
            RAD_CASE(49): result = min(va, vb); break;
            RAD_CASE(50): result = max(va, vb); break;
            RAD_CASE(51): result = rad_smin(va, vb, k[0].x); break;
            RAD_CASE(52): result = rad_smax(va, vb, k[0].x); break;
            RAD_CASE(53): result = max(va, -vb); break;
            RAD_CASE(54): result = rad_smax(va, -vb, k[0].x); break;
            RAD_CASE(55): {
                const float4 k0 = k[0], k1 = k[1];
                const RADFieldPoints pp = P[b & (RAD_FIELD_POINTS - 1)];
                const float4 qx = fabs(pp.x - k0.x) - k0.w, qy = fabs(pp.y - k0.y) - k1.x, qz = fabs(pp.z - k0.z) - k1.y;
                const float4 box = rad_len3(max(qx, 0.0f), max(qy, 0.0f), max(qz, 0.0f)) + min(max(qx, max(qy, qz)), 0.0f);
                const float4 h = va / k1.z;
                result = select(max(h, box), sqrt(h * h + box * box), (h > 0) & (box > 0));
                break;
            }
            RAD_CASE(60): result = k[0].x; break;
            RAD_CASE(61): result = va + vb; break;
            RAD_CASE(62): result = va - vb; break;
            RAD_CASE(63): result = va * vb; break;
            RAD_CASE(64): result = va / vb; break;
            RAD_CASE(65): result = -va; break;
            RAD_CASE(66): result = fabs(va); break;
            RAD_CASE(67): RAD_CASE(76): result = va * va; break;
            RAD_CASE(68): result = sqrt(va); break;
            RAD_CASE(69): result = precise::sin(va); break;
            RAD_CASE(70): result = precise::cos(va); break;
            RAD_CASE(71): result = precise::exp(va); break;
            RAD_CASE(72): result = precise::log(va); break;
            RAD_CASE(73): result = precise::pow(va, float4(k[0].x)); break;
            RAD_CASE(74): result = precise::atan2(va, vb); break;
            RAD_CASE(75): { const float4 k0 = k[0]; result = min(max(va, k0.x), k0.y); break; }
            RAD_CASE(77): result = va + vb * vb; break;
            RAD_CASE(78): result = rad_triwave(va, k[0].x); break;
            RAD_CASE(79): result = rad_field_smooth(va, vb, k[0].x, 1.0f); break;
            RAD_CASE(80): result = rad_field_smooth(va, vb, k[0].x, -1.0f); break;
            default: result = 0; break;
            }
        }
        const uint post = (ins.x >> 12) & 0xFu;
        if (post != 0) {
            const float4 other = V[(ins.x >> 20) & (RAD_FIELD_VALUES - 1)];
            const float kk = as_type<float>(ins.y);
            const bool swap = (ins.x >> 28) & 1u;
            const float4 l = swap ? other : result, r = swap ? result : other;
            switch (post) {
            case 1: result = min(l, r); break;
            case 2: result = max(l, r); break;
            case 3: result = rad_smin(l, r, kk); break;
            case 4: result = rad_smax(l, r, kk); break;
            case 5: result = max(l, -r); break;
            case 6: result = rad_smax(l, -r, kk); break;
            case 7: result = result * kk; break;
            default: result = result - kk; break;
            }
        }
        V[d & (RAD_FIELD_VALUES - 1)] = result;
    }
    return 0;
}
#endif

// ---------------------------------------------------------------- lookup

// A point's field value bound: within the band the part's exact field
// (Lipschitz bound `lipschitz`), elsewhere a signed lower bound of the
// distance to the surface (lipschitz 1). |value| / lipschitz never exceeds
// the distance to the surface, so it is a safe sphere-tracing step.
struct RADFieldSample {
    float value;
    float lipschitz;
};

// A part's grid (its part record), read once per ray or stencil.
struct RADFieldGrid {
    float3 origin;
    float inverse_brick;
    float3 lo;        // the part's box
    float brick;
    float3 hi;
    float margin;
    uint3 dims;
    uint cells;       // word offset of the top cells
    bool valid;       // false: a mesh-only part (every point is outside the grid)
};

RADFieldGrid rad_field_grid(device const uint *w, uint part) {
    const uint record = w[4] + 24 * part;
    RADFieldGrid g;
    g.valid = w[record] == 0;
    g.cells = w[record + 1];
    g.dims = uint3(w[record + 2], w[record + 3], w[record + 4]);
    g.origin = rad_f3(w, record + 6);
    g.brick = rad_f(w, record + 9);
    g.inverse_brick = rad_f(w, record + 10);
    g.margin = rad_f(w, record + 11);
    g.lo = rad_f3(w, record + 12);
    g.hi = rad_f3(w, record + 15);
    return g;
}

// The cell a point lies in. A band cell's tape is exact within the cell grown
// by the margin, so it serves nearby points too (rad_field_sample_cached).
struct RADFieldCell {
    float3 corner;
    float size;
    uint word;   // the tape's first instruction, RAD_FIELD_FAR, or 0 outside the grid
    float bound; // band: Lipschitz bound; far: signed distance bound
};

RADFieldCell rad_field_cell(device const uint *w, thread const RADFieldGrid &g, float3 p) {
    RADFieldCell c;
    const float3 cell = floor((p - g.origin) * g.inverse_brick);
    if (!g.valid || any(cell < 0.0f) || any(cell >= float3(g.dims))) {
        c.corner = p;
        c.size = 0;
        c.word = 0;
        c.bound = 0;
        return c;
    }
    const uint3 ci = uint3(cell);
    float3 corner = g.origin + float3(ci) * g.brick;
    float size = g.brick;
    // One word per cell: far (bit 31: sign in bit 30, the bound's float bits
    // >> 1), refined (bit 30: octant cells at word (rest << 3), index x + 2y
    // + 4z), else a tape (header record at word << 2: the Lipschitz bound).
    uint word = w[g.cells + ci.x + g.dims.x * (ci.y + g.dims.y * ci.z)];
    for (uint level = 0; level < 16 && (word >> 30) == 1u; ++level) {
        size *= 0.5f;
        const float3 middle = corner + size;
        const uint3 upper = uint3(p >= middle);
        corner += float3(upper) * size;
        word = w[((word & 0x3FFFFFFFu) << 3) + (upper.x | upper.y << 1 | upper.z << 2)];
    }
    c.corner = corner;
    c.size = size;
    if (word & RAD_FIELD_FAR_CELL) {
        const float bound = as_type<float>((word & 0x3FFFFFFFu) << 1);
        c.word = RAD_FIELD_FAR;
        c.bound = (word & RAD_FIELD_REFINED) ? -bound : bound;
    } else {
        c.word = (word << 2) + 4;
        c.bound = rad_f(w, word << 2);
    }
    return c;
}

// The sample at p from cell c (which must hold p: rad_field_cell_holds).
RADFieldSample rad_field_sample_cell(device const uint *w, thread const RADFieldGrid &g, thread const RADFieldCell &c, float3 p) {
    RADFieldSample sample;
    sample.lipschitz = 1;
    if (c.word == 0) {
        // Outside the grid the distance to the part's box bounds the distance to its surface.
        sample.value = length(max(max(g.lo - p, p - g.hi), 0.0f));
        return sample;
    }
    if (c.word == RAD_FIELD_FAR) {
        // At least the cell's bound, and at least the margin plus the
        // distance to the cell's faces (the surface lies outside the cell
        // grown by the margin).
        const float3 inner3 = min(p - c.corner, c.corner + c.size - p);
        const float inner = max(min(inner3.x, min(inner3.y, inner3.z)), 0.0f);
        const float distance = max(fabs(c.bound), g.margin + inner);
        sample.value = signbit(c.bound) ? -distance : distance;
        return sample;
    }
    sample.value = rad_field_run(w, c.word, p);
    sample.lipschitz = c.bound;
    return sample;
}

// Whether cell c gives p's sample: a band cell within its cell grown by the
// margin, a far cell within itself (outside the grid: never).
inline bool rad_field_cell_holds(thread const RADFieldCell &c, thread const RADFieldGrid &g, float3 p) {
    const float grow = c.word == RAD_FIELD_FAR ? 0.0f : g.margin;
    return c.word != 0 && all(p >= c.corner - grow) && all(p <= c.corner + c.size + grow);
}

// The sample at p, looking its cell up again only when `cell` does not hold p.
RADFieldSample rad_field_sample_cached(device const uint *w, thread const RADFieldGrid &g, thread RADFieldCell &cell, float3 p) {
    if (!rad_field_cell_holds(cell, g, p)) cell = rad_field_cell(w, g, p);
    return rad_field_sample_cell(w, g, cell, p);
}

RADFieldSample rad_field_sample(device const uint *w, uint part, float3 p) {
    const RADFieldGrid g = rad_field_grid(w, part);
    const RADFieldCell c = rad_field_cell(w, g, p);
    return rad_field_sample_cell(w, g, c, p);
}

// The field value within the band, a signed distance bound elsewhere.
float rad_field_distance(device const uint *w, uint part, float3 p) {
    return rad_field_sample(w, part, p).value;
}

// The tetrahedral stencil of gradients.
constant float3 rad_field_stencil[4] = {float3(1, -1, -1), float3(-1, -1, 1), float3(-1, 1, -1), float3(1, 1, 1)};

// The unit outward normal by tetrahedral differences with step eps (zero
// when the differences vanish), reusing `cell`. With the gradient pass, a
// band cell holding the whole stencil runs its tape once for the four
// samples.
float3 rad_field_gradient_cached(device const uint *w, thread const RADFieldGrid &g, thread RADFieldCell &cell, float3 p, float eps) {
    float3 sum = 0;
#if RAD_FIELD_GRADIENT_PASS
    if (!rad_field_cell_holds(cell, g, p)) cell = rad_field_cell(w, g, p);
    if (cell.word != 0 && cell.word != RAD_FIELD_FAR && rad_field_cell_holds(cell, g, p - eps) &&
        rad_field_cell_holds(cell, g, p + eps)) {
        const float4 f = rad_field_run4(w, cell.word, p, eps);
        sum = float3(f.x - f.y - f.z + f.w, -f.x - f.y + f.z + f.w, -f.x + f.y - f.z + f.w);
    } else
#endif
    {
        for (uint i = 0; i < 4; ++i)
            sum += rad_field_stencil[i] * rad_field_sample_cached(w, g, cell, p + eps * rad_field_stencil[i]).value;
    }
    const float len = length(sum);
    return len > 0 ? sum / len : float3(0);
}

float3 rad_field_gradient(device const uint *w, uint part, float3 p, float eps) {
    const RADFieldGrid g = rad_field_grid(w, part);
    RADFieldCell cell = rad_field_cell(w, g, p);
    return rad_field_gradient_cached(w, g, cell, p, eps);
}

struct RADFieldHit {
    bool hit;
    bool started_inside;  // the ray started inside the solid: the hit is where it leaves
    float t;              // ray parameter of the hit (t1 on a miss)
    float3 position;
    float3 normal;        // unit outward normal at the hit
    uint steps;           // field samples taken (march and bisection; the normal's four are not counted)
};

// Sphere tracing of the unit-direction ray o + t d over [t0, t1]: steps of
// `relaxation` (1 to 2) times |f| / L until |f| < tolerance. An
// over-relaxed step is kept only when the unbounding spheres of its two
// samples overlap (Keinert et al. 2014), so no surface between them is
// skipped; otherwise the march returns to the previous sample and steps
// by |f| / L from there on. A sign change between samples is bisected. A
// ray that starts inside reports where it leaves (as rsdf_raycast does).
// One loop takes every sample (march, bisection and, without the gradient
// pass, the normal's four) at a single call of the interpreter: one copy of
// it in the shader, shared by a SIMD group's rays whatever phase each is in.
// With the gradient pass the normal follows the loop
// (rad_field_gradient_cached).
RADFieldHit rad_field_trace(device const uint *w, uint part, float3 o, float3 d, float t0, float t1,
                            float tolerance, uint max_steps, float eps, float relaxation) {
    const RADFieldGrid g = rad_field_grid(w, part);
    RADFieldHit result;
    result.hit = false;
    result.started_inside = false;
    result.t = t1;
    result.position = o + t1 * d;
    result.normal = float3(0);
    float t = t0;
    float3 x = o + t * d;
    RADFieldCell cell = rad_field_cell(w, g, x);
    float omega = clamp(relaxation, 1.0f, 1.99f);
    float previous_t = t, previous = 0, previous_radius = 0, step = 0, previous_lipschitz = 1;
    float a = 0, b = 0, fa = 0, m = 0;  // bisection
    float3 gradient = 0;
    uint steps = 0, phase = 0, k = 0, bisections = 0;  // phase 0 march, 1 bisection, 2 normal
    for (uint guard = 0; guard <= max_steps + 4; ++guard) {
        RADFieldSample s = rad_field_sample_cached(w, g, cell, x);
        if (phase == 2) {
            gradient += rad_field_stencil[k] * s.value;
            if (++k == 4) break;
            x = result.position + eps * rad_field_stencil[k];
            continue;
        }
        ++steps;
        bool finish = false;  // the bisection ends
        if (phase == 0) {
            if (steps == 1) {
                result.started_inside = s.value < 0;
                previous = s.value;
                previous_lipschitz = s.lipschitz;
            }
            if (steps >= max_steps) break;  // the last sample of the budget is not tested
            if (fabs(s.value) < tolerance) {
                result.hit = true;
            } else if (s.value * previous < 0) {
                a = previous_t;
                b = t;
                fa = previous;
                phase = 1;
                m = 0.5f * (a + b);
                if (m <= a || m >= b) finish = true;
                else x = o + m * d;
            } else {
                float radius = fabs(s.value) / s.lipschitz;
                if (omega > 1 && radius + previous_radius < step) {
                    // The relaxed step may have passed a surface: step back.
                    omega = 1;
                    t = previous_t;
                    radius = previous_radius;
                    s.value = previous;
                    s.lipschitz = previous_lipschitz;
                } else {
                    previous_t = t;
                    previous = s.value;
                    previous_radius = radius;
                    previous_lipschitz = s.lipschitz;
                }
                if (t >= t1) break;
                step = omega * radius;
                t = min(t + step, t1);
                x = o + t * d;
                continue;
            }
        } else {  // a bisection sample at m
            if (fabs(s.value) < tolerance) {
                a = b = m;
                finish = true;
            } else {
                if ((s.value < 0) == (fa < 0)) { a = m; fa = s.value; } else { b = m; }
                m = 0.5f * (a + b);
                if (++bisections >= 40 || steps >= max_steps || m <= a || m >= b) finish = true;
                else x = o + m * d;
            }
        }
        if (finish) {
            t = 0.5f * (a + b);
            result.hit = true;
        }
        if (result.hit) {
            result.t = t;
            result.position = o + t * d;
#if RAD_FIELD_GRADIENT_PASS
            break;
#else
            phase = 2;
            x = result.position + eps * rad_field_stencil[0];
#endif
        }
    }
    result.steps = steps;
    if (result.hit) {
#if RAD_FIELD_GRADIENT_PASS
        result.normal = rad_field_gradient_cached(w, g, cell, result.position, eps);
#else
        const float len = length(gradient);
        result.normal = len > 0 ? gradient / len : float3(0);
#endif
    }
    return result;
}
"""#

    /// The evaluator functions plus the compute kernels of FieldProgramEvaluator.
    static let library = functions + #"""

struct RADFieldRay {
    packed_float3 origin;
    float t0;
    packed_float3 direction;
    float t1;
};

struct RADFieldTraceResult {
    packed_float3 position;
    float t;
    packed_float3 normal;
    uint flags;  // 1 hit, 2 started inside; steps << 8
};

struct RADFieldTraceParameters {
    uint part;
    uint max_steps;
    float tolerance;
    float eps;
    float relaxation;
};

kernel void rad_field_distance_kernel(device const uint *program [[buffer(0)]],
                                      device const packed_float3 *points [[buffer(1)]],
                                      device float *values [[buffer(2)]],
                                      constant uint2 &part_count [[buffer(3)]],
                                      uint index [[thread_position_in_grid]]) {
    if (index >= part_count.y) return;
    values[index] = rad_field_distance(program, part_count.x, float3(points[index]));
}

kernel void rad_field_gradient_kernel(device const uint *program [[buffer(0)]],
                                      device const packed_float3 *points [[buffer(1)]],
                                      device packed_float3 *normals [[buffer(2)]],
                                      constant uint2 &part_count [[buffer(3)]],
                                      constant float &eps [[buffer(4)]],
                                      uint index [[thread_position_in_grid]]) {
    if (index >= part_count.y) return;
    normals[index] = packed_float3(rad_field_gradient(program, part_count.x, float3(points[index]), eps));
}

kernel void rad_field_trace_kernel(device const uint *program [[buffer(0)]],
                                   device const RADFieldRay *rays [[buffer(1)]],
                                   device RADFieldTraceResult *results [[buffer(2)]],
                                   constant RADFieldTraceParameters &parameters [[buffer(3)]],
                                   constant uint &count [[buffer(4)]],
                                   uint index [[thread_position_in_grid]]) {
    if (index >= count) return;
    const RADFieldRay ray = rays[index];
    const RADFieldHit hit = rad_field_trace(program, parameters.part, float3(ray.origin), float3(ray.direction), ray.t0,
                                            ray.t1, parameters.tolerance, parameters.max_steps, parameters.eps,
                                            parameters.relaxation);
    RADFieldTraceResult out;
    out.position = packed_float3(hit.position);
    out.t = hit.t;
    out.normal = packed_float3(hit.normal);
    out.flags = (hit.hit ? 1u : 0u) | (hit.started_inside ? 2u : 0u) | (min(hit.steps, 0xFFFFFFu) << 8);
    results[index] = out;
}
"""#
}
