// Custom fields (traced expressions): tape compilation and the proven slope
// bound of a field over a box (SDF-DESIGN.md, "Custom fields").
//
// The analysis runs every node of a field's tape over a box with interval
// arithmetic, carrying for each value its range, a range per gradient
// component and an upper bound of the gradient's length (`n`). Per-axis
// ranges lose the correlation between axes (the gradient of |p| ranges over
// [-1, 1]^3 near the origin, a box of length sqrt(3)), so the length bound
// follows its own rules (Cauchy-Schwarz, and Gershgorin bounds of J^T J for
// LENGTH and EVAL_AT), which give exactly 1 for length(x, y, z). A box's
// bound is the smaller of n and the length of the per-axis box.
//
// Rounding is outward: sums, products, quotients and square roots are
// rounded toward the bound they feed with error-free transformations (an
// exact result stays exact), and library functions are widened by two ulps.
#include "kernel.h"

#include <algorithm>
#include <cstring>
#include <numeric>
#include <unordered_map>

namespace rsdf {
namespace {

// ------------------------------------------------------------- directed rounding
//
// Inputs are finite; a result that overflows comes back infinite, and every
// non-finite value makes its box fail (see check_finite), so nothing below
// needs infinite arithmetic.

constexpr double kTiny = 0x1p-960;  // below this, error terms may not be exact: widen unconditionally

inline double up(double x) { return std::nextafter(x, INFINITY); }
inline double down(double x) { return std::nextafter(x, -INFINITY); }

inline double sum_error(double a, double b, double s) {  // TwoSum: a + b - s exactly
    const double bv = s - a;
    return (a - (s - bv)) + (b - bv);
}
inline double add_hi(double a, double b) {
    const double s = a + b;
    return sum_error(a, b, s) > 0 ? up(s) : s;
}
inline double add_lo(double a, double b) {
    const double s = a + b;
    return sum_error(a, b, s) < 0 ? down(s) : s;
}
inline double mul_hi(double a, double b) {
    if (a == 0 || b == 0) return 0;
    const double p = a * b;
    if (std::fabs(p) < kTiny) return up(p);
    return std::fma(a, b, -p) > 0 ? up(p) : p;
}
inline double mul_lo(double a, double b) {
    if (a == 0 || b == 0) return 0;
    const double p = a * b;
    if (std::fabs(p) < kTiny) return down(p);
    return std::fma(a, b, -p) < 0 ? down(p) : p;
}
// a / b, b != 0: the remainder a - q b is exact, and the true quotient
// exceeds q when the remainder has the sign of b.
inline double div_hi(double a, double b) {
    if (a == 0) return 0;
    const double q = a / b;
    if (std::fabs(q) < kTiny) return up(q);
    const double r = std::fma(-q, b, a);
    return (r > 0) == (b > 0) && r != 0 ? up(q) : q;
}
inline double div_lo(double a, double b) {
    if (a == 0) return 0;
    const double q = a / b;
    if (std::fabs(q) < kTiny) return down(q);
    const double r = std::fma(-q, b, a);
    return (r > 0) != (b > 0) && r != 0 ? down(q) : q;
}
inline double sqrt_hi(double a) {  // a >= 0
    const double s = std::sqrt(a);
    return std::fma(-s, s, a) > 0 ? up(s) : s;
}
inline double sqrt_lo(double a) {
    if (!(a > 0)) return 0;
    const double s = std::sqrt(a);
    return std::fma(-s, s, a) < 0 ? std::fmax(down(s), 0.0) : s;
}
inline double up2(double x) { return up(up(x)); }
inline double down2(double x) { return down(down(x)); }

// ------------------------------------------------------------- intervals

struct Iv {
    double lo, hi;
};

inline Iv point(double c) { return {c, c}; }
inline double mag(Iv a) { return std::fmax(std::fabs(a.lo), std::fabs(a.hi)); }
inline double mig(Iv a) { return a.lo > 0 ? a.lo : a.hi < 0 ? -a.hi : 0.0; }
inline bool is_zero(Iv a) { return a.lo == 0 && a.hi == 0; }
inline Iv add(Iv a, Iv b) { return {add_lo(a.lo, b.lo), add_hi(a.hi, b.hi)}; }
inline Iv sub(Iv a, Iv b) { return {add_lo(a.lo, -b.hi), add_hi(a.hi, -b.lo)}; }
inline Iv neg(Iv a) { return {-a.hi, -a.lo}; }
inline Iv hull(Iv a, Iv b) { return {std::fmin(a.lo, b.lo), std::fmax(a.hi, b.hi)}; }
inline Iv symmetric(double m) { return {-m, m}; }
inline Iv intersect(Iv a, Iv b) { return {std::fmax(a.lo, b.lo), std::fmin(a.hi, b.hi)}; }

Iv mul(Iv a, Iv b) {
    if (a.lo == a.hi && b.lo == b.hi) {
        return {mul_lo(a.lo, b.lo), mul_hi(a.lo, b.lo)};
    }
    const double lo = std::fmin(std::fmin(mul_lo(a.lo, b.lo), mul_lo(a.lo, b.hi)),
                                std::fmin(mul_lo(a.hi, b.lo), mul_lo(a.hi, b.hi)));
    const double hi = std::fmax(std::fmax(mul_hi(a.lo, b.lo), mul_hi(a.lo, b.hi)),
                                std::fmax(mul_hi(a.hi, b.lo), mul_hi(a.hi, b.hi)));
    return {lo, hi};
}

Iv sqr(Iv a) {
    if (a.lo >= 0) return {mul_lo(a.lo, a.lo), mul_hi(a.hi, a.hi)};
    if (a.hi <= 0) return {mul_lo(a.hi, a.hi), mul_hi(a.lo, a.lo)};
    const double m = mag(a);
    return {0, mul_hi(m, m)};
}

Iv div(Iv a, Iv b) {  // 0 not in b
    const double lo = std::fmin(std::fmin(div_lo(a.lo, b.lo), div_lo(a.lo, b.hi)),
                                std::fmin(div_lo(a.hi, b.lo), div_lo(a.hi, b.hi)));
    const double hi = std::fmax(std::fmax(div_hi(a.lo, b.lo), div_hi(a.lo, b.hi)),
                                std::fmax(div_hi(a.hi, b.lo), div_hi(a.hi, b.hi)));
    return {lo, hi};
}

// Is phase + 2 pi k in [a.lo - tol, a.hi + tol] for an integer k? The
// interval is narrower than 2 pi; a few candidates around the estimate cover
// the rounding of the estimate itself.
bool holds_phase(Iv a, double phase, double tol) {
    constexpr double kTwoPi = 6.283185307179586;
    const double k0 = std::floor((a.lo - phase) / kTwoPi);
    for (int d = -1; d <= 2; ++d) {
        const double x = phase + (k0 + d) * kTwoPi;
        if (x >= a.lo - tol && x <= a.hi + tol) return true;
    }
    return false;
}

// Range of sin (cosine: phase shifted) over an interval: the endpoint values
// and any extremum inside (with a tolerance for the rounding of its position).
Iv periodic_range(Iv a, bool cosine) {
    if (!(a.hi - a.lo < 6.0)) return {-1, 1};
    const double f0 = cosine ? std::cos(a.lo) : std::sin(a.lo), f1 = cosine ? std::cos(a.hi) : std::sin(a.hi);
    double lo = std::fmin(f0, f1), hi = std::fmax(f0, f1);
    const double tol = 1e-12 * std::fmax(1.0, mag(a));
    if (holds_phase(a, cosine ? 0.0 : M_PI_2, tol)) hi = 1;
    if (holds_phase(a, cosine ? M_PI : -M_PI_2, tol)) lo = -1;
    return {std::fmax(-1.0, down2(lo)), std::fmin(1.0, up2(hi))};
}

// |(a mod 2P) - P| with the floored remainder: fmod is exact and cannot
// overflow (a / 2P can), so the result lies in [0, P] (as evaluate.cpp).
inline double triwave_value(double a, double period) {
    const double twice = 2 * period;
    double rest = std::fmod(a, twice);
    if (rest < 0) rest += twice;
    return std::fabs(rest - period);
}

// a^e over an interval whose domain was checked: integer powers of any
// sign (negative ones away from 0), other powers of nonnegative values.
Iv power_range(Iv a, double e) {
    if (e == 0) return {1, 1};
    if (e == 1) return a;
    if (e == 2) return sqr(a);
    const bool integer = std::floor(e) == e;
    const bool even = integer && std::fmod(e, 2.0) == 0;
    double lo, hi;
    if (even && e > 0) {
        const double small = mig(a), large = mag(a);
        lo = std::pow(small, e);
        hi = std::pow(large, e);
    } else {  // monotone over the interval (0 excluded where it matters)
        const double f0 = std::pow(a.lo, e), f1 = std::pow(a.hi, e);
        lo = std::fmin(f0, f1);
        hi = std::fmax(f0, f1);
    }
    lo = down2(lo);
    if (even || !integer) lo = std::fmax(lo, 0.0);
    return {lo, up2(hi)};
}

// ------------------------------------------------------------- values over a box

struct Val {
    Iv v;          // value range
    Iv g[3];       // gradient component ranges
    double n;      // upper bound of the gradient's length
    uint32_t wrap; // 0, or k: atan2 across its cut, defined only modulo 2 pi k over the box
};

enum Problem : uint8_t {
    PROBLEM_NONE,
    PROBLEM_DIVISOR,
    PROBLEM_SQRT_NEGATIVE,
    PROBLEM_SQRT_SLOPE,
    PROBLEM_LOG,
    PROBLEM_POW_NEGATIVE,
    PROBLEM_POW_ZERO,
    PROBLEM_POW_SLOPE,
    PROBLEM_ATAN2_POLE,
    PROBLEM_ATAN2_CUT,
    PROBLEM_OVERFLOW,
    PROBLEM_WARP_RANGE,
};

const char *problem_text(uint8_t problem) {
    switch (problem) {
    case PROBLEM_DIVISOR: return "the divisor may be 0";
    case PROBLEM_SQRT_NEGATIVE: return "the argument of sqrt may be negative";
    case PROBLEM_SQRT_SLOPE:
        return "the argument of sqrt may be 0 where it changes, so its slope is unbounded (use length() for "
               "distances)";
    case PROBLEM_LOG: return "the argument of log may be 0 or negative";
    case PROBLEM_POW_NEGATIVE: return "a fractional power of a value that may be negative";
    case PROBLEM_POW_ZERO: return "a negative power of a value that may be 0";
    case PROBLEM_POW_SLOPE: return "a power below 1 of a value that may be 0 where it changes (unbounded slope)";
    case PROBLEM_ATAN2_POLE: return "atan2 may reach its pole (both arguments 0), where its slope is unbounded";
    case PROBLEM_ATAN2_CUT:
        return "atan2 jumps by 2 pi where its first argument changes sign with a negative second argument; "
               "use the angle only through sin or cos (of whole multiples), or keep that half-plane out of the "
               "bounds";
    case PROBLEM_OVERFLOW: return "the value or its slope overflows";
    case PROBLEM_WARP_RANGE: return "the evaluation point of EVAL_AT may exceed 1e150 in magnitude";
    default: return "no finite bound";
    }
}

inline bool finite_iv(Iv a) { return std::isfinite(a.lo) && std::isfinite(a.hi); }

bool finite_val(const Val &v) {
    return finite_iv(v.v) && finite_iv(v.g[0]) && finite_iv(v.g[1]) && finite_iv(v.g[2]) && std::isfinite(v.n);
}

inline double norm_hi(const Iv g[3]) {
    double sum = 0;
    for (int a = 0; a < 3; ++a) {
        const double m = mag(g[a]);
        sum = add_hi(sum, mul_hi(m, m));
    }
    return sqrt_hi(sum);
}

// Gershgorin bound of the largest eigenvalue of J^T J, J's rows the gradient
// ranges of `rows`: |J^T u| <= sqrt(that) for a unit vector u.
template <class Row>
double gram_bound(size_t count, const Row &row) {
    Iv m[3][3];
    for (auto &line : m)
        for (Iv &entry : line) entry = point(0);
    for (size_t i = 0; i < count; ++i) {
        const Iv *g = row(i);
        for (int a = 0; a < 3; ++a) {
            m[a][a] = add(m[a][a], sqr(g[a]));
            for (int b = a + 1; b < 3; ++b) m[a][b] = add(m[a][b], mul(g[a], g[b]));
        }
    }
    double bound = 0;
    for (int a = 0; a < 3; ++a) {
        double sum = 0;
        for (int b = 0; b < 3; ++b) sum = add_hi(sum, mag(a <= b ? m[a][b] : m[b][a]));
        bound = std::fmax(bound, sum);
    }
    return bound;
}

inline uint32_t combine_wrap(uint32_t a, uint32_t b) {
    if (!a) return b;
    if (!b) return a;
    return std::gcd(a, b);
}

// The value s(c) +- L r of a shape over the box [lo, hi] (c its centre, r
// its half diagonal, L the shape's Lipschitz bound over the box), plus a
// margin for the rounding of s(c).
Iv shape_range(const rsdf_graph &graph, uint32_t shape, const double *lo, const double *hi, double L) {
    double c[3], r2 = 0, scale = 0;
    for (int a = 0; a < 3; ++a) {
        c[a] = lo[a] + 0.5 * (hi[a] - lo[a]);
        const double half = std::fmax(add_hi(hi[a], -c[a]), add_hi(c[a], -lo[a]));
        r2 = add_hi(r2, mul_hi(half, half));
        scale = std::fmax(scale, std::fabs(c[a]));
    }
    const double r = sqrt_hi(r2);
    double s = 0;
    eval_batch(graph, shape, 1, &c[0], &c[1], &c[2], &s);
    const double spread = add_hi(mul_hi(L, r), 1e-12 * (std::fabs(s) + scale + r));
    return {add_lo(s, -spread), add_hi(s, spread)};
}

struct BoxResult {
    double bound;     // INFINITY when the box failed
    Iv value;
    uint32_t node;    // failing node
    uint8_t problem;
};

// Runs the tape over one box. `vals` is scratch (3 coordinates + slots).
BoxResult evaluate_box(const rsdf_graph &graph, const FieldTape &tape, const double *lo, const double *hi,
                       std::vector<Val> &vals) {
    vals.resize(3 + tape.slots);
    for (int a = 0; a < 3; ++a) {
        Val &c = vals[a];
        c.v = {lo[a], hi[a]};
        for (int b = 0; b < 3; ++b) c.g[b] = point(a == b ? 1.0 : 0.0);
        c.n = 1;
        c.wrap = 0;
    }
    const uint32_t *args = tape.args.data();
    for (const FieldTape::Instr &ins : tape.code) {
        Val r;
        r.wrap = 0;
        const uint32_t *ops = args + ins.first;
        const Val &A = ins.count ? vals[ops[0]] : vals[0];  // first operand (unused by leaves)
        auto fail = [&](uint8_t problem) { return BoxResult{INFINITY, {-INFINITY, INFINITY}, ins.node, problem}; };
        // Operations that cannot pass an angle across atan2's cut.
        auto wrapped = [&](uint32_t count) {
            for (uint32_t i = 0; i < count; ++i)
                if (vals[ops[i]].wrap) return true;
            return false;
        };
        switch (ins.op) {
        case FieldTape::kShape: {
            const double L = shape_lipschitz(graph, ins.node, lo, hi);
            r.v = shape_range(graph, ins.node, lo, hi, L);
            for (Iv &g : r.g) g = symmetric(L);
            r.n = L;
            break;
        }
        case OP_CONST:
            r.v = point(ins.p0);
            for (Iv &g : r.g) g = point(0);
            r.n = 0;
            break;
        case OP_ADD: case OP_SUB: {
            const Val &B = vals[ops[1]];
            const bool plus = ins.op == OP_ADD;
            r.v = plus ? add(A.v, B.v) : sub(A.v, B.v);
            for (int a = 0; a < 3; ++a) r.g[a] = plus ? add(A.g[a], B.g[a]) : sub(A.g[a], B.g[a]);
            r.n = add_hi(A.n, B.n);
            r.wrap = combine_wrap(A.wrap, B.wrap);
            break;
        }
        case OP_MUL: {
            const Val &B = vals[ops[1]];
            if (A.wrap || B.wrap) {
                // An angle times a whole constant m stays defined modulo 2 pi k |m|.
                const Val &angle = A.wrap ? A : B, &factor = A.wrap ? B : A;
                const bool constant = factor.v.lo == factor.v.hi && is_zero(factor.g[0]) && is_zero(factor.g[1]) &&
                                      is_zero(factor.g[2]) && factor.wrap == 0;
                if (!constant || std::floor(factor.v.lo) != factor.v.lo || std::fabs(factor.v.lo) > 1e6)
                    return fail(PROBLEM_ATAN2_CUT);
                const uint32_t m = static_cast<uint32_t>(std::fabs(factor.v.lo));
                if (m != 0 && angle.wrap > 0x7fffffffu / m) return fail(PROBLEM_ATAN2_CUT);
                r.wrap = angle.wrap * m;
            }
            if (ops[0] == ops[1]) {  // a * a
                r.v = sqr(A.v);
                const Iv twice = {2 * A.v.lo, 2 * A.v.hi};
                for (int a = 0; a < 3; ++a) r.g[a] = mul(twice, A.g[a]);
                r.n = mul_hi(2 * mag(A.v), A.n);
            } else {
                r.v = mul(A.v, B.v);
                for (int a = 0; a < 3; ++a) r.g[a] = add(mul(A.v, B.g[a]), mul(B.v, A.g[a]));
                r.n = add_hi(mul_hi(mag(A.v), B.n), mul_hi(mag(B.v), A.n));
            }
            break;
        }
        case OP_DIV: {
            const Val &B = vals[ops[1]];
            if (wrapped(2)) return fail(PROBLEM_ATAN2_CUT);
            const double below = mig(B.v);
            if (!(below > 0)) return fail(PROBLEM_DIVISOR);
            r.v = div(A.v, B.v);
            for (int a = 0; a < 3; ++a) r.g[a] = div(sub(A.g[a], mul(r.v, B.g[a])), B.v);
            r.n = div_hi(add_hi(A.n, mul_hi(mag(r.v), B.n)), below);
            break;
        }
        case OP_NEG:
            r.v = neg(A.v);
            for (int a = 0; a < 3; ++a) r.g[a] = neg(A.g[a]);
            r.n = A.n;
            r.wrap = A.wrap;
            break;
        case OP_ABS:
            if (A.wrap) return fail(PROBLEM_ATAN2_CUT);
            if (A.v.lo >= 0) {
                r = A;
            } else if (A.v.hi <= 0) {
                r.v = neg(A.v);
                for (int a = 0; a < 3; ++a) r.g[a] = neg(A.g[a]);
                r.n = A.n;
            } else {
                r.v = {0, mag(A.v)};
                for (int a = 0; a < 3; ++a) r.g[a] = symmetric(mag(A.g[a]));
                r.n = A.n;
            }
            break;
        case OP_SQUARE: {
            if (A.wrap) return fail(PROBLEM_ATAN2_CUT);
            r.v = sqr(A.v);
            const Iv twice = {2 * A.v.lo, 2 * A.v.hi};
            for (int a = 0; a < 3; ++a) r.g[a] = mul(twice, A.g[a]);
            r.n = mul_hi(2 * mag(A.v), A.n);
            break;
        }
        case OP_SQRT: {
            if (A.wrap) return fail(PROBLEM_ATAN2_CUT);
            if (A.v.lo < 0) return fail(PROBLEM_SQRT_NEGATIVE);
            r.v = {sqrt_lo(A.v.lo), sqrt_hi(A.v.hi)};
            if (A.v.lo > 0) {
                const Iv twice = {2 * r.v.lo, 2 * r.v.hi};
                for (int a = 0; a < 3; ++a) r.g[a] = div(A.g[a], twice);
                r.n = div_hi(A.n, twice.lo);
            } else if (A.n == 0) {
                for (Iv &g : r.g) g = point(0);
                r.n = 0;
            } else {
                return fail(PROBLEM_SQRT_SLOPE);
            }
            break;
        }
        case OP_SIN: case OP_COS: {
            // Continuous even across atan2's cut: whole multiples of 2 pi drop out.
            const bool cosine = ins.op == OP_COS;
            r.v = periodic_range(A.v, cosine);
            Iv slope = periodic_range(A.v, !cosine);
            if (cosine) slope = neg(slope);
            for (int a = 0; a < 3; ++a) r.g[a] = mul(slope, A.g[a]);
            r.n = mul_hi(mag(slope), A.n);
            break;
        }
        case OP_EXP:
            if (A.wrap) return fail(PROBLEM_ATAN2_CUT);
            r.v = {std::fmax(down2(std::exp(A.v.lo)), 0.0), up2(std::exp(A.v.hi))};
            for (int a = 0; a < 3; ++a) r.g[a] = mul(r.v, A.g[a]);
            r.n = mul_hi(r.v.hi, A.n);
            break;
        case OP_LOG:
            if (A.wrap) return fail(PROBLEM_ATAN2_CUT);
            if (!(A.v.lo > 0)) return fail(PROBLEM_LOG);
            r.v = {down2(std::log(A.v.lo)), up2(std::log(A.v.hi))};
            for (int a = 0; a < 3; ++a) r.g[a] = div(A.g[a], A.v);
            r.n = div_hi(A.n, A.v.lo);
            break;
        case OP_POWC: {
            if (A.wrap) return fail(PROBLEM_ATAN2_CUT);
            const double e = ins.p0;
            const bool integer = std::floor(e) == e;
            const bool flat = A.n == 0;
            if (integer) {
                if (e < 0 && mig(A.v) == 0) return fail(PROBLEM_POW_ZERO);
            } else {
                if (A.v.lo < 0) return fail(PROBLEM_POW_NEGATIVE);
                if (e < 0 && A.v.lo == 0) return fail(PROBLEM_POW_ZERO);
                if (e < 1 && A.v.lo == 0 && !flat) return fail(PROBLEM_POW_SLOPE);
            }
            r.v = power_range(A.v, e);
            if (e == 0 || flat) {
                for (Iv &g : r.g) g = point(0);
                r.n = 0;
            } else {
                const Iv base = power_range(A.v, e - 1);
                const Iv slope = mul(point(e), base);
                for (int a = 0; a < 3; ++a) r.g[a] = mul(slope, A.g[a]);
                r.n = mul_hi(mag(slope), A.n);
            }
            break;
        }
        case OP_ATAN2: {
            const Val &X = vals[ops[1]];  // A is y
            if (wrapped(2)) return fail(PROBLEM_ATAN2_CUT);
            const double mx = mig(X.v), my = mig(A.v);
            const double r2min = add_lo(mul_lo(mx, mx), mul_lo(my, my));
            if (!(r2min > 0)) return fail(PROBLEM_ATAN2_POLE);
            const double pi_hi = up(M_PI);
            if (A.v.lo <= 0 && A.v.hi >= 0 && X.v.lo < 0) {
                r.v = {-pi_hi, pi_hi};
                r.wrap = 1;
            } else {  // a box away from the origin and the cut: the extreme angles are at corners
                double lo_angle = INFINITY, hi_angle = -INFINITY;
                for (double yy : {A.v.lo, A.v.hi})
                    for (double xx : {X.v.lo, X.v.hi}) {
                        const double angle = std::atan2(yy, xx);
                        lo_angle = std::fmin(lo_angle, angle);
                        hi_angle = std::fmax(hi_angle, angle);
                    }
                r.v = {std::fmax(down2(lo_angle), -pi_hi), std::fmin(up2(hi_angle), pi_hi)};
            }
            const Iv r2 = add(sqr(X.v), sqr(A.v));
            for (int a = 0; a < 3; ++a) r.g[a] = div(sub(mul(X.v, A.g[a]), mul(A.v, X.g[a])), r2);
            // |x grad y - y grad x| <= |(x, y)| sqrt(|grad x|^2 + |grad y|^2)
            r.n = div_hi(sqrt_hi(add_hi(mul_hi(X.n, X.n), mul_hi(A.n, A.n))), sqrt_lo(r2min));
            break;
        }
        case OP_MIN: case OP_MAX: {
            const bool lowest = ins.op == OP_MIN;
            double edge = lowest ? INFINITY : -INFINITY, other = edge;
            for (uint32_t i = 0; i < ins.count; ++i) {
                const Iv &v = vals[ops[i]].v;
                if (lowest) {
                    edge = std::fmin(edge, v.hi);   // min of the upper ends
                    other = std::fmin(other, v.lo);
                } else {
                    edge = std::fmax(edge, v.lo);   // max of the lower ends
                    other = std::fmax(other, v.hi);
                }
            }
            r.v = lowest ? Iv{other, edge} : Iv{edge, other};
            bool any = false;
            r.n = 0;
            for (uint32_t i = 0; i < ins.count; ++i) {
                const Val &c = vals[ops[i]];
                // A child can be the extreme somewhere in the box only if its range reaches past the edge.
                if (lowest ? !(c.v.lo <= edge) : !(c.v.hi >= edge)) continue;
                if (c.wrap) return fail(PROBLEM_ATAN2_CUT);
                for (int a = 0; a < 3; ++a) r.g[a] = any ? hull(r.g[a], c.g[a]) : c.g[a];
                r.n = std::fmax(r.n, c.n);
                any = true;
            }
            break;
        }
        case OP_CLAMP: {
            const double low = ins.p0, high = ins.p1;
            if (low == high) {  // constant
                r.v = point(low);
                for (Iv &g : r.g) g = point(0);
                r.n = 0;
                break;
            }
            if (A.wrap) return fail(PROBLEM_ATAN2_CUT);
            r.v = {std::fmin(std::fmax(A.v.lo, low), high), std::fmin(std::fmax(A.v.hi, low), high)};
            if (A.v.hi <= low || A.v.lo >= high) {
                for (Iv &g : r.g) g = point(0);
                r.n = 0;
            } else if (A.v.lo >= low && A.v.hi <= high) {
                for (int a = 0; a < 3; ++a) r.g[a] = A.g[a];
                r.n = A.n;
            } else {
                for (int a = 0; a < 3; ++a) r.g[a] = hull(A.g[a], point(0));
                r.n = A.n;
            }
            break;
        }
        case OP_LENGTH: {
            if (wrapped(ins.count)) return fail(PROBLEM_ATAN2_CUT);
            Iv sum = point(0);
            double norms = 0, columns[3] = {0, 0, 0};
            for (uint32_t i = 0; i < ins.count; ++i) {
                const Val &c = vals[ops[i]];
                sum = add(sum, sqr(c.v));
                norms = add_hi(norms, mul_hi(c.n, c.n));
                for (int a = 0; a < 3; ++a) {
                    const double m = mag(c.g[a]);
                    columns[a] = add_hi(columns[a], mul_hi(m, m));
                }
            }
            r.v = {sqrt_lo(sum.lo), sqrt_hi(sum.hi)};
            // grad = sum_i u_i grad c_i with u = c / |c| a unit vector:
            // |d/dx_a| <= sqrt(sum_i (dc_i/dx_a)^2) by Cauchy-Schwarz.
            for (int a = 0; a < 3; ++a) r.g[a] = symmetric(sqrt_hi(columns[a]));
            if (r.v.lo > 0) {
                for (int a = 0; a < 3; ++a) {
                    Iv total = point(0);
                    for (uint32_t i = 0; i < ins.count; ++i) {
                        const Val &c = vals[ops[i]];
                        const Iv u = intersect(div(c.v, r.v), Iv{-1, 1});
                        total = add(total, mul(u, c.g[a]));
                    }
                    r.g[a] = intersect(r.g[a], total);
                }
            }
            const double gram = gram_bound(ins.count, [&](size_t i) { return vals[ops[i]].g; });
            r.n = std::fmin(sqrt_hi(norms), sqrt_hi(gram));
            break;
        }
        case OP_TRIWAVE: {
            if (A.wrap) return fail(PROBLEM_ATAN2_CUT);
            // |u - P| for u = a mod 2P: maxima P at even multiples of P,
            // minima 0 at odd ones, slope +-1 between.
            const double period = ins.p0, twice = 2 * period;
            r.n = A.n;
            const double width = add_hi(A.v.hi, -A.v.lo);
            if (!(width < twice)) {
                r.v = {0, period};
                for (int a = 0; a < 3; ++a) r.g[a] = symmetric(mag(A.g[a]));
                break;
            }
            // lo's position u0 in [0, 2P] (fmod is exact; adding 2P rounds),
            // and the interval as [u0, u0 + width] with kinks at P, 2P, 3P.
            double u0 = std::fmod(A.v.lo, twice);
            if (u0 < 0) u0 += twice;
            const double u1 = add_hi(u0, width);
            const double tol = 1e-15 * (twice + width);
            auto holds = [&](double kink) { return kink >= u0 - tol && kink <= u1 + tol; };
            const bool minimum = holds(period) || holds(3 * period);
            const bool maximum = u0 <= tol || holds(twice);
            const double t0 = triwave_value(A.v.lo, period), t1 = triwave_value(A.v.hi, period);
            double low = std::fmin(t0, t1), high = std::fmax(t0, t1);
            if (minimum) low = 0;
            if (maximum) high = period;
            if (minimum || maximum) {
                for (int a = 0; a < 3; ++a) r.g[a] = symmetric(mag(A.g[a]));
            } else {  // one linear piece: descending on [0, P], ascending on [P, 2P]
                const bool descending = u0 < period;
                for (int a = 0; a < 3; ++a) r.g[a] = descending ? neg(A.g[a]) : A.g[a];
            }
            // The exact function and its evaluation both lie in [0, P].
            const double margin = 1e-15 * twice;
            r.v = {std::fmax(0.0, low - margin), std::fmin(period, high + margin)};
            break;
        }
        case OP_SMIN: case OP_SMAX: {
            // min(a, b) - h^2 k / 4, h = max(k - |a - b|, 0) / k (SMAX mirrored):
            // its gradient is a convex combination of the operands' (weights
            // 1 - h/2 and h/2), so its length is at most the larger of theirs.
            const Val &B = vals[ops[1]];
            if (wrapped(2)) return fail(PROBLEM_ATAN2_CUT);
            const bool lowest = ins.op == OP_SMIN;
            const double k = ins.p0;
            const Iv d = sub(A.v, B.v);
            if (d.lo >= k || d.hi <= -k) {  // |a - b| >= k throughout: exactly one operand
                const bool first = lowest ? d.hi <= -k : d.lo >= k;
                r = first ? A : B;
                break;
            }
            const double quarter = up(0.25 * k);
            r.v = lowest ? Iv{add_lo(std::fmin(A.v.lo, B.v.lo), -quarter), std::fmin(A.v.hi, B.v.hi)}
                         : Iv{std::fmax(A.v.lo, B.v.lo), add_hi(std::fmax(A.v.hi, B.v.hi), quarter)};
            for (int a = 0; a < 3; ++a) r.g[a] = hull(A.g[a], B.g[a]);
            r.n = std::fmax(A.n, B.n);
            break;
        }
        case OP_EVAL_AT: {
            if (wrapped(3)) return fail(PROBLEM_ATAN2_CUT);
            double image_lo[3], image_hi[3], norms = 0;
            Iv const *rows[3];
            for (int j = 0; j < 3; ++j) {
                const Val &c = vals[ops[j]];
                if (!(mag(c.v) <= 1e150)) return fail(PROBLEM_WARP_RANGE);
                image_lo[j] = c.v.lo;
                image_hi[j] = c.v.hi;
                rows[j] = c.g;
                norms = add_hi(norms, mul_hi(c.n, c.n));
            }
            const double L = shape_lipschitz(graph, ins.shape, image_lo, image_hi);
            r.v = shape_range(graph, ins.shape, image_lo, image_hi, L);
            // d/dx_a s(F(p)) = grad s . column a of J_F: at most L |column a|.
            for (int a = 0; a < 3; ++a) {
                double column = 0;
                for (int j = 0; j < 3; ++j) {
                    const double m = mag(rows[j][a]);
                    column = add_hi(column, mul_hi(m, m));
                }
                r.g[a] = symmetric(mul_hi(L, sqrt_hi(column)));
            }
            const double gram = gram_bound(3, [&](size_t j) { return rows[j]; });
            r.n = mul_hi(L, std::fmin(sqrt_hi(norms), sqrt_hi(gram)));
            break;
        }
        default: throw Error(format("internal error: field opcode %u in a tape", ins.op));
        }
        r.n = std::fmin(r.n, norm_hi(r.g));
        if (!finite_val(r)) return fail(PROBLEM_OVERFLOW);
        vals[ins.dst] = r;
    }
    const Val &result = vals[tape.result];
    if (result.wrap) return BoxResult{INFINITY, {-INFINITY, INFINITY}, tape.root, PROBLEM_ATAN2_CUT};
    return BoxResult{std::fmin(result.n, norm_hi(result.g)), result.v, tape.root, PROBLEM_NONE};
}

struct Leaf {
    double lo[3], hi[3];
    BoxResult result;
    uint64_t order;  // creation order (deterministic tie-break)
};

std::string region_text(const double *lo, const double *hi) {
    return format("[%.6g, %.6g] x [%.6g, %.6g] x [%.6g, %.6g]", lo[0], hi[0], lo[1], hi[1], lo[2], hi[2]);
}

}  // namespace

// ------------------------------------------------------------- analysis

FieldBound analyze_field(const rsdf_graph &graph, const FieldTape &tape, const double *lo, const double *hi,
                         uint64_t max_boxes) {
    // Best-first branch and bound: split the box with the largest bound (the
    // earliest on ties) into octants. Every box's centre also gets a bound of
    // its own (a degenerate box, which is the gradient's length there for
    // smooth fields), and the largest of those is a slope the field attains,
    // so splitting stops once the largest box bound is within 1% of it: no
    // further splitting can lower the bound by more than 1%. A budget of
    // boxes caps the work; the bound is the largest leaf bound either way.
    FieldBound out;
    std::vector<Val> vals;
    std::vector<Leaf> leaves;
    std::vector<uint8_t> live;
    double attained = 0;
    auto add_leaf = [&](const double *box_lo, const double *box_hi) {
        Leaf leaf;
        double centre[3];
        for (int a = 0; a < 3; ++a) {
            leaf.lo[a] = box_lo[a];
            leaf.hi[a] = box_hi[a];
            centre[a] = box_lo[a] + 0.5 * (box_hi[a] - box_lo[a]);
        }
        leaf.result = evaluate_box(graph, tape, leaf.lo, leaf.hi, vals);
        leaf.order = leaves.size();
        const BoxResult at_centre = evaluate_box(graph, tape, centre, centre, vals);
        if (std::isfinite(at_centre.bound)) attained = std::fmax(attained, at_centre.bound);
        leaves.push_back(leaf);
        live.push_back(1);
    };
    // A max-heap of live leaves by bound, then by creation order.
    const auto later = [&](size_t a, size_t b) {
        const double ba = leaves[a].result.bound, bb = leaves[b].result.bound;
        return ba != bb ? ba < bb : leaves[a].order > leaves[b].order;
    };
    std::vector<size_t> heap;
    add_leaf(lo, hi);
    heap.push_back(0);
    uint64_t boxes = 1;
    while (!heap.empty()) {
        const Leaf &best = leaves[heap.front()];
        const double largest = best.result.bound;
        if (!(largest > 0)) break;                                         // a constant field
        if (std::isfinite(largest) && largest <= attained * 1.01) break;  // within 1% of an attained slope
        int parts[3], count = 1;
        double middle[3];
        for (int a = 0; a < 3; ++a) {
            middle[a] = best.lo[a] + 0.5 * (best.hi[a] - best.lo[a]);
            parts[a] = middle[a] > best.lo[a] && middle[a] < best.hi[a] ? 2 : 1;
            count *= parts[a];
        }
        if (count == 1 || boxes + static_cast<uint64_t>(count) > max_boxes) break;
        const size_t parent = heap.front();
        std::pop_heap(heap.begin(), heap.end(), later);
        heap.pop_back();
        live[parent] = 0;
        const Leaf box = leaves[parent];
        for (int i = 0; i < parts[0]; ++i)
            for (int j = 0; j < parts[1]; ++j)
                for (int k = 0; k < parts[2]; ++k) {
                    const int pick[3] = {i, j, k};
                    double child_lo[3], child_hi[3];
                    for (int a = 0; a < 3; ++a) {
                        const bool split = parts[a] == 2;
                        child_lo[a] = split && pick[a] == 1 ? middle[a] : box.lo[a];
                        child_hi[a] = split && pick[a] == 0 ? middle[a] : box.hi[a];
                    }
                    add_leaf(child_lo, child_hi);
                    heap.push_back(leaves.size() - 1);
                    std::push_heap(heap.begin(), heap.end(), later);
                }
        boxes += static_cast<uint64_t>(count);
    }
    double current = 0;
    for (size_t i = 0; i < leaves.size(); ++i)
        if (live[i]) current = std::fmax(current, leaves[i].result.bound);
    out.boxes = boxes;
    out.value_lo = INFINITY;
    out.value_hi = -INFINITY;
    for (size_t i = 0; i < leaves.size(); ++i) {
        if (!live[i]) continue;
        out.value_lo = std::fmin(out.value_lo, leaves[i].result.value.lo);
        out.value_hi = std::fmax(out.value_hi, leaves[i].result.value.hi);
    }
    if (std::isfinite(current)) {
        out.ok = true;
        out.bound = current;
        return out;
    }
    // No finite bound: name the problem of the smallest failed box (earliest
    // on ties) and the region covered by the boxes failing the same way.
    const Leaf *worst = nullptr;
    auto volume = [](const Leaf &leaf) {
        double v = 1;
        for (int a = 0; a < 3; ++a) v *= leaf.hi[a] - leaf.lo[a];
        return v;
    };
    for (size_t i = 0; i < leaves.size(); ++i) {
        const Leaf &leaf = leaves[i];
        if (!live[i] || std::isfinite(leaf.result.bound)) continue;
        if (!worst || volume(leaf) < volume(*worst) || (volume(leaf) == volume(*worst) && leaf.order < worst->order))
            worst = &leaf;
    }
    double region_lo[3] = {INFINITY, INFINITY, INFINITY}, region_hi[3] = {-INFINITY, -INFINITY, -INFINITY};
    for (size_t i = 0; i < leaves.size(); ++i) {
        const Leaf &leaf = leaves[i];
        if (!live[i] || std::isfinite(leaf.result.bound) || leaf.result.node != worst->result.node ||
            leaf.result.problem != worst->result.problem)
            continue;
        for (int a = 0; a < 3; ++a) {
            region_lo[a] = std::fmin(region_lo[a], leaf.lo[a]);
            region_hi[a] = std::fmax(region_hi[a], leaf.hi[a]);
        }
    }
    out.ok = false;
    out.bound = INFINITY;
    out.failed_node = worst->result.node;
    std::copy(region_lo, region_lo + 3, out.failed_lo);
    std::copy(region_hi, region_hi + 3, out.failed_hi);
    out.problem = format("node %u (%s): ", worst->result.node, op_name(graph.nodes[worst->result.node].op)) +
                  problem_text(worst->result.problem) + " in the region " + region_text(region_lo, region_hi) +
                  format(" (for example the box %s)", region_text(worst->lo, worst->hi).c_str());
    return out;
}

// ------------------------------------------------------------- regional slope bounds
//
// A shape's stepping bound is global except below a TWIST of an inexact
// child or a LOFT of inexact profiles (Node::regional): the twist map
// stretches by sigma(rate r) = (w + sqrt(w^2 + 4)) / 2 (w = rate r) at radius
// r, and a loft's z slope is |b - a| / h, both of which grow with the
// distance from the node's box. shape_lipschitz walks such a subtree with
// the box mapped into each node's coordinates; the bound is the largest
// product of the factors along a path times the bound where the path leaves
// the regional nodes.

namespace {

struct Region {
    uint32_t node;
    double factor;
    double lo[3], hi[3];
    double radius;  // a bound of the points' distance from the z axis (below a TWIST), or infinity
};

// Widens a mapped box for the rounding of its computation.
void pad_box(double *lo, double *hi) {
    for (int a = 0; a < 3; ++a) {
        const double pad = 1e-12 * (std::fabs(lo[a]) + std::fabs(hi[a])) + kTiny;
        lo[a] -= pad;
        hi[a] += pad;
    }
}

// The box around the image M p + t of [lo, hi] (M row-major).
void affine_box(const double m[9], const double t[3], const double *lo, const double *hi, double *out_lo,
                double *out_hi) {
    double centre[3], half[3];
    for (int a = 0; a < 3; ++a) {
        centre[a] = lo[a] + 0.5 * (hi[a] - lo[a]);
        half[a] = std::fmax(add_hi(hi[a], -centre[a]), add_hi(centre[a], -lo[a]));
    }
    for (int i = 0; i < 3; ++i) {
        double c = t[i], h = 0;
        for (int j = 0; j < 3; ++j) {
            c += m[3 * i + j] * centre[j];
            h = add_hi(h, mul_hi(std::fabs(m[3 * i + j]), half[j]));
        }
        out_lo[i] = c - h;
        out_hi[i] = c + h;
    }
    pad_box(out_lo, out_hi);
}

// LOFT over a box: ((1 - t) a + t b) c0 with t = clamp((z - z0) / h), then
// a slab. The blend's xy slope is at most max(L_a, L_b) c0 and its z slope
// |b - a| c0 / h. Inside the profiles' box |b - a| <= c1 (loft_spread() in
// graph.cpp: max(L_a, L_b) D for profiles shown nonempty); it grows by at
// most L_a + L_b per unit of distance from that box, and from its value at
// the region's centre.
double loft_lipschitz(const rsdf_graph &graph, const Node &node, const double *lo, const double *hi) {
    const double *p = graph.param_values(node);
    const uint32_t *kids = graph.child_ids(node);
    const Node &a = graph.nodes[kids[0]], &b = graph.nodes[kids[1]];
    const double la = a.step_lipschitz, lb = b.step_lipschitz, lab = std::fmax(la, lb), lsum = add_hi(la, lb);
    double gap2 = 0, centre[2], half2 = 0;
    for (int i = 0; i < 2; ++i) {
        const double low = std::min({a.lo[i], a.hi[i], b.lo[i], b.hi[i]});
        const double high = std::max({a.lo[i], a.hi[i], b.lo[i], b.hi[i]});
        const double gap = std::fmax(0.0, std::fmax(add_hi(low, -lo[i]), add_hi(hi[i], -high)));
        gap2 = add_hi(gap2, mul_hi(gap, gap));
        centre[i] = lo[i] + 0.5 * (hi[i] - lo[i]);
        const double half = std::fmax(add_hi(hi[i], -centre[i]), add_hi(centre[i], -lo[i]));
        half2 = add_hi(half2, mul_hi(half, half));
    }
    double spread = add_hi(node.c1, mul_hi(lsum, sqrt_hi(gap2)));
    double va = 0, vb = 0;
    eval_batch(graph, kids[0], 1, &centre[0], &centre[1], nullptr, &va);
    eval_batch(graph, kids[1], 1, &centre[0], &centre[1], nullptr, &vb);
    const double radius = sqrt_hi(half2);
    const double at_centre = add_hi(add_hi(std::fabs(vb - va), mul_hi(lsum, radius)),
                                    1e-12 * (std::fabs(va) + std::fabs(vb) + std::fabs(centre[0]) +
                                             std::fabs(centre[1]) + radius));
    if (std::isfinite(at_centre)) spread = std::fmin(spread, at_centre);
    const double slope = div_hi(spread, add_lo(p[1], -p[0]));
    const double blend = mul_hi(up2(node.c0), sqrt_hi(add_hi(mul_hi(lab, lab), mul_hi(slope, slope))));
    return std::fmax(1.0, blend);  // the slab
}

}  // namespace

double shape_lipschitz(const rsdf_graph &graph, uint32_t root, const double *lo, const double *hi) {
    const Node &top = graph.nodes[root];
    if (!top.regional) return top.step_lipschitz;
    std::vector<Region> stack(1);
    stack[0].node = root;
    stack[0].factor = 1;
    stack[0].radius = INFINITY;
    std::copy(lo, lo + 3, stack[0].lo);
    std::copy(hi, hi + 3, stack[0].hi);
    double bound = 0;
    while (!stack.empty()) {
        const Region region = stack.back();
        stack.pop_back();
        const Node &node = graph.nodes[region.node];
        if (!node.regional) {
            bound = std::fmax(bound, mul_hi(region.factor, node.step_lipschitz));
            continue;
        }
        const double *p = graph.param_values(node);
        const uint32_t *kids = graph.child_ids(node);
        Region next = region;
        next.node = kids[0];
        next.radius = INFINITY;  // kept only by nodes that keep coordinates
        switch (node.op) {
        case OP_TWIST: {  // radii and heights are kept; the child sees the cylinder of the box's radius
            const double rx = std::fmax(std::fabs(region.lo[0]), std::fabs(region.hi[0]));
            const double ry = std::fmax(std::fabs(region.lo[1]), std::fabs(region.hi[1]));
            // A twist below a twist keeps the outer cylinder's radius, which
            // its box (that cylinder's square) would grow by sqrt(2).
            const double r = std::fmin(sqrt_hi(add_hi(mul_hi(rx, rx), mul_hi(ry, ry))), region.radius);
            const double w = mul_hi(std::fabs(p[0]), r);
            const double stretch = mul_hi(0.5, add_hi(w, sqrt_hi(add_hi(mul_hi(w, w), 4.0))));
            next.factor = mul_hi(region.factor, mul_hi(up2(node.c0), stretch));
            next.lo[0] = next.lo[1] = -r;
            next.hi[0] = next.hi[1] = r;
            next.radius = r;
            stack.push_back(next);
            break;
        }
        case OP_LOFT:
            bound = std::fmax(bound, mul_hi(region.factor, loft_lipschitz(graph, node, region.lo, region.hi)));
            break;
        case OP_TRANSFORM: {  // child(R^T (p - t) / s) s: the scale cancels
            double m[9], t[3];
            for (int i = 0; i < 3; ++i)
                for (int j = 0; j < 3; ++j) m[3 * i + j] = p[3 * j + i] / p[12];
            for (int i = 0; i < 3; ++i) t[i] = -(m[3 * i] * p[9] + m[3 * i + 1] * p[10] + m[3 * i + 2] * p[11]);
            affine_box(m, t, region.lo, region.hi, next.lo, next.hi);
            stack.push_back(next);
            break;
        }
        case OP_MIRROR: {  // child(p - 2 (n.p - d) n): a reflection
            double m[9], t[3];
            for (int i = 0; i < 3; ++i) {
                for (int j = 0; j < 3; ++j) m[3 * i + j] = (i == j ? 1.0 : 0.0) - 2 * p[i] * p[j];
                t[i] = 2 * p[3] * p[i];
            }
            affine_box(m, t, region.lo, region.hi, next.lo, next.hi);
            stack.push_back(next);
            break;
        }
        case OP_STRETCH:  // child(p / s) min(s): at most the child's slope over p / s
            for (int a = 0; a < 3; ++a) {
                next.lo[a] = region.lo[a] / p[a];
                next.hi[a] = region.hi[a] / p[a];
            }
            pad_box(next.lo, next.hi);
            stack.push_back(next);
            break;
        case OP_ELONGATE:  // child(p - clamp(p, -e, e)): nondecreasing along each axis
            for (int a = 0; a < 3; ++a) {
                next.lo[a] = region.lo[a] - std::fmin(std::fmax(region.lo[a], -p[a]), p[a]);
                next.hi[a] = region.hi[a] - std::fmin(std::fmax(region.hi[a], -p[a]), p[a]);
            }
            pad_box(next.lo, next.hi);
            stack.push_back(next);
            break;
        case OP_OFFSET: case OP_SHELL:
            next.radius = region.radius;
            stack.push_back(next);
            break;
        case OP_UNION: case OP_INTERSECTION: case OP_DIFFERENCE:  // (smooth) minima and maxima: the largest child slope
            next.radius = region.radius;
            for (uint32_t i = 0; i < node.child_count; ++i) {
                next.node = kids[i];
                stack.push_back(next);
            }
            break;
        default:  // no other node reads a 3D child
            bound = std::fmax(bound, mul_hi(region.factor, node.step_lipschitz));
            break;
        }
    }
    return bound;
}

// ------------------------------------------------------------- tapes

void build_tape(const rsdf_graph &graph, uint32_t root, FieldTape &tape) {
    tape = FieldTape();
    tape.root = root;
    // The values the field needs: its field nodes (not through shapes) and
    // the shapes they read at the base point. EVAL_AT reads its shape at
    // other points, so that shape is part of its instruction.
    std::unordered_map<uint32_t, uint32_t> position;
    std::vector<uint32_t> members, stack{root};
    while (!stack.empty()) {
        const uint32_t id = stack.back();
        stack.pop_back();
        if (!position.emplace(id, 0).second) continue;
        members.push_back(id);
        const Node &node = graph.nodes[id];
        if (!node.field) continue;
        const uint32_t *kids = graph.child_ids(node);
        for (uint32_t i = node.op == OP_EVAL_AT ? 1 : 0; i < node.child_count; ++i) stack.push_back(kids[i]);
    }
    std::sort(members.begin(), members.end());  // children precede parents
    const size_t count = members.size();
    for (size_t i = 0; i < count; ++i) position[members[i]] = static_cast<uint32_t>(i);
    constexpr uint32_t kNone = UINT32_MAX;
    auto is_coord = [&](size_t i) { return graph.nodes[members[i]].op == OP_COORD; };
    auto operands = [&](size_t i, std::vector<uint32_t> &out) {  // member positions read by member i
        out.clear();
        const Node &node = graph.nodes[members[i]];
        if (!node.field) return;
        const uint32_t *kids = graph.child_ids(node);
        for (uint32_t c = node.op == OP_EVAL_AT ? 1 : 0; c < node.child_count; ++c) out.push_back(position[kids[c]]);
    };
    std::vector<uint32_t> number(count, kNone), last(count, 0), reads;
    uint32_t instructions = 0;
    for (size_t i = 0; i < count; ++i)
        if (!is_coord(i)) number[i] = instructions++;
    for (size_t i = 0; i < count; ++i) {
        if (number[i] == kNone) continue;
        operands(i, reads);
        for (uint32_t j : reads) last[j] = std::max(last[j], number[i]);
    }
    last[position[root]] = kNone;  // the result stays live
    std::vector<uint32_t> operand(count, 0), free_slots;
    tape.code.reserve(instructions);
    for (size_t i = 0; i < count; ++i) {
        const Node &node = graph.nodes[members[i]];
        const double *p = graph.param_values(node);
        if (is_coord(i)) {
            operand[i] = static_cast<uint32_t>(p[0]);
            continue;
        }
        FieldTape::Instr ins{};
        ins.op = node.field ? node.op : FieldTape::kShape;
        ins.node = members[i];
        ins.first = static_cast<uint32_t>(tape.args.size());
        operands(i, reads);
        for (uint32_t j : reads) tape.args.push_back(operand[j]);
        ins.count = static_cast<uint32_t>(reads.size());
        if (node.op == OP_EVAL_AT) ins.shape = graph.child_ids(node)[0];
        if (node.field && node.param_count > 0) ins.p0 = p[0];
        if (node.field && node.param_count > 1) ins.p1 = p[1];
        uint32_t slot;
        if (!free_slots.empty()) {
            slot = free_slots.back();
            free_slots.pop_back();
        } else {
            slot = tape.slots++;
        }
        ins.dst = 3 + slot;
        operand[i] = ins.dst;
        // Free the operands read for the last time (after choosing the
        // destination, so no instruction writes a slot it reads).
        for (uint32_t j : reads)
            if (number[j] != kNone && last[j] == number[i]) {
                free_slots.push_back(operand[j] - 3);
                last[j] = kNone - 1;  // freed (an operand may appear twice)
            }
        tape.code.push_back(ins);
    }
    tape.result = operand[position[root]];
}

uint64_t field_budget(const FieldTape &tape) {
    constexpr uint64_t kWork = uint64_t(1) << 24;
    return std::max<uint64_t>(1, std::min<uint64_t>(kFieldBoxes, kWork / (2 * std::max<uint64_t>(tape.visits, 1))));
}

void implicit_analysis_box(const double *lo, const double *hi, double *analysis_lo, double *analysis_hi) {
    double extent = 0;
    for (int a = 0; a < 3; ++a) extent = std::fmax(extent, hi[a] - lo[a]);
    const double grow = 1e-6 * extent;
    for (int a = 0; a < 3; ++a) {
        analysis_lo[a] = lo[a] - grow;
        analysis_hi[a] = hi[a] + grow;
    }
}

}  // namespace rsdf

// ------------------------------------------------------------- C API

using namespace rsdf;

namespace {

constexpr uint64_t kMaxFieldBoxes = 1 << 20;
constexpr int kFieldSamples = 4096;
constexpr uint64_t kSampleWork = uint64_t(1) << 26;  // node visits of the gradient sample

void write_message(rsdf_field_report &report, const std::string &message) {
    const size_t length = std::min(message.size(), sizeof report.message - 1);
    std::memcpy(report.message, message.data(), length);
    report.message[length] = 0;
}

}  // namespace

extern "C" {

int rsdf_field_analyze(const rsdf_graph *graph, uint32_t node, const double *lo, const double *hi,
                       uint32_t max_boxes, rsdf_field_report *report) {
    return guarded([&] {
               const rsdf_graph &g = checked_graph(graph);
               const Node &target = checked_node(g, node);
               if (!report) throw Error("report pointer is null");
               std::memset(report, 0, sizeof *report);
               if (!target.tape)
                   throw Error(target.field
                                   ? format("node %u (%s) is a field read by other fields; analyze its root field or "
                                            "its IMPLICIT node", node, op_name(target.op))
                                   : format("node %u (%s) is not a field or an IMPLICIT node", node,
                                            op_name(target.op)));
               const FieldTape &tape = g.tapes[target.tape - 1];
               double box_lo[3], box_hi[3];
               if (lo || hi) {
                   if (!lo || !hi) throw Error("pass both box corners, or neither for an IMPLICIT node's bounds");
                   for (int a = 0; a < 3; ++a) {
                       if (!(std::fabs(lo[a]) <= 1e150) || !(std::fabs(hi[a]) <= 1e150))
                           throw Error("the box must be finite with coordinates of at most 1e150 in magnitude");
                       if (!(lo[a] <= hi[a])) throw Error(format("the box has lo > hi along %c", "xyz"[a]));
                       box_lo[a] = lo[a];
                       box_hi[a] = hi[a];
                   }
               } else {
                   if (target.op != OP_IMPLICIT)
                       throw Error(format("node %u (%s) has no bounds of its own; pass the box to analyze", node,
                                          op_name(target.op)));
                   const double *p = g.param_values(target);
                   for (int a = 0; a < 3; ++a) {
                       box_lo[a] = p[a];
                       box_hi[a] = p[3 + a];
                   }
               }
               const uint64_t budget = max_boxes == 0 ? field_budget(tape) : std::min<uint64_t>(max_boxes, kMaxFieldBoxes);
               double analysis_lo[3], analysis_hi[3];
               if (lo || hi) {
                   std::copy(box_lo, box_lo + 3, analysis_lo);
                   std::copy(box_hi, box_hi + 3, analysis_hi);
               } else {
                   implicit_analysis_box(box_lo, box_hi, analysis_lo, analysis_hi);
               }
               if (tape.visits > kMaxProvenVisits) {  // as IMPLICIT refuses it at load
                   report->bound = INFINITY;
                   report->value_lo = -INFINITY;
                   report->value_hi = INFINITY;
                   report->failed_node = tape.root;
                   std::copy(analysis_lo, analysis_lo + 3, report->failed_lo);
                   std::copy(analysis_hi, analysis_hi + 3, report->failed_hi);
                   write_message(*report, format("node %u (%s): the field costs %llu node visits per point; a slope "
                                                 "bound can be proven only for fields of at most %llu",
                                                 tape.root, op_name(g.nodes[tape.root].op),
                                                 static_cast<unsigned long long>(tape.visits),
                                                 static_cast<unsigned long long>(kMaxProvenVisits)));
                   return;
               }
               // An IMPLICIT node's own bounds and default budget: the proof made
               // at load (its L), with statistics sampled once per proof.
               const Proof *proof = !lo && !hi && max_boxes == 0 && target.proof ? &g.proofs[target.proof - 1]
                                                                                   : nullptr;
               const FieldBound bound = proof ? proof->bound : analyze_field(g, tape, analysis_lo, analysis_hi, budget);
               report->bound = bound.bound;
               report->value_lo = bound.value_lo;
               report->value_hi = bound.value_hi;
               report->boxes = bound.boxes;
               report->ok = bound.ok ? 1 : 0;
               if (!bound.ok) {
                   report->failed_node = bound.failed_node;
                   std::copy(bound.failed_lo, bound.failed_lo + 3, report->failed_lo);
                   std::copy(bound.failed_hi, bound.failed_hi + 3, report->failed_hi);
                   write_message(*report, bound.problem);
                   return;  // no statistics without a bound: the field may be undefined there
               }
               if (proof) {
                   std::lock_guard<std::mutex> lock(g.proofs_mutex);
                   if (proof->sampled) {
                       report->samples = proof->samples;
                       report->gradient_min = proof->gradient_min;
                       report->gradient_median = proof->gradient_median;
                       report->gradient_max = proof->gradient_max;
                       report->quality = bound.bound > 0 ? report->gradient_median / bound.bound : 0.0;
                       return;
                   }
               }
               // |grad f| at a fixed low-discrepancy sample (Roberts' R3
               // sequence) of the box, by tetrahedral differences of f.
               const uint32_t field_node = target.op == OP_IMPLICIT ? tape.root : node;
               double extent = 0;
               for (int a = 0; a < 3; ++a) extent = std::fmax(extent, box_hi[a] - box_lo[a]);
               const double step = std::fmax(1e-6 * extent, 1e-9);
               constexpr double kPhi = 1.2207440846057596;
               const double alpha[3] = {1 / kPhi, 1 / (kPhi * kPhi), 1 / (kPhi * kPhi * kPhi)};
               static const double kTetra[4][3] = {{1, -1, -1}, {-1, -1, 1}, {-1, 1, -1}, {1, 1, 1}};
               // Four evaluations a sample: large fields get fewer samples, so
               // sampling visits at most about 2^26 nodes (at least 64 samples).
               const int samples = static_cast<int>(std::clamp<uint64_t>(
                   kSampleWork / (4 * std::max<uint64_t>(tape.visits, 1)), 64, kFieldSamples));
               std::vector<double> magnitudes;
               magnitudes.reserve(samples);
               constexpr int kPoints = kBatch / 4;
               double x[kBatch], y[kBatch], z[kBatch], values[kBatch];
               for (int first = 0; first < samples; first += kPoints) {
                   const int n = std::min(kPoints, samples - first);
                   for (int i = 0; i < n; ++i) {
                       double q[3];
                       for (int a = 0; a < 3; ++a) {
                           const double u = 0.5 + alpha[a] * (first + i + 1);
                           q[a] = box_lo[a] + (u - std::floor(u)) * (box_hi[a] - box_lo[a]);
                       }
                       for (int k = 0; k < 4; ++k) {
                           x[4 * i + k] = q[0] + step * kTetra[k][0];
                           y[4 * i + k] = q[1] + step * kTetra[k][1];
                           z[4 * i + k] = q[2] + step * kTetra[k][2];
                       }
                   }
                   eval_batch(g, field_node, 4 * n, x, y, z, values);
                   for (int i = 0; i < n; ++i) {
                       double gradient[3] = {0, 0, 0};
                       for (int k = 0; k < 4; ++k)
                           for (int a = 0; a < 3; ++a) gradient[a] += values[4 * i + k] * kTetra[k][a];
                       const double length = std::sqrt(gradient[0] * gradient[0] + gradient[1] * gradient[1] +
                                                       gradient[2] * gradient[2]) / (4 * step);
                       if (std::isfinite(length)) magnitudes.push_back(length);
                   }
               }
               report->samples = magnitudes.size();
               if (!magnitudes.empty()) {
                   std::sort(magnitudes.begin(), magnitudes.end());
                   const size_t m = magnitudes.size();
                   report->gradient_min = magnitudes.front();
                   report->gradient_max = magnitudes.back();
                   report->gradient_median =
                       m % 2 ? magnitudes[m / 2] : 0.5 * (magnitudes[m / 2 - 1] + magnitudes[m / 2]);
               }
               report->quality = bound.ok && bound.bound > 0 ? report->gradient_median / bound.bound : 0.0;
               if (proof) {
                   std::lock_guard<std::mutex> lock(g.proofs_mutex);
                   proof->samples = report->samples;
                   proof->gradient_min = report->gradient_min;
                   proof->gradient_median = report->gradient_median;
                   proof->gradient_max = report->gradient_max;
                   proof->sampled = true;
               }
           })
               ? 0
               : -1;
}

}  // extern "C"
