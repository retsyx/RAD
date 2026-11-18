"""Custom fields (traced expressions) in the native SDF kernel, through its C API.

RSDF graphs with field nodes (opcodes 200-223) and IMPLICIT (10) are written
directly with struct. Covers evaluation of every opcode against numpy, the
slope bound IMPLICIT proves at load (never below the field's actual slope,
on fixed-seed random expressions and on lattices and warped shapes), the
failures that name a node and region, meshing of implicit shapes,
determinism and speed.
"""
import ctypes
import math
import os
from pathlib import Path
import re
import subprocess
import sys
import tempfile
import time
import unittest

import numpy as np

ENGINE = Path(__file__).resolve().parents[1]
if str(ENGINE) not in sys.path:
    sys.path.insert(0, str(ENGINE))

from test_sdf_kernel_native import (BOX, INTERSECTION, SPHERE, TRANSFORM, UNION, CIRCLE, DIFFERENCE2, ELLIPSE,
                                    ELLIPSOID, EXTRUDE, LOFT, OFFSET2, POLYGON, REVOLVE, TWIST, Graph, Kernel,
                                    KernelError, box_ref, library_path, rotation_matrix)

IMPLICIT = 10
(CONST, COORD, ADD, SUB, MUL, DIV, NEG, ABS, SQUARE, SQRT, SIN, COS, EXP, LOG, POWC, ATAN2, MIN, MAX, CLAMP, LENGTH,
 TRIWAVE, EVAL_AT, SMIN, SMAX) = range(200, 224)


class FieldReport(ctypes.Structure):
    _fields_ = [("bound", ctypes.c_double), ("value_lo", ctypes.c_double), ("value_hi", ctypes.c_double),
                ("gradient_min", ctypes.c_double), ("gradient_median", ctypes.c_double),
                ("gradient_max", ctypes.c_double), ("quality", ctypes.c_double), ("boxes", ctypes.c_uint64),
                ("samples", ctypes.c_uint64), ("ok", ctypes.c_int32), ("failed_node", ctypes.c_uint32),
                ("failed_lo", ctypes.c_double * 3), ("failed_hi", ctypes.c_double * 3),
                ("message", ctypes.c_char * 512)]


KERNEL = None


def setUpModule():
    global KERNEL
    path = library_path()
    if path is None:
        raise unittest.SkipTest("libRADSDF.dylib not built (swift build --product RADSDF -c release)")
    KERNEL = Kernel(path)
    function = KERNEL.lib.rsdf_field_analyze
    function.restype = ctypes.c_int
    function.argtypes = [ctypes.c_void_p, ctypes.c_uint32, ctypes.c_void_p, ctypes.c_void_p, ctypes.c_uint32,
                         ctypes.POINTER(FieldReport)]


def triwave_ref(a, period):
    return np.abs(np.mod(a, 2 * period) - period)


def smooth_ref(a, b, k, lowest):
    sign = 1.0 if lowest else -1.0
    u, v = sign * a, sign * b
    h = np.maximum(k - np.abs(u - v), 0.0) / k
    return sign * (np.minimum(u, v) - h * h * k / 4)


class Fields(Graph):
    """RSDF builder with field helpers; values are node indices."""

    def coords(self):
        return tuple(self.add(COORD, [], [axis]) for axis in range(3))

    def const(self, c):
        return self.add(CONST, [], [c])

    def implicit(self, field, lo, hi):
        return self.add(IMPLICIT, [field], [*lo, *hi])


# ---------------------------------------------------------------- random expressions
#
# A random expression is a nested tuple (op, children, params) that both the
# RSDF writer and the numpy reference evaluate. Arguments of ops with a
# restricted domain are shaped to stay inside it (divisors and logarithms
# above 0.25, square roots of values above 0.25, atan2 with x above 0.5 or
# behind sin/cos), so every expression has a finite slope bound.

SHAPES = ("sphere", "box", "moved", "twisted", "loft")
TWIST_RATE, TWIST_RADII = 0.5, (2.0, 0.6, 0.8)  # an inexact (ellipsoid) child: regional slope bounds
LOFT_Z, LOFT_ELLIPSE, LOFT_CIRCLE = (-1.5, 1.5), (2.5, 0.5), 0.8  # anisotropic profiles


def implicit_ref(value, box):
    """IMPLICIT's field from h = f(clamp(p, B)) / L and the box distance:
    max(h, box), and hypot(h, box) outside B where h > 0."""
    return np.where((value > 0) & (box > 0), np.hypot(value, box), np.maximum(value, box))


def ellipsoid_ref(q, radii):
    radii = np.asarray(radii, float)
    return (np.linalg.norm(q / radii, axis=1) - 1) * radii.min()


def slab_ref(d, dz):
    return np.minimum(np.maximum(d, dz), 0) + np.hypot(np.maximum(d, 0), np.maximum(dz, 0))


def shape_ref(kind, p):
    if kind == "sphere":
        return np.linalg.norm(p, axis=1) - 3.0
    if kind == "box":
        return box_ref(p, [2.0, 3.0, 1.5], 0.5)
    if kind == "twisted":
        rotation = rotation_matrix([1, -1, 2], 30)
        local = (p - np.array([0.5, -0.3, 0.2])) @ rotation
        angle = -TWIST_RATE * local[:, 2]
        c, s = np.cos(angle), np.sin(angle)
        q = np.column_stack([c * local[:, 0] - s * local[:, 1], s * local[:, 0] + c * local[:, 1], local[:, 2]])
        w = TWIST_RATE * math.hypot(TWIST_RADII[0], TWIST_RADII[1])
        return ellipsoid_ref(q, TWIST_RADII) / math.sqrt(1 + w * w)
    if kind == "loft":
        z0, z1 = LOFT_Z
        a = ellipsoid_ref(p[:, :2], LOFT_ELLIPSE)
        b = np.linalg.norm(p[:, :2], axis=1) - LOFT_CIRCLE
        t = np.clip((p[:, 2] - z0) / (z1 - z0), 0, 1)
        slope = math.hypot(2 * LOFT_ELLIPSE[0], 2 * max(LOFT_ELLIPSE[1], LOFT_CIRCLE)) / (z1 - z0)
        blend = ((1 - t) * a + t * b) / math.sqrt(1 + slope * slope)
        return slab_ref(blend, np.abs(p[:, 2] - 0.5 * (z0 + z1)) - 0.5 * (z1 - z0))
    rotation = rotation_matrix([1, 2, 3], 40)
    local = (p - np.array([1.0, -1.0, 0.5])) @ rotation  # p_world = R p_local + t
    return box_ref(local, [2.5, 1.0, 2.0])


def write_shape(g, kind):
    if kind == "sphere":
        return g.add(SPHERE, [], [3.0])
    if kind == "box":
        return g.add(BOX, [], [2.0, 3.0, 1.5, 0.5])
    if kind == "twisted":
        twisted = g.add(TWIST, [g.add(ELLIPSOID, [], list(TWIST_RADII))], [TWIST_RATE])
        return g.move(twisted, [0.5, -0.3, 0.2], rotation_matrix([1, -1, 2], 30))
    if kind == "loft":
        return g.add(LOFT, [g.add(ELLIPSE, [], list(LOFT_ELLIPSE)), g.add(CIRCLE, [], [LOFT_CIRCLE])], list(LOFT_Z))
    return g.move(g.add(BOX, [], [2.5, 1.0, 2.0, 0.0]), [1.0, -1.0, 0.5], rotation_matrix([1, 2, 3], 40))


def random_expression(rng, depth):
    if depth == 0 or rng.random() < 0.15:
        pick = rng.random()
        if pick < 0.55:
            return ("coord", int(rng.integers(3)))
        if pick < 0.85:
            return ("const", float(np.round(rng.uniform(-2, 2), 3)))
        return ("shape", SHAPES[int(rng.integers(len(SHAPES)))])
    sub = lambda: random_expression(rng, depth - 1)  # noqa: E731
    positive = lambda e: ("add", [("square", [e]), ("const", 0.25)])  # noqa: E731
    op = rng.choice(["add", "sub", "mul", "div", "neg", "abs", "square", "sqrt", "sin", "cos", "exp", "log", "powc",
                     "atan2", "angle", "min", "max", "clamp", "length", "triwave", "eval_at", "scale", "smin", "smax"])
    if op in ("add", "sub", "mul", "smin", "smax"):
        return (op, [sub(), sub()])
    if op == "scale":
        return ("mul", [("const", float(np.round(rng.uniform(-1.5, 1.5), 3))), sub()])
    if op == "div":
        return ("div", [sub(), positive(sub())])
    if op in ("neg", "abs", "square", "cos", "sin", "triwave", "clamp"):
        return (op, [sub()])
    if op == "sqrt":
        return ("sqrt", [positive(sub())])
    if op == "exp":
        return ("exp", [("mul", [("const", 0.3), ("sin", [sub()])])])
    if op == "log":
        return ("log", [positive(sub())])
    if op == "powc":
        exponent = float(rng.choice([2, 3, -1, -2, 0.5, 1.5, 2.5]))
        return ("powc", [positive(sub()) if exponent != int(exponent) or exponent < 0 else sub()], exponent)
    if op == "atan2":
        return ("atan2", [sub(), ("add", [("square", [sub()]), ("const", 0.5)])])
    if op == "angle":  # an angle across atan2's cut, through cos of a whole multiple
        return ("cos", [("mul", [("const", float(rng.integers(1, 5))),
                                 ("atan2", [sub(), ("sub", [("const", -0.5), ("square", [sub()])])])])])
    if op in ("min", "max", "length"):
        return (op, [sub() for _ in range(int(rng.integers(1, 4)))])
    return ("eval_at", [SHAPES[int(rng.integers(len(SHAPES)))], sub(), sub(), sub()])


def expression_params(e):
    kind = e[0]
    if kind == "powc":
        return [e[2]]
    if kind == "clamp":
        return [-0.7, 0.9]
    if kind == "triwave":
        return [0.8]
    if kind in ("smin", "smax"):
        return [0.6]
    return []


OPCODES = {"add": ADD, "sub": SUB, "mul": MUL, "div": DIV, "neg": NEG, "abs": ABS, "square": SQUARE, "sqrt": SQRT,
           "sin": SIN, "cos": COS, "exp": EXP, "log": LOG, "powc": POWC, "atan2": ATAN2, "min": MIN, "max": MAX,
           "clamp": CLAMP, "length": LENGTH, "triwave": TRIWAVE, "eval_at": EVAL_AT, "smin": SMIN, "smax": SMAX}


def write_expression(g, e, memo):
    """Writes e into g (shared coordinate and shape nodes); returns its node."""
    kind = e[0]
    if kind == "coord":
        key = ("coord", e[1])
        if key not in memo:
            memo[key] = g.add(COORD, [], [e[1]])
        return memo[key]
    if kind == "const":
        return g.add(CONST, [], [e[1]])
    if kind == "shape":
        key = ("shape", e[1])
        if key not in memo:
            memo[key] = write_shape(g, e[1])
        return memo[key]
    if kind == "eval_at":
        shape = write_expression(g, ("shape", e[1][0]), memo)
        return g.add(EVAL_AT, [shape] + [write_expression(g, c, memo) for c in e[1][1:]])
    return g.add(OPCODES[kind], [write_expression(g, c, memo) for c in e[1]], expression_params(e))


def evaluate_expression(e, p):
    kind = e[0]
    if kind == "coord":
        return p[:, e[1]].copy()
    if kind == "const":
        return np.full(len(p), e[1])
    if kind == "shape":
        return shape_ref(e[1], p)
    if kind == "eval_at":
        warped = np.column_stack([evaluate_expression(c, p) for c in e[1][1:]])
        return shape_ref(e[1][0], warped)
    v = [evaluate_expression(c, p) for c in e[1]]
    with np.errstate(all="ignore"):
        return {
            "add": lambda: v[0] + v[1], "sub": lambda: v[0] - v[1], "mul": lambda: v[0] * v[1],
            "div": lambda: v[0] / v[1], "neg": lambda: -v[0], "abs": lambda: np.abs(v[0]),
            "square": lambda: v[0] * v[0], "sqrt": lambda: np.sqrt(v[0]), "sin": lambda: np.sin(v[0]),
            "cos": lambda: np.cos(v[0]), "exp": lambda: np.exp(v[0]), "log": lambda: np.log(v[0]),
            "powc": lambda: np.power(v[0], e[2]), "atan2": lambda: np.arctan2(v[0], v[1]),
            "min": lambda: np.min(v, axis=0), "max": lambda: np.max(v, axis=0),
            "clamp": lambda: np.clip(v[0], -0.7, 0.9), "length": lambda: np.sqrt(np.sum(np.square(v), axis=0)),
            "triwave": lambda: triwave_ref(v[0], 0.8),
            "smin": lambda: smooth_ref(v[0], v[1], 0.6, True), "smax": lambda: smooth_ref(v[0], v[1], 0.6, False),
        }[kind]()


# ---------------------------------------------------------------- lattices

def gyroid(g, period=10.0, scale=1.0):
    """sin(kx)cos(ky) + sin(ky)cos(kz) + sin(kz)cos(kx) with k = 2 pi / period."""
    x, y, z = g.coords()
    k = g.const(2 * math.pi / period)
    kx, ky, kz = (g.add(MUL, [k, c]) for c in (x, y, z))
    s = [g.add(SIN, [c]) for c in (kx, ky, kz)]
    c = [g.add(COS, [v]) for v in (kx, ky, kz)]
    total = g.add(ADD, [g.add(ADD, [g.add(MUL, [s[0], c[1]]), g.add(MUL, [s[1], c[2]])]), g.add(MUL, [s[2], c[0]])])
    return g.add(MUL, [g.const(scale), total]) if scale != 1.0 else total


def gyroid_ref(p, period=10.0):
    k = 2 * math.pi / period
    x, y, z = (k * p[:, a] for a in range(3))
    return np.sin(x) * np.cos(y) + np.sin(y) * np.cos(z) + np.sin(z) * np.cos(x)


def schwarz_p(g, period=8.0):
    x, y, z = g.coords()
    k = g.const(2 * math.pi / period)
    return g.add(ADD, [g.add(ADD, [g.add(COS, [g.add(MUL, [k, x])]), g.add(COS, [g.add(MUL, [k, y])])]),
                       g.add(COS, [g.add(MUL, [k, z])])])


class FieldCase(unittest.TestCase):
    def setUp(self):
        self.k = KERNEL
        self.handles = []
        self.rng = np.random.default_rng(20261117)

    def tearDown(self):
        for handle in self.handles:
            self.k.lib.rsdf_graph_free(handle)

    def build(self, graph):
        handle = self.k.create(graph.to_bytes())
        self.handles.append(handle)
        return handle

    def analyze(self, handle, node, lo=None, hi=None, max_boxes=0):
        report = FieldReport()
        corner = lambda c: (ctypes.c_double * 3)(*c) if c is not None else None  # noqa: E731
        status = self.k.lib.rsdf_field_analyze(handle, node, corner(lo), corner(hi), max_boxes, ctypes.byref(report))
        if status != 0:
            raise KernelError(self.k.error())
        return report

    def points_in(self, lo, hi, count):
        return self.rng.uniform(lo, hi, size=(count, 3))

    def secant_slopes(self, handle, node, lo, hi, count=4000):
        """|f(q) - f(p)| / |q - p| for points p of the box and q a short step
        along the field's numerical gradient (kept inside the box). Secant
        slopes never exceed a Lipschitz bound and approach |grad f| at smooth
        points. Returns (slopes, rounding slack per slope, values at p)."""
        lo, hi = np.asarray(lo, float), np.asarray(hi, float)
        extent = float(np.max(hi - lo))
        h = 1e-6 * extent
        p = self.points_in(lo + 2e-3 * extent, hi - 2e-3 * extent, count)
        offsets = np.vstack([np.eye(3), -np.eye(3)]) * h
        stencil = (p[:, None, :] + offsets[None, :, :]).reshape(-1, 3)
        values = self.k.evaluate(handle, node, stencil).reshape(count, 6)
        gradient = (values[:, :3] - values[:, 3:]) / (2 * h)
        length = np.linalg.norm(gradient, axis=1)
        direction = np.where(length[:, None] > 0, gradient / np.maximum(length, 1e-300)[:, None], [1.0, 0.0, 0.0])
        step = 1e-4 * extent
        q = np.clip(p + step * direction, lo, hi)
        fp, fq = self.k.evaluate(handle, node, p), self.k.evaluate(handle, node, q)
        distance = np.linalg.norm(q - p, axis=1)
        slopes = np.abs(fq - fp) / distance
        slack = 1e-12 * (1 + np.abs(fp) + np.abs(fq)) / distance
        return slopes, slack, fp

    def assertSound(self, handle, node, lo, hi, report, label=""):
        self.assertEqual(report.ok, 1, f"{label}: {report.message.decode()}")
        slopes, slack, values = self.secant_slopes(handle, node, lo, hi)
        excess = slopes - (report.bound * (1 + 1e-9) + slack)
        self.assertLessEqual(float(np.max(excess)), 0.0,
                             f"{label}: secant slope {np.max(slopes):.17g} exceeds the bound {report.bound:.17g}")
        self.assertLessEqual(report.value_lo, float(np.min(values)) + 1e-12 * (1 + abs(report.value_lo)), label)
        self.assertGreaterEqual(report.value_hi, float(np.max(values)) - 1e-12 * (1 + abs(report.value_hi)), label)
        self.assertLessEqual(report.gradient_max, report.bound * (1 + 1e-4) + 1e-9, label)
        return float(np.max(slopes))


class EvaluationTests(FieldCase):
    def test_every_opcode_matches_numpy(self):
        g = Fields()
        x, y, z = g.coords()
        one, half = g.const(1.0), g.const(0.5)
        ball = g.add(SPHERE, [], [2.5])
        cases = {
            "const": (g.const(-3.25), lambda p: np.full(len(p), -3.25)),
            "coord y": (g.add(COORD, [], [1]), lambda p: p[:, 1]),
            "add": (g.add(ADD, [x, y]), lambda p: p[:, 0] + p[:, 1]),
            "sub": (g.add(SUB, [x, z]), lambda p: p[:, 0] - p[:, 2]),
            "mul": (g.add(MUL, [y, z]), lambda p: p[:, 1] * p[:, 2]),
            "mul same": (g.add(MUL, [z, z]), lambda p: p[:, 2] ** 2),
            "div": (g.add(DIV, [x, g.add(ADD, [g.add(SQUARE, [y]), one])]), lambda p: p[:, 0] / (p[:, 1] ** 2 + 1)),
            "neg": (g.add(NEG, [z]), lambda p: -p[:, 2]),
            "abs": (g.add(ABS, [x]), lambda p: np.abs(p[:, 0])),
            "square": (g.add(SQUARE, [y]), lambda p: p[:, 1] ** 2),
            "sqrt": (g.add(SQRT, [g.add(ADD, [g.add(SQUARE, [x]), half])]), lambda p: np.sqrt(p[:, 0] ** 2 + 0.5)),
            "sin": (g.add(SIN, [x]), lambda p: np.sin(p[:, 0])),
            "cos": (g.add(COS, [y]), lambda p: np.cos(p[:, 1])),
            "exp": (g.add(EXP, [g.add(MUL, [g.const(0.3), z])]), lambda p: np.exp(0.3 * p[:, 2])),
            "log": (g.add(LOG, [g.add(ADD, [g.add(SQUARE, [z]), g.const(0.1)])]), lambda p: np.log(p[:, 2] ** 2 + 0.1)),
            "powc 3": (g.add(POWC, [x], [3]), lambda p: p[:, 0] ** 3),
            "powc -2": (g.add(POWC, [y], [-2]), lambda p: p[:, 1] ** -2.0),
            "powc 2.5": (g.add(POWC, [g.add(ABS, [z])], [2.5]), lambda p: np.abs(p[:, 2]) ** 2.5),
            "powc 0": (g.add(POWC, [x], [0]), lambda p: np.ones(len(p))),
            "atan2": (g.add(ATAN2, [y, x]), lambda p: np.arctan2(p[:, 1], p[:, 0])),
            "min": (g.add(MIN, [x, y, z]), lambda p: np.min(p, axis=1)),
            "min of one": (g.add(MIN, [y]), lambda p: p[:, 1]),
            "max": (g.add(MAX, [x, g.add(NEG, [y])]), lambda p: np.maximum(p[:, 0], -p[:, 1])),
            "clamp": (g.add(CLAMP, [x], [-1, 2]), lambda p: np.clip(p[:, 0], -1, 2)),
            "length": (g.add(LENGTH, [x, y, z]), lambda p: np.linalg.norm(p, axis=1)),
            "length of one": (g.add(LENGTH, [z]), lambda p: np.abs(p[:, 2])),
            "triwave": (g.add(TRIWAVE, [x], [1.5]), lambda p: triwave_ref(p[:, 0], 1.5)),
            "smin": (g.add(SMIN, [x, y], [0.8]), lambda p: smooth_ref(p[:, 0], p[:, 1], 0.8, True)),
            "smax": (g.add(SMAX, [x, z], [0.8]), lambda p: smooth_ref(p[:, 0], p[:, 2], 0.8, False)),
            "smin k=0": (g.add(SMIN, [x, y], [0]), lambda p: np.minimum(p[:, 0], p[:, 1])),
            "eval_at": (g.add(EVAL_AT, [ball, g.add(ADD, [x, one]), g.add(MUL, [y, g.const(2.0)]), z]),
                        lambda p: np.linalg.norm(np.column_stack([p[:, 0] + 1, 2 * p[:, 1], p[:, 2]]), axis=1) - 2.5),
            "shape read as a field": (g.add(ADD, [ball, x]), lambda p: np.linalg.norm(p, axis=1) - 2.5 + p[:, 0]),
        }
        handle = self.build(g)
        p = self.points_in(-4, 4, 5000)
        for name, (node, reference) in cases.items():
            with self.subTest(op=name):
                expected = reference(p)
                actual = self.k.evaluate(handle, node, p)
                np.testing.assert_allclose(actual, expected, rtol=1e-13, atol=1e-13)

    def test_shared_subexpressions_run_once(self):
        # 60 doublings describe 2^60 paths; the tape runs 60 additions.
        g = Fields()
        x, _, _ = g.coords()
        value = x
        for _ in range(60):
            value = g.add(ADD, [value, value])
        handle = self.build(g)
        p = self.points_in(-1, 1, 1000)
        np.testing.assert_array_equal(self.k.evaluate(handle, value, p), p[:, 0] * 2.0 ** 60)

    def test_random_expressions_match_numpy(self):
        for trial in range(200):
            e = random_expression(self.rng, 4)
            g = Fields()
            root = write_expression(g, e, {})
            handle = self.build(g)
            p = self.points_in(-3, 3, 400)
            expected = evaluate_expression(e, p)
            actual = self.k.evaluate(handle, root, p)
            scale = 1 + np.abs(expected)
            with self.subTest(trial=trial, expression=e):
                np.testing.assert_array_less(np.abs(actual - expected), 1e-9 * scale)


class ValidationTests(FieldCase):
    def assertRefused(self, graph, message):
        with self.assertRaises(KernelError) as caught:
            self.build(graph)
        self.assertIn(message, str(caught.exception))
        return str(caught.exception)

    def test_shape_consumers_reject_fields(self):
        for op, params in ((UNION, [0]), (INTERSECTION, [0.5]), (TRANSFORM, [1, 0, 0, 0, 1, 0, 0, 0, 1, 1, 2, 3, 1])):
            with self.subTest(op=op):
                g = Fields()
                x, y, _ = g.coords()
                field = g.add(ADD, [x, y])
                g.add(op, [g.add(SPHERE, [], [1]), field] if op != TRANSFORM else [field], params)
                message = self.assertRefused(g, "is a field (ADD)")
                self.assertIn("through IMPLICIT", message)
        g = Fields()
        x, y, z = g.coords()
        g.add(EVAL_AT, [g.add(ADD, [x, y]), x, y, z])
        self.assertRefused(g, "must be a shape")
        g = Fields()
        g.add(ADD, [g.add(CIRCLE, [], [1]), g.add(COORD, [], [0])])
        self.assertRefused(g, "is 2D but must be 3D")

    def test_arity_and_parameters(self):
        cases = [
            (lambda g, x: g.add(ADD, [x]), "has 1 children; expected 2"),
            (lambda g, x: g.add(EVAL_AT, [g.add(SPHERE, [], [1]), x, x]), "expected 4"),
            (lambda g, x: g.add(MIN, []), "expected at least 1"),
            (lambda g, x: g.add(CONST, [x], [1]), "expected 0"),
            (lambda g, x: g.add(POWC, [x], []), "has 0 parameters; expected 1"),
            (lambda g, x: g.add(NEG, [x], [1]), "has 1 parameters; expected 0"),
            (lambda g, x: g.add(COORD, [], [3]), "axis must be 0"),
            (lambda g, x: g.add(COORD, [], [0.5]), "axis must be 0"),
            (lambda g, x: g.add(CLAMP, [x], [2, 1]), "must not exceed"),
            (lambda g, x: g.add(TRIWAVE, [x], [0]), "period must lie between"),
            (lambda g, x: g.add(TRIWAVE, [x], [1e-310]), "period must lie between"),
            (lambda g, x: g.add(POWC, [x], [2.0 ** 54]), "at most 2^52"),
            (lambda g, x: g.implicit(x, (-1e200, 0, 0), (1, 1, 1)), "within 1e150"),
            (lambda g, x: g.add(CONST, [], [math.nan]), "not finite"),
            (lambda g, x: g.add(POWC, [x], [math.inf]), "not finite"),
            (lambda g, x: g.implicit(x, (0, 0, 0), (1, 0, 1)), "lo < hi along y"),
            (lambda g, x: g.implicit(x, (0, 0, 0), (1, 1, math.inf)), "not finite"),
            (lambda g, x: g.add(IMPLICIT, [x], [0, 0, 0, 1, 1]), "expected 6"),
        ]
        for build, message in cases:
            with self.subTest(message=message):
                g = Fields()
                build(g, g.add(COORD, [], [0]))
                self.assertRefused(g, message)

    def test_queries_of_fields(self):
        g = Fields()
        x, y, z = g.coords()
        inner = g.add(ADD, [x, y])
        root = g.add(MUL, [inner, z])
        handle = self.build(g)
        p = self.points_in(-1, 1, 10)
        np.testing.assert_allclose(self.k.evaluate(handle, root, p), (p[:, 0] + p[:, 1]) * p[:, 2], rtol=1e-15)
        self.assertEqual(self.k.gradient(handle, root, p).shape, (10, 3))
        with self.assertRaisesRegex(KernelError, "read by other fields"):
            self.k.evaluate(handle, inner, p)
        with self.assertRaisesRegex(KernelError, "has no bounds"):
            self.k.bounds(handle, root)
        with self.assertRaisesRegex(KernelError, "is a field"):
            self.k.mesh(handle, root, 0.1, bounds=((-1, -1, -1), (1, 1, 1)))
        with self.assertRaisesRegex(KernelError, "is a field"):
            self.k.raycast(handle, root, [[0, 0, 5]], [[0, 0, -1]])
        with self.assertRaisesRegex(KernelError, "read by other fields"):
            self.analyze(handle, inner, (0, 0, 0), (1, 1, 1))
        with self.assertRaisesRegex(KernelError, "pass the box"):
            self.analyze(handle, root)
        with self.assertRaisesRegex(KernelError, "lo > hi"):
            self.analyze(handle, root, (0, 0, 0), (1, -1, 1))
        report = self.analyze(handle, root, (-1, -1, -1), (1, 1, 1))
        self.assertEqual(report.ok, 1)
        g = Fields()
        g.add(SPHERE, [], [1])
        with self.assertRaisesRegex(KernelError, "not a field or an IMPLICIT"):
            self.analyze(self.build(g), 0, (0, 0, 0), (1, 1, 1))

    def test_report_layout_matches_the_bindings(self):
        import radsdf
        self.assertEqual(ctypes.sizeof(FieldReport), 640)
        self.assertEqual(ctypes.sizeof(radsdf._FieldReport), 640)
        for name, *_ in FieldReport._fields_:
            self.assertEqual(getattr(FieldReport, name).offset, getattr(radsdf._FieldReport, name).offset, name)

    def test_implicit_of_a_shape(self):
        g = Fields()
        ball = g.add(SPHERE, [], [2])
        shape = g.implicit(ball, (-3, -3, -3), (3, 3, 3))
        handle = self.build(g)
        report = self.analyze(handle, shape)
        self.assertEqual(report.bound, 1.0)
        lo, hi, lipschitz = self.k.bounds(handle, shape)
        np.testing.assert_array_equal(lo, [-3, -3, -3])
        np.testing.assert_array_equal(hi, [3, 3, 3])
        self.assertEqual(lipschitz, 1.0)
        p = self.points_in(-5, 5, 2000)
        expected = implicit_ref(np.linalg.norm(np.clip(p, -3, 3), axis=1) - 2, box_ref(p, [3, 3, 3]))
        np.testing.assert_allclose(self.k.evaluate(handle, shape, p), expected, rtol=0, atol=1e-13)


class BoundTests(FieldCase):
    def test_random_expressions_never_exceed_their_bound(self):
        tightness = []
        for trial in range(300):
            e = random_expression(self.rng, 4)
            if e[0] == "shape":  # analysis takes fields: read the shape through one
                e = ("add", [e, ("const", 0.0)])
            g = Fields()
            root = write_expression(g, e, {})
            handle = self.build(g)
            centre = self.rng.uniform(-3, 3, 3)
            half = self.rng.uniform(0.3, 4, 3)
            lo, hi = centre - half, centre + half
            report = self.analyze(handle, root, lo, hi)
            with self.subTest(trial=trial, expression=e):
                largest = self.assertSound(handle, root, lo, hi, report, f"trial {trial}")
                if largest > 0:
                    tightness.append(report.bound / largest)
        # Diagnostics: how far above the steepest secant found the bounds lie.
        print(f"\nrandom expressions: bound / steepest secant median {np.median(tightness):.3f}, "
              f"90th percentile {np.percentile(tightness, 90):.3f}")

    def test_lattices_and_warped_shapes(self):
        def sheet(g):
            return g.add(SUB, [g.add(ABS, [gyroid(g)]), g.const(0.35)])

        def twisted(g):  # a twist written as a warp: the box at (x cos az + y sin az, y cos az - x sin az, z)
            x, y, z = g.coords()
            a = g.add(MUL, [g.const(0.15), z])
            c, s = g.add(COS, [a]), g.add(SIN, [a])
            u = g.add(ADD, [g.add(MUL, [x, c]), g.add(MUL, [y, s])])
            v = g.add(SUB, [g.add(MUL, [y, c]), g.add(MUL, [x, s])])
            return g.add(EVAL_AT, [g.add(BOX, [], [4, 1.5, 8, 0.2]), u, v, z])

        def bent(g):
            x, y, z = g.coords()
            return g.add(EVAL_AT, [g.add(SPHERE, [], [3]),
                                   g.add(ADD, [x, g.add(MUL, [g.const(0.8), g.add(SIN, [y])])]), y, z])

        def polar(g):  # a six-petal outline; the box crosses atan2's cut
            x, y, _ = g.coords()
            angle = g.add(ATAN2, [y, x])
            return g.add(SUB, [g.add(LENGTH, [x, y]),
                               g.add(ADD, [g.const(5), g.add(COS, [g.add(MUL, [g.const(6), angle])])])])

        def blend(g):
            return g.add(MIN, [twisted(g), bent(g), g.add(MUL, [g.const(0.5), schwarz_p(g)])])

        cases = {
            "gyroid": (gyroid, (-20, -20, -20), (20, 20, 20)),
            "gyroid sheet": (sheet, (-7, -3, -11), (13, 9, 5)),
            "schwarz p": (schwarz_p, (-8, -8, -8), (8, 8, 8)),
            "twisted box": (twisted, (-6, -6, -9), (6, 6, 9)),
            "bent sphere": (bent, (-5, -5, -5), (5, 5, 5)),
            "polar flower (across the cut)": (polar, (-8, -8, -1), (-1, 8, 1)),
            "blend": (blend, (-6, -6, -6), (6, 6, 6)),
        }
        for name, (make, lo, hi) in cases.items():
            with self.subTest(field=name):
                g = Fields()
                node = make(g)
                handle = self.build(g)
                report = self.analyze(handle, node, lo, hi)
                largest = self.assertSound(handle, node, lo, hi, report, name)
                print(f"\n{name}: bound {report.bound:.4f}, steepest secant {largest:.4f}, sampled max "
                      f"{report.gradient_max:.4f}, quality {report.quality:.3f}, boxes {report.boxes}")


class FailureTests(FieldCase):
    """Fields without a finite slope bound fail at load, naming the node and region."""

    def refused(self, make, lo, hi, *fragments):
        g = Fields()
        root, culprit = make(g)
        g.implicit(root, lo, hi)
        with self.assertRaises(KernelError) as caught:
            self.build(g)
        message = str(caught.exception)
        self.assertIn("(IMPLICIT): its field has no finite slope bound over its bounds", message)
        self.assertIn(f"node {culprit} (", message)
        self.assertIn("in the region [", message)
        for fragment in fragments:
            self.assertIn(fragment, message)
        # The same failure through rsdf_field_analyze of the field root.
        g.records.pop()
        handle = self.build(g)
        report = self.analyze(handle, root, lo, hi)
        self.assertEqual(report.ok, 0)
        self.assertEqual(report.bound, math.inf)
        self.assertEqual(report.failed_node, culprit)
        self.assertIn(fragments[0], report.message.decode())
        self.assertEqual(report.samples, 0, "no gradient statistics where the field may be undefined")
        return report

    def test_division_across_zero(self):
        def make(g):
            x, _, _ = g.coords()
            divide = g.add(DIV, [g.const(1), x])
            return g.add(ADD, [divide, g.const(2)]), divide
        report = self.refused(make, (-1, -2, -3), (1, 2, 3), "the divisor may be 0", "(DIV)")
        self.assertLessEqual(report.failed_lo[0], 0)
        self.assertGreaterEqual(report.failed_hi[0], 0)
        self.assertLess(report.failed_hi[0] - report.failed_lo[0], 0.5)  # a thin slab around x = 0

    def test_sqrt_and_log_at_zero(self):
        def distance(g):  # sqrt(x^2 + y^2 + z^2): use LENGTH, which has no pole
            x, y, z = g.coords()
            root = g.add(SQRT, [g.add(ADD, [g.add(ADD, [g.add(SQUARE, [x]), g.add(SQUARE, [y])]),
                                            g.add(SQUARE, [z])])])
            return g.add(SUB, [root, g.const(1)]), root
        report = self.refused(distance, (-2, -2, -2), (2, 2, 2), "sqrt may be 0", "length()")
        for a in range(3):
            self.assertLessEqual(report.failed_lo[a], 0)
            self.assertGreaterEqual(report.failed_hi[a], 0)

        def negative(g):
            x, _, _ = g.coords()
            root = g.add(SQRT, [x])
            return root, root
        self.refused(negative, (-1, 0, 0), (1, 1, 1), "sqrt may be negative")

        def logarithm(g):
            _, y, _ = g.coords()
            root = g.add(LOG, [g.add(SQUARE, [y])])
            return root, root
        self.refused(logarithm, (-1, -1, -1), (1, 1, 1), "log may be 0 or negative")

    def test_atan2_pole_and_cut(self):
        def pole(g):
            x, y, _ = g.coords()
            angle = g.add(ATAN2, [y, x])
            return g.add(SIN, [angle]), angle
        report = self.refused(pole, (-1, -1, -1), (1, 1, 1), "pole")
        self.assertLessEqual(report.failed_lo[0], 0)
        self.assertGreaterEqual(report.failed_hi[0], 0)

        def bare(g):  # the angle itself jumps by 2 pi across the negative x axis
            x, y, _ = g.coords()
            angle = g.add(ATAN2, [y, x])
            return g.add(ADD, [angle, g.const(1)]), g.records.__len__() - 1
        self.refused(bare, (-5, -1, -1), (-1, 1, 1), "jumps by 2 pi")

        def fractional(g):  # 2.5 turns is not a whole number: cos(2.5 angle) jumps too
            x, y, _ = g.coords()
            scaled = g.add(MUL, [g.const(2.5), g.add(ATAN2, [y, x])])
            return g.add(COS, [scaled]), scaled
        self.refused(fractional, (-5, -1, -1), (-1, 1, 1), "jumps by 2 pi")

        # Whole multiples through cos are continuous across the cut.
        g = Fields()
        x, y, _ = g.coords()
        field = g.add(COS, [g.add(MUL, [g.const(3), g.add(ATAN2, [y, x])])])
        shape = g.implicit(field, (-5, -2, -1), (-1, 2, 1))
        handle = self.build(g)
        report = self.analyze(handle, field, (-5, -2, -1), (-1, 2, 1))
        self.assertSound(handle, field, (-5, -2, -1), (-1, 2, 1), report, "cos(3 angle)")
        self.assertLessEqual(report.bound, 3.0 * 1.1)  # |grad| = 3 |sin 3a| / r <= 3 for r >= 1
        self.assertEqual(self.analyze(handle, shape).ok, 1)

    def test_powers_and_overflow(self):
        def fractional(g):
            x, _, _ = g.coords()
            root = g.add(POWC, [x], [0.5])
            return root, root
        self.refused(fractional, (-1, 0, 0), (1, 1, 1), "fractional power")

        def reciprocal(g):
            _, _, z = g.coords()
            root = g.add(POWC, [z], [-1])
            return root, root
        self.refused(reciprocal, (-1, -1, -1), (1, 1, 1), "negative power")

        def root_of_zero(g):  # x^(1/3) of |x|: infinite slope at 0
            x, _, _ = g.coords()
            root = g.add(POWC, [g.add(ABS, [x])], [1 / 3])
            return root, root
        self.refused(root_of_zero, (-1, -1, -1), (1, 1, 1), "power below 1")

        def huge(g):
            x, _, _ = g.coords()
            root = g.add(EXP, [g.add(MUL, [g.const(1000), x])])
            return root, root
        self.refused(huge, (0, 0, 0), (1, 1, 1), "overflows")


def monte_carlo_fraction(rng, field, lo, hi, count=400000):
    p = rng.uniform(lo, hi, size=(count, 3))
    inside = field(p) < 0
    return float(np.mean(inside)), float(np.std(inside) / math.sqrt(count))


class ImplicitShapeTests(FieldCase):
    def test_sphere_written_as_length(self):
        g = Fields()
        x, y, z = g.coords()
        field = g.add(SUB, [g.add(LENGTH, [x, y, z]), g.const(10)])
        shape = g.implicit(field, (-12, -12, -12), (12, 12, 12))
        grown = g.add(25, [shape], [1.0])  # OFFSET: an implicit shape is an ordinary exact shape
        handle = self.build(g)
        report = self.analyze(handle, shape)
        self.assertEqual(report.bound, 1.0)  # exactly: Gershgorin of J^T J for LENGTH
        self.assertAlmostEqual(report.quality, 1.0, places=6)
        self.assertEqual((report.value_lo, report.ok), (-10.0, 1))
        lo, hi, lipschitz = self.k.bounds(handle, shape)
        np.testing.assert_array_equal(lo, [-12] * 3)
        np.testing.assert_array_equal(hi, [12] * 3)
        self.assertEqual(lipschitz, 1.0)
        p = self.points_in(-12, 12, 4000)
        np.testing.assert_allclose(self.k.evaluate(handle, shape, p), np.linalg.norm(p, axis=1) - 10, atol=1e-13)
        far = self.points_in(-30, 30, 4000)
        expected = implicit_ref(np.linalg.norm(np.clip(far, -12, 12), axis=1) - 10, box_ref(far, [12, 12, 12]))
        np.testing.assert_allclose(self.k.evaluate(handle, shape, far), expected, atol=1e-13)
        mesh = self.k.mesh(handle, shape, 0.2)
        self.assertEqual((mesh.stats.boundary_edges, mesh.stats.nonmanifold_edges), (0, 0))
        self.assertAlmostEqual(mesh.stats.volume / (4 / 3 * math.pi * 1000), 1, delta=0.003)
        self.assertAlmostEqual(mesh.stats.area / (4 * math.pi * 100), 1, delta=0.01)
        mesh = self.k.mesh(handle, grown, 0.2)
        self.assertEqual(mesh.stats.boundary_edges, 0)
        self.assertAlmostEqual(mesh.stats.volume / (4 / 3 * math.pi * 11 ** 3), 1, delta=0.003)
        hits, distances = self.k.raycast(handle, shape, [[0, 0, 40]], [[0, 0, -1]])
        self.assertTrue(hits[0])
        self.assertAlmostEqual(distances[0], 30, places=4)

    def gyroid_block(self, g, half=20.0, thickness=0.35):
        field = g.add(SUB, [g.add(ABS, [gyroid(g)]), g.const(thickness)])
        lattice = g.implicit(field, (-half,) * 3, (half,) * 3)
        return g.add(INTERSECTION, [g.add(BOX, [], [half, half, half, 0]), lattice], [0])

    def test_gyroid_sheet_in_a_box(self):
        g = Fields()
        block = self.gyroid_block(g)
        handle = self.build(g)
        mesh = self.k.mesh(handle, block, 0.25)
        self.assertEqual((mesh.stats.boundary_edges, mesh.stats.nonmanifold_edges), (0, 0))
        self.assertGreater(mesh.stats.volume, 0)
        fraction, error = monte_carlo_fraction(self.rng, lambda p: np.abs(gyroid_ref(p)) - 0.35, -20, 20)
        self.assertAlmostEqual(mesh.stats.volume / (fraction * 40 ** 3), 1, delta=0.01 + 4 * error / fraction)
        # Signs follow the sheet.
        p = self.points_in(-19.9, 19.9, 20000)
        sheet = np.abs(gyroid_ref(p)) - 0.35
        values = self.k.evaluate(handle, block, p)
        clear = np.abs(sheet) > 1e-9
        np.testing.assert_array_equal(np.sign(values[clear]), np.sign(sheet[clear]))
        # The lattice is max(sheet / L, box_B) with L the bound proven at load.
        report = self.analyze(handle, block - 2)
        self.assertEqual(report.ok, 1)
        self.assertGreaterEqual(report.bound, 1.088)  # the gyroid's steepest slope, sqrt(3) 2 pi / 10
        np.testing.assert_allclose(self.k.evaluate(handle, block - 2, p),
                                   np.maximum(sheet / report.bound, box_ref(p, [20, 20, 20])), rtol=1e-12, atol=1e-12)

    def test_implicit_shapes_cull_like_any_shape(self):
        # Twenty implicit balls in one union (indexed children, whole-batch
        # culling) give exactly the brute-force minimum of their fields.
        g = Fields()
        x, y, z = g.coords()
        centres = self.rng.uniform(-30, 30, size=(20, 3))
        radii = self.rng.uniform(1, 4, size=20)
        balls = []
        for c, r in zip(centres, radii):
            moved = [g.add(SUB, [axis, g.const(value)]) for axis, value in zip((x, y, z), c)]
            field = g.add(SUB, [g.add(LENGTH, moved), g.const(r)])
            balls.append(g.implicit(field, c - r - 1, c + r + 1))
        union = g.add(UNION, balls, [0])
        handle = self.build(g)
        p = self.points_in(-35, 35, 20000)
        expected = np.full(len(p), np.inf)
        for c, r in zip(centres, radii):
            lo, hi = c - r - 1, c + r + 1
            inside = np.linalg.norm(np.clip(p, lo, hi) - c, axis=1) - r
            expected = np.minimum(expected, implicit_ref(inside, box_ref(p - c, [r + 1] * 3)))
        np.testing.assert_allclose(self.k.evaluate(handle, union, p), expected, rtol=0, atol=1e-12)
        mesh = self.k.mesh(handle, union, 0.25)
        self.assertEqual(mesh.stats.boundary_edges, 0)

    def test_deterministic_for_any_thread_count(self):
        g = Fields()
        block = self.gyroid_block(g, half=8.0)
        data = g.to_bytes()
        meshes, values, bounds = [], [], []
        p = self.points_in(-9, 9, 30000)
        for threads in (1, 3, 8):
            handle = self.k.create(data)
            self.handles.append(handle)
            mesh = self.k.mesh(handle, block, 0.2, threads=threads)
            meshes.append((mesh.positions.tobytes(), mesh.triangles.tobytes(), mesh.normals.tobytes()))
            values.append(self.k.evaluate(handle, block, p, threads=threads).tobytes())
            report = self.analyze(handle, block - 2)  # the IMPLICIT node
            bounds.append((report.bound, report.boxes, report.value_lo, report.value_hi, report.gradient_median))
        self.assertEqual(len(set(meshes)), 1)
        self.assertEqual(len(set(values)), 1)
        self.assertEqual(len(set(bounds)), 1)

    def test_large_fields_load_within_a_work_budget(self):
        # Noise from 2,000 waves: a 10,000-instruction tape gets fewer boxes,
        # so the proof at load stays near a second at most.
        g = Fields()
        x, y, z = g.coords()
        total = g.add(SUB, [g.add(LENGTH, [x, y, z]), g.const(8)])
        rng = np.random.default_rng(5)
        for _ in range(2000):
            direction = rng.normal(size=3)
            direction *= rng.uniform(0.5, 2) / np.linalg.norm(direction)
            phase = g.add(ADD, [g.add(ADD, [g.add(MUL, [g.const(direction[0]), x]),
                                            g.add(MUL, [g.const(direction[1]), y])]),
                                g.add(MUL, [g.const(direction[2]), z])])
            total = g.add(ADD, [total, g.add(MUL, [g.const(0.0005), g.add(SIN, [phase])])])
        shape = g.implicit(total, (-10, -10, -10), (10, 10, 10))
        data = g.to_bytes()
        start = time.perf_counter()
        handle = self.k.create(data)
        self.handles.append(handle)
        elapsed = time.perf_counter() - start
        report = self.analyze(handle, shape)
        print(f"\n2,000-wave field: load {elapsed:.3f} s, {report.boxes} boxes, bound {report.bound:.4f}")
        self.assertLess(elapsed, 4.0)
        self.assertLess(report.boxes, 32768)
        self.assertEqual(report.ok, 1)
        self.assertGreaterEqual(report.bound, report.gradient_max)
        mesh = self.k.mesh(handle, shape, 0.5)
        self.assertEqual(mesh.stats.boundary_edges, 0)

    def test_gyroid_block_loads_and_meshes_quickly(self):
        g = Fields()
        block = self.gyroid_block(g)
        data = g.to_bytes()
        start = time.perf_counter()
        handle = self.k.create(data)
        self.handles.append(handle)
        loaded = time.perf_counter()
        mesh = self.k.mesh(handle, block, 0.25)
        meshed = time.perf_counter()
        print(f"\n40 mm gyroid block: load {loaded - start:.3f} s, mesh at 0.25 mm {meshed - loaded:.3f} s, "
              f"{mesh.stats.triangle_count} triangles")
        self.assertLess(meshed - start, 5.0)


def field_corpus(rng, mutations=300):
    """RSDF graphs with fields: lattices, implicit shapes in booleans, fields
    that fail to load, random expressions (as roots and inside IMPLICIT), and
    randomly corrupted copies of them."""
    graphs = []
    g = Fields()
    field = g.add(SUB, [g.add(ABS, [gyroid(g, period=6.0)]), g.const(0.3)])
    lattice = g.implicit(field, (-6, -6, -6), (6, 6, 6))
    block = g.add(INTERSECTION, [g.add(BOX, [], [5, 5, 5, 0.5]), lattice], [0.4])
    g.add(25, [g.add(UNION, [block, g.move(g.add(SPHERE, [], [3]), [6, 0, 0])], [0.5])], [0.2])
    graphs.append(g.to_bytes())
    g = Fields()
    x, y, z = g.coords()
    ball = g.implicit(g.add(SUB, [g.add(LENGTH, [x, y, z]), g.const(4)]), (-5, -5, -5), (5, 5, 5))
    warped = g.add(EVAL_AT, [ball, g.add(ADD, [x, g.add(SIN, [y])]), y, g.add(MUL, [z, g.const(1.5)])])
    g.implicit(g.add(MAX, [warped, g.add(NEG, [g.add(COS, [g.add(MUL, [g.const(3), g.add(ATAN2, [y, x])])])])]),
               (-7, -7, -7), (-1, 7, 7))
    graphs.append(g.to_bytes())
    g = Fields()
    x, _, _ = g.coords()
    g.implicit(g.add(DIV, [g.const(1), x]), (-1, -1, -1), (1, 1, 1))
    graphs.append(g.to_bytes())
    for _ in range(40):
        e = random_expression(rng, 4)
        if e[0] == "shape":
            e = ("add", [e, ("const", 0.0)])
        g = Fields()
        root = write_expression(g, e, {})
        if rng.random() < 0.5:
            g.implicit(root, (-3, -2, -4), (4, 3, 2))
        graphs.append(g.to_bytes())
    sources = list(graphs)
    for trial in range(mutations):
        mutated = bytearray(sources[trial % len(sources)])
        for _ in range(1 + trial % 4):
            mutated[int(rng.integers(16, len(mutated)))] = int(rng.integers(0, 256))
        if trial % 7 == 0:
            mutated = mutated[:int(rng.integers(0, len(mutated)))]
        graphs.append(bytes(mutated))
    return graphs


class RobustnessTests(FieldCase):
    def test_huge_arguments_stay_bounded(self):
        # Arguments far beyond 2^53, where whole numbers are no longer one
        # apart: periodic ranges widen to their full extent and finish.
        g = Fields()
        x, y, z = g.coords()
        wave = g.add(TRIWAVE, [g.add(ADD, [x, g.const(1e20)])], [1.0])
        sine = g.add(SIN, [g.add(ADD, [y, g.const(1e300)])])
        field = g.add(SUB, [g.add(LENGTH, [x, y, z]), g.add(ADD, [g.const(3), g.add(MUL, [g.const(0.1),
                                                                                       g.add(ADD, [wave, sine])])])])
        shape = g.implicit(field, (-5, -5, -5), (5, 5, 5))
        handle = self.build(g)
        report = self.analyze(handle, shape)
        self.assertEqual(report.ok, 1)
        self.assertEqual(self.k.mesh(handle, shape, 0.25).stats.boundary_edges, 0)

    def test_corrupted_field_graphs_never_crash(self):
        loaded = 0
        p = self.points_in(-8, 8, 64)
        for data in field_corpus(self.rng):
            handle = self.k.lib.rsdf_graph_create(data, len(data))
            if not handle:
                self.assertTrue(self.k.error())
                continue
            loaded += 1
            try:
                for node in range(self.k.lib.rsdf_graph_node_count(handle)):
                    if self.k.lib.rsdf_node_dimension(handle, node) == 3:  # (loft profiles are 2D)
                        status = self.k.lib.rsdf_evaluate(handle, node, p.ctypes.data, len(p),
                                                          np.empty(len(p)).ctypes.data, 2)
                        if status != 0:
                            self.assertTrue(self.k.error())
                    report = FieldReport()
                    if self.k.lib.rsdf_field_analyze(handle, node, None, None, 500, ctypes.byref(report)) == 0:
                        self.assertIn(report.ok, (0, 1))
            finally:
                self.k.lib.rsdf_graph_free(handle)
        self.assertGreater(loaded, 40)


class RegionalBoundTests(FieldCase):
    """Shapes read as fields: a twist of an inexact child and a loft of
    inexact profiles grow steeper away from their boxes, so their slope bound
    depends on the box analysed (SDF-DESIGN.md, "Custom fields")."""

    def test_twisted_inexact_child_far_from_its_box(self):
        g = Fields()
        twisted = g.add(TWIST, [g.add(ELLIPSOID, [], [5, 1, 1])], [math.radians(10)])
        root = g.add(MAX, [twisted])
        lo, hi = (-20, -20, -5), (20, 20, 5)
        handle = self.build(g)
        report = self.analyze(handle, root, lo, hi)
        steepest = self.assertSound(handle, root, lo, hi, report, "twisted ellipsoid")
        self.assertGreater(steepest, 2.5)  # 2.2 times the twist's stepping bound near its box
        self.assertGreaterEqual(report.bound, report.gradient_max)
        # Normalized by that bound, the solid is 1-Lipschitz across its whole box.
        g.records.pop()
        solid = g.implicit(g.add(SUB, [twisted, g.const(1.0)]), lo, hi)
        handle = self.build(g)
        slopes, slack, _ = self.secant_slopes(handle, solid, lo, hi)
        self.assertLessEqual(float(np.max(slopes - slack)), 1 + 1e-9)

    def test_nested_twists_keep_the_outer_cylinder(self):
        # The inner twist sees the outer twist's cylinder, not the square box
        # around it, whose corner radius is sqrt(2) larger.
        g = Fields()
        inner = g.add(TWIST, [g.add(ELLIPSOID, [], [4, 1, 1])], [math.radians(12)])
        root = g.add(MAX, [g.add(TWIST, [inner], [math.radians(-7)])])
        lo, hi = (-8, -8, -1), (8, 8, 1)
        handle = self.build(g)
        report = self.analyze(handle, root, lo, hi)
        self.assertSound(handle, root, lo, hi, report, "nested twists")
        self.assertLess(report.bound, 3.5)  # 3.21; 4.26 with the inner radius grown by sqrt(2)

    def test_loft_of_anisotropic_profiles_far_from_its_box(self):
        g = Fields()
        loft = g.add(LOFT, [g.add(ELLIPSE, [], [10, 1]), g.add(CIRCLE, [], [1])], [0, 10])
        root = g.add(MAX, [loft])
        lo, hi = (-200, -200, -200), (200, 200, 200)
        handle = self.build(g)
        report = self.analyze(handle, root, lo, hi)
        self.assertGreater(self.assertSound(handle, root, lo, hi, report, "loft"), 5)

    def test_loft_of_an_empty_profile(self):
        # A profile emptied by a covering cutter or by an inward offset wider
        # than itself is far above the profiles' diagonal inside their box,
        # where |b - a| <= L D needs both profiles nonempty.
        for label in ("covering cutter", "inward offset"):
            g = Fields()
            disc = g.add(CIRCLE, [], [1])
            if label == "covering cutter":
                empty = g.add(DIFFERENCE2, [disc, g.add(CIRCLE, [], [100])], [0])
            else:
                empty = g.add(OFFSET2, [disc], [-50])
            loft = g.add(LOFT, [empty, g.add(CIRCLE, [], [1])], [0, 10])
            root = g.add(MAX, [loft])
            handle = self.build(g)
            for lo, hi in [((-2, -2, 0.5), (2, 2, 9.5)), ((5, 5, 1), (8, 8, 9))]:
                report = self.analyze(handle, root, lo, hi)
                self.assertGreater(self.assertSound(handle, root, lo, hi, report, label), 4)
            # The loft's own stepping bound: rays from inside its box reach
            # the surface (the field's first zero along each ray).
            origins = np.array([[0.0, 0.0, 9.0], [0.2, 0.1, 8.0], [0.0, 0.3, 9.5]])
            directions = np.tile([0.0, 0.0, 1.0], (3, 1))
            hits, distances = self.k.raycast(handle, loft, origins, directions, max_distance=20)
            for origin, hit, distance in zip(origins, hits, distances):
                z = np.linspace(origin[2], 10, 100001)
                values = self.k.evaluate(handle, loft, np.column_stack([np.full_like(z, origin[0]),
                                                                         np.full_like(z, origin[1]), z]))
                self.assertTrue(hit, label)
                self.assertAlmostEqual(distance, z[np.argmax(values <= 0)] - origin[2], delta=1e-3)

    def test_exact_children_keep_their_stepping_bound(self):
        # A twisted box's gradient points away from the box outside it, so the
        # twist's stepping bound holds everywhere.
        g = Fields()
        rate, half = 0.3, (2.0, 2.0, 10.0)
        root = g.add(MAX, [g.add(TWIST, [g.add(BOX, [], [*half, 0])], [rate])])
        handle = self.build(g)
        report = self.analyze(handle, root, (-9, -9, -30), (9, 9, 30))
        w = rate * math.hypot(half[0], half[1])
        self.assertAlmostEqual(report.bound, (w + math.sqrt(w * w + 4)) / 2 / math.sqrt(1 + w * w), places=9)


class ProofWorkTests(FieldCase):
    """Load-time proofs: shared proofs, a per-field cost limit and a per-graph
    work limit, each failing with a clear message."""

    def wide_union(self, g):
        sphere = g.add(SPHERE, [], [0.5])
        return g.add(UNION, [sphere] * 65536, [0])  # 65,537 node visits per point

    def test_copies_of_one_solid_share_its_proof(self):
        # Copies reading one field node, and copies of separately built
        # identical fields (as a loop calling implicit() builds them).
        g = Fields()
        wave = gyroid(g)
        field = g.add(SUB, [g.add(ABS, [wave]), g.const(0.3)])
        lo, hi = (-10, -10, -10), (10, 10, 10)
        copies = [g.move(g.implicit(field, lo, hi), [25.0 * i, 0, 0]) for i in range(200)]
        copies += [g.move(g.implicit(g.add(SUB, [g.add(ABS, [wave]), g.const(0.3)]), lo, hi), [25.0 * i, 30, 0])
                   for i in range(200)]
        g.add(UNION, copies, [0])
        started = time.perf_counter()
        self.build(g)  # one 70 ms proof, not 400
        self.assertLess(time.perf_counter() - started, 5.0)

    def test_fields_too_costly_to_prove(self):
        g = Fields()
        x, y, z = g.coords()
        shape = self.wide_union(g)
        reads = [g.add(EVAL_AT, [shape, g.add(ADD, [x, g.const(float(i))]), y, z]) for i in range(130)]
        field = g.add(MIN, reads)
        g.implicit(field, (-1, -1, -1), (1, 1, 1))
        with self.assertRaisesRegex(KernelError, r"costs \d+ node visits per point; a slope bound can be proven "
                                                 r"only for fields of at most 8388608"):
            self.build(g)
        g.records.pop()
        report = self.analyze(self.build(g), field, (-1, -1, -1), (1, 1, 1))
        self.assertEqual(report.ok, 0)
        self.assertIn("at most 8388608", report.message.decode())

    def test_many_distinct_fields_share_the_work(self):
        # 24 separately bounded cells of a periodic field (that never meets
        # the 1% stop) cost about 2^24 visits each at the default budget, more
        # than the graph's 2^28: each distinct proof gets an equal share of the
        # work left, so the graph loads with looser bounds, and analysing an
        # IMPLICIT over its own bounds reports that proof.
        g = Fields()
        wave = gyroid(g)
        for _ in range(240):  # cheap visits: a chain of negations
            wave = g.add(NEG, [wave])
        field = g.add(SUB, [g.add(ABS, [wave]), g.const(0.3)])
        cells = [g.implicit(field, (10.0 * i - 5, -5, -5), (10.0 * i + 5, 5, 5)) for i in range(24)]
        g.add(UNION, cells, [0])
        handle = self.build(g)
        last = self.analyze(handle, cells[-1])
        grow = 1e-6 * 10
        lo, hi = (230 - 5 - grow, -5 - grow, -5 - grow), (230 + 5 + grow, 5 + grow, 5 + grow)
        full = self.analyze(handle, field, lo, hi)  # its default budget
        self.assertEqual((last.ok, full.ok), (1, 1))
        self.assertLess(last.boxes, full.boxes)
        self.assertGreaterEqual(last.bound, full.bound)
        p = self.points_in((226, -4, -4), (234, 4, 4), 500)
        np.testing.assert_allclose(self.k.evaluate(handle, cells[-1], p), (np.abs(gyroid_ref(p)) - 0.3) / last.bound,
                                   rtol=1e-9, atol=1e-12)

    def test_costly_fields_sample_fewer_points(self):
        # The gradient sample visits at most about 2^26 nodes: a field of
        # about 49,000 visits a point gets a few hundred sample points.
        g = Fields()
        x, y, z = g.coords()
        sphere = g.add(SPHERE, [], [0.5])
        shape = g.add(UNION, [sphere] * 4096, [0])
        field = g.add(MIN, [g.add(EVAL_AT, [shape, g.add(ADD, [x, g.const(float(i))]), y, z]) for i in range(12)])
        handle = self.build(g)
        report = self.analyze(handle, field, (-1, -1, -1), (1, 1, 1))
        self.assertEqual(report.ok, 1)
        self.assertTrue(64 <= report.samples < 4096, report.samples)
        one_box = self.analyze(handle, field, (-1, -1, -1), (1, 1, 1), max_boxes=1)
        self.assertEqual(one_box.samples, report.samples)  # set by the field's cost, not its budget

    def test_graph_proof_work_is_capped(self):
        # Each proof of this 4.2 M-visit field evaluates at least one box and
        # its centre (2^23 visits): 31 distinct proofs fit in 2^28, so 40 fail
        # before any proof runs, stating the visits their first boxes need.
        g = Fields()
        x, y, z = g.coords()
        shape = self.wide_union(g)
        reads = [g.add(EVAL_AT, [shape, g.add(ADD, [x, g.const(float(i))]), y, z]) for i in range(64)]
        field = g.add(MIN, reads)
        solids = [g.implicit(field, (-1 - 1e-3 * i, -1, -1), (1, 1, 1)) for i in range(40)]
        g.add(UNION, solids, [0])
        started = time.perf_counter()
        with self.assertRaisesRegex(KernelError, r"node \d+ \(IMPLICIT\): proving the slope bounds of this design's "
                                                 r"40 distinct IMPLICIT fields needs at least (\d+) node visits \(a "
                                                 r"box and its centre each\), more than the 268435456 one design "
                                                 r"may spend") as caught:
            self.build(g)
        self.assertLess(time.perf_counter() - started, 5.0)
        needed = int(re.search(r"needs at least (\d+)", str(caught.exception)).group(1))
        self.assertTrue(40 * 2 * 64 * 65537 <= needed < 41 * 2 * 64 * 65537, needed)

    def test_cheap_proofs_of_costly_fields_load(self):
        # 40 distinct proofs (one field over 40 cells) of a field reading a
        # 33,000-visit shape each stop after one box: what loading decides by
        # is the work they spend (about 2.6 M visits), not their equal shares
        # of the graph's 2^28 (101 boxes, below 256 and the 254-box default).
        g = Fields()
        sphere = g.add(SPHERE, [], [0.6])
        field = g.add(SUB, [g.add(UNION, [sphere] * 33000, [0]), g.const(0.1)])
        cells = [g.implicit(field, (2.0 * i, -1, -1), (2.0 * i + 2, 1, 1)) for i in range(40)]
        g.add(UNION, cells, [0])
        started = time.perf_counter()
        handle = self.build(g)
        self.assertLess(time.perf_counter() - started, 5.0)
        for cell in (cells[0], cells[-1]):
            report = self.analyze(handle, cell)
            self.assertEqual((report.ok, report.boxes), (1, 1))
            self.assertAlmostEqual(report.bound, 1.0, places=9)


class RangeTests(FieldCase):
    """Values far from the origin, tiny periods and constant clamps."""

    def test_tiny_triangle_wave_periods_stay_finite(self):
        g = Fields()
        x, _, _ = g.coords()
        period = 1e-300
        wave = g.add(TRIWAVE, [x], [period])
        handle = self.build(g)
        values = self.k.evaluate(handle, wave, [[1.0, 0, 0], [-1.0, 0, 0], [1e10, 0, 0], [0.3, 0, 0]])
        self.assertTrue(np.all(np.isfinite(values)) and np.all(values >= 0) and np.all(values <= period))
        report = self.analyze(handle, wave, (-1, -1, -1), (1, 1, 1))
        self.assertEqual(report.ok, 1)
        self.assertGreaterEqual(report.value_lo, 0.0)
        self.assertLessEqual(report.value_hi, period)

    def test_triangle_waves_of_huge_arguments_match_exact_remainders(self):
        g = Fields()
        x, _, _ = g.coords()
        wave = g.add(TRIWAVE, [x], [0.7])
        handle = self.build(g)
        a = np.concatenate([self.rng.uniform(-1e20, 1e20, 200), self.rng.uniform(-1e6, 1e6, 200)])
        values = self.k.evaluate(handle, wave, np.column_stack([a, np.zeros_like(a), np.zeros_like(a)]))
        rest = np.fmod(a, 1.4)
        np.testing.assert_allclose(values, np.abs(np.where(rest < 0, rest + 1.4, rest) - 0.7), rtol=0, atol=1e-15)

    def test_constant_clamps_and_smooth_extrema_keep_the_slope(self):
        g = Fields()
        x, y, z = g.coords()
        ball = lambda dx: g.add(SUB, [g.add(LENGTH, [g.add(SUB, [x, g.const(dx)]), y, z]), g.const(4.0)])  # noqa: E731
        constant = g.add(ADD, [g.add(CLAMP, [x], [3, 3]), ball(0.0)])
        blend = g.add(SMIN, [ball(3.0), ball(-3.0)], [0.5])
        cut = g.add(SMAX, [ball(3.0), g.add(NEG, [ball(-3.0)])], [2.0])
        handle = self.build(g)
        for node in (constant, blend, cut):
            report = self.analyze(handle, node, (-9, -6, -6), (9, 6, 6))
            self.assertSound(handle, node, (-9, -6, -6), (9, 6, 6), report)
            self.assertEqual(report.bound, 1.0)

    def test_field_roots_have_no_value_at_unproven_warp_points(self):
        g = Fields()
        x, y, z = g.coords()
        root = g.add(EVAL_AT, [g.add(SPHERE, [], [1.0]), g.add(LOG, [x]), y, z])
        far = g.add(EVAL_AT, [g.add(SPHERE, [], [1.0]), g.add(MUL, [x, g.const(1e200)]), y, z])
        handle = self.build(g)
        values = self.k.evaluate(handle, root, [[-1.0, 0, 0], [1.0, 0, 0]])
        self.assertTrue(np.isnan(values[0]))
        self.assertAlmostEqual(values[1], -1.0)
        self.assertTrue(np.isnan(self.k.evaluate(handle, far, [[2.0, 0, 0]])[0]))

    def test_polygons_far_away_are_outside(self):
        g = Fields()
        polygon = g.add(POLYGON, [], [0, 0, 1, 0, 1, 1, 0.5, 0.4, 0, 1])
        extruded = g.add(EXTRUDE, [polygon], [-1, 1])
        revolved = g.add(REVOLVE, [polygon], [0.5])
        handle = self.build(g)
        for far in (1e16, 1e20, 1e100):
            points = [[0.3, far, 0], [far, 0.3, 0], [-far, 0.5, 0], [0.5, -far, 0]]
            self.assertTrue(np.all(self.k.evaluate(handle, extruded, points) > 0), far)
            self.assertTrue(np.all(self.k.evaluate(handle, revolved, [[0, 0, far], [0, 0, -far]]) > 0), far)


@unittest.skipUnless(os.environ.get("RAD_RUN_SANITIZER_FUZZ") == "1", "set RAD_RUN_SANITIZER_FUZZ=1")
class FieldSanitizerTests(unittest.TestCase):
    """The field corpus through a native driver built with AddressSanitizer
    and UndefinedBehaviorSanitizer, and with ThreadSanitizer."""

    def run_driver(self, flags, environment):
        kernel = Path(__file__).resolve().parents[2] / "Sources" / "RADSDFKernel"
        with tempfile.TemporaryDirectory() as directory:
            folder = Path(directory)
            driver = folder / "field_driver"
            built = subprocess.run(["clang++", "-std=c++17", "-O1", "-g", "-fno-math-errno", "-fno-omit-frame-pointer",
                                    *flags, "-I", str(kernel / "include"), "-I", str(kernel),
                                    *map(str, sorted(kernel.glob("*.cpp"))),
                                    str(Path(__file__).with_name("field_sanitizer_driver.cpp")), "-o", str(driver)],
                                   capture_output=True, text=True)
            self.assertEqual(built.returncode, 0, built.stderr[-4000:])
            paths = []
            for index, data in enumerate(field_corpus(np.random.default_rng(99), mutations=120)):
                path = folder / f"graph-{index}.rsdf"
                path.write_bytes(data)
                paths.append(str(path))
            run = subprocess.run([str(driver), *paths], capture_output=True, text=True,
                                 env={**os.environ, **environment})
            report = run.stdout[-2000:] + run.stderr[-6000:]
            self.assertEqual(run.returncode, 0, report)
            for marker in ("ThreadSanitizer", "AddressSanitizer", "runtime error"):
                self.assertNotIn(marker, run.stderr, report)

    def test_fields_have_no_memory_or_arithmetic_faults(self):
        self.run_driver(["-fsanitize=address,undefined", "-fno-sanitize-recover=undefined"],
                        {"ASAN_OPTIONS": "detect_leaks=0:abort_on_error=0"})

    def test_fields_have_no_data_races(self):
        self.run_driver(["-fsanitize=thread"], {"TSAN_OPTIONS": "halt_on_error=1"})
