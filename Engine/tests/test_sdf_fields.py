"""Custom fields (traced expressions) above the kernel: radsdf's Field API,
implicit() and verify_field, and the worker's builds, design errors and
custom field verification (rad_verify).

Tracing and encoding need no kernel; evaluation, proofs and builds run the
native kernel (RAD_SDF_LIBRARY, or the bindings' search path) and are
skipped when it cannot be loaded.
"""
import json
import math
from pathlib import Path
import struct
import sys
import unittest

import numpy as np

ENGINE = Path(__file__).resolve().parents[1]
if str(ENGINE) not in sys.path:
    sys.path.insert(0, str(ENGINE))

import radsdf  # noqa: E402
from radsdf import (Field, Opcode, SDFError, atan2, box, clamp, cos, exp, implicit, intersection,  # noqa: E402
                    length, log, maximum, minimum, sin, sphere, sqrt, triwave, union, verify_field, x, y, z)
from radsdf_assembly import Assembly  # noqa: E402
import sdf_build  # noqa: E402
from sdf_build import ModelError  # noqa: E402
from sdf_test_support import assert_closed_and_outward, build, native, temporary_folder  # noqa: E402

GYROID = """k = 2 * math.pi / 10.0                        # 10 mm cells
g = sin(k*x)*cos(k*y) + sin(k*y)*cos(k*z) + sin(k*z)*cos(k*x)
lattice = implicit(abs(g) - 0.35, bounds=((-20, -20, -20), (20, 20, 20)))
result = intersection(box(40), lattice)
"""
# A 60 x 10 x 10 bar bent around the Z axis through (0, 40): arc length
# along the bar's centre circle (radius 40) becomes the bar's x.
WARPED = """bar = box((60, 10, 10))
r = 40.0
bent = bar.at(r * atan2(x, r - y), r - length(x, r - y), z)
result = Assembly("Warped")
result.add(implicit(bent, bounds=((-32, -6, -6), (32, 16, 6))), id="bent", name="Bent bar")
result.add(sphere(3, center=(0, 30, 0)), id="ball", name="Ball")
"""
POLE = """f = 1 / x
result = implicit(f, ((-1, -1, -1), (1, 1, 1)))
"""
GYROID_SLOPE = math.sqrt(3) * 2 * math.pi / 10  # the gyroid's steepest slope


def kernel_available():
    try:
        radsdf.kernel_version()
        return True
    except SDFError:
        return False


def needs_kernel(case):
    return unittest.skipUnless(kernel_available(), "libRADSDF.dylib not built")(case)


def records(field):
    """(opcode, children, params) of every RSDF record of a field."""
    data, _ = radsdf.serialize(field)
    _, _, count, _ = struct.unpack_from("<4sIII", data)
    offset, found = 16, []
    for _ in range(count):
        opcode, children, params, _ = struct.unpack_from("<4I", data, offset)
        offset += 16
        kids = struct.unpack_from(f"<{children}I", data, offset)
        offset += 4 * children
        values = struct.unpack_from(f"<{params}d", data, offset)
        offset += 8 * params
        found.append((opcode, kids, values))
    return found


class TracingTests(unittest.TestCase):
    """Fields record native expression nodes; no kernel is needed."""

    def test_operations_record_their_opcodes(self):
        cases = [(x + 1, Opcode.ADD), (2 - y, Opcode.SUB), (x * y, Opcode.MUL), (x / 2, Opcode.DIV), (-z, Opcode.NEG),
                 (abs(x), Opcode.ABS), (x ** 2, Opcode.SQUARE), (x ** 3, Opcode.POWC), (sqrt(x), Opcode.SQRT),
                 (sin(x), Opcode.SIN), (cos(x), Opcode.COS), (exp(x), Opcode.EXP), (log(x), Opcode.LOG),
                 (atan2(y, x), Opcode.ATAN2), (minimum(x, y, 1), Opcode.MIN), (maximum(x, y), Opcode.MAX),
                 (clamp(x, 0, 1), Opcode.CLAMP), (length(x, y, z), Opcode.LENGTH), (triwave(x, 2), Opcode.TRIWAVE),
                 (sphere(1).at(x, y, 2 * z), Opcode.EVAL_AT), (np.sin(x), Opcode.SIN),
                 (np.float64(2) * x, Opcode.MUL), (np.hypot(x, y), Opcode.LENGTH), (2 ** x, Opcode.EXP)]
        for field, opcode in cases:
            self.assertIsInstance(field, Field)
            self.assertEqual(records(field)[-1][0], opcode, opcode.name)
        self.assertEqual(records(x ** 3)[-1][2], (3.0,))
        self.assertEqual(records(clamp(x, -1, 2))[-1][2], (-1.0, 2.0))
        self.assertEqual(records(triwave(x, 2.5))[-1][2], (2.5,))
        self.assertIs(+x, x)
        self.assertIs(x ** 1, x)
        self.assertEqual(records(x ** 0), [(Opcode.CONST, (), (1.0,))])
        self.assertIs(minimum(y), y)

    def test_shapes_are_read_directly_and_shared_subexpressions_encode_once(self):
        ball = sphere(2)
        self.assertIs(ball.field._node, ball, "A shape's distance needs no extra node.")
        g = sin(x)
        found = records(g * g + g + ball.field - ball.field)
        self.assertEqual(sum(1 for opcode, _, _ in found if opcode == Opcode.SIN), 1)
        self.assertEqual(sum(1 for opcode, _, _ in found if opcode == Opcode.SPHERE), 1)
        self.assertEqual(sum(1 for opcode, _, _ in found if opcode == Opcode.COORD), 1)
        # Sixty doublings describe 2**60 paths in sixty nodes.
        field = x
        for _ in range(60):
            field = field + field
        self.assertEqual(len(records(field)), 61)

    def test_implicit_is_a_bounded_solid_node(self):
        lattice = implicit(sin(x) - 0.5, bounds=((-1, -2, -3), (1, 2, 3)))
        self.assertIsInstance(lattice, radsdf.SDF3)
        self.assertEqual(lattice.opcode, Opcode.IMPLICIT)
        self.assertEqual(lattice.params, (-1.0, -2.0, -3.0, 1.0, 2.0, 3.0))
        self.assertEqual(implicit(x, radsdf.Bounds((0, 0, 0), (1, 1, 1))).params, (0.0, 0.0, 0.0, 1.0, 1.0, 1.0))
        moved = intersection(box(4), lattice).translate(1, 0, 0)  # an ordinary solid from here on
        self.assertEqual(moved.opcode, Opcode.TRANSFORM)
        self.assertEqual(implicit(sphere(1), ((-2, -2, -2), (2, 2, 2))).children[0].opcode, Opcode.SPHERE)

    def test_polymorphic_functions_keep_numbers_numbers(self):
        self.assertEqual(sin(0.5), math.sin(0.5))
        self.assertIsInstance(sin(0.5), float)
        self.assertEqual(cos(1), math.cos(1))
        self.assertEqual(exp(1), math.e)
        self.assertEqual(log(math.e), 1.0)
        self.assertEqual(sqrt(9), 3.0)
        self.assertEqual(atan2(1, -1), math.atan2(1, -1))
        self.assertEqual(minimum(3, 1, 2), 1)
        self.assertEqual(maximum([3, 7, 2]), 7)
        self.assertEqual(clamp(5, 0, 2), 2)
        self.assertEqual(length(3, 4), 5.0)
        self.assertEqual(length((1, 2, 2)), 3.0)
        self.assertEqual(triwave(0, 1), 1)
        self.assertEqual(triwave(2.5, 1), 0.5)
        self.assertEqual(triwave(-0.25, 1), 0.75)
        values = np.linspace(-3, 3, 7)
        np.testing.assert_allclose(sin(values), np.sin(values))
        np.testing.assert_allclose(minimum(values, 0), np.minimum(values, 0))
        np.testing.assert_allclose(clamp(values, -1, 1), np.clip(values, -1, 1))
        np.testing.assert_allclose(length(values, values), np.hypot(values, values))
        np.testing.assert_allclose(triwave(values, 1.5), np.abs(np.mod(values, 3) - 1.5))
        with self.assertRaises(ValueError):
            sqrt(-1)  # numbers keep math's domain errors

    def test_program_namespace_keeps_math_and_numpy(self):
        namespace = sdf_build.program_namespace(sdf_build.RADBuildMode(False))
        for name in ("Field", "x", "y", "z", "sin", "cos", "exp", "log", "sqrt", "atan2", "minimum", "maximum",
                     "clamp", "length", "triwave", "implicit", "verify_field"):
            self.assertIs(namespace[name], getattr(radsdf, name), name)
        self.assertIs(namespace["math"], math)
        self.assertIs(namespace["np"], np)
        script = ("a = math.sin(0.5) + math.sqrt(2.0)\n"
                  "b = np.sin(np.linspace(0, 1, 5)).sum() + np.sqrt(4.0)\n"
                  "c = sin(0.5) + sqrt(2.0) + min(1, 2) + max(3, 4) + abs(-1)\n"
                  "d = sin(k * x) if (k := 2.0) else None\n")
        exec(compile(script, "<RAD model>", "exec"), namespace, namespace)
        self.assertEqual(namespace["a"], math.sin(0.5) + math.sqrt(2.0))
        self.assertAlmostEqual(namespace["b"], float(np.sin(np.linspace(0, 1, 5)).sum()) + 2.0)
        self.assertEqual(namespace["c"], math.sin(0.5) + math.sqrt(2.0) + 1 + 4 + 1)
        self.assertIsInstance(namespace["d"], Field)

    def test_misuse_explains_the_rule(self):
        ball = sphere(1)
        cases = [
            (lambda: x > 0, "minimum(), maximum() or clamp()"),
            (lambda: x == y, "minimum(), maximum() or clamp()"),
            (lambda: bool(x), "minimum(), maximum() or clamp()"),
            (lambda: x if x else y, "minimum(), maximum() or clamp()"),
            (lambda: min(x, y), "minimum(), maximum() or clamp()"),
            (lambda: math.sin(x), "radsdf's sin"),
            (lambda: float(x), "radsdf's sin"),
            (lambda: x % 2, "triwave"),
            (lambda: x // 2, "triwave"),
            (lambda: x / 0, "divided by zero"),
            (lambda: x ** y, "exponent of a custom field must be a number"),
            (lambda: (-2) ** x, "positive base"),
            (lambda: x + ball, "shape.field"),
            (lambda: sin(ball), "shape.field"),
            (lambda: x + np.ones(3), "not arrays"),
            (lambda: x * math.inf, "finite"),
            (lambda: x + True, "not booleans"),
            (lambda: np.tan(x), "cannot take a custom field"),
            (lambda: ball - x, "implicit(field, bounds)"),
            (lambda: union(ball, x), "implicit(field, bounds)"),
            (lambda: x.translate(1, 0, 0), "has no attribute"),
            (lambda: sphere(x), "must be a number"),
            (lambda: triwave(x, 0), "greater than zero"),
            (lambda: triwave(x, y), "must be a number"),
            (lambda: clamp(x, 2, 1), "lo <= hi"),
            (lambda: implicit(x, ((0, 0, 0), (1, 1, math.inf))), "finite"),
            (lambda: implicit(x, ((0, 0, 0), (1, 0, 1))), "lo < hi"),
            (lambda: implicit(x, ((0, 0), (1, 1))), "3D"),
            (lambda: implicit(1.5, ((0, 0, 0), (1, 1, 1))), "needs a custom field"),
            (lambda: implicit(radsdf.circle(1), ((0, 0, 0), (1, 1, 1))), "3D field"),
            (lambda: radsdf.circle(1).at(x, y, z), "has no attribute"),
            (lambda: ball.at(x, y), "three coordinates"),
            (lambda: Assembly().add(x), "implicit(field, bounds)"),
            (lambda: verify_field(x), "needs the bounds"),
        ]
        for action, message in cases:
            with self.subTest(message=message):
                with self.assertRaises((SDFError, AttributeError, TypeError)) as caught:
                    action()
                self.assertIn(message, str(caught.exception))
        with self.assertRaises(AttributeError):
            x._node = None


@needs_kernel
class EvaluationTests(unittest.TestCase):
    """The kernel evaluates traced fields exactly as Python arithmetic would."""

    def test_fields_evaluate_like_numpy(self):
        points = np.random.default_rng(7).uniform(-2, 2, (200, 3))
        px, py, pz = points.T
        ball = sphere(1.5, center=(0.5, 0, 0))
        cases = [
            (x * y - z / 3 + 1, px * py - pz / 3 + 1),
            (abs(x) ** 3 + y ** 2, np.abs(px) ** 3 + py ** 2),
            (sin(2 * x) * cos(y) + exp(z / 4), np.sin(2 * px) * np.cos(py) + np.exp(pz / 4)),
            (sqrt(x * x + 1) + log(y * y + 1), np.sqrt(px * px + 1) + np.log(py * py + 1)),
            (atan2(y, x + 3), np.arctan2(py, px + 3)),
            (minimum(x, y, z) - maximum(x, 0.5), np.minimum(np.minimum(px, py), pz) - np.maximum(px, 0.5)),
            (clamp(x, -0.5, 0.5) + clamp(y, x, 1), np.clip(px, -0.5, 0.5) + np.maximum(px, np.minimum(py, 1))),
            (length(x, y, z) - 1, np.sqrt(px ** 2 + py ** 2 + pz ** 2) - 1),
            (triwave(x, 0.75), np.abs(np.mod(px, 1.5) - 0.75)),
            (ball.field * 2, 2 * (np.linalg.norm(points - (0.5, 0, 0), axis=1) - 1.5)),
            (ball.at(x + 1, 2 * y, z), np.linalg.norm(np.stack([px + 0.5, 2 * py, pz], axis=1), axis=1) - 1.5),
        ]
        for field, expected in cases:
            np.testing.assert_allclose(field.evaluate(points), expected, rtol=1e-12, atol=1e-12)
        self.assertEqual(x(np.array([1.5, 2.0, 3.0])), 1.5)

    def test_implicit_normalizes_by_the_proven_bound(self):
        k = 2 * math.pi / 10
        g = sin(k * x) * cos(k * y) + sin(k * y) * cos(k * z) + sin(k * z) * cos(k * x)
        bounds = ((-20, -20, -20), (20, 20, 20))
        lattice = implicit(abs(g) - 0.35, bounds)
        report = verify_field(lattice)
        self.assertTrue(report["ok"])
        self.assertGreaterEqual(report["bound"], GYROID_SLOPE)
        self.assertLess(report["bound"], 1.6)
        self.assertEqual(report["bound"], verify_field(abs(g) - 0.35, bounds)["bound"])
        points = np.random.default_rng(3).uniform(-19, 19, (100, 3))
        np.testing.assert_allclose(lattice.evaluate(points),
                                   (np.abs(g.evaluate(points)) - 0.35) / report["bound"], rtol=1e-12, atol=1e-15)
        self.assertGreater(lattice.evaluate((30.0, 0.0, 0.0)), 9.99, "Outside the bounds: their box's distance.")
        self.assertEqual(lattice.bounds.as_list(), [[-20, -20, -20], [20, 20, 20]])
        self.assertEqual(lattice.lipschitz, 1.0)
        # The graph a build loads proves the same bound.
        data, roots = radsdf.serialize([lattice])
        with radsdf.Graph.from_bytes(data, roots) as graph:
            self.assertEqual(graph.analyze_field(roots[0])["bound"], report["bound"])

    def test_verify_field_reports_quality_and_failures(self):
        sphere_report = verify_field(length(x, y, z) - 5, ((-6, -6, -6), (6, 6, 6)))
        self.assertEqual(sphere_report["bound"], 1.0)
        self.assertAlmostEqual(sphere_report["quality"], 1.0, places=4)
        self.assertEqual(sphere_report["value_range"][0], -5.0)
        self.assertEqual(sphere_report["samples"], 4096)
        self.assertEqual(verify_field(sphere(5).field, ((-6, -6, -6), (6, 6, 6)))["bound"], 1.0)
        stretched = verify_field(2 * x + 1, ((0, 0, 0), (1, 1, 1)))
        self.assertEqual(stretched["bound"], 2.0)
        # Analysed over the bounds grown by 1e-6 of their extent, as implicit() is.
        np.testing.assert_allclose(stretched["value_range"], (1 - 2e-6, 3 + 2e-6), rtol=0, atol=1e-12)
        failed = verify_field(1 / x, ((-1, -1, -1), (1, 1, 1)))
        self.assertFalse(failed["ok"])
        self.assertEqual(failed["bound"], math.inf)
        self.assertEqual(failed["quality"], 0.0)
        self.assertTrue(failed["message"].startswith("a division (/): the divisor may be 0 in the region"))
        self.assertNotIn("node", failed["message"])
        lo, hi = failed["region"]
        self.assertTrue(lo[0] <= 0 <= hi[0])
        for field, problem in [(sqrt(x), "sqrt()"), (log(x + 1), "log()"), (atan2(y, x), "atan2()"),
                               (x ** -1, "a power (**)")]:
            report = verify_field(field, ((-1, -1, -1), (1, 1, 1)))
            self.assertFalse(report["ok"])
            self.assertTrue(report["message"].startswith(problem), report["message"])
        # Away from the pole the same fields have bounds.
        self.assertTrue(verify_field(1 / x, ((1, -1, -1), (2, 1, 1)))["ok"])
        # The angle jumps across the negative x axis; cos of a whole multiple does not.
        across = ((-2, -1, -1), (-0.5, 1, 1))
        self.assertTrue(verify_field(cos(6 * atan2(y, x)) * length(x, y), across)["ok"])
        self.assertIn("atan2 jumps", verify_field(atan2(y, x), across)["message"])

    def test_loading_an_unbounded_field_names_the_construct(self):
        with self.assertRaises(SDFError) as caught:
            implicit(1 / (x - 0.5), ((-1, -1, -1), (1, 1, 1))).bounds
        message = str(caught.exception)
        self.assertIn("implicit(): its field has no finite slope bound", message)
        self.assertIn("a division (/): the divisor may be 0", message)
        self.assertNotRegex(message, r"node \d+")

    def test_implicit_solids_mesh_and_combine(self):
        ball = implicit(length(x, y, z) - 5, ((-6, -6, -6), (6, 6, 6)))
        mesh = ball.mesh(cell_size=0.25)
        assert_closed_and_outward(self, mesh.positions, mesh.triangles)
        self.assertAlmostEqual(mesh.stats["volume"], 4 / 3 * math.pi * 125, delta=0.01 * 4 / 3 * math.pi * 125)
        shell = ball.shell(1)  # a true distance (quality 1) offsets exactly
        self.assertAlmostEqual(shell.evaluate((4.5, 0.0, 0.0)), -0.5, places=9)
        hits, distances = ball.raycast((-20.0, 0.0, 0.0), (1.0, 0.0, 0.0))
        self.assertTrue(hits)
        self.assertAlmostEqual(distances, 15.0, places=3)


@needs_kernel
class BuildTests(unittest.TestCase):
    """The worker builds custom fields, reports them and explains failures."""

    def test_gyroid_part_builds_and_reports_its_field(self):
        result, folder = build(self, GYROID, threads=4)
        self.assertEqual(result["solids"], 1)
        self.assertEqual(result["mesh"]["boundaryEdges"], 0)
        field, = result["customFields"]
        self.assertEqual(set(field), {"solids", "line", "bounds", "slopeBound", "quality"})
        self.assertEqual((field["solids"], field["line"]), ([0], 3))
        self.assertEqual(field["bounds"], [-20, -20, -20, 20, 20, 20])
        self.assertGreaterEqual(field["slopeBound"], GYROID_SLOPE)
        self.assertLess(field["slopeBound"], 1.6)
        self.assertTrue(0.3 < field["quality"] <= 1)
        # A sheet of half-width 0.35 / |grad g| fills a fixed share of the cube.
        self.assertGreater(result["volume"], 0.15 * 40 ** 3)
        self.assertLess(result["volume"], 0.35 * 40 ** 3)
        published = json.loads((folder / "result.json").read_text())
        self.assertEqual(published["customFields"], result["customFields"])

    def test_warped_part_builds_with_the_bent_volume(self):
        result, _ = build(self, WARPED)
        self.assertEqual(result["solids"], 2)
        field, = result["customFields"]
        self.assertEqual((field["solids"], field["line"]), ([0], 5))
        bent = result["partStats"][0]
        # Bending keeps the volume: (1.5 rad / 2) * (45^2 - 35^2) * 10 = 6000 mm^3.
        self.assertAlmostEqual(bent["volume"], 6000, delta=60)
        self.assertEqual(bent["components"], 1)
        self.assertAlmostEqual(bent["bounds"][1], -5, delta=0.1)  # the bar's middle stays at y = -5..5
        self.assertAlmostEqual(bent["bounds"][4], 40 - 35 * math.cos(0.75), delta=0.1)

    def test_design_errors_name_the_part_and_line(self):
        cases = [
            (POLE, ["The design: the custom field of implicit() at line 2 has no finite slope bound within its "
                    "bounds ((-1, -1, -1), (1, 1, 1)): a division (/): the divisor may be 0 in the region",
                    "rad_verify"]),
            ("result = sin(x) + y\n", ["result is a custom field", "implicit(field, bounds)"]),
            ("result = [box(2), x]\n", ["result[1] is a custom field"]),
            ("result = Assembly()\nresult.add(x - 1)\n", ["line 2", "implicit(field, bounds)"]),
            ("result = union(box(2), y)\n", ["line 1", "implicit(field, bounds)"]),
            ("result = box(2) if x > 0 else box(3)\n", ["line 1", "minimum(), maximum() or clamp()"]),
            ("a = Assembly()\na.add(box(1), id='cube', name='Cube')\n"
             "a.add(implicit(sqrt(x * x + y * y) - 1, ((-2, -2, -2), (2, 2, 2))), id='rod', name='Rod')\nresult = a\n",
             ['Part "Rod"', "implicit() at line 3", "sqrt()", "use length() for distances"]),
        ]
        for script, details in cases:
            with self.subTest(script=script):
                with self.assertRaises(ModelError) as caught:
                    sdf_build.build(spec(script), ENGINE, temporary_folder(self), 2)
                for detail in details:
                    self.assertIn(detail, str(caught.exception))
                self.assertNotRegex(str(caught.exception), r"node \d+")

    def test_summary_lists_at_most_32_fields(self):
        script = ("balls = [implicit(length(x - 3 * i, y, z) - 1, ((3 * i - 1.5, -1.5, -1.5), (3 * i + 1.5, 1.5, 1.5)))\n"
                  "         for i in range(40)]\n"
                  "result = union(*balls)\n")
        result = sdf_build.build(spec(script), ENGINE, temporary_folder(self), 4)
        self.assertEqual(len(result["customFields"]), 32)
        self.assertTrue(all(field["slopeBound"] == 1.0 and field["line"] == 1 for field in result["customFields"]))
        self.assertIn("40 implicit() shapes; the summary lists the first 32", result["warnings"][-1])
        verified = sdf_build.build(spec(script, verifyFields=True), ENGINE, temporary_folder(self), 4)
        self.assertEqual((verified["fieldCount"], len(verified["fields"])), (40, 40))

    def test_designs_without_fields_report_none(self):
        result = sdf_build.build(spec("result = box(2)\n"), ENGINE, temporary_folder(self), 2)
        self.assertNotIn("customFields", result)


@needs_kernel
class VerifyFieldsTests(unittest.TestCase):
    """rad_verify's build mode: run the program, analyse every implicit()
    shape without loading or meshing the design."""

    def verify(self, script, **extra):
        folder = temporary_folder(self)
        result = sdf_build.build(spec(script, verifyFields=True, **extra), ENGINE, folder, 2)
        self.assertEqual(json.loads((folder / "result.json").read_text()), result)
        self.assertEqual(sorted(path.name for path in folder.iterdir()), ["result.json"], "No mesh or graph.")
        return result

    def test_reports_bound_quality_and_owners(self):
        result = self.verify(WARPED)
        self.assertEqual(set(result), {"solids", "parts", "fields", "fieldCount", "buildSeconds", "threads", "runtime"})
        self.assertEqual((result["solids"], result["fieldCount"]), (2, 1))
        field, = result["fields"]
        self.assertTrue(field["ok"])
        self.assertEqual((field["solids"], field["line"]), ([0], 5))
        self.assertEqual(field["bounds"], [-32, -6, -6, 32, 16, 6])
        self.assertGreaterEqual(field["slopeBound"], field["gradient"]["max"])
        self.assertEqual(set(field["gradient"]), {"min", "median", "max"})
        self.assertAlmostEqual(field["gradient"]["median"], 1.0, places=3, msg="Bending keeps distances near the bar.")
        self.assertAlmostEqual(field["quality"], field["gradient"]["median"] / field["slopeBound"], places=12)
        built, _ = build(self, WARPED)
        self.assertEqual(built["customFields"][0]["slopeBound"], field["slopeBound"])

    def test_failures_are_reported_not_raised(self):
        result = self.verify("good = implicit(length(x, y, z) - 1, ((-2, -2, -2), (2, 2, 2)))\n"
                             "bad = implicit(log(x), ((-1, -1, -1), (1, 1, 1)))\n"
                             "result = [good, bad.translate(5, 0, 0)]\n")
        good, bad = result["fields"]
        self.assertTrue(good["ok"])
        self.assertEqual(good["slopeBound"], 1.0)
        self.assertEqual((bad["ok"], bad["solids"], bad["line"]), (False, [1], 2))
        self.assertTrue(bad["message"].startswith("log(): the argument of log may be 0 or negative in the region"))
        self.assertNotIn("slopeBound", bad)
        self.assertLessEqual(bad["region"][0], 0)
        self.assertEqual(len(bad["region"]), 6)

    def test_shared_fields_list_every_part_and_nested_failures_explain(self):
        shared = self.verify("ball = implicit(length(x, y, z) - 1, ((-2, -2, -2), (2, 2, 2)))\n"
                             "result = [ball, ball.translate(4, 0, 0)]\n")
        self.assertEqual(shared["fields"][0]["solids"], [0, 1])
        nested = self.verify("inner = implicit(1 / x, ((-1, -1, -1), (1, 1, 1)))\n"
                             "outer = implicit(inner.field * 2, ((-1, -1, -1), (1, 1, 1)))\n"
                             "result = outer\n")
        inner, outer = nested["fields"]
        self.assertFalse(inner["ok"])
        self.assertFalse(outer["ok"])
        self.assertIn("implicit() at line 1", outer["message"])

    def test_programs_without_fields_and_request_validation(self):
        self.assertEqual(self.verify("result = box(2)\n")["fields"], [])
        with self.assertRaises(ModelError):
            sdf_build.validate_spec(spec("result = box(2)\n", verifyFields=False))
        with self.assertRaises(ModelError) as caught:
            self.verify("result = x\n")
        self.assertIn("custom field", str(caught.exception))

    def test_worker_runs_verification_requests(self):
        folder = temporary_folder(self) / "verify"
        folder.mkdir()
        (folder / "request.json").write_text(json.dumps(spec(POLE, verifyFields=True)))
        import worker
        result = worker.run(folder / "request.json", folder, 2)
        self.assertFalse(result["fields"][0]["ok"])
        self.assertTrue((folder / "result.json").is_file())


class ReviewRegressionTests(unittest.TestCase):
    """Bounds of shapes read far from their boxes, limits, blends and cavities."""

    def test_twisted_inexact_shapes_far_from_their_box(self):
        twisted = radsdf.ellipsoid((5, 1, 1)).twist(10)
        bounds = ((-20, -20, -5), (20, 20, 5))
        report = verify_field(twisted.field, bounds)
        self.assertTrue(report["ok"])
        self.assertGreaterEqual(report["bound"], report["gradient"]["max"])
        rng = np.random.default_rng(3)
        a = rng.uniform(bounds[0], bounds[1], (100_000, 3))
        direction = rng.normal(size=a.shape)
        b = a + 1e-3 * direction / np.linalg.norm(direction, axis=1)[:, None]
        slopes = np.abs(twisted.evaluate(a) - twisted.evaluate(b)) / np.linalg.norm(a - b, axis=1)
        self.assertGreater(slopes.max(), 2.5)
        self.assertLessEqual(slopes.max(), report["bound"])
        solid = implicit(twisted.field - 1, bounds)
        inside = (np.all(a > np.array(bounds[0]) + 1e-2, axis=1) & np.all(a < np.array(bounds[1]) - 1e-2, axis=1))
        normalized = np.abs(solid.evaluate(a) - solid.evaluate(b)) / np.linalg.norm(a - b, axis=1)
        self.assertLessEqual(normalized[inside].max(), 1 + 1e-6)

    def test_smooth_extrema_blend_like_unions_and_keep_the_slope(self):
        a, b = length(x - 3, y, z) - 4, length(x + 3, y, z) - 4
        bounds = ((-9, -6, -6), (9, 6, 6))
        blend = minimum(a, b, smooth=0.5)
        self.assertEqual(blend._node.opcode, Opcode.SMIN)
        self.assertEqual(verify_field(blend, bounds)["bound"], 1.0)
        self.assertEqual(verify_field(maximum(a, -b, smooth=2), bounds)["bound"], 1.0)
        shapes = union(sphere(4, center=(3, 0, 0)), sphere(4, center=(-3, 0, 0)), smooth=0.5)
        points = np.random.default_rng(5).uniform((-8.9, -5.9, -5.9), (8.9, 5.9, 5.9), (2000, 3))
        np.testing.assert_allclose(implicit(blend, bounds).evaluate(points), shapes.evaluate(points), atol=1e-12)
        self.assertAlmostEqual(minimum(1.0, 1.2, smooth=0.5), 0.955)
        np.testing.assert_allclose(maximum(np.array([1.0, 3.0]), 1.2, smooth=0.5), [1.245, 3.0])
        self.assertIs(minimum(a, smooth=0.5), a)
        with self.assertRaisesRegex(SDFError, "smooth must not be negative"):
            minimum(a, b, smooth=-1)

    def test_wide_extrema_load(self):
        # 70,000 operands in groups of 1,024 keep few values alive at a time.
        beads = minimum([length(x - 0.001 * i, y, z) - 1 for i in range(70_000)])
        report = verify_field(beads, ((-2, -2, -2), (72, 2, 2)))
        self.assertTrue(report["ok"], report.get("message"))
        self.assertAlmostEqual(beads.evaluate([[0.0, 0, 0]])[0], -1.0)

    def test_limits(self):
        with self.assertRaisesRegex(SDFError, "at most 2\\*\\*52"):
            x ** (2.0 ** 54)
        self.assertEqual((x ** (2.0 ** 52))._node.opcode, Opcode.POWC)
        for period in (1e-310, 1e301):
            with self.assertRaisesRegex(SDFError, "between 1e-300 and 1e300"):
                triwave(x, period)
        with self.assertRaisesRegex(SDFError, "within 1e150"):
            implicit(x, ((-1e200, 0, 0), (1, 1, 1)))
        values = np.array([1e20, -1e20, 5.5, -0.5, 3e15])
        rest = np.fmod(values, 4.0)
        expected = np.abs(np.where(rest < 0, rest + 4.0, rest) - 2.0)
        np.testing.assert_array_equal(triwave(values, 2.0), expected)
        self.assertEqual([triwave(float(v), 2.0) for v in values], list(expected))
        np.testing.assert_array_equal(triwave(x, 2.0).evaluate(np.column_stack([values, 0 * values, 0 * values])),
                                      expected)
        self.assertEqual(verify_field(clamp(x, 3, 3) + length(x, y, z) - 5, ((-8, -8, -8), (8, 8, 8)))["bound"], 1.0)

    def test_twisted_edges_crease_as_single_lines(self):
        # A twisted bar's creases are its 12 edges meeting at 8 corners, with
        # no stray detours or spurs beside the helical edges.
        bar = box((10, 10, 60))
        rate = math.radians(90) / 60
        c, s = cos(rate * z), sin(rate * z)
        corner = math.hypot(5, 5) + 0.5
        warped = implicit(bar.at(c * x + s * y, -s * x + c * y, z), ((-corner, -corner, -30.5), (corner, corner, 30.5)))
        for shape in (warped, bar.twist(90 / 60)):
            for cell in (0.35, 0.25):
                segments = shape.mesh(cell_size=cell).creases.reshape(-1, 2, 3)
                ends, degree = {}, {}
                for index, segment in enumerate(segments):
                    for point in map(tuple, segment):
                        degree[point] = degree.get(point, 0) + 1
                        ends.setdefault(point, []).append(index)
                junctions = [point for point, count in degree.items() if count != 2]
                self.assertEqual(len(junctions), 8, (shape.opcode, cell))
                self.assertTrue(all(degree[point] == 3 for point in junctions), (shape.opcode, cell))

    def test_sealed_cavities_are_not_pieces(self):
        result, folder = build(self, "result = box(10) - sphere(3)")
        stats, = result["partStats"]
        self.assertEqual((stats["components"], stats["cavities"]), (1, 1))
        self.assertFalse([warning for warning in result["warnings"] if "pieces" in warning])
        report, _ = native(folder, result, "validate")
        self.assertTrue(report["ok"], report["warnings"])
        self.assertEqual((report["parts"][0]["components"], report["parts"][0]["cavities"]), (1, 1))
        split, folder = build(self, "result = union(box(4), sphere(2, center=(10, 0, 0)))")
        self.assertEqual(split["partStats"][0]["components"], 2)
        self.assertTrue([warning for warning in split["warnings"] if "2 disconnected pieces" in warning])

    def test_validation_compares_sealed_cavities(self):
        # A 0.02 mm lid over a pocket is open at the preview cell and seals a
        # cavity at the validation cell: a wall thinner than the preview.
        result, folder = build(self, "result = box(20) - box((10, 10, 10 - 0.02), center=(0, 0, 5 - 0.01))")
        self.assertEqual((result["partStats"][0]["components"], result["partStats"][0]["cavities"]), (1, 0))
        report, _ = native(folder, result, "validate")
        self.assertFalse(report["ok"])
        self.assertEqual((report["parts"][0]["cavities"], report["parts"][0]["previewCavities"]), (1, 0))
        self.assertTrue([warning for warning in report["warnings"] if "1 sealed cavities" in warning], report)

    def test_points_outside_implicit_bounds_snap_onto_the_surface(self):
        # Outside its bounds an implicit() solid's field keeps a gradient away
        # from the solid, so measured points beyond a face snap to the surface.
        script = "result = implicit(x * x + y * y + z * z - 100.0, bounds=((-11, -11, -11), (11, 11, 11)))"
        result, folder = build(self, script)
        for point in ([11.5, 0, 0], [0, 11.6, 0], [11.4, 3, 0]):
            report, _ = native(folder, result, "measure", entities=[{"kind": "point", "solid": 0, "point": point}])
            snapped, = report["points"]
            self.assertAlmostEqual(float(np.linalg.norm(snapped["point"])), 10.0, delta=1e-4)
            np.testing.assert_allclose(snapped["normal"], np.array(snapped["point"]) / 10, atol=1e-4)

    def test_specks_below_a_cell_are_pieces_not_cavities(self):
        # A speck far below the cell (0.078 mm) around a lattice point at
        # (phi + k) h meshes as a tiny closed piece, not a sealed void.
        result, _ = build(self, "result = union(box((20, 12, 12)), sphere(0.018, center=(0.031, 6.515, 0.029)))")
        stats, = result["partStats"]
        self.assertEqual((stats["components"], stats["cavities"]), (2, 0))
        self.assertTrue([warning for warning in result["warnings"] if "2 disconnected pieces" in warning])

    def test_voids_about_a_cell_across_are_cavities(self):
        # A void about a preview cell (0.78 mm) across folds into a sliver
        # enclosing as little volume as a speck's; material around it makes
        # it a sealed cavity, so the block stays one piece and validates.
        for radius in (0.4, 0.45):
            result, folder = build(self, f"result = box(200) - sphere({radius}, center=(10.31, -20.77, 5.13))")
            stats, = result["partStats"]
            self.assertEqual((stats["components"], stats["cavities"]), (1, 1), radius)
            self.assertFalse([warning for warning in result["warnings"] if "pieces" in warning], radius)
            report, _ = native(folder, result, "validate")
            self.assertTrue(report["ok"], report)

    def test_many_voids_about_a_cell_across_are_all_cavities(self):
        # Each small component is classified from 26 field samples; with many
        # of them the samples span several evaluation batches.
        rng = np.random.default_rng(7)
        centres = []
        while len(centres) < 100:
            c = rng.uniform(-9, 9, 3)
            if all(np.linalg.norm(c - d) > 2 for d in centres):
                centres.append(c)
        block = implicit(maximum(maximum(abs(x), abs(y)), abs(z)) - 10 + 0.0 * sin(x) * cos(y) * sin(z),
                         ((-10.5, -10.5, -10.5), (10.5, 10.5, 10.5)))
        for c in centres:
            block = block - sphere(0.225, center=tuple(float(v) for v in c))
        for threads in (1, 8):
            stats = block.mesh(cell_size=0.5, threads=threads).stats
            self.assertGreater(stats["cavities"], 20, threads)
            self.assertEqual(stats["components"] - stats["cavities"], 1, (threads, stats["components"], stats["cavities"]))

    def test_chamfered_corners_keep_their_short_edges(self):
        # Three 0.433 mm edges meet at each corner of a box with 0.5 mm edge
        # chamfers; their crease fragments hang from junctions where the other
        # lines turn, so they are edges, not spurs beside a line.
        planes = []
        for sx in (-1, 1):
            for sy in (-1, 1):
                planes += [radsdf.half_space((sx, sy, 0), (sx * 9.75, sy * 9.75, 0)),
                           radsdf.half_space((sx, 0, sy), (sx * 9.75, 0, sy * 9.75)),
                           radsdf.half_space((0, sx, sy), (0, sx * 9.75, sy * 9.75))]
        chamfered = intersection(box(20), *planes)

        def distance(p, a, b):
            t = np.clip((p - a) @ (b - a) / ((b - a) @ (b - a)), 0, 1)
            return np.linalg.norm(a + t * (b - a) - p)

        for cell in (0.1, 0.25):
            segments = chamfered.mesh(cell_size=cell).creases.reshape(-1, 2, 3).astype(np.float64)
            middles = segments.mean(axis=1)
            for corner in np.array(np.meshgrid([-1, 1], [-1, 1], [-1, 1])).T.reshape(-1, 3):
                centre = 9.75 * corner
                edges = [(centre, np.where(np.arange(3) == axis, 10.0, 9.5) * corner) for axis in range(3)]
                near = np.linalg.norm(middles - centre, axis=1) < 0.8
                along = [index for index in np.flatnonzero(near)
                         if min(distance(middles[index], a, b) for a, b in edges) < 0.5 * cell]
                length = float(np.linalg.norm(segments[along, 1] - segments[along, 0], axis=1).sum())
                self.assertGreater(length, 0.55, (cell, corner.tolist()))  # of 3 x 0.433 mm


def spec(script, **extra):
    return {"title": "Fields", "summary": "Custom field test.", "units": "mm", "script": script, **extra}


if __name__ == "__main__":
    unittest.main()
