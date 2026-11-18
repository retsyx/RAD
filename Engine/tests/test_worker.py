"""Worker integration: requests, receipts, programs, parts, previews, the
Examples and the native measurement and export operations."""
import json
import math
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
from unittest import mock

import numpy as np

from sdf_test_support import (ROOT, assert_closed_and_outward, build, native, read_3mf, script_spec,
                              signed_volume)
import sdf_build
from sdf_build import ModelError, validate_spec, validation_hash
import worker


def stl_asset(folder, name, payload=b"solid reference\nendsolid reference\n"):
    (folder / name).write_bytes(payload)


class ValidationTests(unittest.TestCase):
    def test_validation_receipt_ignores_source_edits_and_display_metadata(self):
        spec = script_spec("result = box((1, 2, 3))")
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            original = validation_hash(spec, root)
            read = Path.read_bytes
            sources = {path.resolve() for path in (ROOT / "Engine").glob("*.py")}

            def changed(path):
                data = read(path)
                return data + b"\n# Unrelated edit\n" if Path(path).resolve() in sources else data

            # Only the rules tag (VALIDATION_RULES_VERSION, kernel, mesher
            # parameters) retires saved receipts.
            with mock.patch.object(Path, "read_bytes", changed):
                self.assertEqual(validation_hash(spec, root), original)
            self.assertEqual(validation_hash({**spec, "title": "Saved title", "summary": "Saved summary"}, root),
                             original)
            self.assertNotEqual(validation_hash({**spec, "script": "result = box((2, 2, 3))"}, root), original)

    def test_identical_programs_rebuild_identically_and_keep_their_receipt(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            request = root / "request.json"
            spec = script_spec("result = box((1, 2, 3)) - cylinder(0.3, 4)")
            request.write_text(json.dumps(spec))
            first = worker.run(request, root / "first", 2)
            self.assertRegex(first["validatedHash"], r"^[0-9a-f]{64}$")
            spec.update(title="Final saved title", summary="Final user-facing description")
            request.write_text(json.dumps(spec))
            replay = worker.run(request, root / "replay", 3)
            self.assertEqual(replay["validatedHash"], first["validatedHash"])
            for key in ("bounds", "volume", "area", "solids", "partStats", "mesh", "partRoots"):
                self.assertEqual(first[key], replay[key])
            self.assertEqual((root / "first/model.rsdf").read_bytes(), (root / "replay/model.rsdf").read_bytes())
            spec["script"] = "result = box((2, 2, 3))"
            request.write_text(json.dumps(spec))
            self.assertNotEqual(worker.run(request, root / "changed", 2)["validatedHash"], first["validatedHash"])

    def test_validation_hash_covers_input_bytes_runtime_and_validation_rules(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            spec = {**script_spec("result = box(1)"), "inputAssets": [
                {"id": "a", "name": "Reference", "format": "stl", "path": "input-000.stl"}]}
            stl_asset(root, "input-000.stl")
            first = validation_hash(spec, root)
            spec["inputAssets"][0]["name"] = "Renamed reference"
            self.assertEqual(validation_hash(spec, root), first, "Asset names are display metadata.")
            stl_asset(root, "input-000.stl", b"changed asset bytes")
            changed = validation_hash(spec, root)
            self.assertNotEqual(first, changed)
            runtime = {"python": "3.14.0", "numpy": "0", "kernel": "changed"}
            self.assertNotEqual(validation_hash(spec, root, runtime), changed)
            with mock.patch.object(sdf_build, "VALIDATION_RULES_VERSION", sdf_build.VALIDATION_RULES_VERSION + 1):
                self.assertNotEqual(validation_hash(spec, root), changed)
            with mock.patch.object(sdf_build, "PREVIEW_CELLS_ACROSS", sdf_build.PREVIEW_CELLS_ACROSS * 2):
                self.assertNotEqual(validation_hash(spec, root), changed, "Mesher parameters are build rules.")

    def test_validation_hash_covers_input_presence_identity_format_and_order(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            for name in ("input-000.stl", "input-001.stl", "input-000.obj"):
                stl_asset(root, name, b"identical reference bytes")
            spec = script_spec("result = box(1)")
            no_inputs = validation_hash(spec, root)
            spec["inputAssets"] = [
                {"id": "a", "name": "A", "format": "stl", "path": "input-000.stl"},
                {"id": "b", "name": "B", "format": "stl", "path": "input-001.stl"}]
            original = validation_hash(spec, root)
            self.assertNotEqual(no_inputs, original)
            spec["inputAssets"][0]["id"], spec["inputAssets"][1]["id"] = "b", "a"
            self.assertNotEqual(original, validation_hash(spec, root))
            spec["inputAssets"][0]["id"], spec["inputAssets"][1]["id"] = "a", "b"
            spec["inputAssets"][0].update(format="obj", path="input-000.obj")
            self.assertNotEqual(original, validation_hash(spec, root))
            spec["inputAssets"][0].update(format="stl", path="input-000.stl")
            (root / "input-001.stl").unlink()
            with self.assertRaisesRegex(ModelError, "missing"):
                validation_hash(spec, root)

    def test_script_validation_happens_before_execution(self):
        invalid = [script_spec(None), script_spec(17), script_spec("x" * 200_001),
                   script_spec("é" * 100_001), script_spec(""), script_spec(" \n")]
        missing = script_spec("result = box(1)")
        del missing["script"]
        invalid.append(missing)
        for value in invalid:
            with self.subTest(value=str(value)[:120]), self.assertRaises(ModelError):
                validate_spec(value)
        for source in ("result = box((1, 2, 3))", "#" + "é" * 99_999 + "x"):
            value = script_spec(source)
            self.assertIs(validate_spec(value), value)

    def test_rejects_missing_metadata_unsupported_units_and_extra_protocol_fields(self):
        valid = script_spec("result = box((1, 2, 3))")
        invalid = [None, [], "not a request"]
        for field in ("title", "summary", "units"):
            value = dict(valid)
            del value[field]
            invalid.append(value)
        for field, content in (("title", ""), ("title", "x" * 161), ("title", 17),
                               ("summary", ""), ("summary", "x" * 6001),
                               ("units", "in"), ("units", None)):
            invalid.append({**valid, field: content})
        # Construction requests expose one complete program, without a second
        # operation protocol, removed BREP inputs or arbitrary metadata.
        for field, content in (("nodes", []), ("result", "result"), ("code", "print('unrecognized field')"),
                               ("operation", "build"), ("sourceAsset", "source.step"),
                               ("sourceValidatedBREP", "source.validated.brep")):
            invalid.append({**valid, field: content})
        for asset in ({"id": "a", "name": "A", "format": "step", "path": "input-000.step"},
                      {"id": "a", "name": "A", "format": "stl", "path": "../input-000.stl"},
                      {"id": "a", "name": "A", "format": "stl", "path": "input-001.stl"},
                      {"id": "a", "name": "A", "format": "stl", "path": "input-000.stl", "validatedBREP": "x"}):
            invalid.append({**valid, "inputAssets": [asset]})
        invalid.append({**valid, "inputAssets": [{"id": "a", "name": "A", "format": "stl", "path": "input-000.stl"},
                                                 {"id": "a", "name": "B", "format": "obj", "path": "input-001.obj"}]})
        for value in invalid:
            with self.subTest(value=str(value)[:120]), self.assertRaises(ModelError):
                validate_spec(value)

    def test_script_errors_are_actionable_and_result_must_be_solid(self):
        cases = [("a = 1", "must assign"), ("result = 42", "must be a 3D shape"),
                 ("result = circle(4)", "2D profile"),
                 ("result = [box(1), 'lid']", r"result\[1\] is a str"),
                 ("if True\n  result = None", "SyntaxError"),
                 ("raise ValueError('bad profile')", "line 1: ValueError: bad profile"),
                 ("raise SystemExit(0)", "SystemExit"),
                 ("result = box(1).translate((2e6, 0, 0))", "size limit"),
                 ("result = half_space()", "unbounded"),
                 ("result = imported_assets['missing']", "not available")]
        for source, diagnostic in cases:
            with self.subTest(source=source), self.assertRaisesRegex(ModelError, diagnostic):
                build(self, source)

    def test_rejects_empty_intersection(self):
        with self.assertRaisesRegex(ModelError, "is empty"):
            build(self, "a = sphere(1)\nresult = a & a.translate((50, 0, 0))")

    def test_cli_errors_are_actionable(self):
        with tempfile.TemporaryDirectory() as directory:
            request = Path(directory) / "bad.json"
            request.write_text(json.dumps(script_spec("raise ValueError(\"invalid radius\")")))
            command = [sys.executable, str(ROOT / "Engine" / "worker.py"), "--request", str(request), "--output",
                       directory]
            result = subprocess.run(command, capture_output=True, text=True, timeout=120)
            self.assertEqual(result.returncode, 1)
            self.assertIn("Python design at line 1: ValueError: invalid radius", result.stderr)
            self.assertFalse((Path(directory) / "model.rsdf").exists())
            report = json.loads((Path(directory) / "error.json").read_text())
            self.assertEqual(report["failureKind"], "design")


class ErrorReportTests(unittest.TestCase):
    def test_error_report_bounds_multibyte_messages_after_json_encoding(self):
        with tempfile.TemporaryDirectory() as directory:
            folder = Path(directory)
            for character in ("é", "\0"):
                with self.subTest(character=repr(character)):
                    worker.write_error_report(ModelError(character * 40_000), folder)
                    payload = (folder / "error.json").read_bytes()
                    self.assertLessEqual(len(payload), 65_536)
                    report = json.loads(payload)
                    self.assertEqual(report["formatVersion"], 1)
                    self.assertEqual(report["stage"], "worker")
                    self.assertTrue(report["message"])
                    self.assertEqual(set(report["message"]), {character})
                    self.assertEqual(list(folder.glob(".error-*")), [])

    def test_error_report_states_whether_the_design_or_its_runtime_failed(self):
        cases = [(ModelError("Python design at line 2: PermissionError: denied"), "design"),
                 (ValueError("A shape value is invalid"), "design"),
                 (sdf_build.HostError("Thread count must be between 1 and 256."), "runtime"),
                 (PermissionError(1, "Operation not permitted"), "runtime"),
                 (ImportError("dlopen(libRADSDF.dylib): Library not loaded"), "runtime")]
        with tempfile.TemporaryDirectory() as directory:
            folder = Path(directory)
            for error, kind in cases:
                with self.subTest(error=repr(error)):
                    worker.write_error_report(error, folder)
                    self.assertEqual(json.loads((folder / "error.json").read_bytes())["failureKind"], kind)

    def test_meshing_failures_report_their_stage_and_cell_size(self):
        with tempfile.TemporaryDirectory() as directory:
            folder = Path(directory)
            request = folder / "request.json"
            # A wall far thinner than the preview cell leaves no surface.
            request.write_text(json.dumps(script_spec("result = box((100, 100, 0.001))")))
            with self.assertRaises(ModelError) as caught:
                worker.run(request, folder / "out", 2)
            worker.write_error_report(caught.exception, folder)
            report = json.loads((folder / "error.json").read_text())
            self.assertEqual(report["stage"], "preview_meshing")
            self.assertGreater(report["cellSize"], 0)
            self.assertFalse((folder / "out" / "result.json").exists())


class GeometryFileResourceLimitTests(unittest.TestCase):
    def test_worker_allows_512_mb_files_and_preserves_a_stricter_inherited_limit(self):
        # Running a program sets an irreversible hard cap. Run each case in a
        # child so the unittest process stays unaffected.
        code = """
import errno
import json
from pathlib import Path
import resource
import signal
import sys

sys.path.insert(0, sys.argv[1])
import sdf_build
requested = int(sys.argv[2])
if requested:
    resource.setrlimit(resource.RLIMIT_FSIZE, (requested, requested))
before = resource.getrlimit(resource.RLIMIT_FSIZE)[1]
sdf_build.run_program('result = box(1)')
expected = 512_000_000 if before == resource.RLIM_INFINITY else min(512_000_000, before)
actual = resource.getrlimit(resource.RLIMIT_FSIZE)
assert actual == (expected, expected), (actual, expected)
signal.signal(signal.SIGXFSZ, signal.SIG_IGN)
path = Path(sys.argv[3])
with path.open('wb') as stream:
    stream.truncate(expected)
assert path.stat().st_size == expected
blocked = False
try:
    with path.open('r+b') as stream:
        stream.truncate(expected + 1)
except OSError as error:
    blocked = error.errno == errno.EFBIG
assert blocked, 'The worker file cap did not prevent a larger file'
print(json.dumps({'limit': expected, 'blocked': blocked}))
"""
        with tempfile.TemporaryDirectory() as directory:
            for inherited in (0, 96_000_000):
                with self.subTest(inherited=inherited):
                    result = subprocess.run([sys.executable, "-c", code, str(ROOT / "Engine"), str(inherited),
                                             str(Path(directory) / "output.bin")],
                                            capture_output=True, text=True, timeout=60)
                    self.assertEqual(result.returncode, 0, result.stderr)
                    report = json.loads(result.stdout)
                    self.assertEqual(report["limit"], inherited or 512_000_000)
                    self.assertTrue(report["blocked"])


def assert_part_ownership(test, preview):
    """Unique ids; leaves own exactly one solid; every solid is owned once;
    display arrays name existing solids and every solid owns triangles."""
    seen, owners, pending = set(), [], list(preview["parts"])
    while pending:
        part = pending.pop()
        test.assertNotIn(part["id"], seen)
        seen.add(part["id"])
        if part["children"] is None:
            test.assertEqual(len(part["solidIndices"]), 1)
            owners.extend(part["solidIndices"])
        else:
            test.assertIsInstance(part["children"], list)
            test.assertEqual(part["solidIndices"], [])
            pending.extend(part["children"])
    test.assertEqual(sorted(owners), list(range(preview["solids"])))
    test.assertEqual(set(np.asarray(preview["triangleSolids"]).tolist()), set(range(preview["solids"])))
    test.assertEqual(len(preview["edgeSolids"]), len(preview["edges"]))


class GeometryTests(unittest.TestCase):
    def test_primitives_have_expected_volume(self):
        cases = [("box((10, 20, 30), center=(0, 0, 15))", 6000, 1e-6),
                 ("cylinder(5, 10)", math.pi * 250, 2e-3),
                 ("sphere(5)", 4 / 3 * math.pi * 125, 2e-3),
                 ("cone(5, 0.001, 10)", math.pi * 250 / 3, 1e-2),
                 ("torus(10, 2)", 2 * math.pi ** 2 * 10 * 4, 2e-3),
                 ("capsule((0, 0, 0), (0, 0, 10), 2)", math.pi * 4 * 10 + 4 / 3 * math.pi * 8, 2e-3)]
        for expression, volume, tolerance in cases:
            with self.subTest(expression=expression):
                result, _ = build(self, "result = " + expression)
                self.assertAlmostEqual(result["volume"] / volume, 1, delta=tolerance)
                assert_closed_and_outward(self, result["vertices"], result["triangles"])

    def test_profiles_patterns_booleans_and_finishing(self):
        def volume(script):
            return build(self, script)[0]

        extruded = volume("result = polygon([(0, 0), (10, 0), (10, 20), (0, 20)]).extrude(5)")
        self.assertAlmostEqual(extruded["volume"], 1000, delta=1e-3)
        revolved = volume("result = rect((5, 10), center=(2.5, 5)).revolve()")
        self.assertAlmostEqual(revolved["volume"] / (math.pi * 250), 1, delta=3e-3)
        self.assertAlmostEqual(revolved["bounds"][5] - revolved["bounds"][2], 10, delta=1e-3)
        patterned = volume("result = box(2, center=(0, 0, 1)).linear_array(3, (5, 0, 0))")
        self.assertEqual(patterned["partStats"][0]["components"], 3)
        self.assertAlmostEqual(patterned["volume"], 24, delta=1e-3)
        self.assertTrue(any("disconnected pieces" in warning for warning in patterned["warnings"]))
        intersection = volume("a = box(10, center=(0, 0, 5))\nresult = a & a.translate((5, 0, 0))")
        self.assertAlmostEqual(intersection["volume"], 500, delta=1e-3)
        cut = volume("result = box(10, center=(5, 5, 5)) - cylinder(2, 12, center=(5, 5, 5))")
        self.assertAlmostEqual(cut["volume"] / (1000 - math.pi * 40), 1, delta=1e-3)
        rounded = volume("result = box(10, radius=1)")
        self.assertGreater(rounded["volume"], 1000 - 12 * (1 - math.pi / 4) * 10)
        self.assertLess(rounded["volume"], 1000)
        # A 1 mm wall inside a 10 mm cube, with its top removed above z = 4.
        shelled = volume("result = box(10).shell(1) & half_space((0, 0, 1), (0, 0, 4))")
        self.assertAlmostEqual(shelled["volume"], 10 * 10 * 9 - 8 * 8 * 8, delta=0.05)
        rotated = volume("result = box((2, 3, 4), center=(1, 1.5, 2)).rotate_z(90).translate((10, 20, 30))")
        self.assertAlmostEqual(rotated["volume"], 24, delta=1e-3)
        for actual, expected in zip(rotated["bounds"], [7, 20, 30, 10, 22, 34]):
            self.assertAlmostEqual(actual, expected, delta=1e-4)
        # A frustum lofted between circles: pi h (r1^2 + r1 r2 + r2^2) / 3.
        lofted = volume("result = circle(3).loft_to(circle(2), 4)")
        self.assertAlmostEqual(lofted["volume"] / (math.pi * 4 * (9 + 6 + 4) / 3), 1, delta=3e-3)
        self.assertAlmostEqual(lofted["bounds"][5] - lofted["bounds"][2], 4, delta=1e-3)
        smooth = volume("result = union(sphere(3), sphere(3, center=(4, 0, 0)), smooth=1.5)")
        sharp = volume("result = union(sphere(3), sphere(3, center=(4, 0, 0)))")
        self.assertGreater(smooth["volume"], sharp["volume"], "A smooth blend fills the crease between shapes.")

    def test_preview_meshes_are_closed_outward_and_split_only_at_creases(self):
        result, _ = build(self, "result = [box((4, 6, 8)), sphere(3, center=(20, 0, 0))]")
        vertices, triangles = np.asarray(result["vertices"]), np.asarray(result["triangles"])
        solids = np.asarray(result["triangleSolids"])
        for solid in range(2):
            assert_closed_and_outward(self, vertices, triangles[solids == solid])
        self.assertTrue(np.allclose(np.linalg.norm(result["normals"], axis=1), 1, atol=1e-5))
        # Triangles are contiguous per part.
        self.assertTrue((np.diff(solids) >= 0).all())
        # Crease segments copy their part's vertex positions bit for bit.
        points = {tuple(point) for point in np.asarray(vertices, dtype=np.float32).tolist()}
        edges = np.asarray(result["edges"], dtype=np.float32).reshape(-1, 2, 3)
        self.assertGreater(len(edges), 0)
        self.assertTrue(all(tuple(point) in points for point in edges.reshape(-1, 3).tolist()))
        self.assertEqual(set(np.asarray(result["edgeSolids"]).tolist()), {0}, "A sphere has no creases.")
        box_normals = np.asarray(result["normals"])[np.unique(triangles[solids == 0])]
        axes = np.abs(box_normals).max(axis=1)
        self.assertTrue((axes > 0.999).all(), "Box faces shade with their own normals, split at the edges.")

    def test_multisolid_mesh_mapping(self):
        result, _ = build(self, "result = [box((4, 6, 8), center=(0, 0, 4)), box((4, 6, 8), center=(20, 0, 4))]")
        self.assertEqual(result["solids"], 2)
        triangles, mapping = np.asarray(result["triangles"]), np.asarray(result["triangleSolids"])
        vertices = np.asarray(result["vertices"])
        self.assertEqual(len(mapping), len(triangles))
        owned = [set(np.unique(triangles[mapping == solid]).tolist()) for solid in range(2)]
        self.assertFalse(owned[0] & owned[1])
        x_bounds = [(vertices[list(indices), 0].min(), vertices[list(indices), 0].max()) for indices in owned]
        self.assertEqual([tuple(round(float(value), 4) for value in bounds) for bounds in x_bounds],
                         [(-2, 2), (18, 22)])
        assert_part_ownership(self, result)
        self.assertEqual([part["name"] for part in result["parts"]], ["Part 1", "Part 2"])
        self.assertEqual([part["id"] for part in result["parts"]], ["solid/0", "solid/1"])
        edges = np.asarray(result["edges"]).reshape(-1, 6)
        for solid, expected in enumerate([(-2, 2), (18, 22)]):
            x = edges[np.asarray(result["edgeSolids"]) == solid][:, [0, 3]]
            self.assertEqual((round(float(x.min()), 4), round(float(x.max()), 4)), expected)
        for stats, center in zip(result["partStats"], (0, 20)):
            self.assertAlmostEqual(stats["volume"], 192, delta=1e-3)
            self.assertAlmostEqual(stats["centroid"][0], center, delta=1e-4)

    def test_named_nested_assembly_placements_compose_from_the_root(self):
        source = """
bolt = cylinder(1, 6, center=(0, 0, 3))
result = Assembly('Machine', id='machine')
left = result.folder('Left', id='left', at=Placement(translate=(10, 0, 0)))
left.add(bolt, id='bolt-left', name='Bolt', at=Placement(translate=(0, 5, 0)))
right = result.folder('Right', id='right', at=Placement(translate=(0, 0, 20), rotate=(90, 'x')))
right.add(bolt, id='bolt-right', name='Bolt')
"""
        result, _ = build(self, source)
        assert_part_ownership(self, result)
        left, right = result["partStats"]
        for actual, expected in zip(left["centroid"], (10, 5, 3)):
            self.assertAlmostEqual(actual, expected, delta=1e-3)
        # Rotating +Z about X by 90 degrees points the bolt along -Y.
        for actual, expected in zip(right["centroid"], (0, -3, 20)):
            self.assertAlmostEqual(actual, expected, delta=1e-3)
        self.assertAlmostEqual(right["bounds"][4] - right["bounds"][1], 6, delta=1e-3)

    def test_part_ids_survive_reparenting_root_rename_and_duplicate_instances(self):
        source = """
body = box((2, 3, 4))
result = Assembly('Old title')
left = result.folder('Left')
left.add(body.translate((10, 0, 0)), name='Bolt', id='bolt-left')
right = result.folder('Right')
right.add(body.translate((20, 0, 0)), name='Bolt', id='bolt-right')
right.add(body.translate((30, 0, 0)), name='Spacer')
"""

        def identities(parts):
            return {part["id"] for part in parts} | set().union(*(identities(part["children"] or [])
                                                                  for part in parts))

        first, _ = build(self, source)
        # Move a uniquely named part to another folder and rename the root.
        changed = source.replace("right.add(body.translate((30", "left.add(body.translate((30")
        second, _ = build(self, changed.replace("Old title", "New title"))
        assert_part_ownership(self, first)
        assert_part_ownership(self, second)
        self.assertEqual(identities(first["parts"]) - {"name:Old title"},
                         identities(second["parts"]) - {"name:New title"})
        self.assertTrue({"name:Spacer", "bolt-left", "bolt-right"} <= identities(first["parts"]))
        # Duplicate names without explicit ids stay distinct and deterministic.
        anonymous = source.replace(", id='bolt-left'", "").replace(", id='bolt-right'", "")
        third, _ = build(self, anonymous)
        fourth, _ = build(self, anonymous)
        assert_part_ownership(self, third)
        self.assertEqual(third["parts"], fourth["parts"])
        with self.assertRaisesRegex(ModelError, "unique"):
            build(self, source.replace("id='bolt-right'", "id='bolt-left'"))


class ExampleTests(unittest.TestCase):
    def test_examples_build_closed_parts_and_export_3mf(self):
        examples = sorted((ROOT / "Examples").glob("*.json"))
        self.assertTrue(examples)
        for example in examples:
            with self.subTest(example=example.name), tempfile.TemporaryDirectory() as directory:
                output = Path(directory) / "build"
                output.mkdir()
                result = worker.run(example, output, 4)
                self.assertEqual(result["threads"], 4)
                self.assertEqual(result["mesh"]["boundaryEdges"], 0)
                self.assertGreater(len(result["triangles"]), 10)
                self.assertEqual(set(np.asarray(result["triangleSolids"]).tolist()), set(range(result["solids"])))
                assert_part_ownership(self, result)
                triangles = np.asarray(result["triangles"])
                for solid in range(result["solids"]):
                    assert_closed_and_outward(self, result["vertices"],
                                              triangles[np.asarray(result["triangleSolids"]) == solid])
                corners = np.asarray(result["vertices"], dtype=np.float64)[triangles]
                self.assertAlmostEqual(signed_volume(corners) / result["volume"], 1, delta=1e-6)
                report, folder = native(output, result, "export")
                exported, _ = read_3mf(folder / "model.3mf")
                self.assertEqual(len(exported), report["triangles"])
                self.assertAlmostEqual(signed_volume(exported) / result["volume"], 1, delta=0.01)


class MeasurementTests(unittest.TestCase):
    def metric(self, report, identity):
        return next(item["values"] for item in report["metrics"] if item["id"] == identity)

    def test_curved_surface_minimum_distance_uses_the_fields_not_preview_triangles(self):
        result, output = build(self, "result = [sphere(2), box(1, center=(10.5, 0.5, 0.5)), "
                                     "cylinder(1.5, 4, center=(0, 7, 0))]")
        with mock.patch.object(sdf_build, "build", side_effect=AssertionError("No program replay")):
            report, folder = native(output, result, "measure",
                                    entities=[{"kind": "part", "solid": 0}, {"kind": "part", "solid": 1}])
        self.assertAlmostEqual(self.metric(report, "pair:distance")[0], 8, places=7)
        for actual, expected in zip(self.metric(report, "pair:delta"), [8, 0, 0]):
            self.assertAlmostEqual(actual, expected, places=5)
        # The box's nearest point is its corner; finite-difference gradients
        # place closest points within a few nanometres of such edges.
        for actual, expected in zip(report["closestPoints"][0] + report["closestPoints"][1], [2, 0, 0, 10, 0, 0]):
            self.assertAlmostEqual(actual, expected, places=5)
        self.assertEqual(report["closestPoints"][1], self.metric(report, "pair:second"))
        self.assertFalse((folder / "mesh.bin").exists())
        self.assertFalse((folder / "model.3mf").exists())
        # Sphere to cylinder: 7 - 1.5 - 2 between curved surfaces.
        report, _ = native(output, result, "measure",
                           entities=[{"kind": "part", "solid": 2}, {"kind": "part", "solid": 0}])
        self.assertAlmostEqual(self.metric(report, "pair:distance")[0], 3.5, places=7)

    def test_points_snap_to_their_part_and_measure_to_parts_at_poses(self):
        result, output = build(self, "result = [sphere(2), box((2, 2, 2), center=(10, 0, 0))]")
        report, _ = native(output, result, "measure", entities=[{"kind": "point", "solid": 0, "point": [0, 0, 2.3]}])
        self.assertEqual(report["metrics"], [])
        [point] = report["points"]
        self.assertEqual(point["solid"], 0)
        for actual, expected in zip(point["point"] + point["normal"], [0, 0, 2, 0, 0, 1]):
            self.assertAlmostEqual(actual, expected, places=7)
        moved = [{"solid": 1, "matrix": [1, 0, 0, -5, 0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1]}]
        report, _ = native(output, result, "measure", solidTransforms=moved,
                           entities=[{"kind": "point", "solid": 0, "point": [2, 0, 0]}, {"kind": "part", "solid": 1}])
        self.assertAlmostEqual(self.metric(report, "pair:distance")[0], 2, places=7)
        for actual, expected in zip(self.metric(report, "pair:second"), [4, 0, 0]):
            self.assertAlmostEqual(actual, expected, places=7)
        report, _ = native(output, result, "measure", entities=[
            {"kind": "point", "solid": 1, "point": [11, 0.5, 0]}, {"kind": "point", "solid": 0, "point": [0, 2, 0]}])
        self.assertAlmostEqual(self.metric(report, "pair:distance")[0], math.hypot(11, 1.5), places=7)
        self.assertEqual([item["solid"] for item in report["points"]], [1, 0])
        # Overlapping parts touch: distance zero at a common point.
        report, _ = native(output, result, "measure", solidTransforms=[
            {"solid": 1, "matrix": [1, 0, 0, -9, 0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1]}],
            entities=[{"kind": "part", "solid": 0}, {"kind": "part", "solid": 1}])
        self.assertEqual(self.metric(report, "pair:distance"), [0.0])
        self.assertEqual(report["closestPoints"][0], report["closestPoints"][1])

    def test_selected_3mf_exports_only_requested_parts_in_original_coordinates(self):
        script = ("result = [box((1, 2, 3), center=(-9.5, 4, 5.5)), box((2, 3, 4), center=(21, -3.5, 8)), "
                  "cylinder(2, 5, center=(40, 8, -0.5))]")
        result, output = build(self, script)
        for selected in ([1], [0, 2]):
            with self.subTest(selected=selected):
                report, folder = native(output, result, "export", solidIndices=selected)
                self.assertEqual(report["solids"], len(selected))
                corners, parts = read_3mf(folder / "model.3mf")
                self.assertEqual(len(corners), report["triangles"])
                volume = sum(result["partStats"][index]["volume"] for index in selected)
                # Simplification keeps flat faces and inscribes chords in the
                # cylinder's side (2 mm radius, 5 mm long), removing on average
                # two thirds of their depth, which is at most the tolerance.
                curved = 2 * math.pi * 2 * 5 if 2 in selected else 0.0
                self.assertAlmostEqual(signed_volume(corners) / volume, 1,
                                       delta=2e-3 + 2 / 3 * report["tolerance"] * curved / volume)
                lo = np.min([result["partStats"][index]["bounds"][:3] for index in selected], axis=0)
                hi = np.max([result["partStats"][index]["bounds"][3:] for index in selected], axis=0)
                self.assertTrue(np.allclose(corners.reshape(-1, 3).min(axis=0), lo, atol=0.05))
                self.assertTrue(np.allclose(corners.reshape(-1, 3).max(axis=0), hi, atol=0.05))
                self.assertEqual(len(parts), len(selected))
                for _, _, vertices, triangles in parts:
                    assert_closed_and_outward(self, vertices, triangles)
                # Half the default preview cell: the design spans 52 mm.
                self.assertEqual(report["cellSize"], 52 / 256 / 2)
        report, folder = native(output, result, "export", solidIndices=[0], cellSize=0.25)
        self.assertEqual(report["cellSize"], 0.25)

    def test_export_cell_size_respects_the_triangle_budget(self):
        result, output = build(self, "result = sphere(10)")
        with mock.patch("sdf_ops.EXPORT_TRIANGLE_BUDGET", 20_000):
            report, folder = native(output, result, "export", cellSize=0.05)
        self.assertLessEqual(report["triangles"], 20_000)
        self.assertGreater(report["cellSize"], 0.05)
        corners, _ = read_3mf(folder / "model.3mf")
        self.assertAlmostEqual(signed_volume(corners) / (4 / 3 * math.pi * 1000), 1, delta=0.01)

    def test_selection_operations_reject_invalid_indices_and_items(self):
        result, output = build(self, "result = box((1, 2, 3))")
        for selected in ([], [-1], [1], [True], [0, 0], ["0"], "0"):
            with self.subTest(selected=selected), self.assertRaises(ModelError):
                native(output, result, "export", solidIndices=selected)
        invalid = [[], [{"kind": "part", "solid": 0}] * 3, [{"kind": "face", "solid": 0}],
                   [{"kind": "part", "solid": 1}], [{"kind": "part", "solid": True}],
                   [{"kind": "point", "solid": 0}], [{"kind": "point", "solid": 0, "point": [0, 0]}],
                   [{"kind": "point", "solid": 0, "point": [0, 0, math.inf]}],
                   [{"kind": "part", "solid": 0, "point": [0, 0, 0]}],
                   [{"kind": "part", "solid": 0}, {"kind": "part", "solid": 0}],
                   [{"kind": "face", "solidIndex": 0, "localIndex": 0}]]
        for entities in invalid:
            with self.subTest(entities=entities), self.assertRaises(ModelError):
                native(output, result, "measure", entities=entities)
        for fields in ({"cellSize": 0}, {"cellSize": -1}, {"cellSize": "fine"}, {"solidColors": []},
                       {"individualParts": [[0], [0]]}, {"solidTransforms": [{"solid": 0, "matrix": [2] * 16}]}):
            with self.subTest(fields=fields), self.assertRaises(ModelError):
                native(output, result, "export", **fields)


if __name__ == "__main__":
    unittest.main()
