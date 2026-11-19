"""GPU field programs (radsdf_gpu.h, Engine/sdf_field.py; SDF-DESIGN.md, "GPU
field programs"): the compiler, brick pruning, the float64 reference
interpreter and the build's field.bin. Needs the native kernel."""
import json
import math
import os
from pathlib import Path
import shutil
import struct
import subprocess
import sys
import tempfile
import unittest

import numpy as np

ENGINE = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ENGINE))
sys.path.insert(0, str(ENGINE / "tests"))
import radsdf
from radsdf import *  # noqa: F401,F403 (the program API)
from radsdf import SDFError
import sdf_build
import sdf_field


def kernel_available():
    try:
        radsdf.kernel_version()
        return True
    except SDFError:
        return False


PART_WORDS = 24
FAR = 0x80000000      # cell flags (one word per cell)
REFINED = 0x40000000


def part_record(words, part):
    at = int(words[4]) + PART_WORDS * part
    record = words[at:at + PART_WORDS]
    floats = record.view(np.float32).astype(np.float64)
    return {"status": int(record[0]), "grid": int(record[1]), "dims": [int(value) for value in record[2:5]],
            "origin": floats[6:9], "brick": floats[9], "margin": floats[11], "cells": int(record[23])}


def cells_of(words, part, points):
    """(kind, tape) per point as the shader looks it up: kind -1 outside the
    grid, 0 far, 1 a tape (the word offset of its first instruction)."""
    record = part_record(words, part)
    dims = np.array(record["dims"])
    kinds, found = np.full(len(points), -1), np.zeros(len(points), np.int64)
    for index, point in enumerate(points):
        cell = np.floor((point - record["origin"]) / record["brick"]).astype(np.int64)
        if not np.all((cell >= 0) & (cell < dims)):
            continue
        corner, size = record["origin"] + record["brick"] * cell, record["brick"]
        word = int(words[record["grid"] + int(cell[0] + dims[0] * (cell[1] + dims[1] * cell[2]))])
        while word >> 30 == 1:
            size *= 0.5
            octant = 0
            for axis in range(3):
                if point[axis] >= corner[axis] + size:
                    octant |= 1 << axis
                    corner[axis] += size
            word = int(words[((word & 0x3FFFFFFF) << 3) + octant])
        kinds[index] = 0 if word & FAR else 1
        found[index] = 0 if word & FAR else (word << 2) + 4
    return kinds, found


def leaf_cells(words, part):
    """(cell word, level) of every cell that is not refined."""
    record = part_record(words, part)
    top = int(np.prod(record["dims"]))
    stack, leaves = [(int(words[record["grid"] + cell]), 0) for cell in range(top)], []
    while stack:
        word, level = stack.pop()
        if word >> 30 == 1:
            stack += [(int(words[((word & 0x3FFFFFFF) << 3) + octant]), level + 1) for octant in range(8)]
        else:
            leaves.append((word, level))
    return leaves


def tape_length(words, word):
    """Instructions of the tape a (tape) cell word names."""
    offset = (word << 2) + 4  # after the header record
    count = 0
    while int(words[offset + 4 * count]) & 0x7F:
        count += 1
    return count


def every_opcode():
    """Parts that together use every RSDF opcode."""
    k = 2 * math.pi / 6.0
    star = [((10 if i % 2 else 6) * math.cos(i * math.pi / 60), (10 if i % 2 else 6) * math.sin(i * math.pi / 60))
            for i in range(120)]
    a = union(sphere(5), box((6, 4, 3), center=(4, 0, 0), radius=0.5),
              cylinder(2, 8, center=(-3, 0, 0), axis=(1, 1, 0)), smooth=1.5)
    b = difference(a, cone(2, 0.5, 5, center=(0, 0, 2)), capsule((-5, -5, 0), (5, 5, 0), 1), smooth=0.5)
    booleans = union(b, torus(6, 1.2, center=(0, 0, -3)).scale((1.0, 1.3, 0.8)),
                     ellipsoid((3, 2, 1.5), center=(8, 4, 2)).rotate(30, axis=(1, 2, 3)))
    modifiers = union(box((4, 4, 12), radius=0.3).twist(8).translate(0, 0, 1),
                      sphere(2, center=(3, 3, 0)).mirrored_copy((1, 0, 0)).elongate(1, 0.5, 0).translate(0, 0, 8),
                      box((10, 10, 2)).shell(0.6, side="outside").offset(0.2).translate(0, 0, -8),
                      intersection(sphere(9), half_space((0, 0, 1), (0, 0, 2)), box(14), smooth=0.4).translate(14, 0, 0),
                      sphere(3).shell(0.5, side="center").translate(-12, 0, 0),
                      sphere(3).shell(0.5, side="inside").translate(-12, 8, 0))
    prof = union(polygon(star), rect((8, 3), center=(9, 0), radius=0.5), smooth=0.5)
    prof2 = difference(circle(5), slot((-2, 0), (2, 0), 1), ellipse((1.5, 3), center=(0, 2)).rotate(20))
    prof3 = intersection(rect((10, 10)), half_plane((1, 1), (2, 0)), circle(6).scale((1.0, 0.5)))
    profiles = union(prof.extrude(4), prof2.outline(0.4).extrude(6, taper=-1).translate(0, 0, 6),
                     prof3.mirrored_copy((1, 0)).offset(0.3).revolve(14).translate(0, 0, -10),
                     circle(3).loft_to(rect((5, 2)), 8).translate(-18, 0, 0))
    g = sin(k * x) * cos(k * y) + sin(k * y) * cos(k * z) + sin(k * z) * cos(k * x)
    lattice = implicit(abs(g) - 0.5, bounds=((-8, -8, -8), (8, 8, 8)))
    warped = implicit(sphere(6).at(x + 0.5 * sin(y), y, z * 1.2) + 0.1 * triwave(x, 1.5),
                      bounds=((-9, -9, -9), (9, 9, 9)))
    misc = implicit(length(x, y, z) - 4 + 0.2 * exp(-(x ** 2)) + 0.1 * log(2 + y * y) + 0.05 * sqrt(1 + z * z)
                    + 0.1 * atan2(y, x + 20) - 0.1 * clamp(x / 3, -1, 1) + minimum(x, 1) * 0.01
                    + maximum(y, -1) * 0.01 + (x * x) ** 1.5 * 0.0001 + minimum(x, y, smooth=0.5) * 0.01
                    + maximum(x, y, smooth=0.5) * 0.01 + box(6).field * 0.01, bounds=((-7, -7, -7), (7, 7, 7)))
    fields = union(intersection(box(14), lattice), warped.translate(20, 0, 0), misc.translate(0, 20, 0))
    v = np.array([[6, 0, 0], [-6, 0, 0], [0, 6, 0], [0, -6, 0], [0, 0, 6], [0, 0, -6]], float)
    t = np.array([[0, 2, 4], [2, 1, 4], [1, 3, 4], [3, 0, 4], [2, 0, 5], [1, 2, 5], [3, 1, 5], [0, 3, 5]])
    bv = np.array([[x_, y_, z_] for x_ in (2, 9) for y_ in (-2, 2) for z_ in (-2, 2)], float)
    bt = np.array([[0, 1, 3], [0, 3, 2], [4, 6, 7], [4, 7, 5], [0, 4, 5], [0, 5, 1], [2, 3, 7], [2, 7, 6],
                   [0, 2, 6], [0, 6, 4], [1, 5, 7], [1, 7, 3]])
    meshed = difference(mesh_solid(np.vstack([v, bv]), np.vstack([t, bt + 6])), sphere(2, center=(6, 0, 0)),
                        smooth=0.3)
    return [booleans, modifiers, profiles, fields, meshed]


@unittest.skipUnless(kernel_available(), "the native SDF kernel is not available")
class FieldProgramTests(unittest.TestCase):
    CELL = 0.3

    @classmethod
    def setUpClass(cls):
        cls.shapes = every_opcode()
        cls.graph = radsdf.Graph(cls.shapes)
        cls.program = sdf_field.FieldProgram(cls.graph, cls.graph.roots, 8 * cls.CELL, 0.1 * cls.CELL,
                                             min_brick_size=0.5 * cls.CELL)

    @classmethod
    def tearDownClass(cls):
        cls.program.close()
        cls.graph.close()

    def sample(self, part, count, rng):
        """Points near the part's surface and across its bounds."""
        root = self.graph.roots[part]
        mesh = self.graph.mesh(root, cell_size=self.CELL)
        surface = mesh.positions.astype(np.float64)
        near = surface[rng.integers(0, len(surface), count)] + rng.normal(0, self.CELL, (count, 3))
        bounds = self.graph.bounds(root)
        spread = bounds.lo - 1 + (bounds.hi - bounds.lo + 2) * rng.random((count // 2, 3))
        return np.vstack([near, spread]), surface

    def test_every_opcode_compiles(self):
        self.assertEqual([part["status"] for part in self.program.parts], ["field"] * 5)
        words = self.program.words
        self.assertEqual(struct.unpack("<4sII", words[:3].tobytes()), (b"RFPG", 2, 5))
        self.assertEqual(int(words[3]), len(words))
        self.assertLessEqual(max(part["valueRegisters"] for part in self.program.parts), 16)
        self.assertLessEqual(max(part["pointRegisters"] for part in self.program.parts), 8)

    def test_unpruned_tapes_match_the_kernel_bitwise(self):
        rng = np.random.default_rng(1)
        for part, root in enumerate(self.graph.roots):
            points, _ = self.sample(part, 3000, rng)
            with self.subTest(part=part):
                np.testing.assert_array_equal(self.program.evaluate(part, points, mode=1),
                                              self.graph.evaluate(root, points))

    def test_pruned_tapes_are_exact_in_their_bricks_and_far_values_conservative(self):
        rng = np.random.default_rng(2)
        for part, root in enumerate(self.graph.roots):
            points, surface = self.sample(part, 6000, rng)
            kernel = self.graph.evaluate(root, points)
            values = self.program.evaluate(part, points, mode=0)
            kinds, _ = cells_of(self.program.words, part, points)
            band = kinds == 1
            with self.subTest(part=part):
                self.assertGreater(band.sum(), 1000)
                # Bitwise the full field wherever a tape runs (brick or octant).
                np.testing.assert_array_equal(values[band], kernel[band])
                # Elsewhere a signed lower bound of the distance to the surface.
                far = np.flatnonzero(~band)
                self.assertTrue(np.all(np.sign(values[far]) == np.sign(kernel[far])))
                for index in far[:400]:
                    nearest = np.sqrt(np.min(np.sum((surface - points[index]) ** 2, axis=1)))
                    self.assertLessEqual(abs(values[index]), nearest + self.CELL)

    def test_refined_bricks_stay_exact_and_short(self):
        # Tapes longer than 4 x 4 instructions split their bricks down to the
        # minimum size (8 cells -> 0.5 cell: four levels).
        words = self.program.words
        for part, info in enumerate(self.program.parts):
            record = part_record(words, part)
            top = int(np.prod(record["dims"]))
            self.assertEqual(info["cells"], record["cells"])
            self.assertLessEqual(info["levels"], 4)
            leaves = [(tape_length(words, word), level) for word, level in leaf_cells(words, part) if not word & FAR]
            with self.subTest(part=part):
                self.assertEqual(max(length for length, _ in leaves), info["maxInstructions"])
                for length, level in leaves:
                    if length > 4 * 4:
                        self.assertEqual(level, 4)
                if info["levels"]:
                    self.assertGreater(record["cells"], top)
        self.assertLess(np.median([tape_length(words, word) for word, _ in leaf_cells(words, 0) if not word & FAR]), 8)

    def test_deterministic_for_any_thread_count(self):
        programs = [sdf_field.FieldProgram(self.graph, self.graph.roots, 8 * self.CELL, 0.1 * self.CELL, threads=t,
                                           min_brick_size=0.5 * self.CELL) for t in (1, 3, 0)]
        try:
            for program in programs[1:]:
                np.testing.assert_array_equal(program.words, programs[0].words)
        finally:
            for program in programs:
                program.close()

    def test_limits_leave_parts_to_meshes(self):
        program = sdf_field.FieldProgram(self.graph, self.graph.roots, 8 * self.CELL, 0.1 * self.CELL,
                                         max_instructions=6, min_brick_size=8 * self.CELL)
        try:
            statuses = [part["status"] for part in program.parts]
            self.assertIn("meshOnly", statuses)
            self.assertIn("field", statuses)
            for part, info in enumerate(program.parts):
                if info["status"] == "meshOnly":
                    self.assertIn("exceeds 6 instructions", info["reason"])
                    self.assertEqual(part_record(program.words, part)["status"], 1)
                    with self.assertRaisesRegex(SDFError, "mesh only"):
                        program.evaluate(part, np.zeros((1, 3)))
            tiny = sdf_field.FieldProgram(self.graph, self.graph.roots, 8 * self.CELL, 0.1 * self.CELL,
                                          max_words=20000)
            try:
                self.assertTrue(all(part["status"] == "meshOnly" or part["words"] < 20000 for part in tiny.parts))
                self.assertIn("size limit", " ".join(part.get("reason", "") for part in tiny.parts))
            finally:
                tiny.close()
        finally:
            program.close()

    def test_invalid_arguments(self):
        for options in ({"brick_size": 0, "margin": 0}, {"brick_size": 1, "margin": -1},
                        {"brick_size": math.inf, "margin": 0}):
            with self.subTest(options=options), self.assertRaises(SDFError):
                sdf_field.FieldProgram(self.graph, self.graph.roots, options["brick_size"], options["margin"])
        with self.assertRaises(SDFError):
            sdf_field.FieldProgram(self.graph, [], 1, 0)
        with radsdf.Graph([circle(2)]) as flat, self.assertRaisesRegex(SDFError, "not a 3D shape"):
            sdf_field.FieldProgram(flat, flat.roots, 1, 0)

    def test_folds_join_the_instructions_computing_their_operands(self):
        # A smooth union's folds and a difference become post operations of
        # the leaves computing their operands (the scale after the difference
        # stays: one post operation per instruction): 6 records, not 9, with
        # the kernel's values bitwise.
        shape = difference(union(sphere(2), sphere(2, center=(2, 0, 0)), sphere(2, center=(1, 2, 0)), smooth=1),
                           box(1, center=(1, 1, 1))).scale(1.5)
        with radsdf.Graph([shape]) as graph:
            program = sdf_field.FieldProgram(graph, graph.roots, 64.0, 0.05)
            try:
                words = program.words
                word = max((word for word, _ in leaf_cells(words, 0) if not word & FAR),
                           key=lambda word: tape_length(words, word))
                start = (word << 2) + 4
                records = words[start:start + 4 * tape_length(words, word)].reshape(-1, 4)
                self.assertEqual([int(x) & 0x7F for x in records[:, 0]], [1, 20, 20, 20, 21, 40])
                self.assertEqual([int(x) >> 12 & 0xF for x in records[:, 0]], [0, 0, 3, 3, 5, 0])  # SMIN, SMIN, DIFF
                points = np.random.default_rng(3).uniform(-4, 6, (5000, 3))
                kinds, _ = cells_of(words, 0, points)
                np.testing.assert_array_equal(program.evaluate(0, points)[kinds == 1],
                                              graph.evaluate(graph.roots[0], points)[kinds == 1])
                np.testing.assert_array_equal(program.evaluate(0, points, mode=1), graph.evaluate(graph.roots[0], points))
            finally:
                program.close()

    def test_pruning_drops_distant_operands(self):
        # 64 spheres in a row: each brick keeps the few nearby ones.
        row = union(*[sphere(1, center=(3 * i, 0, 0)) for i in range(64)])
        with radsdf.Graph([row]) as graph:
            program = sdf_field.FieldProgram(graph, graph.roots, 2.0, 0.05)
            try:
                lengths = [tape_length(program.words, word) for word, _ in leaf_cells(program.words, 0)
                           if not word & FAR]
                self.assertLessEqual(max(lengths), 7)
                points = np.column_stack([np.linspace(-2, 192, 4000), np.full(4000, 0.7), np.zeros(4000)])
                kinds, _ = cells_of(program.words, 0, points)
                np.testing.assert_array_equal(program.evaluate(0, points)[kinds == 1],
                                              graph.evaluate(graph.roots[0], points)[kinds == 1])
            finally:
                program.close()


@unittest.skipUnless(kernel_available(), "the native SDF kernel is not available")
class FieldProgramBuildTests(unittest.TestCase):
    def test_build_publishes_field_bin(self):
        script = ("result = [box((10, 10, 4), radius=1) - cylinder(2, 10), "
                  "union(sphere(3, center=(20, 0, 0)), sphere(2, center=(23, 0, 0)), smooth=1)]")
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory)
            result = sdf_build.build({"title": "Test", "summary": "A test design.", "units": "mm", "script": script},
                                     output, output, 2)
            on_disk = json.loads((output / "result.json").read_text())
            record = on_disk["fieldProgram"]
            self.assertEqual(record, result["fieldProgram"])
            self.assertEqual({key: record[key] for key in ("path", "version")}, {"path": "field.bin", "version": 2})
            cell = on_disk["mesh"]["cellSize"]
            self.assertAlmostEqual(record["brickSize"], sdf_field.BRICK_CELLS * cell)
            self.assertAlmostEqual(record["margin"], sdf_field.MARGIN_CELLS * cell)
            self.assertEqual([part["solid"] for part in record["parts"]], [0, 1])
            self.assertEqual([part["status"] for part in record["parts"]], ["field", "field"])
            words = np.fromfile(output / "field.bin", dtype="<u4")
            self.assertEqual(len(words), record["words"])
            self.assertEqual(int(words[0]), 0x47504652)
            self.assertFalse((output / "field.bin.tmp").exists())
            # The published program is the one a fresh compile of model.rsdf gives.
            with radsdf.Graph.from_bytes((output / "model.rsdf").read_bytes(), on_disk["partRoots"]) as graph:
                again = sdf_field.FieldProgram(graph, graph.roots, record["brickSize"], record["margin"],
                                               min_brick_size=sdf_field.MIN_BRICK_CELLS * cell)
                try:
                    np.testing.assert_array_equal(again.words, words)
                finally:
                    again.close()

    def test_compile_failures_do_not_fail_builds(self):
        self.assertEqual(sdf_field.publish(None, "kernel said no", Path(".")), {"error": "kernel said no"})


if __name__ == "__main__":
    unittest.main()


@unittest.skipUnless(os.environ.get("RAD_RUN_SANITIZER_FUZZ") == "1" and shutil.which("clang++") and kernel_available(),
                     "Set RAD_RUN_SANITIZER_FUZZ=1 (needs clang++) to compile GPU programs under sanitizers.")
class GPUProgramSanitizerTests(unittest.TestCase):
    """The field corpus and the every-opcode design compiled into GPU
    programs by a native driver built with AddressSanitizer and
    UndefinedBehaviorSanitizer, and with ThreadSanitizer."""

    def run_driver(self, flags, environment):
        from test_sdf_fields_native import field_corpus
        kernel = ENGINE.parent / "Sources" / "RADSDFKernel"
        with tempfile.TemporaryDirectory() as directory:
            folder = Path(directory)
            driver = folder / "gpu_driver"
            built = subprocess.run(["clang++", "-std=c++17", "-O1", "-g", "-fno-math-errno", "-fno-omit-frame-pointer",
                                    *flags, "-I", str(kernel / "include"), "-I", str(kernel),
                                    *map(str, sorted(kernel.glob("*.cpp"))),
                                    str(Path(__file__).with_name("gpu_program_sanitizer_driver.cpp")), "-o", str(driver)],
                                   capture_output=True, text=True)
            self.assertEqual(built.returncode, 0, built.stderr[-4000:])
            corpus = list(field_corpus(np.random.default_rng(5), mutations=40))
            corpus.append(radsdf._encode(every_opcode())[0])
            paths = []
            for index, data in enumerate(corpus):
                path = folder / f"graph-{index}.rsdf"
                path.write_bytes(data)
                paths.append(str(path))
            run = subprocess.run([str(driver), *paths], capture_output=True, text=True,
                                 env={**os.environ, **environment})
            report = run.stdout[-2000:] + run.stderr[-6000:]
            self.assertEqual(run.returncode, 0, report)
            for marker in ("ThreadSanitizer", "AddressSanitizer", "runtime error"):
                self.assertNotIn(marker, run.stderr, report)

    def test_programs_have_no_memory_or_arithmetic_faults(self):
        self.run_driver(["-fsanitize=address,undefined", "-fno-sanitize-recover=undefined"],
                        {"ASAN_OPTIONS": "detect_leaks=0:abort_on_error=0"})

    def test_programs_have_no_data_races(self):
        self.run_driver(["-fsanitize=thread"], {"TSAN_OPTIONS": "halt_on_error=1"})
