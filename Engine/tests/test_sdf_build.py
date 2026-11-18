"""The assembly API (radsdf_assembly) and the SDF build pipeline (sdf_build).

Assembly, motion, coupling and normalization tests need no kernel. Build
tests run the native kernel (RAD_SDF_LIBRARY, or the bindings' search path)
and are skipped when it cannot be loaded.
"""
import json
import math
import os
from pathlib import Path
import struct
import sys
import tempfile
import unittest
import uuid

import numpy as np

ENGINE = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ENGINE))
import radsdf
from radsdf import SDFError
from radsdf_assembly import Assembly, Motion, Part, Placement, cam, gear, geneva, rack, screw
from assembly_motion import coupled_position, validate_motion
import sdf_build
from sdf_build import ModelError


def kernel_available():
    try:
        radsdf.kernel_version()
        return True
    except SDFError:
        return False


def spec(script, **extra):
    return {"title": "Test", "summary": "A test design.", "units": "mm", "script": script, **extra}


def normalize(assembly):
    return sdf_build.normalize_result(assembly)


def leaves(parts):
    found = []
    for part in parts:
        if part["children"] is None:
            found.append(part)
        else:
            found.extend(leaves(part["children"]))
    return found


def physical(parts, solid):
    return next(part for part in leaves(parts) if part["solidIndices"] == [solid])


class AssemblyAPITests(unittest.TestCase):
    def test_colors(self):
        part = Assembly().add(radsdf.sphere(1), color="#F6C915")
        self.assertEqual(part.color, (246 / 255, 201 / 255, 21 / 255))
        self.assertEqual(Assembly().add(radsdf.sphere(1), color=(0.1, 0.2, 1)).color, (0.1, 0.2, 1.0))
        self.assertEqual(Assembly().add(radsdf.sphere(1), color=np.array([0, 0.5, 1])).color, (0.0, 0.5, 1.0))
        for bad in ("#FFF", "red", "#GG0000", (1.2, 0, 0), (True, 0, 0), (0, 0), (math.nan, 0, 0), 3):
            with self.subTest(color=bad), self.assertRaisesRegex(SDFError, "RRGGBB"):
                Assembly().add(radsdf.sphere(1), color=bad)

    def test_print_down_ids_and_names(self):
        part = Assembly().add(radsdf.sphere(1), print_down=(0, 0, -2))
        self.assertEqual(part.print_down, (0.0, 0.0, -1.0))
        self.assertEqual(Assembly().add(radsdf.sphere(1), print_down="z").print_down, (0.0, 0.0, 1.0))
        with self.assertRaisesRegex(SDFError, "nonzero"):
            Assembly().add(radsdf.sphere(1), print_down=(0, 0, 0))
        with self.assertRaisesRegex(SDFError, "finite"):
            Assembly().add(radsdf.sphere(1), print_down=(0, math.inf, 1))
        # The app refuses Unicode control and format characters (such as
        # zero-width joiners) in identities and names, so programs do too.
        for bad in ("", "   ", "a\nb", "a\u200db", "x" * 1025, 7):
            with self.subTest(id=bad), self.assertRaises(SDFError):
                Assembly().add(radsdf.sphere(1), id=bad)
        with self.assertRaisesRegex(SDFError, "format characters"):
            Assembly().add(radsdf.sphere(1), name="Wheel\u200b")
        self.assertEqual(Assembly().add(radsdf.sphere(1), id=" keep spaces ").id, " keep spaces ")
        self.assertEqual(Assembly().add(radsdf.sphere(1), name="  Wheel ").name, "Wheel")
        self.assertIsNone(Assembly().add(radsdf.sphere(1), name="   ").name)
        with self.assertRaisesRegex(SDFError, "at most 160"):
            Assembly().add(radsdf.sphere(1), name="n" * 161)

    def test_reference_flags(self):
        self.assertIsNone(Assembly().add(radsdf.sphere(1)).reference)
        self.assertIs(Assembly().add(radsdf.sphere(1), reference=True).reference, True)
        self.assertIs(Assembly().add(radsdf.sphere(1), reference=np.bool_(False)).reference, False)
        self.assertIs(Assembly().folder("Board", reference=True).reference, True)
        self.assertIs(Assembly("Kit", reference=False).reference, False)
        for bad in (1, 0, "yes", "True", (True,)):
            with self.subTest(reference=bad), self.assertRaisesRegex(SDFError, "reference must be True or False"):
                Assembly().add(radsdf.sphere(1), reference=bad)
        with self.assertRaisesRegex(SDFError, "add_assembly"):
            Assembly().add(Assembly("Board"), reference=True)

    def test_placements(self):
        placement = Placement(translate=(1, 2, 3), rotate=(90, "z"))
        expected = [[0, -1, 0, 1], [1, 0, 0, 2], [0, 0, 1, 3], [0, 0, 0, 1]]
        np.testing.assert_allclose(placement.matrix, expected, atol=1e-15)
        composed = Placement(translate=(10, 0, 0)) @ placement
        np.testing.assert_allclose(composed.matrix[:3, 3], (11, 2, 3))
        np.testing.assert_allclose(Placement.from_matrix(expected).matrix, expected, atol=1e-15)
        for bad in (np.diag([2, 2, 2, 1]), np.diag([1, 1, -1, 1]), [[1, 0.5, 0, 0], [0, 1, 0, 0], [0, 0, 1, 0],
                                                                    [0, 0, 0, 1]], np.eye(3), "x"):
            with self.subTest(matrix=bad), self.assertRaises(SDFError):
                Placement.from_matrix(bad)
        with self.assertRaisesRegex(SDFError, "angle in degrees, axis"):
            Placement(rotate=90)
        part = Assembly().add(radsdf.sphere(1), at=expected)
        np.testing.assert_allclose(part.at.matrix, expected, atol=1e-15)

    def test_structure_rules(self):
        root = Assembly("Root")
        with self.assertRaisesRegex(SDFError, "extrude or revolve"):
            root.add(radsdf.circle(1))
        with self.assertRaisesRegex(SDFError, "union"):
            root.add([radsdf.sphere(1), radsdf.sphere(2)])
        part = root.add(radsdf.sphere(1))
        self.assertIsInstance(part, Part)
        with self.assertRaisesRegex(SDFError, "takes a shape"):
            root.add(part)
        child = Assembly("Child")
        self.assertIs(root.add(child), child)
        with self.assertRaisesRegex(SDFError, "already belongs"):
            Assembly().add_assembly(child)
        with self.assertRaisesRegex(SDFError, "contain itself"):
            child.add_assembly(root)
        with self.assertRaisesRegex(SDFError, "add_assembly"):
            root.add(Assembly(), name="x")
        with self.assertRaises(AttributeError):
            part.color = "#000000"
        with self.assertRaises(AttributeError):
            root.name = "Other"
        with self.assertRaisesRegex(AttributeError, "set assembly.motion"):
            root.metadata["rad_motion"] = {}
        grand = child.folder("Grand")
        last = grand.add(radsdf.sphere(3))
        first_after = root.add(radsdf.sphere(4))
        self.assertEqual(root.parts(), [part, last, first_after])
        self.assertEqual(root.children, (part, child, first_after))


class MotionBuilderTests(unittest.TestCase):
    def arm(self):
        root = Assembly("Hinged arm")
        base = root.add(radsdf.box((40, 30, 6)), id="base", name="Base")
        arm = root.add(radsdf.box((50, 12, 5)), name="Arm", at=Placement(translate=(20, 0, 10)))
        return root, base, arm

    def test_builder_produces_the_motion_schema(self):
        root, base, arm = self.arm()
        motion = root.motion
        self.assertIsInstance(motion, Motion)
        motion.ground(base)
        hinge = motion.revolute("hinge", name="Arm angle",
                                parent=motion.connector("base-pivot", base, origin=(0, 0, 10)),
                                child=motion.connector("arm-pivot", arm, origin=(0, 0, 10)), limits=(-120, 120))
        self.assertEqual(hinge, "hinge")
        normalized = normalize(root)
        self.assertEqual(normalized.motion, {
            "grounded": ["base"],
            "connectors": [{"id": "base-pivot", "part": "base", "origin": [0.0, 0.0, 10.0],
                            "xAxis": [1.0, 0.0, 0.0], "zAxis": [0.0, 0.0, 1.0]},
                           {"id": "arm-pivot", "part": "name:Arm", "origin": [0.0, 0.0, 10.0],
                            "xAxis": [1.0, 0.0, 0.0], "zAxis": [0.0, 0.0, 1.0]}],
            "joints": [{"id": "hinge", "name": "Arm angle", "type": "revolute", "parent": "base-pivot",
                        "child": "arm-pivot", "position": 0.0, "limits": [-120.0, 120.0]}],
            "couplings": []})

    def test_raw_dictionaries_and_part_frames(self):
        root, base, arm = self.arm()
        root.motion = {"grounded": ["base"], "connectors": [], "joints": []}
        self.assertEqual(normalize(root).motion, {"grounded": ["base"], "connectors": [], "joints": [],
                                                  "couplings": []})
        root, base, arm = self.arm()
        motion = root.motion
        motion.ground("base")
        motion.slider("slide", motion.connector("rail", base, (0, 0, 0), z_axis="x"),
                      motion.connector("carriage", arm, (0, 0, 0), z_axis="x", frame="part"), limits=(0, 5))
        connectors = normalize(root).motion["connectors"]
        self.assertEqual(connectors[1]["origin"], [20.0, 0.0, 10.0])
        np.testing.assert_allclose(connectors[0]["xAxis"], (0, 1, 0))
        np.testing.assert_allclose(connectors[1]["zAxis"], (1, 0, 0))
        with self.assertRaisesRegex(SDFError, "frame='world'"):
            motion.definition()

    def test_builder_errors(self):
        root, base, arm = self.arm()
        motion = root.motion
        motion.connector("a", base, (0, 0, 0))
        with self.assertRaisesRegex(SDFError, "already used"):
            motion.connector("a", arm, (0, 0, 0))
        with self.assertRaisesRegex(SDFError, "perpendicular"):
            motion.connector("b", arm, (0, 0, 0), x_axis=(1, 0, 0), z_axis=(1, 0, 1))
        with self.assertRaisesRegex(SDFError, "not defined"):
            motion.revolute("j", "a", "missing")
        with self.assertRaisesRegex(SDFError, "not a joint"):
            motion.couple("c", "j", "k", **gear(10, 20))
        with self.assertRaisesRegex(SDFError, "physical part"):
            motion.ground(Assembly())
        motion.connector("b", arm, (0, 0, 0))
        motion.connector("c", arm, (1, 0, 0))
        motion.revolute("j", "a", "b")
        motion.revolute("k", "a", "c")
        with self.assertRaisesRegex(SDFError, "linear"):
            motion.couple("c", "j", "k", type="spline")
        with self.assertRaisesRegex(SDFError, "ratio"):
            motion.couple("c", "j", "k", samples=[[0, 0], [1, 1]], ratio=2)
        # Validation of the finished definition reports design errors.
        with self.assertRaisesRegex(ModelError, "grounded physical part"):
            normalize(root)
        child = root.folder("Sub")
        child.motion.ground("base")
        with self.assertRaisesRegex(ModelError, "root assembly"):
            normalize(root)

    def test_definition_needs_explicit_ids(self):
        root, base, arm = self.arm()
        root.motion.ground(arm)
        with self.assertRaisesRegex(SDFError, "explicit id"):
            root.motion.definition()
        root, base, arm = self.arm()
        root.motion.ground(base)
        self.assertEqual(root.motion.definition()["grounded"], ["base"])

    def test_followers_take_derived_positions(self):
        root = Assembly()
        ring = root.add(radsdf.box(4), id="frame")
        a = root.add(radsdf.sphere(1), id="a")
        b = root.add(radsdf.sphere(1), id="b")
        m = root.motion
        m.ground(ring)
        driver = m.revolute("drive", m.connector("f1", ring, (0, 0, 0)), m.connector("a1", a, (0, 0, 0)),
                            position=30)
        follower = m.revolute("follow", m.connector("f2", ring, (5, 0, 0)), m.connector("b1", b, (5, 0, 0)))
        m.couple("mesh", driver, follower, **gear(20, 40))
        motion = normalize(root).motion
        self.assertEqual(motion["couplings"], [{"id": "mesh", "type": "linear", "driver": "drive",
                                                "follower": "follow", "ratio": -0.5, "offset": 0.0}])
        self.assertEqual(motion["joints"][1]["position"], -15.0)


def coupled_parts():
    """A frame with a driver and a follower revolute joint, as validate_motion input."""
    parts = [{"id": name, "name": name, "solidIndices": [index], "children": None}
             for index, name in enumerate(("frame", "crank", "star"))]
    connectors = [{"id": "f1", "part": "frame", "origin": [0, 0, 0]},
                  {"id": "c1", "part": "crank", "origin": [0, 0, 0]},
                  {"id": "f2", "part": "frame", "origin": [50, 0, 0]},
                  {"id": "s1", "part": "star", "origin": [50, 0, 0]}]
    joints = [{"id": "crank", "type": "revolute", "parent": "f1", "child": "c1"},
              {"id": "star", "type": "revolute", "parent": "f2", "child": "s1"}]
    return parts, {"grounded": ["frame"], "connectors": connectors, "joints": joints}


class CouplingHelperTests(unittest.TestCase):
    def test_linear_helpers(self):
        self.assertEqual(gear(20, 40), {"type": "linear", "ratio": -0.5})
        self.assertEqual(gear(12, 36, internal=True), {"type": "linear", "ratio": 12 / 36})
        self.assertAlmostEqual(rack(10)["ratio"], 2 * math.pi * 10 / 360)
        self.assertLess(rack(-10)["ratio"], 0)
        self.assertEqual(screw(1.5), {"type": "linear", "ratio": 1.5 / 360})
        for call in (lambda: gear(0, 10), lambda: gear(10, -1), lambda: rack(0), lambda: screw(math.nan)):
            with self.assertRaises(SDFError):
                call()

    def test_cam_tables(self):
        table = cam(lambda x: 2 * x, 0, 10, step=3)
        self.assertEqual([x for x, _ in table["samples"]], [0, 2.5, 5, 7.5, 10])
        self.assertEqual(table["samples"][-1], [10.0, 20.0])
        self.assertNotIn("period", table)
        lift = cam(lambda x: 5 * (1 - math.cos(math.radians(x))), 0, 360, period=360)
        self.assertEqual((lift["period"], lift["advance"]), (360.0, 0.0))
        self.assertEqual(lift["samples"][-1][1], lift["samples"][0][1])
        self.assertEqual(len(lift["samples"]), 721)
        indexing = cam(lambda x: x / 4, 0, 360, period=360)
        self.assertEqual(indexing["advance"], 90.0)
        partial = cam(lambda x: x, 0, 300, period=360)
        self.assertEqual(partial["advance"], 0.0)
        with self.assertRaisesRegex(SDFError, "do not close"):
            cam(lambda x: x, 0, 360, period=360, advance=10)
        with self.assertRaisesRegex(SDFError, "step of at least"):
            cam(lambda x: x, 0, 10000, step=1)
        with self.assertRaisesRegex(SDFError, "failed at"):
            cam(lambda x: 1 / (x - 5), 0, 10, step=5)
        with self.assertRaisesRegex(SDFError, "requires a period"):
            cam(lambda x: x, 0, 10, advance=1)
        parts, definition = coupled_parts()
        definition["couplings"] = [{"id": "lift", "driver": "crank", "follower": "star", **lift}]
        self.assertEqual(validate_motion(definition, parts)["joints"][1]["position"], 0.0)

    def test_geneva_tables_close_and_follow_the_mechanism(self):
        for slots in (3, 4, 5, 6, 8):
            with self.subTest(slots=slots):
                table = geneva(slots, crank_at_center=15)
                samples, pitch = table["samples"], 360 / slots
                self.assertEqual((table["period"], table["advance"]), (360.0, -pitch))
                closing = samples[0][1] + table["advance"]
                self.assertLessEqual(abs(samples[-1][1] - closing), 1e-6 * max(1, abs(closing)))
                self.assertAlmostEqual(samples[-1][0] - samples[0][0], 360.0, places=9)
                self.assertTrue(all(b[0] > a[0] for a, b in zip(samples, samples[1:])))
                self.assertTrue(all(b[0] - a[0] <= 0.5 + 1e-9 for a, b in zip(samples[1:-1], samples[2:-1])))
                # The exact engagement ends equal the analytic dwell values.
                s, w = math.sin(math.pi / slots), 90 - 180 / slots
                for psi, expected in ((-w, 0.0), (w, -pitch)):
                    phi = math.degrees(math.atan2(s * math.sin(math.radians(psi)),
                                                  1 - s * math.cos(math.radians(psi))))
                    self.assertAlmostEqual(-(phi + 180 / slots), expected, places=9)
                # Independent kinematics: pin at crank angle psi, star centre at distance C.
                center_distance = 50.0
                crank_radius = center_distance * s
                def slot_angle(psi):
                    pin = crank_radius * np.array([math.cos(math.radians(psi)), math.sin(math.radians(psi))])
                    slot = math.degrees(math.atan2(pin[1], pin[0] - center_distance)) - 180.0
                    return (slot + 180.0) % 360.0 - 180.0 - 180 / slots
                for x, star in samples[1:-1]:
                    self.assertAlmostEqual(star, slot_angle(x - 15), delta=1e-9)
                for psi in np.linspace(-w, w, 37):
                    # Linear interpolation between samples at most 0.5 degrees apart.
                    self.assertAlmostEqual(coupled_position(table, 15 + psi), slot_angle(psi), delta=0.05)
                # Dwell, then one pitch per turn over many turns.
                self.assertEqual(coupled_position(table, 15 - 120), 0.0)
                self.assertAlmostEqual(coupled_position(table, 15 + 3 * 360 + 100), -4 * pitch, places=9)
                parts, definition = coupled_parts()
                definition["couplings"] = [{"id": "index", "driver": "crank", "follower": "star", **table}]
                motion = validate_motion(definition, parts)
                self.assertAlmostEqual(motion["joints"][1]["position"], coupled_position(table, 0.0), places=12)

    def test_geneva_options(self):
        self.assertEqual(geneva(6, sign=1)["advance"], 60.0)
        self.assertLessEqual(len(geneva(4, step=0.1)["samples"]), 4096)
        for bad in ({"slots": 2}, {"slots": 6.5}, {"slots": True}, {"slots": 6, "sign": 0},
                    {"slots": 6, "step": 0}, {"slots": 6, "step": 0.001}):
            with self.subTest(arguments=bad), self.assertRaises(SDFError):
                geneva(**bad)


class NormalizationTests(unittest.TestCase):
    def test_generic_results(self):
        shape = radsdf.sphere(2)
        single = sdf_build.normalize_result(shape)
        self.assertEqual(single.parts, [{"id": "solid/0", "name": "Part 1", "solidIndices": [0], "children": None}])
        self.assertIs(single.shapes[0], shape)
        for result in ([shape, radsdf.box(1)], (shape, radsdf.box(1))):
            parts = sdf_build.normalize_result(result).parts
            self.assertEqual([(part["id"], part["name"], part["solidIndices"]) for part in parts],
                             [("solid/0", "Part 1", [0]), ("solid/1", "Part 2", [1])])
        cases = [([], "empty list"), (radsdf.circle(1), "2D profile"), ([shape, radsdf.rect(1)], r"result\[1\] is a 2D"),
                 ([shape, [shape]], "list, not a 3D shape"), ([Assembly()], "an Assembly"),
                 (Assembly().add(shape), "is a Part"), (3, "not int"), (None, "not NoneType")]
        for result, message in cases:
            with self.subTest(result=result), self.assertRaisesRegex(ModelError, message):
                sdf_build.normalize_result(result)
        with self.assertRaisesRegex(ModelError, "at most 4096"):
            sdf_build.normalize_result([shape] * 4097)

    def test_identities_names_and_folders(self):
        root = Assembly()
        root.add(radsdf.sphere(1), id="case", name="Case")
        root.add(radsdf.sphere(1), name="PCB")
        root.add(radsdf.sphere(1), name="Lid", id="name:Lid")
        root.add(radsdf.sphere(1), name="Lid")
        root.add(radsdf.sphere(1))
        group = root.folder("Screws")
        group.add(radsdf.sphere(1), name="Screw")
        group.add(radsdf.sphere(1), name="Screw")
        root.folder("Empty").folder("Nested")
        root.add(radsdf.sphere(1), id="user-folder:mine", name="Mine", color="#FF0000")
        normalized = normalize(root)
        parts = normalized.parts
        self.assertEqual([(part["id"], part["name"]) for part in parts],
                         [("case", "Case"), ("name:PCB", "PCB"), ("name:Lid", "Lid"),
                          ("path:[\"Lid\"]", "Lid"), ("path:[\"@4\"]", "Part 5"), ("name:Screws", "Screws"),
                          ("user-folder:mine", "Mine")])
        self.assertEqual([(part["id"], part["solidIndices"]) for part in parts[5]["children"]],
                         [('path:["Screws","Screw"]', [5]), ('generated:path:["Screws","Screw"]', [6])])
        self.assertEqual(parts[6], {"id": "user-folder:mine", "name": "Mine", "solidIndices": [], "children": [
            {"id": 'body:["user-folder:mine",0]', "name": "Mine", "solidIndices": [7], "children": None,
             "color": {"red": 1.0, "green": 0.0, "blue": 0.0}}]})
        self.assertEqual(len(normalized.shapes), 8)
        sdf_build.validate_parts(parts, 8)

    def test_named_roots_and_duplicates(self):
        root = Assembly("Robot")
        root.add(radsdf.sphere(1), name="Head")
        self.assertEqual(normalize(root).parts[0]["id"], "root")
        self.assertEqual(normalize(root).parts[0]["name"], "Robot")
        root = Assembly(id="robot")
        root.add(radsdf.sphere(1))
        self.assertEqual([part["id"] for part in normalize(root).parts], ["robot"])
        root = Assembly()
        root.add(radsdf.sphere(1), id="root")
        sub = root.folder(None)
        sub.add(radsdf.sphere(1))
        self.assertEqual([part["id"] for part in normalize(root).parts], ["root", 'path:["@1"]'])
        root = Assembly()
        root.add(radsdf.sphere(1), id="same")
        root.folder("F", id="same").add(radsdf.sphere(1))
        with self.assertRaisesRegex(ModelError, "'same' is used more than once"):
            normalize(root)
        with self.assertRaisesRegex(ModelError, "contains no parts"):
            normalize(Assembly("Nothing"))
        deep = root = Assembly()
        for _ in range(34):
            deep = deep.folder("Level")
        deep.add(radsdf.sphere(1))
        with self.assertRaisesRegex(ModelError, "levels deep"):
            normalize(root)

    def test_inherited_colors_print_down_and_placements(self):
        root = Assembly("Car", color="#0000FF", print_down=(0, 0, -1), at=Placement(translate=(100, 0, 0)))
        body = root.add(radsdf.box(10), name="Body")
        wheels = root.folder("Wheels", color=(0.1, 0.1, 0.1), print_down="y",
                             at=Placement(translate=(0, 5, 0), rotate=(90, "x")))
        wheels.add(radsdf.cylinder(3, 2), name="Front", at=Placement(translate=(0, 0, 7)))
        wheels.add(radsdf.cylinder(3, 2), name="Rear", color="#FFFFFF", print_down=(1, 0, 0))
        normalized = normalize(root)
        parts = normalized.parts
        self.assertEqual(physical(parts, 0)["color"], {"red": 0.0, "green": 0.0, "blue": 1.0})
        self.assertEqual(physical(parts, 1)["color"], {"red": 0.1, "green": 0.1, "blue": 0.1})
        self.assertEqual(physical(parts, 2)["color"], {"red": 1.0, "green": 1.0, "blue": 1.0})
        self.assertNotIn("color", parts[0])
        self.assertNotIn("color", parts[0]["children"][1])
        self.assertEqual(normalized.print_down, {0: [0.0, 0.0, -1.0], 1: [0.0, 1.0, 0.0], 2: [1.0, 0.0, 0.0]})
        # Placements compose from the root down: car, wheels folder, then the part.
        expected = Placement(translate=(100, 0, 0)) @ Placement(translate=(0, 5, 0), rotate=(90, "x")) @ \
            Placement(translate=(0, 0, 7))
        front = normalized.shapes[1]
        self.assertEqual(front.opcode, radsdf.Opcode.TRANSFORM)
        params = np.array(front.params)
        placed = np.eye(4)
        placed[:3, :3], placed[:3, 3] = params[:9].reshape(3, 3), params[9:12]
        np.testing.assert_allclose(placed, expected.matrix, atol=1e-12)
        self.assertEqual(params[12], 1.0)
        self.assertEqual(front.children[0].opcode, radsdf.Opcode.CYLINDER)
        self.assertEqual(normalized.shapes[0].params[9:12], (100.0, 0.0, 0.0))
        self.assertIs(normalized.shapes[0].children[0], body.shape)

    def test_reference_parts_inherit_like_colors_and_have_no_print_down(self):
        root = Assembly("Enclosure", print_down=(0, 0, -1))
        root.add(radsdf.box(10), name="Base")
        pcb = root.folder("PCB", reference=True)
        pcb.add(radsdf.box(5), name="Board")
        parts = pcb.folder("Components")
        parts.add(radsdf.sphere(1), name="Connector")
        parts.add(radsdf.sphere(1), name="Printed spacer", reference=False, print_down="z")
        root.add(radsdf.sphere(2), name="Probe", reference=True, print_down=(1, 0, 0))
        normalized = normalize(root)
        self.assertEqual(normalized.reference, [1, 2, 4])
        self.assertEqual(normalized.print_down, {0: [0.0, 0.0, -1.0], 3: [0.0, 0.0, 1.0]})
        # The flag is build metadata, not part of the saved parts hierarchy.
        self.assertTrue(all(set(part) == {"id", "name", "solidIndices", "children"} for part in leaves(normalized.parts)))
        plain = Assembly("Plain")
        plain.add(radsdf.box(1), reference=False)
        self.assertEqual(normalize(plain).reference, [])
        self.assertEqual(sdf_build.normalize_result([radsdf.box(1), radsdf.sphere(1)]).reference, [])

    def test_repeated_nodes_are_rejected(self):
        root = Assembly()
        part = root.add(radsdf.sphere(1))
        root._children.append(part)
        with self.assertRaisesRegex(ModelError, "appears twice"):
            normalize(root)


class ProgramTests(unittest.TestCase):
    def test_namespace(self):
        namespace = sdf_build.program_namespace(sdf_build.RADBuildMode(False))
        for name in radsdf.__all__:
            self.assertIs(namespace[name], getattr(radsdf, name))
        for name in ("Assembly", "Part", "Placement", "Motion", "gear", "rack", "screw", "cam", "geneva"):
            self.assertIn(name, namespace)
        self.assertIs(namespace["sdf"], radsdf)
        self.assertIs(namespace["math"], math)
        self.assertIs(namespace["np"], np)
        self.assertEqual(namespace["imported_assets"], {})
        self.assertEqual(namespace["__name__"], "__rad_model__")
        self.assertNotIn("cq", namespace)
        with self.assertRaisesRegex(KeyError, "'board' is not available in this build"):
            namespace["imported_assets"]["board"]

    def test_results_and_print_mode(self):
        result, mode = sdf_build.run_program("result = sdf.sphere(2) | box(1)\n")
        self.assertIsInstance(result, radsdf.SDF3)
        self.assertFalse(mode.printing_read)
        result, mode = sdf_build.run_program("result = sphere(3 if rad.print else 2)\n", printing=True)
        self.assertTrue(mode.printing_read)
        self.assertEqual(result.params, (3.0,))
        self.assertEqual(repr(mode), "rad")

    def test_errors_carry_program_line_numbers(self):
        cases = [("a = 1\nraise ValueError('boom')\n", "Python design at line 2: ValueError: boom"),
                 ("a = (\n", "Python design at line 1: SyntaxError"),
                 ("import sys\nsys.exit(3)\n", "Python design at line 2: SystemExit: 3"),
                 ("x = 1\n\nresult = box(-1)\n", "Python design at line 3: SDFError: box size"),
                 ("result = Assembly()\nresult.add(circle(1))\n", "line 2: SDFError: A part needs a 3D solid"),
                 ("import no_such_module_here\n", "line 1: ModuleNotFoundError"),
                 ("x = 1\n", "must assign its final shape, list of shapes or Assembly to result")]
        for script, message in cases:
            with self.subTest(script=script), self.assertRaisesRegex(ModelError, message.replace("(", r"\(")):
                sdf_build.run_program(script)


class SpecTests(unittest.TestCase):
    def test_build_requests(self):
        sdf_build.validate_spec(spec("result = sphere(1)"))
        sdf_build.validate_spec(spec("result = sphere(1)", print=True))
        inputs = [{"id": "a", "name": "A", "format": "stl", "path": "input-000.stl"},
                  {"id": "b", "name": "B", "format": "obj", "path": "input-001.obj"}]
        sdf_build.validate_spec(spec("result = sphere(1)", inputAssets=inputs))
        # Input names follow the app's limit of 256 UTF-8 bytes.
        sdf_build.validate_spec(spec("result = sphere(1)", inputAssets=[dict(inputs[0], name="é" * 128)]))
        bad = [spec("x", inputAssets=[dict(inputs[0], name="é" * 129)]), spec("x", print=False), spec("x", sourceAsset="source.step"), spec(" "), spec("x" * 200_001),
               dict(spec("x"), units="in"), dict(spec("x"), title=""), dict(spec("x"), operation="build"),
               spec("x", inputAssets=[{"id": "a", "name": "A", "format": "step", "path": "input-000.step"}]),
               spec("x", inputAssets=[dict(inputs[0], validatedBREP="input-000.validated.brep")]),
               spec("x", inputAssets=[inputs[1]]), spec("x", inputAssets=[inputs[0], dict(inputs[1], id="a")])]
        for request in bad:
            with self.subTest(request=request), self.assertRaises(ModelError):
                sdf_build.validate_spec(request)


class WriteResultTests(unittest.TestCase):
    def test_sections_are_packed_in_order_and_the_input_is_kept(self):
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory)
            preview = {"solids": 1, "vertices": [[0, 0, 0], [1, 0, 0], [0, 1, 0]], "normals": np.zeros((3, 3)),
                       "triangles": [[0, 1, 2]], "edges": np.zeros((1, 6), np.float32), "triangleSolids": [0],
                       "edgeSolids": [0]}
            sdf_build.write_preview_result(preview, output)
            self.assertIn("vertices", preview)
            result = json.loads((output / "result.json").read_text())
            sections = result["meshBuffers"]["sections"]
            self.assertEqual(list(sections), ["positions", "normals", "triangles", "edges", "triangleSolids",
                                              "edgeSolids"])
            self.assertEqual([(item["offset"], item["count"]) for item in sections.values()],
                             [(0, 9), (36, 9), (72, 3), (84, 6), (108, 1), (112, 1)])
            self.assertEqual((output / "mesh.bin").stat().st_size, 116)
            self.assertEqual(sorted(path.name for path in output.iterdir()), ["mesh.bin", "result.json"])
            for malformed in ({"vertices": [[0, 0]]}, {"triangles": [[0, 1, 2], [3]]}, {"triangleSolids": [2**40]}):
                with self.subTest(malformed=malformed), self.assertRaisesRegex(ModelError, "malformed"):
                    sdf_build.write_preview_result(malformed, output)
            self.assertEqual(sorted(path.name for path in output.iterdir()), ["mesh.bin", "result.json"])


SECTION_TYPES = {"positions": ("<f4", 3), "normals": ("<f4", 3), "triangles": ("<u4", 3), "edges": ("<f4", 6),
                 "triangleSolids": ("<i4", 1), "edgeSolids": ("<i4", 1)}

RESULT_KEYS = ["buildID", "artifactPath", "partRoots", "bounds", "volume", "area", "solids", "parts", "partStats",
               "mesh", "assemblyMotion", "printVersion", "printDown", "validatedHash", "buildSeconds", "threads",
               "runtime", "warnings", "meshBuffers"]

ARM = '''
result = Assembly("Hinged arm")
base = result.add(box((42, 32, 6), center=(0, 0, 3)) - cylinder(4, 20, center=(0, 0, 3)), id="base",
                  name="Base", color="#336699")
arm = result.add(box((50, 12, 5), radius=1), id="arm", name="Arm", color=(0.2, 0.4, 0.8),
                 print_down=(0, 0, -1), at=Placement(translate=(20, 0, 10 + (1 if rad.print else 0))))
m = result.motion
m.ground(base)
m.revolute("hinge", m.connector("base-pivot", base, (0, 0, 10)), m.connector("arm-pivot", arm, (0, 0, 10)),
           name="Arm angle", limits=(-120, 120))
'''


def read_buffers(output, result):
    data = (output / "mesh.bin").read_bytes()
    arrays, expected = {}, 0
    for name, section in result["meshBuffers"]["sections"].items():
        dtype, width = SECTION_TYPES[name]
        if section["offset"] != expected:
            raise AssertionError(f"{name} is not contiguous")
        values = np.frombuffer(data, dtype=dtype, count=section["count"], offset=section["offset"])
        arrays[name] = values.reshape(-1, width) if width > 1 else values
        expected += 4 * section["count"]
    if expected != len(data):
        raise AssertionError("mesh.bin has trailing bytes")
    return arrays


def rsdf_node_count(data):
    magic, version, count, reserved = struct.unpack_from("<4sIII", data)
    if (magic, version, reserved) != (b"RSDF", 1, 0):
        raise AssertionError("bad RSDF header")
    offset = 16
    for _ in range(count):
        _, children, params, _ = struct.unpack_from("<4I", data, offset)
        offset += 16 + 4 * children + 8 * params
    if offset != len(data):
        raise AssertionError("RSDF length mismatch")
    return count


@unittest.skipUnless(kernel_available(), "the native SDF kernel is not available")
class BuildTests(unittest.TestCase):
    def build(self, script, threads=4, **extra):
        temporary = tempfile.TemporaryDirectory(prefix="sdf-build-")
        self.addCleanup(temporary.cleanup)
        directory = Path(temporary.name)
        request = spec(script, **extra)
        sdf_build.build(request, directory, directory, threads)
        result = json.loads((directory / "result.json").read_text())
        return directory, request, result, read_buffers(directory, result)

    def assert_mesh_contract(self, result, arrays):
        positions, triangles = arrays["positions"], arrays["triangles"]
        solids = result["solids"]
        self.assertEqual(len(arrays["normals"]), len(positions))
        self.assertEqual(len(arrays["triangleSolids"]), len(triangles))
        self.assertEqual(len(arrays["edgeSolids"]), len(arrays["edges"]))
        self.assertLessEqual(len(triangles), sdf_build.MAX_TRIANGLES)
        self.assertLessEqual(len(positions), sdf_build.MAX_VERTICES)
        self.assertLessEqual(len(arrays["edges"]), sdf_build.MAX_EDGE_SEGMENTS)
        owners = arrays["triangleSolids"]
        self.assertTrue(np.all(np.diff(owners) >= 0), "triangles are contiguous per part")
        self.assertEqual(sorted(set(owners.tolist())), list(range(solids)))
        lengths = np.linalg.norm(arrays["normals"].astype(np.float64), axis=1)
        self.assertLess(np.max(np.abs(lengths - 1)), 1e-5)
        lo, hi = np.array(result["bounds"][:3]), np.array(result["bounds"][3:])
        self.assertTrue(np.all(positions >= lo) and np.all(positions <= hi))
        self.assertEqual(lo.tolist(), positions.min(axis=0).astype(np.float64).tolist())
        key = lambda values: np.ascontiguousarray(values, dtype=np.float32).view(np.dtype((np.void, 12))).ravel()
        self.assertEqual([stats["solid"] for stats in result["partStats"]], list(range(solids)))
        for stats in result["partStats"]:
            solid = stats["solid"]
            own = triangles[owners == solid]
            self.assertEqual(stats["triangles"], len(own))
            vertices = np.unique(own)
            part_positions = positions[vertices]
            np.testing.assert_array_equal(stats["bounds"], [*part_positions.min(axis=0).tolist(),
                                                            *part_positions.max(axis=0).tolist()])
            # Crease endpoints are exact copies of the same part's vertex positions.
            ends = key(arrays["edges"][arrays["edgeSolids"] == solid].reshape(-1, 3))
            self.assertTrue(np.isin(ends, key(part_positions)).all())
            # Closed and consistently oriented once split vertices are welded by position.
            _, welded = np.unique(key(positions), return_inverse=True)
            corners = welded.ravel()[own]
            directed = np.concatenate([corners[:, [0, 1]], corners[:, [1, 2]], corners[:, [2, 0]]])
            forward = set(map(tuple, directed.tolist()))
            self.assertEqual(len(forward), len(directed), "no directed edge repeats")
            self.assertTrue(all((b, a) in forward for a, b in forward), "every edge has a reverse")
            # Outward winding: the signed volume matches the reported volume.
            points = positions.astype(np.float64)[own]
            signed = np.einsum("ij,ij->i", points[:, 0], np.cross(points[:, 1], points[:, 2])).sum() / 6
            self.assertAlmostEqual(signed, stats["volume"], delta=1e-4 * stats["volume"])

    def test_result_and_mesh_buffers_follow_the_protocol(self):
        output, request, result, arrays = self.build(ARM)
        # The program reads rad.print, so it defines a distinct printing version.
        self.assertEqual(list(result), RESULT_KEYS)
        self.assertIs(result["printVersion"], True)
        self.assertEqual(str(uuid.UUID(result["buildID"])), result["buildID"])
        self.assertEqual(result["artifactPath"], str((output / "model.rsdf").resolve()))
        count = rsdf_node_count((output / "model.rsdf").read_bytes())
        self.assertEqual(len(result["partRoots"]), 2)
        self.assertTrue(all(0 <= root < count for root in result["partRoots"]))
        self.assertEqual(result["solids"], 2)
        self.assertEqual(result["parts"], [{"id": "root", "name": "Hinged arm", "solidIndices": [], "children": [
            {"id": "base", "name": "Base", "solidIndices": [0], "children": None,
             "color": {"red": 0x33 / 255, "green": 0x66 / 255, "blue": 0x99 / 255}},
            {"id": "arm", "name": "Arm", "solidIndices": [1], "children": None,
             "color": {"red": 0.2, "green": 0.4, "blue": 0.8}}]}])
        self.assertEqual(result["printDown"], [{"solid": 1, "direction": [0.0, 0.0, -1.0]}])
        self.assertEqual(result["assemblyMotion"]["joints"][0]["limits"], [-120.0, 120.0])
        self.assertEqual(result["mesh"]["boundaryEdges"], 0)
        self.assertAlmostEqual(result["mesh"]["cellSize"], 66 / 256)
        self.assertEqual(result["validatedHash"], sdf_build.validation_hash(request, output))
        self.assertEqual(set(result["runtime"]), {"python", "numpy", "kernel"})
        self.assertEqual(result["runtime"]["kernel"], radsdf.kernel_version())
        self.assertEqual(result["threads"], 4)
        self.assertGreater(result["buildSeconds"], 0)
        self.assertEqual(result["warnings"], [])
        self.assertEqual(sorted(path.name for path in output.iterdir()), ["mesh.bin", "model.rsdf", "result.json"])
        self.assert_mesh_contract(result, arrays)
        base, arm = result["partStats"]
        self.assertAlmostEqual(base["volume"], 42 * 32 * 6 - math.pi * 16 * 6, delta=0.002 * base["volume"])
        self.assertEqual(base["components"], 1)
        np.testing.assert_allclose(arm["bounds"], [-5, -6, 7.5, 45, 6, 12.5], atol=0.02)
        np.testing.assert_allclose(arm["centroid"], [20, 0, 10], atol=0.01)
        self.assertAlmostEqual(result["volume"], base["volume"] + arm["volume"])
        self.assertAlmostEqual(result["area"], base["area"] + arm["area"])

    @unittest.skipUnless(os.environ.get("RAD_SDF_LIBRARY"), "needs RAD_SDF_LIBRARY for the C API")
    def test_the_artifact_loads_in_the_kernel_and_its_roots_are_the_placed_parts(self):
        import ctypes
        output, _, result, arrays = self.build(ARM)
        data = (output / "model.rsdf").read_bytes()
        library = ctypes.CDLL(os.environ["RAD_SDF_LIBRARY"])
        library.rsdf_graph_create.restype = ctypes.c_void_p
        library.rsdf_graph_create.argtypes = [ctypes.c_char_p, ctypes.c_size_t]
        library.rsdf_graph_free.argtypes = [ctypes.c_void_p]
        library.rsdf_evaluate.argtypes = [ctypes.c_void_p, ctypes.c_uint32, ctypes.c_void_p, ctypes.c_uint64,
                                          ctypes.c_void_p, ctypes.c_int32]
        graph = library.rsdf_graph_create(data, len(data))
        self.assertTrue(graph)
        self.addCleanup(library.rsdf_graph_free, graph)
        # Mesh vertices of each part lie on that part's root surface, in world coordinates.
        for solid, root in enumerate(result["partRoots"]):
            own = np.unique(arrays["triangles"][arrays["triangleSolids"] == solid])
            points = np.ascontiguousarray(arrays["positions"][own[::97]], dtype=np.float64)
            distances = np.empty(len(points))
            self.assertEqual(library.rsdf_evaluate(graph, root, points.ctypes.data, len(points),
                                                   distances.ctypes.data, 1), 0)
            self.assertLess(np.max(np.abs(distances)), 0.25 * result["mesh"]["cellSize"])

    def test_print_builds_and_receipts(self):
        output, request, result, _ = self.build(ARM, print=True)
        self.assertEqual(list(result), RESULT_KEYS)
        self.assertIs(result["printVersion"], True)
        np.testing.assert_allclose(result["partStats"][1]["centroid"], [20, 0, 11], atol=0.01)
        design = sdf_build.validation_hash(spec(ARM), output)
        self.assertNotEqual(result["validatedHash"], design)
        self.assertEqual(sdf_build.validation_hash(dict(spec(ARM), title="Other", summary="Other"), output), design)
        self.assertNotEqual(sdf_build.validation_hash(spec(ARM + "\n"), output), design)
        _, _, plain, _ = self.build("result = box(10)")
        self.assertEqual(list(plain), [key for key in RESULT_KEYS
                                       if key not in ("assemblyMotion", "printVersion", "printDown")])
        self.assertEqual(plain["parts"], [{"id": "solid/0", "name": "Part 1", "solidIndices": [0], "children": None}])
        self.assertAlmostEqual(plain["volume"], 1000, delta=0.5)
        self.assertAlmostEqual(plain["area"], 600, delta=0.5)
        rules = sdf_build.validation_rules(radsdf.kernel_version())
        for fragment in (f"kernel={radsdf.kernel_version()}", "cells=256", "triangles=1000000", "crease=35",
                         "refine=6"):
            self.assertIn(fragment, rules)

    def test_a_geneva_drive_builds_with_its_coupling(self):
        script = '''
N, C = 6, 60.0
A = C * math.sin(math.pi / N)
frame = box((150, 80, 4), center=(30, 0, -3))
crank = cylinder(A * 0.9, 6, center=(0, 0, 3)) | cylinder(3, 12, center=(A, 0, 6))
star = cylinder(C * math.cos(math.pi / N), 6, center=(C, 0, 3))
for k in range(N):
    angle = 180 + (k + 0.5) * 360 / N
    star = star - box((40, 6.6, 8)).rotate(angle).translate((C + 30 * math.cos(math.radians(angle)),
                                                             30 * math.sin(math.radians(angle)), 3))
result = Assembly("Geneva drive")
f = result.add(frame, id="frame", name="Frame")
c = result.add(crank, id="crank", name="Crank", at=Placement(translate=(0, 0, 0.5)))
s = result.add(star, id="star", name="Star", at=Placement(translate=(0, 0, 0.5)))
m = result.motion
m.ground(f)
drive = m.revolute("drive", m.connector("frame-crank", f, (0, 0, 0)), m.connector("crank-axis", c, (0, 0, 0)))
index = m.revolute("index", m.connector("frame-star", f, (C, 0, 0)), m.connector("star-axis", s, (C, 0, 0)))
m.couple("geneva", drive, index, **geneva(N))
'''
        _, _, result, arrays = self.build(script)
        motion = result["assemblyMotion"]
        self.assertEqual([joint["position"] for joint in motion["joints"]], [0.0, -30.0])
        coupling = motion["couplings"][0]
        self.assertEqual((coupling["type"], coupling["period"], coupling["advance"]), ("table", 360.0, -60.0))
        self.assertEqual(coupling["samples"], geneva(6)["samples"])
        self.assertEqual(result["solids"], 3)
        self.assert_mesh_contract(result, arrays)

    def test_input_assets_are_part_of_the_receipt(self):
        with tempfile.TemporaryDirectory() as directory:
            directory = Path(directory)
            (directory / "input-000.stl").write_bytes(b"solid a\nendsolid a\n")
            (directory / "input-001.obj").write_bytes(b"v 0 0 0\n")
            assets = [{"id": "a", "name": "A", "format": "stl", "path": "input-000.stl"},
                      {"id": "b", "name": "B", "format": "obj", "path": "input-001.obj"}]
            request = spec("result = box(1)", inputAssets=assets)
            first = sdf_build.validation_hash(request, directory)
            renamed = spec("result = box(1)", inputAssets=[dict(assets[0], name="Z"), assets[1]])
            self.assertEqual(sdf_build.validation_hash(renamed, directory), first)
            (directory / "input-001.obj").write_bytes(b"v 0 0 1\n")
            self.assertNotEqual(sdf_build.validation_hash(request, directory), first)
            (directory / "input-000.stl").unlink()
            with self.assertRaisesRegex(ModelError, "missing"):
                sdf_build.validation_hash(request, directory)

    def test_builds_are_deterministic_for_any_thread_count(self):
        script = "result = [sphere(5) | box((4, 4, 14)), torus(6, 1.5, center=(20, 0, 0))]"
        first_output, _, first, _ = self.build(script, threads=1)
        second_output, _, second, _ = self.build(script, threads=8)
        self.assertEqual((first_output / "mesh.bin").read_bytes(), (second_output / "mesh.bin").read_bytes())
        self.assertEqual((first_output / "model.rsdf").read_bytes(), (second_output / "model.rsdf").read_bytes())
        self.assertEqual(first["validatedHash"], second["validatedHash"])
        self.assertNotEqual(first["buildID"], second["buildID"])

    def test_design_errors_name_the_part(self):
        cases = [
            ("result = Assembly()\nresult.add(half_space(), name='Floor')",
             r'Part "Floor" \(id name:Floor\) is unbounded: .* Intersect it with a box'),
            ("result = [box(10), sphere(1) & sphere(1, center=(10, 0, 0))]", r"Part 2 \(result\[1\]\) is empty"),
            ("result = [box(50), sphere(0.01, center=(0, 0, 30))]",
             r"Part 2 \(result\[1\]\) has no surface at the preview cell size of 0\.2\d* mm: it is empty or "
             r"thinner than about 0\.4\d* mm everywhere\. Make its features thicker"),
            ("result = sphere(1, center=(2e6, 0, 0))", "The design exceeds the size limit"),
            ("result = Assembly('Empty')", "contains no parts"),
            ("result = circle(3)", "2D profile"),
            ("result = box(1)\nresult = undefined_name", "Python design at line 2: NameError"),
        ]
        for script, message in cases:
            with self.subTest(script=script), tempfile.TemporaryDirectory() as directory:
                with self.assertRaisesRegex(ModelError, message):
                    sdf_build.build(spec(script), Path(directory), Path(directory), 2)
                self.assertFalse((Path(directory) / "result.json").exists())
                self.assertFalse((Path(directory) / "model.rsdf").exists())

    def test_the_preview_budget_coarsens_the_cell_size(self):
        _, _, result, arrays = self.build("result = box(1).grid_array((8, 8, 8), (2, 2, 2))", threads=8)
        default = 15 / 256
        self.assertGreater(result["mesh"]["cellSize"], default)
        self.assertLessEqual(len(arrays["triangles"]), sdf_build.MAX_TRIANGLES)
        self.assertGreater(len(arrays["triangles"]), sdf_build.MAX_TRIANGLES / 2)
        self.assertIn("instead of 0.05859 mm", result["warnings"][0])
        self.assertIn("The design has 512 disconnected pieces", result["warnings"][1])
        self.assertAlmostEqual(result["volume"], 512, delta=1)
        self.assertEqual(result["partStats"][0]["components"], 512)

    def test_the_preview_budget_counts_the_surface_before_refinement(self):
        # The mesher refines features narrower than a cell, which these beads
        # are at the probe's cells but not at the preview's: counting the
        # probe's refined triangles would coarsen a preview that fits.
        post = radsdf.union(radsdf.cylinder(4, 2, center=(0, 0, 1)), radsdf.cylinder(4, 2, center=(0, 0, 3)), smooth=0.6)
        shape = post.grid_array((3, 3, 1), (10, 10, 1))
        graph = radsdf.Graph([shape])
        self.addCleanup(graph.close)
        box = (shape.bounds.lo, shape.bounds.hi)
        default = float(np.max(box[1] - box[0])) / sdf_build.PREVIEW_CELLS_ACROSS
        probe = graph.mesh(graph.roots[0], cell_size=default * sdf_build.PREVIEW_PROBE_FACTOR,
                           crease_angle=sdf_build.PREVIEW_CREASE_ANGLE)
        self.assertGreater(len(probe.triangles), 1.5 * sdf_build.unrefined_triangles(probe))
        from unittest import mock
        with mock.patch.object(sdf_build, "MAX_TRIANGLES", 500_000):
            cell, expected = sdf_build.preview_cell_size(graph, [box], 2)
            self.assertEqual(cell, expected)
            preview = graph.mesh(graph.roots[0], cell_size=cell, crease_angle=sdf_build.PREVIEW_CREASE_ANGLE)
            self.assertLess(len(preview.triangles), sdf_build.MAX_TRIANGLES)

    def test_too_detailed_designs_fail_after_two_remeshes(self):
        calls = []
        real = sdf_build._mesh_part

        def dense(graph, shape, label, cell, threads):
            calls.append(cell)
            mesh = real(graph, shape, label, cell, threads)
            return radsdf.Mesh(np.zeros((sdf_build.MAX_VERTICES + 1, 3)), np.tile([0.0, 0.0, 1.0], (
                sdf_build.MAX_VERTICES + 1, 1)), mesh.triangles, mesh.creases, mesh.stats)
        from unittest import mock
        with tempfile.TemporaryDirectory() as directory, mock.patch.object(sdf_build, "_mesh_part", dense):
            with self.assertRaisesRegex(ModelError, "too detailed to preview within 1,000,000 triangles"):
                sdf_build.build(spec("result = [sphere(2), sphere(2, center=(9, 0, 0))]"), Path(directory),
                                Path(directory), 2)
        # One pass and two remeshes, each stopping at the first part over the limits.
        self.assertEqual(len(calls), 3)
        self.assertTrue(calls[0] < calls[1] < calls[2])

    def test_the_meshed_graph_is_the_published_artifact(self):
        # The parts' graph is encoded once; the kernel meshes model.rsdf's bytes.
        encoded, loaded = [], []
        encode, load = radsdf._encode, radsdf.Graph.from_bytes

        def counting_encode(roots, **options):
            encoded.append(len(roots))
            return encode(roots, **options)

        def recording_load(data, roots):
            loaded.append((data, roots))
            return load(data, roots)
        from unittest import mock
        with mock.patch.object(radsdf, "_encode", counting_encode), \
                mock.patch.object(radsdf.Graph, "from_bytes", recording_load):
            output, _, result, _ = self.build("result = [sphere(5), box(4, center=(10, 0, 0))]")
        self.assertEqual(encoded, [2])
        self.assertEqual(loaded, [((output / "model.rsdf").read_bytes(), result["partRoots"])])


@unittest.skipUnless(kernel_available(), "the native SDF kernel is not available")
class WorkerTests(unittest.TestCase):
    """The worker's process framing around sdf_build."""

    def worker(self, *arguments, stdin=None):
        import subprocess
        return subprocess.run([sys.executable, str(ENGINE / "worker.py"), *map(str, arguments)], input=stdin,
                              capture_output=True, text=True, timeout=120)

    def test_command_line_builds_and_failures(self):
        with tempfile.TemporaryDirectory() as directory:
            directory = Path(directory)
            (directory / "request.json").write_text(json.dumps(spec("result = box((30, 20, 4)) - cylinder(3, 10)")))
            completed = self.worker("--request", directory / "request.json", "--output", directory / "out",
                                    "--threads", 2)
            self.assertEqual(completed.returncode, 0, completed.stderr)
            self.assertEqual(json.loads(completed.stdout), {"status": "complete", "solids": 1})
            result = json.loads((directory / "out" / "result.json").read_text())
            self.assertAlmostEqual(result["volume"], 2400 - math.pi * 36, delta=1.5)
            (directory / "bad.json").write_text(json.dumps(spec("x = 1\nresult = box(-1)\n")))
            failed = self.worker("--request", directory / "bad.json", "--output", directory / "bad")
            self.assertEqual(failed.returncode, 1)
            self.assertIn("RAD CAD engine: Python design at line 2: SDFError", failed.stderr)
            report = json.loads((directory / "bad" / "error.json").read_text())
            self.assertEqual((report["formatVersion"], report["stage"], report["failureKind"]), (1, "worker", "design"))
            self.assertFalse((directory / "bad" / "result.json").exists())
            for removed in ("--defer-step", "--verify-later", "--validated-hash=" + "0" * 64):
                self.assertEqual(self.worker("--request", directory / "request.json", "--output", directory / "x",
                                             removed).returncode, 2)

    def test_prepared_build_reads_its_options_from_a_fifo(self):
        import subprocess
        for options, code in (({"threads": 3}, 0), ({"threads": 3, "validatedHash": "0" * 64}, 1),
                              ({"threads": 3, "deferStep": True}, 1)):
            with self.subTest(options=options), tempfile.TemporaryDirectory() as directory:
                output = Path(directory)
                (output / "request.json").write_text(json.dumps(spec("result = sphere(5)")))
                os.mkfifo(output / "start", 0o600)
                process = subprocess.Popen([sys.executable, str(ENGINE / "worker.py"), "--await-build",
                                            str(output / "start"), "--output", str(output)],
                                           stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
                with open(output / "start", "w") as stream:
                    stream.write(json.dumps(options))
                _, stderr = process.communicate(timeout=120)
                self.assertEqual(process.returncode, code, stderr)
                if code == 0:
                    self.assertEqual(json.loads((output / "result.json").read_text())["threads"], 3)
                else:
                    self.assertIn("options are invalid", (output / "error.json").read_text())

    def test_serving_answers_native_operations_only(self):
        with tempfile.TemporaryDirectory() as directory:
            directory = Path(directory)
            (directory / "runtime.json").write_text(json.dumps({"operation": "runtime"}))
            (directory / "build.json").write_text(json.dumps(spec("result = sphere(1)")))
            (directory / "dxf.json").write_text(json.dumps({"operation": "dxf"}))
            lines = "".join(json.dumps({"id": index, "request": str(directory / name), "output": str(directory / str(index)),
                                        "threads": 2}) + "\n"
                            for index, name in enumerate(("runtime.json", "build.json", "dxf.json")))
            completed = self.worker("--serve", stdin=lines)
            replies = [json.loads(line) for line in completed.stdout.splitlines()]
            self.assertEqual([reply["status"] for reply in replies], ["complete", "error", "error"])
            self.assertIn("only native geometry operations", replies[1]["message"])
            runtime = json.loads((directory / "0" / "result.json").read_text())
            self.assertEqual(runtime, {"runtime": sdf_build.runtime_info()})
            self.assertFalse((directory / "1" / "result.json").exists())
            self.assertTrue((directory / "1" / "error.json").exists())


class PoseRecordTests(unittest.TestCase):
    def test_rigid_pose_records(self):
        from assembly_motion import AssemblyMotionError, validate_pose_records
        turn = [0, -1, 0, 5, 1, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1]
        self.assertEqual(validate_pose_records([{"solid": 1, "matrix": turn}], 2), {1: [float(v) for v in turn]})
        self.assertEqual(validate_pose_records([], 0), {})
        scaled = [2, 0, 0, 0, 0, 2, 0, 0, 0, 0, 2, 0, 0, 0, 0, 1]
        mirrored = [1, 0, 0, 0, 0, 1, 0, 0, 0, 0, -1, 0, 0, 0, 0, 1]
        bad = [([{"solid": 2, "matrix": turn}], 2), ([{"solid": 0, "matrix": turn}] * 2, 2),
               ([{"solid": 0, "matrix": scaled}], 1), ([{"solid": 0, "matrix": mirrored}], 1),
               ([{"solid": 0, "matrix": turn[:12]}], 1), ([{"solid": True, "matrix": turn}], 1),
               ({"solid": 0}, 1), ([{"solid": 0, "matrix": turn, "extra": 1}], 1),
               ([{"solid": 0, "matrix": turn[:15] + [2]}], 1), ([{"solid": 0, "matrix": [math.nan] * 16}], 1)]
        for records, count in bad:
            with self.subTest(records=records), self.assertRaises(AssemblyMotionError):
                validate_pose_records(records, count)
