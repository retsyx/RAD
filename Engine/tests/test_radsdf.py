"""radsdf: shape graphs, RSDF encoding, mesh helpers and the native kernel binding.

Encoding tests decode serialize() output with a reader written from the RSDF
specification in SDF-DESIGN.md, independent of the module. Native tests run
when libRADSDF.dylib loads and are skipped otherwise.
"""
import copy
import ctypes
import math
import os
from pathlib import Path
import re
import struct
import subprocess
import sys
import tempfile
import threading
import time
import unittest
import warnings
from unittest import mock

import numpy as np

ENGINE = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ENGINE))
import radsdf
from radsdf import SDFError

# The RSDF opcode table (SDF-DESIGN.md).
OPCODES = {1: "SPHERE", 2: "BOX", 3: "CYLINDER", 4: "CONE", 5: "CAPSULE", 6: "TORUS", 7: "ELLIPSOID",
           8: "HALFSPACE", 20: "TRANSFORM", 21: "STRETCH", 22: "MIRROR", 23: "TWIST", 24: "ELONGATE",
           25: "OFFSET", 26: "SHELL", 30: "UNION", 31: "INTERSECTION", 32: "DIFFERENCE", 40: "EXTRUDE",
           41: "TAPERED_EXTRUDE", 42: "REVOLVE", 43: "LOFT", 100: "CIRCLE", 101: "RECT", 102: "POLYGON",
           103: "SEGMENT", 104: "ELLIPSE", 105: "HALFPLANE", 120: "TRANSFORM2", 121: "STRETCH2",
           122: "MIRROR2", 123: "OFFSET2", 124: "OUTLINE2", 130: "UNION2", 131: "INTERSECTION2",
           132: "DIFFERENCE2"}
OPCODES[9] = "MESH"  # imported triangle meshes (test_mesh_import.py)

ROT_X90 = [[1, 0, 0], [0, 0, -1], [0, 1, 0]]
ROT_Y90 = [[0, 0, 1], [0, 1, 0], [-1, 0, 0]]
ROT_Z90 = [[0, -1, 0], [1, 0, 0], [0, 0, 1]]


def decode(data):
    """[(name, children, params)] of RSDF bytes, checking the framing rules."""
    magic, version, count, reserved = struct.unpack_from("<4sIII", data)
    if (magic, version, reserved) != (b"RSDF", 1, 0):
        raise AssertionError(f"bad header {(magic, version, reserved)}")
    offset, records = 16, []
    for index in range(count):
        opcode, child_count, param_count, reserved = struct.unpack_from("<4I", data, offset)
        offset += 16
        children = struct.unpack_from(f"<{child_count}I", data, offset)
        offset += 4 * child_count
        params = struct.unpack_from(f"<{param_count}d", data, offset)
        offset += 8 * param_count
        if reserved != 0 or any(child >= index for child in children):
            raise AssertionError(f"record {index} is not in topological order")
        records.append((OPCODES[opcode], children, params))
    if offset != len(data):
        raise AssertionError("trailing bytes")
    return records


def N(name, params=(), *children):
    """An expected node: (name, params, children)."""
    return name, tuple(float(value) for value in np.ravel(params)), children


def T(rotation=np.eye(3), offset=(0, 0, 0), scale=1.0):
    """Expected TRANSFORM / TRANSFORM2 parameters."""
    return [*np.ravel(rotation), *offset, scale]


class GraphCase(unittest.TestCase):
    def tree(self, shape):
        data, (root,) = radsdf.serialize(shape)
        records = decode(data)

        def build(index):
            name, children, params = records[index]
            return name, params, tuple(build(child) for child in children)
        return build(root)

    def assertGraph(self, shape, expected):
        self.assertNode(self.tree(shape), expected, "root")

    def assertNode(self, actual, expected, path):
        self.assertEqual(actual[0], expected[0], path)
        self.assertEqual(len(actual[1]), len(expected[1]), f"{path} params")
        np.testing.assert_allclose(actual[1], expected[1], rtol=0, atol=1e-12, err_msg=path)
        self.assertEqual(len(actual[2]), len(expected[2]), f"{path} children")
        for number, (child, want) in enumerate(zip(actual[2], expected[2])):
            self.assertNode(child, want, f"{path}/{number}:{want[0]}")

    def assertRotation(self, params, direction_from, direction_to):
        rotation = np.reshape(params[:9], (3, 3))
        np.testing.assert_allclose(rotation @ rotation.T, np.eye(3), atol=1e-12)
        self.assertAlmostEqual(np.linalg.det(rotation), 1.0, places=12)
        np.testing.assert_allclose(rotation @ direction_from, direction_to, atol=1e-12)


class ConstructorEncodingTests(GraphCase):
    def test_primitives_sit_at_the_origin(self):
        self.assertGraph(radsdf.sphere(2), N("SPHERE", [2]))
        self.assertGraph(radsdf.sphere(2, center=(1, 2, 3)),
                         N("TRANSFORM", T(offset=(1, 2, 3)), N("SPHERE", [2])))
        self.assertGraph(radsdf.box((2, 4, 6), center=(1, 0, 0), radius=0.5),
                         N("TRANSFORM", T(offset=(1, 0, 0)), N("BOX", [1, 2, 3, 0.5])))
        self.assertGraph(radsdf.box(3), N("BOX", [1.5, 1.5, 1.5, 0]))
        self.assertGraph(radsdf.ellipsoid((1, 2, 3), center=(1, 1, 1)),
                         N("TRANSFORM", T(offset=(1, 1, 1)), N("ELLIPSOID", [1, 2, 3])))
        self.assertGraph(radsdf.cylinder(1, 4), N("CYLINDER", [1, 2]))
        self.assertGraph(radsdf.cone(2, 0, 6), N("CONE", [2, 0, 3]))
        self.assertGraph(radsdf.torus(10, 2), N("TORUS", [10, 2]))

    def test_box_radius_is_clamped_to_the_smallest_half_extent(self):
        self.assertGraph(radsdf.box((2, 4, 6), radius=1 + 1e-13), N("BOX", [1, 2, 3, 1]))

    def test_axis_and_center_become_one_transform(self):
        shape = radsdf.cylinder(1, 4, center=(0, 0, 5), axis="x")
        name, params, (child,) = self.tree(shape)
        self.assertEqual((name, child), ("TRANSFORM", ("CYLINDER", (1.0, 2.0), ())))
        self.assertRotation(params, (0, 0, 1), (1, 0, 0))
        self.assertEqual(params[9:], (0.0, 0.0, 5.0, 1.0))
        name, params, (child,) = self.tree(radsdf.cone(2, 1, 6, axis="-z"))
        self.assertEqual(child, ("CONE", (2.0, 1.0, 3.0), ()))
        self.assertRotation(params, (0, 0, 1), (0, 0, -1))
        name, params, (child,) = self.tree(radsdf.torus(10, 2, axis=(0, 3, 0)))
        self.assertEqual(child, ("TORUS", (10.0, 2.0), ()))
        self.assertRotation(params, (0, 0, 1), (0, 1, 0))
        name, params, _ = self.tree(radsdf.cylinder(1, 1, axis=(1, 1, 1)))
        self.assertRotation(params, (0, 0, 1), np.ones(3) / math.sqrt(3))

    def test_quarter_turn_axes_are_exact(self):
        _, params, _ = self.tree(radsdf.cylinder(1, 4, axis="x"))
        self.assertEqual(params[:9], tuple(np.ravel(ROT_Y90).astype(float)))

    def test_shapes_with_explicit_geometry_need_no_transform(self):
        self.assertGraph(radsdf.capsule((0, 0, 0), (0, 0, 5), 1), N("CAPSULE", [0, 0, 0, 0, 0, 5, 1]))
        self.assertGraph(radsdf.half_space((0, 0, 2), (1, 1, 5)), N("HALFSPACE", [0, 0, 1, 5]))
        self.assertGraph(radsdf.half_space("-x", (3, 0, 0)), N("HALFSPACE", [-1, 0, 0, -3]))
        self.assertGraph(radsdf.slot((0, 0), (5, 0), 1), N("SEGMENT", [0, 0, 5, 0, 1]))
        self.assertGraph(radsdf.half_plane((0, 2), (0, 1)), N("HALFPLANE", [0, 1, 1]))

    def test_profiles(self):
        self.assertGraph(radsdf.circle(2), N("CIRCLE", [2]))
        self.assertGraph(radsdf.circle(2, center=(1, 1)),
                         N("TRANSFORM2", T(np.eye(2), (1, 1)), N("CIRCLE", [2])))
        self.assertGraph(radsdf.rect((4, 2), radius=0.5), N("RECT", [2, 1, 0.5]))
        self.assertGraph(radsdf.ellipse((3, 1), center=(0, 2)),
                         N("TRANSFORM2", T(np.eye(2), (0, 2)), N("ELLIPSE", [3, 1])))

    def test_polygon_points_are_stored_in_order_without_repeats(self):
        self.assertGraph(radsdf.polygon([(0, 0), (4, 0), (4, 3), (0, 0)]),
                         N("POLYGON", [0, 0, 4, 0, 4, 3]))
        self.assertGraph(radsdf.polygon([(0, 0), (4, 0), (4, 0), (4, 3), (0, 3)]),
                         N("POLYGON", [0, 0, 4, 0, 4, 3, 0, 3]))
        clockwise = radsdf.polygon([(0, 0), (0, 3), (4, 3), (4, 0)])
        self.assertEqual(clockwise.params, (0.0, 0.0, 0.0, 3.0, 4.0, 3.0, 4.0, 0.0))

    def test_regular_polygon_and_arc_points(self):
        params = np.reshape(radsdf.regular_polygon(6, 2, center=(1, 0)).params, (-1, 2))
        self.assertEqual(len(params), 6)
        np.testing.assert_allclose(params[0], (3, 0), atol=1e-12)
        np.testing.assert_allclose(np.linalg.norm(params - (1, 0), axis=1), 2, atol=1e-12)
        points = radsdf.arc_points((0, 0), 2, 0, 90, segments=4)
        self.assertEqual(points.shape, (5, 2))
        np.testing.assert_allclose(points[[0, -1]], [(2, 0), (0, 2)], atol=1e-12)

    def test_involute_gear_profile(self):
        gear = radsdf.involute_gear(20, 2)
        self.assertEqual(gear.opcode, radsdf.Opcode.POLYGON)
        radii = np.linalg.norm(np.reshape(gear.params, (-1, 2)), axis=1)
        # Pitch radius 20; tips at pitch + module, roots at pitch - 1.25 module.
        self.assertAlmostEqual(radii.max(), 22.0, places=9)
        self.assertAlmostEqual(radii.min(), 17.5, places=9)


class MethodEncodingTests(GraphCase):
    def setUp(self):
        self.s = radsdf.sphere(1)
        self.leaf = N("SPHERE", [1])

    def test_translate_and_moved(self):
        expected = N("TRANSFORM", T(offset=(1, 2, 3)), self.leaf)
        self.assertGraph(self.s.translate(1, 2, 3), expected)
        self.assertGraph(self.s.translate((1, 2, 3)), expected)
        self.assertGraph(self.s.moved([1, 2, 3]), expected)

    def test_rotations(self):
        self.assertGraph(self.s.rotate(90, "x", center=(0, 1, 0)),
                         N("TRANSFORM", T(ROT_X90, (0, 1, -1)), self.leaf))
        self.assertGraph(self.s.rotate_y(90), N("TRANSFORM", T(ROT_Y90), self.leaf))
        self.assertGraph(self.s.rotate_z(90), N("TRANSFORM", T(ROT_Z90), self.leaf))
        self.assertGraph(self.s.rotate_x(-90), N("TRANSFORM", T(np.transpose(ROT_X90)), self.leaf))
        name, params, _ = self.tree(self.s.rotate(30, (1, 2, 3)))
        axis = np.array([1, 2, 3]) / math.sqrt(14)
        self.assertRotation(params, axis, axis)
        _, params, _ = self.tree(self.s.orient((1, 0, 0)))
        self.assertRotation(params, (0, 0, 1), (1, 0, 0))
        _, params, _ = self.tree(self.s.orient((0, 0, 1), up=(0, 1, 0)))
        self.assertRotation(params, (0, 1, 0), (0, 0, 1))

    def test_transform_matrix(self):
        matrix = np.eye(4)
        matrix[:3, :3] = 2 * np.array(ROT_Z90)
        matrix[:3, 3] = (1, 2, 3)
        expected = N("TRANSFORM", T(ROT_Z90, (1, 2, 3), 2), self.leaf)
        self.assertGraph(self.s.transform(matrix), expected)
        self.assertGraph(self.s.transform(matrix[:3]), expected)
        # A slightly inexact rotation is snapped to the nearest rotation.
        sloppy = np.eye(4)
        sloppy[:3, :3] += 1e-8 * np.arange(9).reshape(3, 3)
        _, params, _ = self.tree(self.s.transform(sloppy))
        rotation = np.reshape(params[:9], (3, 3))
        np.testing.assert_allclose(rotation @ rotation.T, np.eye(3), rtol=0, atol=1e-14)
        np.testing.assert_allclose(rotation, np.eye(3), rtol=0, atol=1e-7)

    def test_scale(self):
        self.assertGraph(self.s.scale(2, center=(1, 0, 0)),
                         N("TRANSFORM", T(offset=(-1, 0, 0), scale=2), self.leaf))
        self.assertGraph(self.s.scale((2, 2, 2)), N("TRANSFORM", T(scale=2), self.leaf))
        self.assertGraph(self.s.scale((1, 2, 3)), N("STRETCH", [1, 2, 3], self.leaf))
        self.assertGraph(self.s.scale((1, 2, 3), center=(1, 1, 1)),
                         N("TRANSFORM", T(offset=(1, 1, 1)),
                           N("STRETCH", [1, 2, 3], N("TRANSFORM", T(offset=(-1, -1, -1)), self.leaf))))
        self.assertGraph(radsdf.circle(1).scale((2, 3)), N("STRETCH2", [2, 3], N("CIRCLE", [1])))

    def test_mirrors(self):
        self.assertGraph(self.s.mirror((2, 0, 0), (3, 0, 0)), N("MIRROR", [1, 0, 0, 3], self.leaf))
        self.assertGraph(self.s.mirror("y"), N("MIRROR", [0, 1, 0, 0], self.leaf))
        self.assertGraph(radsdf.circle(1).mirror((0, 3), (0, 2)), N("MIRROR2", [0, 1, 2], N("CIRCLE", [1])))
        records = decode(radsdf.serialize(self.s.mirrored_copy("z", (0, 0, 1)))[0])
        self.assertEqual(records, [("SPHERE", (), (1.0,)), ("MIRROR", (0,), (0.0, 0.0, 1.0, 1.0)),
                                   ("UNION", (0, 1), (0.0,))])

    def test_modifiers(self):
        self.assertGraph(self.s.offset(0.5), N("OFFSET", [0.5], self.leaf))
        self.assertGraph(self.s.shell(1), N("SHELL", [1, 0], self.leaf))
        self.assertGraph(self.s.shell(1, side="outside"), N("SHELL", [1, 1], self.leaf))
        self.assertGraph(self.s.shell(1, side="center"), N("SHELL", [1, 2], self.leaf))
        self.assertGraph(self.s.elongate(1, -2, 0), N("ELONGATE", [1, 2, 0], self.leaf))
        self.assertGraph(self.s.twist(90), N("TWIST", [math.pi / 2], self.leaf))
        circle = N("CIRCLE", [1])
        self.assertGraph(radsdf.circle(1).offset(-0.25), N("OFFSET2", [-0.25], circle))
        self.assertGraph(radsdf.circle(1).outline(0.4), N("OUTLINE2", [0.4], circle))

    def test_cut_intersects_a_half_space(self):
        self.assertGraph(self.s.cut(), N("INTERSECTION", [0], self.leaf, N("HALFSPACE", [0, 0, 1, 0])))
        self.assertGraph(self.s.cut((0, 0, 1), (0, 0, 2), keep="above"),
                         N("INTERSECTION", [0], self.leaf, N("HALFSPACE", [0, 0, -1, -2])))

    def test_arrays_are_unions_of_transformed_copies(self):
        self.assertGraph(self.s.linear_array(3, (5, 0, 0)),
                         N("UNION", [0], self.leaf, N("TRANSFORM", T(offset=(5, 0, 0)), self.leaf),
                           N("TRANSFORM", T(offset=(10, 0, 0)), self.leaf)))
        self.assertGraph(self.s.grid_array((2, 2, 1), (3, 4, 5)),
                         N("UNION", [0], self.leaf, N("TRANSFORM", T(offset=(0, 4, 0)), self.leaf),
                           N("TRANSFORM", T(offset=(3, 0, 0)), self.leaf),
                           N("TRANSFORM", T(offset=(3, 4, 0)), self.leaf)))
        placed = radsdf.sphere(1, center=(5, 0, 0))
        self.assertGraph(placed.polar_array(4),
                         N("UNION", [0], N("TRANSFORM", T(offset=(5, 0, 0)), self.leaf),
                           N("TRANSFORM", T(ROT_Z90, (0, 5, 0)), self.leaf),
                           N("TRANSFORM", T(np.diag([-1, -1, 1]), (-5, 0, 0)), self.leaf),
                           N("TRANSFORM", T(np.transpose(ROT_Z90), (0, -5, 0)), self.leaf)))
        _, _, copies = self.tree(placed.polar_array(3, angle=90))
        offsets = [copy[1][9:12] for copy in copies]
        root = math.sqrt(0.5) * 5
        np.testing.assert_allclose(offsets, [(5, 0, 0), (root, root, 0), (0, 5, 0)], atol=1e-12)
        self.assertGraph(radsdf.circle(1).grid_array((1, 2), (0, 3)),
                         N("UNION2", [0], N("CIRCLE", [1]), N("TRANSFORM2", T(np.eye(2), (0, 3)), N("CIRCLE", [1]))))
        _, _, copies = self.tree(radsdf.circle(1, center=(2, 0)).polar_array(2))
        self.assertEqual([copy[1][4:6] for copy in copies], [(2.0, 0.0), (-2.0, 0.0)])

    def test_booleans(self):
        a, b, c = radsdf.sphere(1), radsdf.box(1), radsdf.cylinder(1, 2)
        A, B, C = N("SPHERE", [1]), N("BOX", [0.5, 0.5, 0.5, 0]), N("CYLINDER", [1, 1])
        self.assertGraph(a | b, N("UNION", [0], A, B))
        self.assertGraph(a & b, N("INTERSECTION", [0], A, B))
        self.assertGraph(a - b, N("DIFFERENCE", [0], A, B))
        self.assertGraph(radsdf.union([a, (b,)], c, smooth=2), N("UNION", [2], A, B, C))
        self.assertGraph(a.intersection(b, smooth=0.5), N("INTERSECTION", [0.5], A, B))
        self.assertGraph(a.difference(b, c, smooth=1), N("DIFFERENCE", [1], A, B, C))
        self.assertGraph(radsdf.union(shape for shape in (a, b)), N("UNION", [0], A, B))
        self.assertIs(radsdf.union(a), a)
        self.assertIs(radsdf.difference(a), a)
        disc, square = radsdf.circle(1), radsdf.rect(1)
        D, S = N("CIRCLE", [1]), N("RECT", [0.5, 0.5, 0])
        self.assertGraph(disc | square, N("UNION2", [0], D, S))
        self.assertGraph(disc & square, N("INTERSECTION2", [0], D, S))
        self.assertGraph(disc - square, N("DIFFERENCE2", [0], D, S))

    def test_unshared_exact_boolean_chains_are_written_as_one_record(self):
        a, b, c, d = (radsdf.sphere(radius) for radius in (1, 2, 3, 4))
        A, B, C, D = (N("SPHERE", [radius]) for radius in (1, 2, 3, 4))
        body = a
        for cutter in (b, c, d):
            body = body - cutter
        self.assertEqual(len(body.children), 2)  # nodes stay as built
        self.assertGraph(body, N("DIFFERENCE", [0], A, B, C, D))
        self.assertGraph((a | b) | c, N("UNION", [0], A, B, C))
        self.assertGraph(a | (b | c), N("UNION", [0], A, N("UNION", [0], B, C)))
        self.assertGraph((a & b) & c, N("INTERSECTION", [0], A, B, C))
        self.assertGraph((a | b) - c, N("DIFFERENCE", [0], N("UNION", [0], A, B), C))
        self.assertGraph((radsdf.circle(1) - radsdf.circle(2)) - radsdf.circle(3),
                         N("DIFFERENCE2", [0], N("CIRCLE", [1]), N("CIRCLE", [2]), N("CIRCLE", [3])))
        # Smooth folds keep the operands' nesting.
        smooth = radsdf.union(radsdf.union(a, b, smooth=1), c, smooth=1)
        self.assertGraph(smooth, N("UNION", [1], N("UNION", [1], A, B), C))
        self.assertGraph(radsdf.union(a | b, c, smooth=1), N("UNION", [1], N("UNION", [0], A, B), C))

    def test_shared_intermediate_results_end_a_chain(self):
        a, b, c = (radsdf.sphere(radius) for radius in (1, 2, 3))
        A, B, C = (N("SPHERE", [radius]) for radius in (1, 2, 3))
        first = a - b
        self.assertGraph(radsdf.union(first - c, first.translate(5, 0, 0)),
                         N("UNION", [0], N("DIFFERENCE", [0], N("DIFFERENCE", [0], A, B), C),
                           N("TRANSFORM", T(offset=(5, 0, 0)), N("DIFFERENCE", [0], A, B))))
        # Keeping every intermediate of a loop keeps the encoding linear.
        stages, body = [], radsdf.box(10)
        for x in range(1, 201):
            body = body - radsdf.sphere(1, center=(x, 0, 0))
            stages.append(body)
        records = decode(radsdf.serialize(radsdf.union(stages))[0])
        self.assertEqual(len(records), 1 + 200 * 3 + 1)
        self.assertTrue(all(len(children) == 2 for name, children, _ in records if name == "DIFFERENCE"))

    def test_chains_are_folded_within_the_child_limit(self):
        shapes = [radsdf.sphere(radius) for radius in range(1, 6)]
        leaves = [N("SPHERE", [radius]) for radius in range(1, 6)]
        chain = shapes[0]
        for shape in shapes[1:]:
            chain = chain | shape
        with mock.patch.object(radsdf, "_MAX_CHILDREN", 3):
            self.assertGraph(chain, N("UNION", [0], N("UNION", [0], *leaves[:3]), *leaves[3:]))

    def test_operand_lists_beyond_the_child_limit_nest_as_left_folds(self):
        shapes = [radsdf.sphere(radius) for radius in range(1, 8)]
        leaves = [N("SPHERE", [radius]) for radius in range(1, 8)]
        with mock.patch.object(radsdf, "_MAX_CHILDREN", 3):
            self.assertGraph(radsdf.union(shapes, smooth=1),
                             N("UNION", [1], N("UNION", [1], N("UNION", [1], *leaves[:3]), *leaves[3:5]),
                               *leaves[5:]))
            self.assertGraph(radsdf.difference(shapes[0], shapes[1:4]),
                             N("DIFFERENCE", [0], N("DIFFERENCE", [0], *leaves[:3]), leaves[3]))
            self.assertGraph(radsdf.union(shapes[:3]) | shapes[3],
                             N("UNION", [0], N("UNION", [0], *leaves[:3]), leaves[3]))

    def test_profiles_to_solids(self):
        disc, D = radsdf.circle(1), N("CIRCLE", [1])
        self.assertGraph(disc.extrude(5), N("EXTRUDE", [0, 5], D))
        self.assertGraph(disc.extrude(5, center=True), N("EXTRUDE", [-2.5, 2.5], D))
        self.assertGraph(disc.extrude(5, taper=-0.5), N("TAPERED_EXTRUDE", [0, 5, -0.5], D))
        self.assertGraph(disc.extrude_between(-1, 2), N("EXTRUDE", [-1, 2], D))
        self.assertGraph(disc.revolve(3), N("REVOLVE", [3], D))
        top = radsdf.rect(2)
        self.assertGraph(disc.loft_to(top, 4), N("LOFT", [0, 4], D, N("RECT", [1, 1, 0])))
        self.assertGraph(disc.loft_to(top, 4, center=True), N("LOFT", [-2, 2], D, N("RECT", [1, 1, 0])))

    def test_profile_transforms(self):
        disc, D = radsdf.circle(1), N("CIRCLE", [1])
        self.assertGraph(disc.rotate(90, center=(1, 0)), N("TRANSFORM2", T([[0, -1], [1, 0]], (1, -1)), D))
        self.assertGraph(disc.translate(2, 3).scale(2),
                         N("TRANSFORM2", T(np.eye(2), (4, 6), 2), D))


class TransformCompositionTests(GraphCase):
    @staticmethod
    def homogeneous(params):
        matrix = np.eye(4)
        matrix[:3, :3] = params[12] * np.reshape(params[:9], (3, 3))
        matrix[:3, 3] = params[9:12]
        return matrix

    def test_consecutive_transforms_compose_into_one_node(self):
        rng = np.random.default_rng(7)
        leaf = radsdf.box((1, 2, 3))
        for _ in range(20):
            shape, expected = leaf, np.eye(4)
            for _ in range(6):
                step = int(rng.integers(3))
                if step == 0:
                    offset = rng.uniform(-5, 5, 3)
                    shape = shape.translate(offset)
                    single = self.homogeneous(T(offset=offset))
                elif step == 1:
                    angle, axis, center = rng.uniform(-180, 180), rng.normal(size=3), rng.uniform(-2, 2, 3)
                    shape = shape.rotate(angle, axis, center)
                    rotation = radsdf.rotation_matrix(angle, axis)
                    single = self.homogeneous(T(rotation, center - rotation @ center))
                else:
                    factor = rng.uniform(0.5, 2)
                    shape = shape.scale(factor)
                    single = self.homogeneous(T(scale=factor))
                expected = single @ expected
            name, params, children = self.tree(shape)
            self.assertEqual((name, children), ("TRANSFORM", (("BOX", (0.5, 1.0, 1.5, 0.0), ()),)))
            np.testing.assert_allclose(self.homogeneous(params), expected, atol=1e-9)
            rotation = np.reshape(params[:9], (3, 3))
            np.testing.assert_allclose(rotation @ rotation.T, np.eye(3), rtol=0, atol=1e-12)

    def test_identity_transforms_return_the_shape(self):
        s = radsdf.sphere(1)
        self.assertIs(s.translate(0, 0, 0), s)
        self.assertIs(s.rotate(0, "x", center=(4, 5, 6)), s)
        self.assertIs(s.scale(1), s)
        self.assertIs(s.translate(1, 0, 0).translate(-1, 0, 0), s)
        self.assertIs(s.rotate_z(90).rotate_z(-90), s)
        disc = radsdf.circle(1)
        self.assertIs(disc.rotate(180).rotate(180), disc)
        self.assertIs(disc.scale((2, 2)).scale(0.5), disc)

    def test_composition_leaves_the_inner_transform_unchanged(self):
        inner = radsdf.sphere(1).translate(1, 0, 0)
        outer = inner.translate(0, 2, 0)
        self.assertEqual(inner.params[9:12], (1.0, 0.0, 0.0))
        self.assertEqual(outer.params[9:12], (1.0, 2.0, 0.0))
        self.assertIs(outer.children[0], inner.children[0])

    def test_transforms_do_not_compose_across_other_nodes(self):
        shape = radsdf.sphere(1).translate(1, 0, 0).offset(1).translate(0, 1, 0)
        self.assertGraph(shape, N("TRANSFORM", T(offset=(0, 1, 0)),
                                  N("OFFSET", [1], N("TRANSFORM", T(offset=(1, 0, 0)), N("SPHERE", [1])))))


class GraphEncodingTests(GraphCase):
    def test_record_layout(self):
        data, roots = radsdf.serialize(radsdf.sphere(2))
        self.assertEqual(roots, [0])
        self.assertEqual(data, struct.pack("<4sIII", b"RSDF", 1, 1, 0) + struct.pack("<4Id", 1, 0, 1, 0, 2.0))
        data, roots = radsdf.serialize(radsdf.sphere(1) | radsdf.box(2))
        self.assertEqual(roots, [2])
        self.assertEqual(data[16:], struct.pack("<4Id", 1, 0, 1, 0, 1.0)
                         + struct.pack("<4I4d", 2, 0, 4, 0, 1.0, 1.0, 1.0, 0.0)
                         + struct.pack("<4I2Id", 30, 2, 1, 0, 0, 1, 0.0))

    def test_shared_nodes_are_written_once(self):
        part = radsdf.sphere(1) - radsdf.box(1)
        assembly = radsdf.union(part, part.translate(5, 0, 0), part.mirror())
        records = decode(radsdf.serialize(assembly)[0])
        self.assertEqual([record[0] for record in records],
                         ["SPHERE", "BOX", "DIFFERENCE", "TRANSFORM", "MIRROR", "UNION"])
        self.assertEqual(records[5][1], (2, 3, 4))
        self.assertEqual(records[3][1], (2,))
        self.assertEqual(records[4][1], (2,))
        self.assertEqual(decode(radsdf.serialize(radsdf.union(part, part))[0])[-1][1], (2, 2))

    def test_multiple_roots_share_one_graph(self):
        base = radsdf.box(4)
        holed = base - radsdf.cylinder(1, 10)
        data, roots = radsdf.serialize([holed, base, holed])
        records = decode(data)
        self.assertEqual(len(records), 3)
        self.assertEqual(roots, [2, 0, 2])

    def test_encoding_is_deterministic_and_handles_deep_graphs(self):
        shape = radsdf.sphere(1)
        for _ in range(5000):
            shape = shape.offset(0.001)
        data = radsdf.serialize(shape)[0]
        self.assertEqual(len(decode(data)), 5001)
        self.assertEqual(data, radsdf.serialize(shape)[0])

    def test_limits(self):
        with mock.patch.object(radsdf, "_MAX_NODES", 2), self.assertRaisesRegex(SDFError, "at most 2"):
            radsdf.serialize(radsdf.sphere(1) | radsdf.box(1))
        with mock.patch.object(radsdf, "_MAX_BYTES", 40), self.assertRaisesRegex(SDFError, "size limit"):
            radsdf.serialize(radsdf.sphere(1) | radsdf.box(1))
        with mock.patch.object(radsdf, "_MAX_PARAMS", 6), self.assertRaisesRegex(SDFError, "at most 3 points"):
            radsdf.polygon([(0, 0), (1, 0), (1, 1), (0, 1)])
        with self.assertRaisesRegex(SDFError, "TRANSFORM parameters must be finite"):
            radsdf.sphere(1).scale(1e200).scale(1e200)

    def test_serialize_rejects_non_shapes(self):
        for value in ([], [radsdf.sphere(1), 3], 3, "sphere"):
            with self.assertRaises(SDFError):
                radsdf.serialize(value)


class ShapeObjectTests(unittest.TestCase):
    def test_shapes_are_immutable_values(self):
        s = radsdf.sphere(1)
        with self.assertRaises(AttributeError):
            s.radius = 2
        with self.assertRaises(AttributeError):
            del s._params
        self.assertIs(copy.deepcopy(s), s)
        self.assertEqual((s.opcode, s.params, s.children), (radsdf.Opcode.SPHERE, (1.0,), ()))
        self.assertEqual(repr(s), "<SDF3 SPHERE>")
        self.assertEqual(repr(radsdf.circle(1)), "<SDF2 CIRCLE>")
        self.assertEqual((s.dimension, radsdf.circle(1).dimension), (3, 2))

    def test_opcodes_match_the_specification(self):
        self.assertEqual({member.value: member.name for member in radsdf.Opcode}, OPCODES)

    def test_star_import_exports_only_the_modeling_api(self):
        namespace = {}
        exec("from radsdf import *", namespace)
        for name in ("sphere", "box", "union", "SDF3", "SDFError", "Mesh", "merge_meshes", "auto_cell_size"):
            self.assertIn(name, namespace)
        for name in ("np", "math", "ctypes", "Opcode", "_native"):
            self.assertNotIn(name, namespace)


class ValidationTests(unittest.TestCase):
    CASES = [
        (lambda: radsdf.sphere(0), "sphere radius must be greater than zero"),
        (lambda: radsdf.sphere("big"), "sphere radius must be a number"),
        (lambda: radsdf.sphere(math.inf), "sphere radius must be finite"),
        (lambda: radsdf.sphere(1, center=(0, 0)), "center must contain 3 finite numbers"),
        (lambda: radsdf.sphere(1, center=(0, 0, math.nan)), "center must contain 3 finite numbers"),
        (lambda: radsdf.box((1, 0, 1)), "box size components must be greater than zero"),
        (lambda: radsdf.box(2, radius=1.5), "box radius must not exceed half"),
        (lambda: radsdf.box(2, radius=-1), "box radius must not be negative"),
        (lambda: radsdf.cylinder(1, 2, axis="w"), "axis must be 'x', 'y', 'z' or a 3D vector"),
        (lambda: radsdf.cylinder(1, 2, axis=(0, 0, 0)), "axis must not be zero"),
        (lambda: radsdf.cone(0, 0, 1), "cone needs a nonzero radius"),
        (lambda: radsdf.torus(1, -1), "torus minor radius must be greater than zero"),
        (lambda: radsdf.ellipsoid((1, 2)), "ellipsoid radii must contain 3 finite numbers"),
        (lambda: radsdf.half_space((0, 0, 0)), "half_space normal must not be zero"),
        (lambda: radsdf.rect((1, 2), radius=0.6), "rect radius must not exceed half the smaller side"),
        (lambda: radsdf.polygon([(0, 0), (1, 1)]), "polygon needs at least three finite"),
        (lambda: radsdf.polygon([(0, 0), (1, 1), (2, 2)]), "polygon has no area"),
        (lambda: radsdf.polygon([(0, 0), (0, 0), (1, 1), (1, 1)]), "at least three distinct points"),
        (lambda: radsdf.polygon("square"), "polygon needs at least three finite"),
        (lambda: radsdf.regular_polygon(2, 1), "regular_polygon needs between 3 and"),
        (lambda: radsdf.half_plane((0, 0)), "half_plane normal must not be zero"),
        (lambda: radsdf.involute_gear(5, 1), "involute_gear needs between 6 and 400 teeth"),
        (lambda: radsdf.involute_gear(20, 1, points_per_flank=1), "points_per_flank must be between"),
        (lambda: radsdf.union(), "union needs at least one shape"),
        (lambda: radsdf.intersection([]), "intersection needs at least one shape"),
        (lambda: radsdf.union(radsdf.sphere(1), 5), "Expected a shape, got int"),
        (lambda: radsdf.union(5), "Expected a shape, got int"),
        (lambda: radsdf.difference("box", radsdf.sphere(1)), "Expected a shape, got str"),
        (lambda: radsdf.sphere(1) | radsdf.circle(1), "Cannot combine 2D profiles with 3D solids"),
        (lambda: radsdf.union(radsdf.sphere(1), radsdf.box(1), smooth=-1), "smooth must not be negative"),
        (lambda: radsdf.sphere(1).translate(1, 2), "translate offset takes a 3D vector or three numbers"),
        (lambda: radsdf.circle(1).translate(1, 2, 3), "translate offset takes a 2D vector or two numbers"),
        (lambda: radsdf.sphere(1).rotate("ninety"), "angle must be a number"),
        (lambda: radsdf.sphere(1).scale(0), "scale factor must be greater than zero"),
        (lambda: radsdf.sphere(1).scale((1, -1, 1)), "scale factors components must be greater than zero"),
        (lambda: radsdf.sphere(1).transform(np.eye(3)), "transform needs a finite 4x4 or 3x4 matrix"),
        (lambda: radsdf.sphere(1).transform(np.diag([1, 1, -1, 1])), "use mirror\\(\\) for reflections"),
        (lambda: radsdf.sphere(1).transform(np.diag([1, 2, 1, 1])), "rotation with uniform positive scale"),
        (lambda: radsdf.sphere(1).transform(np.zeros((4, 4)) + np.eye(4) * [0, 0, 0, 1]), "singular"),
        (lambda: radsdf.sphere(1).transform([[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 1, 0], [0, 0, 1, 1]]),
         "last row is 0, 0, 0, 1"),
        (lambda: radsdf.sphere(1).shell(1, side="middle"), "shell side must be 'inside', 'outside' or 'center'"),
        (lambda: radsdf.sphere(1).shell(0), "shell thickness must be greater than zero"),
        (lambda: radsdf.sphere(1).cut(keep="left"), "cut keep must be 'below' or 'above'"),
        (lambda: radsdf.sphere(1).linear_array(0, (1, 0, 0)), "linear_array count must be between 1 and 10000"),
        (lambda: radsdf.sphere(1).linear_array(2.5, (1, 0, 0)), "linear_array count must be a whole number"),
        (lambda: radsdf.sphere(1).grid_array((100, 100, 2), (1, 1, 1)), "grid_array needs three counts"),
        (lambda: radsdf.sphere(1).grid_array(4, (1, 1, 1)), "grid_array needs three counts"),
        (lambda: radsdf.circle(1).grid_array((2, 2, 2), (1, 1)), "grid_array needs two counts"),
        (lambda: radsdf.sphere(1).polar_array(10_001), "polar_array count must be between 1 and 10000"),
        (lambda: radsdf.circle(1).extrude(0), "extrude height must be greater than zero"),
        (lambda: radsdf.circle(1).extrude_between(2, 1), "extrude_between needs z1 > z0"),
        (lambda: radsdf.circle(1).loft_to(radsdf.sphere(1), 2), "loft_to needs another 2D profile"),
        (lambda: radsdf.circle(1).mirror((0, 0)), "mirror normal must not be zero"),
        (lambda: radsdf.circle(1).outline(-1), "outline thickness must be greater than zero"),
        (lambda: radsdf.arc_points((0, 0), 1, 0, 90, segments=0), "arc segments must be between 1 and"),
        (lambda: radsdf.gradient(radsdf.circle(1), (0, 0)), "gradient needs a 3D solid"),
        (lambda: radsdf.raycast(radsdf.circle(1), (0, 0, 0), (1, 0, 0)), "raycast needs a 3D solid"),
    ]

    def test_invalid_arguments_raise_sdf_errors(self):
        for build, message in self.CASES:
            with self.subTest(message=message), self.assertRaisesRegex(SDFError, message):
                build()

    def test_whole_numbers_accept_integral_floats(self):
        self.assertEqual(len(radsdf.sphere(1).linear_array(3.0, (1, 0, 0)).children), 3)
        self.assertEqual(len(radsdf.sphere(1).polar_array(np.int64(4)).children), 4)


def tetrahedron():
    """A closed unit right tetrahedron with outward counter-clockwise triangles."""
    positions = [(0, 0, 0), (1, 0, 0), (0, 1, 0), (0, 0, 1)]
    triangles = [(0, 2, 1), (0, 1, 3), (0, 3, 2), (1, 2, 3)]
    normals = np.array(positions, dtype=float) - 0.25
    normals /= np.linalg.norm(normals, axis=1, keepdims=True)
    return radsdf.Mesh(positions, normals, triangles, [[(0, 0, 0), (1, 0, 0)]])


class MeshHelperTests(unittest.TestCase):
    def test_measurements(self):
        mesh = tetrahedron()
        self.assertEqual(mesh.positions.dtype, np.float32)
        self.assertEqual(mesh.triangles.dtype, np.uint32)
        self.assertEqual(mesh.creases.shape, (1, 2, 3))
        self.assertAlmostEqual(mesh.stats["volume"], 1 / 6)
        self.assertAlmostEqual(mesh.stats["area"], 1.5 + math.sqrt(3) / 2)
        np.testing.assert_allclose(mesh.stats["centroid"], (0.25, 0.25, 0.25))
        self.assertEqual((mesh.stats["bounds_lo"], mesh.stats["bounds_hi"]), ([0, 0, 0], [1, 1, 1]))
        self.assertEqual((mesh.stats["vertex_count"], mesh.stats["triangle_count"],
                          mesh.stats["crease_segment_count"]), (4, 4, 1))
        far = tetrahedron().transformed(np.array([[1, 0, 0, 1e4], [0, 1, 0, 0], [0, 0, 1, 0]]))
        self.assertAlmostEqual(far.stats["volume"], 1 / 6, places=9)

    def test_save_stl(self):
        mesh = tetrahedron()
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / "part.stl"
            mesh.save_stl(path)
            data = path.read_bytes()
        self.assertEqual(len(data), 84 + 50 * 4)
        self.assertTrue(data.startswith(b"RAD"))
        self.assertEqual(struct.unpack_from("<I", data, 80)[0], 4)
        records = np.frombuffer(data[84:], dtype=[("normal", "<f4", (3,)), ("vertices", "<f4", (3, 3)),
                                                  ("attribute", "<u2")])
        np.testing.assert_allclose(records["vertices"], mesh.positions[mesh.triangles])
        np.testing.assert_allclose(np.linalg.norm(records["normal"], axis=1), 1, atol=1e-6)
        outward = records["vertices"].mean(axis=1) - mesh.stats["centroid"]
        self.assertTrue(np.all(np.einsum("ij,ij->i", records["normal"], outward) > 0))
        self.assertTrue(np.all(records["attribute"] == 0))

    def test_transformed(self):
        mesh = tetrahedron()
        moved = mesh.transformed([[0, -1, 0, 5], [1, 0, 0, 0], [0, 0, 1, 0], [0, 0, 0, 1]])
        np.testing.assert_allclose(moved.positions[1], (5, 1, 0))
        np.testing.assert_allclose(moved.normals, mesh.normals @ np.array(ROT_Z90).T, atol=1e-6)
        np.testing.assert_allclose(moved.creases[0], [(5, 0, 0), (5, 1, 0)])
        self.assertAlmostEqual(moved.stats["volume"], 1 / 6)
        np.testing.assert_allclose(moved.stats["centroid"], (4.75, 0.25, 0.25), atol=1e-6)
        mirrored = mesh.transformed(np.diag([-1, 1, 1, 1]))
        self.assertAlmostEqual(mirrored.stats["volume"], 1 / 6)
        np.testing.assert_array_equal(mirrored.triangles, mesh.triangles[:, ::-1])
        stretched = mesh.transformed(np.diag([2, 1, 1, 1]))
        self.assertAlmostEqual(stretched.stats["volume"], 2 / 6)
        np.testing.assert_allclose(np.linalg.norm(stretched.normals, axis=1), 1, atol=1e-6)
        expected = mesh.normals[3] * (0.5, 1, 1)
        np.testing.assert_allclose(stretched.normals[3], expected / np.linalg.norm(expected), atol=1e-6)
        self.assertEqual(mesh.stats["volume"], tetrahedron().stats["volume"])
        with self.assertRaisesRegex(SDFError, "nonsingular"):
            mesh.transformed(np.diag([1, 0, 1, 1]))
        with self.assertRaisesRegex(SDFError, "last row"):
            mesh.transformed(np.ones((4, 4)))

    def test_merge_meshes(self):
        first = radsdf.Mesh(tetrahedron().positions, tetrahedron().normals, tetrahedron().triangles,
                            stats={"cell_size": 0.5, "components": 1, "seconds": 1.0})
        second = tetrahedron().transformed(np.diag([2, 2, 2, 1]))
        second.stats.update(cell_size=1.0, components=1)
        merged = radsdf.merge_meshes([first, second])
        self.assertEqual(merged.positions.shape, (8, 3))
        np.testing.assert_array_equal(merged.triangles[4:], second.triangles + 4)
        self.assertAlmostEqual(merged.stats["volume"], 1 / 6 + 8 / 6)
        np.testing.assert_allclose(merged.stats["centroid"], np.full(3, (0.25 / 6 + 0.5 * 8 / 6) / (9 / 6)))
        self.assertEqual((merged.stats["cell_size"], merged.stats["components"]), (1.0, 2))
        self.assertNotIn("seconds", merged.stats)
        self.assertEqual(merged.creases.shape, (1, 2, 3))
        with self.assertRaisesRegex(SDFError, "at least one mesh"):
            radsdf.merge_meshes([])
        with self.assertRaisesRegex(SDFError, "Mesh objects"):
            radsdf.merge_meshes([first, "mesh"])

    def test_mesh_validation(self):
        positions, normals = np.zeros((3, 3)), np.zeros((3, 3))
        with self.assertRaisesRegex(SDFError, "positions must have shape"):
            radsdf.Mesh(np.zeros((3, 2)), normals, [(0, 1, 2)])
        with self.assertRaisesRegex(SDFError, "one row per position"):
            radsdf.Mesh(positions, np.zeros((2, 3)), [(0, 1, 2)])
        with self.assertRaisesRegex(SDFError, "index existing positions"):
            radsdf.Mesh(positions, normals, [(0, 1, 3)])
        with self.assertRaisesRegex(SDFError, "nonnegative integers"):
            radsdf.Mesh(positions, normals, [(0, -1, 2)])
        empty = radsdf.Mesh([], [], [])
        self.assertEqual((empty.positions.shape, empty.triangles.shape, empty.stats["volume"]),
                         ((0, 3), (0, 3), 0.0))

    def test_auto_cell_size(self):
        self.assertAlmostEqual(radsdf.auto_cell_size(radsdf.Bounds((0, 0, 0), (100, 20, 5))), 0.5)
        self.assertAlmostEqual(radsdf.auto_cell_size(((0, 0, 0), (100, 20, 5)), cells_across=50), 2.0)
        self.assertAlmostEqual(radsdf.auto_cell_size(((0, 0, 0), (1, 1, 1))), 0.02)
        self.assertAlmostEqual(radsdf.auto_cell_size(((0, 0, 0), (1, 1, 1)), minimum=0), 0.005)
        with self.assertRaisesRegex(SDFError, "finite bounds"):
            radsdf.auto_cell_size(radsdf.Bounds.infinite())
        with self.assertRaisesRegex(SDFError, "nonzero extent"):
            radsdf.auto_cell_size(((0, 0, 0), (0, 0, 0)), minimum=0)

    def test_bounds(self):
        bounds = radsdf.Bounds((0, -1, 2), (4, 1, 3))
        np.testing.assert_array_equal(bounds.size, (4, 2, 1))
        np.testing.assert_array_equal(bounds.center, (2, 0, 2.5))
        self.assertTrue(bounds.finite)
        self.assertFalse(radsdf.Bounds.infinite(2).finite)
        self.assertEqual(len(bounds.corners()), 8)
        self.assertEqual(bounds.union(radsdf.Bounds((5, 0, 0), (6, 0, 0))).as_list(), [[0, -1, 0], [6, 1, 3]])
        self.assertTrue(bounds.intersection(radsdf.Bounds((5, 0, 0), (6, 0, 0))).empty)
        self.assertEqual(bounds.expanded(1).as_list(), [[-1, -2, 1], [5, 2, 4]])


class LibraryTests(unittest.TestCase):
    def test_every_header_function_is_declared(self):
        header = (ENGINE.parent / "Sources" / "RADSDFKernel" / "include" / "radsdf.h").read_text()
        self.assertEqual(set(re.findall(r"\b(rsdf_\w+)\s*\(", header)), set(radsdf._PROTOTYPES))

    def test_option_and_stats_layouts_match_the_header(self):
        options, stats = radsdf._MeshOptions, radsdf._MeshStats
        self.assertEqual(ctypes.sizeof(options), 88)
        self.assertEqual((options.max_cells.offset, options.bounds_lo.offset, options.bounds_valid.offset),
                         (16, 32, 80))
        self.assertEqual(ctypes.sizeof(stats), 160)
        self.assertEqual((stats.volume.offset, stats.centroid.offset, stats.cell_size.offset), (56, 72, 144))

    def test_library_search_order(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder).resolve()
            module = root / "Engine" / "radsdf.py"
            build = root / ".build"
            # In increasing order of preference.
            candidates = [build / "out" / "Products" / "Debug" / "libRADSDF.dylib",
                          build / "debug" / "libRADSDF.dylib",
                          build / "out" / "Products" / "Release" / "libRADSDF.dylib",
                          build / "release" / "libRADSDF.dylib",
                          root / "Engine" / "libRADSDF.dylib"]
            environment = {key: value for key, value in os.environ.items() if key != "RAD_SDF_LIBRARY"}
            with mock.patch.object(radsdf, "__file__", str(module)), \
                    mock.patch.dict(os.environ, environment, clear=True):
                with self.assertRaisesRegex(SDFError, "RAD_SDF_LIBRARY"):
                    radsdf._library_path()
                for candidate in candidates:
                    candidate.parent.mkdir(parents=True, exist_ok=True)
                    candidate.touch()
                    self.assertEqual(radsdf._library_path(), candidate)
                with mock.patch.dict(os.environ, {"RAD_SDF_LIBRARY": "/opt/kernel.dylib"}):
                    self.assertEqual(radsdf._library_path(), Path("/opt/kernel.dylib"))

    def test_module_works_without_the_kernel(self):
        script = ("import sys\nsys.path.insert(0, sys.argv[1])\nimport radsdf\n"
                  "data, roots = radsdf.serialize(radsdf.sphere(1) - radsdf.box(1))\n"
                  "print(len(data), roots)\n"
                  "for query in (lambda: radsdf.sphere(1).evaluate((0, 0, 0)), radsdf.kernel_version):\n"
                  "    try:\n        query()\n"
                  "    except radsdf.SDFError as error:\n        print('SDFError', error)\n")
        with tempfile.TemporaryDirectory() as folder:
            missing = Path(folder) / "libRADSDF.dylib"
            result = subprocess.run([sys.executable, "-I", "-c", script, str(ENGINE)], capture_output=True,
                                    text=True, check=True, env=dict(os.environ, RAD_SDF_LIBRARY=str(missing)))
        lines = result.stdout.splitlines()
        data, roots = radsdf.serialize(radsdf.sphere(1) - radsdf.box(1))
        self.assertEqual(lines[0], f"{len(data)} {roots}")
        self.assertEqual(len(lines), 3)
        for line in lines[1:]:
            self.assertTrue(line.startswith("SDFError The native SDF kernel"), line)
            self.assertIn(str(missing), line)


def kernel_library_exists():
    """Native tests skip when the library is missing; one that exists but
    cannot be loaded fails them."""
    try:
        return radsdf._library_path().is_file()
    except SDFError:
        return False


def box_distance(points, half):
    q = np.abs(points) - half
    return np.linalg.norm(np.maximum(q, 0), axis=-1) + np.minimum(q.max(axis=-1), 0)


def capped_cylinder(radial, axial, radius, half_height):
    dr, dz = radial - radius, np.abs(axial) - half_height
    return np.hypot(np.maximum(dr, 0), np.maximum(dz, 0)) + np.minimum(np.maximum(dr, dz), 0)


def capped_cone(points, r1, r2, h):
    """Exact distance to a capped cone along Z, radius r1 at z=-h and r2 at z=+h."""
    qx, qy = np.hypot(points[:, 0], points[:, 1]), points[:, 2]
    k2 = np.array([r2 - r1, 2 * h])
    cax = qx - np.minimum(qx, np.where(qy < 0, r1, r2))
    cay = np.abs(qy) - h
    t = np.clip(((r2 - qx) * k2[0] + (h - qy) * k2[1]) / (k2 @ k2), 0, 1)
    cbx, cby = qx - r2 + k2[0] * t, qy - h + k2[1] * t
    sign = np.where((cbx < 0) & (cay < 0), -1.0, 1.0)
    return sign * np.sqrt(np.minimum(cax ** 2 + cay ** 2, cbx ** 2 + cby ** 2))


def segment_distance(points, a, b):
    a, b = np.asarray(a, float), np.asarray(b, float)
    t = np.clip((points - a) @ (b - a) / ((b - a) @ (b - a)), 0, 1)
    return np.linalg.norm(points - a - t[:, None] * (b - a), axis=1)


@unittest.skipUnless(kernel_library_exists(), "the native SDF kernel library is missing")
class NativeKernelTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        radsdf.kernel_version()  # raises SDFError when the library cannot be used

    def setUp(self):
        self.points = np.random.default_rng(11).uniform(-6, 6, (2000, 3))

    def assertField(self, shape, expected, points=None, atol=1e-9):
        points = self.points if points is None else points
        np.testing.assert_allclose(shape.evaluate(points), expected(points), rtol=0, atol=atol)

    def test_primitives_match_analytic_distances(self):
        p = self.points
        self.assertField(radsdf.sphere(2, center=(1, 2, 3)), lambda p: np.linalg.norm(p - (1, 2, 3), axis=1) - 2)
        self.assertField(radsdf.box((2, 4, 6), center=(1, 0, 0)), lambda p: box_distance(p - (1, 0, 0), (1, 2, 3)))
        self.assertField(radsdf.box(4, radius=1), lambda p: box_distance(p, 1) - 1)
        self.assertField(radsdf.cylinder(1, 4, center=(0, 1, 0), axis="x"),
                         lambda p: capped_cylinder(np.hypot(p[:, 1] - 1, p[:, 2]), p[:, 0], 1, 2))
        self.assertField(radsdf.cone(2, 0.5, 4), lambda p: capped_cone(p, 2, 0.5, 2))
        self.assertField(radsdf.capsule((0, 0, -2), (1, 1, 2), 0.5),
                         lambda p: segment_distance(p, (0, 0, -2), (1, 1, 2)) - 0.5)
        self.assertField(radsdf.torus(4, 1, axis="y"),
                         lambda p: np.hypot(np.hypot(p[:, 0], p[:, 2]) - 4, p[:, 1]) - 1)
        self.assertField(radsdf.ellipsoid((1, 2, 3)),
                         lambda p: (np.linalg.norm(p / (1, 2, 3), axis=1) - 1) * 1)
        self.assertField(radsdf.half_space((0, 0, 1), (0, 0, 2)), lambda p: p[:, 2] - 2)
        self.assertAlmostEqual(radsdf.sphere(1)(p[0]), np.linalg.norm(p[0]) - 1, places=12)

    def test_booleans_and_modifiers(self):
        a, b = radsdf.sphere(2), radsdf.box(3, center=(1, 0, 0))

        def da(p):
            return np.linalg.norm(p, axis=1) - 2

        def db(p):
            return box_distance(p - (1, 0, 0), 1.5)
        self.assertField(a | b, lambda p: np.minimum(da(p), db(p)))
        self.assertField(a & b, lambda p: np.maximum(da(p), db(p)))
        self.assertField(a - b, lambda p: np.maximum(da(p), -db(p)))

        def smooth_min(x, y, k):
            h = np.maximum(k - np.abs(x - y), 0) / k
            return np.minimum(x, y) - h * h * k / 4
        self.assertField(radsdf.union(a, b, smooth=1), lambda p: smooth_min(da(p), db(p), 1))
        self.assertField(radsdf.difference(a, b, smooth=1), lambda p: -smooth_min(-da(p), db(p), 1))
        self.assertField(a.offset(1), lambda p: da(p) - 1)
        self.assertField(a.shell(0.5), lambda p: np.maximum(da(p), -da(p) - 0.5))
        self.assertField(a.shell(0.5, side="outside"), lambda p: np.maximum(-da(p), da(p) - 0.5))
        self.assertField(a.shell(0.5, side="center"), lambda p: np.abs(da(p)) - 0.25)
        self.assertField(a.scale(2, center=(1, 0, 0)), lambda p: np.linalg.norm(p - (-1, 0, 0), axis=1) - 4)
        self.assertField(radsdf.box(2).scale((1, 2, 3)), lambda p: box_distance(p / (1, 2, 3), 1))
        self.assertField(radsdf.sphere(1, center=(3, 0, 0)).mirror(),
                         lambda p: np.linalg.norm(p + (3, 0, 0), axis=1) - 1)
        self.assertField(a.elongate(2, 0, 0),
                         lambda p: np.linalg.norm(p - np.clip(p, (-2, 0, 0), (2, 0, 0)), axis=1) - 2)

    def test_profiles_and_solids_from_profiles(self):
        flat = self.points[:, :2]
        self.assertField(radsdf.circle(1, center=(1, 1)), lambda p: np.linalg.norm(p - 1, axis=1) - 1, flat)
        self.assertField(radsdf.rect((4, 2), radius=0.5), lambda p: box_distance(p, (1.5, 0.5)) - 0.5, flat)
        self.assertField(radsdf.slot((0, 0), (3, 1), 0.5), lambda p: segment_distance(p, (0, 0), (3, 1)) - 0.5, flat)
        self.assertField(radsdf.half_plane((0, 1), (0, 2)), lambda p: p[:, 1] - 2, flat)
        square = [(0, 0), (4, 0), (4, 3), (0, 3)]
        for points in (square, square[::-1]):
            self.assertField(radsdf.polygon(points), lambda p: box_distance(p - (2, 1.5), (2, 1.5)), flat)
        notch = radsdf.polygon([(0, 0), (4, 0), (4, 4), (2, 2), (0, 4)])
        np.testing.assert_allclose(notch.evaluate([(2, 1), (2, 3), (1, 2), (-1, 0)]),
                                   [-1, math.sqrt(2) / 2, -math.sqrt(2) / 2, 1], atol=1e-12)
        disc = radsdf.circle(1)
        self.assertField(radsdf.rect(2).extrude(4),
                         lambda p: (np.minimum(np.maximum(box_distance(p[:, :2], 1), np.abs(p[:, 2] - 2) - 2), 0)
                                    + np.hypot(np.maximum(box_distance(p[:, :2], 1), 0),
                                               np.maximum(np.abs(p[:, 2] - 2) - 2, 0))))
        self.assertField(radsdf.circle(1, center=(4, 0)).revolve(), radsdf.torus(4, 1).evaluate)
        taper = disc.extrude(2, taper=-0.5)
        self.assertAlmostEqual(taper.evaluate((0, 0, 1)), (-1 + 0.25) / math.sqrt(1 + 0.0625), places=12)
        loft = disc.loft_to(radsdf.circle(2), 4)
        self.assertLess(loft.evaluate((0, 0, 2)), 0)
        self.assertGreater(loft.evaluate((2.5, 0, 3.9)), 0)
        twisted = radsdf.box((2, 2, 10)).twist(9)
        for z in (-4.0, 0.0, 3.0):
            angle = math.radians(9 * z)
            corner = np.array([math.cos(angle), math.sin(angle)])
            self.assertLess(twisted.evaluate((*(0.9 * corner), z)), 0)
            self.assertGreater(twisted.evaluate((*(1.1 * corner), z)), 0)

    def test_point_shapes_and_validation(self):
        shape = radsdf.sphere(1)
        self.assertIsInstance(shape.evaluate((0, 0, 0)), float)
        self.assertEqual(shape.evaluate(np.zeros((4, 5, 3))).shape, (4, 5))
        self.assertEqual(shape.evaluate(np.zeros((0, 3))).shape, (0,))
        np.testing.assert_array_equal(shape.contains([(0, 0, 0), (2, 0, 0)]), [True, False])
        self.assertEqual(radsdf.circle(1).evaluate([(0, 0), (3, 0)]).tolist(), [-1, 2])
        with self.assertRaisesRegex(SDFError, r"points must have shape \(\.\.\., 2\)"):
            radsdf.circle(1).evaluate((0, 0, 0))
        with self.assertRaisesRegex(SDFError, "points must be finite"):
            shape.evaluate((0, math.nan, 0))

    def test_results_do_not_depend_on_the_thread_count(self):
        shape = radsdf.union(radsdf.box(4) - radsdf.sphere(2.5), radsdf.torus(5, 0.5), smooth=0.5)
        points = np.random.default_rng(3).uniform(-7, 7, (50_000, 3))
        np.testing.assert_array_equal(shape.evaluate(points, threads=1), shape.evaluate(points, threads=0))
        np.testing.assert_array_equal(shape.evaluate(points, threads=3), shape.evaluate(points))

    def test_gradient(self):
        shape = radsdf.sphere(2, center=(1, 0, 0))
        directions = self.points - (1, 0, 0)
        lengths = np.linalg.norm(directions, axis=1, keepdims=True)
        away = lengths[:, 0] > 0.5  # the field has no gradient at the centre
        np.testing.assert_allclose(shape.gradient(self.points[away]), (directions / lengths)[away], atol=1e-4)
        np.testing.assert_allclose(radsdf.gradient(shape, (1, 0, 5), eps=1e-6), (0, 0, 1), atol=1e-6)
        self.assertEqual(shape.gradient(np.zeros((2, 2, 3))).shape, (2, 2, 3))

    def test_bounds_and_lipschitz(self):
        bounds = radsdf.sphere(2, center=(1, 2, 3)).bounds
        self.assertTrue(np.all(bounds.lo <= (-1, 0, 1)) and np.all(bounds.hi >= (3, 4, 5)))
        np.testing.assert_allclose(bounds.as_list(), [[-1, 0, 1], [3, 4, 5]], atol=0.05)
        disc = radsdf.circle(1, center=(2, 0)).bounds
        self.assertEqual(disc.lo.shape, (2,))
        np.testing.assert_allclose(disc.as_list(), [[1, -1], [3, 1]], atol=0.05)
        plane = radsdf.half_space()
        self.assertFalse(plane.bounds.finite)
        self.assertTrue(radsdf.box(2).cut().bounds.finite)
        self.assertAlmostEqual(radsdf.sphere(1).lipschitz, 1.0, places=9)
        self.assertGreaterEqual(radsdf.box((2, 2, 10)).twist(30).lipschitz, 1.0 - 1e-12)

    def test_raycast(self):
        shape = radsdf.sphere(1)
        hit, distance = shape.raycast((-5, 0, 0), (2, 0, 0))
        self.assertIs(hit, True)
        self.assertAlmostEqual(distance, 4, places=3)
        hits, distances = radsdf.raycast(shape, [(-5, 0, 0), (-5, 3, 0), (0, 0, -5)], (0, 0, 1))
        np.testing.assert_array_equal(hits, [False, False, True])
        self.assertAlmostEqual(distances[2], 4, places=3)
        hits, _ = shape.raycast(np.zeros((2, 3, 3)) - (5, 0, 0), (1, 0, 0), max_distance=3)
        self.assertEqual(hits.shape, (2, 3))
        self.assertFalse(hits.any())
        hit, distance = shape.raycast((0, 0, 5), (0, 0, -1), 10, 1e-6, 64)
        self.assertIs(hit, True)
        self.assertAlmostEqual(distance, 4, places=5)
        with self.assertRaisesRegex(SDFError, "max_steps must be a whole number"):
            radsdf.raycast(shape, (0, 0, 5), (0, 0, -1), 10, 64, 1e-6)
        with self.assertRaisesRegex(SDFError, "ray directions must not be zero"):
            shape.raycast((0, 0, 0), (0, 0, 0))

    def test_sphere_mesh_volume_and_topology(self):
        radius = 10.0
        mesh = radsdf.sphere(radius, center=(1, 2, 3)).mesh()
        stats = mesh.stats
        self.assertAlmostEqual(stats["volume"] / (4 / 3 * math.pi * radius ** 3), 1, delta=0.01)
        self.assertAlmostEqual(stats["area"] / (4 * math.pi * radius ** 2), 1, delta=0.02)
        np.testing.assert_allclose(stats["centroid"], (1, 2, 3), atol=0.05)
        self.assertEqual((stats["boundary_edges"], stats["nonmanifold_edges"], stats["components"]), (0, 0, 1))
        self.assertEqual((stats["vertex_count"], stats["triangle_count"]), (len(mesh.positions), len(mesh.triangles)))
        self.assertAlmostEqual(stats["cell_size"], 0.1, delta=0.001)
        self.assertEqual((mesh.positions.dtype, mesh.normals.dtype, mesh.triangles.dtype, mesh.creases.dtype),
                         (np.float32, np.float32, np.uint32, np.float32))
        self.assertEqual(mesh.creases.shape[1:], (2, 3))
        np.testing.assert_allclose(np.linalg.norm(mesh.normals, axis=1), 1, atol=1e-5)
        np.testing.assert_allclose(np.linalg.norm(mesh.positions - (1, 2, 3), axis=1), radius, atol=0.01)
        own = radsdf.Mesh(mesh.positions, mesh.normals, mesh.triangles).stats
        self.assertAlmostEqual(own["volume"], stats["volume"], delta=1e-4 * stats["volume"])

    def test_box_mesh_keeps_sharp_edges(self):
        mesh = radsdf.box((4, 6, 8)).mesh(cell_size=0.3)
        self.assertAlmostEqual(mesh.stats["volume"] / 192, 1, delta=0.01)
        np.testing.assert_allclose(mesh.stats["bounds_lo"], (-2, -3, -4), atol=0.02)
        self.assertGreater(len(mesh.creases), 0)

    def test_mesh_options_and_errors(self):
        shape = radsdf.sphere(5)
        bounded = shape.mesh(cell_size=0.25, bounds=((-6, -6, -6), (6, 6, 6)), crease_angle=20, threads=2)
        self.assertAlmostEqual(bounded.stats["volume"] / (4 / 3 * math.pi * 125), 1, delta=0.01)
        with self.assertRaisesRegex(SDFError, "Meshing failed: "):
            shape.mesh(cell_size=0.01, max_cells=1000)
        with self.assertRaisesRegex(SDFError, "finite bounds"):
            radsdf.half_space().mesh()
        with self.assertRaisesRegex(SDFError, "crease_angle must be between 0 and 180"):
            shape.mesh(crease_angle=200)
        with self.assertRaisesRegex(SDFError, "hi > lo"):
            shape.mesh(bounds=((0, 0, 0), (0, 1, 1)))
        disc = radsdf.circle(1)
        with self.assertRaisesRegex(SDFError, "mesh needs a 3D solid"):
            radsdf.Graph(disc).mesh(disc)

    def test_mesh_stl_export(self):
        mesh = radsdf.merge_meshes([radsdf.sphere(3).mesh(cell_size=0.2),
                                    radsdf.box(2, center=(10, 0, 0)).mesh(cell_size=0.2)])
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / "design.stl"
            mesh.save_stl(path)
            data = path.read_bytes()
        self.assertEqual(len(data), 84 + 50 * len(mesh.triangles))
        records = np.frombuffer(data[84:], dtype=[("normal", "<f4", (3,)), ("vertices", "<f4", (3, 3)),
                                                  ("attribute", "<u2")])
        np.testing.assert_array_equal(records["vertices"], mesh.positions[mesh.triangles])
        # Facet normals follow the winding; degenerate triangles get zero normals.
        corners = records["vertices"].astype(np.float64)
        cross = np.cross(corners[:, 1] - corners[:, 0], corners[:, 2] - corners[:, 0])
        areas = np.linalg.norm(cross, axis=1)
        solid = areas > 0
        np.testing.assert_allclose(records["normal"][solid], cross[solid] / areas[solid, None], atol=1e-5)
        np.testing.assert_array_equal(records["normal"][~solid], 0)
        sphere = solid & (np.linalg.norm(corners.mean(axis=1), axis=1) < 5)
        outward = np.einsum("ij,ij->i", records["normal"][sphere], corners[sphere].mean(axis=1)) > 0
        self.assertGreater(areas[sphere][outward].sum() / areas[sphere].sum(), 0.999)
        self.assertEqual(mesh.stats["components"], 2)

    def test_graph_handles(self):
        part, other = radsdf.sphere(1), radsdf.box(2, center=(5, 0, 0))
        with radsdf.Graph([part, other]) as graph:
            self.assertEqual(graph.node_count, len(decode(radsdf.serialize([part, other])[0])))
            self.assertEqual(graph.evaluate(other, (5, 0, 0)), -1)
            self.assertEqual(graph.evaluate(graph.roots[0], (0, 0, 0)), -1)
            np.testing.assert_allclose(graph.bounds(other).as_list(), [[4, -1, -1], [6, 1, 1]], atol=0.05)
            with self.assertRaisesRegex(SDFError, "not one of this graph's roots"):
                graph.evaluate(radsdf.sphere(1), (0, 0, 0))
        self.assertTrue(graph.closed)
        with self.assertRaisesRegex(SDFError, "closed"):
            graph.evaluate(part, (0, 0, 0))
        graph.close()
        self.assertIs(radsdf._graph_for(part), radsdf._graph_for(part))

    def test_graphs_load_from_rsdf_bytes(self):
        part, other = radsdf.sphere(1), radsdf.box(2, center=(5, 0, 0))
        data, roots = radsdf.serialize([part, other])
        for buffer in (data, bytearray(data), memoryview(data), np.frombuffer(data, dtype=np.uint8)):
            with self.subTest(buffer=type(buffer).__name__), radsdf.Graph.from_bytes(buffer, roots) as graph:
                self.assertEqual(graph.roots, tuple(roots))
                self.assertEqual(graph.node_count, len(decode(data)))
                self.assertEqual(graph.evaluate(graph.roots[1], (5, 0, 0)), -1)
                self.assertEqual(graph.bounds(roots[0]).as_list(), [[-1, -1, -1], [1, 1, 1]])
                mesh = graph.mesh(roots[0], cell_size=0.1)
                self.assertAlmostEqual(mesh.stats["volume"] / (4 / 3 * math.pi), 1, delta=0.01)
                with self.assertRaisesRegex(SDFError, "not one of this graph's roots"):
                    graph.evaluate(part, (0, 0, 0))
            self.assertTrue(graph.closed)
        with radsdf.Graph.from_bytes(data, np.array(roots[::-1])) as graph:
            self.assertEqual(graph.roots, tuple(roots[::-1]))
            self.assertIs(type(graph.roots[0]), int)
        self.assertEqual(radsdf.Graph.from_bytes(data, []).roots, ())
        last = len(decode(data)) - 1
        errors = [((data[:-8], roots), "Loading the RSDF graph failed: .+"),
                  ((b"RSDX" + data[4:], roots), "Loading the RSDF graph failed: .+"),
                  ((data, [roots[0], last + 1]), rf"root node index {last + 1} is not a node of the graph "
                                                 rf"\(0 to {last}\)"),
                  ((data, [-1]), "root node index -1 is not a node"),
                  ((data, [0.5]), "root node index must be a whole number"),
                  ((data, 3), "roots must be a list of node indices"),
                  ((data, "01"), "roots must be a list of node indices"),
                  (("RSDF", roots), "RSDF data must be bytes, got str")]
        for arguments, message in errors:
            with self.subTest(message=message), self.assertRaisesRegex(SDFError, message):
                radsdf.Graph.from_bytes(*arguments)

    def test_smooth_unions_of_several_shapes_are_not_clipped(self):
        # Three coincident unit spheres blended with smooth=1 form a sphere of
        # radius 1.390625, beyond the k/4 that a blend of two shapes reaches.
        ball = radsdf.sphere(1)
        blend = radsdf.union(ball, ball, ball, smooth=1)
        radius = 1.390625
        self.assertEqual(blend.bounds.as_list(), [[-radius] * 3, [radius] * 3])
        self.assertLess(blend.evaluate((radius - 1e-9, 0, 0)), 0)
        mesh = blend.mesh(cell_size=0.02)
        self.assertAlmostEqual(mesh.stats["volume"] / (4 / 3 * math.pi * radius ** 3), 1, delta=0.005)
        np.testing.assert_allclose(mesh.stats["bounds_hi"], [radius] * 3, atol=0.005)

    def test_kernel_errors_carry_the_kernel_message(self):
        malformed = radsdf.SDF3(radsdf.Opcode.SPHERE, (1.0, 2.0))
        with self.assertRaisesRegex(SDFError, "Building the shape graph failed: .+"):
            malformed.evaluate((0, 0, 0))

    def test_close_waits_for_queries_running_on_other_threads(self):
        # ctypes releases the GIL during kernel calls: close() on another
        # thread must not free the graph under a running query or mesh.
        star = radsdf.polygon([(math.cos(a) * (10 + (i % 2)), math.sin(a) * (10 + (i % 2)))
                               for i, a in enumerate(np.linspace(0, 2 * math.pi, 4000, endpoint=False))])
        solid = star.extrude(4)
        points = np.random.default_rng(5).uniform(-12, 12, (200_000, 3))
        for query in ("evaluate", "mesh"):
            graph = radsdf.Graph(solid)
            results, errors = [], []

            def run():
                try:
                    if query == "evaluate":
                        results.append(graph.evaluate(solid, points, threads=1))
                    else:
                        results.append(graph.mesh(solid, cell_size=0.25, threads=1))
                except Exception as error:  # noqa: BLE001 - reported below
                    errors.append(error)

            worker = threading.Thread(target=run)
            worker.start()
            deadline = time.monotonic() + 10
            while graph._calls == 0 and worker.is_alive() and time.monotonic() < deadline:
                time.sleep(0.001)
            graph.close()
            self.assertTrue(graph.closed)
            with self.assertRaisesRegex(SDFError, "closed"):
                graph.bounds(solid)
            worker.join()
            with self.subTest(query=query):
                self.assertEqual(errors, [])
                self.assertEqual(len(results), 1)
                self.assertFalse(graph._release.alive)  # freed by the query when it returned

    @unittest.skipUnless(hasattr(os, "fork"), "needs fork()")
    def test_processes_forked_after_kernel_use_run_single_threaded(self):
        shape = radsdf.box(10) - radsdf.sphere(6)
        points = np.random.default_rng(2).uniform(-6, 6, (100_000, 3))
        expected = shape.evaluate(points, threads=0)
        with warnings.catch_warnings():  # the kernel's pool threads make fork() warn
            warnings.simplefilter("ignore", DeprecationWarning)
            pid = os.fork()
        if pid == 0:  # child: libdispatch's pool from the parent is unusable here
            status = 1
            try:
                if radsdf._threads(0) == 1 and np.array_equal(shape.evaluate(points, threads=0), expected):
                    status = 0
            finally:
                os._exit(status)
        _, status = os.waitpid(pid, 0)
        self.assertTrue(os.WIFEXITED(status), f"forked child died: status {status}")
        self.assertEqual(os.WEXITSTATUS(status), 0)
        self.assertEqual(radsdf._threads(0), 0)

    def test_crease_angle_zero_creases_every_edge(self):
        ball = radsdf.sphere(1)
        none = ball.mesh(cell_size=0.1).stats["crease_segment_count"]
        every = ball.mesh(cell_size=0.1, crease_angle=0).stats["crease_segment_count"]
        self.assertEqual(none, 0)
        self.assertEqual(every, ball.mesh(cell_size=0.1, crease_angle=1e-9).stats["crease_segment_count"])
        self.assertGreater(every, 1000)

    def test_tiny_shapes_mesh_with_the_default_cell_size(self):
        self.assertAlmostEqual(radsdf.auto_cell_size(((0, 0, 0), (0.01, 0.01, 0.01))), 0.01 / 32)
        for radius in (0.005, 0.01):
            mesh = radsdf.sphere(radius).mesh()
            self.assertAlmostEqual(mesh.stats["volume"] / (4 / 3 * math.pi * radius ** 3), 1, delta=0.02)

    def test_ray_directions_of_any_finite_size_and_boolean_indices(self):
        ball = radsdf.sphere(1)
        for z in (-1e200, -1e-160, -5e-324):
            hit, distance = ball.raycast((0, 0, 10), (0, 0, z))
            self.assertTrue(hit)
            self.assertAlmostEqual(distance, 9, places=4)
        with radsdf.Graph(ball) as graph:
            with self.assertRaisesRegex(SDFError, "node index must be a whole number, not a boolean"):
                graph.evaluate(True, (0, 0, 0))
            self.assertEqual(graph.evaluate(0, (0, 0, 0)), -1)
        data, _ = radsdf.serialize(ball)
        with self.assertRaisesRegex(SDFError, "not a boolean"):
            radsdf.Graph.from_bytes(data, [np.True_])

    def test_lofts_and_grown_ellipsoids_mesh_whole(self):
        loft = radsdf.circle(10).loft_to(radsdf.circle(5), 20)
        np.testing.assert_allclose(loft.bounds.as_list(), [[-10, -10, 0], [10, 10, 20]])
        frustum = math.pi * 20 / 3 * (100 + 50 + 25)
        self.assertAlmostEqual(loft.mesh().stats["volume"] / frustum, 1, delta=0.005)
        grown = radsdf.ellipsoid((30, 22, 16)).offset(2)
        self.assertLessEqual(grown.bounds.lo[0], -33.75)
        mesh = grown.mesh(cell_size=0.5)
        self.assertAlmostEqual(float(mesh.positions[:, 0].max()), 33.75, delta=0.25)
        flat = np.isclose(np.abs(mesh.positions[:, 0]), grown.bounds.hi[0], atol=1e-4)
        self.assertLess(np.count_nonzero(flat), 3)  # no cap cut by the bounds


if __name__ == "__main__":
    unittest.main()
