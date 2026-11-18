"""Imported triangle meshes: the kernel's MESH node (exact distances, signs and
mesh validation, through the C API with hand-written RSDF), the radsdf
readers and mesh_solid, and builds whose programs use imported STL or OBJ
inputs as solids."""
import json
import math
import os
from pathlib import Path
import shutil
import struct
import subprocess
import tempfile
import time
import unittest

import numpy as np

from test_sdf_kernel_native import Kernel, KernelError, library_path
from sdf_test_support import native, read_stl, signed_volume as stl_volume
import radsdf
import sdf_build
import worker

MESH, TRANSFORM, UNION, DIFFERENCE, SPHERE = 9, 20, 30, 32, 1


# ------------------------------------------------------------------ test meshes

def outward(vertices, triangles):
    return (vertices, triangles) if signed_volume(vertices, triangles) > 0 else (vertices, triangles[:, ::-1])


def cube_mesh(size=2.0, center=(0.0, 0.0, 0.0)):
    h = size / 2
    vertices = np.array([[x, y, z] for x in (-h, h) for y in (-h, h) for z in (-h, h)]) + center
    quads = [(0, 1, 3, 2), (4, 6, 7, 5), (0, 4, 5, 1), (2, 3, 7, 6), (0, 2, 6, 4), (1, 5, 7, 3)]
    return outward(vertices, np.array([tri for a, b, c, d in quads for tri in ((a, b, c), (a, c, d))]))


def sphere_mesh(radius=1.0, rings=16, segments=32):
    vertices = [(0.0, 0.0, radius)]
    for i in range(1, rings):
        theta = math.pi * i / rings
        vertices += [(radius * math.sin(theta) * math.cos(2 * math.pi * j / segments),
                      radius * math.sin(theta) * math.sin(2 * math.pi * j / segments),
                      radius * math.cos(theta)) for j in range(segments)]
    vertices.append((0.0, 0.0, -radius))
    triangles = [(0, 1 + j, 1 + (j + 1) % segments) for j in range(segments)]
    for i in range(rings - 2):
        a, b = 1 + i * segments, 1 + (i + 1) * segments
        for j in range(segments):
            k = (j + 1) % segments
            triangles += [(a + j, b + j, b + k), (a + j, b + k, a + k)]
    last = 1 + (rings - 2) * segments
    triangles += [(last + j, len(vertices) - 1, last + (j + 1) % segments) for j in range(segments)]
    return outward(np.array(vertices), np.array(triangles))


def torus_mesh(major=2.0, minor=0.6, n=32, m=16):
    i, j = np.meshgrid(np.arange(n), np.arange(m), indexing="ij")
    u, w = 2 * np.pi * i / n, 2 * np.pi * j / m
    vertices = np.stack([(major + minor * np.cos(w)) * np.cos(u), (major + minor * np.cos(w)) * np.sin(u),
                         minor * np.sin(w)], axis=-1).reshape(-1, 3)
    a, b = i * m + j, ((i + 1) % n) * m + j
    c, d = ((i + 1) % n) * m + (j + 1) % m, i * m + (j + 1) % m
    triangles = np.concatenate([np.stack([a, b, c], -1).reshape(-1, 3), np.stack([a, c, d], -1).reshape(-1, 3)])
    return outward(vertices, triangles)


def sliver_cube():
    """A closed cube whose edge v0-v1 is split by its midpoint m on one side and
    closed by the zero-area triangle (v1, m, v0)."""
    vertices, triangles = cube_mesh()
    vertices = np.vstack([vertices, (vertices[0] + vertices[1]) / 2])
    rows = [list(row) for row in triangles]
    for index, row in enumerate(rows):
        for k in range(3):
            if (row[k], row[(k + 1) % 3]) == (0, 1):
                apex = row[(k + 2) % 3]
                rows[index:index + 1] = [[0, 8, apex], [8, 1, apex]]
                return vertices, np.array(rows + [[1, 8, 0]])
    raise AssertionError("cube has no edge 0-1")


def combined(*meshes):
    offsets = np.cumsum([0] + [len(vertices) for vertices, _ in meshes[:-1]])
    return (np.vstack([vertices for vertices, _ in meshes]),
            np.vstack([triangles + offset for (_, triangles), offset in zip(meshes, offsets)]))


def other_diagonals(triangles):
    """The quads of cube_mesh (pairs of triangles sharing a diagonal), split
    along their other diagonals."""
    out = []
    for (p, q, r), second in triangles.reshape(-1, 2, 3):
        second = np.roll(second, -list(second).index(p))  # (p, r, s)
        out += [(p, q, second[2]), (q, r, second[2])]
    return np.array(out)


def welded(vertices, triangles):
    """Exact-duplicate vertices merged (bodies that share vertices)."""
    unique, inverse = np.unique(np.round(vertices, 9), axis=0, return_inverse=True)
    return unique, inverse.reshape(-1)[triangles]


def signed_volume(vertices, triangles):
    a, b, c = (vertices[triangles[:, k]] for k in range(3))
    return float(np.einsum("ij,ij->i", a, np.cross(b, c)).sum() / 6)


def mesh_params(vertices, triangles):
    return np.concatenate([[len(vertices), len(triangles)], np.ravel(vertices), np.ravel(triangles)])


def rsdf(*records):
    """RSDF bytes of (opcode, children, params) records."""
    out = [b"RSDF", struct.pack("<III", 1, len(records), 0)]
    for op, children, params in records:
        out += [struct.pack("<IIII", op, len(children), len(params), 0), struct.pack(f"<{len(children)}I", *children),
                np.asarray(params, dtype="<f8").tobytes()]
    return b"".join(out)


def stl_bytes(vertices, triangles, header=b"test mesh"):
    records = np.zeros(len(triangles), dtype=radsdf._STL_RECORD)
    records["vertices"] = vertices[triangles]
    return header.ljust(80, b" ") + struct.pack("<I", len(triangles)) + records.tobytes()


def ascii_stl(vertices, triangles):
    lines = ["solid reference"]
    for corners in vertices[triangles]:
        lines += ["  facet normal 0 0 0", "    outer loop"]
        lines += [f"      vertex {float(x)!r} {float(y)!r} {float(z)!r}" for x, y, z in corners]
        lines += ["    endloop", "  endfacet"]
    return ("\n".join(lines) + "\nendsolid reference\n").encode()


# ------------------------------------------------------------------ brute-force references

def triangle_distances(points, vertices, triangles):
    """Unsigned distance from every point to the nearest triangle."""
    a, b, c = (vertices[triangles[:, k]][None] for k in range(3))
    out = np.empty(len(points))
    for start in range(0, len(points), 128):
        p = points[start:start + 128, None, :]
        normal = np.cross(b - a, c - a)
        nn = np.sum(normal * normal, -1)
        height = np.sum((p - a) * normal, -1) / nn
        projected = p - height[..., None] * normal
        inside = np.ones(height.shape, dtype=bool)
        for x, y in ((a, b), (b, c), (c, a)):
            inside &= np.sum(np.cross(y - x, projected - x) * normal, -1) >= 0
        best = np.where(inside, height * height * nn, np.inf)
        for x, y in ((a, b), (b, c), (c, a)):
            e = y - x
            t = np.clip(np.sum((p - x) * e, -1) / np.sum(e * e, -1), 0, 1)
            best = np.minimum(best, np.sum((p - x - t[..., None] * e) ** 2, -1))
        out[start:start + 128] = np.sqrt(best.min(axis=1))
    return out


def winding_numbers(points, vertices, triangles):
    """Generalized winding numbers (1 inside a closed outward mesh, 0 outside)."""
    a, b, c = (vertices[triangles[:, k]][None] for k in range(3))
    out = np.empty(len(points))
    for start in range(0, len(points), 128):
        p = points[start:start + 128, None, :]
        x, y, z = a - p, b - p, c - p
        lx, ly, lz = (np.linalg.norm(q, axis=-1) for q in (x, y, z))
        det = np.sum(x * np.cross(y, z), -1)
        den = lx * ly * lz + np.sum(x * y, -1) * lz + np.sum(y * z, -1) * lx + np.sum(z * x, -1) * ly
        out[start:start + 128] = np.sum(2 * np.arctan2(det, den), axis=1) / (4 * math.pi)
    return out


def reference_field(points, vertices, triangles):
    distances = triangle_distances(points, vertices, triangles)
    return np.where(winding_numbers(points, vertices, triangles) > 0.5, -distances, distances)


def box_field(points, center, half):
    q = np.abs(points - center) - half
    return np.linalg.norm(np.maximum(q, 0), axis=1) + np.minimum(q.max(axis=1), 0)


# ------------------------------------------------------------------ kernel

@unittest.skipUnless(library_path(), "libRADSDF is not built (swift build --product RADSDF -c release)")
class KernelMeshTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.kernel = Kernel(library_path())

    def load(self, vertices, triangles):
        handle = self.kernel.create(rsdf((MESH, [], mesh_params(vertices, triangles))))
        self.addCleanup(self.kernel.lib.rsdf_graph_free, handle)
        return handle

    def rejection(self, vertices, triangles):
        with self.assertRaises(KernelError) as caught:
            self.load(vertices, triangles)
        return str(caught.exception)

    def test_preview_resolves_close_facets_on_blended_meshes(self):
        def shell(stations, segments, clustering=0):
            uniform = np.linspace(0, 2 * np.pi, segments, endpoint=False)
            angle = uniform + clustering * np.cos(uniform)
            rings = [
                np.column_stack((center + rx * np.cos(angle) + 0.33 * flare * np.cos(3 * angle),
                                 (ry - 0.72 * flare * np.sin(3 * angle)) * np.sin(angle),
                                 np.full(segments, z)))
                for z, center, rx, ry, flare in stations
            ]
            last = (len(rings) - 1) * segments
            triangles = []
            for start in range(0, last, segments):
                for i in range(segments):
                    a, b = start + i, start + (i + 1) % segments
                    triangles.extend(((a, b, b + segments), (a, b + segments, a + segments)))
            for i in range(1, segments - 1):
                triangles.extend(((0, i + 1, i), (last, last + i, last + i + 1)))
            return radsdf.mesh_solid(np.vstack(rings), triangles)

        flange = shell(((0, 0, 5, 7, 1), (0.75, 0, 5.25, 7.8, 1), (2.3, 0.2, 5, 6.5, 0.96),
                        (4.3, 0.45, 4.6, 5.13, 0.48), (6.6, 0.6, 3.55, 3.6, 0.09),
                        (8.1, 0.65, 3.32, 3.27, 0)), 48, clustering=-0.7)
        tube = shell(((3, 0.8, 2.9, 2.6, 0), (5.6, 0.68, 3.26, 3.13, 0),
                      (8, 0.55, 3.4, 3.5, 0), (13, 0.4, 3.54, 3.62, 0)), 25)
        data, (root,) = radsdf.serialize(radsdf.union(flange, tube, smooth=1.35))
        graph = radsdf.Graph.from_bytes(data, [root])
        cell = 0.47
        mesh = graph.native_mesh(root, cell_size=cell, threads=4, _preview=True, _facet_detail=True).mesh()

        self.assertEqual(mesh.stats["boundary_edges"], 0)
        self.assertEqual(mesh.stats["nonmanifold_edges"], 0)
        self.assertEqual(mesh.stats["components"], 1)
        points = mesh.positions[mesh.triangles].astype(float)
        normals = mesh.normals[mesh.triangles].astype(float)
        areas = np.linalg.norm(np.cross(points[:, 1] - points[:, 0], points[:, 2] - points[:, 0]), axis=1)
        barycentric = [[1 / 3] * 3, [0.6, 0.2, 0.2], [0.2, 0.6, 0.2], [0.2, 0.2, 0.6]]
        samples = np.einsum("ij,tjk->tik", barycentric, points).reshape(-1, 3)
        lighting = np.einsum("ij,tjk->tik", barycentric, normals).reshape(-1, 3)
        lighting /= np.linalg.norm(lighting, axis=1)[:, None]
        # Compare what the preview interpolates with the exact visible surface.
        ray = np.array([0, 1.0, 0])
        origins = samples - 0.25 * ray
        hits, distance = graph.raycast(root, origins, ray, max_distance=0.65, tolerance=cell * 1e-6, threads=4)
        exact = graph.gradient(root, origins + distance[:, None] * ray, eps=cell * 1e-4, threads=4)
        front = (hits & (samples[:, 1] < 0) & (samples[:, 2] > 3) & (samples[:, 2] < 8)
                 & (exact[:, 1] < -0.4) & (abs(distance - 0.25) < 0.1))
        error = np.einsum("ij,ij->i", lighting, exact) < math.cos(math.radians(4))
        weights = np.repeat(areas, len(barycentric))
        self.assertGreater(weights[front].sum(), 200)
        self.assertLess(weights[front & error].sum() / weights[front].sum(), 0.045)


    def test_distances_match_brute_force_and_inward_meshes_turn_outward(self):
        rng = np.random.default_rng(1)
        for name, (vertices, triangles) in {"cube": cube_mesh(), "sphere": sphere_mesh(),
                                            "torus": torus_mesh()}.items():
            with self.subTest(name):
                handle = self.load(vertices, triangles)
                lo, hi, lipschitz = self.kernel.bounds(handle, 0)
                np.testing.assert_array_equal(lo, vertices.min(axis=0))
                np.testing.assert_array_equal(hi, vertices.max(axis=0))
                self.assertEqual(lipschitz, 1.0)
                points = rng.uniform(lo - 1, hi + 1, size=(1500, 3))
                expected = reference_field(points, vertices, triangles)
                self.assertTrue((expected < 0).any() and (expected > 0).any())
                np.testing.assert_allclose(self.kernel.evaluate(handle, 0, points), expected, rtol=0, atol=1e-12)
                inward = self.load(vertices, triangles[:, ::-1])
                np.testing.assert_allclose(self.kernel.evaluate(inward, 0, points), expected, rtol=0, atol=1e-12)

    def test_signs_hold_beside_edges_and_vertices(self):
        vertices, triangles = cube_mesh()
        handle = self.load(vertices, triangles)
        points, expected = [], []
        for distance in (1e-12, 1e-9, 1e-6, 1e-3, 0.4):
            for corner in vertices:  # along the corner's diagonal, out and in
                direction = corner / np.linalg.norm(corner)
                points += [corner + distance * direction, corner - distance * direction]
                expected += [distance, -distance / math.sqrt(3)]
            for x in (-1, 1):  # beside the vertical edges, out and in
                for y in (-1, 1):
                    for z in (-0.7, 0.0, 0.3):
                        direction = np.array([x, y, 0]) / math.sqrt(2)
                        edge = np.array([x, y, z])
                        points += [edge + distance * direction, edge - distance * direction]
                        expected += [distance, -distance / math.sqrt(2)]
        values = self.kernel.evaluate(handle, 0, np.array(points))
        np.testing.assert_array_equal(np.sign(values), np.sign(expected))
        np.testing.assert_allclose(values, expected, rtol=1e-9, atol=1e-15)
        rng = np.random.default_rng(2)
        for vertices, triangles in (sphere_mesh(1.0, 24, 48), torus_mesh(2.0, 0.6, 48, 24)):
            handle = self.load(vertices, triangles)
            a, b = vertices[triangles[:, 0]], vertices[triangles[:, 1]]
            features = np.vstack([vertices, (a + b) / 2])  # vertices and edge midpoints
            directions = rng.normal(size=features.shape)
            directions /= np.linalg.norm(directions, axis=1)[:, None]
            for scale in (1e-7, 1e-4, 1e-2):
                points = features + scale * directions
                values = self.kernel.evaluate(handle, 0, points)
                inside = winding_numbers(points, vertices, triangles) > 0.5
                np.testing.assert_array_equal(values < 0, inside)
                np.testing.assert_allclose(np.abs(values), triangle_distances(points, vertices, triangles),
                                           rtol=0, atol=1e-12)

    def test_invalid_meshes_are_rejected_with_their_problem(self):
        vertices, triangles = cube_mesh()
        other = lambda center: cube_mesh(center=center)  # noqa: E731
        cases = [
            ((vertices, triangles[1:]), "3 edges belong to only one triangle"),
            ((vertices, np.vstack([triangles[:1, ::-1], triangles[1:]])), "join triangles that face opposite ways"),
            (welded(*combined((vertices, triangles), other((0, 0, 0)))),
             "18 edges are shared by more than two triangles that do not pair up"),
            (welded(*combined((vertices, triangles), (other((2, 2, 0))[0], other((2, 2, 0))[1][:, ::-1]))),
             "1 edge is shared by more than two triangles that do not pair up"),
            ((vertices, np.vstack([triangles, [[0, 0, 1]]])), "degenerate"),
            (sliver_cube(), "degenerate"),
            ((np.vstack([vertices, [[3, 3, 3]]]), triangles), "vertex 8 is not used"),
            ((np.array([[0, 0, 0], [1, 0, 0], [0, 1, 0], [1, 1, 0.0]]),
              np.array([[0, 2, 1], [0, 1, 3], [0, 3, 2], [1, 2, 3]])), "encloses no volume"),
            ((vertices, triangles[:3]), "at least four triangles"),
            ((vertices, np.vstack([triangles[:-1], [[0, 1, 9]]])), "refers to vertex 9"),
            ((vertices * 1e13, triangles), "beyond 1e12"),
        ]
        for (case_vertices, case_triangles), problem in cases:
            with self.subTest(problem):
                message = self.rejection(case_vertices, case_triangles)
                self.assertIn("node 0 (MESH)", message)
                self.assertIn(problem, message)
        for params, problem in [([8.5, 12], "vertex count must be a whole number"),
                                ([3, 2_000_001], "triangle count must be a whole number of at most 2000000"),
                                (mesh_params(vertices, triangles)[:-1], "need 62")]:
            with self.subTest(problem), self.assertRaisesRegex(KernelError, problem):
                self.kernel.create(rsdf((MESH, [], params)))

    def test_bodies_unite_and_inner_shells_are_cavities(self):
        rng = np.random.default_rng(3)
        reverse = lambda mesh: (mesh[0], mesh[1][:, ::-1])  # noqa: E731
        points = rng.uniform(-3.5, 3.5, size=(2500, 3))
        # Overlapping bodies: their union, with exact distances outside.
        vertices, triangles = combined(cube_mesh(2), cube_mesh(2, center=(1.0, 0.5, 0.25)))
        expected = np.minimum(box_field(points, (0, 0, 0), 1), box_field(points, (1.0, 0.5, 0.25), 1))
        np.testing.assert_allclose(self.kernel.evaluate(self.load(vertices, triangles), 0, points), expected,
                                   rtol=0, atol=1e-12)
        # A separate body facing inward is turned outward.
        vertices, triangles = combined(cube_mesh(2), reverse(cube_mesh(1, center=(2.5, 0, 0))))
        expected = np.minimum(box_field(points, (0, 0, 0), 1), box_field(points, (2.5, 0, 0), 0.5))
        np.testing.assert_allclose(self.kernel.evaluate(self.load(vertices, triangles), 0, points), expected,
                                   rtol=0, atol=1e-12)
        # A hollow ball with an island in its cavity, given right side out or inside out.
        vertices, triangles = combined(sphere_mesh(3.0, 16, 32), reverse(sphere_mesh(2.0, 16, 32)),
                                       sphere_mesh(1.0, 16, 32))
        expected = reference_field(points, vertices, triangles)
        for case in (triangles, triangles[:, ::-1]):
            np.testing.assert_allclose(self.kernel.evaluate(self.load(vertices, case), 0, points), expected,
                                       rtol=0, atol=1e-12)

    def test_shells_facing_inward_are_cavities_only_inside_a_body_solid(self):
        rng = np.random.default_rng(5)
        reverse = lambda mesh: (mesh[0], mesh[1][:, ::-1])  # noqa: E731
        points = rng.uniform(-4, 4, size=(3000, 3))
        field = lambda lo, hi: box_field(points, (np.array(lo) + hi) / 2, (np.array(hi) - lo) / 2)  # noqa: E731

        def box(lo, hi):
            vertices, triangles = cube_mesh(1.0)
            return vertices * (np.array(hi) - lo) + (np.array(hi) + lo) / 2, triangles
        # An inward shell overlapping a body is an inverted body: their union.
        vertices, triangles = combined(box((0, 0, 0), (1, 1, 1)), reverse(box((0.5, 0.5, 0.5), (1.2, 1.2, 1.2))))
        handle = self.load(vertices, triangles)
        np.testing.assert_allclose(self.kernel.evaluate(handle, 0, points),
                                   np.minimum(field((0, 0, 0), (1, 1, 1)), field((0.5, 0.5, 0.5), (1.2, 1.2, 1.2))),
                                   rtol=0, atol=1e-12)
        self.assertAlmostEqual(self.kernel.mesh(handle, 0, 0.02).stats.volume, 1.218, delta=1e-9)
        # A cavity belongs to the body whose solid holds it, not to a smaller
        # body that pokes into it.
        vertices, triangles = combined(box((-3, -3, -3), (3, 3, 3)), reverse(box((-2, -2, -2), (2, 2, 2))),
                                       box((-2.5, -2.5, -2.5), (0.5, 2.5, 2.5)))
        handle = self.load(vertices, triangles)
        hollow = np.maximum(field((-3, -3, -3), (3, 3, 3)), -field((-2, -2, -2), (2, 2, 2)))
        expected = np.minimum(hollow, field((-2.5, -2.5, -2.5), (0.5, 2.5, 2.5)))
        values = self.kernel.evaluate(handle, 0, points)
        np.testing.assert_array_equal(values < 0, expected < 0)
        outside = expected > 0
        np.testing.assert_allclose(values[outside], expected[outside], rtol=0, atol=1e-12)
        self.assertAlmostEqual(self.kernel.evaluate(handle, 0, np.array([[1.5, 0, 0]]))[0], 0.5, delta=1e-12)
        self.assertAlmostEqual(self.kernel.mesh(handle, 0, 0.05).stats.volume, 192.0, delta=1e-6)
        # An inward shell in a cavity is an island (even-odd nesting), as are
        # the deeper levels of an onion of alternately facing boxes.
        for facing in ((1, -1, -1), (1, -1, 1, -1, 1, -1)):
            sizes = [3.0 - 0.5 * k for k in range(len(facing))]
            vertices, triangles = combined(*[box((-size,) * 3, (size,) * 3) if way > 0 else
                                             reverse(box((-size,) * 3, (size,) * 3))
                                             for size, way in zip(sizes, facing)])
            expected = np.full(len(points), np.inf)
            for k in range(0, len(sizes), 2):  # solid layers: shell k minus shell k + 1
                layer = field((-sizes[k],) * 3, (sizes[k],) * 3)
                if k + 1 < len(sizes):
                    layer = np.maximum(layer, -field((-sizes[k + 1],) * 3, (sizes[k + 1],) * 3))
                expected = np.minimum(expected, layer)
            values = self.kernel.evaluate(self.load(vertices, triangles), 0, points)
            np.testing.assert_array_equal(values < 0, expected < 0)
            np.testing.assert_allclose(values, expected, rtol=0, atol=1e-12)
        # A cavity that touches the shell around it is not a cavity: it is
        # turned outward. Two cavities that cross each other are an error.
        vertices, triangles = combined(box((0, 0, 0), (2, 2, 2)), reverse(box((0, 0.5, 0.5), (1, 1.5, 1.5))))
        self.assertLess(self.kernel.evaluate(self.load(vertices, triangles), 0, np.array([[0.5, 1, 1]]))[0], 0)
        message = self.rejection(*combined(box((0, 0, 0), (2, 2, 2)), reverse(box((0.5, 0.5, 0.5), (1.2, 1.2, 1.2))),
                                           reverse(box((1, 1, 1), (1.5, 1.5, 1.5)))))
        self.assertIn("two shells facing inward (cavities) inside one body intersect", message)

    def test_values_outside_the_vertex_box_are_positive(self):
        rng = np.random.default_rng(6)
        directions = rng.normal(size=(4000, 3))
        directions /= np.linalg.norm(directions, axis=1)[:, None]
        reverse = lambda mesh: (mesh[0], mesh[1][:, ::-1])  # noqa: E731
        meshes = {"cube": cube_mesh(), "sphere": sphere_mesh(), "inverted overlap": combined(
            cube_mesh(2), reverse(cube_mesh(1.4, center=(0.8, 0.8, 0.8))))}
        for name, (vertices, triangles) in meshes.items():
            handle = self.load(vertices, triangles)
            for scale in (1e3, 1e15, 1e16, 1e17, 1e40, 1e100, 1e149):
                with self.subTest(name, scale=scale):
                    points = directions * scale
                    values = self.kernel.evaluate(handle, 0, points)
                    self.assertTrue((values > 0).all())
                    self.assertLessEqual(np.max(np.abs(values - np.linalg.norm(points, axis=1)) / scale), 3e-3)

    def test_touching_bodies_that_share_vertices_keep_their_own_surfaces(self):
        rng = np.random.default_rng(7)
        # Random points, and a grid like the mesher's (one irrational offset on
        # every axis), whose rows run exactly through shared face diagonals.
        phi = 2 - (1 + math.sqrt(5)) / 2
        axis = -3 + 0.1 * (np.arange(80) + phi)
        grid = np.stack(np.meshgrid(axis, axis, axis, indexing="ij"), -1).reshape(-1, 3)
        points = np.vstack([rng.uniform(-3, 5, size=(3000, 3)), grid])
        cube = lambda center: cube_mesh(2, center=center)  # noqa: E731
        cases = {"face": [(0, 0, 0), (0, 0, 2)], "edge": [(0, 0, 0), (2, 2, 0)], "vertex": [(0, 0, 0), (2, 2, 2)],
                 "four around an edge": [(0, 0, 0), (2, 0, 0), (0, 2, 0), (2, 2, 0)],
                 "block of eight": [(x, y, z) for x in (0, 2) for y in (0, 2) for z in (0, 2)],
                 "block of 27": [(x, y, z) for x in (0, 2, 4) for y in (0, 2, 4) for z in (0, 2, 4)]}
        for name, centers in cases.items():
            for inverted in (False, True):
                with self.subTest(name, inverted=inverted):
                    vertices, triangles = welded(*combined(*[cube(center) for center in centers]))
                    if inverted:
                        triangles = triangles[:, ::-1]
                    handle = self.load(vertices, triangles)
                    expected = np.min([box_field(points, center, 1) for center in centers], axis=0)
                    np.testing.assert_allclose(self.kernel.evaluate(handle, 0, points), expected, rtol=0, atol=1e-12)
                    mesh = self.kernel.mesh(handle, 0, 0.1)
                    self.assertAlmostEqual(mesh.stats.volume, 8.0 * len(centers), delta=1e-6)
        # The coincident faces of stacked boxes may also be triangulated differently.
        top = cube((0, 0, 2))
        vertices, triangles = welded(*combined(cube((0, 0, 0)), (top[0], other_diagonals(top[1]))))
        np.testing.assert_allclose(self.kernel.evaluate(self.load(vertices, triangles), 0, points),
                                   np.minimum(box_field(points, (0, 0, 0), 1), box_field(points, (0, 0, 2), 1)),
                                   rtol=0, atol=1e-12)

    def test_union_of_many_overlapping_bodies_matches_the_union_of_their_fields(self):
        rng = np.random.default_rng(8)
        # A row of boxes along +x (rays from the row cross more than 32
        # bodies), and boxes turned at random about their centres.
        row = [(np.eye(3), np.array([0.5 * i, 0.0, 0.0]), np.full(3, 0.5)) for i in range(40)]
        turned = []
        for _ in range(30):
            q = rng.normal(size=4)
            q /= np.linalg.norm(q)
            w, x, y, z = q
            rotation = np.array([[1 - 2 * (y * y + z * z), 2 * (x * y - w * z), 2 * (x * z + w * y)],
                                 [2 * (x * y + w * z), 1 - 2 * (x * x + z * z), 2 * (y * z - w * x)],
                                 [2 * (x * z - w * y), 2 * (y * z + w * x), 1 - 2 * (x * x + y * y)]])
            turned.append((rotation, rng.uniform(-1, 1, 3), rng.uniform(0.2, 1.0, 3)))
        for boxes in (row, turned):
            meshes = []
            for rotation, center, half in boxes:
                vertices, triangles = cube_mesh(2.0)
                meshes.append(((vertices * half) @ rotation.T + center, triangles))
            vertices, triangles = combined(*meshes)
            handle = self.load(vertices, triangles)
            lo, hi = vertices.min(axis=0), vertices.max(axis=0)
            phi = 2 - (1 + math.sqrt(5)) / 2
            axes = [lo[k] - 0.3 + 0.1 * (np.arange(int((hi[k] - lo[k] + 0.6) / 0.1)) + phi) for k in range(3)]
            grid = np.stack(np.meshgrid(*axes, indexing="ij"), -1).reshape(-1, 3)
            points = np.vstack([grid, rng.uniform(lo - 0.5, hi + 0.5, size=(5000, 3))])
            expected = np.min([box_field((points - center) @ rotation, 0, half)
                               for rotation, center, half in boxes], axis=0)
            values = self.kernel.evaluate(handle, 0, points)
            np.testing.assert_array_equal(values < 0, expected < 0)
            np.testing.assert_allclose(values, expected, rtol=0, atol=1e-12)

    def test_decoding_many_shells_facing_both_ways_is_fast(self):
        tetrahedron = np.array([[0, 0, 0], [0.5, 0, 0], [0, 0.5, 0], [0, 0, 0.5]])
        faces = np.array([[0, 2, 1], [0, 1, 3], [0, 3, 2], [1, 2, 3]])
        side = 40
        corners = np.stack(np.meshgrid(*[np.arange(side, dtype=float)] * 3, indexing="ij"), -1).reshape(-1, 3)
        vertices = (tetrahedron[None] + corners[:, None]).reshape(-1, 3)
        triangles = (faces[None] + 4 * np.arange(len(corners))[:, None, None]).copy()
        timings = []
        for inverted in (False, True):
            if inverted:
                triangles[1::2] = triangles[1::2, :, ::-1]
                vertices = vertices.reshape(-1, 4, 3)
                vertices[1::2] = vertices[1::2] * 0.9 + corners[1::2, None] * 0.1  # keep the total outward
                vertices = vertices.reshape(-1, 3)
            data = rsdf((MESH, [], mesh_params(vertices, triangles.reshape(-1, 3))))
            started = time.perf_counter()
            handle = self.kernel.create(data)
            timings.append(time.perf_counter() - started)
            self.addCleanup(self.kernel.lib.rsdf_graph_free, handle)
        # 64,000 shells, half of them inverted bodies turned outward: decoding
        # stays near the cost of the all-outward mesh (it once took seconds).
        self.assertLess(timings[1], 4 * timings[0] + 0.25, timings)
        self.assertLess(self.kernel.evaluate(handle, 0, np.array([[0.05, 0.05, 1.05]]))[0], 0)  # turned outward
        self.assertGreater(self.kernel.evaluate(handle, 0, np.array([[0.6, 0.6, 0.6]]))[0], 0)

    def test_queries_are_deterministic_and_compose_with_other_nodes(self):
        vertices, triangles = torus_mesh(2.0, 0.6, 64, 32)
        rng = np.random.default_rng(4)
        points = rng.uniform(-3, 3, size=(5000, 3))
        rotation = np.array([[0.0, -1.0, 0.0], [1.0, 0.0, 0.0], [0.0, 0.0, 1.0]])
        handle = self.kernel.create(rsdf(
            (MESH, [], mesh_params(vertices, triangles)),
            (TRANSFORM, [0], [*rotation.ravel(), 1.0, 2.0, 3.0, 2.0]),
            (SPHERE, [], [0.5]),
            (DIFFERENCE, [0, 2], [0.0])))
        self.addCleanup(self.kernel.lib.rsdf_graph_free, handle)
        values = self.kernel.evaluate(handle, 0, points)
        np.testing.assert_array_equal(values, self.kernel.evaluate(handle, 0, points, threads=1))
        singles = np.array([self.kernel.evaluate(handle, 0, point[None])[0] for point in points[:300]])
        np.testing.assert_array_equal(singles, values[:300])
        local = (points - [1.0, 2.0, 3.0]) @ rotation / 2.0
        np.testing.assert_allclose(self.kernel.evaluate(handle, 1, points),
                                   2.0 * self.kernel.evaluate(handle, 0, local), rtol=0, atol=1e-12)
        np.testing.assert_allclose(self.kernel.evaluate(handle, 3, points),
                                   np.maximum(values, 0.5 - np.linalg.norm(points, axis=1)), rtol=0, atol=1e-15)
        hits, distances = self.kernel.raycast(handle, 0, [[5.0, 0.0, 0.0]], [[-1.0, 0.0, 0.0]])
        self.assertTrue(hits[0])
        self.assertAlmostEqual(distances[0], 5.0 - 2.6, delta=1e-5)  # the outer equator's vertex

    def test_meshing_a_mesh_node_gives_a_closed_surface(self):
        vertices, triangles = torus_mesh(2.0, 0.6, 64, 32)
        handle = self.load(vertices, triangles)
        mesh = self.kernel.mesh(handle, 0, 0.03)
        self.assertEqual((mesh.stats.boundary_edges, mesh.stats.nonmanifold_edges, mesh.stats.components), (0, 0, 1))
        self.assertAlmostEqual(mesh.stats.volume / signed_volume(vertices, triangles), 1.0, delta=0.005)


# ------------------------------------------------------------------ bindings

class ReaderTests(unittest.TestCase):
    def test_binary_and_ascii_stl_weld_back_to_the_indexed_mesh(self):
        vertices, triangles = torus_mesh()
        vertices = vertices.astype(np.float32).astype(np.float64)  # what an STL can hold exactly
        for name, data in [("binary", stl_bytes(vertices, triangles)),
                           ("binary named solid", stl_bytes(vertices, triangles, header=b"solid but binary")),
                           ("ascii", ascii_stl(vertices, triangles))]:
            with self.subTest(name):
                read_vertices, read_triangles = radsdf.load_stl(data)
                self.assertEqual(read_vertices.dtype, np.float64)
                self.assertEqual(read_triangles.dtype, np.int64)
                self.assertEqual((len(read_vertices), len(read_triangles)), (len(vertices), len(triangles)))
                np.testing.assert_array_equal(read_vertices[read_triangles], vertices[triangles])
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "torus.stl"
            path.write_bytes(stl_bytes(vertices, triangles))
            for source in (path, str(path)):
                np.testing.assert_array_equal(radsdf.load_stl(source)[1], read_triangles)

    def test_obj_faces_use_relative_indices_references_weights_and_fans(self):
        vertices, _ = cube_mesh(2.0)
        lines = ["# a cube of quads", "mtllib unused.mtl", "o cube"]
        lines += ["v %r %r %r 2.0" % tuple(map(float, 2.0 * point)) for point in vertices]  # weights halve the coordinates
        lines += ["vt 0 0", "vn 0 0 1", "s off"]
        quads = [(0, 1, 3, 2), (4, 6, 7, 5), (0, 4, 5, 1), (2, 3, 7, 6), (0, 2, 6, 4), (1, 5, 7, 3)]
        for number, quad in enumerate(quads):
            corners = [f"{index - 8}/1/1" if number % 2 else f"{index + 1}//1" for index in quad]
            lines.append("f " + " ".join(corners) + "  # face %d" % number)
        read_vertices, read_triangles = radsdf.load_obj("\n".join(lines).encode())
        self.assertEqual((len(read_vertices), len(read_triangles)), (8, 12))
        self.assertAlmostEqual(abs(signed_volume(read_vertices, read_triangles)), 8.0)
        solid = radsdf.mesh_solid(read_vertices, read_triangles)
        self.assertAlmostEqual(solid.evaluate((0.0, 0.0, 0.0)), -1.0)
        self.assertAlmostEqual(solid.evaluate((2.0, 2.0, 2.0)), math.sqrt(3))
        pentagon = "\n".join(["v 0 0 0", "v 2 0 0", "v 3 1 0", "v 1 2 0", "v -1 1 0", "v 1 1 1",
                              "f 1 2 3 4 5", "f 1 6 2", "f 2 6 3", "f 3 6 4", "f 4 6 5", "f 5 6 1"])
        read_vertices, read_triangles = radsdf.load_obj(pentagon.encode())
        self.assertEqual(len(read_triangles), 3 + 5)
        np.testing.assert_array_equal(read_triangles[:3], [[0, 1, 2], [0, 2, 3], [0, 3, 4]])

    def test_welding_merges_near_duplicates_and_drops_collapsed_triangles(self):
        vertices, triangles = sphere_mesh(10.0, 12, 24)
        rng = np.random.default_rng(5)
        corners = vertices[triangles] + rng.normal(scale=1e-9, size=(len(triangles), 3, 3))
        # Copies on either side of a weld cell boundary (the tolerance is 1e-6 * 20 mm).
        corners[0, 0] = vertices[triangles[0, 0]] + 4e-7 * np.array([1.0, 1.0, 1.0])
        corners[0, 1] = vertices[triangles[0, 1]] - 4e-7 * np.array([1.0, 1.0, 1.0])
        soup = corners.reshape(-1, 3)
        read_vertices, read_triangles = radsdf.load_stl(ascii_stl(soup, np.arange(len(soup)).reshape(-1, 3)))
        self.assertEqual((len(read_vertices), len(read_triangles)), (len(vertices), len(triangles)))
        collapsed = np.vstack([soup, soup[:1] + 1e-8])
        rows = np.vstack([np.arange(len(soup)).reshape(-1, 3), [[0, len(soup), 1]]])
        welded_vertices, welded_triangles = radsdf._weld(collapsed, rows)
        self.assertEqual((len(welded_vertices), len(welded_triangles)), (len(vertices), len(triangles)))
        apart = np.array([[0.0, 0, 0], [1.0, 0, 0], [0, 1.0, 0], [3e-6, 0, 0]])
        self.assertEqual(len(radsdf._weld(apart, np.array([[0, 1, 2], [3, 2, 1]]))[0]), 4)

    def test_stl_variants_written_by_other_tools_are_read(self):
        vertices, triangles = cube_mesh(10.0)
        binary, text = stl_bytes(vertices, triangles), ascii_stl(vertices, triangles)
        streamed = binary[:80] + struct.pack("<I", 0) + binary[84:]
        lines = text.decode().split("\n")
        variants = {
            "binary + trailing bytes, 'solid' header": stl_bytes(vertices, triangles, header=b"solid x") + b"\0\0",
            "binary + trailing bytes": binary + b"\n\n",
            "binary with count 0 (streamed)": streamed,
            "ascii with a byte order mark": b"\xef\xbb\xbf" + text,
            "ascii, CR LF lines": text.replace(b"\n", b"\r\n"),
            "ascii, CR lines": text.replace(b"\n", b"\r"),
            "ascii, capitals and tabs": "\n".join("\t" + line.upper() for line in lines).encode(),
        }
        for name, data in variants.items():
            with self.subTest(name):
                read_vertices, read_triangles = radsdf.load_stl(data)
                self.assertEqual((len(read_vertices), len(read_triangles)), (8, 12))
                np.testing.assert_array_equal(read_vertices[read_triangles], vertices[triangles])
        # A facet with four vertex lines and one with two are not regrouped.
        vertex_lines = [index for index, line in enumerate(lines) if "vertex" in line]
        moved = lines[:]
        moved.insert(vertex_lines[2] + 1, lines[vertex_lines[3]])
        del moved[vertex_lines[3] + 1]
        with self.assertRaisesRegex(radsdf.SDFError, "exactly three vertex lines"):
            radsdf.load_stl("\n".join(moved).encode())

    def test_obj_vertex_colours_and_line_breaks(self):
        vertices, triangles = cube_mesh(2.0)
        faces = "".join("f %d %d %d\n" % tuple(index + 1 for index in row) for row in triangles)
        for name, suffix in [("rgb", " 0.5 0.25 1"), ("rgba", " 0.5 0.25 1 1")]:
            with self.subTest(name):
                obj = "".join("v %r %r %r%s\n" % (*map(float, point), suffix) for point in vertices) + faces
                read_vertices, read_triangles = radsdf.load_obj(obj.encode())
                np.testing.assert_array_equal(read_vertices[read_triangles], vertices[triangles])
        plain = "".join("v %r %r %r\n" % tuple(map(float, point)) for point in vertices) + faces
        for name, data in [("CR lines", plain.replace("\n", "\r").encode()),
                           ("CR LF lines", plain.replace("\n", "\r\n").encode()),
                           ("byte order mark", b"\xef\xbb\xbf" + plain.encode()),
                           ("Unicode line separator in a comment", ("# note\u2028f 1 2 3\n" + plain).encode()),
                           ("latin-1 NEL byte in a comment", b"# note\x85f 1 2 3\n" + plain.encode())]:
            with self.subTest(name):
                self.assertEqual(len(radsdf.load_obj(data)[1]), 12)
        with self.assertRaisesRegex(radsdf.SDFError, "optional weight or colour"):
            radsdf.load_obj(b"v 0 0 0 1 1\n")

    def test_readers_mend_t_junction_slivers_and_keep_touching_bodies(self):
        vertices, triangles = sliver_cube()
        with self.assertRaisesRegex(radsdf.SDFError, "1 triangle is degenerate"):
            radsdf.mesh_solid(vertices, triangles, weld=False)
        read_vertices, read_triangles = radsdf.load_stl(stl_bytes(vertices, triangles))
        self.assertEqual(len(read_triangles), len(triangles))  # the sliver goes, its neighbour splits in two
        solid = radsdf.mesh_solid(read_vertices, read_triangles)
        self.assertAlmostEqual(solid.evaluate((0.0, 0.0, 0.0)), -1.0)
        self.assertAlmostEqual(solid.evaluate((-1.5, -1.0, 0.0)), 0.5)
        # Bodies that touch along faces, edges or points, or nearly do, stay separate bodies.
        for name, offset in [("face", (0, 0, 2)), ("edge", (2, 2, 0)), ("point", (2, 2, 2)), ("gap", (0, 0, 2 + 1e-7))]:
            with self.subTest(name):
                data = stl_bytes(*combined(cube_mesh(2.0), cube_mesh(2.0, center=offset)))
                solid = radsdf.mesh_solid(*radsdf.load_stl(data))
                self.assertAlmostEqual(solid.mesh(cell_size=0.05).stats["volume"], 16.0, delta=1e-6)

    def test_welding_a_crafted_chain_is_fast(self):
        n = 64_000
        index = np.arange(n)
        step = 0.9e-6  # just within the weld tolerance of a unit mesh
        chain = np.stack([np.zeros(n), np.where(index < n // 2, index, n - index) * step, index * step], 1) + 0.25
        tetrahedron = np.array([[0, 0, 0], [1, 0, 0], [0, 1, 0], [0, 0, 1.0]])
        faces = np.array([[0, 2, 1], [0, 1, 3], [0, 3, 2], [1, 2, 3]])
        vertices = np.vstack([tetrahedron, chain])
        triangles = np.vstack([faces, np.stack([index[:-2], index[1:-1], index[2:]], 1) + 4])
        started = time.perf_counter()
        welded_vertices, welded_triangles = radsdf._weld(vertices, triangles)
        self.assertLess(time.perf_counter() - started, 2.0)  # once quadratic: minutes
        self.assertEqual((len(welded_vertices), len(welded_triangles)), (4, 4))

    def test_unreadable_files_are_explained(self):
        vertices, triangles = cube_mesh()
        cases = [
            (radsdf.load_stl, b"", "empty"),
            (radsdf.load_stl, stl_bytes(vertices, triangles)[:-7], "lists 12 triangles but holds only 11"),
            (radsdf.load_stl, b"neither binary nor text" * 3, "neither a binary STL"),
            (radsdf.load_stl, b"solid empty\nendsolid empty\n", "no triangles"),
            (radsdf.load_stl, b"solid q\nfacet normal 0 0 1\nouter loop\nvertex 0 0 0\nvertex 1 0 0\n"
                              b"vertex 0 1 0\nvertex 1 1 0\nendloop\nendfacet\nendsolid q\n", "exactly three"),
            (radsdf.load_stl, b"solid q\nfacet normal 0 0 1\nouter loop\nvertex 0 0 x\nvertex 1 0 0\n"
                              b"vertex 0 1 0\nendloop\nendfacet\nendsolid q\n", "must be numbers"),
            (radsdf.load_obj, b"v 0 0 0\nf 1 2 3\n", "line 2: '2' refers to a vertex"),
            (radsdf.load_obj, b"v 0 0 0 0\n", "weight must not be zero"),
            (radsdf.load_obj, b"v 0 0\n", "three coordinates"),
            (radsdf.load_obj, b"v 0 0 0\nv 1 0 0\nf 1 2\n", "at least three"),
            (radsdf.load_obj, b"# nothing\nv 0 0 0\n", "no faces"),
            (radsdf.load_obj, "/no/such/mesh.obj", "could not read"),
            (radsdf.load_stl, 42, "file path or the file's bytes"),
        ]
        for reader, source, problem in cases:
            with self.subTest(problem), self.assertRaisesRegex(radsdf.SDFError, problem):
                reader(source)


@unittest.skipUnless(library_path(), "libRADSDF is not built (swift build --product RADSDF -c release)")
class MeshSolidTests(unittest.TestCase):
    def test_mesh_solids_evaluate_serialize_and_load_from_bytes(self):
        vertices, triangles = sphere_mesh(5.0)
        solid = radsdf.mesh_solid(vertices, triangles)
        self.assertIsInstance(solid, radsdf.SDF3)
        self.assertEqual(solid.opcode, radsdf.Opcode.MESH)
        np.testing.assert_array_equal(solid.bounds.lo, vertices.min(axis=0))
        np.testing.assert_array_equal(solid.bounds.hi, vertices.max(axis=0))
        self.assertEqual(solid.lipschitz, 1.0)
        points = np.random.default_rng(6).uniform(-6, 6, size=(400, 3))
        expected = reference_field(points, vertices, triangles)
        np.testing.assert_allclose(solid.evaluate(points), expected, rtol=0, atol=1e-12)
        design = radsdf.union(solid, solid.translate((20, 0, 0)), radsdf.box(3))
        data, roots = radsdf.serialize(design)
        self.assertEqual(data.count(struct.pack("<4I", 9, 0, 2 + 3 * len(vertices) + 3 * len(triangles), 0)), 1)
        with radsdf.Graph.from_bytes(data, roots) as graph:
            np.testing.assert_array_equal(graph.evaluate(roots[0], points), design.evaluate(points))
        mesh = (solid - radsdf.box(4, center=(0, 0, 5))).mesh(cell_size=0.1)
        self.assertEqual((mesh.stats["boundary_edges"], mesh.stats["nonmanifold_edges"]), (0, 0))

    def test_mesh_solid_checks_its_arguments_and_reports_mesh_problems(self):
        vertices, triangles = cube_mesh()
        cases = [((vertices[:, :2], triangles), "vertices must be an \\(N, 3\\) array"),
                 ((vertices, triangles[:, :2]), "triangles must be an \\(M, 3\\) array"),
                 ((vertices, triangles + 0.5), "triangles must be an \\(M, 3\\) array"),
                 ((vertices, triangles + 8), "must index the 8 vertices"),
                 ((np.where(vertices > 0, np.nan, vertices), triangles), "finite"),
                 ((vertices, triangles[:3]), "at least four triangles"),
                 ((vertices, triangles[1:]), "mesh_solid: the mesh is not a closed, consistently oriented manifold")]
        for arguments, problem in cases:
            with self.subTest(problem), self.assertRaisesRegex(radsdf.SDFError, problem):
                radsdf.mesh_solid(*arguments)
        # Welding is optional; unused vertices are always dropped.
        spare = np.vstack([vertices, [[9.0, 9.0, 9.0]]])
        self.assertEqual(radsdf.mesh_solid(spare, triangles, weld=False).params[0], 8)


# ------------------------------------------------------------------ builds

@unittest.skipUnless(library_path(), "libRADSDF is not built (swift build --product RADSDF -c release)")
class ImportedAssetBuildTests(unittest.TestCase):
    def build(self, files, script, assets):
        folder = Path(tempfile.mkdtemp())
        self.addCleanup(lambda: __import__("shutil").rmtree(folder, True))
        for name, data in files.items():
            (folder / name).write_bytes(data)
        spec = {"title": "Imported mesh", "summary": "Mesh input test", "units": "mm", "script": script,
                "inputAssets": assets}
        (folder / "request.json").write_text(json.dumps(spec))
        return worker.run(folder / "request.json", folder / "output", 2), folder, spec

    def test_an_imported_stl_unions_with_a_primitive_and_meshes_closed(self):
        vertices, triangles = cube_mesh(10.0)
        assets = [{"id": "asset-cube", "name": "cube.stl", "format": "stl", "path": "input-000.stl"},
                  {"id": "asset-copy", "name": "copy.stl", "format": "stl", "path": "input-001.stl"}]
        data = stl_bytes(vertices, triangles)
        script = ("cube = imported_assets['asset-cube']\n"
                  "assert cube is imported_assets['asset-copy'] and len(imported_assets) == 2\n"
                  "result = cube | box((30, 4, 4))\n")
        result, folder, spec = self.build({"input-000.stl": data, "input-001.stl": data}, script, assets)
        self.assertEqual(result["solids"], 1)
        self.assertEqual((result["mesh"]["boundaryEdges"], result["mesh"]["nonmanifoldEdges"]), (0, 0))
        self.assertAlmostEqual(result["volume"] / (1000 + 480 - 160), 1.0, delta=0.005)
        np.testing.assert_allclose(result["bounds"], [-15, -5, -5, 15, 5, 5], atol=0.05)
        self.assertEqual(result["validatedHash"], sdf_build.validation_hash(spec, folder))
        changed = dict(spec)
        (folder / "input-001.stl").write_bytes(stl_bytes(vertices * 1.01, triangles))
        self.assertNotEqual(sdf_build.validation_hash(changed, folder), result["validatedHash"])

    def test_an_imported_obj_works_in_assemblies(self):
        obj = "".join("v %r %r %r\n" % tuple(map(float, point)) for point in cube_mesh(4.0)[0])
        obj += "".join("f %d %d %d %d\n" % tuple(i + 1 for i in quad)
                       for quad in [(0, 1, 3, 2), (4, 6, 7, 5), (0, 4, 5, 1), (2, 3, 7, 6), (0, 2, 6, 4),
                                    (1, 5, 7, 3)])
        assets = [{"id": "asset-block", "name": "block.obj", "format": "obj", "path": "input-000.obj"}]
        script = ("result = Assembly('Kit')\n"
                  "result.add(imported_assets['asset-block'] - cylinder(1, 10), id='block', name='Block')\n"
                  "result.add(sphere(2).translate((10, 0, 0)), id='ball', name='Ball')\n")
        result, _, _ = self.build({"input-000.obj": obj.encode()}, script, assets)
        self.assertEqual(result["solids"], 2)
        block = next(stats for stats in result["partStats"] if stats["solid"] == 0)
        self.assertAlmostEqual(block["volume"] / (64 - math.pi * 4), 1.0, delta=0.01)

    def test_native_operations_work_on_imported_mesh_parts(self):
        vertices, triangles = cube_mesh(10.0)
        assets = [{"id": "asset-cube", "name": "cube.stl", "format": "stl", "path": "input-000.stl"}]
        script = ("result = Assembly('Fixture')\n"
                  "result.add(imported_assets['asset-cube'], id='cube', name='Cube')\n"
                  "result.add(sphere(2).translate((12, 0, 0)), id='ball', name='Ball')\n")
        result, folder, _ = self.build({"input-000.stl": stl_bytes(vertices, triangles)}, script, assets)
        output = folder / "output"
        probed, _ = native(output, result, "probe", solids=[0], points=[[0, 0, 0], [8, 0, 0], [0, 0, -6]])
        np.testing.assert_allclose([row[0]["distance"] for row in probed["samples"]], [-5, 3, 1], atol=1e-9)
        measured, _ = native(output, result, "measure", entities=[{"kind": "part", "solid": 0},
                                                                   {"kind": "part", "solid": 1}])
        distance = next(metric["values"] for metric in measured["metrics"] if metric["id"] == "pair:distance")
        self.assertAlmostEqual(distance[0], 5.0, delta=1e-4)
        oriented, _ = native(output, result, "print_orient", solidIndices=[0])
        self.assertAlmostEqual(oriented["orientations"][0]["bounds"][2] - oriented["orientations"][0]["bounds"][5],
                               -10.0, delta=0.2)
        exported, export_folder = native(output, result, "export", solidIndices=[0])
        self.assertEqual(exported["solids"], 1)
        corners, _ = read_stl(export_folder / "model.stl")
        self.assertAlmostEqual(stl_volume(corners) / 1000.0, 1.0, delta=0.005)

    def test_unusable_assets_and_unknown_ids_are_named(self):
        vertices, triangles = cube_mesh(10.0)
        assets = [{"id": "asset-open", "name": "open part.stl", "format": "stl", "path": "input-000.stl"}]
        with self.assertRaises(sdf_build.ModelError) as caught:
            self.build({"input-000.stl": stl_bytes(vertices, triangles[1:])},
                       "result = imported_assets['asset-open'] | box(1)", assets)
        self.assertRegex(str(caught.exception), "^The imported STL input 'open part.stl' cannot be used as a solid: "
                                                "the mesh is not a closed")
        self.assertEqual(caught.exception.diagnostic, {
            "stage": "input_import", "geometryScope": "input_asset",
            "inputAsset": {"id": "asset-open", "name": "open part.stl", "format": "stl"}})
        with self.assertRaisesRegex(sdf_build.ModelError, "'asset-other' is not available in this build"):
            self.build({"input-000.stl": stl_bytes(vertices, triangles)}, "result = imported_assets['asset-other']",
                       assets)
        # Unused inputs are never read as meshes, but stay in the receipt; a
        # membership test reads none either.
        result, _, _ = self.build({"input-000.stl": b"not a mesh"}, "result = box(1)", assets)
        self.assertEqual(result["solids"], 1)
        script = "result = box(5) if 'asset-open' in imported_assets and 7 not in imported_assets else sphere(1)"
        result, _, _ = self.build({"input-000.stl": b"not a mesh"}, script, assets)
        self.assertAlmostEqual(result["volume"], 125, delta=0.5)
        with self.assertRaisesRegex(sdf_build.ModelError, "list it in rad_build assetIDs"):
            self.build({}, "result = imported_assets['asset-new']", [])

    def test_inverted_overlapping_shells_and_thin_sheets_build(self):
        reverse = lambda mesh: (mesh[0], mesh[1][:, ::-1])  # noqa: E731
        assets = [{"id": "asset-a", "name": "part.stl", "format": "stl", "path": "input-000.stl"}]
        data = stl_bytes(*combined(cube_mesh(10.0, center=(5, 5, 5)), reverse(cube_mesh(7.0, center=(8.5, 8.5, 8.5)))))
        result, _, _ = self.build({"input-000.stl": data}, "result = imported_assets['asset-a']", assets)
        self.assertAlmostEqual(result["volume"], 1218.0, delta=1.0)  # the inverted body joins the union
        np.testing.assert_allclose(result["bounds"], [0, 0, 0, 12, 12, 12], atol=0.05)
        # A sheet thinner than its own preview cells fails alone after its
        # program ran (rad_import still registers it), and builds as a cutter.
        vertices, triangles = cube_mesh(1.0)
        data = stl_bytes(vertices * [300, 300, 0.5], triangles)
        with self.assertRaises(sdf_build.ModelError) as caught:
            self.build({"input-000.stl": data}, "result = imported_assets['asset-a']", assets)
        self.assertEqual(caught.exception.diagnostic["stage"], "preview_meshing")
        result, _, _ = self.build({"input-000.stl": data},
                                  "result = box((40, 40, 20)) - imported_assets['asset-a'].offset(0.3)", assets)
        self.assertAlmostEqual(result["volume"], 40 * 40 * 20 - 40 * 40 * 1.1, delta=20)

    def test_builds_free_the_graphs_of_program_queries(self):
        vertices, triangles = cube_mesh(10.0)
        assets = [{"id": "asset-cube", "name": "cube.stl", "format": "stl", "path": "input-000.stl"}]
        script = "cube = imported_assets['asset-cube']\nassert cube.evaluate((0, 0, 0)) < 0\nresult = cube\n"
        cached = []
        original = radsdf._release_graphs

        def release():
            cached.extend(radsdf._graph_cache.values())
            original()
        radsdf._release_graphs = release
        try:
            self.build({"input-000.stl": stl_bytes(vertices, triangles)}, script, assets)
        finally:
            radsdf._release_graphs = original
        self.assertTrue(cached)
        self.assertTrue(all(graph.closed for graph in cached))
        self.assertEqual(len(radsdf._graph_cache), 0)


# ------------------------------------------------------------------ sanitizer fuzzing

def fuzz_meshes(seed, count):
    """Random MESH cases aimed at the decoder's harder paths: shells touching
    with shared vertices, coincident and crossing faces, nested shells facing
    either way, slivers, repeated corners, dropped triangles and corrupted
    indices, at tiny to huge scales."""
    rng = np.random.default_rng(seed)
    reverse = lambda mesh: (mesh[0], mesh[1][:, ::-1])  # noqa: E731
    bases = [cube_mesh(), sphere_mesh(1.0, 6, 8), torus_mesh(2.0, 0.6, 8, 6), sliver_cube(),
             welded(*combined(cube_mesh(), cube_mesh(center=(0, 0, 2)))),
             welded(*combined(cube_mesh(), cube_mesh(center=(2, 2, 0)), cube_mesh(center=(2, 0, 0)))),
             welded(*combined(cube_mesh(), cube_mesh(center=(2, 2, 2)))),
             welded(*combined(cube_mesh(), cube_mesh())),
             combined(cube_mesh(6), reverse(cube_mesh(4)), cube_mesh(2)),
             combined(cube_mesh(6), reverse(cube_mesh(4)), reverse(cube_mesh(2))),
             combined(cube_mesh(2), reverse(cube_mesh(1.4, center=(0.8, 0.8, 0.8)))),
             combined(cube_mesh(6), reverse(cube_mesh(2, center=(1, 0, 0))), reverse(cube_mesh(2, center=(2, 0, 0))))]
    for _ in range(count):
        vertices, triangles = bases[rng.integers(len(bases))]
        vertices, triangles = vertices.astype(np.float64).copy(), np.array(triangles, dtype=np.int64)
        for _ in range(int(rng.integers(1, 4))):
            operation = int(rng.integers(10))
            if operation == 0 and len(triangles) > 4:  # holes
                triangles = triangles[rng.random(len(triangles)) > 0.05]
            elif operation == 1:  # some triangles turned around
                flipped = rng.random(len(triangles)) < 0.1
                triangles[flipped] = triangles[flipped][:, ::-1]
            elif operation == 2:  # two vertices merged
                a, b = rng.integers(len(vertices), size=2)
                triangles[triangles == b] = a
            elif operation == 3:  # jitter at some scale
                vertices = vertices + rng.normal(scale=rng.choice([1e-12, 1e-6, 1e-2, 0.3]), size=vertices.shape)
            elif operation == 4:  # repeated triangles
                triangles = np.vstack([triangles, triangles[rng.integers(len(triangles), size=int(rng.integers(1, 4)))]])
            elif operation == 5:  # scale
                vertices = vertices * rng.choice([1e-100, 1e-6, 1e6, 1e11])
            elif operation == 6:  # a copy of the whole mesh, moved, facing either way
                moved = (vertices + rng.normal(scale=rng.choice([0.0, 0.5, 2.0]), size=3),
                         triangles if rng.random() < 0.5 else triangles[:, ::-1])
                vertices, triangles = combined((vertices, triangles), moved)
                if rng.random() < 0.5 and triangles.max() < len(vertices):
                    vertices, triangles = welded(vertices, triangles)
            elif operation == 7 and triangles.max() < len(vertices):  # vertex order shuffled
                order = rng.permutation(len(vertices))
                vertices, triangles = vertices[order], np.argsort(order)[triangles]
            elif operation == 8:  # triangle order and corners rotated
                triangles = np.roll(triangles[rng.permutation(len(triangles))], int(rng.integers(3)), axis=1)
            else:  # an index out of range or a corner repeated
                row, column = int(rng.integers(len(triangles))), int(rng.integers(3))
                triangles[row, column] = triangles[row, (column + 1) % 3] if rng.random() < 0.5 else len(vertices)
        yield rsdf((MESH, [], mesh_params(vertices, triangles)))


@unittest.skipUnless(os.environ.get("RAD_RUN_SANITIZER_FUZZ") == "1" and shutil.which("clang++"),
                     "Set RAD_RUN_SANITIZER_FUZZ=1 (needs clang++) to fuzz the MESH decoder under sanitizers.")
class MeshSanitizerFuzzTests(unittest.TestCase):
    def test_random_meshes_decode_and_query_without_sanitizer_reports(self):
        kernel = Path(__file__).resolve().parents[2] / "Sources" / "RADSDFKernel"
        count = int(os.environ.get("RAD_FUZZ_CASES", "1500"))
        with tempfile.TemporaryDirectory() as directory:
            folder = Path(directory)
            driver = folder / "mesh_fuzz_driver"
            flags = ["-std=c++17", "-O1", "-g", "-fsanitize=address,undefined", "-fno-sanitize-recover=undefined",
                     "-fno-omit-frame-pointer"]
            built = subprocess.run(["clang++", *flags, "-I", str(kernel / "include"), "-I", str(kernel),
                                    *map(str, sorted(kernel.glob("*.cpp"))),
                                    str(Path(__file__).with_name("mesh_fuzz_driver.cpp")), "-o", str(driver)],
                                   capture_output=True, text=True)
            self.assertEqual(built.returncode, 0, built.stderr[-4000:])
            cases = []
            for index, data in enumerate(fuzz_meshes(int(os.environ.get("RAD_FUZZ_SEED", "11")), count)):
                cases.append(folder / f"case-{index:05d}.rsdf")
                cases[-1].write_bytes(data)
            for start in range(0, len(cases), 250):
                run = subprocess.run([str(driver), *map(str, cases[start:start + 250])], capture_output=True, text=True,
                                     env={**os.environ, "ASAN_OPTIONS": "detect_leaks=0:abort_on_error=0"})
                report = run.stdout[-2000:] + run.stderr[-6000:]
                self.assertEqual(run.returncode, 0, report)
                self.assertNotIn("runtime error", run.stderr, report)
                self.assertNotIn("AddressSanitizer", run.stderr, report)


if __name__ == "__main__":
    unittest.main()
