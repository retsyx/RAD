"""Feature-preserving mesh simplification (SDF-DESIGN.md, "Mesh
simplification"): rsdf_mesh_simplify through radsdf's Graph.mesh(simplify=),
and the worker's 3MF export with its tolerance.

Simplified meshes must stay closed and manifold, keep every changed triangle
within the tolerance of the field (sampled densely here, independently of the
kernel's own samples), keep crease lines, corners and orientation, and be
identical for any thread count. Native tests are skipped without the kernel.
"""
import ctypes
import hashlib
import itertools
import math
import os
from pathlib import Path
import shutil
import subprocess
import tempfile
import time
import unittest
from unittest import mock

import numpy as np

from sdf_test_support import assert_closed_and_outward, build, native, read_3mf, signed_volume
import radsdf
from radsdf import SDFError, box, cone, cylinder, involute_gear, rect, regular_polygon, sphere, torus, union
import sdf_build
import sdf_ops

TOLERANCE = 0.01
# Changed triangles near curved creases measure up to 31% above the original
# mesh's own largest deviation nearby (SDF-DESIGN.md, "Mesh simplification").
LOCAL_EXCESS = 1.5


def kernel_available():
    try:
        radsdf.kernel_version()
        return True
    except SDFError:
        return False


def plate_with_holes(columns=10, rows=4):
    holes = [cylinder(1.5, 10, center=(x * 5 - 2.5 * (columns - 1), y * 5 - 2.5 * (rows - 1), 0))
             for x in range(columns) for y in range(rows)]
    return box((5 * columns, 5 * rows, 5)) - union(*holes)


def gear():
    return involute_gear(24, 2.0).extrude(8, center=True) - cylinder(4, 10)


def enclosure(depth=14):
    outer = rect((59, 33.6), radius=4).extrude(depth, center=True)
    inner = rect((56.6, 31.2), radius=2.8).extrude(depth + 2, center=True).translate((0, 0, 2.2))
    shape = outer - inner - box((14, 8, 6), center=(0, -16, 2))
    for x in (-21, 21):
        for y in (-10, 10):
            shape = (shape | cylinder(2.8, depth - 1.2, center=(x, y, 0.6))) - cylinder(1.2, depth + 2,
                                                                                        center=(x, y, 1))
    return shape


def triangle_points(mesh, divisions=6):
    """Barycentric grid points of every triangle (including corners and sides),
    as an array (points per triangle, triangles, 3)."""
    corners = mesh.positions.astype(np.float64)[mesh.triangles]
    a, b, c = corners[:, 0], corners[:, 1], corners[:, 2]
    return np.stack([(i * a + j * b + (divisions - i - j) * c) / divisions
                     for i in range(divisions + 1) for j in range(divisions + 1 - i)])


def deviation(graph, node, mesh, divisions=6):
    """The largest |f| over a dense sampling of each triangle (per triangle)."""
    points = triangle_points(mesh, divisions)
    values = np.abs(graph.evaluate(node, points.reshape(-1, 3))).reshape(points.shape[:2])
    return values.max(axis=0)


def segment_distances(points, segments):
    """Distance from each point to the nearest of the segments (S, 2, 3)."""
    a, b = segments[:, 0].astype(np.float64), segments[:, 1].astype(np.float64)
    ab = b - a
    length2 = np.maximum((ab * ab).sum(axis=1), 1e-300)
    out = np.empty(len(points))
    for start in range(0, len(points), 64):
        p = points[start:start + 64, None, :]
        t = np.clip(((p - a) * ab).sum(axis=2) / length2, 0, 1)
        out[start:start + 64] = np.sqrt((((a + t[..., None] * ab) - p) ** 2).sum(axis=2)).min(axis=1)
    return out


def crease_points(creases):
    creases = creases.astype(np.float64)
    return np.concatenate([creases[:, 0], creases[:, 1], 0.5 * (creases[:, 0] + creases[:, 1])])


def qualities(mesh):
    corners = mesh.positions.astype(np.float64)[mesh.triangles]
    a, b, c = corners[:, 0], corners[:, 1], corners[:, 2]
    doubled = np.linalg.norm(np.cross(b - a, c - a), axis=1)
    return 2 * math.sqrt(3) * doubled / (((b - a) ** 2) + ((c - b) ** 2) + ((a - c) ** 2)).sum(axis=1)


def facings(graph, node, mesh):
    """Cosine between each facet normal and the field gradient at its centroid."""
    corners = mesh.positions.astype(np.float64)[mesh.triangles]
    normals = np.cross(corners[:, 1] - corners[:, 0], corners[:, 2] - corners[:, 0])
    normals /= np.linalg.norm(normals, axis=1, keepdims=True)
    return (normals * graph.gradient(node, corners.mean(axis=1))).sum(axis=1)


def triangle_keys(mesh):
    """Each triangle's corner positions, rotated to start at the smallest."""
    corners = mesh.positions[mesh.triangles]
    keys = []
    for rows in corners.tolist():
        rows = [tuple(row) for row in rows]
        first = min(range(3), key=lambda k: rows[k])
        keys.append(tuple(rows[first:] + rows[:first]))
    return keys


def digest(mesh):
    return hashlib.sha256(b"".join(array.tobytes() for array in
                                   (mesh.positions, mesh.normals, mesh.triangles, mesh.creases))).hexdigest()


def corners_of(mesh):
    return mesh.positions.astype(np.float64)[mesh.triangles]


def field_deviation(graph, nodes, corners, divisions):
    """(largest |f| over a barycentric grid of each triangle, the grid point
    where it peaks) for triangle corners (T, 3, 3); |f| is the smallest over
    `nodes`."""
    largest, peaks = np.empty(len(corners)), np.empty((len(corners), 3))
    for start in range(0, len(corners), 50_000):
        chunk = corners[start:start + 50_000]
        a, b, c = chunk[:, 0], chunk[:, 1], chunk[:, 2]
        points = np.stack([(i * a + j * b + (divisions - i - j) * c) / divisions
                           for i in range(divisions + 1) for j in range(divisions + 1 - i)])
        values = np.min([np.abs(graph.evaluate(node, points.reshape(-1, 3))) for node in nodes],
                        axis=0).reshape(points.shape[:2])
        largest[start:start + len(chunk)] = values.max(axis=0)
        peaks[start:start + len(chunk)] = points[values.argmax(axis=0), np.arange(len(chunk))]
    return largest, peaks


def local_excess(graph, nodes, simplified, original, tolerance, radius=0.25):
    """Each simplified triangle's largest |f| (dense samples) over the
    original mesh's largest |f| around the point where it peaks (triangles
    whose centroids lie in the 27 cells of `radius` there), or over the
    tolerance where that is larger: new deviation shows as ratios above 1,
    deviation the original mesh already has nearby does not."""
    deviation, peaks = field_deviation(graph, nodes, simplified, 16)
    reference, _ = field_deviation(graph, nodes, original, 6)
    centroids = original.mean(axis=1)
    origin = np.floor(centroids.min(axis=0) / radius) - 2
    cells = lambda points: (np.floor(points / radius) - origin).astype(np.int64)  # noqa: E731
    keys = cells(centroids)
    dims = keys.max(axis=0) + 3
    linear = lambda k: (k[:, 0] * dims[1] + k[:, 1]) * dims[2] + k[:, 2]  # noqa: E731
    unique, inverse = np.unique(linear(keys), return_inverse=True)
    largest = np.zeros(len(unique))
    np.maximum.at(largest, inverse, reference)
    near, at_peaks = np.zeros(len(peaks)), np.clip(cells(peaks), 1, dims - 2)
    for offset in itertools.product((-1, 0, 1), repeat=3):
        key = linear(at_peaks + np.array(offset))
        index = np.clip(np.searchsorted(unique, key), 0, len(unique) - 1)
        near = np.maximum(near, np.where(unique[index] == key, largest[index], 0.0))
    return deviation / np.maximum(near, tolerance)


def edge_turns(mesh):
    """(midpoints, normal cosines) of the edges two triangles share, split
    vertices welded by position."""
    welded, index = np.unique(mesh.positions, axis=0, return_inverse=True)
    triangles = index.reshape(-1)[mesh.triangles]
    corners = corners_of(mesh)
    normals = np.cross(corners[:, 1] - corners[:, 0], corners[:, 2] - corners[:, 0])
    lengths = np.linalg.norm(normals, axis=1, keepdims=True)
    normals = np.divide(normals, lengths, out=np.zeros_like(normals), where=lengths > 0)
    edges = np.sort(np.concatenate([triangles[:, [0, 1]], triangles[:, [1, 2]], triangles[:, [2, 0]]]), axis=1)
    owners = np.tile(np.arange(len(triangles)), 3)
    order = np.lexsort((edges[:, 1], edges[:, 0]))
    edges, owners = edges[order], owners[order]
    shared = np.flatnonzero(np.all(edges[1:] == edges[:-1], axis=1))
    midpoints = welded.astype(np.float64)[edges[shared]].mean(axis=1)
    return midpoints, (normals[owners[shared]] * normals[owners[shared + 1]]).sum(axis=1)


@unittest.skipUnless(kernel_available(), "the native SDF kernel is not available")
class SimplifyTests(unittest.TestCase):
    def meshes(self, shape, cell, tolerance=TOLERANCE, threads=0, **options):
        graph = radsdf.Graph(shape)
        self.addCleanup(graph.close)
        full = graph.mesh(shape, cell_size=cell, threads=threads, **options)
        simple = graph.mesh(shape, cell_size=cell, threads=threads, simplify=tolerance, **options)
        return graph, graph.roots[0], full, simple

    def assert_sound(self, graph, node, full, simple, tolerance=TOLERANCE):
        """Closed, manifold, same pieces, small volume change, and within the
        tolerance (or the full mesh's own deviation where that is larger)."""
        self.assertEqual(simple.stats["unsimplified_triangles"], len(full.triangles))
        self.assertEqual(simple.stats["simplify_tolerance"], tolerance)
        self.assertLess(len(simple.triangles), len(full.triangles))
        for key in ("boundary_edges", "nonmanifold_edges", "components"):
            self.assertEqual(simple.stats[key], full.stats[key], key)
        assert_closed_and_outward(self, simple.positions, simple.triangles)
        self.assertAlmostEqual(simple.stats["volume"] / full.stats["volume"], 1, delta=0.005)
        worst = float(deviation(graph, node, simple).max())
        if worst > tolerance:  # only where the full mesh deviates as much (sampled alike)
            self.assertLessEqual(worst, float(deviation(graph, node, full).max()) * 1.001)
        # And locally: no triangle deviates much beyond the full mesh near it.
        self.assertLessEqual(float(local_excess(graph, [node], corners_of(simple), corners_of(full), tolerance).max()),
                             LOCAL_EXCESS)

    def assert_creases_kept(self, full, simple, tolerance=TOLERANCE):
        """Every crease point of the full mesh lies within the tolerance of a
        simplified crease segment, and the other way round."""
        self.assertGreater(len(simple.creases), 0)
        self.assertLessEqual(segment_distances(crease_points(full.creases), simple.creases).max(), tolerance)
        self.assertLessEqual(segment_distances(crease_points(simple.creases), full.creases).max(), tolerance)

    def test_a_box_collapses_to_twelve_triangles_with_exact_corners(self):
        graph, node, full, simple = self.meshes(box((20, 10, 5)), 0.1)
        self.assertGreater(len(full.triangles), 100_000)
        self.assertEqual(len(simple.triangles), 12)
        self.assertEqual(len(simple.creases), 12)
        corners = {tuple(point) for point in np.unique(simple.positions, axis=0).tolist()}
        self.assertEqual(corners, {(x, y, z) for x in (-10, 10) for y in (-5, 5) for z in (-2.5, 2.5)})
        self.assert_sound(graph, node, full, simple)
        self.assertAlmostEqual(simple.stats["volume"], 1000, places=9)

    def test_a_flat_plate_with_holes_collapses_massively_and_keeps_its_rims(self):
        graph, node, full, simple = self.meshes(plate_with_holes(), 0.1)
        self.assertLess(len(simple.triangles), 0.02 * len(full.triangles))
        self.assert_sound(graph, node, full, simple)
        self.assert_creases_kept(full, simple)
        # Every hole keeps a rim on both faces: the crease segments at each hole's top.
        top = simple.creases[np.all(np.abs(simple.creases[:, :, 2] - 2.5) < 1e-4, axis=1)]
        for x in range(10):
            for y in range(4):
                center = np.array([x * 5 - 22.5, y * 5 - 7.5])
                radii = np.linalg.norm(top[:, :, :2].reshape(-1, 2) - center, axis=1)
                ring = radii[radii < 2]
                self.assertGreater(len(ring), 20)
                self.assertTrue(np.all(np.abs(ring - 1.5) <= TOLERANCE), (x, y, ring.min(), ring.max()))

    def test_a_sphere_simplifies_moderately_within_the_tolerance(self):
        graph, node, full, simple = self.meshes(sphere(10), 0.25)
        ratio = len(simple.triangles) / len(full.triangles)
        self.assertGreater(ratio, 0.03)
        self.assertLess(ratio, 0.5)
        self.assert_sound(graph, node, full, simple)
        self.assertEqual(len(simple.creases), 0)
        # Smooth shading normals stay the field's gradients.
        normals = graph.gradient(node, simple.positions.astype(np.float64))
        self.assertGreater(float((normals * simple.normals).sum(axis=1).min()), 0.999)

    def test_gears_keep_their_tooth_edges(self):
        graph, node, full, simple = self.meshes(gear(), 0.1)
        self.assertLess(len(simple.triangles), 0.05 * len(full.triangles))
        self.assert_sound(graph, node, full, simple)
        self.assert_creases_kept(full, simple)

    def test_the_enclosure_simplifies_within_the_tolerance(self):
        graph, node, full, simple = self.meshes(enclosure(), 0.1)
        self.assertLess(len(simple.triangles), 0.02 * len(full.triangles))
        self.assert_sound(graph, node, full, simple)
        self.assert_creases_kept(full, simple)

    def test_changed_triangles_never_flip_or_become_needles(self):
        for name, shape, cell in (("sphere", sphere(10), 0.2), ("plate", plate_with_holes(4, 2), 0.1),
                                  ("gear", gear(), 0.1), ("rounded box", box((20, 10, 5), radius=1.5), 0.1),
                                  ("blend", union(sphere(4), box((8, 3, 3)), smooth=1.0), 0.05)):
            with self.subTest(shape=name):
                graph, node, full, simple = self.meshes(shape, cell)
                original = set(triangle_keys(full))
                changed = np.array([key not in original for key in triangle_keys(simple)])
                self.assertTrue(changed.any())
                self.assertGreater(float(facings(graph, node, simple)[changed].min()), 0.25)
                self.assertGreaterEqual(float(qualities(simple)[changed].min()),
                                        min(0.05, float(qualities(full).min())) * 0.999)

    def test_changed_triangles_never_fold_onto_their_neighbours(self):
        # These once folded flaps back onto adjacent triangles (normals nearly
        # opposite) where the full mesh turns by about 90 degrees.
        holes = union(*[cylinder(0.8, 10, axis=axis) for axis in ((1, 0, 0), (0, 1, 0), (0, 0, 1))])
        cases = (("knife edges", box((6, 6, 6)) & box((6, 6, 6)).rotate(45, (1, 1, 0)), 0.08),
                 ("crossing holes", sphere(4) - holes, 0.08),
                 ("turned boxes", box((6, 4, 3)).rotate(25, (1, 0, 2)) | box((5, 5, 2)).rotate(-40, (0, 1, 1)), 0.06),
                 ("cut torus", torus(4, 1.5) - box((3, 10, 10), center=(2, 0, 0)).rotate(15, (0, 1, 0)), 0.07))
        for name, shape, cell in cases:
            with self.subTest(shape=name):
                graph, node, full, simple = self.meshes(shape, cell)
                self.assertLess(len(simple.triangles), 0.05 * len(full.triangles))
                points, turns = edge_turns(full)
                sharp = points[turns < -0.8]
                folds, turns = edge_turns(simple)
                for point in folds[turns < -0.9]:  # only where the full mesh turns as sharply nearby
                    self.assertTrue(len(sharp) and np.linalg.norm(sharp - point, axis=1).min() < 4 * cell,
                                    f"a fold at {point.round(3).tolist()}")

    def test_changed_triangles_stay_near_the_full_mesh_at_curved_creases(self):
        # Dual contouring places vertices of curved creases off the crease, so
        # the full mesh exceeds the tolerance there; changed triangles may
        # exceed it only about as much as the full mesh nearby.
        for name, shape in (("frustum", cone(5, 2, 8)), ("cone", cone(5, 0, 10)),
                            ("turned cylinder", cylinder(5, 10).rotate(37, (1, 2, 3))),
                            ("tapered prism", regular_polygon(5, 6).extrude(8, taper=10)),
                            ("twisted bar", box((4, 4, 30)).twist(6))):
            with self.subTest(shape=name):
                graph, node, full, simple = self.meshes(shape, 0.1)
                ratios = local_excess(graph, [node], corners_of(simple), corners_of(full), TOLERANCE)
                self.assertLessEqual(float(ratios.max()), LOCAL_EXCESS)

    def test_native_meshes_report_their_stats_before_copying(self):
        graph = radsdf.Graph(gear())
        self.addCleanup(graph.close)
        node = graph.roots[0]
        full = graph.mesh(node, cell_size=0.2)
        native = graph.native_mesh(node, cell_size=0.2)
        self.assertEqual(native.stats["triangle_count"], len(full.triangles))
        simple = native.mesh(TOLERANCE)
        self.assertTrue(native.closed)
        self.assertEqual(digest(simple), digest(graph.mesh(node, cell_size=0.2, simplify=TOLERANCE)))
        self.assertEqual(simple.stats["unsimplified_triangles"], len(full.triangles))
        with self.assertRaisesRegex(SDFError, "freed"):
            native.mesh()
        native.close()
        with graph.native_mesh(node, cell_size=0.2) as unused:
            self.assertFalse(unused.closed)
        self.assertTrue(unused.closed)
        self.assertEqual(digest(graph.native_mesh(node, cell_size=0.2).mesh()), digest(full))
        # Simplifying needs the graph; a closed one fails cleanly and still frees the mesh.
        other = radsdf.Graph(sphere(2))
        pending = other.native_mesh(other.roots[0], cell_size=0.2)
        other.close()
        with self.assertRaisesRegex(SDFError, "closed"):
            pending.mesh(TOLERANCE)
        self.assertTrue(pending.closed)

    def test_results_do_not_depend_on_the_thread_count(self):
        for name, shape, cell in (("plate", plate_with_holes(4, 2), 0.1), ("gear", gear(), 0.1),
                                  ("rotated box", box((20, 10, 5)).rotate(30, (1, 2, 3)), 0.1)):
            with self.subTest(shape=name):
                digests = {digest(self.meshes(shape, cell, threads=threads)[3]) for threads in (1, 2, 3, 8)}
                self.assertEqual(len(digests), 1)

    def test_explicit_bounds_clip_and_the_simplified_mesh_stays_closed(self):
        shape = sphere(10)
        bounds = ((-20, -20, -20), (20, 20, 4))
        graph, node, full, simple = self.meshes(shape, 0.2, bounds=bounds)
        self.assertLess(len(simple.triangles), 0.3 * len(full.triangles))
        assert_closed_and_outward(self, simple.positions, simple.triangles)
        self.assertAlmostEqual(float(simple.positions[:, 2].max()), 4, places=4)
        self.assert_creases_kept(full, simple)
        # The flat cut keeps its crease circle as closely as the full mesh does.
        def rim(mesh):
            cap = mesh.creases[np.all(np.abs(mesh.creases[:, :, 2] - 4) < 1e-4, axis=1)]
            return np.abs(np.linalg.norm(cap[:, :, :2].reshape(-1, 2), axis=1) - math.sqrt(84))
        self.assertGreater(len(rim(simple)), 40)
        self.assertLessEqual(rim(simple).max(), max(TOLERANCE, rim(full).max()))

    def test_a_larger_tolerance_simplifies_further_and_zero_keeps_the_mesh(self):
        graph = radsdf.Graph(gear())
        self.addCleanup(graph.close)
        node = graph.roots[0]
        full = graph.mesh(node, cell_size=0.1)
        counts = [len(graph.mesh(node, cell_size=0.1, simplify=tolerance).triangles)
                  for tolerance in (0.002, 0.01, 0.05)]
        self.assertGreater(len(full.triangles), counts[0])
        self.assertGreater(counts[0], counts[1])
        self.assertGreater(counts[1], counts[2])
        self.assertEqual(digest(graph.mesh(node, cell_size=0.1, simplify=0)), digest(full))
        coarse = graph.mesh(node, cell_size=0.1, simplify=0.05)
        self.assertLessEqual(float(deviation(graph, node, coarse).max()),
                             max(0.05, float(deviation(graph, node, full).max())) * 1.001)

    def test_non_manifold_regions_stay_as_they_are(self):
        # A wall thinner than a cell pinches into non-manifold edges and pieces.
        graph, node, full, simple = self.meshes(box((10, 0.06, 10)).rotate(33, (1, 2, 3)), 0.1)
        self.assertGreater(full.stats["nonmanifold_edges"], 100)
        for key in ("nonmanifold_edges", "boundary_edges", "components"):
            self.assertEqual(simple.stats[key], full.stats[key], key)
        self.assertLess(len(simple.triangles), len(full.triangles))
        self.assertAlmostEqual(simple.stats["volume"], full.stats["volume"], delta=0.01 * full.stats["volume"])

    def test_simplified_meshes_simplify_again(self):
        graph = radsdf.Graph(plate_with_holes(4, 2))
        self.addCleanup(graph.close)
        node = graph.roots[0]
        library = radsdf._native()
        options = radsdf._MeshOptions(cell_size=0.1, crease_angle=35, threads=0, refine_steps=6)
        simplify = radsdf._SimplifyOptions(tolerance=TOLERANCE, threads=0)
        with graph._live() as handle:
            built = library.rsdf_mesh_build(handle, node, ctypes.byref(options))
            once = library.rsdf_mesh_simplify(handle, node, built, ctypes.byref(simplify))
            twice = library.rsdf_mesh_simplify(handle, node, once, ctypes.byref(simplify))
            self.assertTrue(built and once and twice)
            counts = []
            for mesh in (built, once, twice):
                stats = radsdf._MeshStats()
                library.rsdf_mesh_get_stats(mesh, ctypes.byref(stats))
                counts.append((stats.triangle_count, stats.boundary_edges, stats.nonmanifold_edges))
                library.rsdf_mesh_free(mesh)
        self.assertGreater(counts[0][0], counts[1][0])
        self.assertGreaterEqual(counts[1][0], counts[2][0])
        self.assertEqual([count[1:] for count in counts], [(0, 0)] * 3)

    def test_the_c_api_reports_invalid_requests(self):
        graph = radsdf.Graph([sphere(2), box(2)])
        self.addCleanup(graph.close)
        twin = radsdf.Graph([sphere(2), box(2)])
        self.addCleanup(twin.close)
        library = radsdf._native()
        options = radsdf._MeshOptions(cell_size=0.2, crease_angle=35, threads=0, refine_steps=6)
        with graph._live() as handle:
            built = library.rsdf_mesh_build(handle, graph.roots[0], ctypes.byref(options))
            try:
                cases = [(graph.roots[1], built, 0.01, "built from node"), (graph.roots[0], built, 0.0, "tolerance"),
                         (graph.roots[0], built, -1.0, "tolerance"), (graph.roots[0], built, math.nan, "tolerance"),
                         (graph.roots[0], built, math.inf, "tolerance"), (graph.roots[0], None, 0.01, "mesh is null")]
                for node, mesh, tolerance, message in cases:
                    with self.subTest(message=message, tolerance=tolerance):
                        result = library.rsdf_mesh_simplify(handle, node, mesh, ctypes.byref(
                            radsdf._SimplifyOptions(tolerance=tolerance)))
                        self.assertFalse(result)
                        self.assertIn(message, library.rsdf_last_error().decode())
                self.assertFalse(library.rsdf_mesh_simplify(handle, graph.roots[0], built, None))
                self.assertIn("options are null", library.rsdf_last_error().decode())
                # The same nodes in another graph: the mesh belongs to its own graph.
                with twin._live() as other:
                    self.assertFalse(library.rsdf_mesh_simplify(other, graph.roots[0], built, ctypes.byref(
                        radsdf._SimplifyOptions(tolerance=0.01))))
                    self.assertIn("another graph", library.rsdf_last_error().decode())
            finally:
                library.rsdf_mesh_free(built)
        with self.assertRaisesRegex(SDFError, "simplify must not be negative"):
            sphere(1).mesh(0.1, simplify=-0.1)

    def test_an_empty_solid_simplifies_to_an_empty_mesh(self):
        empty = box(1) & box(1, center=(5, 0, 0))
        graph = radsdf.Graph(empty)
        self.addCleanup(graph.close)
        mesh = graph.mesh(empty, cell_size=0.1, bounds=((-1, -1, -1), (6, 1, 1)), simplify=TOLERANCE)
        self.assertEqual(len(mesh.triangles), 0)
        self.assertEqual(mesh.stats["unsimplified_triangles"], 0)

    def test_benchmark(self):
        rows = []
        for name, shape, cell in (("sphere r10", sphere(10), 0.05), ("gear 24t", gear(), 0.06),
                                  ("enclosure", enclosure(), 0.07)):
            graph = radsdf.Graph(shape)
            self.addCleanup(graph.close)
            start = time.perf_counter()
            full = graph.mesh(shape, cell_size=cell)
            meshed = time.perf_counter()
            simple = graph.mesh(shape, cell_size=cell, simplify=TOLERANCE)
            simplified = time.perf_counter() - meshed - (meshed - start)
            rows.append((name, len(full.triangles), len(simple.triangles), meshed - start, simplified))
            # About a second per million triangles leaves room for slower machines.
            self.assertLess(simplified, 5 * max(1.0, len(full.triangles) / 1e6))
        print()
        for name, before, after, mesh_seconds, seconds in rows:
            print(f"  simplify {name}: {before:,} -> {after:,} triangles ({after / before:.2%}) in "
                  f"{seconds * 1000:.0f} ms ({seconds / before * 1e9:.0f} ns per triangle; meshing "
                  f"{mesh_seconds * 1000:.0f} ms)")


@unittest.skipUnless(kernel_available(), "the native SDF kernel is not available")
class ExportToleranceTests(unittest.TestCase):
    FIXTURE = """
result = [box((40, 30, 4)) - union(*[cylinder(2, 10, center=(x, y, 0)) for x in (-12, 0, 12) for y in (-8, 8)]),
          sphere(6, center=(40, 0, 0))]
"""

    def setUp(self):
        self.result, self.output = build(self, self.FIXTURE)

    def export(self, **fields):
        report, folder = native(self.output, self.result, "export", **fields)
        corners, _ = read_3mf(folder / "model.3mf")
        return report, corners

    def test_exports_are_simplified_by_default_and_report_both_counts(self):
        report, corners = self.export()
        self.assertEqual(report["tolerance"], min(sdf_ops.EXPORT_TOLERANCE,
                                                  sdf_ops.EXPORT_TOLERANCE_CELLS * report["cellSize"]))
        self.assertEqual(report["triangles"], len(corners))
        self.assertLess(report["triangles"], 0.1 * report["meshedTriangles"])
        full_report, full = self.export(tolerance=0)
        self.assertEqual(full_report["tolerance"], 0)
        self.assertEqual(full_report["triangles"], full_report["meshedTriangles"])
        self.assertEqual(full_report["meshedTriangles"], report["meshedTriangles"])
        self.assertEqual(full_report["cellSize"], report["cellSize"])
        self.assertAlmostEqual(signed_volume(corners) / signed_volume(full), 1, delta=2e-3)
        expected = sum(stats["volume"] for stats in self.result["partStats"])
        self.assertAlmostEqual(signed_volume(corners) / expected, 1, delta=3e-3)
        # Every written triangle's centroid stays within the tolerance of its
        # part's field, and densely sampled, no triangle deviates much beyond
        # the full export mesh nearby.
        graph = radsdf.Graph.from_bytes((self.output / "model.rsdf").read_bytes(), self.result["partRoots"])
        self.addCleanup(graph.close)
        points = corners.mean(axis=1)
        distance = np.min([np.abs(graph.evaluate(node, points)) for node in graph.roots], axis=0)
        self.assertLessEqual(float(distance.max()), report["tolerance"])
        ratios = local_excess(graph, graph.roots, corners, full, report["tolerance"])
        self.assertLessEqual(float(ratios.max()), LOCAL_EXCESS)

    def test_explicit_tolerances_and_individual_files(self):
        coarse, _ = self.export(tolerance=0.05)
        default, _ = self.export()
        self.assertEqual(coarse["tolerance"], min(0.05, sdf_ops.EXPORT_TOLERANCE_CELLS * coarse["cellSize"]))
        self.assertGreater(coarse["tolerance"], default["tolerance"])
        self.assertLess(coarse["triangles"], default["triangles"])
        report, folder = native(self.output, self.result, "export", individualParts=[[1], [0]], tolerance=0.02)
        self.assertEqual(report["tolerance"], 0.02)
        total = sum(len(read_3mf(folder / record["path"])[0]) for record in report["partFiles"])
        self.assertEqual(total, report["triangles"])

    def test_tolerances_are_capped_at_a_quarter_of_the_cell(self):
        for value in (1, 1e6):
            with self.subTest(tolerance=value):
                report, _ = self.export(tolerance=value)
                self.assertEqual(report["tolerance"], sdf_ops.EXPORT_TOLERANCE_CELLS * report["cellSize"])

    def test_invalid_tolerances_are_refused_before_meshing(self):
        for value in (-0.01, "0.01", True, [0.01], 1e309):
            with self.subTest(value=value):
                with (mock.patch.object(sdf_ops, "closed_meshes", side_effect=AssertionError("No meshing")),
                      self.assertRaisesRegex(sdf_build.ModelError, "tolerance")):
                    native(self.output, self.result, "export", tolerance=value)

    def test_the_triangle_budget_counts_the_meshes_before_simplification(self):
        report, _ = self.export()
        budget = report["meshedTriangles"] // 2
        self.assertGreater(budget, report["triangles"])
        copied, copy = [], radsdf.NativeMesh.mesh

        def counted(native, simplify=None):
            copied.append((native.stats["triangle_count"], simplify))
            return copy(native, simplify)

        with (mock.patch.object(sdf_ops, "EXPORT_TRIANGLE_BUDGET", budget),
              mock.patch.object(sdf_ops, "budget_cell", side_effect=lambda source, solids, cell, *rest: cell),
              mock.patch.object(radsdf.NativeMesh, "mesh", counted)):
            limited, _ = self.export()
        self.assertLessEqual(limited["meshedTriangles"], budget)
        self.assertGreater(limited["cellSize"], report["cellSize"])
        # Passes over the budget simplify nothing: one simplification per part.
        self.assertEqual(len(copied), 2)
        self.assertEqual(sum(count for count, _ in copied), limited["meshedTriangles"])
        self.assertTrue(all(simplify == limited["tolerance"] for _, simplify in copied))


@unittest.skipUnless(kernel_available(), "the native SDF kernel is not available")
class ThinWallExportTests(unittest.TestCase):
    def test_large_tolerances_keep_both_sides_of_thin_walls_apart(self):
        # A 0.1 mm spherical shell meshes two export cells (0.05 mm) thick: a
        # tolerance above half the wall used to let its two sides cross.
        result, output = build(self, "result = sphere(6).shell(0.1)")
        report, folder = native(output, result, "export", tolerance=1)
        self.assertEqual(report["tolerance"], sdf_ops.EXPORT_TOLERANCE_CELLS * report["cellSize"])
        self.assertLess(report["tolerance"], 0.025)
        self.assertLess(report["triangles"], 0.1 * report["meshedTriangles"])
        corners, _ = read_3mf(folder / "model.3mf")
        a, b, c = corners[:, 0], corners[:, 1], corners[:, 2]
        points = np.concatenate([(i * a + j * b + (6 - i - j) * c) / 6 for i in range(7) for j in range(7 - i)])
        # Every sample stays on its own side of the wall's middle (radius 5.95).
        self.assertGreater(float(np.abs(np.linalg.norm(points, axis=1) - 5.95).min()), 0.025)


@unittest.skipUnless(os.environ.get("RAD_RUN_SANITIZER_FUZZ") == "1" and shutil.which("clang++") and kernel_available(),
                     "Set RAD_RUN_SANITIZER_FUZZ=1 (needs clang++) to run simplification under sanitizers.")
class SimplifySanitizerTests(unittest.TestCase):
    """Parallel simplification of crease-heavy meshes in a native driver built
    with ThreadSanitizer, and with AddressSanitizer and
    UndefinedBehaviorSanitizer."""
    SHAPES = (("frustum", cone(5, 2, 8), 0.1), ("gear", gear(), 0.15),
              ("double cone", cone(3, 0, 10) | cone(3, 0, 10).rotate(180, (1, 0, 0)), 0.05),
              ("knife edges", box((6, 6, 6)) & box((6, 6, 6)).rotate(45, (1, 1, 0)), 0.08))

    def run_driver(self, flags, environment):
        kernel = Path(__file__).resolve().parents[2] / "Sources" / "RADSDFKernel"
        with tempfile.TemporaryDirectory() as directory:
            folder = Path(directory)
            driver = folder / "simplify_driver"
            built = subprocess.run(["clang++", "-std=c++17", "-O1", "-g", "-fno-math-errno", "-fno-omit-frame-pointer",
                                    *flags, "-I", str(kernel / "include"), "-I", str(kernel),
                                    *map(str, sorted(kernel.glob("*.cpp"))),
                                    str(Path(__file__).with_name("simplify_sanitizer_driver.cpp")), "-o", str(driver)],
                                   capture_output=True, text=True)
            self.assertEqual(built.returncode, 0, built.stderr[-4000:])
            arguments = []
            for index, (_, shape, cell) in enumerate(self.SHAPES):
                data, roots = radsdf.serialize([shape])
                (folder / f"shape-{index}.rsdf").write_bytes(data)
                arguments += [str(folder / f"shape-{index}.rsdf"), str(roots[0]), str(cell)]
            run = subprocess.run([str(driver), *arguments], capture_output=True, text=True,
                                 env={**os.environ, **environment})
            report = run.stdout[-2000:] + run.stderr[-6000:]
            self.assertEqual(run.returncode, 0, report)
            for marker in ("ThreadSanitizer", "AddressSanitizer", "runtime error"):
                self.assertNotIn(marker, run.stderr, report)

    def test_parallel_simplification_has_no_data_races(self):
        self.run_driver(["-fsanitize=thread"], {"TSAN_OPTIONS": "halt_on_error=1"})

    def test_simplification_has_no_memory_or_arithmetic_faults(self):
        self.run_driver(["-fsanitize=address,undefined", "-fno-sanitize-recover=undefined"],
                        {"ASAN_OPTIONS": "detect_leaks=0:abort_on_error=0"})


if __name__ == "__main__":
    unittest.main()
