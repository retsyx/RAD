"""Native SDF kernel (libRADSDF) checks through its C API.

RSDF graphs are written directly with struct, independently of any Python
binding module. The library comes from RAD_SDF_LIBRARY or the SwiftPM build
directory (`swift build --product RADSDF -c release`).
"""
import ctypes
import math
import os
from pathlib import Path
import struct
import time
import unittest

import numpy as np

ROOT = Path(__file__).resolve().parents[2]

SPHERE, BOX, CYLINDER, CONE, CAPSULE, TORUS, ELLIPSOID, HALFSPACE = 1, 2, 3, 4, 5, 6, 7, 8
TRANSFORM, STRETCH, MIRROR, TWIST, ELONGATE, OFFSET, SHELL = 20, 21, 22, 23, 24, 25, 26
UNION, INTERSECTION, DIFFERENCE = 30, 31, 32
EXTRUDE, TAPERED_EXTRUDE, REVOLVE, LOFT = 40, 41, 42, 43
CIRCLE, RECT, POLYGON, SEGMENT, ELLIPSE, HALFPLANE = 100, 101, 102, 103, 104, 105
TRANSFORM2, STRETCH2, MIRROR2, OFFSET2, OUTLINE2 = 120, 121, 122, 123, 124
UNION2, INTERSECTION2, DIFFERENCE2 = 130, 131, 132

IDENTITY = (1, 0, 0, 0, 1, 0, 0, 0, 1)


def library_path():
    if os.environ.get("RAD_SDF_LIBRARY"):
        return Path(os.environ["RAD_SDF_LIBRARY"])
    # Both SwiftPM build systems link .build/<configuration> to their
    # products; the default one writes them to out/Products/<Configuration>.
    for folder in ("release", "out/Products/Release", "debug", "out/Products/Debug"):
        candidate = ROOT / ".build" / folder / "libRADSDF.dylib"
        if candidate.exists():
            return candidate
    return None


class MeshOptions(ctypes.Structure):
    _fields_ = [("cell_size", ctypes.c_double), ("crease_angle", ctypes.c_double), ("max_cells", ctypes.c_uint64),
                ("threads", ctypes.c_int32), ("refine_steps", ctypes.c_int32),
                ("bounds_lo", ctypes.c_double * 3), ("bounds_hi", ctypes.c_double * 3),
                ("bounds_valid", ctypes.c_int32), ("flags", ctypes.c_uint32)]


class MeshStats(ctypes.Structure):
    _fields_ = [(name, ctypes.c_uint64) for name in (
        "vertex_count", "triangle_count", "crease_segment_count", "active_cells", "boundary_edges",
        "nonmanifold_edges", "components", "cavities")] + [
        ("volume", ctypes.c_double), ("area", ctypes.c_double), ("cavity_volume", ctypes.c_double),
        ("centroid", ctypes.c_double * 3),
        ("bounds_lo", ctypes.c_double * 3), ("bounds_hi", ctypes.c_double * 3), ("cell_size", ctypes.c_double),
        ("seconds", ctypes.c_double)]


class KernelError(Exception):
    pass


class Kernel:
    """Thin ctypes wrapper over radsdf.h."""

    def __init__(self, path):
        lib = self.lib = ctypes.CDLL(str(path))
        p, u32, u64, i32, dbl = ctypes.c_void_p, ctypes.c_uint32, ctypes.c_uint64, ctypes.c_int32, ctypes.c_double
        signatures = {
            "rsdf_abi_version": (u32, []), "rsdf_kernel_version": (ctypes.c_char_p, []),
            "rsdf_last_error": (ctypes.c_char_p, []),
            "rsdf_graph_create": (p, [ctypes.c_char_p, ctypes.c_size_t]), "rsdf_graph_free": (None, [p]),
            "rsdf_graph_node_count": (u32, [p]), "rsdf_node_dimension": (ctypes.c_int, [p, u32]),
            "rsdf_bounds": (ctypes.c_int, [p, u32, p, p, p]),
            "rsdf_evaluate": (ctypes.c_int, [p, u32, p, u64, p, i32]),
            "rsdf_gradient": (ctypes.c_int, [p, u32, p, u64, dbl, p, i32]),
            "rsdf_raycast": (ctypes.c_int, [p, u32, p, p, u64, dbl, dbl, i32, p, p, i32]),
            "rsdf_mesh_build": (p, [p, u32, ctypes.POINTER(MeshOptions)]), "rsdf_mesh_free": (None, [p]),
            "rsdf_mesh_get_stats": (None, [p, ctypes.POINTER(MeshStats)]),
            "rsdf_mesh_positions": (ctypes.POINTER(ctypes.c_float), [p]),
            "rsdf_mesh_normals": (ctypes.POINTER(ctypes.c_float), [p]),
            "rsdf_mesh_triangles": (ctypes.POINTER(ctypes.c_uint32), [p]),
            "rsdf_mesh_creases": (ctypes.POINTER(ctypes.c_float), [p]),
        }
        for name, (restype, argtypes) in signatures.items():
            function = getattr(lib, name)
            function.restype, function.argtypes = restype, argtypes

    def error(self):
        return self.lib.rsdf_last_error().decode()

    def create(self, data):
        handle = self.lib.rsdf_graph_create(data, len(data))
        if not handle:
            raise KernelError(self.error())
        return handle

    def check(self, status):
        if status != 0:
            raise KernelError(self.error())

    def evaluate(self, handle, node, points, threads=0):
        points = np.ascontiguousarray(points, dtype=np.float64)
        out = np.empty(len(points))
        self.check(self.lib.rsdf_evaluate(handle, node, points.ctypes.data, len(points), out.ctypes.data, threads))
        return out

    def gradient(self, handle, node, points, eps=0.0, threads=0):
        points = np.ascontiguousarray(points, dtype=np.float64)
        out = np.empty((len(points), 3))
        self.check(self.lib.rsdf_gradient(handle, node, points.ctypes.data, len(points), eps, out.ctypes.data,
                                          threads))
        return out

    def raycast(self, handle, node, origins, directions, max_distance=1e3, tolerance=1e-6, max_steps=512):
        origins = np.ascontiguousarray(origins, dtype=np.float64)
        directions = np.ascontiguousarray(directions, dtype=np.float64)
        hits = np.zeros(len(origins), dtype=np.uint8)
        distances = np.zeros(len(origins))
        self.check(self.lib.rsdf_raycast(handle, node, origins.ctypes.data, directions.ctypes.data, len(origins),
                                         max_distance, tolerance, max_steps, hits.ctypes.data,
                                         distances.ctypes.data, 0))
        return hits.astype(bool), distances

    def bounds(self, handle, node):
        lo, hi, lipschitz = np.full(3, np.nan), np.full(3, np.nan), ctypes.c_double()
        self.check(self.lib.rsdf_bounds(handle, node, lo.ctypes.data, hi.ctypes.data, ctypes.byref(lipschitz)))
        dimension = self.lib.rsdf_node_dimension(handle, node)
        return lo[:dimension], hi[:dimension], lipschitz.value

    def mesh(self, handle, node, cell_size, threads=0, bounds=None, max_cells=0, crease_angle=0.0, preview=False):
        options = MeshOptions(cell_size=cell_size, crease_angle=crease_angle, max_cells=max_cells, threads=threads,
                              flags=1 if preview else 0)
        if bounds is not None:
            options.bounds_lo[:] = bounds[0]
            options.bounds_hi[:] = bounds[1]
            options.bounds_valid = 1
        mesh = self.lib.rsdf_mesh_build(handle, node, ctypes.byref(options))
        if not mesh:
            raise KernelError(self.error())
        try:
            stats = MeshStats()
            self.lib.rsdf_mesh_get_stats(mesh, ctypes.byref(stats))

            def array(pointer, count, width):
                return np.ctypeslib.as_array(pointer, shape=(count, width)).copy() if count else np.zeros((0, width))

            return Mesh(stats, array(self.lib.rsdf_mesh_positions(mesh), stats.vertex_count, 3),
                        array(self.lib.rsdf_mesh_normals(mesh), stats.vertex_count, 3),
                        array(self.lib.rsdf_mesh_triangles(mesh), stats.triangle_count, 3).astype(np.int64),
                        array(self.lib.rsdf_mesh_creases(mesh), stats.crease_segment_count, 6))
        finally:
            self.lib.rsdf_mesh_free(mesh)


class Mesh:
    def __init__(self, stats, positions, normals, triangles, creases):
        self.stats, self.positions, self.normals, self.triangles, self.creases = (
            stats, positions, normals, triangles, creases)


class Graph:
    """Builds RSDF version 1 bytes."""

    def __init__(self):
        self.records = []

    def add(self, op, children=(), params=()):
        self.records.append((op, list(children), [float(v) for v in params]))
        return len(self.records) - 1

    def to_bytes(self):
        out = [b"RSDF", struct.pack("<III", 1, len(self.records), 0)]
        for op, children, params in self.records:
            out.append(struct.pack("<IIII", op, len(children), len(params), 0))
            out.append(struct.pack(f"<{len(children)}I", *children))
            out.append(struct.pack(f"<{len(params)}d", *params))
        return b"".join(out)

    # Convenience constructors.
    def move(self, child, offset, rotation=IDENTITY, scale=1.0):
        return self.add(TRANSFORM, [child], [*np.ravel(rotation), *offset, scale])

    def move2(self, child, offset, rotation=(1, 0, 0, 1), scale=1.0):
        return self.add(TRANSFORM2, [child], [*np.ravel(rotation), *offset, scale])


def rotation_matrix(axis, degrees):
    axis = np.asarray(axis, dtype=float) / np.linalg.norm(axis)
    x, y, z = axis
    c, s = math.cos(math.radians(degrees)), math.sin(math.radians(degrees))
    t = 1 - c
    return np.array([[t * x * x + c, t * x * y - s * z, t * x * z + s * y],
                     [t * x * y + s * z, t * y * y + c, t * y * z - s * x],
                     [t * x * z - s * y, t * y * z + s * x, t * z * z + c]])


# ------------------------------------------------------------- reference fields

def length(v):
    return np.sqrt(np.sum(v * v, axis=1))


def box_ref(p, half, r=0.0):
    q = np.abs(p) - (np.asarray(half) - r)
    return length(np.maximum(q, 0)) + np.minimum(q.max(axis=1), 0) - r


def segment_distance(p, a, b):
    ab = b - a
    t = np.clip(((p - a) @ ab) / (ab @ ab), 0, 1)
    return length(p - a - t[:, None] * ab)


def polygon_ref(p, poly):
    """Brute-force signed distance: nearest edge, even-odd inside test."""
    poly = np.asarray(poly, dtype=float)
    d = np.min([segment_distance(p, poly[i], poly[(i + 1) % len(poly)]) for i in range(len(poly))], axis=0)
    inside = np.zeros(len(p), dtype=bool)
    for i in range(len(poly)):
        a, b = poly[i], poly[(i + 1) % len(poly)]
        straddle = (a[1] > p[:, 1]) != (b[1] > p[:, 1])
        with np.errstate(divide="ignore", invalid="ignore"):
            cross_x = a[0] + (p[:, 1] - a[1]) * (b[0] - a[0]) / (b[1] - a[1])
        inside ^= straddle & (p[:, 0] < cross_x)
    return np.where(inside, -d, d)


def slab_ref(d, w):
    return np.minimum(np.maximum(d, w), 0) + np.hypot(np.maximum(d, 0), np.maximum(w, 0))


def smin_ref(a, b, k):
    if k == 0:
        return np.minimum(a, b)
    h = np.maximum(k - np.abs(a - b), 0) / k
    return np.minimum(a, b) - h * h * k / 4


def smax_ref(a, b, k):
    return -smin_ref(-a, -b, k)


def star_points(points=5, outer=10.0, inner=4.0, phase=0.3):
    angles = phase + np.arange(2 * points) * math.pi / points
    radii = np.where(np.arange(2 * points) % 2 == 0, outer, inner)
    return np.column_stack([radii * np.cos(angles), radii * np.sin(angles)])


def gear_points(teeth=40, root=18.0, tip=20.0):
    """A gear-like outline: trapezoidal teeth between root and tip circles."""
    points = []
    pitch = 2 * math.pi / teeth
    for tooth in range(teeth):
        start = tooth * pitch
        for fraction, radius in ((0.0, root), (0.18, tip), (0.42, tip), (0.6, root)):
            angle = start + fraction * pitch
            points.append((radius * math.cos(angle), radius * math.sin(angle)))
    return np.array(points)


def shoelace(poly):
    x, y = poly[:, 0], poly[:, 1]
    return 0.5 * abs(np.sum(x * np.roll(y, -1) - np.roll(x, -1) * y))


def perimeter(poly):
    return float(np.sum(length(np.roll(poly, -1, axis=0) - poly)))


KERNEL = None


def setUpModule():
    global KERNEL
    path = library_path()
    if path is None:
        raise unittest.SkipTest("libRADSDF.dylib not built (swift build --product RADSDF -c release)")
    KERNEL = Kernel(path)


class KernelCase(unittest.TestCase):
    def setUp(self):
        self.k = KERNEL
        self.handles = []
        self.rng = np.random.default_rng(1234)

    def tearDown(self):
        for handle in self.handles:
            self.k.lib.rsdf_graph_free(handle)

    def build(self, graph):
        handle = self.k.create(graph.to_bytes())
        self.handles.append(handle)
        return handle

    def points(self, count, scale, dimension=3):
        return self.rng.uniform(-scale, scale, size=(count, dimension))

    def assertField(self, graph, node, points, expected, tolerance=1e-12):
        actual = self.k.evaluate(self.build(graph), node, points)
        np.testing.assert_allclose(actual, expected, rtol=0, atol=tolerance * max(1.0, np.max(np.abs(expected))))


class EvaluationTests(KernelCase):
    def test_abi_and_version(self):
        self.assertEqual(self.k.lib.rsdf_abi_version(), 2)
        self.assertTrue(self.k.lib.rsdf_kernel_version())

    def test_3d_primitives_match_analytic_values(self):
        p = self.points(4000, 12)
        a, b = np.array([-3.0, 1.0, 2.0]), np.array([4.0, -2.0, 5.0])
        n = np.array([1.0, 2.0, -2.0]) / 3
        cases = [
            (SPHERE, [5], length(p) - 5),
            (BOX, [5, 3, 2, 0], box_ref(p, [5, 3, 2])),
            (BOX, [5, 3, 2, 1.5], box_ref(p, [5, 3, 2], 1.5)),
            (CYLINDER, [4, 3], slab_ref(np.hypot(p[:, 0], p[:, 1]) - 4, np.abs(p[:, 2]) - 3)),
            (CAPSULE, [*a, *b, 1.5], segment_distance(p, a, b) - 1.5),
            (TORUS, [6, 2], np.hypot(np.hypot(p[:, 0], p[:, 1]) - 6, p[:, 2]) - 2),
            (ELLIPSOID, [5, 3, 2], (length(p / [5, 3, 2]) - 1) * 2),
            (HALFSPACE, [*n, 1.5], p @ n - 1.5),
        ]
        for op, params, expected in cases:
            with self.subTest(op=op, params=params):
                graph = Graph()
                self.assertField(graph, graph.add(op, [], params), p, expected)

    def test_cone_is_exact(self):
        # The cone is the revolution of a trapezoid; its exact distance is the
        # 2D distance in the (radius, z) half plane to the cap and side edges.
        p = self.points(4000, 10)
        r1, r2, h = 5.0, 2.0, 4.0
        q = np.column_stack([np.hypot(p[:, 0], p[:, 1]), p[:, 2]])
        corners = np.array([[0, -h], [r1, -h], [r2, h], [0, h]])
        d = np.min([segment_distance(q, corners[i], corners[i + 1]) for i in range(3)], axis=0)
        inside = polygon_ref(q, corners) < 0
        graph = Graph()
        self.assertField(graph, graph.add(CONE, [], [r1, r2, h]), p, np.where(inside, -d, d))
        apex = Graph()  # one radius may be zero
        apex_node = apex.add(CONE, [], [0, 3, 2])
        q = np.column_stack([np.hypot(p[:, 0], p[:, 1]), p[:, 2]])
        corners = np.array([[0, -2], [0, -2], [3, 2], [0, 2]])
        d = np.minimum(segment_distance(q, corners[1], corners[2]), segment_distance(q, corners[2], corners[3]))
        inside = polygon_ref(q, corners[1:]) < 0
        self.assertField(apex, apex_node, p, np.where(inside, -d, d))

    def test_3d_operations(self):
        p = self.points(4000, 12)
        rotation = rotation_matrix([1, 2, 3], 37)
        offset, scale = np.array([1.0, -2.0, 0.5]), 1.7
        local = ((p - offset) @ rotation) / scale
        sphere = length(p) - 4
        graph = Graph()
        ball = graph.add(SPHERE, [], [4])
        block = graph.add(BOX, [], [3, 2, 1, 0])
        cases = [
            (graph.add(TRANSFORM, [block], [*rotation.ravel(), *offset, scale]), box_ref(local, [3, 2, 1]) * scale),
            (graph.add(STRETCH, [ball], [2, 1, 0.5]), (length(p / [2, 1, 0.5]) - 4) * 0.5),
            (graph.add(MIRROR, [graph.move(ball, [5, 0, 0])], [1, 0, 0, 1]), length(p - [-3, 0, 0]) - 4),
            (graph.add(ELONGATE, [ball], [2, 0, 1]), length(p - np.clip(p, [-2, 0, -1], [2, 0, 1])) - 4),
            (graph.add(OFFSET, [ball], [1.5]), sphere - 1.5),
            (graph.add(SHELL, [ball], [1, 0]), np.maximum(sphere, -sphere - 1)),
            (graph.add(SHELL, [ball], [1, 1]), np.maximum(-sphere, sphere - 1)),
            (graph.add(SHELL, [ball], [1, 2]), np.abs(sphere) - 0.5),
        ]
        rate = 0.15
        angle = -rate * p[:, 2]
        twisted = np.column_stack([np.cos(angle) * p[:, 0] - np.sin(angle) * p[:, 1],
                                   np.sin(angle) * p[:, 0] + np.cos(angle) * p[:, 1], p[:, 2]])
        rho = math.hypot(3, 2)
        cases.append((graph.add(TWIST, [block], [rate]),
                      box_ref(twisted, [3, 2, 1]) / math.sqrt(1 + (rate * rho) ** 2)))
        c1, c2 = graph.move(ball, [2, 0, 0]), graph.move(graph.add(SPHERE, [], [3]), [-3, 1, 0])
        c3 = graph.move(graph.add(SPHERE, [], [2]), [0, 0, 3])
        f1, f2, f3 = length(p - [2, 0, 0]) - 4, length(p - [-3, 1, 0]) - 3, length(p - [0, 0, 3]) - 2
        for k in (0.0, 1.5):
            cases += [
                (graph.add(UNION, [c1, c2, c3], [k]), smin_ref(smin_ref(f1, f2, k), f3, k)),
                (graph.add(INTERSECTION, [c1, c2, c3], [k]), smax_ref(smax_ref(f1, f2, k), f3, k)),
                (graph.add(DIFFERENCE, [c1, c2, c3], [k]), smax_ref(smax_ref(f1, -f2, k), -f3, k)),
            ]
        for node, expected in cases:
            with self.subTest(op=graph.records[node][0]):
                self.assertField(graph, node, p, expected)

    def test_profile_operations(self):
        p = self.points(4000, 12)
        radius = np.hypot(p[:, 0], p[:, 1])
        graph = Graph()
        disc, small = graph.add(CIRCLE, [], [4]), graph.add(CIRCLE, [], [2])
        z0, z1, taper = -1.0, 5.0, 1.5
        t = np.clip((p[:, 2] - z0) / (z1 - z0), 0, 1)
        w = np.abs(p[:, 2] - 2) - 3
        loft_c = math.hypot(1, math.hypot(8, 8) / (z1 - z0))  # D: diagonal of the profiles' box
        cases = [
            (graph.add(EXTRUDE, [disc], [z0, z1]), slab_ref(radius - 4, w)),
            (graph.add(TAPERED_EXTRUDE, [disc], [z0, z1, taper]),
             slab_ref((radius - 4 - taper * t) / math.sqrt(1 + (taper / (z1 - z0)) ** 2), w)),
            (graph.add(REVOLVE, [graph.move2(small, [6, 0])], [0]), np.hypot(radius - 6, p[:, 2]) - 2),
            (graph.add(REVOLVE, [small], [6]), np.hypot(radius - 6, p[:, 2]) - 2),
            (graph.add(LOFT, [disc, small], [z0, z1]),
             slab_ref(((1 - t) * (radius - 4) + t * (radius - 2)) / loft_c, w)),
        ]
        for node, expected in cases:
            with self.subTest(op=graph.records[node][0]):
                self.assertField(graph, node, p, expected)

    def test_2d_operations(self):
        p = self.points(4000, 12, dimension=2)
        star = star_points()
        graph = Graph()
        disc = graph.add(CIRCLE, [], [4])
        theta = math.radians(25)
        rotation = np.array([[math.cos(theta), -math.sin(theta)], [math.sin(theta), math.cos(theta)]])
        local = ((p - [1, -2]) @ rotation) / 1.5
        n = np.array([3.0, 4.0]) / 5
        f_disc = length(p) - 4
        f_rect = box_ref(p - [3, 0], [2, 1])
        rect = graph.move2(graph.add(RECT, [], [2, 1, 0]), [3, 0])
        cases = [
            (disc, f_disc),
            (graph.add(RECT, [], [5, 3, 1]), box_ref(p, [5, 3], 1)),
            (graph.add(POLYGON, [], star.ravel()), polygon_ref(p, star)),
            (graph.add(POLYGON, [], star[::-1].ravel()), polygon_ref(p, star)),  # clockwise input
            (graph.add(SEGMENT, [], [-3, 1, 4, 2, 1.5]),
             segment_distance(p, np.array([-3.0, 1.0]), np.array([4.0, 2.0])) - 1.5),
            (graph.add(ELLIPSE, [], [5, 2]), (length(p / [5, 2]) - 1) * 2),
            (graph.add(HALFPLANE, [], [*n, 2]), p @ n - 2),
            (graph.add(TRANSFORM2, [graph.add(RECT, [], [3, 2, 0.5])], [*rotation.ravel(), 1, -2, 1.5]),
             box_ref(local, [3, 2], 0.5) * 1.5),
            (graph.add(STRETCH2, [disc], [2, 0.5]), (length(p / [2, 0.5]) - 4) * 0.5),
            (graph.add(MIRROR2, [graph.move2(disc, [5, 0])], [0, 1, 1]), length(p - [5, 2]) - 4),
            (graph.add(OFFSET2, [disc], [-1]), f_disc + 1),
            (graph.add(OUTLINE2, [disc], [1]), np.abs(f_disc) - 0.5),
            (graph.add(UNION2, [disc, rect], [1]), smin_ref(f_disc, f_rect, 1)),
            (graph.add(INTERSECTION2, [disc, rect], [0]), np.maximum(f_disc, f_rect)),
            (graph.add(DIFFERENCE2, [disc, rect], [0.5]), smax_ref(f_disc, -f_rect, 0.5)),
        ]
        for node, expected in cases:
            with self.subTest(op=graph.records[node][0]):
                self.assertField(graph, node, p, expected)

    def test_polygon_branch_and_bound_matches_brute_force(self):
        gear = gear_points(teeth=60)
        graph = Graph()
        node = graph.add(POLYGON, [], gear[::-1].ravel())
        handle = self.build(graph)
        scattered = self.points(3000, 25, dimension=2)
        near = gear[self.rng.integers(0, len(gear), 3000)] + self.rng.normal(scale=0.2, size=(3000, 2))
        for p in (scattered, near):
            np.testing.assert_allclose(self.k.evaluate(handle, node, p), polygon_ref(p, gear), rtol=0, atol=1e-11)

    def test_indexed_children_match_the_linear_scan(self):
        # From 16 children a boolean finds candidate children through a
        # bounding volume hierarchy. Its fold must equal the same left fold
        # written as a 15-child scan followed by the last child, which here is
        # an empty solid (an inverted box) that still has a field.
        p = self.points(20000, 14)
        for op in (UNION, DIFFERENCE):
            for k in (0.0, 0.7):
                g = Graph()
                children = [g.move(g.add(SPHERE, [], [1 + 0.1 * i]), self.rng.uniform(-10, 10, 3)) for i in range(15)]
                ball = g.add(SPHERE, [], [1])
                empty = g.add(INTERSECTION, [ball, g.move(ball, [2.5, 0, 0])], [0])
                linear = g.add(op, [g.add(op, children, [k]), empty], [k])
                indexed = g.add(op, children + [empty], [k])
                handle = self.build(g)
                lo, hi, _ = self.k.bounds(handle, empty)
                self.assertGreater(lo[0], hi[0])
                with self.subTest(op=op, k=k):
                    self.assertTrue(np.array_equal(self.k.evaluate(handle, linear, p),
                                                   self.k.evaluate(handle, indexed, p)))
        g = Graph()  # unbounded children are always candidates
        cuts = [g.move(g.add(SPHERE, [], [1]), [x, 0, 0]) for x in range(-8, 9)]
        node = g.add(DIFFERENCE, [g.add(BOX, [], [10, 10, 10, 0]), *cuts, g.add(HALFSPACE, [], [0, 0, -1, -5])], [0])
        expected = np.max([box_ref(p, [10, 10, 10])] + [1 - length(p - [x, 0, 0]) for x in range(-8, 9)]
                          + [p[:, 2] - 5], axis=0)
        actual = self.k.evaluate(self.build(g), node, p)
        np.testing.assert_array_equal(np.sign(actual), np.sign(expected))
        self.assertTrue(np.all(actual <= expected + 1e-12))

    def test_deep_nesting(self):
        g = Graph()
        node = g.add(SPHERE, [], [1])
        for _ in range(100000):  # far deeper than a thread's call stack could recurse
            node = g.add(OFFSET, [node], [1e-4])
        p = self.points(1000, 15)
        np.testing.assert_allclose(self.k.evaluate(self.build(g), node, p), length(p) - 11, atol=1e-9)
        nested = Graph()
        node = nested.add(SPHERE, [], [1])
        for _ in range(30000):  # each transform keeps three point buffers alive
            node = nested.move(node, [0, 0, 1e-3])
        with self.assertRaisesRegex(KernelError, "point buffers"):
            self.build(nested)

    def test_whole_batch_culling_keeps_values_exact_near_surfaces(self):
        graph = Graph()
        holes = [graph.move(graph.add(CYLINDER, [], [0.6, 5]), [x, y, 0]) for x in range(-8, 9, 2)
                 for y in range(-8, 9, 2)]
        plate = graph.add(BOX, [], [10, 10, 1, 0])
        node = graph.add(DIFFERENCE, [plate, *holes], [0])
        ball = graph.add(UNION, [graph.move(graph.add(SPHERE, [], [1]), [x, 0, 0]) for x in range(-8, 9, 4)], [0])
        handle = self.build(graph)
        p = self.points(20000, 11) * [1, 1, 0.15]
        cylinders = [slab_ref(np.hypot(p[:, 0] - x, p[:, 1] - y) - 0.6, np.abs(p[:, 2]) - 5)
                     for x in range(-8, 9, 2) for y in range(-8, 9, 2)]
        expected = np.max([box_ref(p, [10, 10, 1])] + [-c for c in cylinders], axis=0)
        actual = self.k.evaluate(handle, node, p)
        np.testing.assert_array_equal(np.sign(actual), np.sign(expected))
        near = np.abs(expected) < 0.3
        np.testing.assert_allclose(actual[near], expected[near], atol=1e-12)
        self.assertTrue(np.all(actual <= expected + 1e-12))  # a distance bound
        spheres = np.min([length(p - [x, 0, 0]) - 1 for x in range(-8, 9, 4)], axis=0)
        union = self.k.evaluate(handle, ball, p)
        np.testing.assert_array_equal(np.sign(union), np.sign(spheres))
        self.assertTrue(np.all(union <= spheres + 1e-12))

    def test_culling_skips_only_children_separated_from_the_batch(self):
        # A batch outside a difference's base (or inside a union) whose box
        # overlaps a child's box may lie inside that child: the child still
        # counts, so values are exact however points are batched. Above a
        # hole, the field is the distance to the rim, not to the base plane.
        graph = Graph()
        plate = graph.add(BOX, [], [10, 10, 1, 0])
        node = graph.add(DIFFERENCE, [plate, graph.add(CYLINDER, [], [3, 2])], [0])
        pair = graph.add(UNION, [graph.add(SPHERE, [], [2]), graph.move(graph.add(SPHERE, [], [2]), [1, 0, 0])], [0])
        handle = self.build(graph)
        above = np.array([[0, 0, 1.0], [0.5, 0, 1.0], [0, 0, 1.5]])
        expected = [1.0, 1.0, 0.5]
        for points in (above, above[:1], np.vstack([above, [[30, 30, 30]] * 5])):
            count = min(len(points), len(expected))
            np.testing.assert_allclose(self.k.evaluate(handle, node, points)[:count], expected[:count], atol=1e-12)
        inside = np.array([[1.0, 0, 0], [0.5, 0, 0]])
        np.testing.assert_allclose(self.k.evaluate(handle, pair, inside), [-2.0, -1.5], atol=1e-12)
        hits, distances = self.k.raycast(handle, node, [[0, 0, 5.0], [5, 5, 5.0]], [[0, 0, -1.0]] * 2,
                                         max_distance=20)
        self.assertEqual(hits.tolist(), [False, True], "A ray down a through hole hits nothing")
        self.assertAlmostEqual(distances[1], 4, places=5)

    def assertBatchIndependent(self, handle, node, points, expected, atol=1e-12):
        """Values equal `expected` evaluated together, one point at a time, in
        reverse order and next to a far point (which widens every batch)."""
        np.testing.assert_allclose(self.k.evaluate(handle, node, points), expected, rtol=0, atol=atol)
        np.testing.assert_allclose(self.k.evaluate(handle, node, points[::-1])[::-1], expected, rtol=0, atol=atol)
        for i in range(0, len(points), max(1, len(points) // 40)):
            alone = self.k.evaluate(handle, node, points[i:i + 1])[0]
            widened = self.k.evaluate(handle, node, np.vstack([points[i:i + 1], [[400.0, 400.0, 400.0]]]))[0]
            self.assertAlmostEqual(alone, expected[i], delta=atol)
            self.assertAlmostEqual(widened, expected[i], delta=atol)

    def test_culling_keeps_values_of_fields_that_underestimate_distance(self):
        # Ellipsoids, stretches, intersections and twists are below the true
        # distance outside their boxes: culling scales box distances by their
        # distance ratio, so no blend or cut is skipped while it still counts.
        g = Graph()
        ball = g.move(g.add(SPHERE, [], [10]), [27.5, 0, 0])
        egg = g.add(ELLIPSOID, [], [14, 3, 8])
        node = g.add(UNION, [ball, egg], [3.0])
        handle = self.build(g)
        expected = smin_ref(np.array([0.2]), np.array([(math.sqrt((17.3 / 14) ** 2) - 1) * 3]), 3.0)
        self.assertAlmostEqual(expected[0], -0.3179, places=4)
        self.assertBatchIndependent(handle, node, np.array([[17.3, 0, 0]]), expected)

        g = Graph()
        cigar = g.add(STRETCH, [g.add(SPHERE, [], [1])], [8, 1, 1])
        bead = g.move(g.add(SPHERE, [], [1]), [10.5, 0, 0])
        cutter = g.add(ELLIPSOID, [], [10, 1, 1])
        bar = g.move(g.add(BOX, [], [3, 2, 2, 0]), [9, 0, 0])
        bars = g.add(INTERSECTION, [g.move(g.add(BOX, [], [10, 1, 1, 0]), [10, 1, 0]),
                                    g.move(g.add(BOX, [], [1, 10, 1, 0]), [1, 10, 0])], [0])
        twisted = g.move(g.add(TWIST, [g.add(BOX, [], [3, 0.5, 2, 0])], [0.6]), [0, -6, 0])
        blobs = g.add(UNION, [bead, cigar], [3.0])
        dented = g.add(DIFFERENCE, [bar, cutter], [1.5])
        corner = g.add(UNION, [g.move(g.add(SPHERE, [], [1]), [6, 6, 1.05]), bars], [5.6])
        many = g.add(UNION, [bead, cigar, twisted] + [g.move(g.add(SPHERE, [], [0.5]), [x, 4, 0])
                                                      for x in range(-8, 9)], [1.2])
        handle = self.build(g)
        p = np.vstack([self.points(3000, 13) * [1, 0.6, 0.6], [[6, 6, 0], [1, 1, 0], [12.4, 0, 0], [12.3, 0.1, 0]]])
        stretched = (length(p / [8, 1, 1]) - 1) * 1
        beads = length(p - [10.5, 0, 0]) - 1
        ellipsoid = (length(p / [10, 1, 1]) - 1) * 1
        bar_values = box_ref(p - [9, 0, 0], [3, 2, 2])
        cross = np.maximum(box_ref(p - [10, 1, 0], [10, 1, 1]), box_ref(p - [1, 10, 0], [1, 10, 1]))
        angle = -0.6 * p[:, 2]
        q = p - [0, -6, 0]
        local = np.column_stack([np.cos(angle) * q[:, 0] - np.sin(angle) * q[:, 1],
                                 np.sin(angle) * q[:, 0] + np.cos(angle) * q[:, 1], q[:, 2]])
        twist_values = box_ref(local, [3, 0.5, 2]) / math.sqrt(1 + (0.6 * math.hypot(3, 0.5)) ** 2)
        fold = smin_ref(smin_ref(beads, stretched, 1.2), twist_values, 1.2)
        for x in range(-8, 9):
            fold = smin_ref(fold, length(p - [x, 4, 0]) - 0.5, 1.2)
        cases = [(blobs, smin_ref(beads, stretched, 3.0)), (dented, smax_ref(bar_values, -ellipsoid, 1.5)),
                 (corner, smin_ref(length(p - [6, 6, 1.05]) - 1, cross, 5.6)), (many, fold)]
        for node, expected in cases:
            with self.subTest(node=node):
                self.assertBatchIndependent(handle, node, p, expected, atol=1e-11)

    def test_query_order_never_changes_results(self):
        # Large requests on costly graphs are visited in Morton order; outputs
        # keep the request order and match a reversed request exactly.
        g = Graph()
        holes = [g.move(g.add(CYLINDER, [], [0.6, 5]), [x, y, 0]) for x in range(-8, 9, 2) for y in range(-8, 9, 2)]
        plate = g.add(DIFFERENCE, [g.add(BOX, [], [10, 10, 1, 0]), *holes], [0.2])
        handle = self.build(g)
        p = self.points(30000, 11)
        values = self.k.evaluate(handle, plate, p)
        np.testing.assert_array_equal(self.k.evaluate(handle, plate, p[::-1])[::-1], values)
        normals = self.k.gradient(handle, plate, p[:5000])
        np.testing.assert_array_equal(self.k.gradient(handle, plate, p[4999::-1])[::-1], normals)
        origins = np.column_stack([self.rng.uniform(-11, 11, (4000, 2)), np.full(4000, 6.0)])
        directions = np.column_stack([self.rng.normal(scale=0.2, size=(4000, 2)), np.full(4000, -1.0)])
        hits, distances = self.k.raycast(handle, plate, origins, directions)
        reverse = self.k.raycast(handle, plate, origins[::-1], directions[::-1])
        np.testing.assert_array_equal(reverse[0][::-1], hits)
        np.testing.assert_array_equal(reverse[1][::-1], distances)
        self.assertGreater(hits.sum(), 1000)

    def test_polygon_signs_near_straight_vertices_and_spikes(self):
        # A vertex whose edges are collinear up to rounding takes its edges'
        # side (pseudonormal), however the nearest-feature distance tie falls.
        for delta in (1e-9, -1e-9, 1e-12, -1e-12, 3e-17, -3e-17):
            g = Graph()
            node = g.add(POLYGON, [], [0, -1, 2, -1, 2, 0, 1, delta, 0, 0])
            handle = self.build(g)
            xs = 1 + np.linspace(-2e-8, 2e-8, 4001)
            for y, sign in ((0.5, 1), (-0.5, -1)):
                values = self.k.evaluate(handle, node, np.column_stack([xs, np.full_like(xs, y)]))
                self.assertTrue(np.all(np.sign(values) == sign), f"delta={delta} y={y}")
        a, b = np.array([0.1, 0.7]), np.array([3.3, 2.9])  # points interpolated on an edge
        poly = np.array([a, a + 0.3 * (b - a), a + 0.6 * (b - a), b, (3.3, -1.0), (0.1, -1.0)])
        g = Graph()
        node = g.add(POLYGON, [], poly.ravel())
        handle = self.build(g)
        along = (b - a) / np.linalg.norm(b - a)
        inward = np.array([along[1], -along[0]])
        for t in (0.3, 0.6):
            for depth in (0.05, 0.5, -0.05, -0.5):
                p = a + t * (b - a) + depth * inward + np.linspace(-1e-9, 1e-9, 201)[:, None] * along
                values = self.k.evaluate(handle, node, p)
                self.assertTrue(np.all(np.sign(values) == -np.sign(depth)), f"t={t} depth={depth}")
        # Zero-width spikes (edges running back along each other) are removed.
        g = Graph()
        node = g.add(POLYGON, [], [0, 0, 2, 0, 2, 1, 1, 1, 1, 3, 1, 1, 0, 1])
        handle = self.build(g)
        np.testing.assert_allclose(self.k.evaluate(handle, node, [[0.5, 2], [1.5, 2], [1, 3.5], [1, 0.5]]),
                                   [1, 1, 2.5, -0.5], atol=1e-12)
        lo, hi, _ = self.k.bounds(handle, node)
        np.testing.assert_array_equal(lo, [0, 0])
        np.testing.assert_array_equal(hi, [2, 1])


class BoundsAndQueryTests(KernelCase):
    def assertBounds(self, graph, node, lo, hi, lipschitz=1.0):
        actual_lo, actual_hi, actual_lipschitz = self.k.bounds(self.build(graph), node)
        np.testing.assert_allclose(actual_lo, lo, atol=1e-12)
        np.testing.assert_allclose(actual_hi, hi, atol=1e-12)
        self.assertAlmostEqual(actual_lipschitz, lipschitz)

    def test_bounds_and_lipschitz(self):
        inf = math.inf
        g = Graph()
        box = g.add(BOX, [], [3, 2, 1, 0])
        rotated = g.move(box, [1, 2, 3], rotation_matrix([0, 0, 1], 90), 2.0)
        self.assertBounds(g, box, [-3, -2, -1], [3, 2, 1])
        self.assertBounds(g, rotated, [-3, -4, 1], [5, 8, 5])
        self.assertBounds(g, g.add(UNION, [box, g.move(box, [10, 0, 0])], [2]), [-3.5, -2.5, -1.5], [13.5, 2.5, 1.5])
        self.assertBounds(g, g.add(INTERSECTION, [box, g.move(box, [2, 0, 0])], [0]), [-1, -2, -1], [3, 2, 1])
        self.assertBounds(g, g.add(DIFFERENCE, [box, g.move(box, [2, 0, 0])], [0]), [-3, -2, -1], [3, 2, 1])
        self.assertBounds(g, g.add(HALFSPACE, [], [0, 0, 1, 2]), [-inf, -inf, -inf], [inf, inf, 2])
        self.assertBounds(g, g.add(HALFSPACE, [], [-1, 0, 0, 2]), [-2, -inf, -inf], [inf, inf, inf])
        self.assertBounds(g, g.add(HALFSPACE, [], [0.6, 0.8, 0, 2]), [-inf] * 3, [inf] * 3)
        rho = math.hypot(3, 2)
        self.assertBounds(g, g.add(TWIST, [box], [0.3]), [-rho, -rho, -1], [rho, rho, 1])
        self.assertBounds(g, g.add(SHELL, [box], [0.5, 1]), [-3.5, -2.5, -1.5], [3.5, 2.5, 1.5])
        self.assertBounds(g, g.add(ELONGATE, [box], [1, 0, 2]), [-4, -2, -3], [4, 2, 3])
        rect = g.add(RECT, [], [2, 1, 0])
        self.assertBounds(g, rect, [-2, -1], [2, 1])
        self.assertBounds(g, g.add(EXTRUDE, [rect], [1, 4]), [-2, -1, 1], [2, 1, 4])
        self.assertBounds(g, g.add(TAPERED_EXTRUDE, [rect], [0, 4, 0.5]), [-2.5, -1.5, 0], [2.5, 1.5, 4])
        self.assertBounds(g, g.add(REVOLVE, [rect], [5]), [-7, -7, -1], [7, 7, 1])
        self.assertBounds(g, g.add(POLYGON, [], star_points().ravel()),
                          star_points().min(axis=0), star_points().max(axis=0))
        self.assertBounds(g, g.add(HALFPLANE, [], [0, 1, 3]), [-inf, -inf], [inf, 3])
        empty = g.add(INTERSECTION, [box, g.move(box, [20, 0, 0])], [0])
        lo, hi, _ = self.k.bounds(self.build(g), empty)
        self.assertTrue(np.any(lo > hi))

    def test_smooth_union_bounds_cover_the_whole_fold(self):
        # A left fold of n coincident children sinks exactly k * x_(n-1) below
        # them, x_1 = 1/4 and x_(m+1) = x_m + (1 - x_m)^2 / 4 (SDF-DESIGN.md);
        # three unit spheres with k = 1 reach radius 1.390625, not 1.25.
        reach = [0.0]
        for _ in range(9):
            reach.append(reach[-1] + (1 - reach[-1]) ** 2 / 4)
        self.assertEqual(reach[1:3], [0.25, 0.390625])
        g = Graph()
        ball, disc = g.add(SPHERE, [], [1]), g.add(CIRCLE, [], [1])
        cases = []
        for n in (2, 3, 4, 10):
            for k in (0.5, 2.0):
                cases.append((n, k, g.add(UNION, [ball] * n, [k]), g.add(UNION2, [disc] * n, [k])))
        handle = self.build(g)
        for n, k, solid, profile in cases:
            with self.subTest(children=n, k=k):
                r = 1 + k * reach[n - 1]
                self.assertBounds(g, solid, [-r] * 3, [r] * 3)
                self.assertBounds(g, profile, [-r] * 2, [r] * 2)
                inside, outside = r - 1e-9, r + 1e-9
                solid_values = self.k.evaluate(handle, solid, [[inside, 0, 0], [0, 0, -outside]])
                profile_values = self.k.evaluate(handle, profile, [[0, inside], [-outside, 0]])
                for values in (solid_values, profile_values):
                    self.assertLess(values[0], 0)
                    self.assertGreater(values[1], 0)

    def test_loft_bounds_span_the_slab(self):
        g = Graph()
        node = g.add(LOFT, [g.add(CIRCLE, [], [1]), g.add(CIRCLE, [], [0.5])], [0, 2])
        self.assertBounds(g, node, [-1, -1, 0], [1, 1, 2])
        self.assertLess(self.k.evaluate(self.build(g), node, [[0, 0, 1]])[0], 0)

    def test_bounds_hold_grown_fields_that_underestimate_distance(self):
        # Offsets, outer shells, outlines, tapers and blends of fields below the
        # true distance reach further than their growth; no point outside the
        # reported bounds may be inside.
        g = Graph()
        ball = g.add(SPHERE, [], [1])
        egg = g.add(ELLIPSOID, [], [10, 1, 1])
        flat = g.add(ELLIPSE, [], [10, 1])
        twisted = g.add(TWIST, [g.add(CYLINDER, [], [1, 2])], [math.radians(115)])
        tapered_rect = g.add(TAPERED_EXTRUDE, [g.add(RECT, [], [1, 1, 0])], [0, 1, 2])
        pair = g.add(UNION, [g.move(egg, [0, 0.5, 0]), g.move(g.add(ELLIPSOID, [], [2, 6, 1]), [3, 0, 0])], [1.0])
        gap = g.add(INTERSECTION, [g.move(ball, [-1.1, 0, 0]), g.move(ball, [1.1, 0, 0])], [0])
        cases = {
            "ellipsoid offset": g.add(OFFSET, [egg], [1]),
            "stretched sphere offset": g.add(OFFSET, [g.add(STRETCH, [ball], [10, 1, 1])], [1]),
            "twisted cylinder offset": g.add(OFFSET, [twisted], [0.5]),
            "ellipsoid outer shell": g.add(SHELL, [egg], [1, 1]),
            "ellipsoid centred shell": g.add(SHELL, [egg], [2, 2]),
            "ellipse outline": g.add(EXTRUDE, [g.add(OUTLINE2, [flat], [2])], [0, 1]),
            "ellipse taper": g.add(TAPERED_EXTRUDE, [flat], [0, 4, 1]),
            "taper offset": g.add(OFFSET, [tapered_rect], [0.5]),
            "blend offset": g.add(OFFSET, [pair], [1]),
            "blend": g.add(UNION, [egg, g.move(egg, [0, 0.3, 0])], [1.0]),
            "empty intersection offset": g.add(OFFSET, [gap], [0.5]),
        }
        handle = self.build(g)
        lo, hi, _ = self.k.bounds(handle, gap)
        self.assertGreater(lo[0], hi[0])  # empty, but its field is small between the balls
        for name, node in cases.items():
            with self.subTest(case=name):
                lo, hi, _ = self.k.bounds(handle, node)
                self.assertTrue(np.all(np.isfinite(lo)) and np.all(lo < hi))
                size = hi - lo
                p = self.rng.uniform(lo - 0.5 * size, hi + 0.5 * size, size=(200000, 3))
                outside = np.any((p < lo) | (p > hi), axis=1)
                values = self.k.evaluate(handle, node, p[outside])
                self.assertGreater(values.min(), 0)
                # The solid does reach near its bounds: they are not grossly loose.
                inside = self.k.evaluate(handle, node, p[~outside])
                self.assertTrue(np.any(inside < 0))

    def test_raycast_step_budget_never_overflows(self):
        g = Graph()
        ball = g.add(SPHERE, [], [1])
        handle = self.build(g)
        for steps in (2**31 - 61, 2**31 - 60, 2**31 - 1):
            hits, distances = self.k.raycast(handle, ball, [[0, 0, 5.0]], [[0, 0, -1.0]], max_steps=steps)
            self.assertTrue(hits[0])
            self.assertAlmostEqual(distances[0], 4.0, places=5)

    def test_graphs_needing_too_many_visits_per_point_are_refused(self):
        # Every use of a shared subgraph is evaluated again: `s = s | s moved`
        # doubles the work per point, so 40 doublings would take ~1e12 visits.
        def doublings(levels):
            g = Graph()
            node = g.add(SPHERE, [], [1])
            for _ in range(levels):
                node = g.add(UNION, [node, g.move(node, [1e-3, 0, 0])], [0])
            return g, node

        g, node = doublings(20)
        self.assertLess(self.k.evaluate(self.build(g), node, [[0, 0, 0]])[0], 0)
        g, node = doublings(40)
        with self.assertRaisesRegex(KernelError, r"node \d+ \(UNION\): evaluating it would visit more than"):
            self.build(g)

    def test_loft_lipschitz_bound_holds_between_opposite_corners(self):
        # |b - a| reaches the diagonal of the box around both profiles.
        g = Graph()
        a = g.move2(g.add(CIRCLE, [], [0.05]), [1.95, 1.95])
        b = g.move2(g.add(CIRCLE, [], [0.05]), [-1.95, -1.95])
        nodes = [g.add(LOFT, [a, b], [0, height]) for height in (1.0, 0.2, 0.05)]
        handle = self.build(g)
        for node, height in zip(nodes, (1.0, 0.2, 0.05)):
            _, _, lipschitz = self.k.bounds(handle, node)
            p = np.column_stack([self.rng.uniform(-2, 2, (100000, 2)), self.rng.uniform(0, height, 100000)])
            step = self.rng.normal(size=(100000, 3))
            step *= 1e-4 * min(height, 1) / length(step)[:, None]
            slopes = np.abs(self.k.evaluate(handle, node, p + step) - self.k.evaluate(handle, node, p)) / length(step)
            self.assertLessEqual(slopes.max(), lipschitz * (1 + 1e-6), f"height {height}")

    def test_gradients(self):
        g = Graph()
        ball = g.move(g.add(SPHERE, [], [5]), [1, 2, 3])
        box = g.add(BOX, [], [4, 3, 2, 0])
        handle = self.build(g)
        p = self.points(500, 10) + [1, 2, 3]
        expected = (p - [1, 2, 3]) / length(p - [1, 2, 3])[:, None]
        # The tetrahedral stencil is first-order accurate: error ~ eps x curvature.
        np.testing.assert_allclose(self.k.gradient(handle, ball, p), expected, atol=1e-5)  # default eps
        np.testing.assert_allclose(self.k.gradient(handle, ball, p, eps=1e-7), expected, atol=1e-6)
        faces = np.array([[3.9, 0.5, 0.3], [-3.9, 0.1, 0.2], [1, 2.95, -1], [0.5, -0.2, -1.97]])
        np.testing.assert_allclose(self.k.gradient(handle, box, faces, eps=1e-6),
                                   [[1, 0, 0], [-1, 0, 0], [0, 1, 0], [0, 0, -1]], atol=1e-9)
        np.testing.assert_allclose(length(self.k.gradient(handle, ball, p, eps=1e-3)), 1, atol=1e-12)
        profile = Graph()
        disc = profile.add(CIRCLE, [], [1])
        with self.assertRaisesRegex(KernelError, "3D"):
            self.k.gradient(self.build(profile), disc, p)

    def test_raycasts(self):
        g = Graph()
        ball = g.add(SPHERE, [], [10])
        twisted = g.add(TWIST, [g.add(BOX, [], [6, 2, 8, 0])], [0.1])
        handle = self.build(g)
        heights = np.linspace(-12, 12, 49)
        origins = np.column_stack([np.full_like(heights, -50), heights, np.zeros_like(heights)])
        directions = np.tile([2.0, 0, 0], (len(heights), 1))  # normalised by the kernel
        hits, distances = self.k.raycast(handle, ball, origins, directions)
        expected = np.abs(heights) < 10
        np.testing.assert_array_equal(hits, expected)
        np.testing.assert_allclose(distances[expected], 50 - np.sqrt(100 - heights[expected] ** 2), atol=1e-5)
        hits, distances = self.k.raycast(handle, ball, [[0, 0, 0]], [[0, 0, 1]])  # starts inside: exits
        self.assertTrue(hits[0])
        self.assertAlmostEqual(distances[0], 10, places=5)
        hits, _ = self.k.raycast(handle, ball, [[-50, 0, 0]], [[1, 0, 0]], max_distance=30)
        self.assertFalse(hits[0])
        # Hits on a twisted box land on its surface.
        origins = np.column_stack([np.full(40, -40.0), np.linspace(-4, 4, 40), np.linspace(-6, 6, 40)])
        hits, distances = self.k.raycast(handle, twisted, origins, np.tile([1.0, 0.05, 0], (40, 1)))
        self.assertGreater(hits.sum(), 20)
        direction = np.array([1.0, 0.05, 0]) / np.linalg.norm([1.0, 0.05, 0])
        surface = origins[hits] + distances[hits, None] * direction
        np.testing.assert_allclose(self.k.evaluate(handle, twisted, surface), 0, atol=1e-5)


def malformed_cases():
    def header(count=1, version=1, reserved=0, magic=b"RSDF"):
        return magic + struct.pack("<III", version, count, reserved)

    def record(op, children, params, reserved=0):
        return (struct.pack("<IIII", op, len(children), len(params), reserved)
                + struct.pack(f"<{len(children)}I", *children)
                + struct.pack(f"<{len(params)}d", *params))

    sphere = record(SPHERE, [], [1.0])
    reflection = (-1, 0, 0, 0, 1, 0, 0, 0, 1)
    skewed = (1, 0.1, 0, 0, 1, 0, 0, 0, 1)
    return [
        (b"", "truncated"),
        (b"RSDF" + struct.pack("<I", 1), "truncated"),
        (header(magic=b"RSDX") + sphere, "magic"),
        (header(version=2) + sphere, "version"),
        (header(reserved=7) + sphere, "reserved"),
        (header(count=0), "no nodes"),
        (header(count=3) + sphere, "truncated"),
        (header() + sphere[:-4], "truncated"),
        (header() + sphere + b"\0", "trailing"),
        (header() + record(99, [], [1.0]), "unknown opcode"),
        (header() + record(SPHERE, [], [1.0], reserved=1), "reserved"),
        (header() + record(SPHERE, [], [1.0, 2.0]), "parameters"),
        (header() + record(POLYGON, [], [0, 0, 1, 0]), "parameters"),
        (header() + record(POLYGON, [], [0, 0, 1, 0, 1]), "parameters"),
        (header(2) + sphere + record(SPHERE, [0], [1.0]), "children"),
        (header() + record(UNION, [], [0.0]), "children"),
        (header(2) + sphere + record(LOFT, [0], [0, 1]), "children"),
        (header(2) + sphere + record(OFFSET, [1], [1.0]), "does not precede"),
        (header(2) + sphere + record(OFFSET, [5], [1.0]), "does not precede"),
        (header() + record(SPHERE, [], [math.nan]), "not finite"),
        (header() + record(SPHERE, [], [math.inf]), "not finite"),
        (header() + record(SPHERE, [], [-1.0]), "positive"),
        (header() + record(BOX, [], [1, 1, 1, 2]), "edge radius"),
        (header(2) + sphere + record(TRANSFORM, [0], [*reflection, 0, 0, 0, 1]), "determinant"),
        (header(2) + sphere + record(TRANSFORM, [0], [*skewed, 0, 0, 0, 1]), "orthonormal"),
        (header(2) + sphere + record(TRANSFORM, [0], [*IDENTITY, 0, 0, 0, 0]), "scale"),
        (header(2) + sphere + record(TRANSFORM2, [0], [1, 0, 0, 1, 0, 0, 1]), "2D"),
        (header(2) + record(CIRCLE, [], [1.0]) + record(UNION, [0], [0.0]), "2D"),
        (header(3) + sphere + record(CIRCLE, [], [1.0]) + record(UNION2, [0, 1], [0.0]), "3D"),
        (header(2) + sphere + record(EXTRUDE, [0], [0, 1]), "3D"),
        (header(2) + record(CIRCLE, [], [1.0]) + record(EXTRUDE, [0], [1, 1]), "z1"),
        (header() + record(HALFSPACE, [], [0, 0, 2, 1]), "unit length"),
        (header(2) + sphere + record(SHELL, [0], [1, 3]), "mode"),
        (header(2) + sphere + record(UNION, [0], [-1]), "blend"),
        (header(2) + record(HALFSPACE, [], [0, 0, 1, 0]) + record(TWIST, [0], [0.1]), "bounded"),
        (header() + record(POLYGON, [], [0, 0, 1, 1, 2, 2]), "zero area"),
        (header() + record(POLYGON, [], [0, 0, 0, 0, 1, 1, 0, 0]), "three distinct"),
        (header() + record(CONE, [], [0, 0, 1]), "cone radius"),
    ]


class MalformedInputTests(KernelCase):
    def test_malformed_graphs_report_errors(self):
        for data, message in malformed_cases():
            with self.subTest(message=message, data=data[:40]):
                with self.assertRaises(KernelError) as caught:
                    self.handles.append(self.k.create(data))
                self.assertIn(message, str(caught.exception))

    def test_random_corruption_never_crashes(self):
        g = Graph()
        poly = g.add(POLYGON, [], gear_points(teeth=8).ravel())
        body = g.add(EXTRUDE, [g.add(UNION2, [poly, g.add(CIRCLE, [], [3])], [0.5])], [0, 4])
        g.add(DIFFERENCE, [body, g.move(g.add(SPHERE, [], [2]), [0, 0, 4]), g.add(TWIST, [body], [0.2])], [0.3])
        data = bytearray(g.to_bytes())
        p = self.points(64, 25)
        created = 0
        for trial in range(1500):
            mutated = bytearray(data)
            for _ in range(1 + trial % 4):
                mutated[self.rng.integers(16, len(mutated))] = self.rng.integers(0, 256)
            if trial % 7 == 0:
                mutated = mutated[:self.rng.integers(0, len(mutated))]
            handle = self.k.lib.rsdf_graph_create(bytes(mutated), len(mutated))
            if not handle:
                self.assertTrue(self.k.error())
                continue
            created += 1
            try:
                for node in range(self.k.lib.rsdf_graph_node_count(handle)):
                    dimension = self.k.lib.rsdf_node_dimension(handle, node)
                    values = self.k.evaluate(handle, node, p[:, :dimension])
                    self.assertEqual(values.shape, (64,))
            finally:
                self.k.lib.rsdf_graph_free(handle)
        self.assertGreater(created, 0)

    def test_query_errors(self):
        g = Graph()
        ball = g.add(SPHERE, [], [1])
        disc = g.add(CIRCLE, [], [1])
        plane = g.add(HALFSPACE, [], [0, 0, 1, 0])
        handle = self.build(g)
        with self.assertRaisesRegex(KernelError, "does not exist"):
            self.k.evaluate(handle, 9, np.zeros((1, 3)))
        with self.assertRaisesRegex(KernelError, "not finite"):
            self.k.evaluate(handle, ball, [[0, math.nan, 0]])
        with self.assertRaisesRegex(KernelError, "zero"):
            self.k.raycast(handle, ball, [[0, 0, 5]], [[0, 0, 0]])
        with self.assertRaisesRegex(KernelError, "2D"):
            self.k.mesh(handle, disc, 0.1)
        with self.assertRaisesRegex(KernelError, "unbounded"):
            self.k.mesh(handle, plane, 0.1)
        with self.assertRaisesRegex(KernelError, "cell_size"):
            self.k.mesh(handle, ball, 0.0)
        with self.assertRaisesRegex(KernelError, "max_cells"):
            self.k.mesh(handle, ball, 0.01, max_cells=1000)
        self.assertEqual(self.k.lib.rsdf_node_dimension(handle, 99), -1)
        self.assertIn("does not exist", self.k.error())


class MeshTests(KernelCase):
    def mesh(self, graph, node, cell_size, **options):
        mesh = self.k.mesh(self.build(graph), node, cell_size, **options)
        stats = mesh.stats
        self.assertGreater(stats.triangle_count, 0)
        self.assertEqual(stats.boundary_edges, 0)
        self.assertEqual(stats.nonmanifold_edges, 0)
        self.assertGreater(stats.volume, 0)  # outward orientation
        self.assertTrue(np.all(np.isfinite(mesh.positions)))
        np.testing.assert_allclose(length(mesh.normals.astype(float)), 1, atol=1e-5)
        self.assertLess(int(mesh.triangles.max()), stats.vertex_count)
        return mesh

    def assertMeasures(self, mesh, volume, area=None, volume_tolerance=0.004, area_tolerance=0.01):
        self.assertAlmostEqual(mesh.stats.volume / volume, 1, delta=volume_tolerance)
        if area is not None:
            self.assertAlmostEqual(mesh.stats.area / area, 1, delta=area_tolerance)

    def test_closed_accurate_meshes(self):
        g = Graph()
        star, gear = star_points(), gear_points()
        holes = [(x, y) for x in (-12, 0, 12) for y in (-7, 7)]
        plate = g.add(DIFFERENCE, [g.add(BOX, [], [20, 15, 5, 0])] +
                      [g.move(g.add(CYLINDER, [], [3, 8]), [x, y, 0]) for x, y in holes], [0])
        r, R = 2.0, 6.0
        cases = {
            "sphere": (g.add(SPHERE, [], [10]), 0.25, 4 / 3 * math.pi * 1000, 4 * math.pi * 100),
            "box": (g.move(g.add(BOX, [], [6, 4, 3, 0]), [0.31, -0.17, 0.05]), 0.3, 6 * 8 * 12,
                    2 * (12 * 8 + 12 * 6 + 8 * 6)),
            "rounded box": (g.add(BOX, [], [6, 4, 3, 1]), 0.2, None, None),
            "cylinder": (g.add(CYLINDER, [], [5, 4]), 0.2, math.pi * 25 * 8, 2 * math.pi * 5 * 8 + 2 * math.pi * 25),
            "torus": (g.add(TORUS, [], [R, r]), 0.15, 2 * math.pi ** 2 * R * r * r, 4 * math.pi ** 2 * R * r),
            "drilled box": (plate, 0.25, 40 * 30 * 10 - len(holes) * math.pi * 9 * 10,
                            2 * 40 * 30 + 2 * (40 + 30) * 10 - 2 * len(holes) * math.pi * 9
                            + len(holes) * 2 * math.pi * 3 * 10),
            "star": (g.add(EXTRUDE, [g.add(POLYGON, [], star.ravel())], [0, 6]), 0.15, shoelace(star) * 6,
                     2 * shoelace(star) + perimeter(star) * 6),
            "gear": (g.add(EXTRUDE, [g.add(POLYGON, [], gear.ravel())], [-2.5, 2.5]), 0.08, shoelace(gear) * 5,
                     2 * shoelace(gear) + perimeter(gear) * 5),
        }
        # Rounded box: Steiner formulas for a box (half extents a, b, c) grown by r.
        a, b, c, rr = 5.0, 3.0, 2.0, 1.0
        rounded_volume = 8 * a * b * c + 8 * (a * b + b * c + c * a) * rr + 2 * math.pi * (a + b + c) * rr ** 2 \
            + 4 / 3 * math.pi * rr ** 3
        rounded_area = 8 * (a * b + b * c + c * a) + 4 * math.pi * (a + b + c) * rr + 4 * math.pi * rr ** 2
        cases["rounded box"] = (cases["rounded box"][0], 0.2, rounded_volume, rounded_area)
        for name, (node, cell, volume, area) in cases.items():
            with self.subTest(shape=name):
                mesh = self.mesh(g, node, cell)
                self.assertMeasures(mesh, volume, area)
                self.assertEqual(mesh.stats.components, 1)

    def test_smooth_union_volume(self):
        g = Graph()
        node = g.add(UNION, [g.move(g.add(SPHERE, [], [4]), [-3, 0, 0]), g.move(g.add(SPHERE, [], [3]), [3, 0, 0])],
                     [2.0])
        mesh = self.mesh(g, node, 0.15)
        # Reference volume by midpoint integration of the same field.
        step = 0.05
        lo, hi, _ = self.k.bounds(self.build(g), node)
        axes = [np.arange(lo[i] + step / 2, hi[i], step) for i in range(3)]
        grid = np.stack(np.meshgrid(*axes, indexing="ij"), axis=-1).reshape(-1, 3)
        reference = np.count_nonzero(self.k.evaluate(self.build(g), node, grid) < 0) * step ** 3
        self.assertMeasures(mesh, reference)
        self.assertEqual(mesh.stats.components, 1)

    def test_smooth_union_of_several_children_is_not_clipped(self):
        # Three coincident unit spheres blended with k = 1 form a sphere of
        # radius 1.390625; bounds expanded by only k/4 would cut it at 1.25.
        g = Graph()
        ball = g.add(SPHERE, [], [1])
        node = g.add(UNION, [ball] * 3, [1.0])
        radius = 1.390625
        mesh = self.mesh(g, node, 0.02)
        self.assertMeasures(mesh, 4 / 3 * math.pi * radius ** 3, 4 * math.pi * radius ** 2)
        np.testing.assert_allclose(mesh.stats.bounds_lo, [-radius] * 3, atol=0.005)
        np.testing.assert_allclose(mesh.stats.bounds_hi, [radius] * 3, atol=0.005)
        self.assertEqual(mesh.stats.components, 1)

    def test_normals_point_outward(self):
        g = Graph()
        node = g.add(DIFFERENCE, [g.add(BOX, [], [5, 5, 5, 1]), g.add(SPHERE, [], [4])], [0.5])
        mesh = self.mesh(g, node, 0.2)
        corners = mesh.positions[mesh.triangles].astype(float)
        facet = np.cross(corners[:, 1] - corners[:, 0], corners[:, 2] - corners[:, 0])
        facet /= np.maximum(length(facet), 1e-30)[:, None]
        shading = mesh.normals[mesh.triangles].astype(float).sum(axis=1)
        self.assertGreater(np.mean(np.sum(facet * shading, axis=1) > 0), 0.99)
        self.assertEqual(mesh.stats.components, 2)  # outer shell and inner cavity
        self.assertEqual(mesh.stats.cavities, 1)  # the sphere's void, enclosing negative volume
        self.assertAlmostEqual(mesh.stats.cavity_volume, 4 / 3 * math.pi * 4 ** 3, delta=3.0)

    def test_sharp_box_corners(self):
        g = Graph()
        center = np.array([0.123, 0.456, 0.789])
        node = g.move(g.add(BOX, [], [5, 4, 3, 0]), center)
        for cell in (0.37, 0.23, 0.5, 0.1):
            mesh = self.mesh(g, node, cell)
            for corner in np.array(np.meshgrid([-5, 5], [-4, 4], [-3, 3])).T.reshape(-1, 3):
                nearest = np.min(length(mesh.positions.astype(float) - (center + corner)))
                self.assertLess(nearest, 0.01 * cell, msg=f"corner {center + corner} at h={cell}")

    def test_oblique_shallow_crease_follows_the_implicit_ridge(self):
        # The roof planes meet at an 18-degree change of normal. The ridge
        # crosses the sampling lattice obliquely and lies below the threshold
        # for a drawn edge, but it still separates the two shading normals.
        g = Graph()
        angle = math.radians(18 / 2)
        s, c = math.sin(angle), math.cos(angle)
        body = g.add(BOX, [], [4, 4, 3, 0])
        left = g.add(HALFSPACE, [], [-s, 0, c, 2 * c])
        right = g.add(HALFSPACE, [], [s, 0, c, 2 * c])
        roof = g.add(INTERSECTION, [body, left, right], [0])
        rotation = rotation_matrix([1, 2, 3], 29)
        offset = np.array([0.137, -0.159, 0.213])
        node = g.move(roof, offset, rotation)
        cell = 0.43
        mesh = self.mesh(g, node, cell, preview=True)

        points = (mesh.positions.astype(float) - offset) @ rotation
        local = points[mesh.triangles]
        x, y, z = local.transpose(2, 0, 1)
        top = (np.abs(y).max(axis=1) < 2.7) & (np.abs(x).max(axis=1) < 1.4) & (z.min(axis=1) > 1.65)
        local = local[top]
        self.assertGreater(len(local), 100)
        x = local[:, :, 0]
        # A triangle that has two corners across the exact ridge uses a
        # blended normal where the hard transition should be.
        crossing = (x.min(axis=1) < -0.02 * cell) & (x.max(axis=1) > 0.02 * cell)
        self.assertLess(np.mean(crossing), 0.03)

        side = np.sign(x.mean(axis=1))
        normals = mesh.normals[mesh.triangles[top]].astype(float) @ rotation
        cosine = normals[:, :, 0] * side[:, None] * s + normals[:, :, 2] * c
        self.assertGreater(np.percentile(cosine, 5), math.cos(math.radians(2)))
        self.assertEqual(mesh.stats.boundary_edges, 0)
        self.assertEqual(mesh.stats.nonmanifold_edges, 0)
        self.assertEqual(mesh.stats.components, 1)

    def test_curved_shallow_crease_preserves_the_normals_on_both_sides(self):
        # A plane clipped near the sphere's pole leaves a circular crease
        # with a 22-degree change of normal. Rotate it off the sampling axes.
        g = Graph()
        radius, angle = 5, math.radians(22)
        cut_height = radius * math.cos(angle)
        rim_radius = radius * math.sin(angle)
        ball = g.add(SPHERE, [], [radius])
        plane = g.add(HALFSPACE, [], [0, 0, 1, cut_height])
        clipped = g.add(INTERSECTION, [ball, plane], [0])
        rotation = rotation_matrix([1, 2, 3], 29)
        offset = np.array([0.137, -0.159, 0.213])
        node = g.move(clipped, offset, rotation)
        cell = 0.3
        mesh = self.mesh(g, node, cell, threads=1, preview=True)
        parallel = self.mesh(g, node, cell, threads=4, preview=True)
        for key in ("positions", "normals", "triangles", "creases"):
            np.testing.assert_array_equal(getattr(mesh, key), getattr(parallel, key))

        points = (mesh.positions.astype(float) - offset) @ rotation
        local = points[mesh.triangles]
        radial = np.hypot(local[:, :, 0], local[:, :, 1])
        near_rim = (np.abs(radial - rim_radius).max(axis=1) < 0.65) & (local[:, :, 2].min(axis=1) > cut_height - 0.5)
        self.assertGreater(np.count_nonzero(near_rim), 500)
        radial = radial[near_rim]
        crossing = (radial.min(axis=1) < rim_radius - 0.02 * cell) & (radial.max(axis=1) > rim_radius + 0.02 * cell)
        self.assertLess(np.mean(crossing), 0.01)

        local = local[near_rim]
        normals = mesh.normals[mesh.triangles[near_rim]].astype(float) @ rotation
        center = local.mean(axis=1)
        on_cap = np.hypot(center[:, 0], center[:, 1]) < rim_radius
        outward = local / np.linalg.norm(local, axis=2)[:, :, None]
        cosine = np.where(on_cap[:, None], normals[:, :, 2], np.sum(normals * outward, axis=2))
        self.assertGreater(np.percentile(cosine, 5), math.cos(math.radians(2)))
        self.assertEqual(mesh.stats.boundary_edges, 0)
        self.assertEqual(mesh.stats.nonmanifold_edges, 0)
        self.assertEqual(mesh.stats.components, 1)

    def test_faces_beside_creases_shade_with_their_own_normals(self):
        # A crease vertex lies where faces meet, so its gradient is a blend of
        # theirs: flat faces must shade flat up to their edges. A blend
        # narrower than a cell (smooth 0.4 at cell 0.72) is not resolved by
        # the mesh, but must not tilt the faces beside it by the crease angle.
        g = Graph()
        rotation = rotation_matrix([1, 2, 0.5], 11)
        cylinder = g.move(g.add(CYLINDER, [], [5, 4]), [0.13, -0.27, 0.31], rotation)
        block = g.move(g.add(BOX, [], [4, 4, 1.5, 0]), [0, 0, 0], rotation_matrix([0, 0, 1], 9))
        slotted = g.add(DIFFERENCE, [block, g.move(g.add(BOX, [], [0.75, 10, 10, 0]), [0, 0, 2])], [0.4])
        axis, up = rotation[:, 2], np.array([0.0, 0.0, 1.0])
        cases = [(cylinder, 0.3, axis, 0.5), (cylinder, 0.37, axis, 0.5), (slotted, 0.72, up, 20)]
        for node, cell, normal, tolerance in cases:
            mesh = self.mesh(g, node, cell, preview=True)
            corners = mesh.positions[mesh.triangles].astype(float)
            facet = np.cross(corners[:, 1] - corners[:, 0], corners[:, 2] - corners[:, 0])
            facet /= length(facet)[:, None]
            for face in (normal, -normal):
                with self.subTest(cell=cell, face=face.tolist()):
                    flat = facet @ face > math.cos(math.radians(1))
                    self.assertGreater(np.count_nonzero(flat), 50)
                    shading = mesh.normals[mesh.triangles[flat]].reshape(-1, 3).astype(float) @ face
                    self.assertGreater(shading.min(), math.cos(math.radians(tolerance)))

    def test_buried_cylinder_caps_do_not_crease_the_outer_wall(self):
        # Overlapping coaxial cylinders have one continuous cylindrical wall.
        # Inside it, a triangle centroid can sample either buried cap.
        g = Graph()
        rotation = rotation_matrix([1, 2, 0], 41)
        axis = rotation[:, 2]
        cylinder = g.add(CYLINDER, [], [4, 3])
        for overlap in (0.01, 0.001):
            with self.subTest(overlap=overlap):
                offset = (3 - overlap / 2) * axis
                node = g.add(UNION, [g.move(cylinder, -offset, rotation), g.move(cylinder, offset, rotation)], [0])
                mesh = self.mesh(g, node, 0.72)

                axial = mesh.positions @ axis
                join = np.abs(axial) < 2
                radial = mesh.positions[join] - axial[join, None] * axis
                radial /= length(radial)[:, None]
                self.assertGreater(np.min(np.sum(radial * mesh.normals[join], axis=1)), math.cos(math.radians(2)))
                ends = mesh.creases.reshape(-1, 2, 3)
                self.assertFalse(np.any(np.abs(ends @ axis) < 2))
                self.assertGreater(len(ends), 0)  # the exposed end rims still crease

    def test_a_blend_narrower_than_a_cell_is_refined_on_the_surface(self):
        # A smooth union of two flush cylinders bulges into a bead k/4 high
        # and about two cells wide. Unrefined, its QEF vertices lie above the
        # crest or dent into it, triangles span it, and its crest creases in
        # some grid columns but not others: it shades as splotches.
        g = Graph()
        center = np.array([0.13, -0.27, 0.31])
        low = g.move(g.add(CYLINDER, [], [8, 2]), center + [0, 0, -2])
        high = g.move(g.add(CYLINDER, [], [8, 3]), center + [0, 0, 3])
        node = g.add(UNION, [low, high], [0.6])
        handle = self.build(g)

        def on_bead(points):
            offset = points - center
            return (np.abs(offset[..., 2]) < 0.6) & (np.hypot(offset[..., 0], offset[..., 1]) > 7.5)

        for cell in (0.43, 0.47, 0.5, 0.53, 0.5325, 0.56):
            with self.subTest(cell=cell):
                mesh = self.mesh(g, node, cell)
                points = mesh.positions.astype(float)
                self.assertLess(np.max(np.abs(self.k.evaluate(handle, node, points[on_bead(points)]))), 1e-5)
                corners = points[mesh.triangles]
                corners = corners[on_bead(corners.mean(axis=1))]
                facet = np.cross(corners[:, 1] - corners[:, 0], corners[:, 2] - corners[:, 0])
                facet /= length(facet)[:, None]
                field = self.k.gradient(handle, node, corners.mean(axis=1))
                field /= length(field)[:, None]
                deviation = np.degrees(np.arccos(np.clip(np.sum(facet * field, axis=1), -1, 1)))
                self.assertLess(np.percentile(deviation, 95), 9.5)
                ends = mesh.creases.reshape(-1, 2, 3).astype(float)
                self.assertFalse(np.any(np.all(on_bead(ends), axis=1)))

    def test_crease_segments(self):
        g = Graph()
        box = g.move(g.add(BOX, [], [5, 4, 3, 0]), [0.2, 0.1, -0.3])
        mesh = self.mesh(g, box, 0.3)
        self.assertGreater(mesh.stats.crease_segment_count, 0)
        ends = mesh.creases.reshape(-1, 3).astype(float) - [0.2, 0.1, -0.3]
        on_edge = np.sum(np.abs(np.abs(ends) - [5, 4, 3]) < 1e-3, axis=1) >= 2
        self.assertGreater(np.mean(on_edge), 0.95)
        total = np.sum(length(mesh.creases[:, 3:] - mesh.creases[:, :3]))
        self.assertAlmostEqual(total / (4 * (10 + 8 + 6)), 1, delta=0.05)
        sphere = self.mesh(g, g.add(SPHERE, [], [10]), 0.25)
        self.assertEqual(sphere.stats.crease_segment_count, 0)

    def test_explicit_bounds_clip_and_close(self):
        g = Graph()
        node = g.add(SPHERE, [], [5])
        mesh = self.mesh(g, node, 0.1, bounds=([-6, -6, 0], [6, 6, 6]))
        self.assertMeasures(mesh, 2 / 3 * math.pi * 125, 3 * math.pi * 25)
        empty = g.add(INTERSECTION, [node, g.move(node, [20, 0, 0])], [0])
        self.assertEqual(self.k.mesh(self.build(g), empty, 0.1).stats.triangle_count, 0)

    def test_deterministic_for_any_thread_count(self):
        g = Graph()
        node = g.add(UNION, [g.add(EXTRUDE, [g.add(POLYGON, [], gear_points(teeth=12).ravel())], [0, 4]),
                             g.move(g.add(SPHERE, [], [6]), [0, 0, 4])], [1.0])
        handle = self.build(g)
        meshes = [self.k.mesh(handle, node, 0.2, threads=threads) for threads in (1, 4, 0)]
        for other in meshes[1:]:
            for name in ("positions", "normals", "triangles", "creases"):
                self.assertTrue(np.array_equal(getattr(meshes[0], name), getattr(other, name)), name)
            self.assertEqual(meshes[0].stats.volume, other.stats.volume)
        p = self.points(100000, 20)
        values = [self.k.evaluate(handle, node, p, threads=threads) for threads in (1, 4, 0)]
        self.assertTrue(np.array_equal(values[0], values[1]) and np.array_equal(values[0], values[2]))


    def test_lofts_mesh_and_combine(self):
        g = Graph()
        R, r, height = 1.0, 0.5, 2.0
        loft = g.add(LOFT, [g.add(CIRCLE, [], [R]), g.add(CIRCLE, [], [r])], [0, height])
        frustum = math.pi * height / 3 * (R * R + R * r + r * r)
        self.assertMeasures(self.mesh(g, loft, 0.02), frustum)
        # As a later union child and as a cutter the loft is never skipped.
        joined = g.add(UNION, [g.move(g.add(SPHERE, [], [0.3]), [5, 0, 0]), loft], [0])
        block = g.move(g.add(BOX, [], [2, 2, 2, 0]), [0, 0, 1])
        pocket = g.add(DIFFERENCE, [block, loft], [0])
        handle = self.build(g)
        loft_value = self.k.evaluate(handle, loft, [[0, 0, 1]])[0]
        self.assertAlmostEqual(self.k.evaluate(handle, joined, [[0, 0, 1]])[0], loft_value, places=12)
        self.assertAlmostEqual(self.k.evaluate(handle, pocket, [[0, 0, 1]])[0], -loft_value, places=12)
        self.assertMeasures(self.mesh(g, joined, 0.02), frustum + 4 / 3 * math.pi * 0.3 ** 3)
        self.assertMeasures(self.mesh(g, pocket, 0.02), 64 - frustum)

    def test_culled_children_never_open_meshes(self):
        # The same fields with the culled child made unbounded (never culled)
        # are the controls.
        g = Graph()
        far = g.add(HALFSPACE, [], [0, 0, 1, -1e6])
        cigar = g.add(STRETCH, [g.add(SPHERE, [], [1])], [8, 1, 1])
        blob = g.add(UNION, [g.move(g.add(SPHERE, [], [1]), [10.5, 0, 0]), cigar], [3])
        blob_control = g.add(UNION, [g.move(g.add(SPHERE, [], [1]), [10.5, 0, 0]), g.add(UNION, [cigar, far], [0])], [3])
        bar = g.move(g.add(BOX, [], [3, 2, 2, 0]), [9, 0, 0])
        rod = g.move(g.add(STRETCH, [g.add(CYLINDER, [], [0.8, 1])], [1, 1, 6]), [0, 0, 0], rotation_matrix([0, 1, 0], 90))
        dent = g.add(DIFFERENCE, [bar, rod], [2])
        dent_control = g.add(DIFFERENCE, [bar, g.add(UNION, [rod, far], [0])], [2])
        bounds = ([-12, -5, -5], [16, 5, 5])
        for node, control, cell in ((blob, blob_control, 0.04), (dent, dent_control, 0.05)):
            with self.subTest(node=node):
                mesh = self.mesh(g, node, cell)
                reference = self.mesh(g, control, cell, bounds=bounds)
                self.assertEqual(mesh.stats.components, 1)
                self.assertAlmostEqual(mesh.stats.volume / reference.stats.volume, 1, delta=1e-3)
        self.assertLess(self.k.mesh(self.build(g), dent, 0.05).stats.volume, 95)  # the dent is cut

    def test_grown_booleans_of_exact_shapes_stay_closed(self):
        # SHELL and OFFSET read values away from the zero level set: culling
        # must keep values, not only signs.
        g = Graph()
        bar = g.add(UNION, [g.add(BOX, [], [10, 5, 5, 0]), g.move(g.add(BOX, [], [5, 5, 5, 0]), [8, 0, 0])], [0])
        hollow = g.add(SHELL, [bar], [1, 0])
        plate = g.add(DIFFERENCE, [g.add(BOX, [], [5, 5, 1, 0]), g.add(CYLINDER, [], [2, 5])], [0])
        grown = g.add(OFFSET, [plate], [0.7])
        handle = self.build(g)
        np.testing.assert_allclose(self.k.evaluate(handle, hollow, [[9.5, 0, 0]]), [2.5], atol=1e-12)
        np.testing.assert_allclose(self.k.evaluate(handle, grown, [[0, 0, 1.5]]), [1.3], atol=1e-12)
        hits, _ = self.k.raycast(handle, grown, [[0, 0, 5.0]], [[0, 0, -1.0]], max_distance=20)
        self.assertFalse(hits[0])
        mesh = self.mesh(g, hollow, 0.1)
        self.assertMeasures(mesh, 23 * 10 * 10 - 21 * 8 * 8, volume_tolerance=1e-6)
        self.assertEqual(mesh.stats.components, 2)
        mesh = self.mesh(g, grown, 0.1)
        self.assertEqual(np.count_nonzero(np.hypot(mesh.positions[:, 0], mesh.positions[:, 1]) < 1.0), 0)

    def test_rotated_boxes_mesh_without_pinches(self):
        # Ambiguous faces near sharp edges are decided by the field at the face
        # centre: the bilinear rule split these cells into two vertices.
        cases = [((0.5, 0.5, 0.5), [[-0.8609518015782385, 0.2968878623443728, 0.41306124552150203, 0.611656186051831],
                                    [0.0998065360877074, -0.6976428620279087, 0.7094597186702578, -0.2478694131670547],
                                    [0.49879920881388634, 0.6520368350439237, 0.5710055297740505, -0.7702320226107788]]),
                 ((1.0, 0.5, 0.25), [[-0.5373952243062219, -0.7279114951664748, 0.4258535289244096, 0.6624574144582671],
                                     [0.6874450494619515, -0.670605210309796, -0.27876146770242344, -0.17112581749074018],
                                     [0.4884932720755945, 0.14294581879111187, 0.860779191213994, -0.38888894660698226]])]
        for half, matrix in cases:
            matrix = np.array(matrix)
            g = Graph()
            node = g.move(g.add(BOX, [], [*half, 0]), matrix[:, 3], matrix[:, :3])
            for threads in (1, 0):
                mesh = self.mesh(g, node, 0.1, threads=threads)
                self.assertEqual(mesh.stats.components, 1)
                self.assertMeasures(mesh, 8 * np.prod(half), volume_tolerance=0.01)

    def test_offsets_of_empty_solids_and_single_precision_limits(self):
        g = Graph()
        ball = g.add(SPHERE, [], [1])
        gap = g.add(INTERSECTION, [g.move(ball, [-1.1, 0, 0]), g.move(ball, [1.1, 0, 0])], [0])
        grown = g.add(OFFSET, [gap], [0.5])
        self.assertEqual(self.k.mesh(self.build(g), gap, 0.05).stats.triangle_count, 0)
        mesh = self.mesh(g, grown, 0.05)
        self.assertEqual(mesh.stats.components, 1)
        far = g.move(ball, [1e6, -1e6, 1e6])
        with self.assertRaisesRegex(KernelError, "single-precision"):
            self.k.mesh(self.build(g), far, 0.05)
        self.mesh(g, far, 1.0)


class BenchmarkTests(KernelCase):
    def test_benchmarks(self):
        g = Graph()
        parts = []
        for i in range(8):  # 16 nodes
            parts.append(g.move(g.add(SPHERE, [], [2 + 0.2 * i]), [6 * math.cos(i), 6 * math.sin(i), i - 4]))
        for i in range(6):  # 12 nodes
            block = g.add(BOX, [], [1.5, 1, 2, 0.3])
            parts.append(g.move(block, [i * 2 - 5, 0, 0], rotation_matrix([0, 0, 1], 15 * i)))
        body = g.add(UNION, parts, [0.5])
        cutters = [g.move(g.add(CYLINDER, [], [0.8, 6]), [x, 0, 0]) for x in (-4, 0, 4)]  # 6 nodes
        cut = g.add(DIFFERENCE, [body, *cutters], [0])
        rim = g.add(EXTRUDE, [g.add(OUTLINE2, [g.add(CIRCLE, [], [9])], [1])], [-1, 1])  # 3 nodes
        tree = g.add(UNION, [cut, rim], [0.3])
        self.assertEqual(len(g.records), 40)
        handle = self.build(g)
        points = self.points(5_000_000, 12)
        start = time.perf_counter()
        self.k.evaluate(handle, tree, points)
        evaluate_seconds = time.perf_counter() - start

        sphere = Graph()
        sphere_node = sphere.add(SPHERE, [], [50])
        sphere_mesh = self.k.mesh(self.build(sphere), sphere_node, 0.25)

        plate = Graph()
        holes = [plate.move(plate.add(CYLINDER, [], [1.5, 4]), [x * 5 - 47.5, y * 5 - 22.5, 0])
                 for x in range(20) for y in range(10)]
        plate_node = plate.add(DIFFERENCE, [plate.add(BOX, [], [50, 25, 2.5, 0]), *holes], [0])
        plate_mesh = self.k.mesh(self.build(plate), plate_node, 0.1)
        self.assertEqual(plate_mesh.stats.boundary_edges, 0)
        self.assertEqual(plate_mesh.stats.nonmanifold_edges, 0)
        expected = 100 * 50 * 5 - 200 * math.pi * 1.5 ** 2 * 5
        self.assertAlmostEqual(plate_mesh.stats.volume / expected, 1, delta=0.002)
        print(f"\n  evaluate 5M points, 40-node tree: {evaluate_seconds * 1000:.0f} ms "
              f"({5e6 / evaluate_seconds / 1e6:.1f} M points/s)")
        for name, mesh in (("100 mm sphere, h=0.25", sphere_mesh), ("plate with 200 holes, h=0.1", plate_mesh)):
            s = mesh.stats
            print(f"  mesh {name}: {s.seconds * 1000:.0f} ms, {s.triangle_count:,} triangles, "
                  f"{s.active_cells:,} active cells, {s.crease_segment_count:,} crease segments")


if __name__ == "__main__":
    unittest.main()
