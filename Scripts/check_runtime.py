"""Smoke-test the packaged standalone Python, NumPy and, when given, RAD's SDF kernel.

With RAD_SDF_LIBRARY naming a libRADSDF.dylib, the check also loads the kernel
through ctypes with this interpreter (exercising code signing and library
validation), then evaluates and meshes a sphere through the C API in radsdf.h.
Payload preparation runs without it, because the payload is prepared before
the kernel is built.
"""
import ctypes
import ensurepip
import math
import os
import ssl
import struct
import venv

import numpy as np

# The macOS 14 NumPy wheel performs linear algebra with Accelerate.
matrix = np.array([[4.0, 1.0, 0.0], [1.0, 3.0, 1.0], [0.0, 1.0, 2.0]])
solution = np.linalg.solve(matrix, [1.0, 2.0, 3.0])
assert np.allclose(matrix @ solution, [1.0, 2.0, 3.0])
values, vectors = np.linalg.eigh(matrix)
assert np.allclose(vectors @ np.diag(values) @ vectors.T, matrix)
assert ssl.OPENSSL_VERSION_INFO >= (3,) and ensurepip.version() and venv.EnvBuilder
checks = ["NumPy linear algebra", "ssl", "ensurepip", "venv", "ctypes"]


class MeshOptions(ctypes.Structure):
    _fields_ = [("cell_size", ctypes.c_double), ("crease_angle", ctypes.c_double),
                ("max_cells", ctypes.c_uint64), ("threads", ctypes.c_int32),
                ("refine_steps", ctypes.c_int32), ("bounds_lo", ctypes.c_double * 3),
                ("bounds_hi", ctypes.c_double * 3), ("bounds_valid", ctypes.c_int32),
                ("max_depth", ctypes.c_int32), ("tolerance", ctypes.c_double), ("max_triangles", ctypes.c_uint64)]


class MeshStats(ctypes.Structure):
    _fields_ = [("vertex_count", ctypes.c_uint64), ("triangle_count", ctypes.c_uint64),
                ("crease_segment_count", ctypes.c_uint64), ("active_cells", ctypes.c_uint64),
                ("boundary_edges", ctypes.c_uint64), ("nonmanifold_edges", ctypes.c_uint64),
                ("components", ctypes.c_uint64), ("cavities", ctypes.c_uint64), ("volume", ctypes.c_double),
                ("area", ctypes.c_double), ("cavity_volume", ctypes.c_double), ("centroid", ctypes.c_double * 3), ("bounds_lo", ctypes.c_double * 3),
                ("bounds_hi", ctypes.c_double * 3), ("cell_size", ctypes.c_double),
                ("seconds", ctypes.c_double), ("max_deviation", ctypes.c_double), ("tolerance", ctypes.c_double),
                ("refined_cells", ctypes.c_uint64), ("balanced_cells", ctypes.c_uint64),
                ("flagged_triangles", ctypes.c_uint64), ("max_level", ctypes.c_uint32), ("passes", ctypes.c_uint32)]


def check_kernel(path):
    kernel = ctypes.CDLL(path)
    doubles = ctypes.POINTER(ctypes.c_double)
    for name, restype, argtypes in [
            ("rsdf_abi_version", ctypes.c_uint32, []),
            ("rsdf_kernel_version", ctypes.c_char_p, []),
            ("rsdf_last_error", ctypes.c_char_p, []),
            ("rsdf_graph_create", ctypes.c_void_p, [ctypes.c_char_p, ctypes.c_size_t]),
            ("rsdf_graph_free", None, [ctypes.c_void_p]),
            ("rsdf_evaluate", ctypes.c_int, [ctypes.c_void_p, ctypes.c_uint32, doubles, ctypes.c_uint64,
                                             doubles, ctypes.c_int32]),
            ("rsdf_mesh_build", ctypes.c_void_p, [ctypes.c_void_p, ctypes.c_uint32, ctypes.POINTER(MeshOptions)]),
            ("rsdf_mesh_get_stats", None, [ctypes.c_void_p, ctypes.POINTER(MeshStats)]),
            ("rsdf_mesh_free", None, [ctypes.c_void_p])]:
        function = getattr(kernel, name)
        function.restype, function.argtypes = restype, argtypes
    assert kernel.rsdf_abi_version() == 3, kernel.rsdf_abi_version()
    # RSDF version 1: one SPHERE node (opcode 1, no children, radius 5).
    graph_bytes = b"RSDF" + struct.pack("<III", 1, 1, 0) + struct.pack("<IIIId", 1, 0, 1, 0, 5.0)
    graph = kernel.rsdf_graph_create(graph_bytes, len(graph_bytes))
    assert graph, kernel.rsdf_last_error()
    try:
        points = np.array([[0.0, 0.0, 0.0], [5.0, 0.0, 0.0], [0.0, 0.0, 8.0]])
        distances = np.empty(len(points))
        status = kernel.rsdf_evaluate(graph, 0, points.ctypes.data_as(doubles), len(points),
                                      distances.ctypes.data_as(doubles), 2)
        assert status == 0, kernel.rsdf_last_error()
        assert np.allclose(distances, [-5.0, 0.0, 3.0]), distances
        options = MeshOptions(cell_size=0.25, crease_angle=35.0, threads=2, refine_steps=6)
        mesh = kernel.rsdf_mesh_build(graph, 0, ctypes.byref(options))
        assert mesh, kernel.rsdf_last_error()
        try:
            stats = MeshStats()
            kernel.rsdf_mesh_get_stats(mesh, ctypes.byref(stats))
        finally:
            kernel.rsdf_mesh_free(mesh)
        expected = 4.0 / 3.0 * math.pi * 5.0 ** 3
        assert stats.triangle_count and stats.boundary_edges == 0 and stats.nonmanifold_edges == 0, \
            (stats.triangle_count, stats.boundary_edges, stats.nonmanifold_edges)
        assert abs(stats.volume - expected) < 0.01 * expected, stats.volume
    finally:
        kernel.rsdf_graph_free(graph)
    version = (kernel.rsdf_kernel_version() or b"").decode("utf-8", "replace")
    return f"SDF kernel {version or 'unknown version'} (ABI 3, evaluation, meshing)"


library = os.environ.get("RAD_SDF_LIBRARY")
if library:
    checks.append(check_kernel(library))
print("Standalone runtime checks passed: " + ", ".join(checks) + ".")
