"""Exact-face proxies (sdf_build.build_proxies; SDF-DESIGN.md, "Proxy meshes"):
closed, outward meshes of each part's offset surface that lie outside the
solid within a stated shell. Needs the native kernel."""
import json
from pathlib import Path
import sys
import tempfile
import unittest
from unittest import mock

import numpy as np

ENGINE = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ENGINE))
import radsdf
from radsdf import SDFError
import sdf_build
import sdf_field


def kernel_available():
    try:
        radsdf.kernel_version()
        return True
    except SDFError:
        return False


def spec(script):
    return {"title": "Test", "summary": "A test design.", "units": "mm", "script": script}


# A concave base/wall crease (folds the mesher leaves for the repair), holes
# and a separate rounded part.
BRACKET = '''
base = rect((80, 50), radius=1).extrude(6)
wall = rect((80, 6), center=(0, 22), radius=1).extrude_between(6, 50)
holes = [cylinder(3.25, 8, center=(x, -6, 3)) for x in (-25, 25)]
holes += [cylinder(3.25, 10, center=(x, 23, 30), axis=(0, 1, 0)) for x in (-25, 25)]
result = [difference(base | wall, *holes), sphere(8, center=(0, -60, 8)) | box((30, 4, 4), center=(0, -60, 8))]
'''
PROXY_SECTIONS = {"proxyPositions": ("<f4", 3), "proxyTriangles": ("<u4", 3), "proxySolids": ("<i4", 1)}
# Barycentric weights of a dense triangle sample (55 points, corners and edges included).
DENSE = np.array([(i / 9, j / 9, (9 - i - j) / 9) for i in range(10) for j in range(10 - i)])


def proxy_arrays(output, result):
    data = (output / "mesh.bin").read_bytes()
    arrays = {}
    for name, (dtype, width) in PROXY_SECTIONS.items():
        section = result["meshBuffers"]["sections"].get(name)
        if section is None:
            continue
        values = np.frombuffer(data, dtype=dtype, count=section["count"], offset=section["offset"])
        arrays[name] = values.reshape(-1, width) if width > 1 else values
    return arrays


@unittest.skipUnless(kernel_available(), "the native SDF kernel is not available")
class ProxyBuildTests(unittest.TestCase):
    def build(self, script, threads=4):
        temporary = tempfile.TemporaryDirectory(prefix="sdf-proxy-")
        self.addCleanup(temporary.cleanup)
        output = Path(temporary.name)
        returned = sdf_build.build(spec(script), output, output, threads)
        result = json.loads((output / "result.json").read_text())
        return output, result, proxy_arrays(output, result), returned

    def test_proxies_follow_the_protocol(self):
        output, result, arrays, returned = self.build(BRACKET)
        keys = list(result)
        self.assertEqual(keys[keys.index("fieldProgram") + 1], "proxy")
        proxy = result["proxy"]
        self.assertEqual(list(proxy), ["offset", "cellSize", "tolerance", "buildSeconds", "parts"])
        cell = result["mesh"]["cellSize"]
        self.assertAlmostEqual(proxy["offset"], sdf_build.PROXY_OFFSET_CELLS * cell)
        self.assertAlmostEqual(proxy["cellSize"], sdf_build.PROXY_CELL_FACTOR * cell)
        self.assertAlmostEqual(proxy["tolerance"], sdf_build.PROXY_TOLERANCE_FRACTION * proxy["offset"])
        self.assertGreater(proxy["buildSeconds"], 0)
        self.assertEqual([part["solid"] for part in proxy["parts"]], [0, 1])
        self.assertEqual([part["status"] for part in proxy["parts"]], ["proxy", "proxy"])
        for part in proxy["parts"]:
            self.assertEqual(list(part), ["solid", "status", "triangles", "shell", "minimum", "maxDeviation"])
            self.assertGreater(part["minimum"], sdf_build.PROXY_MINIMUM_FRACTION * proxy["offset"])
            self.assertGreaterEqual(part["shell"], sdf_build.PROXY_SHELL_FACTOR * proxy["offset"])
        positions, triangles, solids = arrays["proxyPositions"], arrays["proxyTriangles"], arrays["proxySolids"]
        self.assertEqual(len(solids), len(triangles))
        self.assertTrue(np.all(np.diff(solids) >= 0), "triangles are grouped by solid in ascending order")
        self.assertEqual([int(np.count_nonzero(solids == part["solid"])) for part in proxy["parts"]],
                         [part["triangles"] for part in proxy["parts"]])
        self.assertLessEqual(len(triangles), sdf_build.PROXY_TRIANGLE_BUDGET)
        self.assertEqual(int(triangles.max()) + 1, len(positions))
        for solid in (0, 1):  # each part's vertices are one block, used by its triangles only
            used = np.unique(triangles[solids == solid])
            self.assertEqual(int(used[-1] - used[0]) + 1, len(used))
        # The returned result keeps the arrays; result.json carries only their sections.
        np.testing.assert_array_equal(returned["proxyVertices"], positions)
        np.testing.assert_array_equal(returned["proxyTriangles"], triangles)
        np.testing.assert_array_equal(returned["proxyTriangleSolids"], solids)
        for key in ("proxyVertices", "proxyTriangles", "proxyTriangleSolids"):
            self.assertNotIn(key, result)
        # model.rsdf holds the design's nodes only: no appended offset nodes.
        normalized, _ = sdf_build.run_design(spec(BRACKET), output)
        data, roots, _ = sdf_build.encode_design(normalized)
        self.assertEqual((output / "model.rsdf").read_bytes(), data)
        self.assertEqual(result["partRoots"], roots)

    def test_proxies_are_closed_outside_outward_and_within_their_shells(self):
        output, result, arrays, _ = self.build(BRACKET)
        delta = result["proxy"]["offset"]
        positions = arrays["proxyPositions"].astype(np.float64)
        with radsdf.Graph.from_bytes((output / "model.rsdf").read_bytes(), result["partRoots"]) as graph:
            for part, stats in zip(result["proxy"]["parts"], result["partStats"]):
                solid, root = part["solid"], result["partRoots"][part["solid"]]
                own = arrays["proxyTriangles"][arrays["proxySolids"] == solid].astype(np.int64)
                self.assertTrue(sdf_build.closed_surface(own, len(positions)))
                corners = positions[own]
                values = graph.evaluate(root, np.einsum("sk,tkd->tsd", DENSE, corners).reshape(-1, 3))
                self.assertGreater(values.min(), 0, "every point of every triangle is outside the solid")
                self.assertLessEqual(values.max(), part["shell"])
                self.assertLessEqual(part["minimum"], values.min() + 0.15 * delta)
                normals, usable = sdf_build._triangle_normals(corners)
                gradients = graph.gradient(root, corners.mean(axis=1))
                facing = np.einsum("td,td->t", normals, gradients)
                self.assertGreater(facing.sum(), 0.97 * np.linalg.norm(normals, axis=1).sum(), "outward")
                volume = np.einsum("td,td->t", corners[:, 0], np.cross(corners[:, 1], corners[:, 2])).sum() / 6
                self.assertGreater(volume, stats["volume"])
                _, _, flipped = sdf_build.proxy_samples(graph, root, positions, own, delta, 2)
                self.assertFalse(flipped.any())

    def test_parts_without_a_field_program_get_no_proxy(self):
        build = sdf_field.build

        def one_mesh_only(graph, cell, threads):
            program, failure = build(graph, cell, threads)
            program.parts[1] = {"status": "meshOnly", "reason": "over a limit"}
            return program, failure

        with mock.patch.object(sdf_field, "build", one_mesh_only):
            _, result, arrays, _ = self.build(BRACKET)
        self.assertEqual(result["fieldProgram"]["parts"][1]["status"], "meshOnly")
        self.assertEqual(result["proxy"]["parts"][1], {"solid": 1, "status": "none", "reason": "no GPU field program"})
        self.assertEqual(result["proxy"]["parts"][0]["status"], "proxy")
        self.assertEqual(set(arrays["proxySolids"].tolist()), {0})
        with mock.patch.object(sdf_field, "build", lambda graph, cell, threads: (None, "the kernel said no")):
            output, result, arrays, _ = self.build("result = box(10)")
        self.assertEqual(result["fieldProgram"], {"error": "the kernel said no"})
        self.assertEqual(result["proxy"], {"error": "the kernel said no"})
        self.assertEqual(arrays, {})
        self.assertEqual(result["solids"], 1)

    def test_failed_checks_retry_and_never_fail_the_build(self):
        failures = iter(["first", "second"])
        with mock.patch.object(sdf_build, "checked_proxy", side_effect=lambda *args: (next(failures, "more"), None)), \
                mock.patch.object(sdf_build, "mesh_proxy", wraps=sdf_build.mesh_proxy) as meshed:
            _, result, arrays, _ = self.build("result = box(10)")
        self.assertEqual(result["proxy"]["parts"], [{"solid": 0, "status": "none", "reason": "second"}])
        tolerances = [call.args[3] for call in meshed.call_args_list]
        self.assertEqual(len(tolerances), 2)
        self.assertAlmostEqual(tolerances[1], tolerances[0] / 2)
        self.assertEqual(arrays, {})
        self.assertGreater(result["volume"], 0)

    def test_the_budget_coarsens_or_omits_proxies(self):
        with mock.patch.object(sdf_build, "PROXY_TRIANGLE_BUDGET", 3000):
            _, result, arrays, _ = self.build(BRACKET)
        parts = result["proxy"]["parts"]
        self.assertLessEqual(len(arrays.get("proxyTriangles", ())), 3000)
        for part in parts:
            if part["status"] == "proxy":
                if "cellSize" in part:
                    self.assertAlmostEqual(part["cellSize"], 2 * result["proxy"]["cellSize"])
            else:
                self.assertIn("budget", part["reason"])
        self.assertIn("proxy", [part["status"] for part in parts])

    def test_proxies_do_not_depend_on_the_thread_count(self):
        outputs = [self.build(BRACKET, threads) for threads in (1, 3)]
        (_, first, one, _), (_, second, three, _) = outputs
        self.assertEqual([dict(first["proxy"], buildSeconds=0)], [dict(second["proxy"], buildSeconds=0)])
        for name in PROXY_SECTIONS:
            self.assertEqual(one[name].tobytes(), three[name].tobytes())


@unittest.skipUnless(kernel_available(), "the native SDF kernel is not available")
class ProxyGeometryTests(unittest.TestCase):
    DELTA = 0.1

    def test_offset_nodes_grow_their_roots(self):
        shapes = [radsdf.box((10, 8, 4), radius=1), radsdf.sphere(3, center=(12, 0, 0))]
        data, roots = radsdf.serialize(shapes)
        augmented, nodes = sdf_build.offset_graph_bytes(data, roots, self.DELTA)
        self.assertEqual(augmented[16:len(data)], data[16:])
        points = np.random.default_rng(3).uniform(-8, 16, size=(500, 3))
        with radsdf.Graph.from_bytes(augmented, roots) as graph:
            self.assertEqual(graph.roots, tuple(roots))
            self.assertEqual(nodes, [graph.node_count - 2, graph.node_count - 1])
            for root, node in zip(roots, nodes):
                np.testing.assert_allclose(graph.evaluate(node, points), graph.evaluate(root, points) - self.DELTA,
                                           atol=1e-12)

    def triangle_flipped(self, shape, corners):
        corners = np.array([corners], dtype=np.float64)
        with radsdf.Graph([shape]) as graph:
            normals, usable = sdf_build._triangle_normals(corners)
            gradients = graph.gradient(graph.roots[0], corners.mean(axis=1))
            return bool(sdf_build._flipped(graph, graph.roots[0], corners, normals, usable, gradients, self.DELTA,
                                           1)[0])

    def test_folds_are_flipped_but_ridges_are_not(self):
        slab = radsdf.box((10, 10, 2))  # top face at z = 1
        z = 1 + self.DELTA
        outward = [(0, 0, z), (1, 0, z), (0, 1, z)]
        inward = [(0, 0, z), (0, 1, z), (1, 0, z)]
        self.assertFalse(self.triangle_flipped(slab, outward))
        self.assertTrue(self.triangle_flipped(slab, inward))
        # Two slabs 2 delta apart: their offset surfaces touch on the ridge between them.
        gap = slab | radsdf.box((10, 10, 2), center=(0, 0, 2 + 2 * self.DELTA))
        self.assertFalse(self.triangle_flipped(gap, outward))
        self.assertFalse(self.triangle_flipped(gap, inward))
        # A sliver has no normal to face against anything.
        self.assertFalse(self.triangle_flipped(slab, [(0, 0, z), (1, 0, z), (2, 1e-9, z)]))

    @mock.patch.object(sdf_build, "PROXY_SIMPLIFY_FRACTION", 0.0)
    def test_repair_untangles_a_fold(self):
        shape = radsdf.box((10, 10, 4), radius=1)
        data, roots = radsdf.serialize([shape])
        augmented, nodes = sdf_build.offset_graph_bytes(data, roots, self.DELTA)
        with radsdf.Graph.from_bytes(augmented, roots) as graph:
            mesh = sdf_build.mesh_proxy(graph, nodes[0], 0.5, self.DELTA / 3, None, 2)
            positions, triangles = sdf_build.weld_proxy(mesh.positions, mesh.triangles)
            self.assertTrue(sdf_build.closed_surface(triangles, len(positions)))
            _, _, flipped = sdf_build.proxy_samples(graph, roots[0], positions, triangles, self.DELTA, 2)
            self.assertFalse(flipped.any())
            # Push a vertex in the middle of the top face past a neighbour: its fan folds.
            top = np.nonzero((np.abs(positions[:, 2] - (2 + self.DELTA)) < 1e-4)
                             & np.all(np.abs(positions[:, :2]) < 2, axis=1))[0]
            vertex = int(top[0])
            ring = np.unique(triangles[np.any(triangles == vertex, axis=1)])
            neighbour = positions[[index for index in ring if index != vertex][0]]
            folded = positions.copy()
            folded[vertex] = neighbour + 0.6 * (neighbour - positions[vertex])
            lowest, _, flipped = sdf_build.proxy_samples(graph, roots[0], folded, triangles, self.DELTA, 2)
            self.assertTrue(flipped.any())
            repair = sdf_build.ProxyRepair(graph, roots[0], folded, triangles, self.DELTA, 0.1 * self.DELTA)
            self.assertEqual(repair.run(np.nonzero(flipped)[0]), [])
            self.assertGreater(repair.edits, 0)
            repaired = repair.positions.astype(np.float32)
            self.assertTrue(sdf_build.closed_surface(repair.triangles, len(repaired)))
            lowest, _, flipped = sdf_build.proxy_samples(graph, roots[0], repaired, repair.triangles, self.DELTA, 2)
            self.assertFalse(flipped.any())
            self.assertGreater(lowest, 0.1 * self.DELTA)
            # checked_proxy repairs the same fold in a mesh.
            fake = radsdf.Mesh(folded.astype(np.float32), mesh.normals[:len(folded)], triangles, mesh.creases,
                               mesh.stats)
            failure, checked = sdf_build.checked_proxy(graph, roots[0], fake, self.DELTA, 2)
            self.assertIsNone(failure)
            _, _, flipped = sdf_build.proxy_samples(graph, roots[0], checked[0], checked[1], self.DELTA, 2)
            self.assertFalse(flipped.any())

    def test_weld_merges_coincident_vertices_in_order_of_first_use(self):
        positions = np.array([[0, 0, 0], [1, 0, 0], [0, 1, 0], [1, 0, 0], [0, 0, 1]], dtype=np.float32)
        triangles = np.array([[0, 1, 2], [3, 2, 4]], dtype=np.uint32)
        welded, faces = sdf_build.weld_proxy(positions, triangles)
        np.testing.assert_array_equal(welded, positions[[0, 1, 2, 4]])
        np.testing.assert_array_equal(faces, [[0, 1, 2], [1, 2, 3]])
        tetrahedron = np.array([[0, 2, 1], [0, 1, 3], [1, 2, 3], [2, 0, 3]])
        self.assertTrue(sdf_build.closed_surface(tetrahedron, 4))
        self.assertFalse(sdf_build.closed_surface(tetrahedron[:3], 4))
        self.assertFalse(sdf_build.closed_surface(np.array([[0, 2, 1], [0, 1, 3], [1, 2, 3], [2, 3, 0]]), 4))


if __name__ == "__main__":
    unittest.main()
