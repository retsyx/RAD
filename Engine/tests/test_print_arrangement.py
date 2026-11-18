"""Printing orientation from part meshes, checked against the parts' fields."""
import math
import unittest
from unittest import mock

import numpy as np

from sdf_test_support import build, native
import radsdf
from sdf_build import ModelError
import sdf_ops


def matrix_of(values):
    return np.asarray(values, dtype=np.float64).reshape(4, 4)


class PrintArrangementTests(unittest.TestCase):
    def source(self, script):
        """(Source, build result) of a program's parts."""
        result, output = build(self, script)
        data = (output / "model.rsdf").read_bytes()
        source = sdf_ops.Source(radsdf.Graph.from_bytes(data, result["partRoots"]), data,
                                tuple(result["partRoots"]))
        self.addCleanup(source.close)
        return source, result, output

    def placed(self, source, solid, record, cell=0.05):
        """Vertices of a fine mesh of the part moved by an orientation."""
        matrix = matrix_of(record[0])
        mesh = source.graph.mesh(source.roots[solid], cell_size=cell)
        return mesh.positions.astype(np.float64) @ matrix[:3, :3].T + matrix[:3, 3], mesh

    def orient(self, source, solids=None, pins=None):
        solids = list(range(source.solids)) if solids is None else solids
        return sdf_ops.print_orientations(source, solids, pins or [None] * len(solids), 2)

    def assert_oriented(self, source, records, solids=None, cell=0.05):
        """Reported bounds match the moved part (to a fraction of its sampling
        cell); footprints are wider than deep."""
        for solid, record in zip(solids or range(source.solids), records):
            points, _ = self.placed(source, solid, record, cell)
            actual = [*points.min(axis=0), *points.max(axis=0)]
            sampling = source.sample_mesh(solid, 2).stats["cell_size"]
            for reported, expected in zip(record[1], actual):
                self.assertAlmostEqual(reported, expected, delta=0.6 * sampling + 0.02)
            box = record[1]
            self.assertGreaterEqual(box[3] - box[0], box[4] - box[1] - 1e-9, "Footprints are turned wider than deep")
            self.assertEqual(record[0][12:], [0.0, 0.0, 0.0, 1.0])
            rotation = matrix_of(record[0])[:3, :3]
            self.assertTrue(np.allclose(rotation @ rotation.T, np.eye(3), atol=1e-12))
            self.assertAlmostEqual(np.linalg.det(rotation), 1, places=12)

    def height(self, record):
        return record[1][5] - record[1][2]

    def assert_height(self, source, solid, record, expected):
        """Heights hold to a fraction of the part's sampling cell."""
        cell = source.sample_mesh(solid, 2).stats["cell_size"]
        self.assertAlmostEqual(self.height(record), expected, delta=0.3 * cell)

    def test_orients_each_part_on_the_bed_without_changing_the_artifact(self):
        source, result, output = self.source("""
result = [box((8, 18, 2)).rotate(61, (1, 2, 1)).translate((30, -5, 17)),
          cylinder(3, 14).rotate_y(90).translate((-9, 21, -4)),
          box((5, 7, 9), center=(20.5, 23.5, 37.5)),
          sphere(4, center=(80, 60, 20)) & half_space((0, 0, -1), (80, 60, 20))]
""")
        artifact = (output / "model.rsdf").read_bytes()
        records = self.orient(source)
        self.assert_oriented(source, records)
        for solid, expected in enumerate((2, 14, 5, 4)):
            # Boxes rest on their largest face, the cylinder on its end and
            # the dome on its flat face.
            self.assert_height(source, solid, records[solid], expected)
        self.assertEqual(artifact, (output / "model.rsdf").read_bytes())

    def test_translated_instances_share_orientation_only_within_one_request(self):
        source, _, _ = self.source("""
part = box((8, 15, 2))
copies = [part.translate((index * 30, index * -4, 10 + index)) for index in range(4)]
rotated = part.rotate(37, (1, 2, 3)).translate((50, 20, 30))
result = copies + [rotated, rotated.translate((100, -20, 40))]
""")
        with mock.patch.object(sdf_ops, "print_orientation", wraps=sdf_ops.print_orientation) as orient:
            records = self.orient(source)
            self.assertEqual(orient.call_count, 2, "Translation shares work; a distinct rotation is scored independently")
            self.orient(source)
            self.assertEqual(orient.call_count, 4, "Orientation statistics must not survive a request")
            self.orient(source, [0, 1], [None, (0, 0, -1)])
            self.assertEqual(orient.call_count, 5, "A pinned instance needs no search and is not shared")
        self.assert_oriented(source, records)
        turns = [matrix_of(record[0])[:3, :3] for record in records]
        for turn in turns[1:4]:
            self.assertTrue(np.array_equal(turn, turns[0]))
        self.assertTrue(np.array_equal(turns[4], turns[5]))
        self.assertEqual([record[1] for record in records[:4]], [records[0][1]] * 4)

    def test_splitting_a_part_into_coplanar_pieces_does_not_change_candidate_directions(self):
        source, _, _ = self.source("""
result = [box((10, 12, 3), center=(5, 6, 1.5)),
          box((5, 12, 3), center=(2.5, 6, 1.5)) | box((5, 12, 3), center=(7.5, 6, 1.5))]
""")
        candidates = []
        for solid in range(2):
            surface = sdf_ops.PrintSurface(source.sample_mesh(solid, 2))
            self.assertAlmostEqual(surface.weights.sum(), 2 * (120 + 30 + 36), delta=0.01)
            directions = sdf_ops.print_candidates(surface)
            candidates.append({tuple(round(float(value), 6) for value in direction) for direction in directions})
        self.assertEqual(candidates[0], candidates[1])
        for record in self.orient(source):
            self.assert_height(source, 0, record, 3)

    def test_freeform_samples_are_area_weighted_unit_normals(self):
        source, _, _ = self.source("""
loft = ellipse((7, 5)).loft_to(ellipse((5, 7)), 9)
result = loft - cylinder(2, 20, center=(4, 0, 0))
""")
        mesh = source.sample_mesh(0, 2)
        surface = sdf_ops.PrintSurface(mesh)
        self.assertGreater(len(surface.normals), len(surface.plane_areas) + 10)
        self.assertTrue(np.allclose(np.linalg.norm(surface.normals, axis=1), 1, atol=1e-12))
        self.assertTrue((surface.weights > 0).all())
        corners = mesh.positions.astype(np.float64)[mesh.triangles]
        area = 0.5 * np.linalg.norm(np.cross(corners[:, 1] - corners[:, 0], corners[:, 2] - corners[:, 0]), axis=1)
        self.assertAlmostEqual(surface.weights.sum(), area.sum(), places=9)
        [record] = self.orient(source)
        self.assert_oriented(source, [record])

    def test_rotated_cylinder_and_tube_print_along_their_axes(self):
        for shape, height, diameter in [("cylinder(3, 17)", 17, 6), ("cylinder(5, 19) - cylinder(4, 20)", 19, 10)]:
            with self.subTest(shape=shape):
                source, _, _ = self.source(f"result = ({shape}).rotate(73, (1, 2, 3)).translate((10, 20, 30))")
                [record] = self.orient(source)
                self.assert_oriented(source, [record])
                box, cell = record[1], source.sample_mesh(0, 2).stats["cell_size"]
                self.assert_height(source, 0, record, height)
                self.assertAlmostEqual(box[3] - box[0], diameter, delta=0.3 * cell)
                self.assertAlmostEqual(box[4] - box[1], diameter, delta=0.3 * cell)

    def test_recess_floor_is_not_mistaken_for_bed_contact(self):
        source, _, _ = self.source("""
tray = box((30, 25, 8), center=(15, 12.5, 4)) - box((26, 21, 7), center=(15, 12.5, 5.5))
result = tray.rotate_x(180).translate((10, 30, 40))
""")
        [record] = self.orient(source)
        self.assert_oriented(source, [record])
        # The tray was built upside down: its outer floor must face the bed again.
        rotation = matrix_of(record[0])[:3, :3]
        self.assertAlmostEqual((rotation @ [0, 0, 1])[2], -1, places=9)
        self.assert_height(source, 0, record, 8)

    def enclosure(self, depth):
        return f"""
depth = {depth}
outer = rect((59, 33.6), radius=4).extrude(depth, center=True)
inner = rect((56.6, 31.2), radius=2.8).extrude(depth + 2, center=True).translate((0, 0, 2.2))
shape = outer - inner - box((14, 8, 6), center=(0, -16, 2))
for x in (-21, 21):
    for y in (-10, 10):
        boss = cylinder(2.8, depth - 1.2, center=(x, y, 0.6))
        bore = cylinder(1.2, depth + 2, center=(x, y, 1))
        shape = (shape | boss) - bore
"""

    def test_rounded_enclosure_halves_rest_on_exterior_bases(self):
        for depth, rotation in ((14, "shape.rotate(73, (1, 2, 3))"), (8, "shape.rotate_y(180)")):
            with self.subTest(depth=depth):
                source, _, _ = self.source(self.enclosure(depth) + f"result = {rotation}\n")
                [record] = self.orient(source)
                self.assert_oriented(source, [record])
                self.assert_height(source, 0, record, depth)
                # The built part's outer floor (-Z before its rotation) faces the bed.
                built = matrix_of(record[0])[:3, :3]
                angle, axis = (73, (1, 2, 3)) if depth == 14 else (180, (0, 1, 0))
                floor = built @ (radsdf.rotation_matrix(angle, axis) @ [0, 0, -1])
                self.assertAlmostEqual(floor[2], -1, places=6, msg="The enclosure must not balance on an edge")

    def test_contact_stability_uses_boundary_support_with_holes_and_islands(self):
        angles = np.linspace(0, 2 * math.pi, 64, endpoint=False)
        circle = np.stack([np.cos(angles), np.sin(angles), np.zeros_like(angles)], axis=1)
        feet = np.array([[x, y, 0] for x in (-12, -9, 9, 12) for y in (-3, 3)], dtype=float)
        rotation = radsdf.rotation_matrix(67, (1, 2, 3))
        direction = rotation @ [0, 0, 1]
        for name, points, centroid in [("disk", 5 * circle, (0, 0, 2)),
                                       ("ring", np.concatenate([5 * circle, 3 * circle]), (0, 0, 2)),
                                       ("separate feet", feet, (0, 0, 4))]:
            with self.subTest(contact=name):
                self.assertTrue(sdf_ops.contact_is_stable(points, (0, 0, 1), centroid))
                moved = points @ rotation.T + (17, -12, 20)
                self.assertTrue(sdf_ops.contact_is_stable(moved, direction, rotation @ centroid + (17, -12, 20)))
                self.assertFalse(sdf_ops.contact_is_stable(points, (0, 0, 1), (100, 0, 2)))

    def test_contact_stability_rejects_an_eccentric_pad(self):
        source, _, _ = self.source("result = box(2, center=(-9, 0, 1)) | box((24, 4, 2), center=(2, 0, 3))")
        surface = sdf_ops.PrintSurface(source.sample_mesh(0, 2))
        self.assertGreater(surface.centroid[0], -8, "The volume centroid lies outside the support pad")
        bottom = [plane for plane, normal in enumerate(surface.plane_normals)
                  if normal[2] < -0.99999 and abs(surface.plane_centers[plane][2]) < 1e-6]
        top = [plane for plane, normal in enumerate(surface.plane_normals)
               if normal[2] > 0.99999 and abs(surface.plane_centers[plane][2] - 4) < 1e-6]
        self.assertTrue(bottom and top)
        support = np.concatenate([surface.support(plane) for plane in bottom])
        self.assertFalse(sdf_ops.contact_is_stable(support, (0, 0, 1), surface.centroid))
        support = np.concatenate([surface.support(plane) for plane in top])
        self.assertTrue(sdf_ops.contact_is_stable(support, (0, 0, -1), surface.centroid))

    def test_parts_without_a_resting_plane_keep_their_supplied_pose(self):
        # Facets of a meshed sphere are not planes it could rest on, and
        # every pose scores alike, so +Z stays up. (Its footprint is as wide as
        # deep, so the quarter turn about Z follows sampling noise.)
        source, _, _ = self.source("result = sphere(4, center=(20, 5, 1))")
        surface = sdf_ops.PrintSurface(source.sample_mesh(0, 2))
        self.assertEqual(len(surface.plane_areas), 0)
        [record] = self.orient(source)
        self.assertTrue(np.allclose(matrix_of(record[0])[:3, 2], [0, 0, 1]))
        self.assert_oriented(source, [record])

    def test_steep_cone_uses_its_curved_normals_to_avoid_downward_overhang(self):
        source, _, _ = self.source("result = cone(10, 1, 4).rotate(151, (1, 2, 3))")
        [record] = self.orient(source)
        self.assert_oriented(source, [record])
        self.assert_height(source, 0, record, 4)
        base = matrix_of(record[0])[:3, :3] @ (radsdf.rotation_matrix(151, (1, 2, 3)) @ [0, 0, -1])
        self.assertAlmostEqual(base[2], -1, places=6, msg="The wide base faces the bed.")

    def test_bracket_avoids_unsupported_planar_undersides(self):
        source, _, _ = self.source("""
bracket = box((20, 14, 2), center=(10, 7, 1)) | box((2, 14, 16), center=(1, 7, 8))
result = bracket.rotate_y(180)
""")
        [record] = self.orient(source)
        points, mesh = self.placed(source, 0, record, cell=0.1)
        bottom = points[:, 2].min()
        corners = points[mesh.triangles]
        normals = np.cross(corners[:, 1] - corners[:, 0], corners[:, 2] - corners[:, 0])
        areas = np.linalg.norm(normals, axis=1) / 2
        unit = normals / np.maximum(2 * areas, 1e-300)[:, None]
        overhang = (unit[:, 2] < -math.sqrt(0.5)) & (corners[:, :, 2].mean(axis=1) > bottom + 1e-3)
        self.assertLess(areas[overhang].sum(), 0.05)

    def test_pinned_direction_faces_the_bed_and_replaces_the_search(self):
        source, _, _ = self.source("result = box((40, 20, 4), center=(20, 10, 2)) | box((4, 20, 30), center=(2, 10, 15))")
        with mock.patch.object(sdf_ops, "print_orientation", side_effect=AssertionError("No search")):
            [pinned] = self.orient(source, [0], [(0, 0, 1)])
            [side] = self.orient(source, [0], [(-1, 0, 0)])
        self.assert_oriented(source, [pinned])
        self.assert_oriented(source, [side])
        self.assert_height(source, 0, pinned, 30)
        self.assert_height(source, 0, side, 40)
        self.assertAlmostEqual((matrix_of(pinned[0])[:3, :3] @ [0, 0, 1])[2], -1, places=12)

    def test_native_orientation_request_returns_rigid_deltas_for_selected_parts(self):
        result, output = build(self, """
result = [box((4, 8, 2), center=(82, 16, 11)),
          cylinder(3, 12).rotate_x(90).translate((-20, 30, 20)),
          box((3, 4, 5), center=(13.5, -38, 72.5))]
""")
        report, folder = native(output, result, "print_orient", solidIndices=[1, 2],
                                printDown=[{"solid": 2, "direction": [0, 0, 2]}])
        self.assertFalse((folder / "model.stl").exists())
        records = report["orientations"]
        self.assertEqual([record["solid"] for record in records], [1, 2])
        data = (output / "model.rsdf").read_bytes()
        source = sdf_ops.Source(radsdf.Graph.from_bytes(data, result["partRoots"]), data, tuple(result["partRoots"]))
        self.addCleanup(source.close)
        self.assert_oriented(source, [(record["matrix"], record["bounds"]) for record in records], [1, 2])
        self.assert_height(source, 1, (records[0]["matrix"], records[0]["bounds"]), 12)
        self.assert_height(source, 2, (records[1]["matrix"], records[1]["bounds"]), 5)
        # The pinned top face now faces the bed.
        self.assertAlmostEqual(records[1]["matrix"][10], -1, places=9)

    def test_orientation_requests_validate_pins_and_export_rejects_removed_arrangement(self):
        result, output = build(self, "result = box((1, 2, 3))")
        for pins in ([{"solid": 0, "direction": [0, 0, 0]}], [{"solid": 1, "direction": [0, 0, 1]}],
                     [{"solid": 0, "direction": [0, 0]}], [{"solid": 0}], "down",
                     [{"solid": 0, "direction": [0, 0, 1]}, {"solid": 0, "direction": [0, 0, -1]}],
                     [{"solid": 0, "direction": [0, 0, True]}], [{"solid": 0, "direction": [0, 0, math.nan]}]):
            with self.subTest(pins=pins), self.assertRaises(ModelError):
                native(output, result, "print_orient", printDown=pins)
        with self.assertRaises(ModelError):
            native(output, result, "export", arrangeFor3DPrinting=True)
        self.assertEqual(list(output.parent.glob("*/model.stl")), [])


if __name__ == "__main__":
    unittest.main()
