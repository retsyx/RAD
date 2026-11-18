"""Batch 3MF export preserves parts, their coordinates and shared print placement."""
import unittest
from unittest import mock

import numpy as np

from sdf_test_support import assert_closed_and_outward, build, native, read_3mf, signed_volume
import sdf_build
import sdf_ops

FIXTURE = """
result = [box((8, 14, 2)).rotate(61, (1, 2, 3)).translate((30, -5, 17)),
          box((5, 7, 9), center=(22.5, 43.5, 37.5)),
          cylinder(3, 17).rotate_y(90).translate((-9, 21, -4))]
"""

PRESENTATION = {
    "0": {"name": 'Base & bracket <Ø2> "left"', "color": [1, 0, 0.5]},
    "2": {"name": "Peg", "color": [0, 0.25, 1]},
}
EXPECTED_PRESENTATION = {
    0: (PRESENTATION["0"]["name"], "#FF0080"),
    1: ("Part 2", "#B3BFCC"),
    2: ("Peg", "#0040FF"),
}


class IndividualThreeMFExportTests(unittest.TestCase):
    def setUp(self):
        self.result, self.output = build(self, FIXTURE)
        self.stats = self.result["partStats"]

    def export(self, **fields):
        original = (self.output / "model.rsdf").read_bytes()
        with (mock.patch.object(sdf_ops, "read_artifact", wraps=sdf_ops.read_artifact) as read,
              mock.patch.object(sdf_build, "build", side_effect=AssertionError("No replay"))):
            report, folder = native(self.output, self.result, "export", **fields)
        self.assertEqual(read.call_count, 1, "All files share one graph load.")
        self.assertEqual(original, (self.output / "model.rsdf").read_bytes())
        return report, folder

    def plate(self, selected):
        """Printing placements as the app sends them: worker orientations,
        then translations that ground and separate the footprints."""
        report, _ = native(self.output, self.result, "print_orient", solidIndices=selected)
        transforms, x = [], 0.0
        for record in report["orientations"]:
            matrix, box = list(record["matrix"]), record["bounds"]
            matrix[3] += x - box[0]
            matrix[7] -= box[1]
            matrix[11] -= box[2]
            transforms.append({"solid": record["solid"], "matrix": matrix})
            x += box[3] - box[0] + 50
        return transforms

    def assert_same_geometry(self, corners, indices, placed=False):
        volume = sum(self.stats[index]["volume"] for index in indices)
        self.assertAlmostEqual(signed_volume(corners) / volume, 1, delta=2e-3)
        if not placed:
            lo = np.min([self.stats[index]["bounds"][:3] for index in indices], axis=0)
            hi = np.max([self.stats[index]["bounds"][3:] for index in indices], axis=0)
            points = corners.reshape(-1, 3)
            self.assertTrue(np.allclose(points.min(axis=0), lo, atol=0.08), (points.min(axis=0), lo))
            self.assertTrue(np.allclose(points.max(axis=0), hi, atol=0.08), (points.max(axis=0), hi))

    def test_individual_subset_and_grouping_preserve_output_order_and_coordinates(self):
        for selected, parts in [([0, 2], [[2], [0]]), ([0, 1, 2], [[2, 0], [1]])]:
            with self.subTest(parts=parts):
                report, folder = self.export(solidIndices=selected, individualParts=parts,
                                             partPresentation=PRESENTATION)
                self.assertFalse((folder / "model.3mf").exists())
                self.assertEqual(report["solids"], len(selected))
                self.assertEqual(len(report["partFiles"]), len(parts))
                total = 0
                for index, (record, expected) in enumerate(zip(report["partFiles"], parts)):
                    self.assertEqual(record, {"path": f"part-{index:04d}.3mf", "solidIndices": expected})
                    corners, meshes = read_3mf(folder / record["path"])
                    self.assert_presentation(meshes, expected)
                    total += len(corners)
                    self.assert_same_geometry(corners, expected)
                self.assertEqual(total, report["triangles"])
                self.assertEqual(sorted(path.name for path in folder.glob("*.3mf")),
                                 [f"part-{index:04d}.3mf" for index in range(len(parts))])

    def assert_presentation(self, meshes, indices):
        self.assertEqual([(name, color) for name, color, _, _ in meshes],
                         [EXPECTED_PRESENTATION[index] for index in indices])
        for _, _, vertices, triangles in meshes:
            self.assertEqual(len(np.unique(vertices, axis=0)), len(vertices), "3MF mesh topology shares vertex indices.")
            assert_closed_and_outward(self, vertices, triangles)

    def test_individual_printing_files_share_one_selected_placement(self):
        selected = [0, 2]
        plate = self.plate(selected)
        with mock.patch.object(sdf_ops, "print_orientations", side_effect=AssertionError("Placement is supplied")):
            report, folder = self.export(solidIndices=selected, individualParts=[[2], [0]], solidTransforms=plate,
                                         partPresentation=PRESENTATION)
            combined, combined_folder = self.export(solidIndices=selected, solidTransforms=plate,
                                                    partPresentation=PRESENTATION)
        exported = {}
        for record in report["partFiles"]:
            corners, meshes = read_3mf(folder / record["path"])
            self.assert_presentation(meshes, record["solidIndices"])
            exported[record["solidIndices"][0]] = corners
        merged, meshes = read_3mf(combined_folder / "model.3mf")
        self.assert_presentation(meshes, selected)
        self.assertEqual(combined["triangles"], report["triangles"])
        self.assertEqual(combined["cellSize"], report["cellSize"])
        self.assertAlmostEqual(signed_volume(merged), sum(signed_volume(item) for item in exported.values()),
                               places=6)
        boxes = {}
        for index, corners in exported.items():
            self.assert_same_geometry(corners, [index], placed=True)
            points = corners.reshape(-1, 3)
            boxes[index] = np.concatenate([points.min(axis=0), points.max(axis=0)])
            self.assertAlmostEqual(boxes[index][2], 0, delta=0.05)
        a, b = boxes[0], boxes[2]
        clearance = max(b[0] - a[3], a[0] - b[3], b[1] - a[4], a[1] - b[4])
        self.assertGreaterEqual(clearance, 50 - 0.1)

    def test_invalid_individual_partitions_fail_before_writing_any_3mf(self):
        invalid = [None, [], {}, "parts", [None], [[]], [[0], [0]], [[0, 0], [2]], [[0]],
                   [[0], [1]], [[0], [True]], [[0], [-1]], [[0], [3]], [[0], ["2"]]]
        for parts in invalid:
            with self.subTest(parts=parts):
                with (mock.patch.object(sdf_ops, "write_3mf", side_effect=AssertionError("No partial output")),
                      self.assertRaises(sdf_build.ModelError)):
                    native(self.output, self.result, "export", solidIndices=[0, 2], individualParts=parts)
        self.assertEqual(list(self.output.parent.glob("*/*.3mf")), [])

    def test_invalid_presentation_fails_before_writing_any_3mf(self):
        invalid = [{"-1": PRESENTATION["0"]}, {"3": PRESENTATION["0"]},
                   {"0": {"name": "Part", "color": [1, 0]}}, {"0": {"name": "Part", "color": [1, 0, 2]}},
                   {"0": {"name": "Part", "color": [1, 0, float("nan")]}}]
        for presentation in invalid:
            with self.subTest(presentation=presentation):
                with mock.patch.object(sdf_ops, "write_3mf", side_effect=AssertionError("No partial output")):
                    with self.assertRaises(sdf_build.ModelError):
                        native(self.output, self.result, "export", partPresentation=presentation)
        self.assertEqual(list(self.output.parent.glob("*/*.3mf")), [])


if __name__ == "__main__":
    unittest.main()
