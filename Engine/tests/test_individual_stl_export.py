"""Batch STL export preserves parts, their coordinates and shared print placement."""
import unittest
from unittest import mock

import numpy as np

from sdf_test_support import build, native, read_stl, signed_volume
import sdf_build
import sdf_ops

FIXTURE = """
result = [box((8, 14, 2)).rotate(61, (1, 2, 3)).translate((30, -5, 17)),
          box((5, 7, 9), center=(22.5, 43.5, 37.5)),
          cylinder(3, 17).rotate_y(90).translate((-9, 21, -4))]
"""


class IndividualSTLExportTests(unittest.TestCase):
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
                report, folder = self.export(solidIndices=selected, individualParts=parts)
                self.assertFalse((folder / "model.stl").exists())
                self.assertEqual(report["solids"], len(selected))
                self.assertEqual(len(report["partFiles"]), len(parts))
                total = 0
                for index, (record, expected) in enumerate(zip(report["partFiles"], parts)):
                    self.assertEqual(record, {"path": f"part-{index:04d}.stl", "solidIndices": expected})
                    corners, _ = read_stl(folder / record["path"])
                    total += len(corners)
                    self.assert_same_geometry(corners, expected)
                self.assertEqual(total, report["triangles"])
                self.assertEqual(sorted(path.name for path in folder.glob("*.stl")),
                                 [f"part-{index:04d}.stl" for index in range(len(parts))])

    def test_individual_printing_files_share_one_selected_placement(self):
        selected = [0, 2]
        plate = self.plate(selected)
        with mock.patch.object(sdf_ops, "print_orientations", side_effect=AssertionError("Placement is supplied")):
            report, folder = self.export(solidIndices=selected, individualParts=[[2], [0]], solidTransforms=plate)
            combined, combined_folder = self.export(solidIndices=selected, solidTransforms=plate)
        exported = {record["solidIndices"][0]: read_stl(folder / record["path"])[0] for record in report["partFiles"]}
        merged, _ = read_stl(combined_folder / "model.stl")
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

    def test_invalid_individual_partitions_fail_before_writing_any_stl(self):
        invalid = [None, [], {}, "parts", [None], [[]], [[0], [0]], [[0, 0], [2]], [[0]],
                   [[0], [1]], [[0], [True]], [[0], [-1]], [[0], [3]], [[0], ["2"]]]
        for parts in invalid:
            with self.subTest(parts=parts):
                with (mock.patch.object(sdf_ops, "write_stl", side_effect=AssertionError("No partial output")),
                      self.assertRaises(sdf_build.ModelError)):
                    native(self.output, self.result, "export", solidIndices=[0, 2], individualParts=parts)
        self.assertEqual(list(self.output.parent.glob("*/*.stl")), [])

    def test_stl_carries_no_colors(self):
        colors = [{"solidIndex": 0, "color": {"red": 0.8, "green": 0.1, "blue": 0.2}}]
        with self.assertRaisesRegex(sdf_build.ModelError, "solidColors"):
            native(self.output, self.result, "export", solidColors=colors)


if __name__ == "__main__":
    unittest.main()
