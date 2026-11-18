"""Imported input assets: records, private files and receipts. Programs read
STL and OBJ inputs as mesh solids through `imported_assets`
(test_mesh_import.py covers the meshes themselves)."""
import json
import os
from pathlib import Path
import tempfile
import unittest

from sdf_test_support import script_spec
from sdf_build import ModelError, validate_spec, validation_hash
import worker


def spec(asset, script=None):
    return {**script_spec(script or "result = box(1)", title="External geometry"), "inputAssets": [asset]}


def cube_obj(low=0, high=1):
    points = [(low, low, low), (high, low, low), (high, high, low), (low, high, low),
              (low, low, high), (high, low, high), (high, high, high), (low, high, high)]
    faces = [(1, 4, 3, 2), (5, 6, 7, 8), (1, 2, 6, 5), (2, 3, 7, 6), (3, 4, 8, 7), (4, 1, 5, 8)]
    return "".join("v %s %s %s\n" % point for point in points) + "".join(
        "f " + " ".join(str(index) for index in face) + "\n" for face in faces)


class AgentAssetTests(unittest.TestCase):
    def run_build(self, folder, value):
        request = folder / "request.json"
        request.write_text(json.dumps(value))
        return worker.run(request, folder / "output", 2)

    def test_staged_mesh_inputs_are_receipted_and_importable(self):
        valid = {"id": "external", "name": "Cube", "format": "obj", "path": "input-000.obj"}
        with tempfile.TemporaryDirectory() as directory:
            folder = Path(directory)
            (folder / "input-000.obj").write_text(cube_obj())
            result = self.run_build(folder, spec(valid))
            self.assertEqual(result["validatedHash"], validation_hash(spec(valid), folder))
            imported = self.run_build(folder, spec(valid, "result = imported_assets['external']"))
            self.assertAlmostEqual(imported["volume"], 1.0, delta=0.01)
            self.assertEqual(imported["validatedHash"],
                             validation_hash(spec(valid, "result = imported_assets['external']"), folder))
            with self.assertRaisesRegex(ModelError, "imported asset 'other' is not available"):
                self.run_build(folder, spec(valid, "result = imported_assets['other']"))
            listed = self.run_build(folder, spec(valid, "assert list(imported_assets) == ['external']\n"
                                                        "result = box(1)"))
            self.assertEqual(listed["solids"], 1)

    def test_invalid_asset_records_paths_and_links_fail_before_replay(self):
        valid = {"id": "external", "name": "Cube", "format": "obj", "path": "input-000.obj"}
        for changed in [{"path": "../cube.obj"}, {"format": "exe"}, {"format": "step"}, {"format": "brep"},
                        {"format": []}, {"id": ""}, {"id": "a\nb"}, {"name": 1}, {"validatedBREP": "x"}]:
            with self.subTest(changed=changed), self.assertRaises(ModelError):
                validate_spec(spec({**valid, **changed}))
        value = spec(valid)
        value["inputAssets"] *= 2
        with self.assertRaises(ModelError):
            validate_spec(value)
        with tempfile.TemporaryDirectory() as directory:
            folder = Path(directory)
            original = folder / "original.obj"
            original.write_text(cube_obj())
            linked = folder / "input-000.obj"
            linked.symlink_to(original)
            with self.assertRaisesRegex(ModelError, "regular private"):
                self.run_build(folder, spec(valid))
            linked.unlink()
            os.link(original, linked)
            with self.assertRaisesRegex(ModelError, "regular private"):
                self.run_build(folder, spec(valid))
            linked.unlink()
            with self.assertRaisesRegex(ModelError, "missing"):
                self.run_build(folder, spec(valid))
            self.assertFalse((folder / "output" / "result.json").exists())

    def test_runtime_query_creates_no_geometry(self):
        with tempfile.TemporaryDirectory() as directory:
            folder = Path(directory)
            request = folder / "request.json"
            request.write_text('{"operation":"runtime"}')
            result = worker.run(request, folder, 2)
            self.assertEqual(set(result), {"runtime"})
            self.assertEqual(set(result["runtime"]), {"python", "numpy", "kernel"})
            self.assertEqual(json.loads((folder / "result.json").read_text()), result)
            self.assertFalse((folder / "model.rsdf").exists())
            self.assertFalse((folder / "mesh.bin").exists())
            with self.assertRaisesRegex(ModelError, "takes no other fields"):
                request.write_text('{"operation":"runtime","solids":[0]}')
                worker.run(request, folder, 2)


if __name__ == "__main__":
    unittest.main()
