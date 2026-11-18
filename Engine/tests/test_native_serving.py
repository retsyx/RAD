"""The long-lived native worker serves only native operations and reuses graphs."""
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile
import unittest
from unittest import mock

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import radsdf
import sdf_ops
import worker
from sdf_build import ModelError

ROOT = Path(__file__).resolve().parents[2]
SCRIPT = "result = [box((2, 3, 4)), sphere(1.5, center=(6, 0, 0))]"


def stage(folder, name, payload):
    directory = folder / name
    directory.mkdir()
    (directory / "request.json").write_text(json.dumps(payload))
    return directory


def built(folder, script=SCRIPT):
    """A build's output folder and its result."""
    output = folder / "build"
    output.mkdir()
    (output / "request.json").write_text(json.dumps({"title": "Parts", "summary": "Test parts", "units": "mm",
                                                     "script": script}))
    return output, worker.run(output / "request.json", output, 2)


def native_spec(output, result, operation, **fields):
    return {"operation": operation, "artifact": "source.rsdf", "artifactIdentity": str(output / "model.rsdf"),
            "partRoots": result["partRoots"], **fields}


class NativeServingTests(unittest.TestCase):
    def test_serving_answers_native_requests_and_refuses_programs(self):
        with tempfile.TemporaryDirectory() as root:
            folder = Path(root)
            output, result = built(folder)
            runtime = stage(folder, "runtime", {"operation": "runtime"})
            probe = stage(folder, "probe", native_spec(output, result, "probe", points=[[0, 0, 0], [6, 0, 0]]))
            shutil.copyfile(output / "model.rsdf", probe / "source.rsdf")
            program = stage(folder, "program", {"title": "Box", "summary": "Generated program", "units": "mm",
                                                "script": "result = box(1)"})
            lines = "".join(json.dumps({"id": index, "request": str(directory / "request.json"),
                                        "output": str(directory), "threads": 1}) + "\n"
                            for index, directory in enumerate([runtime, probe, program], start=1))
            completed = subprocess.run([sys.executable, str(ROOT / "Engine" / "worker.py"), "--serve"], input=lines,
                                       capture_output=True, text=True, timeout=120)
            self.assertEqual(completed.returncode, 0, completed.stderr)
            replies = [json.loads(line) for line in completed.stdout.splitlines()]
            self.assertEqual([reply["id"] for reply in replies], [1, 2, 3])
            self.assertEqual([reply["status"] for reply in replies[:2]], ["complete", "complete"])
            self.assertEqual(set(json.loads((runtime / "result.json").read_text())["runtime"]),
                             {"python", "numpy", "kernel"})
            samples = json.loads((probe / "result.json").read_text())["samples"]
            self.assertAlmostEqual(samples[0][0]["distance"], -1, places=9)
            self.assertAlmostEqual(samples[1][1]["distance"], -1.5, places=9)
            self.assertEqual(replies[2]["status"], "error")
            self.assertIn("only native geometry operations", replies[2]["message"])
            self.assertTrue((program / "error.json").exists())
            self.assertFalse((program / "result.json").exists(), "A refused request publishes no result.")

    def test_serving_reuses_a_graph_by_identity(self):
        with tempfile.TemporaryDirectory() as root:
            folder = Path(root)
            output, result = built(folder)
            staged = folder / "staged"
            staged.mkdir()
            source = staged / "source.rsdf"
            shutil.copyfile(output / "model.rsdf", source)
            spec = native_spec(output, result, "probe")
            with mock.patch.dict(sdf_ops._sources, clear=True):
                first = sdf_ops.load_source(spec, staged, serving=True)
                source.unlink()
                self.assertIs(sdf_ops.load_source(spec, staged, serving=True), first,
                              "A repeated identity reuses the loaded graph.")
                shutil.copyfile(output / "model.rsdf", source)
                for index in range(2):
                    sdf_ops.load_source(dict(spec, artifactIdentity=f"/session/other-{index}/model.rsdf"), staged,
                                        serving=True)
                self.assertEqual(len(sdf_ops._sources), 2, "At most two graphs stay loaded.")
                self.assertTrue(first.graph.closed, "An evicted graph is freed.")
                # Other roots for a cached identity reload the artifact.
                single = sdf_ops.load_source(dict(spec, artifactIdentity="/session/other-1/model.rsdf",
                                                  partRoots=result["partRoots"][:1]), staged, serving=True)
                self.assertEqual(single.solids, 1)
            with mock.patch.dict(sdf_ops._sources, clear=True):
                sdf_ops.execute(spec | {"points": [[0, 0, 0]]}, staged, staged, 1)
                self.assertEqual(len(sdf_ops._sources), 0, "One-shot processes never cache graphs.")

    def test_artifacts_are_checked_before_the_kernel_reads_them(self):
        with tempfile.TemporaryDirectory() as root:
            folder = Path(root)
            output, result = built(folder)
            data = (output / "model.rsdf").read_bytes()
            spec = native_spec(output, result, "probe")
            cases = {
                "header": b"RSDX" + data[4:],
                "version": data[:4] + (2).to_bytes(4, "little") + data[8:],
                "truncated": data[:-8],
                "short": data[:12],
            }
            for name, payload in cases.items():
                with self.subTest(artifact=name), tempfile.TemporaryDirectory() as directory:
                    (Path(directory) / "source.rsdf").write_bytes(payload)
                    with self.assertRaises(ModelError):
                        sdf_ops.load_source(spec, Path(directory))
            invalid_specs = [dict(spec, artifact="model.rsdf"), dict(spec, artifactIdentity=""),
                             dict(spec, partRoots=[]), dict(spec, partRoots=[True]),
                             dict(spec, partRoots=[len(data)]), dict(spec, partRoots="0")]
            for invalid in invalid_specs:
                with self.subTest(spec=invalid), tempfile.TemporaryDirectory() as directory:
                    (Path(directory) / "source.rsdf").write_bytes(data)
                    with self.assertRaises(ModelError):
                        sdf_ops.load_source(invalid, Path(directory))
            with tempfile.TemporaryDirectory() as directory:
                directory = Path(directory)
                with self.assertRaisesRegex(ModelError, "missing"):
                    sdf_ops.load_source(spec, directory)
                (directory / "real.rsdf").write_bytes(data)
                (directory / "source.rsdf").symlink_to(directory / "real.rsdf")
                with self.assertRaises(ModelError):
                    sdf_ops.load_source(spec, directory)
                (directory / "source.rsdf").unlink()
                os.link(directory / "real.rsdf", directory / "source.rsdf")
                with self.assertRaisesRegex(ModelError, "private file"):
                    sdf_ops.load_source(spec, directory)
            # Part roots must be solids, not 2D profiles.
            profile_data, _ = radsdf.serialize([radsdf.circle(1)])
            with tempfile.TemporaryDirectory() as directory:
                (Path(directory) / "source.rsdf").write_bytes(profile_data)
                with self.assertRaisesRegex(ModelError, "2D profile"):
                    sdf_ops.load_source(dict(spec, partRoots=[0]), Path(directory))

    def test_requests_name_only_supported_fields(self):
        with tempfile.TemporaryDirectory() as root:
            folder = Path(root)
            output, result = built(folder)
            shutil.copyfile(output / "model.rsdf", folder / "source.rsdf")
            for spec in (native_spec(output, result, "probe", points=[[0, 0, 0]], features=[]),
                         native_spec(output, result, "dxf"), native_spec(output, result, "verify"),
                         native_spec(output, result, "export", solidColors=[])):
                with self.subTest(operation=spec["operation"]), self.assertRaises(ModelError):
                    sdf_ops.execute(spec, folder, folder, 1)


if __name__ == "__main__":
    unittest.main()
