"""Opt-in macOS sandbox integration: run outside another sandbox.

RAD_RUN_SANDBOX_TESTS=1 RAD_SDF_LIBRARY=<built libRADSDF.dylib> \
    .build/cad-runtime/arm64/expanded/python/bin/python3 -m unittest discover -s Engine/tests -v
Scripts/test.sh builds the kernel and sets RAD_SDF_LIBRARY.
"""
import json
import math
import os
from pathlib import Path
import platform
import subprocess
import sys
import tempfile
import unittest

ROOT = Path(__file__).resolve().parents[2]
KERNEL = os.environ.get("RAD_SDF_LIBRARY", "")
# A 20 x 15 x 6 mm plate with a 6 mm diameter through hole.
PLATE = "result = difference(box((20, 15, 6)), cylinder(3, 10))"
PLATE_VOLUME = 20 * 15 * 6 - math.pi * 3 ** 2 * 6


@unittest.skipUnless(platform.system() == "Darwin" and os.environ.get("RAD_RUN_SANDBOX_TESTS") == "1",
                     "Requires macOS and RAD_RUN_SANDBOX_TESTS=1 outside an enclosing sandbox")
@unittest.skipUnless(KERNEL and Path(KERNEL).is_file(), "Requires RAD_SDF_LIBRARY naming the built libRADSDF.dylib")
class SandboxTests(unittest.TestCase):
    def sandboxed(self, arguments, output):
        """Run the runtime interpreter under Engine/script.sb as RAD's worker launch does."""
        runtime = Path(sys.executable).absolute()
        resolved = runtime.resolve()
        framework = resolved.parent.parent / "Resources/Python.app/Contents/MacOS/Python"
        interpreter = framework if framework.is_file() else resolved
        # The profile matches real paths: pass the kernel's resolved location.
        kernel = Path(KERNEL).resolve()
        parameters = {"RUNTIME": str(runtime.parent.parent.resolve()), "PYTHON_BASE": str(resolved.parent.parent),
                      "WORKER": str(ROOT / "Engine"), "OUTPUT": str(output), "PYTHON": str(interpreter),
                      "ENTRY_PYTHON": str(runtime), "KERNEL": str(kernel)}
        command = ["/usr/bin/sandbox-exec"]
        for key, value in parameters.items():
            command += ["-D", f"{key}={value}"]
        command += ["-f", str(ROOT / "Engine/script.sb"), str(interpreter), *map(str, arguments)]
        environment = {"PATH": "/usr/bin:/bin", "HOME": str(Path.home()), "LANG": "en_US.UTF-8", "LC_ALL": "C",
                       "PYTHONNOUSERSITE": "1", "PYTHONDONTWRITEBYTECODE": "1", "RAD_MODEL_OUTPUT": str(output),
                       "RAD_SDF_LIBRARY": str(kernel), "XDG_CACHE_HOME": str(output / ".cache"), "TMPDIR": str(output)}
        if interpreter != resolved:
            environment["__PYVENV_LAUNCHER__"] = str(runtime)
        return subprocess.run(command, cwd=output, env=environment, capture_output=True, text=True, timeout=180)

    def build(self, source, private_contents="private test data"):
        with tempfile.TemporaryDirectory(prefix="RAD-sandbox-test-") as directory:
            folder = Path(directory).resolve()
            output = folder / "output"
            output.mkdir()
            private = folder / "private.txt"
            private.write_text(private_contents)
            source = source.replace("PRIVATE_FIXTURE", repr(str(private))).replace("OUTSIDE_DESTINATION", repr(str(folder / "forbidden.txt")))
            request = output / "request.json"
            request.write_text(json.dumps({"title": "Sandbox test", "summary": "Native sandbox integration", "units": "mm",
                                           "script": source}))
            process = self.sandboxed([ROOT / "Engine/worker.py", "--request", request, "--output", output,
                                      "--threads", "2"], output)
            self.assertEqual(process.returncode, 0, process.stderr)
            self.assertEqual(private.read_text(), private_contents)
            self.assertFalse((folder / "forbidden.txt").exists())
            with (output / "model.rsdf").open("rb") as artifact:
                self.assertEqual(artifact.read(4), b"RSDF")
            return json.loads((output / "result.json").read_text())

    def test_kernel_loads_and_meshes_inside_the_sandbox(self):
        with tempfile.TemporaryDirectory(prefix="RAD-sandbox-test-") as directory:
            output = Path(directory).resolve() / "output"
            output.mkdir()
            probe = output / "probe.py"
            probe.write_text(f"""
import json, sys
sys.path.insert(0, {str(ROOT / 'Engine')!r})
import numpy as np
from radsdf import box, cylinder, difference
mesh = difference(box((20, 15, 6)), cylinder(3, 10)).mesh(threads=2)
points, triangles = np.asarray(mesh.positions, dtype=float), np.asarray(mesh.triangles)
a, b, c = points[triangles[:, 0]], points[triangles[:, 1]], points[triangles[:, 2]]
print(json.dumps({{"volume": float(np.einsum("ij,ij->i", a, np.cross(b, c)).sum() / 6)}}))
""")
            process = self.sandboxed(["-I", "-B", probe], output)
            self.assertEqual(process.returncode, 0, process.stderr)
            self.assertAlmostEqual(json.loads(process.stdout)["volume"], PLATE_VOLUME, delta=0.01 * PLATE_VOLUME)

    def test_worker_builds_a_program_with_the_native_kernel(self):
        preview = self.build(PLATE)
        self.assertEqual(preview["solids"], 1)
        self.assertAlmostEqual(preview["volume"], PLATE_VOLUME, delta=0.01 * PLATE_VOLUME)

    def test_private_reads_writes_network_and_subprocesses_are_denied(self):
        source = """
from pathlib import Path
import socket
import subprocess
import errno
import os
assert 'OPENAI_API_KEY' not in os.environ and 'CODEX_API_KEY' not in os.environ
for action in [lambda: Path(PRIVATE_FIXTURE).read_text(),
               lambda: Path(OUTSIDE_DESTINATION).write_text('forbidden')]:
    try:
        action()
    except PermissionError:
        pass
    else:
        raise AssertionError('Sandbox allowed access outside the model folder')
try:
    with socket.socket() as connection:
        connection.connect(('127.0.0.1',9))
except OSError as error:
    assert error.errno in (errno.EPERM, errno.EACCES), error
else:
    raise AssertionError('Sandbox allowed networking')
try:
    subprocess.run(['/bin/echo','forbidden'], check=True)
except PermissionError:
    pass
else:
    raise AssertionError('Sandbox allowed subprocess execution')
result = box((4, 5, 6))
"""
        preview = self.build(source)
        self.assertAlmostEqual(preview["volume"], 120, delta=1.2)


if __name__ == "__main__":
    unittest.main()
