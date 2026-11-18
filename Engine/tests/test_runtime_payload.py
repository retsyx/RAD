"""Packaging checks that prevent shipping a runtime tied to a developer Mac."""
import importlib.util
import os
from pathlib import Path
import subprocess
import sys
import unittest

SCRIPTS = Path(__file__).resolve().parents[2] / "Scripts"
sys.path.insert(0, str(SCRIPTS))
SPEC = importlib.util.spec_from_file_location("rad_runtime_audit", SCRIPTS / "audit_runtime.py")
audit = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(audit)
SPEC = importlib.util.spec_from_file_location("rad_runtime_preparation", SCRIPTS / "prepare_runtime.py")
preparation = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(preparation)


class RuntimeNativeAuditTests(unittest.TestCase):
    def test_rejects_external_runtime_libraries_and_search_paths(self):
        for kind in ["dependency", "rpath"]:
            for path in ["/opt/homebrew/opt/lib/libsomething.dylib", "/Users/builder/local/lib", "liblocal.dylib"]:
                with self.subTest(kind=kind, path=path), self.assertRaises(RuntimeError):
                    audit.validate_commands([(kind, path)], "test-image")

    def test_checks_minimum_macos_while_allowing_relocatable_and_system_paths(self):
        audit.validate_commands([("dependency", "@rpath/libRADSDF.dylib"),
                                 ("rpath", "@loader_path/../native libraries"),
                                 ("dependency", "/usr/lib/libSystem.B.dylib"),
                                 ("minimumOS", "14.0")], "test-image")
        with self.assertRaises(RuntimeError):
            audit.validate_commands([("minimumOS", "15.0")], "test-image")

    def test_parses_dependencies_rpaths_and_both_deployment_commands(self):
        commands = audit.load_commands("""
Load command 0
          cmd LC_LOAD_DYLIB
         name @rpath/native library.dylib (offset 24)
Load command 1
          cmd LC_RPATH
         path @loader_path/../native libraries (offset 12)
Load command 2
          cmd LC_BUILD_VERSION
        minos 14.0
Load command 3
          cmd LC_VERSION_MIN_MACOSX
      version 11.0
""")
        self.assertEqual(commands, [("dependency", "@rpath/native library.dylib"),
                                    ("rpath", "@loader_path/../native libraries"),
                                    ("minimumOS", "14.0"), ("minimumOS", "11.0")])


class RuntimePreparationTests(unittest.TestCase):
    def test_identifier_names_the_pinned_packages_and_payload_digest(self):
        digest = "0123456789abcdef" * 4
        self.assertEqual(preparation.runtime_identifier("3.14.10", "# Runtime lock\nnumpy==2.4.6\n", "arm64", digest),
                         "cpython-3.14.10-numpy-2.4.6-arm64-" + digest[:20])
        self.assertEqual(preparation.pinned_packages("NumPy==2.4.6  # pinned\n\nPyYAML==6.0.3\n"),
                         [("numpy", "2.4.6"), ("pyyaml", "6.0.3")])

    def test_lock_requires_exact_pins_and_identifier_stays_within_installer_limit(self):
        for lock in ["", "numpy>=2.4\n", "numpy\n", "numpy==2.4.6 --hash=sha256:00\n", "numpy==2.4.6+local\n"]:
            with self.subTest(lock=lock), self.assertRaises(RuntimeError):
                preparation.pinned_packages(lock)
        lock = "".join(f"package{index}==1.0.0\n" for index in range(20))
        with self.assertRaisesRegex(RuntimeError, "limit"):
            preparation.runtime_identifier("3.14.10", lock, "arm64", "0" * 64)

    def check_runtime(self, environment):
        environment = {"PATH": "/usr/bin:/bin", "PYTHONDONTWRITEBYTECODE": "1", **environment}
        process = subprocess.run([sys.executable, "-I", "-B", str(SCRIPTS / "check_runtime.py")],
                                 env=environment, capture_output=True, text=True, timeout=180)
        self.assertEqual(process.returncode, 0, process.stderr)
        return process.stdout

    def test_runtime_check_needs_no_kernel(self):
        self.assertNotIn("SDF kernel", self.check_runtime({}))

    @unittest.skipUnless(os.environ.get("RAD_SDF_LIBRARY"), "Requires RAD_SDF_LIBRARY naming the built libRADSDF.dylib")
    def test_runtime_check_loads_evaluates_and_meshes_with_the_kernel(self):
        self.assertIn("SDF kernel", self.check_runtime({"RAD_SDF_LIBRARY": os.environ["RAD_SDF_LIBRARY"]}))


if __name__ == "__main__":
    unittest.main()
