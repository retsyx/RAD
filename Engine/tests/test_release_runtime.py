"""Release runtime boundaries: exact bytes, safe staging and native inventory."""
import hashlib
import importlib.util
import json
from pathlib import Path
import subprocess
import sys
import tarfile
import tempfile
import unittest
from unittest.mock import patch

SCRIPTS = Path(__file__).resolve().parents[2] / "Scripts"
sys.path.insert(0, str(SCRIPTS))
SPEC = importlib.util.spec_from_file_location("rad_release_runtime", SCRIPTS / "release_runtime.py")
release = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(release)


class ReleaseRuntimeTests(unittest.TestCase):
    def fixture(self, root):
        expanded = root / "expanded"
        (expanded / "python/bin").mkdir(parents=True)
        (expanded / "python/bin/python3").write_bytes(b"native executable")
        (expanded / "requirements-lock.txt").write_text("numpy==2.4.6\n")
        (expanded / "wheel-hashes.json").write_text("{}\n")
        return expanded

    def manifest(self):
        digest = "a" * 64
        return {"formatVersion": 1, "identifier": "cpython-3.11.18-numpy-2.4.6-arm64-" + digest[:20],
                "architecture": "arm64", "pythonVersion": "3.11.18", "archive": "payload.tar.gz", "sha256": digest}

    def test_inventory_includes_unusual_native_names_once_and_never_follows_symlinks(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / "z.bin").write_bytes(b"\xcf\xfa\xed\xfe" + b"image")
            (root / "a.cpython.so").write_bytes(b"\xca\xfe\xba\xbe" + b"image")
            (root / "script").write_text("#!/bin/sh\n")
            (root / "alias").symlink_to("z.bin")
            self.assertEqual([path.name for path in release.native_images(root)], ["a.cpython.so", "z.bin"])

    def test_packed_bytes_receive_new_install_identity_and_complete_manifest(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            expanded = self.fixture(root)
            source = self.manifest()
            signing = {"identity": "Test Developer ID", "hardenedRuntime": True,
                       "nativeImages": ["python/bin/python3"], "pythonEntitlements": {}}
            manifest = release.pack_runtime(expanded, source, root / "output", signing)
            archive = root / "output/payload.tar.gz"
            actual = hashlib.sha256(archive.read_bytes()).hexdigest()
            self.assertEqual(manifest["sha256"], actual)
            self.assertNotEqual(manifest["identifier"], source["identifier"])
            self.assertTrue(manifest["identifier"].endswith(actual[:20]))
            self.assertEqual(json.loads((root / "output/manifest.json").read_text()), manifest)
            self.assertEqual(manifest["signing"]["nativeImageCount"], 1)
            self.assertNotIn("nativeImages", manifest["signing"])
            self.assertEqual(source, self.manifest())
            with tarfile.open(archive) as bundle:
                self.assertEqual(bundle.extractfile("python/bin/python3").read(), b"native executable")
                self.assertEqual(bundle.extractfile("requirements-lock.txt").read(), b"numpy==2.4.6\n")

    def test_refuses_external_symlinks_and_incomplete_payload_before_publishing(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            expanded = self.fixture(root)
            (expanded / "python/external").symlink_to("/usr/bin/true")
            with self.assertRaisesRegex(ValueError, "external symlink"):
                release.pack_runtime(expanded, self.manifest(), root / "output")
            self.assertFalse((root / "output/manifest.json").exists())
            (expanded / "python/external").unlink()
            (expanded / "wheel-hashes.json").unlink()
            with self.assertRaisesRegex(ValueError, "incomplete"):
                release.pack_runtime(expanded, self.manifest(), root / "output")

    def test_refuses_adhoc_release_and_mutating_the_prepared_runtime_cache(self):
        with self.assertRaisesRegex(ValueError, "Developer ID"):
            release.sign_runtime(Path("/tmp/unused"), "-", production=True)
        cache = SCRIPTS.parent / ".build/cad-runtime/arm64/expanded"
        with self.assertRaisesRegex(ValueError, "staging copy"):
            release.sign_runtime(cache, "-", production=False)

    @staticmethod
    def codesign(calls, team="TEAM123456", hardened=True, timestamp=True,
                 authority="Developer ID Application: RAD Test (TEAM123456)"):
        """Stand-in for codesign: records commands and reports a signature."""
        def run(command, **kwargs):
            calls.append(command)
            details = ""
            if "--display" in command:
                details = (f"Executable={command[-1]}\nflags=0x{0x10000 if hardened else 0x2:x}(runtime)\n"
                           f"Authority={authority}\nTeamIdentifier={team}\n"
                           + ("Timestamp=Nov 16, 2026 at 00:00:00\n" if timestamp else ""))
            return subprocess.CompletedProcess(command, 0, "", details)
        return run

    def test_release_python_has_no_entitlements_and_local_python_only_skips_library_validation(self):
        with tempfile.TemporaryDirectory() as directory:
            expanded = self.fixture(Path(directory))
            (expanded / "python/bin/python3").write_bytes(b"\xcf\xfa\xed\xfe" + b"interpreter")
            (expanded / "python/lib").mkdir()
            (expanded / "python/lib/numpy.so").write_bytes(b"\xcf\xfa\xed\xfe" + b"extension")
            for production, identity in ((True, "Developer ID Application: RAD Test"), (False, "-")):
                calls = []
                with self.subTest(production=production), patch.object(release, "audit_runtime", return_value={}), \
                        patch.object(release, "smoke_runtime") as smoke, \
                        patch.object(release, "checked", side_effect=self.codesign(calls)):
                    signing = release.sign_runtime(expanded, identity, production=production)
                    smoke.assert_called_once()
                    signs = {Path(call[-1]).name: call for call in calls if "--sign" in call}
                    self.assertEqual(set(signs), {"python3", "numpy.so"})
                    self.assertEqual(sorted(signing["nativeImages"]), ["python/bin/python3", "python/lib/numpy.so"])
                    self.assertTrue(all("--options" in call and "runtime" in call for call in signs.values()))
                    self.assertNotIn("--entitlements", signs["numpy.so"])
                    if production:
                        self.assertEqual(signing["pythonEntitlements"], {})
                        self.assertNotIn("--entitlements", signs["python3"])
                        self.assertEqual(signing["teamIdentifier"], "TEAM123456")
                    else:
                        self.assertEqual(signing["pythonEntitlements"],
                                         {"com.apple.security.cs.disable-library-validation": True})
                        self.assertIn("--entitlements", signs["python3"])

    def test_signed_images_require_hardened_timestamped_developer_id_signatures(self):
        library = Path("Contents/Frameworks/libRADSDF.dylib")
        for case in ({"hardened": False}, {"timestamp": False}, {"team": "not set"},
                     {"authority": "Apple Development: RAD Test (TEAM123456)"}):
            with self.subTest(**case), patch.object(release, "checked", side_effect=self.codesign([], **case)):
                with self.assertRaises(RuntimeError):
                    release.sign_image(library, "Developer ID Application: RAD Test", production=True)
        calls = []
        with patch.object(release, "checked", side_effect=self.codesign(calls)):
            self.assertEqual(release.sign_image(library, "Developer ID Application: RAD Test", production=True), "TEAM123456")
        self.assertIn("--timestamp", calls[0])
        with patch.object(release, "checked", side_effect=self.codesign([], timestamp=False, team="not set", authority="")):
            self.assertEqual(release.sign_image(library, "-", production=False), "not set")

    def test_rejects_output_inside_runtime_and_unsupported_source_manifest(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            expanded = self.fixture(root)
            with self.assertRaisesRegex(ValueError, "outside"):
                release.pack_runtime(expanded, self.manifest(), expanded / "output")
            with self.assertRaisesRegex(ValueError, "Apple Silicon"):
                release.pack_runtime(expanded, {**self.manifest(), "architecture": "unsupported"}, root / "output")


if __name__ == "__main__":
    unittest.main()
