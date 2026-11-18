"""Verify privacy checks against real arm64 debug maps and executable strings."""
import json
from pathlib import Path
import platform
import shutil
import subprocess
import sys
import tempfile
import unittest

SCRIPTS = Path(__file__).resolve().parents[2] / "Scripts"
sys.path.insert(0, str(SCRIPTS))
from prepare_app_binaries import (
    KERNEL_LIBRARY,
    NATIVE_IMAGES,
    BinaryPrivacyError,
    prepare_app_binaries,
    read_binary_metadata,
    verify_app_binaries,
    verify_app_symbols,
)


@unittest.skipUnless(platform.system() == "Darwin" and platform.machine() == "arm64",
                     "Native binary privacy checks require Apple Silicon macOS")
class AppBinaryPrivacyTests(unittest.TestCase):
    executable_paths = (
        Path("Contents/MacOS/RAD"),
        Path("Contents/PlugIns/RADThumbnailExtension.appex/Contents/MacOS/RADThumbnailExtension"),
        Path("Contents/Frameworks/libRADSDF.dylib"),
    )

    def compile_fixture(self, root, embedded_path="public fixture", library=False):
        """An arm64 executable, or a dynamic library like the SDF kernel."""
        name = "fixture-library" if library else "fixture"
        source = root / f"{name}.c"
        entry = ("int fixture_check(void) { return fixture_path[0] == 0; }\n" if library
                 else "int main(void) { return puts(fixture_path) < 0; }\n")
        source.write_text("#include <stdio.h>\n"
                          f"const char *fixture_path = {json.dumps(embedded_path)};\n" + entry)
        object_file = root / f"{name}.o"
        subprocess.run(["/usr/bin/xcrun", "clang", "-arch", "arm64", "-g", "-c",
                        str(source), "-o", str(object_file)], check=True,
                       capture_output=True, text=True)
        image = root / (f"lib{name}.dylib" if library else name)
        options = ["-dynamiclib", "-install_name", "@rpath/libRADSDF.dylib"] if library else []
        subprocess.run(["/usr/bin/xcrun", "clang", "-arch", "arm64", "-g", *options,
                        str(object_file), "-o", str(image)], check=True,
                       capture_output=True, text=True)
        return image

    def make_app(self, root, embedded_path="public fixture"):
        executable = self.compile_fixture(root, embedded_path)
        library = self.compile_fixture(root, embedded_path, library=True)
        app = root / "RAD.app"
        for relative in self.executable_paths:
            image = app / relative
            image.parent.mkdir(parents=True)
            shutil.copy2(library if relative.suffix == ".dylib" else executable, image)
        return app

    def run_image(self, image):
        if image.suffix == ".dylib":
            # The worker loads the kernel the same way, with ctypes.
            subprocess.run([sys.executable, "-I", "-c", "import ctypes, sys; "
                            "sys.exit(ctypes.CDLL(sys.argv[1]).fixture_check())", str(image)],
                           check=True, capture_output=True)
        else:
            subprocess.run([str(image)], check=True, capture_output=True)

    def test_shipped_native_images_include_the_kernel_library(self):
        self.assertEqual(NATIVE_IMAGES, self.executable_paths)
        self.assertEqual(KERNEL_LIBRARY, self.executable_paths[2])

    def test_unstripped_debug_maps_are_rejected(self):
        with tempfile.TemporaryDirectory(prefix="rad-binary-privacy-", dir="/private/tmp") as directory:
            root = Path(directory)
            app = self.make_app(root)
            metadata = read_binary_metadata(app / self.executable_paths[0])
            self.assertGreater(metadata["debugRecordCount"], 0)
            with self.assertRaises(BinaryPrivacyError):
                verify_app_binaries(app)

    def test_prepare_retains_matching_symbols_and_strips_both_executables(self):
        with tempfile.TemporaryDirectory(prefix="rad-binary-privacy-", dir="/private/tmp") as directory:
            root = Path(directory)
            app = self.make_app(root)
            before = {relative.name: read_binary_metadata(app / relative)["uuid"]
                      for relative in self.executable_paths}
            symbols = root / "private symbols"
            prepare_app_binaries(app, symbols, build_roots=(root,))
            result = verify_app_binaries(app, build_roots=(root,))
            self.assertEqual(len(result), 3)
            for relative in self.executable_paths:
                with self.subTest(executable=relative.name):
                    executable = app / relative
                    metadata = read_binary_metadata(executable)
                    self.assertEqual(metadata["architecture"], "arm64")
                    self.assertEqual(metadata["debugRecordCount"], 0)
                    self.assertFalse(metadata["hasDWARF"])
                    self.assertEqual(metadata["uuid"], before[relative.name])
                    dwarf = symbols / f"{relative.name}.dSYM/Contents/Resources/DWARF/{relative.name}"
                    self.assertTrue(dwarf.is_file())
                    debug = read_binary_metadata(dwarf)
                    self.assertEqual(debug["uuid"], metadata["uuid"])
                    self.assertTrue(debug["hasDebugInfo"])
                    self.run_image(executable)

    def test_retained_symbols_must_match_the_shipped_executable(self):
        with tempfile.TemporaryDirectory(prefix="rad-binary-privacy-", dir="/private/tmp") as directory:
            root = Path(directory)
            app = self.make_app(root)
            symbols = root / "symbols"
            prepare_app_binaries(app, symbols, build_roots=(root,))
            verify_app_symbols(app, symbols)
            other = root / "different build"
            other.mkdir()
            executable = self.compile_fixture(other, "different public fixture")
            dsym = other / "fixture.dSYM"
            subprocess.run(["/usr/bin/xcrun", "dsymutil", "--arch", "arm64",
                            str(executable), "-o", str(dsym)], check=True, capture_output=True)
            replacement = dsym / "Contents/Resources/DWARF/fixture"
            self.assertNotEqual(read_binary_metadata(replacement)["uuid"],
                                read_binary_metadata(app / self.executable_paths[0])["uuid"])
            shutil.copy2(replacement, symbols / "RAD.dSYM/Contents/Resources/DWARF/RAD")
            with self.assertRaisesRegex(BinaryPrivacyError, "mismatched"):
                verify_app_symbols(app, symbols)

    def test_symbols_cannot_be_retained_inside_the_distributed_app(self):
        with tempfile.TemporaryDirectory(prefix="rad-binary-privacy-", dir="/private/tmp") as directory:
            root = Path(directory)
            app = self.make_app(root)
            symbols = app / "Contents/Resources/symbols"
            with self.assertRaisesRegex(BinaryPrivacyError, "outside the app"):
                prepare_app_binaries(app, symbols)
            with self.assertRaisesRegex(BinaryPrivacyError, "outside the app"):
                verify_app_symbols(app, symbols)
            self.assertFalse(symbols.exists())

    def test_missing_debug_object_fails_before_executable_is_stripped(self):
        with tempfile.TemporaryDirectory(prefix="rad-binary-privacy-", dir="/private/tmp") as directory:
            root = Path(directory)
            app = self.make_app(root)
            executable = app / self.executable_paths[0]
            before = read_binary_metadata(executable)["debugRecordCount"]
            self.assertGreater(before, 0)
            (root / "fixture.o").unlink()
            with self.assertRaises(BinaryPrivacyError):
                prepare_app_binaries(app, root / "symbols", build_roots=(root,))
            self.assertEqual(read_binary_metadata(executable)["debugRecordCount"], before)

    def test_stripping_does_not_hide_embedded_personal_or_checkout_paths(self):
        with tempfile.TemporaryDirectory(prefix="rad-binary-privacy-", dir="/private/tmp") as directory:
            root = Path(directory)
            for name, embedded_path in (
                ("personal", "/Users/private-fixture/internal-checkout/Source.swift"),
                ("checkout", str(root / "checkout/Source.swift")),
            ):
                with self.subTest(kind=name):
                    case = root / name
                    case.mkdir()
                    app = self.make_app(case, embedded_path)
                    for relative in self.executable_paths:
                        subprocess.run(["/usr/bin/xcrun", "strip", "-S", str(app / relative)],
                                       check=True, capture_output=True)
                    self.assertEqual(read_binary_metadata(app / self.executable_paths[0])["debugRecordCount"], 0)
                    with self.assertRaises(BinaryPrivacyError) as error:
                        verify_app_binaries(app, build_roots=(root / "checkout",))
                    self.assertNotIn(embedded_path, str(error.exception))

    def test_missing_or_malformed_executable_fails_closed(self):
        with tempfile.TemporaryDirectory(prefix="rad-binary-privacy-", dir="/private/tmp") as directory:
            root = Path(directory)
            app = self.make_app(root)
            executable = app / self.executable_paths[1]
            executable.unlink()
            with self.assertRaises(BinaryPrivacyError):
                verify_app_binaries(app)
            executable.write_bytes(b"not a Mach-O executable")
            with self.assertRaises(BinaryPrivacyError):
                read_binary_metadata(executable)

    def test_missing_kernel_library_fails_closed(self):
        with tempfile.TemporaryDirectory(prefix="rad-binary-privacy-", dir="/private/tmp") as directory:
            root = Path(directory)
            app = self.make_app(root)
            (app / KERNEL_LIBRARY).unlink()
            with self.assertRaisesRegex(BinaryPrivacyError, "libRADSDF.dylib"):
                prepare_app_binaries(app, root / "symbols", build_roots=(root,))
            # The executables are stripped now; the missing library still fails.
            with self.assertRaisesRegex(BinaryPrivacyError, "libRADSDF.dylib"):
                verify_app_binaries(app, build_roots=(root,))


if __name__ == "__main__":
    unittest.main()
