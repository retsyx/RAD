"""Opaque sRGB colors remain semantic part metadata."""
import copy
import json
import math
from pathlib import Path
import shutil
import sys
import tempfile
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from sdf_build import ModelError, validate_parts
from worker import run


def spec(script):
    return {"title": "Colored assembly", "summary": "Assembly colors", "units": "mm", "script": script}


def flatten(parts):
    return [node for part in parts for node in [part, *flatten(part.get("children") or [])]]


COLORED = """
result = Assembly('Product', id='product', color=(0.2, 0.4, 0.6))
result.add(box((2, 3, 4)), id='body', name='Body')
group = result.folder('Case', id='case', color='#B34D1A')
group.add(box((1, 2, 3)).translate((10, 0, 0)) | box((3, 2, 1)).translate((20, 0, 0)), id='pair', name='Pair')
group.add(box(2), id='lid', name='Lid', at=Placement(translate=(0, 20, 0)), color=(0.8, 0.1, 0.4))
result.add(box(1), id='post', name='Post', at=Placement(translate=(0, 0, 20)))
"""


class PartColorTests(unittest.TestCase):
    def build(self, script):
        folder = Path(tempfile.mkdtemp())
        self.addCleanup(shutil.rmtree, folder)
        (folder / "request.json").write_text(json.dumps(spec(script)))
        return run(folder / "request.json", folder, 2)

    def assert_color(self, node, expected):
        self.assertEqual(set(node["color"]), {"red", "green", "blue"})
        for channel, value in zip(("red", "green", "blue"), expected):
            self.assertAlmostEqual(node["color"][channel], value, places=6)

    def test_parent_inheritance_child_override_and_ownership(self):
        colored = self.build(COLORED)
        nodes = {node["id"]: node for node in flatten(colored["parts"])}
        self.assertEqual([nodes[key]["solidIndices"] for key in ("body", "pair", "lid", "post")],
                         [[0], [1], [2], [3]])
        self.assert_color(nodes["body"], (0.2, 0.4, 0.6))
        self.assert_color(nodes["pair"], (0xB3 / 255, 0x4D / 255, 0x1A / 255))
        self.assert_color(nodes["lid"], (0.8, 0.1, 0.4))
        self.assert_color(nodes["post"], (0.2, 0.4, 0.6))
        for key in ("product", "case"):
            self.assertNotIn("color", nodes[key], "A structural group owns no physical finish")
        plain_source = COLORED
        for color in (", color=(0.2, 0.4, 0.6)", ", color='#B34D1A'", ", color=(0.8, 0.1, 0.4)"):
            plain_source = plain_source.replace(color, "")
        plain = self.build(plain_source)
        stripped = copy.deepcopy(colored["parts"])
        for node in flatten(stripped):
            node.pop("color", None)
        self.assertEqual(stripped, plain["parts"], "Colors must not alter IDs, labels, order or ownership")
        self.assertEqual(colored["partStats"], plain["partStats"])

    def test_anonymous_root_and_user_folder_apply_inherited_color_to_physical_owners(self):
        source = """
result = Assembly(color=(0.25, 0.5, 0.75))
result.add(box((1, 2, 3)), id='body', name='Body')
result.add(box((2, 3, 4)).translate((10, 0, 0)), id='user-folder:test', name='Folder')
"""
        parts = self.build(source)["parts"]
        self.assertEqual([part["id"] for part in parts], ["body", "user-folder:test"])
        self.assert_color(parts[0], (0.25, 0.5, 0.75))
        self.assertNotIn("color", parts[1])
        self.assertEqual(parts[1]["children"][0]["id"], 'body:["user-folder:test",0]')
        self.assert_color(parts[1]["children"][0], (0.25, 0.5, 0.75))

    def test_invalid_and_nonopaque_colors_are_rejected_with_actionable_error(self):
        for color in ("(0.2, 0.4, 0.6, 0.5)", "(0.2, 0.4)", "(1.2, 0, 0)", "(math.nan, 0, 0)", "'#12345'",
                      "'red'", "(True, 0, 0)"):
            source = f"result = Assembly()\nresult.add(box(1), name='Box', color={color})"
            with self.subTest(color=color), self.assertRaisesRegex(ModelError, r"color must be \(r, g, b\)"):
                self.build(source)

    def test_saved_color_schema_is_optional_strict_and_finite(self):
        part = {"id": "body", "name": "Body", "solidIndices": [0], "children": None}
        self.assertEqual(validate_parts([part], 1), [part])
        for color in (None, {"red": 0.1, "green": 0.2, "blue": 0.3}):
            candidate = {**part, "color": color}
            self.assertEqual(validate_parts([candidate], 1), [candidate])
        invalid = [[], "red", {}, {"red": 0, "green": 0}, {"red": 0, "green": 0, "blue": 0, "alpha": 0.5}]
        for value in (-0.01, 1.01, 10 ** 1000, math.nan, math.inf, True, "0.5", None):
            invalid.append({"red": value, "green": 0.2, "blue": 0.3})
        for color in invalid:
            with self.subTest(color=color), self.assertRaisesRegex(ModelError, "finite sRGB"):
                validate_parts([{**part, "color": color}], 1)


if __name__ == "__main__":
    unittest.main()
