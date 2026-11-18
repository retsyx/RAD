"""Programs define their 3D printing version through `rad.print`."""
import json
import math
from pathlib import Path
import shutil
import sys
import tempfile
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import radsdf
import worker
from sdf_build import ModelError, validate_spec, validation_hash

COUNTERBORE = '''
plate = box((40, 20, 6))
plate = plate - cylinder(3, 3, center=(0, 0, 1.5))
plate = plate - cylinder(1.6, 10)
if rad.print:
    # One printed layer closes the shaft where the counterbore steps down.
    plate = plate | cylinder(1.6, 0.2, center=(0, 0, -0.1))
peg = cylinder(3, 15, center=(30, 0, 7.5))
result = Assembly("Kit", id="kit", print_down=(0, 0, 1))
result.add(plate, id="plate", name="Plate")
result.add(peg, id="peg", name="Peg", print_down=[0, 0, -3])
'''


def spec(script, printing=False):
    value = {"title": "Kit", "summary": "Printing version test.", "units": "mm", "script": script}
    if printing:
        value["print"] = True
    return value


class PrintVersionTests(unittest.TestCase):
    def build(self, script, printing=False):
        """(result, graph of the published artifact) of one build."""
        folder = Path(tempfile.mkdtemp())
        self.addCleanup(shutil.rmtree, folder)
        (folder / "request.json").write_text(json.dumps(spec(script, printing)))
        result = worker.run(folder / "request.json", folder, 2)
        graph = radsdf.Graph.from_bytes((folder / "model.rsdf").read_bytes(), result["partRoots"])
        self.addCleanup(graph.close)
        return result, graph

    def test_reading_the_flag_defines_a_printing_version_with_extra_geometry(self):
        ordinary, ordinary_graph = self.build(COUNTERBORE)
        printing, printing_graph = self.build(COUNTERBORE, printing=True)
        self.assertTrue(ordinary["printVersion"])
        self.assertTrue(printing["printVersion"])
        # The printed layer fills the shaft just below the counterbore.
        bridge = [[0, 0, -0.1], [1.2, 0, -0.05], [0, -1.2, -0.15]]
        self.assertTrue((ordinary_graph.evaluate(ordinary_graph.roots[0], bridge) > 0).all())
        self.assertTrue((printing_graph.evaluate(printing_graph.roots[0], bridge) < 0).all())
        cell = 0.04
        volumes = [graph.mesh(graph.roots[0], cell_size=cell).stats["volume"]
                   for graph in (ordinary_graph, printing_graph)]
        self.assertAlmostEqual(volumes[1] - volumes[0], math.pi * 1.6 * 1.6 * 0.2, delta=0.1)
        self.assertEqual([part["id"] for part in ordinary["parts"][0]["children"]], ["plate", "peg"])
        self.assertEqual(ordinary["parts"], printing["parts"])

    def test_pinned_directions_are_inherited_overridden_and_normalized(self):
        result, _ = self.build(COUNTERBORE)
        self.assertEqual(result["printDown"], [{"solid": 0, "direction": [0.0, 0.0, 1.0]},
                                               {"solid": 1, "direction": [0.0, 0.0, -1.0]}])

    def test_reference_parts_are_reported_by_both_versions(self):
        script = COUNTERBORE + (
            "board = result.folder('Board', id='board', reference=True)\n"
            "board.add(box((20, 10, 1.6), center=(0, 0, 10)), id='pcb', name='PCB')\n"
            "board.add(box(2, center=(0, 0, 12)), id='button', name='Button', print_down=(1, 0, 0))\n")
        ordinary, _ = self.build(script)
        printing, _ = self.build(script, printing=True)
        for result in (ordinary, printing):
            self.assertEqual(result["referenceSolids"], [2, 3])
            # Reference parts are never printed, so they have no printing direction.
            self.assertEqual([pin["solid"] for pin in result["printDown"]], [0, 1])
            keys = list(result)
            self.assertEqual(keys.index("referenceSolids"), keys.index("printDown") + 1)
        self.assertEqual(ordinary["parts"], printing["parts"])
        self.assertNotIn("referenceSolids", self.build(COUNTERBORE)[0])

    def test_programs_that_never_read_the_flag_have_no_printing_version(self):
        result, _ = self.build("result = box((4, 5, 6))")
        self.assertNotIn("printVersion", result)
        self.assertNotIn("printDown", result)
        printing, _ = self.build("result = box((4, 5, 6))", printing=True)
        self.assertNotIn("printVersion", printing, "A program that ignores rad.print has one geometry.")

    def test_invalid_pins_and_options_fail_before_acceptance(self):
        for value in ("(0, 0, 0)", "(0, 0)", "'down'", "(0, 0, float('nan'))", "(True, 0, 0)"):
            script = f"result = Assembly()\nresult.add(box(1), name='Box', print_down={value})"
            with self.subTest(value=value), self.assertRaisesRegex(ModelError, "print_down"):
                self.build(script)
        for option in (False, 1, "true"):
            value = spec("result = box(1)")
            value["print"] = option
            with self.subTest(option=option), self.assertRaises(ModelError):
                validate_spec(value)

    def test_printing_version_has_its_own_validation_receipt(self):
        with tempfile.TemporaryDirectory() as directory:
            ordinary = validation_hash(spec(COUNTERBORE), Path(directory))
            printing = validation_hash(spec(COUNTERBORE, printing=True), Path(directory))
            self.assertNotEqual(ordinary, printing)
            self.assertEqual(ordinary, validation_hash(spec(COUNTERBORE), Path(directory)))


if __name__ == "__main__":
    unittest.main()
