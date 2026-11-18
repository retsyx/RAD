"""Moving assembly definitions and posed native geometry operations."""
import copy
import math
import unittest
from unittest import mock

import numpy as np

from sdf_test_support import build, native, read_stl, signed_volume
from assembly_motion import AssemblyMotionError, coupled_position, validate_motion
from sdf_build import ModelError
import sdf_ops


def validate_assembly_motion(definition, parts):
    """The motion rules as builds apply them: violations are design errors."""
    try:
        return validate_motion(definition, parts)
    except AssemblyMotionError as error:
        raise ModelError(str(error)) from error


def connector(identity, part, origin=(0, 0, 0)):
    return {"id": identity, "part": part, "origin": list(origin),
            "xAxis": [1, 0, 0], "zAxis": [0, 0, 1]}


def definition():
    return {"grounded": ["base"],
            "connectors": [connector("base-pin", "base", (5, 0, 0)),
                           connector("arm-pin", "arm", (25, 0, 0))],
            "joints": [{"id": "hinge", "name": "Arm hinge", "type": "revolute",
                        "parent": "base-pin", "child": "arm-pin", "limits": [-90, 90]}]}


def physical_parts():
    return [{"id": "group", "name": "Mechanism", "solidIndices": [], "children": [
        {"id": "base", "name": "Base", "solidIndices": [0], "children": None},
        {"id": "arm", "name": "Arm", "solidIndices": [1], "children": None},
        {"id": "link", "name": "Link", "solidIndices": [2], "children": None}]}]


def rotation_y(angle, translation=(0, 0, 0)):
    cosine, sine = math.cos(math.radians(angle)), math.sin(math.radians(angle))
    x, y, z = translation
    return [cosine, 0, sine, x, 0, 1, 0, y, -sine, 0, cosine, z, 0, 0, 0, 1]


class AssemblyDefinitionTests(unittest.TestCase):
    def test_build_emits_detached_world_frames_without_changing_built_geometry(self):
        motion = definition()
        script = f"""
result = Assembly('Mechanism', at=Placement(translate=(5, 0, 0)))
result.add(box((1, 2, 3), center=(0.5, 1, 1.5)), id='base', name='Base')
result.add(box((2, 3, 4), center=(1, 1.5, 2)), id='arm', name='Arm', at=Placement(translate=(20, 0, 0)))
result.motion = {motion!r}
"""
        preview, _ = build(self, script)
        self.assertEqual(preview["assemblyMotion"]["grounded"], ["base"])
        self.assertEqual(preview["assemblyMotion"]["connectors"], motion["connectors"])
        self.assertEqual(preview["assemblyMotion"]["joints"][0]["position"], 0)
        for actual, expected in zip(preview["bounds"], [5, 0, 0, 27, 3, 4]):
            self.assertAlmostEqual(actual, expected, delta=1e-9)
        self.assertEqual([part["solidIndices"] for part in preview["parts"][0]["children"]], [[0], [1]])
        plain, _ = build(self, "result = box((1, 2, 3))")
        self.assertNotIn("assemblyMotion", plain)

    def test_closed_loop_and_multiple_grounded_parts_are_retained_for_native_solver(self):
        motion = definition()
        motion["grounded"].append("arm")
        motion["connectors"].append(connector("link-pin", "link"))
        motion["joints"] += [
            {"id": "slider", "type": "slider", "parent": "arm-pin", "child": "link-pin", "position": 3},
            {"id": "closure", "type": "fixed", "parent": "link-pin", "child": "base-pin"}]
        actual = validate_assembly_motion(motion, physical_parts())
        self.assertEqual(len(actual["joints"]), 3)
        self.assertEqual(actual["joints"][1]["name"], "slider")
        motion["connectors"][0]["origin"][0] = 900
        self.assertEqual(actual["connectors"][0]["origin"][0], 5)

    def test_small_axis_roundoff_is_normalized_to_orthonormal_frames(self):
        motion = definition()
        motion["connectors"][0]["xAxis"] = [1.0000001, 0, 0.0000001]
        motion["connectors"][0]["zAxis"] = [0, 0, 0.9999999]
        frame = validate_assembly_motion(motion, physical_parts())["connectors"][0]
        self.assertEqual(frame["xAxis"], [1, 0, 0])
        self.assertEqual(frame["zAxis"], [0, 0, 1])

    def test_omitted_connector_axes_use_world_x_and_z(self):
        motion = definition()
        del motion["connectors"][1]["xAxis"]
        del motion["connectors"][1]["zAxis"]
        frame = validate_assembly_motion(motion, physical_parts())["connectors"][1]
        self.assertEqual(frame["xAxis"], [1, 0, 0])
        self.assertEqual(frame["zAxis"], [0, 0, 1])

    def test_invalid_frames_references_limits_and_ungrounded_components_fail(self):
        invalid = []

        def changed(change):
            value = definition()
            change(value)
            invalid.append(value)

        changed(lambda value: value.update(grounded=[]))
        changed(lambda value: value.update(grounded=["group"]))
        changed(lambda value: value.update(grounded=["base", "base"]))
        changed(lambda value: value["connectors"][1].update(id="base-pin"))
        changed(lambda value: value["connectors"][1].update(part="missing"))
        changed(lambda value: value["connectors"][1].update(part="group"))
        changed(lambda value: value["connectors"][1].update(origin=[0, float("nan"), 0]))
        changed(lambda value: value["connectors"][1].update(xAxis=[2, 0, 0]))
        changed(lambda value: value["connectors"][1].update(zAxis=[1, 0, 0]))
        changed(lambda value: value["joints"][0].update(parent="missing"))
        changed(lambda value: value["joints"][0].update(child="base-pin"))
        changed(lambda value: value["joints"][0].update(type="ball"))
        # The app refuses control and format characters such as zero-width spaces.
        changed(lambda value: value["joints"][0].update(name="Hinge\u200b"))
        changed(lambda value: (value["connectors"][1].update(id="arm-pin\u2060"),
                               value["joints"][0].update(child="arm-pin\u2060")))
        changed(lambda value: value["joints"][0].update(position=float("inf")))
        changed(lambda value: value["joints"][0].update(position=True))
        changed(lambda value: value["joints"][0].update(limits=[90, -90]))
        changed(lambda value: value["joints"][0].update(position=91))
        changed(lambda value: value["joints"][0].update(type="fixed"))
        changed(lambda value: value.update(grounded=["link"]))
        changed(lambda value: value["joints"].append(copy.deepcopy(value["joints"][0])))
        changed(lambda value: value.update(joints=value["joints"] * 257))
        for index, value in enumerate(invalid):
            with self.subTest(index=index), self.assertRaises(ModelError):
                validate_assembly_motion(value, physical_parts())

    def test_bad_construction_metadata_rejects_candidate(self):
        script = "result = Assembly()\nresult.add(box((1, 2, 3)), id='box')\nresult.motion = {}"
        with self.assertRaisesRegex(ModelError, "Assembly motion requires"):
            build(self, script)


def coupled_definition(coupling):
    """Two wheels on a grounded base; the link follows the arm."""
    motion = definition()
    motion["connectors"] += [connector("base-link", "base", (40, 0, 0)), connector("link-axis", "link", (40, 0, 0))]
    motion["joints"].append({"id": "index", "name": "Link index", "type": "revolute",
                             "parent": "base-link", "child": "link-axis"})
    motion["couplings"] = [{"id": "drive", "driver": "hinge", "follower": "index", **coupling}]
    return motion


GENEVA_LIKE = {"type": "table", "samples": [[0, 0], [120, 0], [240, -60]], "period": 360, "advance": -60}


class AssemblyCouplingTests(unittest.TestCase):
    def test_linear_and_periodic_couplings_normalize_and_derive_follower_positions(self):
        linear = coupled_definition({"type": "linear", "ratio": -0.5, "offset": 2})
        linear["joints"][0]["position"] = 40
        actual = validate_assembly_motion(linear, physical_parts())
        self.assertEqual(actual["couplings"], [{"id": "drive", "type": "linear", "driver": "hinge",
                                                "follower": "index", "ratio": -0.5, "offset": 2}])
        self.assertEqual(actual["joints"][1]["position"], -18)
        periodic = coupled_definition(GENEVA_LIKE)
        periodic["joints"][0].update(position=60, limits=[-720, 720])
        actual = validate_assembly_motion(periodic, physical_parts())
        self.assertEqual(actual["couplings"][0]["period"], 360)
        self.assertEqual(actual["couplings"][0]["samples"], [[0, 0], [120, 0], [240, -60]])
        self.assertEqual(actual["joints"][1]["position"], 0)
        plain = validate_assembly_motion(definition(), physical_parts())
        self.assertEqual(plain["couplings"], [])

    def test_coupled_positions_repeat_each_period_with_its_advance(self):
        coupling = {"id": "drive", "driver": "hinge", "follower": "index", **GENEVA_LIKE}
        closed = dict(coupling, samples=GENEVA_LIKE["samples"] + [[360, -60]])
        for driver, expected in ((0, 0), (60, 0), (180, -30), (240, -60), (360, -60), (540, -90),
                                 (720, -120), (-180, 30), (-360, 60), (1e6, coupled_position(coupling, 1e6))):
            with self.subTest(driver=driver):
                self.assertAlmostEqual(coupled_position(coupling, driver), expected, places=9)
                self.assertAlmostEqual(coupled_position(closed, driver), expected, places=9)
        bounded = {"type": "table", "samples": [[0, 0], [10, 5], [20, 0]]}
        self.assertEqual([coupled_position(bounded, value) for value in (0, 5, 10, 15, 20)], [0, 2.5, 5, 2.5, 0])
        self.assertEqual(coupled_position({"type": "linear", "ratio": 2, "offset": -1}, 3), 5)

    def test_matching_follower_positions_are_accepted_and_contradictions_rejected(self):
        motion = coupled_definition({"type": "linear", "ratio": 2})
        motion["joints"][0]["position"] = 10
        motion["joints"][1]["position"] = 20
        self.assertEqual(validate_assembly_motion(motion, physical_parts())["joints"][1]["position"], 20)
        motion["joints"][1]["position"] = 21
        with self.assertRaisesRegex(ModelError, "must match its coupling"):
            validate_assembly_motion(motion, physical_parts())

    def test_invalid_couplings_fail(self):
        invalid = []

        def changed(coupling, change=None):
            value = coupled_definition(coupling)
            if change:
                change(value)
            invalid.append(value)

        linear = {"type": "linear", "ratio": 1}
        changed(linear, lambda value: value.update(couplings={}))
        changed({"type": "gear", "ratio": 1})
        changed({"type": "linear"})
        changed({"type": "linear", "ratio": 0})
        changed({"type": "linear", "ratio": float("nan")})
        changed({"type": "linear", "ratio": 1, "samples": [[0, 0], [1, 1]]})
        changed(linear, lambda value: value["couplings"][0].update(driver="missing"))
        changed(linear, lambda value: value["couplings"][0].update(follower="hinge"))
        changed(linear, lambda value: value["couplings"][0].pop("driver"))
        changed(linear, lambda value: value["couplings"].append(dict(value["couplings"][0])))
        changed(linear, lambda value: value["couplings"].append(dict(value["couplings"][0], id="again")))
        changed(linear, lambda value: value["couplings"].append(
            {"id": "back", "type": "linear", "driver": "index", "follower": "hinge", "ratio": 1}))
        changed(linear, lambda value: (value["joints"][1].update(type="fixed"),))
        changed({"type": "table", "samples": [[0, 0]]})
        changed({"type": "table", "samples": [[0, 0], [0, 1]]})
        changed({"type": "table", "samples": [[0, 0], [1]]})
        changed({"type": "table", "samples": [[0, 0], [1, 1]] * 2049})
        changed({"type": "table", "samples": [[0, 0], [360, 1]], "period": 360, "advance": 0})
        changed({"type": "table", "samples": [[0, 0], [361, 1]], "period": 360})
        changed({"type": "table", "samples": [[0, 0], [1, 1]], "period": 0})
        changed({"type": "table", "samples": [[0, 0], [1, 1]], "advance": 1})
        changed({"type": "table", "samples": [[0, 0], [90, 1]]})
        changed({"type": "table", "samples": [[0, 0], [90, 1]]},
                lambda value: value["joints"][0].update(limits=[-10, 90]))
        changed(linear, lambda value: value["joints"][1].update(limits=[5, 10]))
        for index, value in enumerate(invalid):
            with self.subTest(index=index), self.assertRaises(ModelError):
                validate_assembly_motion(value, physical_parts())

    def test_nonperiodic_table_accepts_driver_limits_within_its_samples(self):
        motion = coupled_definition({"type": "table", "samples": [[-90, 0], [0, 10], [90, 0]]})
        motion["joints"][0].update(position=45)
        self.assertEqual(validate_assembly_motion(motion, physical_parts())["joints"][1]["position"], 5)


class NativePoseTests(unittest.TestCase):
    def assert_vector(self, actual, expected, places=7, delta=None):
        self.assertEqual(len(actual), len(expected))
        for a, b in zip(actual, expected):
            if delta is None:
                self.assertAlmostEqual(a, b, places=places)
            else:
                self.assertAlmostEqual(a, b, delta=delta)

    def metric(self, report, identity):
        return next(item["values"] for item in report["metrics"] if item["id"] == identity)

    def test_posed_stl_export_preserves_geometry_source_and_subset(self):
        result, output = build(self, "result = [box((1, 2, 3)), cylinder(2, 5, center=(10, 0, 2.5))]")
        source = (output / "model.rsdf").read_bytes()
        pose = [{"solid": 1, "matrix": rotation_y(90, (20, 5, 7))}]
        report, folder = native(output, result, "export", solidIndices=[1], solidTransforms=pose)
        corners, _ = read_stl(folder / "model.stl")
        self.assertEqual(report["solids"], 1)
        # Simplification keeps the flat ends and inscribes chords in the
        # side, removing on average two thirds of their depth (at most the
        # tolerance) over its 20π mm² for a volume of 20π mm³.
        self.assertAlmostEqual(signed_volume(corners) / (20 * math.pi), 1, delta=2e-3 + 2 / 3 * report["tolerance"])
        points = corners.reshape(-1, 3)
        # The posed cylinder axis runs along +X from x = 20 to 25 around (y 5, z -3).
        self.assert_vector(np.concatenate([points.min(axis=0), points.max(axis=0)]).tolist(),
                           [20, 3, -5, 25, 7, -1], delta=0.01)
        self.assertEqual((output / "model.rsdf").read_bytes(), source)
        self.assert_vector(result["partStats"][1]["centroid"], [10, 0, 2.5], delta=1e-3)

    def test_posed_measurements_use_the_parts_fields_at_their_poses(self):
        result, output = build(self, "result = [box((1, 2, 3), center=(0.5, 1, 1.5)), "
                                     "box((1, 2, 3), center=(10.5, 1, 1.5))]")
        pose = [{"solid": 1, "matrix": rotation_y(0, (5, 4, 0))}]
        report, _ = native(output, result, "measure", solidTransforms=pose,
                           entities=[{"kind": "part", "solid": 0}, {"kind": "part", "solid": 1}])
        self.assertAlmostEqual(self.metric(report, "pair:distance")[0], math.hypot(14, 2), places=7)
        # The closest points lie on parallel edges, found to within 1e-6 mm.
        self.assert_vector(report["closestPoints"][1][:2], [15, 4], places=6)
        # A point on the posed part is given and returned in built coordinates.
        report, _ = native(output, result, "measure", solidTransforms=pose,
                           entities=[{"kind": "point", "solid": 1, "point": [10, 1, 1.5]},
                                     {"kind": "point", "solid": 0, "point": [1, 1, 1.5]}])
        self.assert_vector(report["points"][0]["point"], [10, 1, 1.5])
        self.assert_vector(report["points"][0]["normal"], [-1, 0, 0])
        self.assert_vector(self.metric(report, "pair:first"), [15, 5, 1.5])
        self.assertAlmostEqual(self.metric(report, "pair:distance")[0], math.hypot(14, 4), places=7)
        rotated = [{"solid": 1, "matrix": rotation_y(90, (0, 0, 0))}]
        report, _ = native(output, result, "measure", solidTransforms=rotated,
                           entities=[{"kind": "point", "solid": 1, "point": [11, 1, 1.5]}, {"kind": "part", "solid": 0}])
        # rotation_y(90) maps (11, 1, 1.5) to (1.5, 1, -11).
        self.assert_vector(self.metric(report, "pair:first"), [1.5, 1, -11])
        self.assert_vector(self.metric(report, "pair:second"), [1, 1, 0], places=6)

    def test_clearance_separates_gaps_contact_and_interference_at_each_pose(self):
        result, output = build(self, "result = [box(1, center=(0.5, 0.5, 0.5)), box(1, center=(5.5, 0.5, 0.5)), "
                                     "box(1, center=(0.5, 9.5, 0.5))]")
        poses = [[], [{"solid": 1, "matrix": rotation_y(0, (-4, 0, 0))}],
                 [{"solid": 1, "matrix": rotation_y(0, (-4.5, 0, 0))}]]
        report, _ = native(output, result, "clearance", pairs=[[[0], [1]], [[0, 2], [1]]], poses=poses)
        self.assertEqual(report["tolerance"], sdf_ops.CONTACT_TOLERANCE)
        gap, touching, overlapping = report["samples"]
        self.assertAlmostEqual(gap[0]["distance"], 4, places=9)
        self.assertNotIn("overlapVolume", gap[0])
        self.assertAlmostEqual(gap[0]["closestPoints"][0][0], 1, places=9)
        self.assertAlmostEqual(gap[0]["closestPoints"][1][0], 5, places=9)
        self.assertEqual(touching[0]["distance"], 0)
        self.assertEqual(touching[0]["overlapVolume"], 0)
        self.assertEqual(overlapping[0]["distance"], 0)
        self.assertAlmostEqual(overlapping[0]["overlapVolume"], 0.5, delta=0.01)
        # A grouped side measures the nearest member and its combined overlap.
        self.assertAlmostEqual(overlapping[1]["overlapVolume"], 0.5, delta=0.01)
        self.assertAlmostEqual(gap[1]["distance"], 4, places=9)

    def test_curved_contact_and_thin_interference_are_told_apart(self):
        result, output = build(self, "result = [sphere(2), cylinder(1, 6, center=(5, 0, 0))]")
        poses = [[{"solid": 1, "matrix": rotation_y(0, (-2, 0, 0))}],
                 [{"solid": 1, "matrix": rotation_y(0, (-2.05, 0, 0))}],
                 [{"solid": 1, "matrix": rotation_y(0, (-1.95, 0, 0))}]]
        report, _ = native(output, result, "clearance", pairs=[[[0], [1]]], poses=poses)
        touching, overlapping, gap = (sample[0] for sample in report["samples"])
        self.assertEqual(touching["distance"], 0)
        self.assertLess(touching["overlapVolume"], 1e-6)
        self.assertEqual(overlapping["distance"], 0)
        self.assertGreater(overlapping["overlapVolume"], 1e-4)
        self.assertAlmostEqual(gap["distance"], 0.05, places=7)

    def test_invalid_clearance_requests_fail_before_measuring(self):
        result, output = build(self, "result = [box(1), box(1, center=(5, 0, 0))]")
        valid = {"pairs": [[[0], [1]]], "poses": [[]]}
        invalid = [dict(valid, pairs=[]), dict(valid, pairs=[[[0], [1]]] * 9), dict(valid, pairs=[[[0]]]),
                   dict(valid, pairs=[[[0], []]]), dict(valid, pairs=[[[0], [0]]]), dict(valid, pairs=[[[0], [2]]]),
                   dict(valid, pairs=[[[0, 0], [1]]]), dict(valid, pairs=[[None, [1]]]), dict(valid, poses=[]),
                   dict(valid, poses=[[]] * 362), dict(valid, poses=[[{"solid": 1, "matrix": [2] * 16}]]),
                   dict(valid, extra=True)]
        for spec in invalid:
            with (self.subTest(spec=spec), self.assertRaises(ModelError),
                  mock.patch.object(sdf_ops, "separation", side_effect=AssertionError("No measurement"))):
                native(output, result, "clearance", **spec)

    def test_nonrigid_ambiguous_or_nonfinite_poses_fail_before_use(self):
        identity = rotation_y(0)
        matrices = []
        for index, value in ((0, 2), (0, -1), (1, .1), (12, .1), (15, 0), (3, float("nan"))):
            matrix = identity.copy()
            matrix[index] = value
            matrices.append(matrix)
        invalid = [[{"solid": 0, "matrix": matrix}] for matrix in matrices]
        invalid += [[{"solid": True, "matrix": identity}], [{"solid": 1, "matrix": identity}],
                    [{"solid": 0, "matrix": identity}] * 2, [{"solid": 0, "matrix": identity[:12]}], {}]
        for records in invalid:
            with self.subTest(records=records), self.assertRaises(ModelError):
                sdf_ops.Pose(records, 1)
        pose = sdf_ops.Pose([{"solid": 0, "matrix": rotation_y(90, (1, 2, 3))}], 1)
        point = np.array([[1.0, 0.0, 0.0]])
        self.assert_vector(pose.apply(0, point)[0].tolist(), [1, 2, 2])
        self.assert_vector(pose.local(0, pose.apply(0, point))[0].tolist(), [1, 0, 0])


if __name__ == "__main__":
    unittest.main()
