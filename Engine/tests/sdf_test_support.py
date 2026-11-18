"""Shared helpers for Engine tests: builds through the worker, staged native
operations on a build's artifact, and 3MF readback."""
import itertools
import json
from pathlib import Path
import shutil
import sys
import tempfile
import xml.etree.ElementTree as ET
import zipfile

import numpy as np

ENGINE = Path(__file__).resolve().parents[1]
ROOT = ENGINE.parent
if str(ENGINE) not in sys.path:
    sys.path.insert(0, str(ENGINE))

import worker  # noqa: E402

_counter = itertools.count()


def script_spec(script, title="Python design", **extra):
    return {"title": title, "summary": "SDF geometry test", "units": "mm", "script": script, **extra}


def temporary_folder(test):
    folder = Path(tempfile.mkdtemp())
    test.addCleanup(shutil.rmtree, folder, True)
    return folder


def build(test, script, threads=2, **extra):
    """(result, output folder) of a worker build of `script`."""
    folder = temporary_folder(test) / "build"
    folder.mkdir()
    (folder / "request.json").write_text(json.dumps(script_spec(script, **extra)))
    return worker.run(folder / "request.json", folder, threads), folder


def native(build_folder, result, operation, threads=2, serving=False, **fields):
    """(report, folder) of a native operation on a build's staged artifact."""
    folder = build_folder.parent / f"{operation}-{next(_counter)}"
    folder.mkdir()
    shutil.copyfile(build_folder / "model.rsdf", folder / "source.rsdf")
    request = {"operation": operation, "artifact": "source.rsdf",
               "artifactIdentity": str(build_folder / "model.rsdf"), "partRoots": result["partRoots"], **fields}
    (folder / "request.json").write_text(json.dumps(request))
    report = worker.run(folder / "request.json", folder, threads, native_only=serving)
    assert json.loads((folder / "result.json").read_text()) == json.loads(json.dumps(report))
    return report, folder


def read_3mf(path):
    """Triangle corners and ordered (name, sRGB hex color, vertices, triangles) objects."""
    namespaces = {"": "http://schemas.microsoft.com/3dmanufacturing/core/2015/02",
                  "m": "http://schemas.microsoft.com/3dmanufacturing/material/2015/02"}
    with zipfile.ZipFile(path) as package:
        relations = ET.fromstring(package.read("_rels/.rels"))
        model_path, = [relation.attrib["Target"].lstrip("/") for relation in relations
                       if relation.attrib["Type"] == "http://schemas.microsoft.com/3dmanufacturing/2013/01/3dmodel"]
        types = ET.fromstring(package.read("[Content_Types].xml"))
        assert any(entry.get("Extension") == "model" or entry.get("PartName") == "/" + model_path for entry in types
                   if entry.get("ContentType") == "application/vnd.ms-package.3dmanufacturing-3dmodel+xml")
        model = ET.fromstring(package.read(model_path))
    assert model.attrib["unit"] == "millimeter"
    colors = {group.attrib["id"]: [entry.attrib["color"] for entry in group.findall("m:color", namespaces)]
              for group in model.findall("resources/m:colorgroup", namespaces)}
    objects = {obj.attrib["id"]: obj for obj in model.findall("resources/object", namespaces)}
    items = []
    for item in model.findall("build/item", namespaces):
        assert item.get("transform") is None  # Pose is baked into exported coordinates.
        obj = objects[item.attrib["objectid"]]
        color = colors[obj.attrib["pid"]][int(obj.attrib["pindex"])]
        vertices = np.array([[float(vertex.attrib[axis]) for axis in ("x", "y", "z")]
                             for vertex in obj.findall("mesh/vertices/vertex", namespaces)], dtype=np.float64)
        triangles = np.array([[int(triangle.attrib[index]) for index in ("v1", "v2", "v3")]
                              for triangle in obj.findall("mesh/triangles/triangle", namespaces)], dtype=np.int64)
        assert np.isfinite(vertices).all() and 0 <= triangles.min() and triangles.max() < len(vertices)
        items.append((obj.attrib["name"], color, vertices, triangles))
    return np.concatenate([vertices[triangles] for _, _, vertices, triangles in items]), items


def signed_volume(corners):
    a, b, c = corners[:, 0], corners[:, 1], corners[:, 2]
    return float(np.einsum("ij,ij->i", a, np.cross(b, c)).sum() / 6)


def edge_uses(triangles):
    """Directed edges (3N, 2) of triangles given as vertex index rows."""
    triangles = np.asarray(triangles).reshape(-1, 3)
    directed = np.concatenate([triangles[:, [0, 1]], triangles[:, [1, 2]], triangles[:, [2, 0]]])
    return directed


def assert_closed_and_outward(test, positions, triangles):
    """Every edge joins exactly two triangles with opposite directions
    (watertight, consistently oriented), and the enclosed volume is positive.
    Vertices split for crease shading are welded by their identical positions."""
    positions = np.asarray(positions, dtype=np.float32).reshape(-1, 3)
    _, weld = np.unique(positions, axis=0, return_inverse=True)
    triangles = weld.reshape(-1)[np.asarray(triangles, dtype=np.int64).reshape(-1, 3)]
    directed = edge_uses(triangles)
    unique_directed, counts = np.unique(directed, axis=0, return_counts=True)
    test.assertTrue((counts == 1).all(), "Each directed edge appears once (consistent winding).")
    reverse = {tuple(edge) for edge in unique_directed.tolist()}
    test.assertTrue(all((b, a) in reverse for a, b in unique_directed.tolist()), "Every edge has a twin.")
    welded = np.zeros((int(weld.max()) + 1, 3))
    welded[weld.reshape(-1)] = positions
    test.assertGreater(signed_volume(welded[triangles]), 0)
