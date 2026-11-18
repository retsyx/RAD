"""Shared helpers for Engine tests: builds through the worker, staged native
operations on a build's artifact, and binary STL readback."""
import itertools
import json
from pathlib import Path
import shutil
import struct
import sys
import tempfile

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


def read_stl(path):
    """(triangle corners (N, 3, 3) float64, facet normals (N, 3)) of a binary STL."""
    data = Path(path).read_bytes()
    count = struct.unpack_from("<I", data, 80)[0]
    assert len(data) == 84 + 50 * count, "binary STL size must match its triangle count"
    records = np.frombuffer(data, dtype=np.dtype([("normal", "<f4", (3,)), ("vertices", "<f4", (3, 3)),
                                                  ("attribute", "<u2")]), count=count, offset=84)
    return records["vertices"].astype(np.float64), records["normal"].astype(np.float64)


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
