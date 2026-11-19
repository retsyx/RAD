#!/usr/bin/env python3
"""Validate GPU field programs against the kernel and measure throughput.

Builds each design with the worker (field.bin beside mesh.bin), samples points
near every part's surface and in its bounds, rays toward the surface and
surface normals, evaluates them with the kernel in float64, then runs
Scripts/field-gpu-check (the Metal evaluator) and prints its report.

  RAD_SDF_LIBRARY=.build/release/libRADSDF.dylib \\
    <runtime python> Scripts/field_gpu_validate.py OUT design.json|program.py|synthetic ...

The tool is compiled into OUT. `synthetic` is a built-in design with every
RSDF opcode. Reports land in OUT/<design>/report.json.
"""
import json
import struct
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "Engine"))

import numpy as np  # noqa: E402

import radsdf  # noqa: E402

SYNTHETIC = r'''
k = 2 * math.pi / 6.0
star = [((10 if i % 2 else 6) * math.cos(i * math.pi / 60), (10 if i % 2 else 6) * math.sin(i * math.pi / 60))
        for i in range(120)]
a = union(sphere(5), box((6, 4, 3), center=(4, 0, 0), radius=0.5),
          cylinder(2, 8, center=(-3, 0, 0), axis=(1, 1, 0)), smooth=1.5)
b = difference(a, cone(2, 0.5, 5, center=(0, 0, 2)), capsule((-5, -5, 0), (5, 5, 0), 1), smooth=0.5)
booleans = union(b, torus(6, 1.2, center=(0, 0, -3)).scale((1.0, 1.3, 0.8)),
                 ellipsoid((3, 2, 1.5), center=(8, 4, 2)).rotate(30, axis=(1, 2, 3)))
modifiers = union(box((4, 4, 12), radius=0.3).twist(8).translate(0, 0, 1),
                  sphere(2, center=(3, 3, 0)).mirrored_copy((1, 0, 0)).elongate(1, 0.5, 0).translate(0, 0, 8),
                  box((10, 10, 2)).shell(0.6, side="outside").offset(0.2).translate(0, 0, -8),
                  intersection(sphere(9), half_space((0, 0, 1), (0, 0, 2)), box(14), smooth=0.4).translate(14, 0, 0),
                  sphere(3).shell(0.5, side="center").translate(-12, 0, 0),
                  sphere(3).shell(0.5, side="inside").translate(-12, 8, 0))
prof = union(polygon(star), rect((8, 3), center=(9, 0), radius=0.5), smooth=0.5)
prof2 = difference(circle(5), slot((-2, 0), (2, 0), 1), ellipse((1.5, 3), center=(0, 2)).rotate(20))
prof3 = intersection(rect((10, 10)), half_plane((1, 1), (2, 0)), circle(6).scale((1.0, 0.5)))
profiles = union(prof.extrude(4), prof2.outline(0.4).extrude(6, taper=-1).translate(0, 0, 6),
                 prof3.mirrored_copy((1, 0)).offset(0.3).revolve(14).translate(0, 0, -10),
                 circle(3).loft_to(rect((5, 2)), 8).translate(-18, 0, 0))
g = sin(k * x) * cos(k * y) + sin(k * y) * cos(k * z) + sin(k * z) * cos(k * x)
lattice = implicit(abs(g) - 0.5, bounds=((-8, -8, -8), (8, 8, 8)))
warped = implicit(sphere(6).at(x + 0.5 * sin(y), y, z * 1.2) + 0.1 * triwave(x, 1.5), bounds=((-9, -9, -9), (9, 9, 9)))
misc = implicit(length(x, y, z) - 4 + 0.2 * exp(-(x ** 2)) + 0.1 * log(2 + y * y) + 0.05 * sqrt(1 + z * z)
                + 0.1 * atan2(y, x + 20) - 0.1 * clamp(x / 3, -1, 1) + minimum(x, 1) * 0.01 + maximum(y, -1) * 0.01
                + (x * x) ** 1.5 * 0.0001 + minimum(x, y, smooth=0.5) * 0.01 + maximum(x, y, smooth=0.5) * 0.01
                + box(6).field * 0.01, bounds=((-7, -7, -7), (7, 7, 7)))
fields = union(intersection(box(14), lattice), warped.translate(20, 0, 0), misc.translate(0, 20, 0))
v = np.array([[6, 0, 0], [-6, 0, 0], [0, 6, 0], [0, -6, 0], [0, 0, 6], [0, 0, -6]], float)
t = np.array([[0, 2, 4], [2, 1, 4], [1, 3, 4], [3, 0, 4], [2, 0, 5], [1, 2, 5], [3, 1, 5], [0, 3, 5]])
bv = np.array([[x_, y_, z_] for x_ in (2, 9) for y_ in (-2, 2) for z_ in (-2, 2)], float)
bt = np.array([[0, 1, 3], [0, 3, 2], [4, 6, 7], [4, 7, 5], [0, 4, 5], [0, 5, 1], [2, 3, 7], [2, 7, 6],
               [0, 2, 6], [0, 6, 4], [1, 5, 7], [1, 7, 3]])
meshed = difference(mesh_solid(np.vstack([v, bv]), np.vstack([t, bt + 6])), sphere(2, center=(6, 0, 0)), smooth=0.3)
result = [booleans.translate(-40, 0, 0), modifiers.translate(0, -40, 0), profiles.translate(40, 0, 0),
          fields.translate(0, 40, 0), meshed]
'''


def spec_for(source):
    if source == "synthetic":
        return "synthetic", {"title": "Synthetic", "summary": "every opcode", "units": "mm", "script": SYNTHETIC}
    path = Path(source)
    if path.suffix == ".py":
        return path.stem, {"title": path.stem, "summary": "validation", "units": "mm", "script": path.read_text()}
    spec = json.loads(path.read_text())
    spec = {key: spec[key] for key in ("title", "summary", "units", "script") if key in spec}
    spec.setdefault("summary", "validation")
    spec.setdefault("units", "mm")
    return path.stem.replace(" ", "_"), spec


def mesh_arrays(folder, result):
    manifest = result["meshBuffers"]["sections"]
    data = (folder / "mesh.bin").read_bytes()

    def section(name, dtype, width):
        info = manifest[name]
        values = np.frombuffer(data, dtype=dtype, count=info["count"], offset=info["offset"])
        return values.reshape(-1, width) if width > 1 else values
    return (section("positions", "<f4", 3).astype(np.float64), section("normals", "<f4", 3).astype(np.float64),
            section("triangles", "<u4", 3), section("triangleSolids", "<i4", 1))


def write_samples(folder, result, graph, path, rng, near=20000, scattered=5000, surface=5000, rays=5000):
    positions, normals, triangles, owners = mesh_arrays(folder, result)
    cell = float(result["mesh"]["cellSize"])
    chunks = [struct.pack("<III", 0x53474652, 1, result["solids"])]
    for solid, root in enumerate(graph.roots):
        vertices = np.unique(triangles[owners == solid].reshape(-1))
        rng.shuffle(vertices)
        pick = lambda count: vertices[rng.integers(0, len(vertices), count)]  # noqa: E731
        base = pick(near)
        points = positions[base] + normals[base] * rng.uniform(-1, 1, (near, 1)) * cell \
            + rng.normal(0, 0.2 * cell, (near, 3))
        lo, hi = positions[vertices].min(axis=0) - 2 * cell, positions[vertices].max(axis=0) + 2 * cell
        points = np.vstack([points, lo + (hi - lo) * rng.random((scattered, 3))])
        values = graph.evaluate(root, points)
        on = positions[pick(surface)]
        kernel_normals = graph.gradient(root, on, eps=1e-5)
        same_step = graph.gradient(root, on, eps=1e-3 * cell)  # the GPU's step: float32 error alone
        start = pick(rays)
        origins = positions[start] + 2 * cell * normals[start]
        directions = -normals[start]
        hits, distances = graph.raycast(root, origins, directions, max_distance=4 * cell, tolerance=1e-4 * cell,
                                        max_steps=1024)
        ray_records = np.hstack([origins, directions, np.full((rays, 1), 4 * cell)])
        chunks.append(struct.pack("<IIIId", solid, len(points), surface, rays, cell))
        for array in (points, values, on, kernel_normals, same_step, ray_records,
                      np.column_stack([np.asarray(hits, float), np.asarray(distances, float)])):
            chunks.append(np.ascontiguousarray(array, dtype="<f8").tobytes())
    import os
    chunks.append(frame_rays(result, graph, cell, width=int(os.environ.get("RAD_FIELD_FRAME_WIDTH", 1600))))
    Path(path).write_bytes(b"".join(chunks))


def frame_rays(result, graph, cell, width=1600, inflation=0.25):
    """An orthographic frame of the design (`width` pixels across its
    projected extent): each pixel's ray starts `inflation` cells before the
    kernel's hit on the nearest part (a proxy mesh's start) and may march
    twice that. Rays come per part in 8 x 4 pixel tiles, as SIMD groups
    shade them."""
    lo, hi = np.array(result["bounds"][:3]), np.array(result["bounds"][3:])
    centre, radius = (lo + hi) / 2, np.linalg.norm(hi - lo) / 2
    forward = -np.array([1.0, -2.0, 1.2]) / np.linalg.norm([1.0, -2.0, 1.2])
    right = np.cross(forward, [0, 0, 1.0])
    right /= np.linalg.norm(right)
    up = np.cross(right, forward)
    height = width
    u = (np.arange(width) + 0.5) / width * 2 - 1
    v = (np.arange(height) + 0.5) / height * 2 - 1
    # tile order: 8 x 4 tiles, row-major inside each tile
    tx, ty = np.meshgrid(np.arange(width), np.arange(height))
    order = np.lexsort((tx.ravel() % 8, ty.ravel() % 4, tx.ravel() // 8, ty.ravel() // 4))
    origins = (centre - forward * radius * 1.5)[None, :] + (u[tx.ravel()][:, None] * right + v[ty.ravel()][:, None] * up) * radius
    origins = origins[order]
    best = np.full(len(origins), np.inf)
    owner = np.full(len(origins), -1)
    for solid, root in enumerate(graph.roots):
        hits, distances = graph.raycast(root, origins, forward, max_distance=4 * radius, tolerance=1e-4 * cell,
                                        max_steps=1024)
        closer = np.asarray(hits) & (np.asarray(distances) < best)
        best[closer] = np.asarray(distances)[closer]
        owner[closer] = solid
    chunks = [struct.pack("<II", width, height)]
    start = inflation * cell
    for solid in range(len(graph.roots)):
        mine = np.flatnonzero(owner == solid)
        records = np.hstack([origins[mine] + forward * (best[mine, None] - start), np.tile(forward, (len(mine), 1)),
                             np.full((len(mine), 1), 2 * start)])
        chunks.append(struct.pack("<I", len(mine)))
        chunks.append(np.ascontiguousarray(records, dtype="<f8").tobytes())
    return b"".join(chunks)


def main():
    out = Path(sys.argv[1]).resolve()
    out.mkdir(parents=True, exist_ok=True)
    python = sys.executable
    tool = out / "field-gpu-check"
    sources = [ROOT / "Scripts/field-gpu-check/main.swift"] + [ROOT / f"Sources/RAD/{name}.swift" for name in
                                                               ("FieldProgram", "FieldProgramShaders", "FieldProgramEvaluator")]
    if not tool.exists() or tool.stat().st_mtime < max(source.stat().st_mtime for source in sources):
        subprocess.run(["swiftc", "-O", *map(str, sources), "-o", str(tool)], check=True)
    rng = np.random.default_rng(7)
    for source in sys.argv[2:]:
        name, spec = spec_for(source)
        folder = out / name
        folder.mkdir(exist_ok=True)
        request = folder / "request.json"
        request.write_text(json.dumps(spec))
        started = time.perf_counter()
        build = subprocess.run([python, str(ROOT / "Engine/worker.py"), "--request", str(request), "--output",
                                str(folder)], capture_output=True, text=True)
        if build.returncode != 0:
            print(f"{name}: build failed\n{build.stdout[-2000:]}\n{build.stderr[-2000:]}")
            continue
        wall = time.perf_counter() - started
        result = json.loads((folder / "result.json").read_text())
        with radsdf.Graph.from_bytes((folder / "model.rsdf").read_bytes(), result["partRoots"]) as graph:
            write_samples(folder, result, graph, folder / "samples.bin", rng)
        check = subprocess.run([str(tool), str(folder), str(folder / "samples.bin")], capture_output=True, text=True)
        if check.returncode != 0:
            print(f"{name}: check failed\n{check.stdout[-2000:]}\n{check.stderr[-4000:]}")
            continue
        report = json.loads(check.stdout)
        program = result["fieldProgram"]
        report["build"] = {"buildSeconds": result["buildSeconds"], "wallSeconds": wall,
                           "fieldSeconds": program.get("buildSeconds"), "fieldBytes": 4 * program.get("words", 0),
                           "meshBytes": (folder / "mesh.bin").stat().st_size, "cellSize": result["mesh"]["cellSize"],
                           "meshOnly": [part for part in program.get("parts", []) if part["status"] != "field"]}
        (folder / "report.json").write_text(json.dumps(report, indent=1, sort_keys=True))
        parts = [part for part in report["parts"] if part.get("hasField")]
        worst = lambda key, stat: max((part[key][stat] for part in parts), default=0)  # noqa: E731
        signs = sum(part["signMismatches"] for part in parts), sum(part["signChecked"] for part in parts)
        instructions = report["instructionsPerSurfaceEvaluation"]
        print(f"{name}: parts {len(report['parts'])} meshOnly {len(report['build']['meshOnly'])} "
              f"field {report['build']['fieldBytes'] / 1e6:.2f} MB ({report['build']['fieldSeconds']:.3f} s) "
              f"mesh {report['build']['meshBytes'] / 1e6:.2f} MB build {report['build']['buildSeconds']:.2f} s | "
              f"band err max {worst('bandError', 'max'):.2e} p99 {worst('bandError', 'p99'):.2e} "
              f"rel max {worst('relativeError', 'max'):.1e} | sign {signs[0]}/{signs[1]} "
              f"farSign {sum(part['farSignMismatches'] for part in parts)} | "
              f"normal(1e-3 cell) vs same step p50/p99/max {max((part['normalSameStep']['p50'] for part in parts), default=0):.4f}/"
              f"{max((part['normalSameStep']['p99'] for part in parts), default=0):.3f}/{max((part['normalSameStep']['max'] for part in parts), default=0):.2f} deg, "
              f"vs 1e-5 mm p50/p90 {max((part['normalAngleDegrees']['eps=0.001cell']['p50'] for part in parts), default=0):.3f}/"
              f"{max((part['normalAngleDegrees']['eps=0.001cell']['p90'] for part in parts), default=0):.3f} deg | "
              f"rays agree {min((part['rayHitAgreement'] for part in parts), default=1):.4f} "
              f"dt max {worst('rayDistanceError', 'max'):.1e} | instr/eval p50 {instructions['p50']:.0f} "
              f"p90 {instructions['p90']:.0f} max {instructions['max']:.0f} | "
              f"frame {report['frame']['pixels']} px: {report['frame']['gpuMilliseconds']:.2f} ms, "
              f"{report['frame']['samplesPerPixel']:.1f} samples/px, {report['frame']['evaluationsPerSecond'] / 1e9:.2f} Geval/s, "
              f"instr/sample p50 {report['frame']['instructions']['p50']:.0f} p90 {report['frame']['instructions']['p90']:.0f} "
              f"max {report['frame']['instructions']['max']:.0f}; 2.07 Mpx x 12 samples ~ {report['frame']['estimate2MpxMilliseconds']:.1f} ms")


if __name__ == "__main__":
    main()
