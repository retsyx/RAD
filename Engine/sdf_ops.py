"""Native geometry operations on a build's SDF graph (SDF-DESIGN.md,
"Native operations").

The long-lived serving worker answers these outside the program sandbox and
never runs design programs. Every request stages a clone of a build's
model.rsdf as source.rsdf together with the RSDF node of each part
(partRoots); parts are already placed in the built design's world
coordinates. The loader checks the file strictly before the kernel decodes
it, and a serving worker keeps the two most recently used graphs by their
artifactIdentity. Handlers return plain dictionaries, which the worker
publishes as result.json; STL exports are written beside it.

Measurements are numeric: distances come from the parts' signed distance
fields, seeded by surface samples (vertices of a sampling mesh) and refined
along SDF gradients; volumes and orientations come from meshes.
"""
from __future__ import annotations

import collections
import math
import os
from pathlib import Path
import stat
import struct

import numpy as np

import radsdf
from assembly_motion import AssemblyMotionError, validate_pose_records
from sdf_build import (MAX_GEOMETRY_BYTES, MAX_PARTS, PREVIEW_CELLS_ACROSS, PREVIEW_CREASE_ANGLE,
                       PREVIEW_MINIMUM_CELL, HostError, ModelError, mesh_parts, mesh_pieces, unrefined_triangles)

SOURCE_NAME = "source.rsdf"
SOURCE_FIELDS = frozenset({"operation", "artifact", "artifactIdentity", "partRoots"})
OPERATION_FIELDS = {
    "measure": {"entities", "solidTransforms"},
    "clearance": {"pairs", "poses"},
    "print_orient": {"solidIndices", "printDown"},
    "export": {"solidIndices", "individualParts", "solidTransforms", "cellSize", "tolerance"},
    "validate": {"cellSize"},
    "probe": {"solids", "points", "directions", "maxDistance"},
}
RSDF_HEADER = struct.Struct("<4sIII")
RSDF_RECORD = struct.Struct("<IIII")
RSDF_MAX_NODES = 4_000_000
MESH_OPCODE = 9
TRANSFORM_OPCODE = 20
GRAPH_CACHE_SIZE = 2
# Root records (for sharing printing orientations between placed instances)
# are read from the artifact bytes, which are kept only up to this size.
RECORD_DATA_LIMIT = 64_000_000
COORDINATE_LIMIT = 1e6

# Sampling meshes seed distances and orientations: per part, its largest
# extent over SAMPLE_CELLS_ACROSS (finer once when that has no surface).
SAMPLE_CELLS_ACROSS = 96
SAMPLE_MINIMUM_CELL = 1e-3
SAMPLE_CACHE_TRIANGLES = 4_000_000
CONTACT_TOLERANCE = 1e-6
REFINE_SEEDS = 12
SEED_CANDIDATES = 256
SURFACE_ITERATIONS = 40
PROJECTION_STEPS = 6
OVERLAP_SAMPLE_BUDGET = 2_000_000
OVERLAP_CELLS_ACROSS = 1024
MAX_CLEARANCE_PAIRS = 8
MAX_CLEARANCE_POSES = 361

EXPORT_TRIANGLE_BUDGET = 4_000_000
EXPORT_MINIMUM_CELL = 0.05
EXPORT_MAXIMUM_CELL = 0.5
EXPORT_PROBE_FACTOR = 4
EXPORT_MARGIN = 1.05
MAX_EXPORT_REMESHES = 2
# Export meshes are simplified within this tolerance (mm) or the request's
# "tolerance" (0 disables), at most this fraction of the export cell: walls
# that mesh (at least two cells thick) keep their two sides apart.
EXPORT_TOLERANCE = 0.01
EXPORT_TOLERANCE_CELLS = 0.25
STL_HEADER = b"RAD binary STL, millimetres".ljust(80, b" ")
STL_RECORD = np.dtype([("normal", "<f4", (3,)), ("vertices", "<f4", (3, 3)), ("attribute", "<u2")])

VALIDATE_TRIANGLE_BUDGET = 4_000_000
VALIDATE_VOLUME_CHANGE = 0.02
MAX_WARNINGS = 24

MAX_PROBE_POINTS = 65_536
MAX_PROBE_SAMPLES = 1_048_576
PROBE_TOLERANCE = 1e-5
PROBE_MAX_STEPS = 1024

PRINT_THRESHOLD = math.sqrt(0.5)
PRINT_NORMAL_BIN = 1e-3
PRINT_CONTACT_COSINE = 1.0 - 5e-9  # bed contact: normal within 1e-4 rad of straight down
PRINT_TIE = 1e-4
PRINT_COST_TIE = 0.01
PRINT_PLANE_CELLS = 1.5
PRINT_CONTACT_CELLS = 0.25
PRINT_BED_BAND_CELLS = 1.0


# ------------------------------------------------------------------ sources

_sources = collections.OrderedDict()


def execute(spec, directory, output, threads, *, serving=False):
    """Run one native operation on the staged source.rsdf in `directory`;
    returns the result dictionary. Serving workers reuse loaded graphs."""
    operation = spec.get("operation")
    if operation not in OPERATION_FIELDS:
        raise ModelError(f"Unknown native operation {operation!r}.")
    extra = set(spec) - SOURCE_FIELDS - OPERATION_FIELDS[operation]
    if extra:
        raise ModelError(f"The {operation} request contains unsupported fields: {', '.join(sorted(extra))}.")
    try:
        radsdf.kernel_version()
    except radsdf.SDFError as error:
        raise HostError(str(error)) from error
    source = load_source(spec, Path(directory), serving)
    try:
        return HANDLERS[operation](spec, source, Path(output), threads)
    finally:
        if not serving:
            source.close()


def load_source(spec, directory, serving=False):
    """The request's graph: from the serving cache, or read strictly."""
    if spec.get("artifact") != SOURCE_NAME:
        raise ModelError("Native operations require the private source.rsdf artifact.")
    identity = spec.get("artifactIdentity")
    if not isinstance(identity, str) or not 0 < len(identity) <= 4096:
        raise ModelError("The SDF artifact identity is invalid.")
    roots = spec.get("partRoots")
    if (not isinstance(roots, list) or not 1 <= len(roots) <= MAX_PARTS
            or any(isinstance(root, bool) or not isinstance(root, int) or root < 0 for root in roots)):
        raise ModelError("partRoots must list the RSDF node of each of the design's 1 to 4,096 parts.")
    roots = tuple(roots)
    if serving:
        source = _sources.get(identity)
        if source is not None and source.roots == roots and not source.graph.closed:
            _sources.move_to_end(identity)
            return source
    graph, data = read_artifact(directory / SOURCE_NAME, roots)
    source = Source(graph, data, roots)
    if serving:
        previous = _sources.pop(identity, None)
        if previous is not None:
            previous.close()
        _sources[identity] = source
        while len(_sources) > GRAPH_CACHE_SIZE:
            _sources.popitem(last=False)[1].close()
    return source


def read_artifact(path, roots):
    """(graph, bytes) of a private regular RSDF file holding every root."""
    try:
        descriptor = os.open(path, os.O_RDONLY | os.O_NOFOLLOW)
    except FileNotFoundError:
        raise ModelError("The staged source.rsdf artifact is missing.") from None
    except OSError:
        raise ModelError("The SDF artifact must be a regular private file of at most 512 MB.") from None
    with os.fdopen(descriptor, "rb") as stream:
        info = os.fstat(stream.fileno())
        if (not stat.S_ISREG(info.st_mode) or info.st_nlink != 1
                or not RSDF_HEADER.size < info.st_size <= MAX_GEOMETRY_BYTES):
            raise ModelError("The SDF artifact must be a regular private file of at most 512 MB.")
        data = stream.read(MAX_GEOMETRY_BYTES + 1)
    if len(data) != info.st_size:
        raise ModelError("The SDF artifact changed while it was read.")
    magic, version, nodes, reserved = RSDF_HEADER.unpack_from(data)
    if magic != b"RSDF" or version != 1 or reserved != 0 or not 1 <= nodes <= RSDF_MAX_NODES:
        raise ModelError("The SDF artifact is not a version 1 RSDF graph.")
    for root in roots:
        if root >= nodes:
            raise ModelError(f"Part root {root} is not a node of the SDF artifact, which has {nodes} nodes.")
    try:
        graph = radsdf.Graph.from_bytes(data, roots)
    except radsdf.SDFError as error:
        raise ModelError(f"The SDF artifact could not be loaded: {error}") from error
    try:
        for solid, root in enumerate(roots):
            if graph.bounds(root).lo.shape != (3,):
                raise ModelError(f"Part root {root} of part {solid + 1} is a 2D profile, not a solid.")
    except radsdf.SDFError as error:
        graph.close()
        raise ModelError(f"The SDF artifact could not be loaded: {error}") from error
    except ModelError:
        graph.close()
        raise
    return graph, data


class Source:
    """A loaded build graph with per-part caches (bounds, sampling meshes,
    root records). Parts are addressed by solid index."""

    def __init__(self, graph, data, roots):
        self.graph, self.roots = graph, roots
        self._data = data if len(data) <= RECORD_DATA_LIMIT else None
        self._bounds, self._samples, self._records = {}, collections.OrderedDict(), None
        self._sample_triangles = 0

    @property
    def solids(self):
        return len(self.roots)

    def close(self):
        self.graph.close()

    @staticmethod
    def label(solid):
        return f"Part {solid + 1}"

    def bounds(self, solid):
        """Conservative (lo, hi) of a part in built coordinates."""
        cached = self._bounds.get(solid)
        if cached is None:
            try:
                bounds = self.graph.bounds(self.roots[solid])
            except radsdf.SDFError as error:
                raise ModelError(f"{self.label(solid)} could not be evaluated: {error}") from error
            if np.any(bounds.lo > bounds.hi) or not bounds.finite:
                raise ModelError(f"{self.label(solid)} has no finite, nonempty bounds in the SDF artifact.")
            cached = self._bounds[solid] = (bounds.lo.copy(), bounds.hi.copy())
        return cached

    def lipschitz(self, solid):
        return max(1.0, float(self.graph.lipschitz(self.roots[solid])))

    def mesh(self, solid, cell, threads):
        return self.copied(solid, self.native_mesh(solid, cell, threads), cell)

    def native_mesh(self, solid, cell, threads):
        """The part's mesh kept native (radsdf.NativeMesh), to count first."""
        try:
            return self.graph.native_mesh(self.roots[solid], cell_size=cell, crease_angle=PREVIEW_CREASE_ANGLE,
                                          threads=threads)
        except radsdf.SDFError as error:
            raise self.meshing_error(solid, cell, error) from error

    def copied(self, solid, native, cell, simplify=None):
        """A native_mesh() result as a Mesh, simplified within `simplify` mm
        when given; frees the native mesh."""
        try:
            return native.mesh(simplify)
        except radsdf.SDFError as error:
            raise self.meshing_error(solid, cell, error) from error

    def meshing_error(self, solid, cell, error):
        return ModelError(f"{self.label(solid)} could not be meshed at a cell size of {cell:.4g} mm: {error}")

    def sample_mesh(self, solid, threads):
        """The part's sampling mesh (cached within a byte budget)."""
        mesh = self._samples.get(solid)
        if mesh is not None:
            self._samples.move_to_end(solid)
            return mesh
        lo, hi = self.bounds(solid)
        extent = float(np.max(hi - lo))
        for cells in (SAMPLE_CELLS_ACROSS, 4 * SAMPLE_CELLS_ACROSS):
            mesh = self.mesh(solid, max(extent / cells, SAMPLE_MINIMUM_CELL), threads)
            if len(mesh.triangles):
                break
        else:
            raise ModelError(f"{self.label(solid)} has no surface at a sampling cell size of "
                             f"{max(extent / (4 * SAMPLE_CELLS_ACROSS), SAMPLE_MINIMUM_CELL):.4g} mm.")
        self._samples[solid] = mesh
        self._sample_triangles += len(mesh.triangles)
        while self._sample_triangles > SAMPLE_CACHE_TRIANGLES and len(self._samples) > 1:
            _, evicted = self._samples.popitem(last=False)
            self._sample_triangles -= len(evicted.triangles)
        return mesh

    def root_record(self, solid):
        """(opcode, children, params) of a part's root node, or None when
        the artifact is too large to keep its bytes."""
        if self._records is None:
            if self._data is None:
                return None
            self._records = root_records(self._data, self.roots)
        return self._records[self.roots[solid]]


def root_records(data, roots):
    """{node index: (opcode, children tuple, params tuple)} for the roots."""
    wanted, records = set(roots), {}
    _, _, nodes, _ = RSDF_HEADER.unpack_from(data)
    offset, last = RSDF_HEADER.size, max(wanted)
    for index in range(min(nodes, last + 1)):
        opcode, children, params, _ = RSDF_RECORD.unpack_from(data, offset)
        offset += RSDF_RECORD.size
        if index in wanted:
            child_values = struct.unpack_from(f"<{children}I", data, offset)
            # A MESH root's millions of parameters are never needed here.
            param_values = (struct.unpack_from(f"<{params}d", data, offset + 4 * children)
                            if opcode != MESH_OPCODE else ())
            records[index] = (opcode, child_values, param_values)
        offset += 4 * children + 8 * params
    return records


# ------------------------------------------------------------------ requests

def _integer(value):
    return isinstance(value, int) and not isinstance(value, bool)


def _number(value):
    return isinstance(value, (int, float)) and not isinstance(value, bool) and math.isfinite(value)


def _point(value, limit=COORDINATE_LIMIT):
    return (isinstance(value, list) and len(value) == 3 and all(_number(item) for item in value)
            and all(abs(item) <= limit for item in value))


def selected_solids(indices, solid_count, *, name="The selected parts"):
    """Distinct valid solid indices in request order (None selects all)."""
    if indices is None:
        return list(range(solid_count))
    if (not isinstance(indices, list) or not indices or len(indices) > solid_count
            or any(not _integer(index) or not 0 <= index < solid_count for index in indices)
            or len(set(indices)) != len(indices)):
        raise ModelError(f"{name} contain invalid or duplicate solid indices.")
    return list(indices)


def positive_cell(spec, name):
    value = spec.get("cellSize")
    if value is None:
        return None
    if not _number(value) or value <= 0:
        raise ModelError(f"The {name} cell size must be a positive number of millimetres.")
    return max(float(value), PREVIEW_MINIMUM_CELL)


class Pose:
    """Rigid world-space deltas per solid (missing solids keep their built
    placement): p_posed = R p + t."""

    def __init__(self, records, solid_count):
        try:
            matrices = validate_pose_records(records, solid_count)
        except AssemblyMotionError as error:
            raise ModelError(str(error)) from error
        self.rotations, self.offsets = {}, {}
        for solid, values in matrices.items():
            matrix = np.asarray(values, dtype=np.float64).reshape(4, 4)
            self.rotations[solid], self.offsets[solid] = matrix[:3, :3], matrix[:3, 3]

    def apply(self, solid, points):
        rotation = self.rotations.get(solid)
        return points if rotation is None else points @ rotation.T + self.offsets[solid]

    def rotate(self, solid, vectors):
        rotation = self.rotations.get(solid)
        return vectors if rotation is None else vectors @ rotation.T

    def local(self, solid, points):
        rotation = self.rotations.get(solid)
        return points if rotation is None else (points - self.offsets[solid]) @ rotation

    def box(self, solid, lo, hi):
        """World AABB of a posed local box."""
        if solid not in self.rotations:
            return lo, hi
        corners = np.array([[x, y, z] for x in (lo[0], hi[0]) for y in (lo[1], hi[1]) for z in (lo[2], hi[2])])
        corners = self.apply(solid, corners)
        return corners.min(axis=0), corners.max(axis=0)


# ------------------------------------------------------------------ fields

def group_field(source, solids, pose, points, threads):
    """(value, owner): the smallest posed SDF value of the solids at world
    points (a union of the group) and the solid that attains it."""
    best = np.full(len(points), np.inf)
    owner = np.full(len(points), solids[0], dtype=np.int64)
    for solid in solids:
        values = source.graph.evaluate(source.roots[solid], pose.local(solid, points), threads=threads)
        closer = values < best
        best[closer] = values[closer]
        owner[closer] = solid
    return best, owner


def gradient_step(points):
    """Finite-difference step for measurement gradients: small enough that
    stencils rarely straddle sharp edges, large against rounding."""
    return 1e-8 * (1.0 + float(np.abs(points).max())) if len(points) else 1e-8


def group_normals(source, owner, pose, points, threads):
    """Unit outward SDF normals (world) of each point's owning solid."""
    normals = np.empty_like(points)
    eps = gradient_step(points)
    for solid in np.unique(owner).tolist():
        rows = owner == solid
        local = pose.local(solid, points[rows])
        normals[rows] = pose.rotate(solid, source.graph.gradient(source.roots[solid], local, eps=eps,
                                                                 threads=threads))
    return normals


def project(source, solids, pose, points, threads, steps=PROJECTION_STEPS):
    """World points moved onto the group's surface along its SDF gradients
    (one step for exact distance fields; more for bounds of distance)."""
    points = np.array(points, dtype=np.float64)
    for _ in range(steps):
        values, owner = group_field(source, solids, pose, points, threads)
        if np.all(np.abs(values) <= 1e-12 * (1.0 + np.abs(points).max(axis=1))):
            break
        points = points - values[:, None] * group_normals(source, owner, pose, points, threads)
    return points


def snap(source, solid, point, threads):
    """A point (built coordinates) moved onto its part's surface: projected
    along the field's gradients, else from the nearest vertex of the part's
    sampling mesh (where the gradient leads nowhere, such as a field without
    a slope toward its surface)."""
    built = Pose([], source.solids)
    tolerance = CONTACT_TOLERANCE * (1.0 + float(np.abs(point).max()))

    def projected(start):
        moved = project(source, [solid], built, start[None], threads, steps=16)[0]
        value = float(group_field(source, [solid], built, moved[None], threads)[0][0])
        return moved if np.all(np.isfinite(moved)) and abs(value) <= tolerance else None

    snapped = projected(point)
    if snapped is None:
        positions = source.sample_mesh(solid, threads).positions.astype(np.float64)
        snapped = projected(positions[int(np.argmin(np.sum((positions - point) ** 2, axis=1)))])
    if snapped is None:
        raise ModelError(f"The point {_vector(point)} cannot be moved onto the surface of {source.label(solid)}.")
    return snapped


def posed_samples(source, solids, pose, threads):
    """World positions (float64) of the group's sampling-mesh vertices."""
    return np.concatenate([pose.apply(solid, source.sample_mesh(solid, threads).positions.astype(np.float64))
                           for solid in solids])


def sample_cell(source, solids, threads):
    return max(float(source.sample_mesh(solid, threads).stats.get("cell_size", 0.0) or 0.0) for solid in solids)


def _tangents(normals):
    reference = np.eye(3)[np.argmin(np.abs(normals), axis=1)]
    first = np.cross(normals, reference)
    first /= np.linalg.norm(first, axis=1, keepdims=True)
    return first, np.cross(normals, first)


# Quadratic model stencil in tangent coordinates (in units of the step).
_STENCIL = np.array([[1.0, 0.0], [-1.0, 0.0], [0.0, 1.0], [0.0, -1.0], [1.0, 1.0]])


def surface_minimum(source, near, far, pose, seeds, threads, radius, iterations=SURFACE_ITERATIONS):
    """Points a on `near`'s surface locally minimizing the field of `far`
    (the distance to it, or minus the penetration), from seeds.

    A trust-region Newton method in tangent coordinates: a quadratic model
    fitted to the field at five nearby surface points (projected onto the
    surface) proposes each step, which is kept only when it lowers the field;
    otherwise the region shrinks. This converges quickly even where
    alternating projections crawl (nearly touching curved surfaces), and
    finds crossings that sampling missed. Returns (points, values)."""
    a = project(source, near, pose, seeds, threads)
    value = group_field(source, far, pose, a, threads)[0]
    count = len(a)
    scale = 1.0 + float(np.abs(a).max())
    step = np.full(count, float(radius))
    # Distances are second order in the points' error near a minimum.
    floor = 1e-8 * scale
    active = np.ones(count, dtype=bool)
    for iteration in range(iterations):
        if iteration == 3:
            # Seeds in clearly worse basins stop; the best few continue.
            active &= value <= value.min() + radius
        rows = np.flatnonzero(active)
        if not len(rows):
            break
        points, values, size = a[rows], value[rows], step[rows]
        owner = group_field(source, near, pose, points, threads)[1]
        first, second = _tangents(group_normals(source, owner, pose, points, threads))
        offsets = (_STENCIL[None, :, 0, None] * first[:, None] + _STENCIL[None, :, 1, None] * second[:, None])
        probes = project(source, near, pose, (points[:, None] + offsets * size[:, None, None]).reshape(-1, 3),
                         threads).reshape(len(rows), len(_STENCIL), 3)
        sampled = group_field(source, far, pose, probes.reshape(-1, 3), threads)[0].reshape(len(rows), -1)
        relative = probes - points[:, None]
        u = np.einsum("skj,sj->sk", relative, first)
        v = np.einsum("skj,sj->sk", relative, second)
        # Model f = f0 + g.u + (h11 u^2 + 2 h12 u v + h22 v^2) / 2 through the five probes.
        system = np.stack([u, v, u * u / 2, u * v, v * v / 2], axis=-1)
        rhs = sampled - values[:, None]
        solvable = np.abs(np.linalg.det(system)) > 1e-300
        coefficients = np.zeros((len(rows), 5))
        if solvable.any():
            coefficients[solvable] = np.linalg.solve(system[solvable], rhs[solvable][..., None])[..., 0]
        gradient = coefficients[:, :2]
        hessian = np.stack([coefficients[:, [2, 3]], coefficients[:, [3, 4]]], axis=1)
        determinant = hessian[:, 0, 0] * hessian[:, 1, 1] - hessian[:, 0, 1] ** 2
        convex = solvable & (hessian[:, 0, 0] > 0) & (determinant > 0)
        move = -gradient * (size / np.maximum(np.linalg.norm(gradient, axis=1), 1e-300))[:, None]
        if convex.any():
            move[convex] = -np.linalg.solve(hessian[convex], gradient[convex][..., None])[..., 0]
        length = np.linalg.norm(move, axis=1)
        limit = 2 * size
        move *= np.minimum(1.0, limit / np.maximum(length, 1e-300))[:, None]
        length = np.minimum(length, limit)
        # The model's own estimate of the remaining decrease: none left
        # (a minimum, or a valley of equal values) ends the search.
        predicted = -(np.einsum("si,si->s", gradient, move) + 0.5 * np.einsum("si,sij,sj->s", move, hessian, move))
        candidates = project(source, near, pose, points + move[:, :1] * first + move[:, 1:] * second, threads)
        candidate_values = group_field(source, far, pose, candidates, threads)[0]
        better = candidate_values < values - 1e-12 * scale  # gains below rounding noise do not count
        accepted = rows[better]
        a[accepted], value[accepted] = candidates[better], candidate_values[better]
        step[accepted] = np.clip(2 * length[better], size[better] / 4, radius)
        step[rows[~better]] = size[~better] / 4
        active[rows] = (step[rows] > floor) & ~(better & (length <= floor)) & (predicted > 1e-12 * scale)
    return a, value


def surface_distance(source, solids, pose, points, threads, reach):
    """(distance, hit point) from each world point to the group's surface
    along the group's descending field gradient, by sphere tracing. Boolean
    combinations make fields bounds of distance rather than distances away
    from the surface, so stepping by the field alone can stop short; the
    traced hit is always on the surface. Misses give infinity."""
    values, owner = group_field(source, solids, pose, points, threads)
    directions = -group_normals(source, owner, pose, points, threads)
    distance = np.full(len(points), np.inf)
    for solid in solids:
        rotation = pose.rotations.get(solid)
        local = pose.local(solid, points)
        heading = directions if rotation is None else directions @ rotation
        hits, traced = source.graph.raycast(source.roots[solid], local, heading, max_distance=reach,
                                            tolerance=PROBE_TOLERANCE * 1e-2, max_steps=PROBE_MAX_STEPS,
                                            threads=threads)
        distance = np.where(hits & (traced < distance), traced, distance)
    # Points already on the surface (or inside it) are their own hits.
    distance = np.where(np.abs(values) <= CONTACT_TOLERANCE, 0.0, distance)
    return distance, points + directions * np.where(np.isfinite(distance), distance, 0.0)[:, None]


def separation(source, first, second, pose, threads, seeds=REFINE_SEEDS):
    """Minimum distance between two solid groups at a pose.

    Returns (distance, point on first, point on second, depth). Each group's
    field is evaluated at the other's surface samples. Any negative value
    means the parts interpenetrate: distance 0, both points at the deepest
    point found and depth its penetration. Otherwise the samples with the
    lowest values (bounds of their distance) are traced to the other surface;
    the closest traced samples seed surface_minimum, whose surface points are
    traced again. The smallest traced distance, between two surface points,
    is the result; one within CONTACT_TOLERANCE is contact (distance 0).
    """
    radius = sample_cell(source, (*first, *second), threads)
    boxes = [group_box(source, group, pose, threads) for group in (first, second)]
    reach = 4.0 * float(np.linalg.norm(np.max([box[1] for box in boxes], axis=0)
                                       - np.min([box[0] for box in boxes], axis=0))) + 1.0
    deepest, best = None, None
    for near, far in ((first, second), (second, first)):
        samples = posed_samples(source, near, pose, threads)
        values = group_field(source, far, pose, samples, threads)[0]
        lowest = int(np.argmin(values))
        if values[lowest] < -CONTACT_TOLERANCE and (deepest is None or values[lowest] < deepest[0]):
            deepest = (float(values[lowest]), samples[lowest])
        if deepest is not None:
            continue
        count = min(SEED_CANDIDATES, len(values))
        order = np.argpartition(values, count - 1)[:count]
        traced, _ = surface_distance(source, far, pose, samples[order], threads, reach)
        if best is not None and np.min(traced) > best[0] + radius:
            continue  # refinement cannot get within a sampling cell of the first direction's result
        count = min(seeds, len(order))
        chosen = order[np.argpartition(traced, count - 1)[:count]]
        points, minima = surface_minimum(source, near, far, pose, samples[chosen], threads, radius)
        lowest = int(np.argmin(minima))
        if minima[lowest] < -CONTACT_TOLERANCE:
            deepest = (float(minima[lowest]), points[lowest])
            continue
        candidates = points  # on the near surface; raw mesh vertices may lie off it
        traced, hits = surface_distance(source, far, pose, candidates, threads, reach)
        index = int(np.argmin(traced))
        if np.isfinite(traced[index]) and (best is None or traced[index] < best[0]):
            pair = (candidates[index], hits[index]) if near is first else (hits[index], candidates[index])
            best = (float(traced[index]), *pair)
    if deepest is not None:
        return 0.0, deepest[1], deepest[1].copy(), -deepest[0]
    if best is None:
        raise ModelError("The distance between these parts could not be determined.")
    distance, point_first, point_second = best
    if distance <= CONTACT_TOLERANCE:
        return 0.0, point_first, point_first.copy(), 0.0
    return distance, point_first, point_second, 0.0


_CHILD_OFFSETS = np.array([[x, y, z] for x in (-1, 1) for y in (-1, 1) for z in (-1, 1)], dtype=np.float64)


def group_box(source, solids, pose, threads):
    """World AABB of a group's posed sampling meshes, padded by their cells."""
    boxes = []
    for solid in solids:
        mesh = source.sample_mesh(solid, threads)
        padding = 2.0 * float(mesh.stats.get("cell_size", 0.0) or 0.0)
        positions = mesh.positions.astype(np.float64)
        lo, hi = pose.box(solid, positions.min(axis=0) - padding, positions.max(axis=0) + padding)
        boxes.append((lo, hi))
    return np.min([box[0] for box in boxes], axis=0), np.max([box[1] for box in boxes], axis=0)


def overlap_volume(source, first, second, pose, threads, depth=0.0):
    """Volume common to two solid groups at a pose.

    Integrated on an adaptive grid over the intersection of their posed
    bounds: a block whose centre value of max(f1, f2) proves it entirely
    inside (or outside) both counts whole (or not at all); other blocks are
    split down to a target cell (the region's extent / OVERLAP_CELLS_ACROSS,
    or a quarter of a measured penetration depth) or until
    OVERLAP_SAMPLE_BUDGET evaluations. Each remaining cell counts
    the fraction of its slab below the surface when its centre or a corner
    is inside both. Touching parts give zero.
    """
    lo_first, hi_first = group_box(source, first, pose, threads)
    lo_second, hi_second = group_box(source, second, pose, threads)
    lo, hi = np.maximum(lo_first, lo_second), np.minimum(hi_first, hi_second)
    extent = hi - lo
    if np.any(extent <= 0):
        return 0.0
    size = float(extent.max())
    # Overlaps are often thin sheets: refine far, within the evaluation budget.
    target = max(size / OVERLAP_CELLS_ACROSS, 1e-5)
    if depth > 0:
        target = max(min(target, depth / 4), 1e-5)
    levels = max(0, math.ceil(math.log2(size / (16 * target))))
    side = target * 2 ** levels
    # As in the mesher, an irrational offset keeps faces at simple
    # coordinates off the sample planes.
    origin = lo - (2 - (1 + math.sqrt(5)) / 2) * side
    counts = np.maximum(1, np.ceil((hi - origin) / side - 1e-9)).astype(int)
    axes = [origin[axis] + (np.arange(counts[axis]) + 0.5) * side for axis in range(3)]
    centers = np.stack(np.meshgrid(*axes, indexing="ij"), axis=-1).reshape(-1, 3)
    # The internal stepping bound of twisted, tapered and lofted fields can
    # exceed their reported Lipschitz bound by up to sqrt(2).
    lipschitz = 1.5 * max(source.lipschitz(solid) for solid in (*first, *second))

    def common(points):
        return np.maximum(group_field(source, first, pose, points, threads)[0],
                          group_field(source, second, pose, points, threads)[0])

    volume, evaluated = 0.0, 0
    while len(centers):
        values = common(centers)
        evaluated += len(centers)
        radius = lipschitz * side * math.sqrt(3) / 2
        inside = values < -radius
        boundary = ~inside & (values <= radius)
        volume += np.count_nonzero(inside) * side ** 3
        if side <= target * (1 + 1e-9) or evaluated + 8 * np.count_nonzero(boundary) > OVERLAP_SAMPLE_BUDGET:
            # A remaining cell holds the fraction of its slab below the
            # surface (exact for planar boundaries), but only when one of its
            # centre and corners is inside both: touching parts add nothing.
            cells, centre = centers[boundary], values[boundary]
            corners = common((cells[:, None, :] + _CHILD_OFFSETS[None] * (side / 2)).reshape(-1, 3))
            touched = (centre < 0) | (corners.reshape(len(cells), 8) < 0).any(axis=1)
            volume += float(np.clip(0.5 - centre / side, 0.0, 1.0)[touched].sum()) * side ** 3
            break
        side /= 2
        centers = (centers[boundary][:, None, :] + _CHILD_OFFSETS[None] * (side / 2)).reshape(-1, 3)
    return float(volume)


# ------------------------------------------------------------------ measure

def _vector(values):
    return [float(value) for value in values]


def measure(spec, source, output, threads):
    """Distance between one or two parts or surface points (SDF-DESIGN.md:
    points are snapped to their part's surface in built coordinates; pair
    results are in posed world coordinates)."""
    entities = spec.get("entities")
    if not isinstance(entities, list) or not 1 <= len(entities) <= 2:
        raise ModelError("Measure one or two parts or surface points.")
    parsed = []
    for entity in entities:
        kind = entity.get("kind") if isinstance(entity, dict) else None
        keys = {"kind", "solid", "point"} if kind == "point" else {"kind", "solid"}
        if (kind not in ("part", "point") or set(entity) != keys or not _integer(entity["solid"])
                or not 0 <= entity["solid"] < source.solids or (kind == "point" and not _point(entity["point"]))):
            raise ModelError("Each measured item must be a part or a point on a part, with a valid solid index.")
        parsed.append((kind, entity["solid"], tuple(entity.get("point") or ())))
    if len(parsed) == 2 and parsed[0] == parsed[1]:
        raise ModelError("Choose two different items to measure.")
    pose = Pose(spec.get("solidTransforms", []), source.solids)
    report, points, world = {"metrics": []}, [], []
    for kind, solid, point in parsed:
        if kind == "point":
            snapped = snap(source, solid, np.array(point, dtype=np.float64), threads)
            normal = source.graph.gradient(source.roots[solid], snapped, threads=threads)
            points.append({"solid": solid, "point": _vector(snapped), "normal": _vector(normal)})
            world.append(pose.apply(solid, snapped[None])[0])
        else:
            world.append(None)
    if points:
        report["points"] = points
    if len(parsed) == 1:
        return report
    (kind_a, solid_a, _), (kind_b, solid_b, _) = parsed
    if kind_a == kind_b == "point":
        first, second = world
    elif kind_a == kind_b == "part":
        _, first, second, _ = separation(source, [solid_a], [solid_b], pose, threads)
    else:
        swap = kind_a == "part"
        point = world[1] if swap else world[0]
        part = solid_a if swap else solid_b
        value = float(group_field(source, [part], pose, point[None], threads)[0][0])
        closest = point.copy()
        if value > CONTACT_TOLERANCE:
            lo, hi = group_box(source, [part], pose, threads)
            reach = 4.0 * float(np.linalg.norm(np.maximum(hi, point) - np.minimum(lo, point))) + 1.0
            traced, hits = surface_distance(source, [part], pose, point[None], threads, reach)
            closest = hits[0] if np.isfinite(traced[0]) else project(source, [part], pose, point[None], threads,
                                                                     steps=16)[0]
        first, second = (closest, point) if swap else (point, closest)
    distance = float(np.linalg.norm(second - first))
    if distance <= CONTACT_TOLERANCE:
        distance, second = 0.0, first.copy()

    def metric(identity, label, values):
        report["metrics"].append({"id": identity, "label": label, "values": values, "unit": "mm"})

    metric("pair:distance", "Minimum distance", [distance])
    metric("pair:delta", "Delta X, Y, Z", _vector(second - first))
    metric("pair:first", "Closest point 1", _vector(first))
    metric("pair:second", "Closest point 2", _vector(second))
    report["closestPoints"] = [_vector(first), _vector(second)]
    return report


# ------------------------------------------------------------------ clearance

def clearance(spec, source, output, threads):
    """Minimum distance between part groups at each rigid pose, with the
    common volume of touching or interfering groups."""
    pairs, poses = spec.get("pairs"), spec.get("poses")
    if not isinstance(pairs, list) or not 1 <= len(pairs) <= MAX_CLEARANCE_PAIRS:
        raise ModelError("A clearance check requires between 1 and 8 part pairs.")
    if not isinstance(poses, list) or not 1 <= len(poses) <= MAX_CLEARANCE_POSES:
        raise ModelError("A clearance check requires between 1 and 361 poses.")
    groups = []
    for pair in pairs:
        if not isinstance(pair, list) or len(pair) != 2:
            raise ModelError("Each clearance pair requires two groups of solid indices.")
        if not all(isinstance(group, list) for group in pair):
            raise ModelError("Each clearance pair requires two groups of solid indices.")
        selected = [selected_solids(group, source.solids, name="Clearance groups") for group in pair]
        if set(selected[0]) & set(selected[1]):
            raise ModelError("Each clearance pair requires two nonempty, disjoint groups of solids.")
        groups.append(selected)
    checked = [Pose(records, source.solids) for records in poses]
    samples = []
    for pose in checked:
        row = []
        for first, second in groups:
            distance, a, b, depth = separation(source, first, second, pose, threads)
            item = {"distance": distance, "closestPoints": [_vector(a), _vector(b)]}
            if distance <= CONTACT_TOLERANCE:
                item["overlapVolume"] = overlap_volume(source, first, second, pose, threads, depth)
            row.append(item)
        samples.append(row)
    return {"samples": samples, "tolerance": CONTACT_TOLERANCE}


# ------------------------------------------------------------------ printing

def print_down_direction(value):
    """A pinned printing direction (the world-space outward direction that
    faces the build plate) as a unit vector."""
    if not isinstance(value, (list, tuple)) or len(value) != 3 or not all(_number(item) for item in value):
        raise ModelError("A pinned printing direction must be three finite numbers.")
    length = math.sqrt(sum(float(item) * float(item) for item in value))
    if length < 1e-9:
        raise ModelError("A pinned printing direction must be a nonzero direction.")
    return tuple(float(item) / length for item in value)


def print_rotation(direction):
    """The rotation (3x3) taking `direction` to +Z along the shortest arc;
    -Z, which has no shortest arc, turns half about the Y axis."""
    d = np.asarray(direction, dtype=np.float64)
    x, y, z = np.cross(d, (0.0, 0.0, 1.0))
    w = float(d[2])
    norm = math.sqrt(x * x + y * y + z * z + w * w)
    x, y, z, w = x / norm, y / norm, z / norm, w / norm + 1.0
    if w <= 1e-300:
        x, y, z = (0.0, d[2], -d[1]) if d[2] * d[2] > d[0] * d[0] else (d[1], -d[0], 0.0)
    norm = math.sqrt(x * x + y * y + z * z + w * w)
    x, y, z, w = x / norm, y / norm, z / norm, w / norm
    return np.array([[1 - 2 * (y * y + z * z), 2 * (x * y - w * z), 2 * (x * z + w * y)],
                     [2 * (x * y + w * z), 1 - 2 * (x * x + z * z), 2 * (y * z - w * x)],
                     [2 * (x * z - w * y), 2 * (y * z + w * x), 1 - 2 * (x * x + y * y)]])


def _hull(points):
    """Convex hull (counter-clockwise, no collinear points) of 2D tuples."""
    points = sorted(set(points))

    def cross(a, b, c):
        return (b[0] - a[0]) * (c[1] - a[1]) - (b[1] - a[1]) * (c[0] - a[0])

    lower, upper = [], []
    for chain, sequence in ((lower, points), (upper, reversed(points))):
        for point in sequence:
            while len(chain) >= 2 and cross(chain[-2], chain[-1], point) <= 0:
                chain.pop()
            chain.append(point)
    return lower[:-1] + upper[:-1]


def _plane_basis(normal):
    normal = np.asarray(normal, dtype=np.float64)
    tangent = np.cross(normal, np.eye(3)[np.argmin(np.abs(normal))])
    tangent /= np.linalg.norm(tangent)
    return tangent, np.cross(normal, tangent)


def contact_is_stable(points, direction, centroid):
    """True when the centroid projects strictly inside the convex hull of the
    bed contact points (the support polygon), viewed along `direction`."""
    points = np.asarray(points, dtype=np.float64).reshape(-1, 3)
    if len(points) < 3:
        return False
    tangent, other = _plane_basis(direction)
    relative = points - np.asarray(centroid, dtype=np.float64)
    flat = np.stack([relative @ tangent, relative @ other], axis=1)
    if len(flat) > 64:
        # Points strictly inside the octagon of extreme points cannot be on the hull.
        angles = np.arange(8) * (math.pi / 4)
        axes = np.stack([np.cos(angles), np.sin(angles)], axis=1)
        octagon = flat[np.argmax(flat @ axes.T, axis=0)]
        keep = np.zeros(len(flat), dtype=bool)
        keep[np.argmax(flat @ axes.T, axis=0)] = True
        for a, b in zip(octagon, np.roll(octagon, -1, axis=0)):
            keep |= (b[0] - a[0]) * (flat[:, 1] - a[1]) - (b[1] - a[1]) * (flat[:, 0] - a[0]) <= 0
        flat = flat[keep]
    hull = _hull(zip(flat[:, 0].tolist(), flat[:, 1].tolist()))
    if len(hull) < 3:
        return False
    tolerance = max(1e-7, float(np.ptp(relative, axis=0).max()) * 1e-9)
    return all((b[0] - a[0]) * (0 - a[1]) - (b[1] - a[1]) * (0 - a[0]) > tolerance * math.hypot(b[0] - a[0], b[1] - a[1])
               for a, b in zip(hull, hull[1:] + hull[:1]))


class PrintSurface:
    """Orientation data of a closed part mesh, translated by -shift:
    area-weighted unit triangle normals, vertices, coplanar triangle
    clusters (planar contacts) and the volume centroid and principal axes."""

    def __init__(self, mesh, shift=(0.0, 0.0, 0.0)):
        positions = mesh.positions.astype(np.float64) - np.asarray(shift, dtype=np.float64)
        triangles = mesh.triangles.astype(np.int64)
        corners = positions[triangles]
        cross = np.cross(corners[:, 1] - corners[:, 0], corners[:, 2] - corners[:, 0])
        double = np.linalg.norm(cross, axis=1)
        keep = np.isfinite(double) & (double > 0)
        if not keep.any():
            raise ModelError("A part has no surface to orient for 3D printing.")
        self.positions = positions
        self.triangles = triangles[keep]
        self.normals = cross[keep] / double[keep, None]
        self.weights = double[keep] / 2
        self.centers = corners[keep].mean(axis=1)
        # Float32 mesh coordinates are exact to about 1e-7 of their size.
        self.tolerance = max(1e-6, 1e-6 * float(np.abs(mesh.positions).max()))
        self.cell = float(mesh.stats.get("cell_size", 0.0) or 0.0)
        # Dual contouring places some vertices at sharp edges slightly off the
        # surface, so a plane resting on the bed may sit a little above the
        # lowest vertex.
        self.contact_tolerance = max(self.tolerance, PRINT_CONTACT_CELLS * self.cell)
        # The same artifacts chamfer sharp bottom edges; overhangs within one
        # sampling cell of the bed need no support and are not scored.
        self.bed_band = max(self.tolerance, PRINT_BED_BAND_CELLS * self.cell)
        self._volume_properties(corners)
        self._cluster_planes()
        self._supports = {}

    def _volume_properties(self, corners):
        origin = (self.positions.min(axis=0) + self.positions.max(axis=0)) / 2
        a, b, c = (corners[:, index] - origin for index in range(3))
        determinant = np.einsum("ij,ij->i", a, np.cross(b, c))
        volume = float(determinant.sum()) / 6
        if not volume > 0:
            raise ModelError("A part's mesh encloses no volume and cannot be oriented for 3D printing.")
        total = a + b + c
        mean = (determinant[:, None] * total).sum(axis=0) / (24 * volume)
        second = sum(np.einsum("i,ij,ik->jk", determinant, vector, vector) for vector in (a, b, c, total)) / 120
        central = second - volume * np.outer(mean, mean)
        inertia = np.trace(central) * np.eye(3) - central
        _, vectors = np.linalg.eigh((inertia + inertia.T) / 2)
        axes = vectors.T.copy()
        for axis in axes:  # deterministic signs
            if axis[np.argmax(np.abs(axis))] < 0:
                axis *= -1
        self.centroid, self.axes = origin + mean, axes

    def _cluster_planes(self):
        """Coplanar triangles (normals within a bin, offsets within the
        tolerance) as planes: normal, centre, area, triangle members."""
        keys = np.round(self.normals / PRINT_NORMAL_BIN).astype(np.int64)
        _, bins = np.unique(keys, axis=0, return_inverse=True)
        bins = bins.reshape(-1)
        sums = np.zeros((int(bins.max()) + 1, 3))
        np.add.at(sums, bins, self.normals * self.weights[:, None])
        bin_normals = sums / np.linalg.norm(sums, axis=1, keepdims=True)
        offsets = np.einsum("ij,ij->i", bin_normals[bins], self.centers)
        order = np.lexsort((offsets, bins))
        breaks = np.ones(len(order), dtype=bool)
        breaks[1:] = (bins[order][1:] != bins[order][:-1]) | (np.diff(offsets[order]) > self.tolerance)
        cluster = np.empty(len(order), dtype=np.int64)
        cluster[order] = np.cumsum(breaks) - 1
        sizes = np.bincount(cluster)
        areas = np.bincount(cluster, weights=self.weights)
        # A plane PRINT_PLANE_CELLS cells wide covers at least about 2 cells².
        planar = np.flatnonzero((sizes >= 2) & (areas >= 2.0 * self.cell * self.cell))
        areas = areas[planar]
        # Keep the largest planes; tiny ones neither support nor rank.
        planar = planar[np.argsort(-areas, kind="stable")[:4096]]
        members = [np.flatnonzero(cluster == index) for index in planar] if len(planar) <= 64 else None
        if members is None:
            rank = np.full(len(sizes), -1)
            rank[planar] = np.arange(len(planar))
            selected = np.flatnonzero(rank[cluster] >= 0)
            grouped = selected[np.argsort(rank[cluster[selected]], kind="stable")]
            members = np.split(grouped, np.cumsum(np.bincount(rank[cluster[selected]], minlength=len(planar)))[:-1])
        normals = np.array([(self.normals[rows] * self.weights[rows, None]).sum(axis=0)
                            for rows in members]).reshape(-1, 3)
        normals /= np.maximum(np.linalg.norm(normals, axis=1, keepdims=True), 1e-300)
        # Facets of curved surfaces (single quads, strips along a cylinder)
        # are planar too; a real plane is at least PRINT_PLANE_CELLS sampling
        # cells wide in every in-plane direction.
        wide = [index for index, rows in enumerate(members)
                if self._width(rows, normals[index]) >= PRINT_PLANE_CELLS * self.cell]
        members = [members[index] for index in wide]
        normals = normals[wide].reshape(-1, 3)
        self.plane_members = members
        self.plane_areas = np.array([self.weights[rows].sum() for rows in members])
        self.plane_normals = normals
        self.plane_centers = np.array([(self.centers[rows] * self.weights[rows, None]).sum(axis=0)
                                       / self.weights[rows].sum() for rows in members]).reshape(-1, 3)
        self.plane_bins = np.array([bins[rows[0]] for rows in members], dtype=np.int64)

    def _width(self, rows, normal):
        """Smallest in-plane extent of triangles (along their principal axes)."""
        points = self.positions[np.unique(self.triangles[rows])]
        tangent, other = _plane_basis(normal)
        flat = np.stack([points @ tangent, points @ other], axis=1)
        flat -= flat.mean(axis=0)
        _, axes = np.linalg.eigh(flat.T @ flat)
        return float(np.ptp(flat @ axes[:, 0]))

    def support(self, plane):
        """Hull points (3D) of a plane's triangles: its boundary vertices."""
        points = self._supports.get(plane)
        if points is None:
            rows = self.triangles[self.plane_members[plane]]
            edges = np.sort(np.concatenate([rows[:, [0, 1]], rows[:, [1, 2]], rows[:, [2, 0]]]), axis=1)
            unique, counts = np.unique(edges, axis=0, return_counts=True)
            vertices = np.unique(unique[counts == 1]) if (counts == 1).any() else np.unique(rows)
            candidates = self.positions[vertices]
            tangent, other = _plane_basis(self.plane_normals[plane])
            projected = list(zip((candidates @ tangent).tolist(), (candidates @ other).tolist()))
            hull = set(_hull(projected))
            points = self._supports[plane] = candidates[[index for index, point in enumerate(projected)
                                                         if point in hull]]
        return points


def print_candidates(surface):
    """Candidate up directions: the 48 largest plane groups facing down,
    +Z first, the principal axes of inertia, the world axes (both signs)
    and 32 Fibonacci-sphere directions, deduplicated at 1e-6."""
    grouped = {}

    def add(direction, importance=0.0):
        direction = np.asarray(direction, dtype=np.float64)
        length = np.linalg.norm(direction)
        if not np.isfinite(length) or length < 1e-12:
            return
        direction = direction / length
        key = tuple(round(float(value), 6) for value in direction)
        if key in grouped:
            grouped[key][1] += importance
        else:
            grouped[key] = [direction, importance]

    # Planes split into pieces (or parallel planes) accumulate their areas
    # in one normal bin instead of receiving extra candidates.
    by_bin = {}
    for normal, area, key in zip(surface.plane_normals, surface.plane_areas, surface.plane_bins.tolist()):
        entry = by_bin.setdefault(key, [np.zeros(3), 0.0])
        entry[0] += normal * area
        entry[1] += area
    for normal, area in by_bin.values():
        add(-normal, area)
    directions = [item[0] for item in sorted(grouped.values(), key=lambda item: -item[1])[:48]]
    grouped.clear()
    add((0.0, 0.0, 1.0))  # Keep the supplied pose first in equal-score ties.
    for direction in directions:
        add(direction)
    for axis in surface.axes:
        add(axis)
        add(-axis)
    for axis in np.eye(3):
        add(axis)
        add(-axis)
    for index in range(32):
        z = 1 - 2 * (index + 0.5) / 32
        angle = index * math.pi * (3 - math.sqrt(5))
        radius = math.sqrt(1 - z * z)
        add((radius * math.cos(angle), radius * math.sin(angle), z))
    return np.asarray([item[0] for item in grouped.values()])


def print_orientation(surface):
    """(rotation, bounds): the up direction minimizing the 45-degree overhang
    integral sum(w * max(0, -n.d - sqrt(1/2))^2) over triangles reaching more
    than a sampling cell above the bed (fully from two cells), preferring
    poses whose centroid lies
    strictly inside the support polygon: the convex hull of the planes
    resting on the bed (facing straight down within the contact tolerance of
    the lowest vertex; curved surfaces only touch the bed). Ties prefer more
    resting plane area, then a lower part, then the candidate order."""
    directions = print_candidates(surface)
    origin = surface.positions[0]
    lows, highs = np.full(len(directions), np.inf), np.full(len(directions), -np.inf)
    for start in range(0, len(surface.positions), 4096):
        heights = (surface.positions[start:start + 4096] - origin) @ directions.T
        lows = np.minimum(lows, heights.min(axis=0))
        highs = np.maximum(highs, heights.max(axis=0))
    band = lows + surface.bed_band
    costs = np.zeros(len(directions))
    for start in range(0, len(surface.weights), 2048):
        block = slice(start, start + 2048)
        top = ((surface.positions[surface.triangles[block]] - origin) @ directions.T).max(axis=1)
        penalties = np.maximum(0, -surface.normals[block] @ directions.T - PRINT_THRESHOLD)
        # Ramp in over the next band so discretization does not flip triangles in and out.
        above = np.clip((top - band) / surface.bed_band, 0.0, 1.0)
        costs += surface.weights[block] @ (penalties * penalties * above)
    facing = surface.plane_normals @ directions.T <= -PRINT_CONTACT_COSINE
    levels = (surface.plane_centers - origin) @ directions.T
    contacts = facing & (levels <= lows[None, :] + surface.contact_tolerance)
    contact_areas = surface.plane_areas @ contacts if len(surface.plane_areas) else np.zeros(len(directions))
    heights = highs - lows
    # Mesh discretization perturbs equivalent poses slightly; scores within
    # these tolerances tie, so symmetric parts keep the earlier candidate.
    # Overhang integrals of curved parts vary by about a percent between
    # equivalent poses; heights by a fraction of a sampling cell.
    area = float(surface.weights.sum())
    cost_tolerance = max(1e-10, PRINT_TIE * area * (1 - PRINT_THRESHOLD) ** 2, PRINT_COST_TIE * float(costs.min()))
    area_tolerance = max(1e-10, PRINT_TIE * area)
    height_tolerance = max(surface.contact_tolerance, PRINT_TIE * float(heights.max()))
    stable = {}

    def is_stable(index):
        if index not in stable:
            planes = np.flatnonzero(contacts[:, index])
            stable[index] = bool(len(planes)) and contact_is_stable(
                np.concatenate([surface.support(plane) for plane in planes]), directions[index], surface.centroid)
        return stable[index]

    best = None
    for require_stable_bed in (True, False):
        feasible = np.arange(len(directions))
        if require_stable_bed:
            # Stability is checked lazily, cheapest candidates first.
            ranked = sorted(feasible.tolist(), key=lambda index: (costs[index], index))
            floor = next((costs[index] for index in ranked if is_stable(index)), None)
            if floor is None:
                continue
            feasible = np.array([index for index in ranked if costs[index] <= floor + cost_tolerance
                                 and is_stable(index)])
        else:
            feasible = feasible[costs <= costs.min() + cost_tolerance]
        feasible = feasible[contact_areas[feasible] >= contact_areas[feasible].max() - area_tolerance]
        feasible = feasible[heights[feasible] <= heights[feasible].min() + height_tolerance]
        best = int(feasible.min())
        break
    rotation = print_rotation(directions[best])
    return rotation, oriented_bounds(surface, rotation)


def oriented_bounds(surface, rotation):
    """Bounds of the surface's vertices after `rotation`; a plane resting on
    the bed (facing straight down within the contact tolerance of the
    lowest vertex) defines the bottom exactly."""
    rotated = surface.positions @ rotation.T
    bounds = [*rotated.min(axis=0).tolist(), *rotated.max(axis=0).tolist()]
    if len(surface.plane_areas):
        down = rotation.T @ np.array([0.0, 0.0, 1.0])
        levels = surface.plane_centers @ down
        resting = (surface.plane_normals @ down <= -PRINT_CONTACT_COSINE) & (
            levels <= bounds[2] + surface.contact_tolerance)
        if resting.any():
            bounds[2] = float(levels[resting].max())
    return bounds


def print_orientations(source, solids, pins, threads):
    """(matrix[16], bounds[6]) per solid: the world-space rigid delta from
    the built part to its printing orientation and the oriented part's
    bounds. A part placed by a translation (its root node a TRANSFORM) is
    oriented about its own origin; instances sharing geometry, rotation,
    scale and pin share the work within this request only. A pin (the
    outward direction facing the bed) replaces the search. A quarter turn
    keeps each footprint at least as wide as deep."""
    shared, results = {}, []
    for solid, pin in zip(solids, pins):
        record = source.root_record(solid)
        shift, key = np.zeros(3), (source.roots[solid], pin)
        if record is not None and record[0] == TRANSFORM_OPCODE and len(record[1]) == 1 and len(record[2]) == 13:
            shift = np.asarray(record[2][9:12], dtype=np.float64)
            key = ("instance", record[1][0], tuple(record[2][:9]), record[2][12], pin)
        entry = shared.get(key)
        if entry is None:
            surface = PrintSurface(source.sample_mesh(solid, threads), shift)
            if pin is None:
                turn, bounds = print_orientation(surface)
            else:
                turn = print_rotation(tuple(-value for value in pin))
                bounds = oriented_bounds(surface, turn)
            if bounds[4] - bounds[1] > bounds[3] - bounds[0]:
                turn = np.array([[0.0, -1.0, 0.0], [1.0, 0.0, 0.0], [0.0, 0.0, 1.0]]) @ turn
                bounds = [-bounds[4], bounds[0], bounds[2], -bounds[1], bounds[3], bounds[5]]
            entry = shared[key] = (turn, bounds)
        turn, bounds = entry
        matrix = np.eye(4)
        matrix[:3, :3], matrix[:3, 3] = turn, -turn @ shift
        results.append(([float(value) for value in matrix.reshape(-1)], list(bounds)))
    return results


def print_orient(spec, source, output, threads):
    """Printing orientations of the requested parts, honoring pins."""
    indices = selected_solids(spec.get("solidIndices"), source.solids)
    records = spec.get("printDown", [])
    if not isinstance(records, list) or len(records) > source.solids:
        raise ModelError("Pinned printing directions must be a list of uniquely indexed solids.")
    pins = {}
    for record in records:
        if not isinstance(record, dict) or set(record) != {"solid", "direction"}:
            raise ModelError("Each pinned printing direction requires a solid index and a direction.")
        index = record["solid"]
        if not _integer(index) or not 0 <= index < source.solids or index in pins:
            raise ModelError("Pinned printing directions contain invalid or duplicate solid indices.")
        pins[index] = print_down_direction(record["direction"])
    oriented = print_orientations(source, indices, [pins.get(index) for index in indices], threads)
    return {"orientations": [{"solid": index, "matrix": matrix, "bounds": bounds}
                             for index, (matrix, bounds) in zip(indices, oriented)]}


# ------------------------------------------------------------------ export

def default_cell(source):
    """The build's default preview cell size: the design's largest extent
    (conservative part bounds) over PREVIEW_CELLS_ACROSS."""
    boxes = [source.bounds(solid) for solid in range(source.solids)]
    lo = np.min([box[0] for box in boxes], axis=0)
    hi = np.max([box[1] for box in boxes], axis=0)
    return max(float(np.max(hi - lo)) / PREVIEW_CELLS_ACROSS, PREVIEW_MINIMUM_CELL)


def budget_cell(source, solids, cell, budget, threads):
    """`cell`, coarsened so a probe mesh at EXPORT_PROBE_FACTOR times it
    predicts at most `budget` triangles for the solids (counted before
    refinement, as in preview_cell_size)."""
    probe = cell * EXPORT_PROBE_FACTOR
    triangles = 0.0
    for solid in solids:
        try:
            mesh = source.graph.mesh(source.roots[solid], cell_size=probe, crease_angle=PREVIEW_CREASE_ANGLE,
                                     threads=threads)
            triangles += unrefined_triangles(mesh) * EXPORT_PROBE_FACTOR ** 2
        except radsdf.SDFError:
            lo, hi = source.bounds(solid)
            a, b, c = (float(value) for value in hi - lo)
            triangles += 4.0 * (a * b + b * c + c * a) / (cell * cell)
    if triangles > budget:
        cell *= math.sqrt(triangles / budget) * EXPORT_MARGIN
    return cell


def export_tolerance(spec, cell):
    """The export simplification tolerance at `cell`: the request's
    "tolerance" (0 disables), else EXPORT_TOLERANCE, at most
    EXPORT_TOLERANCE_CELLS of the cell either way."""
    value = spec.get("tolerance")
    if value is None:
        value = EXPORT_TOLERANCE
    elif not _number(value) or value < 0:
        raise ModelError("The export tolerance must be a nonnegative number of millimetres (0 keeps the full mesh).")
    return min(float(value), EXPORT_TOLERANCE_CELLS * cell)


def closed_meshes(source, solids, cell, budget, threads, purpose, tolerance=None):
    """Closed meshes of the solids at `cell` within `budget` triangles in
    total, coarsening (at most MAX_EXPORT_REMESHES times) when a pass
    exceeds it. Every part is meshed and counted (kept native) before any is
    simplified within `tolerance(cell)` (0: none), so the budget bounds the
    meshing work and a pass that exceeds it simplifies nothing. Returns
    (meshes, cell)."""
    for attempt in range(MAX_EXPORT_REMESHES + 1):
        natives, triangles = [], 0
        try:
            for solid in solids:
                natives.append(source.native_mesh(solid, cell, threads))
                stats = natives[-1].stats
                if not stats["triangle_count"] or stats["boundary_edges"]:
                    raise ModelError(f"{source.label(solid)} has no closed surface at the {purpose} cell size of "
                                     f"{cell:.4g} mm; choose a finer cell size or thicken its features.")
                triangles += stats["triangle_count"]
                if triangles > budget:
                    break
            else:
                simplify = tolerance(cell) if tolerance else 0.0
                return [source.copied(solid, native, cell, simplify or None)
                        for solid, native in zip(solids, natives)], cell
        finally:
            for native in natives:
                native.close()
        if attempt == MAX_EXPORT_REMESHES:
            break
        cell *= math.sqrt(triangles / budget) * EXPORT_MARGIN * (len(solids) / len(natives)) ** 0.5
    raise ModelError(f"The {purpose} exceeds {budget:,} triangles even at a cell size of {cell:.4g} mm.")


def write_stl(path, meshes, solids, pose):
    """Binary STL (millimetres) of posed meshes as separate shells, written
    to a temporary name and renamed. Returns its triangle count."""
    count = sum(len(mesh.triangles) for mesh in meshes)
    staged = path.with_name(path.name + ".tmp")
    try:
        with staged.open("wb") as stream:
            stream.write(STL_HEADER)
            stream.write(struct.pack("<I", count))
            for mesh, solid in zip(meshes, solids):
                positions = pose.apply(solid, mesh.positions.astype(np.float64))
                for start in range(0, len(mesh.triangles), 262_144):
                    corners = positions[mesh.triangles[start:start + 262_144].astype(np.int64)]
                    normals = np.cross(corners[:, 1] - corners[:, 0], corners[:, 2] - corners[:, 0])
                    lengths = np.linalg.norm(normals, axis=1, keepdims=True)
                    records = np.zeros(len(corners), dtype=STL_RECORD)
                    records["normal"] = np.divide(normals, lengths, out=np.zeros_like(normals), where=lengths > 0)
                    records["vertices"] = corners
                    stream.write(records.tobytes())
        staged.replace(path)
    finally:
        staged.unlink(missing_ok=True)
    return count


def export(spec, source, output, threads):
    """Binary STL of the selected parts at their poses, re-meshed at export
    resolution and simplified within the export tolerance: one model.stl, or
    part-%04d.stl per individualParts group."""
    indices = selected_solids(spec.get("solidIndices"), source.solids)
    parts = spec.get("individualParts")
    if "individualParts" in spec:
        if not isinstance(parts, list) or not 1 <= len(parts) <= len(indices):
            raise ModelError("Individual STL files require a nonempty list of available parts.")
        used, selected = set(), set(indices)
        for part in parts:
            if (not isinstance(part, list) or not part or any(not _integer(index) or index not in selected
                                                              for index in part)
                    or len(set(part)) != len(part) or used.intersection(part)):
                raise ModelError("Individual STL files must contain distinct selected parts.")
            used.update(part)
        if used != selected:
            raise ModelError("Individual STL files must include every selected part exactly once.")
    pose = Pose(spec.get("solidTransforms", []), source.solids)
    cell = positive_cell(spec, "export")
    if cell is None:
        cell = min(max(default_cell(source) / 2, EXPORT_MINIMUM_CELL), EXPORT_MAXIMUM_CELL)
    export_tolerance(spec, 1.0)  # refuses a malformed tolerance before meshing
    cell = budget_cell(source, indices, cell, EXPORT_TRIANGLE_BUDGET, threads)
    meshes, cell = closed_meshes(source, indices, cell, EXPORT_TRIANGLE_BUDGET, threads, "export",
                                 tolerance=lambda size: export_tolerance(spec, size))
    by_solid = dict(zip(indices, meshes))
    report = {"solids": len(indices), "triangles": sum(len(mesh.triangles) for mesh in meshes), "cellSize": cell,
              "tolerance": export_tolerance(spec, cell),
              "meshedTriangles": sum(int(mesh.stats.get("unsimplified_triangles", len(mesh.triangles)))
                                     for mesh in meshes)}
    if parts is None:
        write_stl(output / "model.stl", meshes, indices, pose)
        return report
    files = []
    for position, part in enumerate(parts):
        name = f"part-{position:04d}.stl"
        write_stl(output / name, [by_solid[index] for index in part], part, pose)
        files.append({"path": name, "solidIndices": part})
    report["partFiles"] = files
    return report


# ------------------------------------------------------------------ validate

class _Labels:
    """The part labels sdf_build's preview meshing names in messages."""
    __slots__ = ("labels",)

    def __init__(self, labels):
        self.labels = labels


def validate(spec, source, output, threads):
    """Re-mesh every part finer than its preview (half the preview cell size
    unless cellSize is given) and report open or non-manifold surfaces,
    parts that split or lose pieces, sealed cavities that open or close, and
    volume changes above 2%."""
    requested = positive_cell(spec, "validation")
    boxes = [source.bounds(solid) for solid in range(source.solids)]
    # The preview is deterministic: replaying its meshing reproduces the
    # build's cell size and part meshes exactly.
    previews, preview_cell, _ = mesh_parts(source.graph, _Labels([source.label(solid)
                                                                 for solid in range(source.solids)]),
                                           boxes, threads)
    cell = preview_cell / 2 if requested is None else requested
    warnings = []
    estimate = sum(len(mesh.triangles) for mesh in previews) * (preview_cell / cell) ** 2
    if estimate > VALIDATE_TRIANGLE_BUDGET:
        coarser = cell * math.sqrt(estimate / VALIDATE_TRIANGLE_BUDGET) * EXPORT_MARGIN
        warnings.append(f"Validation uses a cell size of {coarser:.4g} mm instead of {cell:.4g} mm to stay "
                        f"within {VALIDATE_TRIANGLE_BUDGET:,} triangles.")
        cell = coarser
    parts, ok = [], True
    for solid, preview in enumerate(previews):
        label = source.label(solid)
        mesh = source.mesh(solid, cell, threads)
        stats = mesh.stats
        triangles = len(mesh.triangles)
        boundary = int(stats.get("boundary_edges", 0))
        nonmanifold = int(stats.get("nonmanifold_edges", 0))
        components = mesh_pieces(stats) if triangles else 0  # sealed cavities are not pieces
        cavities = int(stats.get("cavities", 0)) if triangles else 0
        before = mesh_pieces(preview.stats)
        preview_cavities = int(preview.stats.get("cavities", 0))
        volume = float(stats.get("volume", 0.0)) if triangles else 0.0
        preview_volume = float(preview.stats["volume"])
        change = (volume - preview_volume) / preview_volume
        problems = []
        if not triangles:
            problems.append(f"{label} vanishes at {cell:.4g} mm: it is thinner than the preview suggests.")
        else:
            if boundary:
                problems.append(f"{label} is open at {cell:.4g} mm ({boundary} boundary edges).")
            if nonmanifold:
                problems.append(f"{label} has {nonmanifold} non-manifold edges at {cell:.4g} mm: features "
                                f"thinner than about {2 * cell:.3g} mm touch or merge.")
            if components > before:
                problems.append(f"{label} splits into {components} pieces at {cell:.4g} mm (the preview shows "
                                f"{before}): connections thinner than the preview cell size of "
                                f"{preview_cell:.4g} mm are not solid.")
            elif components < before:
                problems.append(f"{label} has {components} pieces at {cell:.4g} mm where the preview shows "
                                f"{before}: some preview pieces vanish or merge at a finer resolution.")
            if cavities != preview_cavities:
                problems.append(f"{label} has {cavities} sealed cavities at {cell:.4g} mm where the preview shows "
                                f"{preview_cavities}: a void, or a wall or opening around one, is thinner than the "
                                f"preview cell size of {preview_cell:.4g} mm.")
            if abs(change) > VALIDATE_VOLUME_CHANGE:
                problems.append(f"{label} changes volume by {change:+.1%} at {cell:.4g} mm ({preview_volume:.6g} "
                                f"to {volume:.6g} mm³): features thinner than the preview cell size of "
                                f"{preview_cell:.4g} mm.")
        ok = ok and not problems
        warnings.extend(problems)
        parts.append({"solid": solid, "closed": bool(triangles) and not boundary, "boundaryEdges": boundary,
                      "nonmanifoldEdges": nonmanifold, "components": components, "cavities": cavities,
                      "previewCavities": preview_cavities, "volume": volume,
                      "previewVolume": preview_volume, "volumeChange": change})
    if len(warnings) > MAX_WARNINGS:
        warnings[MAX_WARNINGS - 1:] = [f"... and {len(warnings) - MAX_WARNINGS + 1} more warnings."]
    return {"ok": ok, "cellSize": cell, "parts": parts, "warnings": warnings}


# ------------------------------------------------------------------ probe

def probe(spec, source, output, threads):
    """Signed distance and unit normal of each requested part at world
    points of the built design; with directions, sphere-traced ray hits."""
    solids = selected_solids(spec.get("solids"), source.solids, name="The probed parts")
    points = spec.get("points")
    if (not isinstance(points, list) or not 1 <= len(points) <= MAX_PROBE_POINTS
            or not all(_point(point, 1e9) for point in points)):
        raise ModelError("Probe 1 to 65,536 points, each three finite coordinates.")
    if len(points) * len(solids) > MAX_PROBE_SAMPLES:
        raise ModelError(f"Probe at most {MAX_PROBE_SAMPLES:,} point and part combinations at once.")
    directions = spec.get("directions")
    if directions is not None and (not isinstance(directions, list) or len(directions) != len(points)
                                   or not all(_point(direction, 1e9) and any(direction) for direction in directions)):
        raise ModelError("Probe directions must give one nonzero finite direction per point.")
    origins = np.asarray(points, dtype=np.float64)
    maximum = spec.get("maxDistance")
    if maximum is not None and (not _number(maximum) or maximum <= 0):
        raise ModelError("The probe's maxDistance must be a positive number of millimetres.")
    if maximum is None and directions is not None:
        boxes = [source.bounds(solid) for solid in solids]
        lo = np.min([box[0] for box in boxes], axis=0)
        hi = np.max([box[1] for box in boxes], axis=0)
        far = np.maximum(np.abs(origins - lo), np.abs(origins - hi))
        maximum = max(1.0, 2.0 * float(np.linalg.norm(far, axis=1).max()))
    unit = None
    if directions is not None:
        unit = np.asarray(directions, dtype=np.float64)
        unit /= np.linalg.norm(unit, axis=1, keepdims=True)
    columns = []
    for solid in solids:
        node = source.roots[solid]
        try:
            distances = source.graph.evaluate(node, origins, threads=threads)
            normals = source.graph.gradient(node, origins, threads=threads)
            hits = traced = None
            if unit is not None:
                hits, traced = source.graph.raycast(node, origins, unit, max_distance=float(maximum),
                                                    tolerance=PROBE_TOLERANCE, max_steps=PROBE_MAX_STEPS,
                                                    threads=threads)
        except radsdf.SDFError as error:
            raise ModelError(f"{source.label(solid)} could not be probed: {error}") from error
        columns.append((distances, normals, hits, traced))
    samples = []
    for row in range(len(origins)):
        values = []
        for distances, normals, hits, traced in columns:
            sample = {"distance": float(distances[row]), "normal": _vector(normals[row])}
            if hits is not None:
                hit, length = bool(hits[row]), min(float(traced[row]), float(maximum))
                sample.update(hit=hit, hitDistance=length)
                if hit:
                    sample["hitPoint"] = _vector(origins[row] + unit[row] * length)
            values.append(sample)
        samples.append(values)
    return {"solids": solids, "samples": samples}


HANDLERS = {"measure": measure, "clearance": clearance, "print_orient": print_orient, "export": export,
            "validate": validate, "probe": probe}
