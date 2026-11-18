"""RAD's SDF build pipeline.

A build runs the design program, normalizes its `result` into physical parts
(one solid index per part), serializes every part's placed shape into one
RSDF graph (model.rsdf), meshes each part with the native kernel at one
preview cell size and publishes mesh.bin and result.json (SDF-DESIGN.md,
"Worker <-> app protocol"). The worker owns process framing and error
reporting; this module owns everything between the request and the outputs.

numpy, radsdf and radsdf_assembly are imported where they are used, so a
missing runtime library is reported as a runtime failure in error.json.
"""
from __future__ import annotations

import collections.abc
import hashlib
import json
import math
from pathlib import Path
import platform
import re
import time
import uuid

from assembly_motion import AssemblyMotionError, has_control_characters, validate_motion

MAX_TRIANGLES = 1_000_000
MAX_VERTICES = 1_000_000
MAX_EDGE_SEGMENTS = 250_000
MAX_SCRIPT_BYTES = 200_000
MAX_GEOMETRY_BYTES = 512_000_000
MAX_PARTS = 4096
MAX_PART_NODES = 8192
MAX_DEPTH = 32
COORDINATE_LIMIT = 1_000_000.0
MAX_WARNINGS = 24

# Preview meshing (one cell size per build). The validation receipt's rules
# tag names every value here, so changing one invalidates old receipts.
PREVIEW_CELLS_ACROSS = 256
PREVIEW_TRIANGLE_BUDGET = MAX_TRIANGLES
PREVIEW_CREASE_ANGLE = 35.0
PREVIEW_MINIMUM_CELL = 1e-4
MAX_PREVIEW_REMESHES = 2
PREVIEW_MARGIN = 1.05
PREVIEW_PROBE_FACTOR = 4

ROOT_KEYS = {"title", "summary", "units", "script"}
FIELD_TO_SOLID = ("a field becomes a solid only through implicit(field, bounds), for example "
                  "implicit(g, bounds=((-20, -20, -20), (20, 20, 20))), inside where the field is negative.")
# Custom fields: the build summary analyses at most this many implicit()
# shapes again for their slope bound and quality; rad_verify reports more.
MAX_SUMMARY_FIELDS = 32
MAX_VERIFIED_FIELDS = 64
IMPLICIT_FAILURE = re.compile(r"node (\d+) \(IMPLICIT\): its field has no finite slope bound over its "
                              r"bounds: (.*)", re.S)
FIELD_ADVICE = ("Every operation needs a finite slope inside the bounds: keep divisors and the arguments of "
                "sqrt and log away from 0 (and sqrt and log from negative values), keep atan2's pole out of "
                "the bounds, or shrink the bounds. rad_verify reports every custom field.")
INPUT_FORMATS = {"stl", "obj"}
FOLDER_PREFIXES = ("user-folder:", "folder:")
VALIDATION_RULES_VERSION = 1


class ModelError(ValueError):
    """An actionable, user-visible modeling error."""

    def __init__(self, message, *, diagnostic=None):
        super().__init__(message)
        self.diagnostic = diagnostic


class HostError(RuntimeError):
    """The app's request or the CAD runtime is at fault, not the design."""


# ------------------------------------------------------------------ requests

def validate_spec(spec):
    """Validate a build request before running any program."""
    if (not isinstance(spec, dict) or not ROOT_KEYS <= set(spec)
            or set(spec) - ROOT_KEYS - {"inputAssets", "print", "verifyFields"}):
        raise ModelError("Design must contain title, summary, units and a complete Python script.")
    if "print" in spec and spec["print"] is not True:
        raise ModelError("The 3D printing build option must be true when present.")
    if "verifyFields" in spec and spec["verifyFields"] is not True:
        raise ModelError("The custom field verification option must be true when present.")
    for name, maximum in (("title", 160), ("summary", 6000)):
        if not isinstance(spec[name], str) or not 1 <= len(spec[name]) <= maximum:
            raise ModelError(f"{name} must be a nonempty string of at most {maximum} characters.")
    if spec["units"] != "mm":
        raise ModelError("Only millimeters (units: mm) are supported.")
    script = spec["script"]
    if not isinstance(script, str) or not script.strip() or len(script.encode("utf-8")) > MAX_SCRIPT_BYTES:
        raise ModelError(f"Python script must be nonempty text of at most {MAX_SCRIPT_BYTES:,} UTF-8 bytes.")
    assets = spec.get("inputAssets", [])
    if not isinstance(assets, list) or len(assets) > 256:
        raise ModelError("Imported geometry contains too many input assets.")
    seen = set()
    for index, asset in enumerate(assets):
        if (not isinstance(asset, dict) or set(asset) != {"id", "name", "format", "path"}
                or not isinstance(asset["id"], str) or not 1 <= len(asset["id"]) <= 1024
                or any(ord(char) < 32 for char in asset["id"]) or asset["id"] in seen
                or not isinstance(asset["name"], str) or not 1 <= len(asset["name"].encode("utf-8")) <= 256
                or not isinstance(asset["format"], str) or asset["format"] not in INPUT_FORMATS
                or asset["path"] != f"input-{index:03d}.{asset['format']}"):
            raise ModelError("Imported geometry requires uniquely identified private STL or OBJ input assets.")
        seen.add(asset["id"])
    return spec


def runtime_info():
    """Versions that define a build's behaviour: Python, numpy and the kernel."""
    import numpy
    import radsdf
    try:
        kernel = radsdf.kernel_version()
    except radsdf.SDFError as error:
        raise HostError(str(error)) from error
    return {"python": platform.python_version(), "numpy": numpy.__version__, "kernel": kernel}


def validation_rules(kernel):
    """The receipt's rules tag: the rules version, the kernel and the mesher
    parameters (the crossing refinement is the one radsdf meshes with)."""
    import radsdf
    return (f"rad-sdf-validation-rules-{VALIDATION_RULES_VERSION};kernel={kernel};"
            f"cells={PREVIEW_CELLS_ACROSS};triangles={PREVIEW_TRIANGLE_BUDGET};vertices={MAX_VERTICES};"
            f"segments={MAX_EDGE_SEGMENTS};crease={PREVIEW_CREASE_ANGLE:g};refine={radsdf._REFINE_STEPS};"
            f"minimum={PREVIEW_MINIMUM_CELL:g};remeshes={MAX_PREVIEW_REMESHES};margin={PREVIEW_MARGIN:g};"
            f"probe={PREVIEW_PROBE_FACTOR}")


def validation_hash(spec, directory, runtime=None):
    """Bind a build to its exact replay inputs, runtime versions and mesher rules.

    This is an accidental-change detector, not an authenticity signature.
    Display metadata (title, summary, asset names) is not part of it.
    """
    import stat
    runtime = runtime_info() if runtime is None else runtime
    digest = hashlib.sha256()

    def add(data):
        digest.update(len(data).to_bytes(8, "big"))
        digest.update(data)
    add(validation_rules(runtime["kernel"]).encode())
    add(json.dumps(runtime, sort_keys=True, separators=(",", ":")).encode())
    construction = {"units": spec["units"], "script": spec["script"],
                    "inputAssets": [{"id": asset["id"], "format": asset["format"]}
                                    for asset in spec.get("inputAssets", [])]}
    # The printing version is a different construction of the same program.
    if spec.get("print"):
        construction["print"] = True
    add(json.dumps(construction, sort_keys=True, separators=(",", ":")).encode())
    for asset in spec.get("inputAssets", []):
        path = Path(directory) / asset["path"]
        try:
            info = path.lstat()
        except FileNotFoundError:
            raise ModelError(f"The imported input {asset['name']!r} is missing.") from None
        if not stat.S_ISREG(info.st_mode) or info.st_nlink != 1 or not 0 < info.st_size <= MAX_GEOMETRY_BYTES:
            raise ModelError("Imported geometry must be a regular private file of at most 512 MB.")
        add(path.read_bytes())
    return digest.hexdigest()


# ------------------------------------------------------------------ programs

class RADBuildMode:
    """The program's `rad` value. `rad.print` is true only while building the
    design's 3D printing version. A program that never reads it has no
    distinct printing version, so its ordinary geometry serves both."""
    __slots__ = ("_printing", "printing_read")

    def __init__(self, printing):
        self._printing = bool(printing)
        self.printing_read = False

    @property
    def print(self):
        self.printing_read = True
        return self._printing

    def __repr__(self):
        return "rad"


class ImportedAssetError(ModelError):
    """An input asset that cannot become a solid; the diagnostic names it."""

    def __init__(self, asset, detail):
        name = asset["name"]
        super().__init__(f"The imported {asset['format'].upper()} input {name!r} cannot be used as a solid: {detail}",
                         diagnostic={"stage": "input_import", "geometryScope": "input_asset",
                                     "inputAsset": {"id": asset["id"], "name": name, "format": asset["format"]}})


class ImportedAssets(collections.abc.Mapping):
    """The program's `imported_assets`: the build's STL and OBJ inputs by asset
    ID, each read on first use as a radsdf mesh solid (SDF-DESIGN.md, MESH).
    Inputs with identical bytes and format share one solid within a build."""

    def __init__(self, assets=(), directory=None):
        self._records = {asset["id"]: asset for asset in assets}
        self._directory = None if directory is None else Path(directory)
        self._solids = {}
        self._by_content = {}

    def __getitem__(self, key):
        solid = self._solids.get(key) if isinstance(key, str) else None
        if solid is not None:
            return solid
        record = self._records.get(key) if isinstance(key, str) else None
        if record is None:
            known = ", ".join(map(repr, self._records)) or "none"
            raise KeyError(f"imported asset {key!r} is not available in this build (its inputs: {known}); "
                           "list it in rad_build assetIDs and in the final response's assetIDs")
        solid = self._solids[key] = self._load(record)
        return solid

    def __contains__(self, key):
        """Membership without reading the mesh."""
        return isinstance(key, str) and key in self._records

    def __iter__(self):
        return iter(self._records)

    def __len__(self):
        return len(self._records)

    def __repr__(self):
        return f"imported_assets({list(self._records)!r})"

    def _load(self, record):
        import radsdf
        data = (self._directory / record["path"]).read_bytes()
        content = (hashlib.sha256(data).hexdigest(), record["format"])
        solid = self._by_content.get(content)
        if solid is None:
            reader = radsdf.load_stl if record["format"] == "stl" else radsdf.load_obj
            try:
                solid = radsdf.mesh_solid(*reader(data), weld=False)
            except radsdf.SDFError as error:
                detail = re.sub(r"^(?:mesh_solid|load_stl|load_obj)(?::\s*|\s+(?=line))", "", str(error))
                raise ImportedAssetError(record, detail) from None
            self._by_content[content] = solid
        return solid


def program_namespace(build_mode, assets=None):
    """The names a design program starts with (SDF-DESIGN.md, Program API)."""
    import numpy as np
    import radsdf
    import radsdf_assembly
    namespace = {}
    for module in (radsdf, radsdf_assembly):
        names = getattr(module, "__all__", None)
        if names is None:
            names = [name for name in vars(module) if not name.startswith("_")]
        namespace.update((name, getattr(module, name)) for name in names)
    namespace.update(sdf=radsdf, math=math, np=np, rad=build_mode,
                     imported_assets=ImportedAssets() if assets is None else assets, __name__="__rad_model__")
    return namespace


def limit_file_size():
    """Bound individual files (including stderr) before generated Python runs.
    The hard limit cannot be raised again by the program's process."""
    import resource
    _, hard = resource.getrlimit(resource.RLIMIT_FSIZE)
    ceiling = MAX_GEOMETRY_BYTES
    if hard != resource.RLIM_INFINITY:
        ceiling = min(ceiling, hard)
    resource.setrlimit(resource.RLIMIT_FSIZE, (ceiling, ceiling))


def run_program(script, printing=False, assets=None):
    """Execute a design program; return its `result` and its RADBuildMode.
    `assets` is the program's ImportedAssets (default: none).

    This is intentionally not a Python-language security boundary. The app
    runs builds in a macOS process sandbox restricting files, subprocesses
    and networking; CLI callers must isolate untrusted programs themselves.
    """
    limit_file_size()
    build_mode = RADBuildMode(printing)
    namespace = program_namespace(build_mode, assets)
    try:
        exec(compile(script, "<RAD model>", "exec"), namespace, namespace)
    except ImportedAssetError:
        raise  # an input file's problem, named with its diagnostic
    except BaseException as error:
        # Keep line numbers useful to the agent's repair loop, including
        # SystemExit, which must never end the worker.
        line = getattr(error, "lineno", None) if isinstance(error, SyntaxError) else None
        trace = error.__traceback__
        while trace is not None:
            if trace.tb_frame.f_code.co_filename == "<RAD model>":
                line = trace.tb_lineno
            trace = trace.tb_next
        location = f" at line {line}" if line else ""
        raise ModelError(f"Python design{location}: {type(error).__name__}: {error}") from error
    if "result" not in namespace:
        raise ModelError("Python design must assign its final shape, list of shapes or Assembly to result.")
    return namespace["result"], build_mode


# ------------------------------------------------------------------ results

def validate_part_color(color):
    """Validate the opaque sRGB metadata shared by previews and saved parts."""
    if (not isinstance(color, dict) or set(color) != {"red", "green", "blue"}
            or any(isinstance(value, bool) or not isinstance(value, (int, float))
                   or not 0 <= value <= 1 or not math.isfinite(value) for value in color.values())):
        raise ModelError("Part colors must contain finite sRGB red, green and blue values between 0 and 1.")
    return color


def validate_parts(parts, solid_count):
    """Check semantic ownership independently of disposable mesh data."""
    if not isinstance(parts, list) or len(parts) > MAX_PART_NODES:
        raise ModelError("The saved parts hierarchy is invalid or too large.")
    pending = [(part, 0) for part in parts]
    identifiers, owners, count = set(), set(), 0
    while pending:
        part, depth = pending.pop()
        count += 1
        if (not isinstance(part, dict) or not {"id", "name", "solidIndices"} <= set(part)
                or set(part) - {"id", "name", "solidIndices", "children", "color"}
                or count > MAX_PART_NODES or depth > MAX_DEPTH):
            raise ModelError("The saved parts hierarchy contains invalid items.")
        identity, name = part["id"], part["name"]
        if (not isinstance(identity, str) or not identity or len(identity.encode("utf-8")) > 1024
                or has_control_characters(identity) or identity in identifiers or not isinstance(name, str)
                or not name.strip() or len(name) > 160 or has_control_characters(name)):
            raise ModelError("The saved parts hierarchy contains invalid or duplicate identities.")
        identifiers.add(identity)
        if part.get("color") is not None:
            validate_part_color(part["color"])
        indices, children = part["solidIndices"], part.get("children")
        if (not isinstance(indices, list) or (children is not None and not isinstance(children, list))
                or (children is None and not indices)):
            raise ModelError("The saved parts hierarchy contains invalid ownership.")
        for index in indices:
            if isinstance(index, bool) or not isinstance(index, int) or not 0 <= index < solid_count or index in owners:
                raise ModelError("Each saved solid must belong to exactly one part.")
            owners.add(index)
        pending.extend((child, depth + 1) for child in children or [])
        if count + len(pending) > MAX_PART_NODES:
            raise ModelError("The saved parts hierarchy is too large.")
    if owners != set(range(solid_count)):
        raise ModelError("The saved parts hierarchy must include every solid.")
    return parts


class Normalized:
    """A program result as physical parts: `shapes[i]` is solid i placed in
    world coordinates and `labels[i]` names it in messages. `reference` is
    the sorted solid indices of reference parts, which are not printed."""
    __slots__ = ("shapes", "labels", "parts", "print_down", "reference", "motion")

    def __init__(self, shapes, labels, parts, print_down=None, reference=None, motion=None):
        self.shapes, self.labels, self.parts = shapes, labels, parts
        self.print_down = print_down or {}
        self.reference = reference or []
        self.motion = motion


def normalize_result(result):
    """Map a program's `result` to physical parts and the PartNode tree.

    A single 3D shape is one part (solid/0, "Part 1"); a list or tuple of 3D
    shapes is one part per shape (solid/<i>, "Part <i+1>"); an Assembly keeps
    its hierarchy, ids, names, colors, print-down directions, reference parts
    and motion.
    """
    from radsdf import SDF2, SDF3, Field
    from radsdf_assembly import Assembly, Part
    if isinstance(result, Assembly):
        return _assembly_result(result)
    if isinstance(result, Field):
        raise ModelError(f"Python design result is a custom field; {FIELD_TO_SOLID}")
    if isinstance(result, SDF3):
        shapes, labels = [result], ["The design"]
    elif isinstance(result, (list, tuple)):
        if not result:
            raise ModelError("Python design result is an empty list; it needs at least one 3D shape.")
        shapes = list(result)
        labels = [f"Part {index + 1} (result[{index}])" for index in range(len(shapes))]
        for index, shape in enumerate(shapes):
            if isinstance(shape, SDF2):
                raise ModelError(f"result[{index}] is a 2D profile; extrude or revolve it into a 3D solid.")
            if isinstance(shape, Field):
                raise ModelError(f"result[{index}] is a custom field; {FIELD_TO_SOLID}")
            if not isinstance(shape, SDF3):
                kind = ("an Assembly; use one Assembly with folders for named groups"
                        if isinstance(shape, Assembly) else f"a {type(shape).__name__}, not a 3D shape")
                raise ModelError(f"result[{index}] is {kind}.")
    elif isinstance(result, SDF2):
        raise ModelError("Python design result is a 2D profile; extrude or revolve it into a 3D solid.")
    elif isinstance(result, Part):
        raise ModelError("Python design result is a Part; assign the Assembly that contains it to result.")
    else:
        raise ModelError("Python design result must be a 3D shape, a list of 3D shapes or an Assembly, "
                         f"not {type(result).__name__}.")
    if len(shapes) > MAX_PARTS:
        raise ModelError(f"The design has {len(shapes)} parts; at most {MAX_PARTS} are supported.")
    parts = [{"id": f"solid/{index}", "name": f"Part {index + 1}", "solidIndices": [index], "children": None}
             for index in range(len(shapes))]
    return Normalized(shapes, labels, parts)


def _assembly_result(root):
    """Normalize an Assembly with the established id, name, folder, body,
    color and print-down rules; reference flags inherit like colors, and
    reference parts, which are never printed, get no print-down direction.
    Solid indices follow parent-before-child insertion order; placements
    compose from the root down."""
    import numpy as np
    import radsdf
    from radsdf_assembly import Assembly, Motion, Part
    seen, name_counts, explicit_ids, pending, nodes = set(), {}, set(), [(root, 0)], 0
    while pending:
        node, depth = pending.pop()
        if id(node) in seen:
            raise ModelError("An assembly or part appears twice in the result; create a new one for each "
                             "instance.")
        seen.add(id(node))
        nodes += 1
        if depth > MAX_DEPTH or nodes > MAX_PART_NODES:
            raise ModelError(f"The assembly is too large: at most {MAX_PART_NODES} parts and folders, "
                             f"nested at most {MAX_DEPTH} levels deep.")
        if node is not root and node.name:
            name_counts[node.name] = name_counts.get(node.name, 0) + 1
        if node.id is not None:
            if node.id in explicit_ids:
                raise ModelError(f"Assembly ids must be unique: {node.id!r} is used more than once.")
            explicit_ids.add(node.id)
        if isinstance(node, Assembly):
            motion = node._motion  # None until the program uses assembly.motion
            if node is not root and motion is not None and not (isinstance(motion, Motion) and motion.empty):
                raise ModelError(f"Motion is read only from the result assembly; define the joints of "
                                 f"{node.name or node.id or 'a sub-assembly'!r} on the root assembly.")
            pending.extend((child, depth + 1) for child in reversed(node.children))
    generated_ids = set(explicit_ids)

    def generated_id(candidate):
        # Explicit ids are used verbatim; generated ones must not collide with them.
        while candidate in generated_ids:
            candidate = f"generated:{candidate}"
        generated_ids.add(candidate)
        return candidate

    shapes, labels, print_down, reference, resolved = [], [], {}, [], {}

    def visit(node, matrix, path, color, down, unprinted):
        if node.at is not None:
            matrix = node.at.matrix if matrix is None else matrix @ node.at.matrix
        color = node.color if node.color is not None else color
        down = node.print_down if node.print_down is not None else down
        unprinted = node.reference if node.reference is not None else unprinted
        name = node.name or ""
        if node.id is not None:
            identity = node.id
        elif node is root:
            identity = generated_id("root")
        elif name and name_counts[name] == 1:
            identity = generated_id(f"name:{name}")
        else:
            identity = generated_id("path:" + json.dumps(path, ensure_ascii=False, separators=(",", ":")))
        if isinstance(node, Part):
            index = len(shapes)
            label = f'Part "{name}" (id {identity})' if name else f"Part {index + 1} (id {identity})"
            try:
                shape = node.shape if matrix is None else node.shape.transform(matrix)
            except radsdf.SDFError as error:
                raise ModelError(f"{label} cannot be placed: {error}") from error
            shapes.append(shape)
            labels.append(label)
            resolved[id(node)] = resolved[identity] = (identity, np.eye(4) if matrix is None else matrix)
            if unprinted:
                reference.append(index)
            elif down is not None:
                print_down[index] = list(down)
            leaf = {"id": identity, "name": name or f"Part {index + 1}", "solidIndices": [index], "children": None}
            if color is not None:
                leaf["color"] = {"red": color[0], "green": color[1], "blue": color[2]}
            if not identity.startswith(FOLDER_PREFIXES):
                return leaf
            # Folder identities stay folders; the physical part becomes their body.
            body = dict(leaf, id=generated_id("body:" + json.dumps([identity, 0], ensure_ascii=False,
                                                                     separators=(",", ":"))),
                        name=name or "Body")
            resolved[id(node)] = resolved[identity] = (body["id"], resolved[identity][1])
            return {"id": identity, "name": name or "Parts", "solidIndices": [], "children": [body]}
        children = []
        for index, child in enumerate(node.children):
            part = visit(child, matrix, path + [child.name or f"@{index}"], color, down, unprinted)
            if part is not None:
                children.append(part)
        if not children:
            return None
        return {"id": identity, "name": name or "Parts", "solidIndices": [], "children": children}

    tree = visit(root, None, [], None, None, False)
    if tree is None:
        raise ModelError("The result assembly contains no parts; add 3D shapes with add().")
    if len(shapes) > MAX_PARTS:
        raise ModelError(f"The design has {len(shapes)} parts; at most {MAX_PARTS} are supported.")
    # An unnamed root without an id is only a container for its children.
    parts = tree["children"] if not root.name and root.id is None else [tree]
    validate_parts(parts, len(shapes))
    motion = _assembly_motion(root, parts, resolved)
    return Normalized(shapes, labels, parts, print_down, reference, motion)


def _assembly_motion(root, parts, resolved):
    import radsdf
    from radsdf_assembly import Motion, Part
    definition = root._motion  # None until the program uses assembly.motion
    if definition is None or (isinstance(definition, Motion) and definition.empty):
        return None
    if isinstance(definition, Motion):
        def resolve(part):
            if isinstance(part, Part):
                if id(part) not in resolved:
                    raise ModelError(f"Motion references part {part.name or part.id or 'unnamed'!r}, "
                                     "which is not in the result assembly.")
                return resolved[id(part)]
            return resolved.get(part, (part, None))
        try:
            definition = definition._build(resolve)
        except radsdf.SDFError as error:
            raise ModelError(str(error)) from error
    try:
        return validate_motion(definition, parts)
    except AssemblyMotionError as error:
        raise ModelError(str(error)) from error


# ------------------------------------------------------------------ geometry

def part_bounds(graph, normalized):
    """Finite, nonempty, in-range conservative bounds (lo, hi) of every part
    (graph.roots holds the parts' nodes in solid order)."""
    import numpy as np
    import radsdf
    boxes = []
    stage = {"stage": "part_bounds"}
    for node, label in zip(graph.roots, normalized.labels):
        try:
            bounds = graph.bounds(node)
        except radsdf.SDFError as error:
            raise ModelError(f"{label} could not be evaluated: {error}", diagnostic=stage) from error
        if np.any(bounds.lo > bounds.hi):
            raise ModelError(f"{label} is empty: its shape encloses no volume, for example an intersection "
                             "of shapes that do not overlap.", diagnostic=stage)
        if not bounds.finite:
            raise ModelError(f"{label} is unbounded: it extends to infinity, for example a half_space or "
                             "half_plane that nothing bounded intersects. Intersect it with a box around "
                             "the part.", diagnostic=stage)
        if np.any(np.abs(bounds.lo) > COORDINATE_LIMIT) or np.any(np.abs(bounds.hi) > COORDINATE_LIMIT):
            raise ModelError(f"{label} exceeds the size limit: coordinates must stay within "
                             f"±{COORDINATE_LIMIT:,.0f} mm.", diagnostic=stage)
        boxes.append((bounds.lo.copy(), bounds.hi.copy()))
    return boxes


def _triangle_estimate(box, cell):
    """Expected preview triangles of a part filling its box: two per crossed
    grid edge, which is exact for boxes and an upper bound for convex parts."""
    a, b, c = (float(value) for value in box[1] - box[0])
    return 4.0 * (a * b + b * c + c * a) / (cell * cell)


def unrefined_triangles(mesh):
    """A mesh's triangles before the mesher refined features narrower than a
    cell: two per active cell of the extracted surface."""
    return 2.0 * mesh.stats.get("active_cells", len(mesh.triangles) / 2)


def preview_cell_size(graph, boxes, threads):
    """(cell size, default cell size). The default is the design's largest
    extent over PREVIEW_CELLS_ACROSS. A probe mesh of every part at
    PREVIEW_PROBE_FACTOR times that size estimates the preview's triangles,
    vertices (both scale with 1/cell^2) and crease segments (1/cell); the
    cell size grows until the estimate fits the preview limits. A part the
    probe cannot mesh counts with its bounding-box estimate. The mesher's
    refinement of features narrower than a cell does not scale that way (at
    the probe's coarse cells more features are that narrow), so the probe
    counts the extracted surface: two triangles per active cell
    (unrefined_triangles), and its vertices in proportion."""
    import numpy as np
    import radsdf
    lo = np.min([box[0] for box in boxes], axis=0)
    hi = np.max([box[1] for box in boxes], axis=0)
    default = max(float(np.max(hi - lo)) / PREVIEW_CELLS_ACROSS, PREVIEW_MINIMUM_CELL)
    probe = default * PREVIEW_PROBE_FACTOR
    area_factor = PREVIEW_PROBE_FACTOR ** 2
    triangles = vertices = segments = 0.0
    for node, box in zip(graph.roots, boxes):
        try:
            mesh = graph.mesh(node, cell_size=probe, crease_angle=PREVIEW_CREASE_ANGLE, threads=threads)
        except radsdf.SDFError:
            triangles += _triangle_estimate(box, default)
            vertices += _triangle_estimate(box, default) / 2
            continue
        unrefined = unrefined_triangles(mesh)
        triangles += unrefined * area_factor
        vertices += len(mesh.positions) * (unrefined / len(mesh.triangles) if len(mesh.triangles) else 1) * area_factor
        segments += len(mesh.creases) * PREVIEW_PROBE_FACTOR
    scale = max(math.sqrt(triangles / MAX_TRIANGLES), math.sqrt(vertices / MAX_VERTICES),
                segments / MAX_EDGE_SEGMENTS)
    return (default * scale * PREVIEW_MARGIN if scale > 1 else default), default


def _mesh_part(graph, node, label, cell, threads, *, facets=False):
    import radsdf
    diagnostic = {"stage": "preview_meshing", "cellSize": cell}
    def make_mesh(facet_detail):
        return graph.native_mesh(node, cell_size=cell, crease_angle=PREVIEW_CREASE_ANGLE,
                                 threads=threads, _preview=True, _facet_detail=facet_detail).mesh()

    try:
        mesh = make_mesh(facets)
        if facets and (mesh.stats["boundary_edges"] or mesh.stats["nonmanifold_edges"]):
            mesh = make_mesh(False)
    except radsdf.SDFError as error:
        raise ModelError(f"{label} could not be meshed at the preview cell size of {cell:.4g} mm: {error}",
                         diagnostic=diagnostic) from error
    if len(mesh.triangles) == 0:
        raise ModelError(f"{label} has no surface at the preview cell size of {cell:.4g} mm: it is empty or "
                         f"thinner than about {2 * cell:.3g} mm everywhere. Make its features thicker, or "
                         "check that booleans do not remove it entirely.", diagnostic=diagnostic)
    boundary = int(mesh.stats.get("boundary_edges", 0))
    if boundary:
        raise ModelError(f"{label} has an open surface at the preview cell size of {cell:.4g} mm ({boundary} "
                         f"boundary edges): features thinner than about {2 * cell:.3g} mm do not close. "
                         "Make thin walls, pins and gaps thicker.", diagnostic=diagnostic)
    return mesh


def mesh_parts(graph, normalized, boxes, threads, *, facets=False):
    """Mesh every part at one preview cell size within the preview limits.

    The first pass uses preview_cell_size; a pass that exceeds the limits
    stops early and is redone at a coarser cell size predicted from the
    triangles so far plus the remaining parts' estimates, at most
    MAX_PREVIEW_REMESHES times. Drop optional facet detail before coarsening.
    Returns (meshes, cell size, default cell size).
    """
    cell, default = preview_cell_size(graph, boxes, threads)
    remeshes = MAX_PREVIEW_REMESHES + int(facets)
    for attempt in range(remeshes + 1):
        meshes, triangles, vertices, segments, remaining = [], 0, 0, 0, None
        for index, (node, label) in enumerate(zip(graph.roots, normalized.labels)):
            mesh = (_mesh_part(graph, node, label, cell, threads, facets=True) if facets else
                    _mesh_part(graph, node, label, cell, threads))
            meshes.append(mesh)
            triangles += len(mesh.triangles)
            vertices += len(mesh.positions)
            segments += len(mesh.creases)
            if triangles > MAX_TRIANGLES or vertices > MAX_VERTICES or segments > MAX_EDGE_SEGMENTS:
                remaining = sum(_triangle_estimate(box, cell) for box in boxes[index + 1:])
                break
        if remaining is None:
            return meshes, cell, default
        if facets:
            facets = False
            continue
        if attempt == remeshes:
            break
        growth = (triangles + remaining) / triangles
        scale = max(math.sqrt(triangles * growth / MAX_TRIANGLES), math.sqrt(vertices * growth / MAX_VERTICES),
                    segments * growth / MAX_EDGE_SEGMENTS)
        cell *= max(scale, 1.0) * PREVIEW_MARGIN
    raise ModelError(f"The design is too detailed to preview within {MAX_TRIANGLES:,} triangles, even at a "
                     f"cell size of {cell:.4g} mm. Simplify fine repeated detail (threads, knurling, "
                     "lattices) or make the design smaller.", diagnostic={"stage": "preview_meshing",
                                                                          "cellSize": cell})


def _unit_normals(positions, triangles, normals):
    """Normals with unit length; zero or invalid ones come from the adjacent
    triangles' area-weighted face normals."""
    import numpy as np
    values = normals.astype(np.float64)
    lengths = np.linalg.norm(values, axis=1)
    if np.all(np.abs(lengths - 1.0) <= 1e-5):
        return normals
    usable = np.isfinite(lengths) & (lengths > 1e-12)
    values[usable] /= lengths[usable, None]
    if not usable.all():
        corners = positions.astype(np.float64)[triangles]
        faces = np.cross(corners[:, 1] - corners[:, 0], corners[:, 2] - corners[:, 0])
        sums = np.zeros_like(values)
        for corner in range(3):
            np.add.at(sums, triangles[:, corner], faces)
        sums_length = np.linalg.norm(sums, axis=1)
        fallback = np.where(sums_length[:, None] > 0, sums / np.maximum(sums_length, 1e-300)[:, None],
                            np.array([0.0, 0.0, 1.0]))
        values[~usable] = fallback[~usable]
    return values.astype(np.float32)


def mesh_pieces(stats):
    """A mesh's separate pieces: its connected surfaces less the sealed
    cavities (inward-facing surfaces of voids inside a piece)."""
    return int(stats.get("components", 1)) - int(stats.get("cavities", 0))


def _warnings(normalized, meshes, cell, default):
    messages = []
    if cell > default * (1 + 1e-9):
        messages.append(f"The preview cell size is {cell:.4g} mm instead of {default:.4g} mm to stay within "
                        f"{MAX_TRIANGLES:,} triangles; details smaller than about {2 * cell:.3g} mm may not show.")
    for label, mesh in zip(normalized.labels, meshes):
        nonmanifold = int(mesh.stats.get("nonmanifold_edges", 0))
        if nonmanifold:
            messages.append(f"{label} has {nonmanifold} non-manifold mesh edges at the preview cell size of "
                            f"{cell:.4g} mm: features thinner than about {2 * cell:.3g} mm touch or merge.")
        pieces = mesh_pieces(mesh.stats)
        if pieces > 1:
            messages.append(f"{label} has {pieces} disconnected pieces; they move and export as one "
                            "rigid part.")
    if len(messages) > MAX_WARNINGS:
        messages[MAX_WARNINGS - 1:] = [f"... and {len(messages) - MAX_WARNINGS + 1} more warnings."]
    return messages


def assemble_preview(normalized, meshes, cell, default, roots, artifact, fingerprint, printing_read):
    """The build result (SDF-DESIGN.md key order) with the display arrays
    under the PREVIEW_SECTIONS keys; write_preview_result moves them to mesh.bin."""
    import numpy as np
    positions, normals, triangles, creases, stats = [], [], [], [], []
    triangle_solids, edge_solids, offset = [], [], 0
    for solid, (label, mesh) in enumerate(zip(normalized.labels, meshes)):
        points = mesh.positions
        part_triangles = mesh.triangles.astype(np.int64)
        positions.append(points)
        normals.append(_unit_normals(points, part_triangles, mesh.normals))
        triangles.append(part_triangles + offset)
        creases.append(mesh.creases.reshape(-1, 6))
        triangle_solids.append(np.full(len(part_triangles), solid, dtype=np.int32))
        edge_solids.append(np.full(len(mesh.creases), solid, dtype=np.int32))
        offset += len(points)
        lo = points.min(axis=0).astype(np.float64)
        hi = points.max(axis=0).astype(np.float64)
        volume = float(mesh.stats["volume"])
        if not volume > 0:
            raise ModelError(f"{label} encloses no volume at the preview cell size of {cell:.4g} mm.",
                             diagnostic={"stage": "preview_meshing", "cellSize": cell})
        if not (math.isfinite(volume) and math.isfinite(float(mesh.stats["area"]))
                and all(math.isfinite(float(value)) for value in mesh.stats["centroid"])
                and np.isfinite(points).all() and np.isfinite(mesh.creases).all()):
            raise HostError(f"The kernel returned invalid mesh data for {label}.")
        stats.append({"solid": solid, "volume": volume, "area": float(mesh.stats["area"]),
                      "bounds": [*lo.tolist(), *hi.tolist()],
                      "centroid": [float(value) for value in mesh.stats["centroid"]],
                      "triangles": len(part_triangles), "components": mesh_pieces(mesh.stats),
                      "cavities": int(mesh.stats.get("cavities", 0))})
    lo = np.min([item["bounds"][:3] for item in stats], axis=0)
    hi = np.max([item["bounds"][3:] for item in stats], axis=0)
    if np.any(np.abs(lo) > COORDINATE_LIMIT) or np.any(np.abs(hi) > COORDINATE_LIMIT):
        raise ModelError(f"The design exceeds the size limit: coordinates must stay within "
                         f"±{COORDINATE_LIMIT:,.0f} mm.", diagnostic={"stage": "part_bounds"})
    preview = {
        "buildID": str(uuid.uuid4()),
        "artifactPath": str(artifact),
        "partRoots": list(roots),
        "bounds": [*lo.tolist(), *hi.tolist()],
        "volume": sum(item["volume"] for item in stats),
        "area": sum(item["area"] for item in stats),
        "solids": len(meshes),
        "parts": normalized.parts,
        "partStats": stats,
        "mesh": {"cellSize": cell,
                 "boundaryEdges": sum(int(mesh.stats.get("boundary_edges", 0)) for mesh in meshes),
                 "nonmanifoldEdges": sum(int(mesh.stats.get("nonmanifold_edges", 0)) for mesh in meshes)},
    }
    if normalized.motion is not None:
        preview["assemblyMotion"] = normalized.motion
    if printing_read:
        preview["printVersion"] = True
    if normalized.print_down:
        preview["printDown"] = [{"solid": index, "direction": normalized.print_down[index]}
                                for index in sorted(normalized.print_down)]
    if normalized.reference:
        preview["referenceSolids"] = list(normalized.reference)
    preview["validatedHash"] = fingerprint
    preview["warnings"] = _warnings(normalized, meshes, cell, default)
    preview.update(vertices=np.concatenate(positions), normals=np.concatenate(normals),
                   triangles=np.concatenate(triangles).astype(np.uint32), edges=np.concatenate(creases),
                   triangleSolids=np.concatenate(triangle_solids), edgeSolids=np.concatenate(edge_solids))
    return preview


# Display arrays travel to the app as packed little-endian 4-byte sections in
# mesh.bin beside compact JSON metadata: (preview key, section, element type,
# values per item). Float32 matches the GPU attributes the app uploads.
PREVIEW_SECTIONS = (
    ("vertices", "positions", "<f4", 3), ("normals", "normals", "<f4", 3), ("triangles", "triangles", "<u4", 3),
    ("edges", "edges", "<f4", 6), ("triangleSolids", "triangleSolids", "<i4", 1),
    ("edgeSolids", "edgeSolids", "<i4", 1),
)


def write_preview_result(preview, output):
    """Publish result.json, moving any display arrays into mesh.bin first.
    The caller's dictionary is not changed."""
    import numpy as np
    output = Path(output)
    result = dict(preview)
    sections, offset = {}, 0
    mesh = output / "mesh.bin"
    temporary = output / "mesh.bin.tmp"
    try:
        with temporary.open("wb") as stream:
            for key, name, dtype, width in PREVIEW_SECTIONS:
                values = result.pop(key, None)
                if values is None:
                    continue
                try:
                    packed = np.ascontiguousarray(values, dtype=dtype)
                except (TypeError, ValueError, OverflowError) as error:
                    raise ModelError(f"The preview {key} data is malformed.") from error
                if packed.ndim != (1 if width == 1 else 2) or (width > 1 and packed.shape[1] != width):
                    raise ModelError(f"The preview {key} data is malformed.")
                stream.write(packed.tobytes())
                sections[name] = {"offset": offset, "count": int(packed.size)}
                offset += packed.size * 4
    except BaseException:
        temporary.unlink(missing_ok=True)
        raise
    if sections:
        temporary.replace(mesh)
        result["meshBuffers"] = {"path": mesh.name, "sections": sections}
    else:
        temporary.unlink()
    staged = output / "result.json.tmp"
    staged.write_text(json.dumps(result, separators=(",", ":"), allow_nan=False))
    staged.replace(output / "result.json")


def write_artifact(data, output):
    """Publish model.rsdf and return its absolute path."""
    path = (Path(output) / "model.rsdf").resolve()
    staged = path.with_name("model.rsdf.tmp")
    try:
        staged.write_bytes(data)
        staged.replace(path)
    finally:
        staged.unlink(missing_ok=True)
    return path


# ------------------------------------------------------------------ custom fields

def implicit_nodes(order):
    """Indices of the IMPLICIT nodes (implicit() shapes) among the encoded nodes."""
    from radsdf import Opcode
    return [index for index, node in enumerate(order) if node.opcode == Opcode.IMPLICIT]


def implicit_owners(order, shapes, wanted):
    """{IMPLICIT node index: solid indices of the parts that read it}, for
    the node indices in `wanted`. Walks the parts' Python graphs, since an
    exact boolean fold encodes without its inner nodes."""
    index_of = {id(node): index for index, node in enumerate(order)}
    owners = {index: [] for index in wanted}
    for solid, shape in enumerate(shapes):
        seen, stack = set(), [shape]
        while stack:
            node = stack.pop()
            if id(node) in seen:
                continue
            seen.add(id(node))
            index = index_of.get(id(node))
            if index in owners:
                owners[index].append(solid)
            stack.extend(node.children)
    return owners


def _owners_text(solids, labels):
    names = [labels[solid] for solid in solids[:3]]
    if len(solids) > 3:
        names.append(f"{len(solids) - 3} more parts")
    return ", ".join(names) or "The design"


def _box_text(params):
    lo, hi = params[:3], params[3:]
    return f"(({', '.join(f'{value:g}' for value in lo)}), ({', '.join(f'{value:g}' for value in hi)}))"


def field_load_error(error, order, normalized):
    """A ModelError naming the part and program line of the implicit() shape
    whose field the kernel could not bound when loading the design, or None
    for other load failures."""
    import radsdf
    match = IMPLICIT_FAILURE.search(str(error))
    if match is None:
        return None
    index = int(match.group(1))
    if index >= len(order) or order[index].opcode != radsdf.Opcode.IMPLICIT:
        return None
    node = order[index]
    solids = implicit_owners(order, normalized.shapes, [index])[index]
    line = radsdf._implicit_line(node)
    where = f"implicit() at line {line}" if line is not None else "an implicit() shape"
    problem = radsdf._describe_field_nodes(match.group(2).strip(), order).rstrip(".")
    return ModelError(f"{_owners_text(solids, normalized.labels)}: the custom field of {where} has no finite slope "
                      f"bound within its bounds {_box_text(node.params)}: {problem}. {FIELD_ADVICE}")


def _finite_numbers(values):
    return all(isinstance(value, (int, float)) and math.isfinite(value) for value in values)


def field_record(node, solids, report):
    """The JSON record of one implicit() shape's analysis (verify_field's
    dictionary): finite values only."""
    import radsdf
    record = {"solids": solids, "bounds": list(node.params)}
    line = radsdf._implicit_line(node)
    if line is not None:
        record["line"] = line
    record["ok"] = bool(report.get("ok"))
    if record["ok"]:
        record["slopeBound"] = report["bound"]
        if _finite_numbers(report["value_range"]):
            record["valueRange"] = list(report["value_range"])
        gradient = report["gradient"]
        if report["samples"] and _finite_numbers(gradient.values()):
            record["gradient"] = {key: gradient[key] for key in ("min", "median", "max")}
        record["quality"] = report["quality"] if math.isfinite(report["quality"]) else 0.0
        record["boxes"] = report["boxes"]
    else:
        record["message"] = report.get("message") or "The field has no finite slope bound within its bounds."
        region = report.get("region")
        if region is not None and _finite_numbers([*region[0], *region[1]]):
            record["region"] = [*region[0], *region[1]]
    return record


def custom_fields(graph, order, normalized):
    """(records, warnings) of the design's implicit() shapes for the build
    summary: each one's parts, bounds, proven slope bound and quality,
    analysed on the loaded graph (the bound is the one the kernel uses)."""
    nodes = implicit_nodes(order)
    if not nodes:
        return [], []
    owners = implicit_owners(order, normalized.shapes, nodes[:MAX_SUMMARY_FIELDS])
    records = []
    for index in nodes[:MAX_SUMMARY_FIELDS]:
        record = field_record(order[index], owners[index], graph.analyze_field(index))
        if record["ok"]:  # always: the loaded graph proved every bound with the same analysis
            records.append({key: record[key] for key in ("solids", "line", "bounds", "slopeBound", "quality")
                            if key in record})
    warnings = []
    if len(nodes) > MAX_SUMMARY_FIELDS:
        warnings.append(f"The design has {len(nodes)} implicit() shapes; the summary lists the first "
                        f"{MAX_SUMMARY_FIELDS} (rad_verify reports up to {MAX_VERIFIED_FIELDS}).")
    return records, warnings


def encode_design(normalized):
    """(RSDF bytes, part root indices, encoded nodes) of the design's parts."""
    import radsdf
    try:
        data, roots, order = radsdf._encode(normalized.shapes, with_order=True)
    except radsdf.SDFError as error:
        raise ModelError(f"The design's shape graph cannot be encoded: {error}") from error
    if len(data) > MAX_GEOMETRY_BYTES:
        raise ModelError("The design's shape graph exceeds 512 MB; share repeated shapes instead of "
                         "rebuilding them.")
    return data, roots, order


def run_design(spec, directory):
    """Run the request's program; returns its Normalized parts and RADBuildMode."""
    import radsdf
    radsdf._implicit_lines.clear()
    result, build_mode = run_program(spec["script"], printing=spec.get("print", False),
                                     assets=ImportedAssets(spec.get("inputAssets", []), directory))
    # Free the graphs the program's queries loaded (mesh_solid's validation
    # graph among them) before the design's graph decodes the same meshes.
    radsdf._release_graphs()
    return normalize_result(result), build_mode


def verify_fields(spec, directory, output, threads, started):
    """rad_verify: run the program and analyse the field of every implicit()
    shape (at most MAX_VERIFIED_FIELDS) as its load would, without loading
    or meshing the design, so fields without a finite slope bound are
    reported rather than failing. Publishes result.json only."""
    import radsdf
    runtime = runtime_info()
    normalized, _ = run_design(spec, directory)
    _, _, order = encode_design(normalized)
    nodes = implicit_nodes(order)
    owners = implicit_owners(order, normalized.shapes, nodes[:MAX_VERIFIED_FIELDS])
    fields = []
    for index in nodes[:MAX_VERIFIED_FIELDS]:
        node = order[index]
        try:
            report = radsdf.verify_field(node)
        except radsdf.SDFError as error:  # it reads another implicit() shape that cannot load
            report = {"ok": False, "message": str(error)}
        fields.append(field_record(node, owners[index], report))
    result = {"solids": len(normalized.shapes), "parts": normalized.parts, "fields": fields,
              "fieldCount": len(nodes), "buildSeconds": time.perf_counter() - started, "threads": threads,
              "runtime": runtime}
    staged = Path(output) / "result.json.tmp"
    staged.write_text(json.dumps(result, separators=(",", ":"), allow_nan=False))
    staged.replace(Path(output) / "result.json")
    return result


def build(spec, directory, output, threads, started=None):
    """Run one build request (inputs in `directory`) and publish model.rsdf,
    mesh.bin and result.json in `output`; returns the result, including the
    display arrays. A request with verifyFields analyses its custom fields
    instead (verify_fields)."""
    import radsdf
    started = time.perf_counter() if started is None else started
    validate_spec(spec)
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    if spec.get("verifyFields"):
        return verify_fields(spec, directory, output, threads, started)
    runtime = runtime_info()
    fingerprint = validation_hash(spec, directory, runtime)
    normalized, build_mode = run_design(spec, directory)
    data, roots, order = encode_design(normalized)
    try:
        graph = radsdf.Graph.from_bytes(data, roots)
    except radsdf.SDFError as error:
        failure = field_load_error(error, order, normalized)
        if failure is not None:
            raise failure from error
        message = radsdf._describe_field_nodes(str(error), order)
        raise ModelError(f"The kernel rejected the design's shape graph: {message}") from error
    with graph:
        boxes = part_bounds(graph, normalized)
        facets = any(node.opcode == radsdf.Opcode.MESH for node in order)
        meshes, cell, default = mesh_parts(graph, normalized, boxes, threads, facets=facets)
        fields, field_warnings = custom_fields(graph, order, normalized)
    artifact = (output / "model.rsdf").resolve()
    preview = assemble_preview(normalized, meshes, cell, default, roots, artifact, fingerprint,
                               build_mode.printing_read)
    if fields:
        preview["customFields"] = fields
    write_artifact(data, output)
    preview.update(buildSeconds=time.perf_counter() - started, threads=threads, runtime=runtime)
    # Keep the protocol's key order: warnings follow the runtime facts.
    warnings = preview.pop("warnings")
    preview["warnings"] = warnings + field_warnings[:max(0, MAX_WARNINGS - len(warnings))]
    write_preview_result(preview, output)
    return preview
