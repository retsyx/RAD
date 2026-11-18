"""RAD assemblies: named part hierarchies, rigid placements and motion.

A program returns a single solid, a list of solids, or an `Assembly`. An
Assembly is a folder of physical parts and sub-folders:

    result = Assembly("Rubber duck")
    result.add(body, id="body", name="Body", color="#F6C915")
    eyes = result.folder("Eyes", id="eyes")
    eyes.add(eye, id="eye-left", name="Left eye", color=(0.05, 0.05, 0.05),
             at=Placement(translate=(9, 7, 38)))

- Every physical part is one rigid body (one solid index), numbered in
  parent-before-child insertion order, even when its shape has several
  disconnected pieces.
- `id` is the part's stable identity; parts without one get an identity
  derived from their name or position in the tree.
- `color` ((r, g, b) in 0..1 or "#RRGGBB"), `print_down` (the outward
  direction that faces the build plate when 3D printing) and `reference`
  are inherited by descendants that do not set their own.
- `reference=True` marks reference parts, such as a circuit board inside an
  enclosure: they are shown, measured and checked for clearance like any
  part, but they are not printed, so 3D printing layouts and exports leave
  them out.
- `at` is a rigid placement (a 4x4 matrix or a Placement) composed from the
  root down and applied to the part's shape.
- `assembly.motion` is the root assembly's motion definition: a `Motion`
  builder (`ground`, `connector`, `fixed`, `revolute`, `slider`, `couple`),
  or a raw dictionary with the same schema. Connector frames are in the
  built design's world coordinates. `gear`, `rack`, `screw`, `cam` and
  `geneva` return coupling fields for `Motion.couple`.

Angles are in degrees and lengths in millimetres.
"""
import math
import numbers

import numpy as np

import radsdf
from assembly_motion import has_control_characters
from radsdf import SDF2, SDF3, Field, SDFError

__all__ = ["Assembly", "Part", "Placement", "Motion", "gear", "rack", "screw", "cam", "geneva"]

MAX_TABLE_SAMPLES = 4096
_AXES = {"x": (1.0, 0.0, 0.0), "y": (0.0, 1.0, 0.0), "z": (0.0, 0.0, 1.0)}


# ------------------------------------------------------------------- validation

def _number(value, label):
    if isinstance(value, bool) or not isinstance(value, numbers.Real):
        raise SDFError(f"{label} must be a finite number.")
    value = float(value)
    if not math.isfinite(value):
        raise SDFError(f"{label} must be a finite number.")
    return value


def _vector(value, label):
    if isinstance(value, (str, bytes)):
        raise SDFError(f"{label} must be three finite numbers.")
    try:
        items = list(value)
    except TypeError:
        raise SDFError(f"{label} must be three finite numbers.") from None
    if len(items) != 3:
        raise SDFError(f"{label} must be three finite numbers.")
    return [_number(item, label) for item in items]


def _direction(value, label):
    """A unit vector from 'x', 'y', 'z' or three finite numbers."""
    if isinstance(value, str) and value.lower() in _AXES:
        return list(_AXES[value.lower()])
    vector = _vector(value, label)
    length = math.sqrt(sum(item * item for item in vector))
    if length < 1e-9:
        raise SDFError(f"{label} must be a nonzero direction.")
    return [item / length for item in vector]


def _text(value, label, maximum):
    """A stripped single-line string of at most `maximum` characters, or None."""
    if value is None:
        return None
    if not isinstance(value, str):
        raise SDFError(f"{label} must be a string.")
    value = value.strip()
    if has_control_characters(value):
        raise SDFError(f"{label} must not contain control or format characters.")
    if len(value) > maximum:
        raise SDFError(f"{label} must be at most {maximum} characters.")
    return value or None


def _identity(value, label="id"):
    """A stable identity: used verbatim, nonempty, at most 1024 UTF-8 bytes."""
    if value is None:
        return None
    if not isinstance(value, str) or not value.strip():
        raise SDFError(f"{label} must be a nonempty string.")
    if len(value.encode("utf-8")) > 1024 or has_control_characters(value):
        raise SDFError(f"{label} must be at most 1024 bytes without control or format characters.")
    return value


def _color(value):
    """(r, g, b) in 0..1 or "#RRGGBB" as an (r, g, b) tuple of floats."""
    if value is None:
        return None
    message = 'color must be (r, g, b) with values between 0 and 1, or "#RRGGBB".'
    if isinstance(value, str):
        text = value.strip()
        if len(text) != 7 or text[0] != "#" or any(char not in "0123456789abcdefABCDEF" for char in text[1:]):
            raise SDFError(message)
        return tuple(int(text[index:index + 2], 16) / 255.0 for index in (1, 3, 5))
    try:
        items = list(value)
    except TypeError:
        raise SDFError(message) from None
    if len(items) != 3:
        raise SDFError(message)
    rgb = []
    for item in items:
        if isinstance(item, bool) or not isinstance(item, numbers.Real):
            raise SDFError(message)
        item = float(item)
        if not math.isfinite(item) or not 0.0 <= item <= 1.0:
            raise SDFError(message)
        rgb.append(item)
    return tuple(rgb)


def _print_down(value):
    if value is None:
        return None
    return tuple(_direction(value, "print_down"))


def _reference(value):
    """True, False, or None to inherit."""
    if value is None:
        return None
    if not isinstance(value, (bool, np.bool_)):
        raise SDFError("reference must be True or False.")
    return bool(value)


def _rigid(matrix, label):
    """A validated rigid 4x4 matrix (proper rotation and translation)."""
    if isinstance(matrix, Placement):
        return matrix.matrix
    try:
        array = np.array(matrix, dtype=float)
    except (TypeError, ValueError):
        array = None
    if array is None or array.shape not in ((4, 4), (3, 4)) or not np.isfinite(array).all():
        raise SDFError(f"{label} must be a Placement or a finite 4x4 rigid matrix.")
    if array.shape == (3, 4):
        array = np.vstack([array, (0.0, 0.0, 0.0, 1.0)])
    if not np.allclose(array[3], (0.0, 0.0, 0.0, 1.0), rtol=0.0, atol=1e-9):
        raise SDFError(f"{label} must be a rigid matrix whose last row is 0, 0, 0, 1.")
    rotation = array[:3, :3]
    if (not np.allclose(rotation.T @ rotation, np.eye(3), rtol=0.0, atol=1e-6)
            or np.linalg.det(rotation) <= 0):
        raise SDFError(f"{label} must be a proper rotation and a translation, without scale, shear "
                       "or reflection.")
    if not np.allclose(rotation.T @ rotation, np.eye(3), rtol=0.0, atol=1e-12):
        u, _, vt = np.linalg.svd(rotation)
        array[:3, :3] = u @ vt
    array[3] = (0.0, 0.0, 0.0, 1.0)
    return array


def _solid(shape, label):
    if isinstance(shape, SDF2):
        raise SDFError(f"{label} needs a 3D solid; extrude or revolve the 2D profile first.")
    if isinstance(shape, Field):
        raise SDFError(f"{label} needs a 3D solid; a custom field becomes one through implicit(field, bounds).")
    if not isinstance(shape, SDF3):
        raise SDFError(f"{label} needs a 3D shape, got {type(shape).__name__}.")
    return shape


# -------------------------------------------------------------------- placement

class Placement:
    """A rigid placement: rotate by `rotate=(angle, axis)` (degrees, about an
    axis through the origin; axis 'x', 'y', 'z' or a vector), then translate.
    `a @ b` applies b first, then a."""
    __slots__ = ("_matrix",)

    def __init__(self, translate=(0.0, 0.0, 0.0), rotate=None):
        matrix = np.eye(4)
        if rotate is not None:
            try:
                angle, axis = rotate
            except (TypeError, ValueError):
                raise SDFError("Placement rotate must be (angle in degrees, axis), "
                               "for example rotate=(90, 'z').") from None
            matrix[:3, :3] = radsdf.rotation_matrix(_number(angle, "Placement rotate angle"),
                                                    _direction(axis, "Placement rotate axis"))
        matrix[:3, 3] = _vector(translate, "Placement translate")
        object.__setattr__(self, "_matrix", matrix)

    @classmethod
    def from_matrix(cls, matrix):
        """A Placement from a rigid 4x4 (or 3x4) matrix."""
        placement = cls.__new__(cls)
        object.__setattr__(placement, "_matrix", _rigid(matrix, "Placement matrix"))
        return placement

    def __setattr__(self, name, value):
        raise AttributeError("Placements are immutable.")

    @property
    def matrix(self):
        """The 4x4 row-major matrix acting on column vectors."""
        return self._matrix.copy()

    def __matmul__(self, other):
        if not isinstance(other, Placement):
            return NotImplemented
        return Placement.from_matrix(self._matrix @ other._matrix)

    def __repr__(self):
        return f"Placement.from_matrix({self._matrix.tolist()})"


def _placement(value):
    """`at` as a Placement, or None."""
    if value is None or isinstance(value, Placement):
        return value
    return Placement.from_matrix(_rigid(value, "at"))


# ------------------------------------------------------------- parts and folders

class Part:
    """A physical part: one rigid body made from one 3D shape. Create parts
    with Assembly.add."""
    __slots__ = ("_shape", "_id", "_name", "_color", "_print_down", "_reference", "_at")

    def __init__(self, shape, id=None, name=None, color=None, print_down=None, at=None, reference=None):
        object.__setattr__(self, "_shape", _solid(shape, "A part"))
        object.__setattr__(self, "_id", _identity(id))
        object.__setattr__(self, "_name", _text(name, "Part name", 160))
        object.__setattr__(self, "_color", _color(color))
        object.__setattr__(self, "_print_down", _print_down(print_down))
        object.__setattr__(self, "_reference", _reference(reference))
        object.__setattr__(self, "_at", _placement(at))

    def __setattr__(self, name, value):
        raise AttributeError("Part attributes are fixed when the part is added.")

    shape = property(lambda self: self._shape, doc="The part's shape before placement.")
    id = property(lambda self: self._id, doc="The explicit stable identity, or None.")
    name = property(lambda self: self._name, doc="The display name, or None.")
    color = property(lambda self: self._color, doc="Own (r, g, b) color, or None to inherit.")
    print_down = property(lambda self: self._print_down, doc="Own unit print-down direction, or None.")
    reference = property(lambda self: self._reference, doc="Own reference flag, or None to inherit.")
    at = property(lambda self: self._at, doc="Own Placement, or None.")

    def __repr__(self):
        label = self._name or self._id or "unnamed"
        return f"<Part {label!r}>"


class Assembly:
    """A folder of physical parts and sub-folders; see the module docstring."""
    __slots__ = ("_name", "_id", "_color", "_print_down", "_reference", "_at", "_children", "_motion", "_parent")

    def __init__(self, name=None, id=None, color=None, print_down=None, at=None, reference=None):
        object.__setattr__(self, "_name", _text(name, "Assembly name", 160))
        object.__setattr__(self, "_id", _identity(id))
        object.__setattr__(self, "_color", _color(color))
        object.__setattr__(self, "_print_down", _print_down(print_down))
        object.__setattr__(self, "_reference", _reference(reference))
        object.__setattr__(self, "_at", _placement(at))
        object.__setattr__(self, "_children", [])
        object.__setattr__(self, "_motion", None)
        object.__setattr__(self, "_parent", None)

    def __setattr__(self, name, value):
        if name == "motion":
            object.__setattr__(self, "_motion", _motion_value(value))
            return
        raise AttributeError(f"Assembly has no settable attribute {name!r}; pass it when creating the "
                             "assembly or use add(), folder() and add_assembly().")

    def __getattr__(self, name):
        if name == "metadata":
            raise AttributeError("Assembly has no metadata: set assembly.motion for motion, and pass id=, "
                                 "color=, print_down= and reference= to add() and folder().")
        raise AttributeError(f"'Assembly' object has no attribute {name!r}")

    name = property(lambda self: self._name, doc="The display name, or None.")
    id = property(lambda self: self._id, doc="The explicit stable identity, or None.")
    color = property(lambda self: self._color, doc="Own (r, g, b) color, or None to inherit.")
    print_down = property(lambda self: self._print_down, doc="Own unit print-down direction, or None.")
    reference = property(lambda self: self._reference, doc="Own reference flag, or None to inherit.")
    at = property(lambda self: self._at, doc="Own Placement, or None.")
    children = property(lambda self: tuple(self._children), doc="Parts and folders in insertion order.")

    @property
    def motion(self):
        """The motion definition: a Motion builder (created on first use), or
        the raw dictionary assigned with `assembly.motion = {...}`."""
        if self._motion is None:
            object.__setattr__(self, "_motion", Motion())
        return self._motion

    def add(self, shape, id=None, name=None, color=None, print_down=None, at=None, reference=None):
        """Add a physical part made from a 3D shape and return the Part."""
        if isinstance(shape, Assembly):
            if any(value is not None for value in (id, name, color, print_down, at, reference)):
                raise SDFError("Pass a sub-assembly's id, name, color, print_down, at and reference when "
                               "creating it, then add it with add_assembly().")
            return self.add_assembly(shape)
        if isinstance(shape, Part):
            raise SDFError("add() takes a shape; each Part belongs to the assembly that created it.")
        if isinstance(shape, (list, tuple)):
            raise SDFError("add() takes one 3D shape; combine pieces with union(...) or add one part each.")
        part = Part(shape, id=id, name=name, color=color, print_down=print_down, at=at, reference=reference)
        self._children.append(part)
        return part

    def folder(self, name, id=None, color=None, print_down=None, at=None, reference=None):
        """Add and return a named sub-assembly."""
        child = Assembly(name, id=id, color=color, print_down=print_down, at=at, reference=reference)
        return self.add_assembly(child)

    def add_assembly(self, assembly):
        """Graft an existing assembly (and its parts) under this one; returns it."""
        if not isinstance(assembly, Assembly):
            raise SDFError("add_assembly() takes an Assembly.")
        if assembly._parent is not None:
            raise SDFError("This assembly already belongs to another assembly; create a new one for "
                           "each instance.")
        node = self
        while node is not None:
            if node is assembly:
                raise SDFError("An assembly cannot contain itself.")
            node = node._parent
        object.__setattr__(assembly, "_parent", self)
        self._children.append(assembly)
        return assembly

    def parts(self):
        """Every Part below this assembly, in solid-index order."""
        found, pending = [], [self]
        while pending:
            node = pending.pop()
            if isinstance(node, Part):
                found.append(node)
            else:
                pending.extend(reversed(node._children))
        return found

    def __repr__(self):
        label = self._name or self._id or "unnamed"
        return f"<Assembly {label!r}: {len(self._children)} children>"


def _motion_value(value):
    if value is None or isinstance(value, (Motion, dict)):
        return value
    raise SDFError("assembly.motion must be a Motion or a dictionary with grounded, connectors, "
                   "joints and optional couplings.")


# ----------------------------------------------------------------------- motion

def _part_reference(part, label):
    if isinstance(part, Part):
        return part
    if not isinstance(part, str):
        raise SDFError(f"{label} must be a physical part: the Part returned by add(), or its id.")
    return _identity(part, label)


def _auto_x_axis(z):
    """A unit X axis perpendicular to z: world X projected, or world Y when z is near X."""
    helper = (1.0, 0.0, 0.0) if abs(z[0]) < 0.9 else (0.0, 1.0, 0.0)
    dot = sum(a * b for a, b in zip(helper, z))
    x = [a - dot * b for a, b in zip(helper, z)]
    length = math.sqrt(sum(item * item for item in x))
    return [item / length for item in x]


class Motion:
    """Builds the root assembly's motion definition (the same dictionary as
    `assembly.motion = {...}`). Joints connect connector frames on two
    physical parts: the child frame aligns with the parent frame, then moves
    by the joint position (revolute: degrees about the parent Z axis; slider:
    millimetres along it). Every moving group needs a grounded part. Methods
    return the new connector or joint id for use in later calls."""
    __slots__ = ("_grounded", "_connectors", "_joints", "_couplings")

    def __init__(self):
        self._grounded = []
        self._connectors = []
        self._joints = []
        self._couplings = []

    @property
    def empty(self):
        return not (self._grounded or self._connectors or self._joints or self._couplings)

    def ground(self, *parts):
        """Fix parts (Parts or part ids) to the world."""
        if not parts:
            raise SDFError("ground() needs at least one part.")
        for part in parts:
            self._grounded.append(_part_reference(part, "A grounded part"))

    def connector(self, id, part, origin, x_axis=None, z_axis=(0.0, 0.0, 1.0), frame="world"):
        """A coordinate frame on a part. `origin`, `x_axis` and `z_axis` are in
        world coordinates of the built design (frame="world"), or in the part's
        own coordinates before its `at` placements (frame="part"). Without
        `x_axis`, X is the world X axis made perpendicular to Z."""
        identity = _identity(id, "Connector id")
        if any(entry["id"] == identity for entry in self._connectors):
            raise SDFError(f"Connector id {identity!r} is already used.")
        if frame not in ("world", "part"):
            raise SDFError("Connector frame must be 'world' or 'part'.")
        z = _direction(z_axis, "Connector z_axis")
        if x_axis is None:
            x = _auto_x_axis(z)
        else:
            x = _direction(x_axis, "Connector x_axis")
            if abs(sum(a * b for a, b in zip(x, z))) > 1e-6:
                raise SDFError(f"Connector {identity!r} x_axis and z_axis must be perpendicular.")
        self._connectors.append({"id": identity, "part": _part_reference(part, "A connector part"),
                                 "origin": _vector(origin, "Connector origin"), "xAxis": x, "zAxis": z,
                                 "frame": frame})
        return identity

    def _connector_id(self, value, label):
        identity = _identity(value, label)
        if not any(entry["id"] == identity for entry in self._connectors):
            raise SDFError(f"{label} {identity!r} is not defined; create it with connector() first.")
        return identity

    def _joint(self, kind, id, parent, child, name, position, limits):
        identity = _identity(id, "Joint id")
        if any(joint["id"] == identity for joint in self._joints):
            raise SDFError(f"Joint id {identity!r} is already used.")
        joint = {"id": identity, "type": kind,
                 "parent": self._connector_id(parent, "Parent connector"),
                 "child": self._connector_id(child, "Child connector")}
        if name is not None:
            joint["name"] = _text(name, "Joint name", 160) or identity
        if position is not None:
            joint["position"] = _number(position, "Joint position")
        if limits is not None:
            try:
                lower, upper = limits
            except (TypeError, ValueError):
                raise SDFError("Joint limits must be (lower, upper).") from None
            lower, upper = _number(lower, "Joint limit"), _number(upper, "Joint limit")
            if lower > upper:
                raise SDFError("Joint limits must be ordered (lower, upper).")
            joint["limits"] = [lower, upper]
        self._joints.append(joint)
        return identity

    def fixed(self, id, parent, child, name=None):
        """Rigidly join two parts at aligned connector frames."""
        return self._joint("fixed", id, parent, child, name, None, None)

    def revolute(self, id, parent, child, name=None, position=None, limits=None):
        """A hinge about the parent connector's Z axis; position and limits in degrees."""
        return self._joint("revolute", id, parent, child, name, position, limits)

    def slider(self, id, parent, child, name=None, position=None, limits=None):
        """A prismatic joint along the parent connector's Z axis; position and limits in mm."""
        return self._joint("slider", id, parent, child, name, position, limits)

    def couple(self, id, driver, follower, type=None, ratio=None, offset=None, samples=None,
               period=None, advance=None):
        """Make the follower joint's position a function of the driver joint's:
        linear (follower = ratio * driver + offset) or a sampled table
        ([[driver, follower], ...], optionally periodic with an advance per
        period). Usually called with a helper: couple(id, a, b, **gear(20, 40))."""
        identity = _identity(id, "Coupling id")
        if any(coupling["id"] == identity for coupling in self._couplings):
            raise SDFError(f"Coupling id {identity!r} is already used.")
        joints = {joint["id"] for joint in self._joints}
        ends = []
        for value, label in ((driver, "Coupling driver"), (follower, "Coupling follower")):
            joint = _identity(value, label)
            if joint not in joints:
                raise SDFError(f"{label} {joint!r} is not a joint; create the joint first.")
            ends.append(joint)
        if type is None:
            type = "linear" if ratio is not None else "table" if samples is not None else None
        coupling = {"id": identity, "type": type, "driver": ends[0], "follower": ends[1]}
        if type == "linear":
            if ratio is None or samples is not None or period is not None or advance is not None:
                raise SDFError("A linear coupling takes ratio and an optional offset.")
            coupling["ratio"] = _number(ratio, "Coupling ratio")
            if coupling["ratio"] == 0:
                raise SDFError("A linear coupling ratio must be nonzero.")
            if offset is not None:
                coupling["offset"] = _number(offset, "Coupling offset")
        elif type == "table":
            if samples is None or ratio is not None or offset is not None:
                raise SDFError("A table coupling takes samples and an optional period and advance.")
            coupling["samples"] = _samples(samples)
            if period is not None:
                coupling["period"] = _number(period, "Coupling period")
                coupling["advance"] = _number(0.0 if advance is None else advance, "Coupling advance")
            elif advance is not None:
                raise SDFError("A coupling advance requires a period.")
        else:
            raise SDFError("Coupling type must be 'linear' (ratio) or 'table' (samples).")
        self._couplings.append(coupling)
        return identity

    def definition(self):
        """The motion dictionary. Parts must be referenced by id, or be Parts
        with an explicit id, and connectors must use frame='world'; the build
        resolves the rest."""
        if any(entry["frame"] == "part" for entry in self._connectors):
            raise SDFError("Connectors with frame='part' are placed when the design is built; "
                           "motion.definition() needs frame='world' connectors.")

        def resolve(part):
            if isinstance(part, Part):
                if part.id is None:
                    raise SDFError("Give the part an explicit id to use it in motion.definition().")
                return part.id, None
            return part, None
        return self._build(resolve)

    def _build(self, resolve):
        """The motion dictionary with part references resolved by `resolve(part)
        -> (part id, 4x4 world placement of the part, or None when the part is
        not in the assembly)`."""
        connectors = []
        for entry in self._connectors:
            part, placement = resolve(entry["part"])
            origin, x, z = entry["origin"], entry["xAxis"], entry["zAxis"]
            if entry["frame"] == "part":
                if placement is None:
                    raise SDFError(f"Connector {entry['id']!r} uses frame='part', but its part {part!r} "
                                   "is not in the assembly.")
                rotation, offset = placement[:3, :3], placement[:3, 3]
                origin = (rotation @ origin + offset).tolist()
                x, z = (rotation @ x).tolist(), (rotation @ z).tolist()
            connectors.append({"id": entry["id"], "part": part, "origin": list(origin),
                               "xAxis": list(x), "zAxis": list(z)})
        return {"grounded": [resolve(part)[0] for part in self._grounded], "connectors": connectors,
                "joints": [dict(joint) for joint in self._joints],
                "couplings": [dict(coupling) for coupling in self._couplings]}

    def __repr__(self):
        return (f"<Motion {len(self._grounded)} grounded, {len(self._connectors)} connectors, "
                f"{len(self._joints)} joints, {len(self._couplings)} couplings>")


# ------------------------------------------------------------- coupling helpers

def _samples(samples):
    """[[driver, follower], ...] with strictly increasing driver positions."""
    if isinstance(samples, (str, bytes)):
        raise SDFError("Coupling samples must be [[driver, follower], ...].")
    try:
        rows = [list(row) for row in samples]
    except TypeError:
        raise SDFError("Coupling samples must be [[driver, follower], ...].") from None
    if not 2 <= len(rows) <= MAX_TABLE_SAMPLES:
        raise SDFError(f"A table coupling needs between 2 and {MAX_TABLE_SAMPLES} samples.")
    points = []
    for row in rows:
        if len(row) != 2:
            raise SDFError("Each coupling sample must be [driver, follower].")
        points.append([_number(row[0], "Coupling sample"), _number(row[1], "Coupling sample")])
    if any(b[0] <= a[0] for a, b in zip(points, points[1:])):
        raise SDFError("Coupling sample driver positions must increase strictly.")
    return points


def _positive(value, label):
    value = _number(value, label)
    if value <= 0:
        raise SDFError(f"{label} must be positive.")
    return value


def _nonzero(value, label):
    value = _number(value, label)
    if value == 0:
        raise SDFError(f"{label} must be nonzero.")
    return value


def gear(n_driver, n_follower, internal=False):
    """Two meshing gears with tooth counts n_driver and n_follower (revolute
    joints with parallel Z axes): ratio -n_driver/n_follower, or
    +n_driver/n_follower when one of them is an internal (ring) gear."""
    ratio = _positive(n_driver, "Driver tooth count") / _positive(n_follower, "Follower tooth count")
    return {"type": "linear", "ratio": ratio if internal else -ratio}


def rack(pitch_radius):
    """A pinion (revolute driver, degrees) moving a rack (slider follower, mm):
    2*pi*pitch_radius/360 mm per degree. A negative radius reverses the rack."""
    return {"type": "linear", "ratio": 2.0 * math.pi * _nonzero(pitch_radius, "Pitch radius") / 360.0}


def screw(lead):
    """A screw (revolute driver, degrees) moving a nut (slider follower, mm):
    lead/360 mm per degree. A negative lead is a left-hand thread."""
    return {"type": "linear", "ratio": _nonzero(lead, "Screw lead") / 360.0}


def cam(fn, start, stop, step=0.5, period=None, advance=None):
    """A table coupling sampling follower = fn(driver) from `start` to `stop`
    (inclusive) at most `step` apart. With `period`, the table repeats every
    period and the follower advances by `advance` per period; when the
    samples span a whole period, `advance` defaults to fn(stop) - fn(start)
    and otherwise to 0."""
    if not callable(fn):
        raise SDFError("cam() needs a function of the driver position.")
    start, stop = _number(start, "cam start"), _number(stop, "cam stop")
    step = _positive(step, "cam step")
    if stop <= start:
        raise SDFError("cam stop must be greater than start.")
    intervals = max(1, math.ceil((stop - start) / step - 1e-9))
    if intervals + 1 > MAX_TABLE_SAMPLES:
        raise SDFError(f"cam would need {intervals + 1} samples; use a step of at least "
                       f"{(stop - start) / (MAX_TABLE_SAMPLES - 1):.6g}.")
    xs = [start + (stop - start) * index / intervals for index in range(intervals + 1)]
    xs[-1] = stop
    samples = []
    for x in xs:
        try:
            y = fn(x)
        except Exception as error:
            raise SDFError(f"cam function failed at {x:.6g}: {type(error).__name__}: {error}") from error
        samples.append([x, _number(y, f"cam function value at {x:.6g}")])
    coupling = {"type": "table", "samples": samples}
    if period is None:
        if advance is not None:
            raise SDFError("A cam advance requires a period.")
        return coupling
    period = _positive(period, "cam period")
    tolerance = 1e-9 * max(1.0, period)
    if stop - start > period + tolerance:
        raise SDFError("cam samples must lie within one period.")
    whole = stop - start >= period - tolerance
    if advance is None:
        advance = samples[-1][1] - samples[0][1] if whole else 0.0
    advance = _number(advance, "cam advance")
    if whole:
        closing = samples[0][1] + advance
        if abs(samples[-1][1] - closing) > 1e-6 * max(1.0, abs(closing)):
            raise SDFError("cam samples do not close: fn(stop) must equal fn(start) + advance.")
        samples[-1][1] = closing
    coupling.update(period=period, advance=advance)
    return coupling


def geneva(slots, crank_at_center=0.0, step=0.5, sign=-1):
    """An external Geneva drive: a crank (revolute driver) indexing a star
    wheel with `slots` slots (revolute follower, Z axes parallel) by
    sign*360/slots per crank turn. The center distance C and crank radius
    A = C*sin(180/slots) give s = sin(pi/slots); the pin is engaged while the
    crank is within w = 90 - 180/slots degrees of `crank_at_center`, the crank
    angle at which the pin points at the star centre, and the star turns
    phi = atan2(s*sin(psi), 1 - s*cos(psi)) at psi degrees from there.

    The star's position is 0 from crank_at_center-180 to crank_at_center-w,
    sign*360/slots from crank_at_center+w, and sign*180/slots at the centre:
    model the star at rest (the line of centres between two slots) and the
    crank with its pin pointing at the star at crank position crank_at_center.
    The table is periodic (360 degrees) and closes exactly."""
    if isinstance(slots, bool) or not isinstance(slots, numbers.Integral) or not 3 <= slots <= 360:
        raise SDFError("geneva slots must be a whole number between 3 and 360.")
    if sign not in (-1, 1):
        raise SDFError("geneva sign must be -1 (star turns opposite to the crank) or 1.")
    center = _number(crank_at_center, "geneva crank_at_center")
    step = _positive(step, "geneva step")
    count = int(slots)
    s = math.sin(math.pi / count)
    half = 180.0 / count
    pitch = 360.0 / count
    w = 90.0 - half
    intervals = max(1, math.ceil(2.0 * w / step - 1e-9))
    if intervals + 3 > MAX_TABLE_SAMPLES:
        raise SDFError(f"geneva would need {intervals + 3} samples; use a larger step.")
    samples = [[center - 180.0, 0.0]]
    for index in range(intervals + 1):
        psi = -w + 2.0 * w * index / intervals
        phi = math.degrees(math.atan2(s * math.sin(math.radians(psi)), 1.0 - s * math.cos(math.radians(psi))))
        samples.append([center + psi, sign * (phi + half)])
    # The engagement ends are exactly the dwell positions (phi(+-w) = +-180/slots).
    samples[1][1] = 0.0
    samples[-1][1] = sign * pitch
    samples.append([center + 180.0, sign * pitch])
    return {"type": "table", "samples": samples, "period": 360.0, "advance": sign * pitch}
