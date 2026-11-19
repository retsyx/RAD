"""RAD's signed distance field (SDF) modeling API.

Programs build solids from the constructor functions below and combine them
with booleans, transforms and modifiers. Shapes are immutable nodes of an
expression graph: each holds an RSDF opcode, its parameters and its child
shapes, exactly as the binary RSDF format stores them (SDF-DESIGN.md).
Distances, gradients, bounds, ray casts and meshes come from the native
kernel (libRADSDF.dylib) through ctypes; building shapes needs no kernel.

Conventions:
- Distances are negative inside, zero on the surface and positive outside,
  in millimetres.
- 3D primitives are centred on their `center` argument (default origin).
- 2D profiles live in the XY plane. `extrude(h)` spans z = 0..h unless
  `center=True`; `revolve()` maps profile x to the radius about Z and
  profile y to Z.
- Angles are in degrees. Rotations follow the right-hand rule.
- Combining shapes never mutates them; every operation returns a new shape.
"""
import collections
import collections.abc
import contextlib
import ctypes
import enum
import functools
import itertools
import math
import numbers
import operator
import os
from pathlib import Path
import re
import struct
import sys
import threading
import weakref

import numpy as np

__all__ = [
    "SDFError", "Bounds", "SDF3", "SDF2", "Graph", "Mesh",
    "sphere", "box", "cylinder", "cone", "capsule", "torus", "ellipsoid", "half_space",
    "circle", "rect", "polygon", "regular_polygon", "slot", "ellipse", "half_plane",
    "arc_points", "involute_gear", "union", "intersection", "difference",
    "rotation_matrix", "gradient", "raycast", "serialize", "auto_cell_size",
    "merge_meshes", "kernel_version",
]

FORMAT_VERSION = 1

# RSDF limits (SDF-DESIGN.md) and modeling limits.
_MAX_NODES = 4_000_000
_MAX_CHILDREN = 65_536
_MAX_PARAMS = 1_000_000
_MAX_BYTES = 512 * 1024 * 1024
_MAX_COPIES = 10_000
_INT32_MAX = 2**31 - 1


class SDFError(ValueError):
    """An invalid shape construction or evaluation request."""


class Opcode(enum.IntEnum):
    """RSDF node opcodes (SDF-DESIGN.md)."""
    SPHERE = 1
    BOX = 2
    CYLINDER = 3
    CONE = 4
    CAPSULE = 5
    TORUS = 6
    ELLIPSOID = 7
    HALFSPACE = 8
    MESH = 9
    IMPLICIT = 10
    TRANSFORM = 20
    STRETCH = 21
    MIRROR = 22
    TWIST = 23
    ELONGATE = 24
    OFFSET = 25
    SHELL = 26
    UNION = 30
    INTERSECTION = 31
    DIFFERENCE = 32
    EXTRUDE = 40
    TAPERED_EXTRUDE = 41
    REVOLVE = 42
    LOFT = 43
    CIRCLE = 100
    RECT = 101
    POLYGON = 102
    SEGMENT = 103
    ELLIPSE = 104
    HALFPLANE = 105
    TRANSFORM2 = 120
    STRETCH2 = 121
    MIRROR2 = 122
    OFFSET2 = 123
    OUTLINE2 = 124
    UNION2 = 130
    INTERSECTION2 = 131
    DIFFERENCE2 = 132
    # Field nodes (custom fields): read only by other field nodes and IMPLICIT.
    CONST = 200
    COORD = 201
    ADD = 202
    SUB = 203
    MUL = 204
    DIV = 205
    NEG = 206
    ABS = 207
    SQUARE = 208
    SQRT = 209
    SIN = 210
    COS = 211
    EXP = 212
    LOG = 213
    POWC = 214
    ATAN2 = 215
    MIN = 216
    MAX = 217
    CLAMP = 218
    LENGTH = 219
    TRIWAVE = 220
    EVAL_AT = 221
    SMIN = 222
    SMAX = 223


# Operations whose opcode depends only on the shape's dimension.
_DIMENSION_OPCODES = {
    3: {"transform": Opcode.TRANSFORM, "stretch": Opcode.STRETCH, "offset": Opcode.OFFSET,
        "union": Opcode.UNION, "intersection": Opcode.INTERSECTION, "difference": Opcode.DIFFERENCE},
    2: {"transform": Opcode.TRANSFORM2, "stretch": Opcode.STRETCH2, "offset": Opcode.OFFSET2,
        "union": Opcode.UNION2, "intersection": Opcode.INTERSECTION2, "difference": Opcode.DIFFERENCE2},
}
_SHELL_MODES = {"inside": 0, "outside": 1, "center": 2}
_NUMBER_WORDS = {2: "two", 3: "three"}


# --------------------------------------------------------------------------- arguments

def _finite(value, name):
    try:
        number = float(value)
    except (TypeError, ValueError):
        raise SDFError(f"{name} must be a number.") from None
    if not math.isfinite(number):
        raise SDFError(f"{name} must be finite.")
    return number


def _positive(value, name):
    number = _finite(value, name)
    if number <= 0:
        raise SDFError(f"{name} must be greater than zero.")
    return number


def _nonnegative(value, name):
    number = _finite(value, name)
    if number < 0:
        raise SDFError(f"{name} must not be negative.")
    return number


def _whole(value, name, low, high, message=None):
    """An integer in [low, high]; integral floats such as 4.0 are accepted,
    booleans are not."""
    if isinstance(value, (bool, np.bool_)):
        raise SDFError(f"{name} must be a whole number, not a boolean.")
    number = _finite(value, name)
    if not number.is_integer():
        raise SDFError(f"{name} must be a whole number.")
    if not low <= number <= high:
        raise SDFError(message or f"{name} must be between {low} and {high}.")
    return int(number)


# The kernel's thread pool (libdispatch) does not survive fork(): a process
# forked after the kernel was loaded uses one thread, which needs no pool.
_forked_child = False


def _after_fork_in_child():
    global _forked_child
    if _library is not None:
        _forked_child = True


if hasattr(os, "register_at_fork"):
    os.register_at_fork(after_in_child=_after_fork_in_child)


def _threads(value):
    threads = _whole(value, "threads", 0, _INT32_MAX)
    return 1 if _forked_child else threads


def _is_scalar(value):
    try:
        return np.ndim(value) == 0
    except ValueError:  # ragged nested sequences
        return False


def _vector(value, size, name):
    try:
        array = np.asarray(value, dtype=float)
    except (TypeError, ValueError):
        array = None
    if array is None or array.ndim == 0 or array.size != size or not np.isfinite(array).all():
        raise SDFError(f"{name} must contain {size} finite numbers.")
    return array.reshape(size)


def _components(args, size, name):
    """Accept f((x, y[, z])) or f(x, y[, z])."""
    if len(args) == 1:
        return _vector(args[0], size, name)
    if len(args) == size:
        return _vector(args, size, name)
    raise SDFError(f"{name} takes a {size}D vector or {_NUMBER_WORDS[size]} numbers.")


def _size(value, size, name):
    """Positive extents: one number for every axis, or one per axis."""
    if _is_scalar(value):
        return np.full(size, _positive(value, name))
    array = _vector(value, size, name)
    if np.any(array <= 0):
        raise SDFError(f"{name} components must be greater than zero.")
    return array


def _unit(value, size, name):
    array = _vector(value, size, name)
    length = float(np.linalg.norm(array))
    if length < 1e-12:
        raise SDFError(f"{name} must not be zero.")
    return array / length


_AXES = {"x": (1.0, 0.0, 0.0), "y": (0.0, 1.0, 0.0), "z": (0.0, 0.0, 1.0),
         "-x": (-1.0, 0.0, 0.0), "-y": (0.0, -1.0, 0.0), "-z": (0.0, 0.0, -1.0)}


def _axis(value, name="axis"):
    if isinstance(value, str):
        key = value.strip().lower()
        if key not in _AXES:
            raise SDFError(f"{name} must be 'x', 'y', 'z' or a 3D vector.")
        return np.array(_AXES[key])
    return _unit(value, 3, name)


_QUARTER_TURNS = ((1.0, 0.0), (0.0, 1.0), (-1.0, 0.0), (0.0, -1.0))


def _cos_sin(degrees):
    """cos and sin of an angle in degrees, exact at multiples of 90 degrees so
    axis-aligned placements produce exact rotation matrices."""
    quarters = degrees / 90.0
    nearest = round(quarters)
    if abs(quarters - nearest) < 1e-12:
        return _QUARTER_TURNS[nearest % 4]
    theta = math.radians(degrees)
    return math.cos(theta), math.sin(theta)


def rotation_matrix(angle, axis=(0.0, 0.0, 1.0)):
    """Right-handed rotation by `angle` degrees about `axis` as a 3x3 array."""
    x, y, z = _axis(axis)
    c, s = _cos_sin(_finite(angle, "angle"))
    t = 1.0 - c
    return np.array([[t * x * x + c, t * x * y - s * z, t * x * z + s * y],
                     [t * x * y + s * z, t * y * y + c, t * y * z - s * x],
                     [t * x * z - s * y, t * y * z + s * x, t * z * z + c]])


def _rotation_between(a, b):
    """The smallest rotation taking direction a to direction b."""
    a = _unit(a, 3, "direction")
    b = _unit(b, 3, "direction")
    cross = np.cross(a, b)
    dot = float(np.clip(np.dot(a, b), -1.0, 1.0))
    if np.linalg.norm(cross) < 1e-12:
        if dot > 0:
            return np.eye(3)
        helper = np.array([1.0, 0.0, 0.0]) if abs(a[0]) < 0.9 else np.array([0.0, 1.0, 0.0])
        return rotation_matrix(180.0, np.cross(a, helper))
    return rotation_matrix(math.degrees(math.acos(dot)), cross)


def _affine(matrix, name):
    """The linear part and translation of a finite 4x4 or 3x4 affine matrix."""
    try:
        array = np.asarray(matrix, dtype=float)
    except (TypeError, ValueError):
        array = None
    if array is None or array.shape not in ((4, 4), (3, 4)) or not np.isfinite(array).all():
        raise SDFError(f"{name} needs a finite 4x4 or 3x4 matrix.")
    if array.shape == (4, 4) and not np.allclose(array[3], (0.0, 0.0, 0.0, 1.0), rtol=0.0, atol=1e-12):
        raise SDFError(f"{name} needs an affine matrix whose last row is 0, 0, 0, 1.")
    return array[:3, :3], array[:3, 3]


# ------------------------------------------------------------------ bounding boxes

class Bounds:
    """An axis-aligned box with corners `lo` and `hi`; components may be infinite."""
    __slots__ = ("lo", "hi")

    def __init__(self, lo, hi):
        self.lo = np.array(lo, dtype=float).reshape(-1)
        self.hi = np.array(hi, dtype=float).reshape(-1)
        if self.lo.shape != self.hi.shape:
            raise SDFError("Bounds corners must have the same dimension.")

    @classmethod
    def infinite(cls, dimension=3):
        return cls(np.full(dimension, -math.inf), np.full(dimension, math.inf))

    @property
    def finite(self):
        return bool(np.isfinite(self.lo).all() and np.isfinite(self.hi).all())

    @property
    def empty(self):
        return bool(np.any(self.lo > self.hi))

    @property
    def size(self):
        return self.hi - self.lo

    @property
    def center(self):
        return (self.lo + self.hi) / 2

    def union(self, other):
        return Bounds(np.minimum(self.lo, other.lo), np.maximum(self.hi, other.hi))

    def intersection(self, other):
        return Bounds(np.maximum(self.lo, other.lo), np.minimum(self.hi, other.hi))

    def expanded(self, amount):
        return Bounds(self.lo - amount, self.hi + amount)

    def corners(self):
        """Every corner of the box, shape (2**dimension, dimension)."""
        return np.array(list(itertools.product(*zip(self.lo, self.hi))))

    def as_list(self):
        return [self.lo.tolist(), self.hi.tolist()]

    def __repr__(self):
        return f"Bounds({self.lo.tolist()}, {self.hi.tolist()})"


def _as_bounds(value, name="bounds"):
    if isinstance(value, Bounds):
        return value
    try:
        lo, hi = value
    except (TypeError, ValueError):
        raise SDFError(f"{name} must be a Bounds or a (lo, hi) pair.") from None
    return Bounds(lo, hi)


# ======================================================================== shapes

class _Shape:
    """One immutable graph node: an RSDF opcode, its parameters and child shapes.

    SDF3 (solids) and SDF2 (XY profiles) share these operations; create
    shapes with the constructor functions rather than directly.
    """
    __slots__ = ("_opcode", "_params", "_children")
    dimension = 0

    def __init__(self, opcode, params=(), children=()):
        opcode = Opcode(opcode)
        params = tuple(map(float, params))
        if not all(map(math.isfinite, params)):
            raise SDFError(f"{opcode.name} parameters must be finite.")
        children = tuple(children)
        kinds = (_Shape, _FieldNode) if opcode == Opcode.IMPLICIT else _Shape
        if not all(isinstance(child, kinds) for child in children):
            raise SDFError(f"{opcode.name} children must be shapes.")
        object.__setattr__(self, "_opcode", opcode)
        object.__setattr__(self, "_params", params)
        object.__setattr__(self, "_children", children)

    def __setattr__(self, name, value):
        raise AttributeError("Shapes are immutable; every operation returns a new shape.")

    def __delattr__(self, name):
        raise AttributeError("Shapes are immutable; every operation returns a new shape.")

    def __copy__(self):
        return self

    def __deepcopy__(self, memo):
        return self

    @property
    def opcode(self):
        return self._opcode

    @property
    def params(self):
        return self._params

    @property
    def children(self):
        return self._children

    def __repr__(self):
        return f"<{type(self).__name__} {self._opcode.name}>"

    # -- native queries -----------------------------------------------------
    def evaluate(self, points, *, threads=0):
        """Signed distances at points of shape (..., dimension); a float for one point."""
        return _graph_for(self).evaluate(self, points, threads=threads)

    __call__ = evaluate

    def contains(self, points):
        """True where points lie strictly inside."""
        return self.evaluate(points) < 0

    @property
    def bounds(self):
        """Conservative axis-aligned bounds (components may be infinite)."""
        return _graph_for(self).bounds(self)

    @property
    def lipschitz(self):
        """A Lipschitz bound of the field (1 for a distance bound)."""
        return _graph_for(self).lipschitz(self)

    # -- booleans -----------------------------------------------------------
    def union(self, *others, smooth=0.0):
        return union(self, *others, smooth=smooth)

    def intersection(self, *others, smooth=0.0):
        return intersection(self, *others, smooth=smooth)

    def difference(self, *others, smooth=0.0):
        return difference(self, *others, smooth=smooth)

    def __or__(self, other):
        return union(self, other)

    def __and__(self, other):
        return intersection(self, other)

    def __sub__(self, other):
        return difference(self, other)

    # -- transforms and modifiers --------------------------------------------
    def translate(self, *offset):
        """Move by an offset vector (or its components)."""
        size = self.dimension
        return _transformed(self, np.eye(size), _components(offset, size, "translate offset"))

    moved = translate

    def scale(self, factor, center=None):
        """Scale uniformly (a number) or per axis (one factor per axis) about
        `center` (default origin). Per-axis scaling keeps a distance bound by
        scaling distances with the smallest factor."""
        size = self.dimension
        center = np.zeros(size) if center is None else _vector(center, size, "center")
        if _is_scalar(factor):
            factor = _positive(factor, "scale factor")
            return _transformed(self, np.eye(size), center - factor * center, factor)
        factors = _size(factor, size, "scale factors")
        if np.all(factors == factors[0]):
            return self.scale(float(factors[0]), center)
        centred = _transformed(self, np.eye(size), -center)
        stretched = type(self)(_DIMENSION_OPCODES[size]["stretch"], factors, (centred,))
        return _transformed(stretched, np.eye(size), center)

    def offset(self, distance):
        """Grow (positive) or shrink (negative) the boundary by `distance`."""
        opcode = _DIMENSION_OPCODES[self.dimension]["offset"]
        return type(self)(opcode, (_finite(distance, "offset distance"),), (self,))

    def linear_array(self, count, step):
        """`count` copies, each displaced by `step` from the previous."""
        count = _whole(count, "linear_array count", 1, _MAX_COPIES,
                       "linear_array count must be between 1 and 10000.")
        step = _vector(step, self.dimension, "array step")
        return union(*[self.translate(step * i) for i in range(count)])

    def grid_array(self, counts, steps):
        """A rectangular array with one count and one step per axis."""
        size = self.dimension
        message = (f"grid_array needs {_NUMBER_WORDS[size]} counts of at least 1 "
                   f"(at most {_MAX_COPIES} copies).")
        try:
            counts = [_whole(count, "grid_array count", 1, _MAX_COPIES, message) for count in counts]
        except TypeError:
            raise SDFError(message) from None
        if len(counts) != size or math.prod(counts) > _MAX_COPIES:
            raise SDFError(message)
        steps = _vector(steps, size, "grid steps")
        return union(*[self.translate(steps * np.array(index))
                       for index in itertools.product(*map(range, counts))])


def _polar_step(count, angle):
    """Angle between polar copies: a full turn divides evenly, a partial arc
    puts copies at both ends."""
    if abs(abs(angle) - 360.0) < 1e-9:
        return angle / count
    return angle / max(count - 1, 1)


class SDF3(_Shape):
    """A solid described by a signed distance bound in 3D."""
    __slots__ = ()
    dimension = 3

    # -- transforms ---------------------------------------------------------
    def rotate(self, angle, axis=(0.0, 0.0, 1.0), center=(0.0, 0.0, 0.0)):
        """Rotate by `angle` degrees about `axis` through `center`."""
        center = _vector(center, 3, "center")
        rotation = rotation_matrix(angle, axis)
        return _transformed(self, rotation, center - rotation @ center)

    def rotate_x(self, angle, center=(0.0, 0.0, 0.0)):
        return self.rotate(angle, "x", center)

    def rotate_y(self, angle, center=(0.0, 0.0, 0.0)):
        return self.rotate(angle, "y", center)

    def rotate_z(self, angle, center=(0.0, 0.0, 0.0)):
        return self.rotate(angle, "z", center)

    def orient(self, direction, up=(0.0, 0.0, 1.0)):
        """Rotate so the shape's `up` direction points along `direction`."""
        return _transformed(self, _rotation_between(up, direction), np.zeros(3))

    def transform(self, matrix):
        """Apply a 4x4 (or 3x4) matrix made of a rotation, a uniform positive
        scale and a translation."""
        linear, offset = _affine(matrix, "transform")
        scale = float(np.cbrt(abs(np.linalg.det(linear))))
        if scale < 1e-9:
            raise SDFError("transform matrix is singular.")
        rotation = linear / scale
        if not np.allclose(rotation.T @ rotation, np.eye(3), atol=1e-6) or np.linalg.det(rotation) < 0:
            raise SDFError("transform must be a rotation with uniform positive scale; "
                           "use mirror() for reflections.")
        if not np.allclose(rotation.T @ rotation, np.eye(3), rtol=0.0, atol=1e-12):
            # Snap a nearly orthonormal input to the nearest rotation.
            u, _, vt = np.linalg.svd(rotation)
            rotation = u @ vt
        return _transformed(self, rotation, offset, scale)

    def mirror(self, normal=(1.0, 0.0, 0.0), point=(0.0, 0.0, 0.0)):
        """Reflect through the plane with `normal` passing through `point`."""
        normal = _axis(normal, "mirror normal")
        distance = normal @ _vector(point, 3, "mirror point")
        return SDF3(Opcode.MIRROR, (*normal, distance), (self,))

    def mirrored_copy(self, normal=(1.0, 0.0, 0.0), point=(0.0, 0.0, 0.0)):
        """This shape plus its reflection."""
        return union(self, self.mirror(normal, point))

    # -- modifiers ----------------------------------------------------------
    def shell(self, thickness, side="inside"):
        """A hollow wall of `thickness` following the surface.

        side: 'inside' keeps the outer surface, 'outside' keeps the inner
        surface, 'center' straddles it.
        """
        thickness = _positive(thickness, "shell thickness")
        mode = _SHELL_MODES.get(side) if isinstance(side, str) else None
        if mode is None:
            raise SDFError("shell side must be 'inside', 'outside' or 'center'.")
        return SDF3(Opcode.SHELL, (thickness, mode), (self,))

    def elongate(self, *amount):
        """Split the shape at its centre planes and insert straight sections
        2 * amount long along each axis."""
        return SDF3(Opcode.ELONGATE, np.abs(_components(amount, 3, "elongation")), (self,))

    def twist(self, degrees_per_mm):
        """Twist about the Z axis by the given rate; the shape must be bounded."""
        rate = math.radians(_finite(degrees_per_mm, "twist rate"))
        return SDF3(Opcode.TWIST, (rate,), (self,))

    def cut(self, normal=(0.0, 0.0, 1.0), point=(0.0, 0.0, 0.0), keep="below"):
        """Keep the side of a plane: 'below' (opposite the normal) or 'above'."""
        if keep not in ("below", "above"):
            raise SDFError("cut keep must be 'below' or 'above'.")
        normal = _axis(normal, "cut normal")
        return intersection(self, half_space(normal if keep == "below" else -normal, point))

    def polar_array(self, count, axis=(0.0, 0.0, 1.0), center=(0.0, 0.0, 0.0), angle=360.0):
        """`count` copies rotated about `axis` through `center`, spread over `angle` degrees."""
        count = _whole(count, "polar_array count", 1, _MAX_COPIES,
                       "polar_array count must be between 1 and 10000.")
        step = _polar_step(count, _finite(angle, "polar_array angle"))
        return union(*[self.rotate(step * i, axis, center) for i in range(count)])

    # -- custom fields --------------------------------------------------------
    @property
    def field(self):
        """The shape's signed distance as a Field, for custom field expressions."""
        return Field(self)

    def at(self, *point):
        """A Field: this shape's signed distance evaluated at the point (fx, fy,
        fz), three fields or numbers (or one triple), instead of at (x, y, z).
        Warping the evaluation point bends, twists or tapers the shape;
        implicit() makes the result a solid."""
        if len(point) == 1 and isinstance(point[0], (list, tuple)):
            point = tuple(point[0])
        if len(point) != 3:
            raise SDFError("at() takes three coordinates (fx, fy, fz): fields or numbers.")
        return Field(_FieldNode(Opcode.EVAL_AT, (), (self, *(_field_operand(value, "at()") for value in point))))

    # -- native queries -----------------------------------------------------
    def gradient(self, points, eps=None, *, threads=0):
        """Unit outward normals (normalised field gradients) at points (..., 3).
        `eps` is the finite-difference step; None lets the kernel choose."""
        return _graph_for(self).gradient(self, points, eps, threads=threads)

    def raycast(self, origins, directions, max_distance=1e5, tolerance=1e-4, max_steps=512, *, threads=0):
        """Sphere-trace rays from `origins` along `directions` (both (..., 3),
        broadcast together). Returns (hits, distances): whether each ray met
        the surface within `max_distance`, and the distance along the
        normalised direction. One ray returns a (bool, float) pair."""
        return _graph_for(self).raycast(self, origins, directions, max_distance, tolerance, max_steps,
                                        threads=threads)

    def mesh(self, cell_size=None, crease_angle=35.0, max_cells=None, threads=0, bounds=None, simplify=None,
             tolerance=None, max_depth=3, max_triangles=None):
        """A closed triangle mesh of the surface (adaptive sharp-feature dual
        contouring: cells split where the mesh strays from the surface).

        cell_size: base grid spacing in mm (default: auto_cell_size of the region).
        crease_angle: degrees between adjacent faces that keep a sharp edge.
        max_cells: fail beyond this many narrow-band cells and stop refining
            beyond this many octree cells (default: kernel limit).
        threads: worker threads (0 = all cores).
        bounds: the region to mesh, a Bounds or (lo, hi); needed for unbounded shapes.
        simplify: a tolerance in mm: merge triangles (keeping sharp edges and
            corners) while the field stays within it at the moved vertices and
            at samples of every changed triangle; between samples, where the
            full mesh already deviates more, and for fields that are not exact
            distances, the surface can lie somewhat further away. None or 0
            keeps the full mesh.
        tolerance: the estimated distance (mm) from the surface that cells
            split to meet at samples of every triangle (default: cell_size / 20).
        max_depth: how many times a cell may halve (0: a uniform grid).
        max_triangles: refinement stops before its estimate exceeds this many
            triangles (default: no limit).
        The mesh's stats report max_deviation, the largest estimated distance
        sampled on its triangles.
        """
        return _graph_for(self).mesh(self, cell_size, crease_angle, max_cells, threads, bounds, simplify,
                                     tolerance=tolerance, max_depth=max_depth, max_triangles=max_triangles)


class SDF2(_Shape):
    """A planar region in the XY plane described by a signed distance bound."""
    __slots__ = ()
    dimension = 2

    # -- transforms ---------------------------------------------------------
    def rotate(self, angle, center=(0.0, 0.0)):
        """Rotate by `angle` degrees counter-clockwise about `center`."""
        center = _vector(center, 2, "center")
        c, s = _cos_sin(_finite(angle, "angle"))
        rotation = np.array([[c, -s], [s, c]])
        return _transformed(self, rotation, center - rotation @ center)

    def mirror(self, normal=(1.0, 0.0), point=(0.0, 0.0)):
        """Reflect across the line with `normal` passing through `point`."""
        normal = _unit(normal, 2, "mirror normal")
        distance = normal @ _vector(point, 2, "mirror point")
        return SDF2(Opcode.MIRROR2, (*normal, distance), (self,))

    def mirrored_copy(self, normal=(1.0, 0.0), point=(0.0, 0.0)):
        """This profile plus its reflection."""
        return union(self, self.mirror(normal, point))

    # -- modifiers ----------------------------------------------------------
    def outline(self, thickness):
        """A band of `thickness` centred on the boundary."""
        return SDF2(Opcode.OUTLINE2, (_positive(thickness, "outline thickness"),), (self,))

    def polar_array(self, count, center=(0.0, 0.0), angle=360.0):
        """`count` copies rotated about `center`, spread over `angle` degrees."""
        count = _whole(count, "polar_array count", 1, _MAX_COPIES,
                       "polar_array count must be between 1 and 10000.")
        step = _polar_step(count, _finite(angle, "polar_array angle"))
        return union(*[self.rotate(step * i, center) for i in range(count)])

    # -- to 3D ---------------------------------------------------------------
    def extrude(self, height, center=False, taper=0.0):
        """Extrude along +Z from z=0 (or centred on z=0). `taper` is the profile
        offset at the top in mm (negative narrows)."""
        height = _positive(height, "extrude height")
        z0, z1 = (-height / 2, height / 2) if center else (0.0, height)
        taper = _finite(taper, "taper")
        if taper:
            return SDF3(Opcode.TAPERED_EXTRUDE, (z0, z1, taper), (self,))
        return SDF3(Opcode.EXTRUDE, (z0, z1), (self,))

    def extrude_between(self, z0, z1):
        """Extrude between the planes z = z0 and z = z1."""
        z0, z1 = _finite(z0, "z0"), _finite(z1, "z1")
        if z1 <= z0:
            raise SDFError("extrude_between needs z1 > z0.")
        return SDF3(Opcode.EXTRUDE, (z0, z1), (self,))

    def revolve(self, offset=0.0):
        """Revolve about Z: profile x is the radius (plus `offset`), profile y is Z."""
        return SDF3(Opcode.REVOLVE, (_finite(offset, "revolve offset"),), (self,))

    def loft_to(self, other, height, center=False):
        """Linear blend from this profile at the bottom to `other` at the top."""
        if not isinstance(other, SDF2):
            raise SDFError("loft_to needs another 2D profile.")
        height = _positive(height, "loft height")
        z0, z1 = (-height / 2, height / 2) if center else (0.0, height)
        return SDF3(Opcode.LOFT, (z0, z1), (self, other))


def _transformed(shape, rotation, offset, scale=1.0):
    """`shape` placed by p_world = scale * rotation @ p_local + offset.

    A transform of a transform composes into a single node, and identity
    transforms return `shape` itself, so placement chains never nest.
    """
    size = shape.dimension
    opcode = _DIMENSION_OPCODES[size]["transform"]
    if _is_identity(rotation, offset, scale):
        return shape
    if shape.opcode == opcode:
        params = shape.params
        inner_rotation = np.reshape(params[:size * size], (size, size))
        inner_offset = np.array(params[size * size:-1])
        offset = scale * (rotation @ inner_offset) + offset
        rotation = rotation @ inner_rotation
        scale = scale * params[-1]
        shape = shape.children[0]
        if _is_identity(rotation, offset, scale):
            return shape
    return type(shape)(opcode, (*rotation.ravel(), *offset, scale), (shape,))


def _is_identity(rotation, offset, scale):
    return scale == 1.0 and not np.any(offset) and np.array_equal(rotation, np.eye(len(offset)))


# ================================================================ constructors (3D)

def sphere(radius, center=(0.0, 0.0, 0.0)):
    """A sphere of `radius` centred at `center`."""
    node = SDF3(Opcode.SPHERE, (_positive(radius, "sphere radius"),))
    return _transformed(node, np.eye(3), _vector(center, 3, "center"))


def box(size, center=(0.0, 0.0, 0.0), radius=0.0):
    """A box with edge lengths `size` (number or (x, y, z)) centred at `center`.
    `radius` rounds every edge and corner."""
    size = _size(size, 3, "box size")
    radius = _nonnegative(radius, "box radius")
    if radius > np.min(size) / 2 + 1e-12:
        raise SDFError("box radius must not exceed half the smallest box dimension.")
    half = size / 2
    node = SDF3(Opcode.BOX, (*half, min(radius, float(np.min(half)))))
    return _transformed(node, np.eye(3), _vector(center, 3, "center"))


def cylinder(radius, height, center=(0.0, 0.0, 0.0), axis=(0.0, 0.0, 1.0)):
    """A capped cylinder centred at `center`, its height along `axis`."""
    node = SDF3(Opcode.CYLINDER, (_positive(radius, "cylinder radius"),
                                  _positive(height, "cylinder height") / 2))
    return _place(node, axis, center)


def cone(radius1, radius2, height, center=(0.0, 0.0, 0.0), axis=(0.0, 0.0, 1.0)):
    """A capped cone/frustum centred at `center`: `radius1` at the bottom
    (toward -axis), `radius2` at the top. Either radius may be zero."""
    r1 = _nonnegative(radius1, "cone radius1")
    r2 = _nonnegative(radius2, "cone radius2")
    if r1 == 0 and r2 == 0:
        raise SDFError("cone needs a nonzero radius.")
    node = SDF3(Opcode.CONE, (r1, r2, _positive(height, "cone height") / 2))
    return _place(node, axis, center)


def capsule(a, b, radius):
    """All points within `radius` of the segment from `a` to `b`."""
    return SDF3(Opcode.CAPSULE, (*_vector(a, 3, "capsule start"), *_vector(b, 3, "capsule end"),
                                 _positive(radius, "capsule radius")))


def torus(major_radius, minor_radius, center=(0.0, 0.0, 0.0), axis=(0.0, 0.0, 1.0)):
    """A torus around `axis`: ring radius `major_radius`, tube radius `minor_radius`."""
    node = SDF3(Opcode.TORUS, (_positive(major_radius, "torus major radius"),
                               _positive(minor_radius, "torus minor radius")))
    return _place(node, axis, center)


def ellipsoid(radii, center=(0.0, 0.0, 0.0)):
    """An axis-aligned ellipsoid with semi-axes `radii` (x, y, z)."""
    node = SDF3(Opcode.ELLIPSOID, _size(radii, 3, "ellipsoid radii"))
    return _transformed(node, np.eye(3), _vector(center, 3, "center"))


def half_space(normal=(0.0, 0.0, 1.0), point=(0.0, 0.0, 0.0)):
    """Everything on the side of the plane opposite `normal` (unbounded; intersect it)."""
    normal = _axis(normal, "half_space normal")
    return SDF3(Opcode.HALFSPACE, (*normal, normal @ _vector(point, 3, "half_space point")))


def _place(node, axis, center):
    """Turn a Z-aligned primitive toward `axis` and move it to `center`."""
    rotation = _rotation_between((0.0, 0.0, 1.0), _axis(axis))
    return _transformed(node, rotation, _vector(center, 3, "center"))


# ================================================================ constructors (2D)

def circle(radius, center=(0.0, 0.0)):
    """A disc of `radius` centred at `center`."""
    node = SDF2(Opcode.CIRCLE, (_positive(radius, "circle radius"),))
    return _transformed(node, np.eye(2), _vector(center, 2, "center"))


def rect(size, center=(0.0, 0.0), radius=0.0):
    """A rectangle with side lengths `size` centred at `center`; `radius` rounds its corners."""
    size = _size(size, 2, "rect size")
    radius = _nonnegative(radius, "rect radius")
    if radius > np.min(size) / 2 + 1e-12:
        raise SDFError("rect radius must not exceed half the smaller side.")
    half = size / 2
    node = SDF2(Opcode.RECT, (*half, min(radius, float(np.min(half)))))
    return _transformed(node, np.eye(2), _vector(center, 2, "center"))


_POLYGON_PREFIX = re.compile(r"^.*?node \d+ \(POLYGON\): ")
_POINT_REFERENCE = re.compile(r"\bpoint (\d+)\b")


def _polygon_failure(node, kept):
    """The kernel's objection to a POLYGON node, naming the caller's points
    (`kept` maps the node's points to them), or None. Without the kernel,
    shapes still build and decoding checks the outline later."""
    try:
        _native()
    except SDFError:
        return None
    try:
        Graph(node).close()
    except SDFError as error:
        message = str(error)
        if not _POLYGON_PREFIX.match(message):
            raise
        detail = _POINT_REFERENCE.sub(lambda match: f"point {kept[int(match.group(1))]}",
                                      _POLYGON_PREFIX.sub("", message, count=1))
        if "not simple" in detail:
            detail += ". Check the order of the points (numbered from 0 as given)"
        return detail + "."
    return None


def polygon(points):
    """A simple polygon through `points` [(x, y), ...] (closed automatically,
    either orientation). Its outline must not cross, touch or overlap itself."""
    try:
        array = np.asarray(points, dtype=float)
    except (TypeError, ValueError):
        array = None
    if (array is None or array.ndim != 2 or array.shape[1] != 2 or len(array) < 3
            or not np.isfinite(array).all()):
        raise SDFError("polygon needs at least three finite (x, y) points.")
    # Repeated consecutive points, including a closing copy of the first
    # point, would form zero-length edges, which have no inside side.
    kept = np.flatnonzero(np.linalg.norm(array - np.roll(array, -1, axis=0), axis=1) > 1e-9)
    array = array[kept]
    if len(array) < 3:
        raise SDFError("polygon needs at least three distinct points.")
    if len(array) > _MAX_PARAMS // 2:
        raise SDFError(f"polygon supports at most {_MAX_PARAMS // 2} points.")
    node = SDF2(Opcode.POLYGON, array.ravel().tolist())
    # A crossing outline is reported first: its lobes can cancel its area.
    failure = _polygon_failure(node, kept)
    if failure is not None and "not simple" in failure:
        raise SDFError(failure)
    area = 0.5 * np.sum(array[:, 0] * np.roll(array[:, 1], -1) - np.roll(array[:, 0], -1) * array[:, 1])
    if abs(area) < 1e-12:
        raise SDFError("polygon has no area.")
    if failure is not None:
        raise SDFError(failure)
    return node


def regular_polygon(sides, radius, center=(0.0, 0.0), angle=0.0):
    """A regular polygon with `sides` corners on a circle of `radius`; the first
    corner is at `angle` degrees."""
    sides = _whole(sides, "regular_polygon sides", 3, _MAX_PARAMS // 2,
                   f"regular_polygon needs between 3 and {_MAX_PARAMS // 2} sides.")
    radius = _positive(radius, "regular_polygon radius")
    start = math.radians(_finite(angle, "angle"))
    theta = start + np.arange(sides) * 2 * math.pi / sides
    return polygon(np.column_stack([radius * np.cos(theta), radius * np.sin(theta)])
                   + _vector(center, 2, "center"))


def slot(a, b, radius):
    """A 2D stadium: points within `radius` of the segment a-b."""
    return SDF2(Opcode.SEGMENT, (*_vector(a, 2, "slot start"), *_vector(b, 2, "slot end"),
                                 _positive(radius, "slot radius")))


def ellipse(radii, center=(0.0, 0.0)):
    """An axis-aligned ellipse with semi-axes `radii` (x, y)."""
    node = SDF2(Opcode.ELLIPSE, _size(radii, 2, "ellipse radii"))
    return _transformed(node, np.eye(2), _vector(center, 2, "center"))


def half_plane(normal=(0.0, 1.0), point=(0.0, 0.0)):
    """Everything on the side of the line opposite `normal` (unbounded)."""
    normal = _unit(normal, 2, "half_plane normal")
    return SDF2(Opcode.HALFPLANE, (*normal, normal @ _vector(point, 2, "half_plane point")))


def arc_points(center, radius, start_angle, end_angle, segments=None):
    """Points along a circular arc (degrees), for building polygons; shape (segments + 1, 2)."""
    center = _vector(center, 2, "center")
    radius = _positive(radius, "arc radius")
    a0 = math.radians(_finite(start_angle, "start_angle"))
    a1 = math.radians(_finite(end_angle, "end_angle"))
    if segments is None:
        segments = max(8, int(abs(a1 - a0) / (2 * math.pi) * 128))
    segments = _whole(segments, "arc segments", 1, _MAX_PARAMS // 2)
    t = np.linspace(a0, a1, segments + 1)
    return np.column_stack([center[0] + radius * np.cos(t), center[1] + radius * np.sin(t)])


def involute_gear(teeth, module, pressure_angle=20.0, clearance=0.25, backlash=0.0, points_per_flank=12):
    """A 2D spur gear profile centred at the origin (pitch radius = module * teeth / 2)."""
    teeth = _whole(teeth, "teeth", 6, 400, "involute_gear needs between 6 and 400 teeth.")
    m = _positive(module, "module")
    alpha = math.radians(_finite(pressure_angle, "pressure_angle"))
    clearance = _nonnegative(clearance, "clearance")
    flank_points = _whole(points_per_flank, "points_per_flank", 2, _MAX_PARAMS // 2)
    pitch = m * teeth / 2
    base = pitch * math.cos(alpha)
    outer = pitch + m
    root = max(pitch - (1 + clearance) * m, base * 0.5)
    # Half tooth angular thickness at the pitch circle, minus backlash.
    half_tooth = math.pi / (2 * teeth) - _finite(backlash, "backlash") / (2 * pitch)

    def involute(radius):
        if radius <= base:
            return 0.0
        a = math.acos(base / radius)
        return math.tan(a) - a

    inv_pitch = involute(pitch)
    radii = np.linspace(max(base, root), outer, flank_points)
    profile = []
    for tooth in range(teeth):
        center_angle = 2 * math.pi * tooth / teeth
        right = [(r, center_angle - (half_tooth + inv_pitch - involute(r))) for r in radii]
        left = [(r, center_angle + (half_tooth + inv_pitch - involute(r))) for r in radii[::-1]]
        gap_start = center_angle + (half_tooth + inv_pitch)
        gap_end = center_angle + 2 * math.pi / teeth - (half_tooth + inv_pitch)
        root_arc = [(root, a) for a in np.linspace(gap_start, gap_end, 4)]
        if radii[0] > root:
            right.insert(0, (root, right[0][1]))
            left.append((root, left[-1][1]))
        profile.extend(right + left + root_arc)
    return polygon([(r * math.cos(a), r * math.sin(a)) for r, a in profile])


# ================================================================== combinators

def _flatten(items):
    flat = []
    for item in items:
        if isinstance(item, _Shape):
            flat.append(item)
        elif isinstance(item, Field):
            raise SDFError(_FIELD_IS_NOT_A_SHAPE)
        elif isinstance(item, collections.abc.Iterable) and not isinstance(item, (str, bytes)):
            flat.extend(_flatten(item))
        else:
            raise SDFError(f"Expected a shape, got {type(item).__name__}.")
    return flat


def _same_dimension(shapes):
    dimension = shapes[0].dimension
    if any(shape.dimension != dimension for shape in shapes):
        raise SDFError("Cannot combine 2D profiles with 3D solids; extrude or revolve the profile first.")


def _boolean(operation, operands, smooth):
    """A boolean node over `operands`: a left fold with blend radius `smooth`.
    Operand lists beyond the RSDF child limit nest as left folds, which
    leaves the field unchanged."""
    kind = type(operands[0])
    opcode = _DIMENSION_OPCODES[kind.dimension][operation]
    while len(operands) > _MAX_CHILDREN:
        operands = [kind(opcode, (smooth,), operands[:_MAX_CHILDREN]), *operands[_MAX_CHILDREN:]]
    return kind(opcode, (smooth,), operands)


def union(*shapes, smooth=0.0):
    """Everything inside any of the shapes; `smooth` blends with that radius."""
    shapes = _flatten(shapes)
    if not shapes:
        raise SDFError("union needs at least one shape.")
    smooth = _nonnegative(smooth, "smooth")
    _same_dimension(shapes)
    return shapes[0] if len(shapes) == 1 else _boolean("union", shapes, smooth)


def intersection(*shapes, smooth=0.0):
    """Only the space inside all of the shapes."""
    shapes = _flatten(shapes)
    if not shapes:
        raise SDFError("intersection needs at least one shape.")
    smooth = _nonnegative(smooth, "smooth")
    _same_dimension(shapes)
    return shapes[0] if len(shapes) == 1 else _boolean("intersection", shapes, smooth)


def difference(base, *cutters, smooth=0.0):
    """`base` with every cutter removed."""
    if isinstance(base, Field):
        raise SDFError(_FIELD_IS_NOT_A_SHAPE)
    if not isinstance(base, _Shape):
        raise SDFError(f"Expected a shape, got {type(base).__name__}.")
    cutters = _flatten(cutters)
    smooth = _nonnegative(smooth, "smooth")
    if not cutters:
        return base
    _same_dimension([base, *cutters])
    return _boolean("difference", [base, *cutters], smooth)


def _solid(shape, operation):
    if not isinstance(shape, SDF3):
        raise SDFError(f"{operation} needs a 3D solid.")


def gradient(shape, points, eps=None):
    """Unit outward normals of a solid at points (..., 3); see SDF3.gradient."""
    _solid(shape, "gradient")
    return shape.gradient(points, eps)


def raycast(shape, origins, directions, max_distance=1e5, tolerance=1e-4, max_steps=512):
    """Sphere-trace rays against a solid; see SDF3.raycast."""
    _solid(shape, "raycast")
    return shape.raycast(origins, directions, max_distance, tolerance, max_steps)


# ================================================================ custom fields
#
# A Field is a scalar function of the point traced from Python arithmetic over
# the symbolic coordinates x, y, z, numbers and shapes' signed distances
# (SDF-DESIGN.md, "Custom fields"). Every operation records a native
# expression node (opcodes 200-223); nothing runs in Python per point. A field
# becomes a solid only through implicit(field, bounds): loading that shape
# proves an upper bound L of |grad field| over the bounds and uses the field
# divided by L, a distance bound every algorithm accepts.

_FIELD_IS_NOT_A_SHAPE = ("A custom field is not a shape: make it a solid with implicit(field, bounds) before "
                         "combining, transforming or using it as a part (use shape.field for arithmetic with a "
                         "shape's signed distance).")
_FIELD_IS_SYMBOLIC = ("A custom field has no value while the program runs (x, y and z are symbolic), so it "
                      "cannot be compared or tested with if/and/or: use minimum(), maximum() or clamp() "
                      "instead (for example maximum(a, b) instead of `a if a > b else b`).")
_FIELD_HAS_NO_NUMBER = ("A custom field has no number value while the program runs (x, y and z are symbolic): "
                        "use radsdf's sin, cos, exp, log, sqrt, atan2, minimum, maximum, clamp, length and "
                        "triwave, which accept fields, instead of math functions.")
_FIELD_JUMPS = ("% and // jump, so custom fields cannot use them; triwave(value, period) is a continuous "
                "repeating pattern.")


class _FieldNode:
    """One immutable node of a custom field (opcodes 200-223): its opcode,
    parameters and children (field nodes or 3D shapes), exactly as RSDF
    stores them. Programs work with Field, which wraps these nodes."""
    __slots__ = ("_opcode", "_params", "_children")
    dimension = 3

    def __init__(self, opcode, params=(), children=()):
        opcode = Opcode(opcode)
        params = tuple(map(float, params))
        if not all(map(math.isfinite, params)):
            raise SDFError("Field constants must be finite numbers.")
        children = tuple(children)
        for child in children:
            if isinstance(child, SDF2):
                raise SDFError("A custom field reads 3D shapes only; extrude or revolve the profile first.")
            if not isinstance(child, (_FieldNode, SDF3)):
                raise SDFError(f"{opcode.name} children must be fields or 3D shapes.")
        object.__setattr__(self, "_opcode", opcode)
        object.__setattr__(self, "_params", params)
        object.__setattr__(self, "_children", children)

    def __setattr__(self, name, value):
        raise AttributeError("Fields are immutable; every operation returns a new field.")

    def __delattr__(self, name):
        raise AttributeError("Fields are immutable; every operation returns a new field.")

    def __copy__(self):
        return self

    def __deepcopy__(self, memo):
        return self

    @property
    def opcode(self):
        return self._opcode

    @property
    def params(self):
        return self._params

    @property
    def children(self):
        return self._children

    def __repr__(self):
        return f"<field node {self._opcode.name}>"


def _constant(value):
    return _FieldNode(Opcode.CONST, (value,))


def _field_number(value, operation):
    """A finite float operand of a field operation (not a field)."""
    if isinstance(value, _Shape):
        raise SDFError(f"{operation}: a shape is not a field; use shape.field for its signed distance, or "
                       "shape.at(fx, fy, fz) to evaluate it at a warped point.")
    if isinstance(value, (bool, np.bool_)):
        raise SDFError(f"{operation} needs numbers or fields, not booleans.")
    if isinstance(value, numbers.Real) or (isinstance(value, np.ndarray) and value.ndim == 0
                                           and value.dtype.kind in "fiu"):
        number = float(value)
        if not math.isfinite(number):
            raise SDFError(f"{operation}: field constants must be finite numbers.")
        return number
    if isinstance(value, (np.ndarray, list, tuple)):
        raise SDFError(f"{operation}: fields combine with single numbers, not arrays or sequences.")
    raise SDFError(f"{operation} needs numbers or fields, got {type(value).__name__}.")


def _field_operand(value, operation):
    """The node of a field operand: a Field's node, or a CONST node for a number."""
    if isinstance(value, Field):
        return value._node
    return _constant(_field_number(value, operation))


def _field_operation(opcode, operands, operation, params=()):
    return Field(_FieldNode(opcode, params, [_field_operand(value, operation) for value in operands]))


class Field:
    """A scalar function of the point, traced from Python: build fields from
    the coordinates x, y, z, numbers, shape.field (a shape's signed distance)
    and shape.at(fx, fy, fz), with + - * /, unary -, abs(), ** (a number
    exponent) and sin, cos, exp, log, sqrt, atan2, minimum, maximum, clamp,
    length and triwave. Running the program records the expression once; the
    kernel evaluates it natively. implicit(field, bounds) makes it a solid
    (inside where the field is negative). Fields have no value while the
    program runs: comparisons and `if field` raise SDFError (use minimum,
    maximum or clamp)."""
    __slots__ = ("_node",)

    def __init__(self, node):
        if not isinstance(node, (_FieldNode, SDF3)):
            raise SDFError("Build fields from x, y, z, numbers, shape.field and shape.at().")
        object.__setattr__(self, "_node", node)

    def __setattr__(self, name, value):
        raise AttributeError("Fields are immutable; every operation returns a new field.")

    def __delattr__(self, name):
        raise AttributeError("Fields are immutable; every operation returns a new field.")

    def __copy__(self):
        return self

    def __deepcopy__(self, memo):
        return self

    def __repr__(self):
        return f"<Field {self._node.opcode.name}>"

    # -- arithmetic -----------------------------------------------------------
    def __add__(self, other):
        return _field_operation(Opcode.ADD, (self, other), "+")

    def __radd__(self, other):
        return _field_operation(Opcode.ADD, (other, self), "+")

    def __sub__(self, other):
        return _field_operation(Opcode.SUB, (self, other), "-")

    def __rsub__(self, other):
        return _field_operation(Opcode.SUB, (other, self), "-")

    def __mul__(self, other):
        return _field_operation(Opcode.MUL, (self, other), "*")

    def __rmul__(self, other):
        return _field_operation(Opcode.MUL, (other, self), "*")

    def __truediv__(self, other):
        if not isinstance(other, Field) and _field_number(other, "/") == 0:
            raise SDFError("A custom field divided by zero.")
        return _field_operation(Opcode.DIV, (self, other), "/")

    def __rtruediv__(self, other):
        return _field_operation(Opcode.DIV, (other, self), "/")

    def __neg__(self):
        return Field(_FieldNode(Opcode.NEG, (), (self._node,)))

    def __pos__(self):
        return self

    def __abs__(self):
        return Field(_FieldNode(Opcode.ABS, (), (self._node,)))

    def __pow__(self, exponent, modulo=None):
        if modulo is not None:
            raise SDFError("pow() of a custom field takes no modulus.")
        if isinstance(exponent, Field):
            raise SDFError("The exponent of a custom field must be a number; use exp(b * log(a)) for a field "
                           "exponent.")
        power = _field_number(exponent, "**")
        if abs(power) > 2.0 ** 52:  # beyond, e - 1 rounds and loses the derivative's parity
            raise SDFError("The exponent of a custom field must be at most 2**52 in magnitude.")
        if power == 0:
            return Field(_constant(1.0))
        if power == 1:
            return self
        if power == 2:
            return Field(_FieldNode(Opcode.SQUARE, (), (self._node,)))
        return Field(_FieldNode(Opcode.POWC, (power,), (self._node,)))

    def __rpow__(self, base):
        base = _field_number(base, "**")
        if base <= 0:
            raise SDFError("A number raised to a custom field's power needs a positive base.")
        return exp(self * math.log(base))

    def __mod__(self, other):
        raise SDFError(_FIELD_JUMPS)

    __rmod__ = __floordiv__ = __rfloordiv__ = __divmod__ = __rdivmod__ = __mod__

    # -- no values while tracing -------------------------------------------------
    def __bool__(self):
        raise SDFError(_FIELD_IS_SYMBOLIC)

    def _compare(self, other):
        raise SDFError(_FIELD_IS_SYMBOLIC)

    __lt__ = __le__ = __gt__ = __ge__ = __eq__ = __ne__ = _compare
    __hash__ = object.__hash__

    def __float__(self):
        raise SDFError(_FIELD_HAS_NO_NUMBER)

    __int__ = __index__ = __complex__ = __float__

    def __array_ufunc__(self, ufunc, method, *inputs, **kwargs):
        """numpy functions with a radsdf counterpart (np.sin(x), 2.0 * x with
        a numpy number, ...) build fields; others raise SDFError."""
        operation = _FIELD_UFUNCS.get(ufunc.__name__) if method == "__call__" and not kwargs else None
        if operation is None:
            raise SDFError(f"numpy's {ufunc.__name__} cannot take a custom field. Fields support + - * / ** and "
                           "abs(), and radsdf's sin, cos, exp, log, sqrt, atan2, minimum, maximum, clamp, length "
                           "and triwave.")
        values = [value if isinstance(value, Field) else _field_number(value, f"np.{ufunc.__name__}")
                  for value in inputs]
        return operation(*values)

    # -- native queries -----------------------------------------------------
    def evaluate(self, points, *, threads=0):
        """Field values at points of shape (..., 3); a float for one point."""
        return _graph_for(self._node).evaluate(self._node, points, threads=threads)

    __call__ = evaluate


x = Field(_FieldNode(Opcode.COORD, (0.0,)))
y = Field(_FieldNode(Opcode.COORD, (1.0,)))
z = Field(_FieldNode(Opcode.COORD, (2.0,)))


def _not_a_shape(value, name):
    if isinstance(value, _Shape):
        raise SDFError(f"{name}() takes numbers or fields; use shape.field for a shape's signed distance.")


def _unary(opcode, scalar, array, name, doc):
    """A function of one number, array or field."""
    def function(value):
        if isinstance(value, Field):
            return Field(_FieldNode(opcode, (), (value._node,)))
        _not_a_shape(value, name)
        return scalar(value) if np.ndim(value) == 0 else array(value)
    function.__name__ = function.__qualname__ = name
    function.__doc__ = doc
    return function


sin = _unary(Opcode.SIN, math.sin, np.sin, "sin", "Sine of radians: a number, an array or a field.")
cos = _unary(Opcode.COS, math.cos, np.cos, "cos", "Cosine of radians: a number, an array or a field.")
exp = _unary(Opcode.EXP, math.exp, np.exp, "exp", "e to the power of a number, an array or a field.")
log = _unary(Opcode.LOG, math.log, np.log, "log", "Natural logarithm of a number, an array or a field (the "
                                                  "field must stay positive within its bounds).")
sqrt = _unary(Opcode.SQRT, math.sqrt, np.sqrt, "sqrt", "Square root of a number, an array or a field (a field "
                                                       "must not reach 0 where it changes; use length() for "
                                                       "distances).")


def atan2(y_value, x_value):
    """The angle of (x_value, y_value) in radians, in (-pi, pi]: numbers,
    arrays or fields. Across the negative x axis the angle jumps by 2 pi, so
    a field may use it there only through sin or cos of whole multiples."""
    if isinstance(y_value, Field) or isinstance(x_value, Field):
        return _field_operation(Opcode.ATAN2, (y_value, x_value), "atan2()")
    _not_a_shape(y_value, "atan2")
    _not_a_shape(x_value, "atan2")
    if np.ndim(y_value) == 0 and np.ndim(x_value) == 0:
        return math.atan2(y_value, x_value)
    return np.arctan2(y_value, x_value)


_EXTREMUM_GROUP = 1024


def _smooth_extremum(values, name, k, lowest):
    """The left fold of the polynomial smooth minimum (or maximum) with
    radius k, as union(..., smooth=k) blends shapes: SMIN/SMAX nodes, whose
    slope bound is the larger of their operands' (a + b would add them)."""
    if any(isinstance(value, Field) for value in values):
        node = _field_operand(values[0], f"{name}()")
        for value in values[1:]:
            node = _FieldNode(Opcode.SMIN if lowest else Opcode.SMAX, (k,), (node, _field_operand(value, f"{name}()")))
        return Field(node)
    for value in values:
        _not_a_shape(value, name)
    sign = 1.0 if lowest else -1.0
    result = values[0]
    for value in values[1:]:
        u, v = np.multiply(sign, result), np.multiply(sign, value)
        h = np.maximum(k - np.abs(u - v), 0.0) / k
        result = sign * (np.minimum(u, v) - h * h * k / 4)
    return float(result) if np.ndim(result) == 0 else result


def _extremum(opcode, values, name, scalar, array, smooth=0):
    if len(values) == 1 and isinstance(values[0], (list, tuple)):
        values = tuple(values[0])
    if not values:
        raise SDFError(f"{name}() needs at least one value.")
    smooth = _field_number(smooth, f"{name}() smooth")
    if smooth < 0:
        raise SDFError(f"{name}() smooth must not be negative.")
    if len(values) == 1 and isinstance(values[0], Field):
        return values[0]
    if smooth > 0 and len(values) > 1:
        return _smooth_extremum(values, name, smooth, opcode == Opcode.MIN)
    if any(isinstance(value, Field) for value in values):
        nodes = [_field_operand(value, f"{name}()") for value in values]
        # Wider extrema nest in groups (min and max are associative): a field
        # keeps every operand of one node alive until it runs, so groups keep
        # few values alive at a time.
        while len(nodes) > _EXTREMUM_GROUP:
            nodes = [_FieldNode(opcode, (), nodes[i:i + _EXTREMUM_GROUP]) if len(nodes) - i > 1 else nodes[i]
                     for i in range(0, len(nodes), _EXTREMUM_GROUP)]
        return Field(_FieldNode(opcode, (), nodes))
    for value in values:
        _not_a_shape(value, name)
    if all(np.ndim(value) == 0 for value in values):
        return scalar(values)
    return functools.reduce(array, values)


def minimum(*values, smooth=0):
    """The smallest of numbers, arrays or fields (also of one list of them).
    smooth=k blends them with radius k, as union(..., smooth=k) blends shapes."""
    return _extremum(Opcode.MIN, values, "minimum", min, np.minimum, smooth)


def maximum(*values, smooth=0):
    """The largest of numbers, arrays or fields (also of one list of them).
    smooth=k blends them with radius k, as intersection(..., smooth=k) does."""
    return _extremum(Opcode.MAX, values, "maximum", max, np.maximum, smooth)


def clamp(value, lo, hi):
    """`value` limited to [lo, hi]: numbers, arrays or fields."""
    if any(isinstance(item, Field) for item in (value, lo, hi)):
        if isinstance(value, Field) and not isinstance(lo, Field) and not isinstance(hi, Field):
            low, high = _field_number(lo, "clamp() lo"), _field_number(hi, "clamp() hi")
            if low > high:
                raise SDFError("clamp() needs lo <= hi.")
            return Field(_FieldNode(Opcode.CLAMP, (low, high), (value._node,)))
        return maximum(lo, minimum(value, hi))
    for item in (value, lo, hi):
        _not_a_shape(item, "clamp")
    if np.ndim(value) == 0 and np.ndim(lo) == 0 and np.ndim(hi) == 0:
        if lo > hi:
            raise SDFError("clamp() needs lo <= hi.")
        return min(max(value, lo), hi)
    return np.clip(value, lo, hi)


def length(*components):
    """Euclidean length sqrt(a**2 + b**2 + ...) of numbers, arrays or fields
    (also of one list of them). length(x, y, z) - r is a sphere's exact
    distance; unlike sqrt, its slope stays bounded at 0."""
    if len(components) == 1 and isinstance(components[0], (list, tuple)):
        components = tuple(components[0])
    if not components:
        raise SDFError("length() needs at least one component.")
    if any(isinstance(value, Field) for value in components):
        if len(components) > _MAX_CHILDREN:
            raise SDFError(f"length() takes at most {_MAX_CHILDREN} components.")
        return _field_operation(Opcode.LENGTH, components, "length()")
    for value in components:
        _not_a_shape(value, "length")
    if all(np.ndim(value) == 0 for value in components):
        return math.hypot(*components)
    return np.sqrt(sum(np.square(value) for value in components))


def triwave(value, period):
    """A continuous triangle wave of slope 1: |(value mod 2*period) - period|,
    between 0 and `period` (a number from 1e-300 to 1e300) and repeating
    every 2*period. Numbers, arrays or fields; use it for repeating patterns
    instead of %."""
    period = _positive(period, "triwave period")
    if not 1e-300 <= period <= 1e300:
        raise SDFError("triwave period must lie between 1e-300 and 1e300.")
    if isinstance(value, Field):
        return Field(_FieldNode(Opcode.TRIWAVE, (period,), (value._node,)))
    _not_a_shape(value, "triwave")
    twice = 2 * period  # the remainder by fmod is exact, as in the kernel
    if np.ndim(value) == 0:
        rest = math.fmod(value, twice)
        return abs((rest + twice if rest < 0 else rest) - period)
    rest = np.fmod(value, twice)
    return np.abs(np.where(rest < 0, rest + twice, rest) - period)


_FIELD_UFUNCS = {
    "add": operator.add, "subtract": operator.sub, "multiply": operator.mul, "true_divide": operator.truediv,
    "negative": operator.neg, "positive": operator.pos, "absolute": abs, "fabs": abs,
    "square": lambda value: value ** 2, "power": operator.pow, "sqrt": sqrt, "sin": sin, "cos": cos,
    "exp": exp, "log": log, "arctan2": atan2, "minimum": minimum, "maximum": maximum, "fmin": minimum,
    "fmax": maximum, "hypot": length,
}


# The design program's line of each implicit() call, for messages naming it:
# {id(shape): (shape, line)}. Builds clear it before running their program.
_implicit_lines = {}


def _program_line():
    """The innermost line of the design program ("<RAD model>") calling in."""
    frame = sys._getframe(1)
    while frame is not None:
        if frame.f_code.co_filename == "<RAD model>":
            return frame.f_lineno
        frame = frame.f_back
    return None


def _implicit_line(shape):
    entry = _implicit_lines.get(id(shape))
    return entry[1] if entry is not None and entry[0] is shape else None


def _field_root(value, operation):
    """The node a custom field operation reads: a Field's node or a 3D shape."""
    if isinstance(value, Field):
        return value._node
    if isinstance(value, SDF3):
        return value
    if isinstance(value, SDF2):
        raise SDFError(f"{operation} needs a 3D field; extrude or revolve the profile first.")
    raise SDFError(f"{operation} needs a custom field built from x, y, z (or a 3D shape), got "
                   f"{type(value).__name__}.")


def _field_box(bounds, operation):
    box = _as_bounds(bounds, f"{operation} bounds")
    if box.lo.shape != (3,):
        raise SDFError(f"{operation} bounds must be 3D: ((x0, y0, z0), (x1, y1, z1)).")
    if not box.finite or np.any(np.abs(box.lo) > 1e150) or np.any(np.abs(box.hi) > 1e150):
        raise SDFError(f"{operation} bounds must be finite, within 1e150 of the origin.")
    if np.any(box.lo >= box.hi):
        raise SDFError(f"{operation} bounds need lo < hi along every axis.")
    return box


def implicit(field, bounds):
    """A solid from a custom field: inside where the field is negative.

    The field is used only within `bounds` ((x0, y0, z0), (x1, y1, z1)) or a
    Bounds, which become the solid's bounds: keep them a little larger than
    the solid, and more where an outward offset, shell or blend of it reaches
    past them (there it sees the box, not the field). When the solid loads,
    the kernel proves an upper bound L of |grad field| over the bounds and
    uses max(field / L, signed distance to the bounds box) inside them and,
    outside, hypot(field(c) / L, d) where field(c) > 0, else d (c the nearest
    point of the box, d the distance to it): a distance bound like every other
    shape's. Where |grad field| is below L that underestimates the distance,
    so the solid's offsets, shells, blends and clearances come out larger
    there (by up to L / |grad field|; inward offsets remove more) and probed
    distances are smaller than true ones (verify_field and rad_verify report
    L and how close the field is to a true distance). A field without a
    finite slope bound (a division by a value that reaches 0, sqrt or log at
    0, atan2 at its pole) fails the build, naming the region.
    """
    node = _field_root(field, "implicit")
    box = _field_box(bounds, "implicit")
    shape = SDF3(Opcode.IMPLICIT, (*box.lo, *box.hi), (node,))
    line = _program_line()
    if line is not None:
        _implicit_lines[id(shape)] = (shape, line)
    return shape


def _analysis_box(lo, hi):
    """The box IMPLICIT analyses: its bounds grown by 1e-6 of their largest
    extent (the kernel's arithmetic, so the bound is the same)."""
    lo, hi = [float(value) for value in lo], [float(value) for value in hi]
    grow = 1e-6 * max(0.0, *(high - low for low, high in zip(lo, hi)))
    return [value - grow for value in lo], [value + grow for value in hi]


_NODE_REFERENCE = re.compile(r"node (\d+) \(([A-Z0-9_]+)\)")
_FIELD_CONSTRUCTS = {
    "CONST": "a number", "COORD": "a coordinate", "ADD": "a sum (+)", "SUB": "a difference (-)",
    "MUL": "a product (*)", "DIV": "a division (/)", "NEG": "a negation (-)", "ABS": "abs()",
    "SQUARE": "a square (** 2)", "SQRT": "sqrt()", "SIN": "sin()", "COS": "cos()", "EXP": "exp()",
    "LOG": "log()", "POWC": "a power (**)", "ATAN2": "atan2()", "MIN": "minimum()", "MAX": "maximum()",
    "CLAMP": "clamp()", "LENGTH": "length()", "TRIWAVE": "triwave()", "EVAL_AT": "shape.at()",
    "SMIN": "minimum(..., smooth=k)", "SMAX": "maximum(..., smooth=k)",
}


def _describe_field_nodes(message, order):
    """`message` with the kernel's references to field and IMPLICIT nodes
    ("node 12 (DIV)") replaced by the Python constructs that made them;
    `order` lists the encoded nodes by index."""
    def describe(match):
        index, name = int(match.group(1)), match.group(2)
        node = order[index] if index < len(order) else None
        if node is None or node.opcode.name != name:
            return match.group(0)
        if node.opcode == Opcode.IMPLICIT:
            line = _implicit_line(node)
            return "implicit()" + (f" at line {line}" if line is not None else "")
        if isinstance(node, _FieldNode):
            return _FIELD_CONSTRUCTS.get(name, match.group(0))
        return match.group(0)
    return _NODE_REFERENCE.sub(describe, message)


def _field_report(report, order=None):
    """A verify_field dictionary of a native rsdf_field_report; with `order`
    (the encoded nodes) the failure names Python constructs, not nodes."""
    ok = bool(report.ok)
    result = {"ok": ok, "bound": float(report.bound) if ok else math.inf,
              "value_range": (float(report.value_lo), float(report.value_hi)),
              "gradient": {"min": float(report.gradient_min), "median": float(report.gradient_median),
                           "max": float(report.gradient_max)},
              "quality": float(report.quality), "boxes": int(report.boxes), "samples": int(report.samples)}
    if not ok:
        message = report.message.decode("utf-8", "replace")
        result["message"] = message if order is None else _describe_field_nodes(message, order)
        result["region"] = (tuple(report.failed_lo), tuple(report.failed_hi))
    return result


def _analyze_field(node, lo, hi):
    """verify_field's analysis of a field root (a field node or a 3D shape,
    read through a one-child MAX) over the box implicit(node, (lo, hi))
    analyses at load."""
    root = node if isinstance(node, _FieldNode) else _FieldNode(Opcode.MAX, (), (node,))
    data, roots, order = _encode([root], with_order=True)
    try:
        graph = Graph.from_bytes(data, roots)
    except SDFError as error:
        raise SDFError(_describe_field_nodes(str(error), order)) from None
    with graph:
        report = graph.analyze_field(roots[0], (lo, hi))
    if "message" in report:
        report["message"] = _describe_field_nodes(report["message"], order)
    return report


def verify_field(field, bounds=None):
    """The proof implicit(field, bounds) makes when its solid loads, as a
    dictionary: `ok` (a finite slope bound exists), `bound` (the proven upper
    bound L of |grad field|, inf without one), `value_range` (proven
    (lo, hi) of the field), `gradient` (min, median and max of |grad field|
    at up to 4,096 sample points, fewer for costly fields), `quality`
    (median / bound: 1 for a true distance, lower where implicit()
    underestimates distances), `boxes` and `samples`; without a bound also
    `message` and `region` ((lo, hi) where it fails). An implicit() shape
    may be passed without bounds.
    """
    if bounds is None and isinstance(field, SDF3) and field.opcode == Opcode.IMPLICIT:
        return _analyze_field(field.children[0], field.params[:3], field.params[3:])
    if bounds is None:
        raise SDFError("verify_field needs the bounds the field is used in (or an implicit() shape).")
    box = _field_box(bounds, "verify_field")
    return _analyze_field(_field_root(field, "verify_field"), box.lo, box.hi)


__all__ += ["Field", "x", "y", "z", "sin", "cos", "exp", "log", "sqrt", "atan2", "minimum", "maximum", "clamp",
            "length", "triwave", "implicit", "verify_field"]


# ================================================================== RSDF encoding

def _node_of(value):
    """The graph node of a shape, a Field or a field node (None otherwise)."""
    if isinstance(value, (_Shape, _FieldNode)):
        return value
    if isinstance(value, Field):
        return value._node
    return None


def _root_list(shapes):
    """The graph nodes of a shape or field, or of a list of them."""
    single = _node_of(shapes)
    if single is not None:
        return [single]
    try:
        roots = list(shapes)
    except TypeError:
        raise SDFError(f"Expected a shape or shapes, got {type(shapes).__name__}.") from None
    if not roots:
        raise SDFError("Expected at least one shape.")
    nodes = [_node_of(root) for root in roots]
    for root, node in zip(roots, nodes):
        if node is None:
            raise SDFError(f"Expected a shape, got {type(root).__name__}.")
    return nodes


_EXACT_FOLDS = frozenset({Opcode.UNION, Opcode.INTERSECTION, Opcode.DIFFERENCE,
                          Opcode.UNION2, Opcode.INTERSECTION2, Opcode.DIFFERENCE2})


def _use_counts(roots):
    """How many times each node is referenced, by parents or as a root."""
    uses = collections.Counter(id(root) for root in roots)
    seen, stack = set(), list(roots)
    while stack:
        node = stack.pop()
        if id(node) not in seen:
            seen.add(id(node))
            uses.update(id(child) for child in node.children)
            stack.extend(node.children)
    return uses


def _record_children(node, uses):
    """The children `node` is written with.

    An exact boolean whose first child is the same exact boolean, used
    nowhere else, continues that child's left fold; `s = s - hole` loops are
    written as one record instead of a chain as deep as the loop. The fold,
    and with it the field, is unchanged.
    """
    children = node.children
    if node.opcode not in _EXACT_FOLDS or node.params != (0.0,):
        return children
    levels, count, first = [], len(children), children[0]
    while (first.opcode == node.opcode and first.params == (0.0,) and uses[id(first)] == 1
           and count - 1 + len(first.children) <= _MAX_CHILDREN):
        levels.append(children[1:])
        children = first.children
        count += len(children) - 1
        first = children[0]
    return [*children, *itertools.chain.from_iterable(reversed(levels))] if levels else children


def _encode(roots, with_order=False):
    """RSDF bytes of every node reachable from `roots`, and each root's index
    (plus the nodes in record order with `with_order`)."""
    uses = _use_counts(roots)
    # Iterative depth-first post-order keeps children before parents, visits
    # shared nodes once and survives graphs deeper than the recursion limit.
    order, index, written = [], {}, {}
    for root in roots:
        stack = [root]
        while stack:
            node = stack[-1]
            if id(node) in index:
                stack.pop()
                continue
            children = written.get(id(node))
            if children is None:
                children = written[id(node)] = _record_children(node, uses)
            pending = [child for child in children if id(child) not in index]
            if pending:
                stack.extend(reversed(pending))
            else:
                stack.pop()
                index[id(node)] = len(order)
                order.append(node)
    if len(order) > _MAX_NODES:
        raise SDFError(f"The shape graph has {len(order)} nodes; at most {_MAX_NODES} are supported.")
    records = [struct.pack("<4sIII", b"RSDF", FORMAT_VERSION, len(order), 0)]
    size = len(records[0])
    for node in order:
        children = [index[id(child)] for child in written[id(node)]]
        params = node.params
        if isinstance(params, np.ndarray):  # MESH parameters stay packed little-endian doubles
            record = struct.pack(f"<4I{len(children)}I", node.opcode, len(children), len(params), 0,
                                 *children) + params.tobytes()
        else:
            record = struct.pack(f"<4I{len(children)}I{len(params)}d", node.opcode, len(children),
                                 len(params), 0, *children, *params)
        size += len(record)
        if size > _MAX_BYTES:
            raise SDFError("The shape graph exceeds the 512 MB RSDF size limit.")
        records.append(record)
    if with_order:
        return b"".join(records), [index[id(root)] for root in roots], order
    return b"".join(records), [index[id(root)] for root in roots]


def serialize(roots):
    """Encode a shape or field, or a list of them, as an RSDF version 1 graph.

    Returns (data, root_indices): the bytes and the node index of each root,
    in order. A node shared by several parents or roots is written once, and
    unshared chains of the same exact boolean are written as one record.
    """
    return _encode(_root_list(roots))


# ================================================================== native kernel

_LIBRARY_NAME = "libRADSDF.dylib"
_LIBRARY_VARIABLE = "RAD_SDF_LIBRARY"
_ABI_VERSION = 3
_REFINE_STEPS = 6  # the kernel's documented default crossing refinement
_MIN_CREASE_ANGLE = 1e-9  # degrees; a crease angle of 0 creases every edge


_MAX_DEPTH = 6  # the kernel's deepest octree refinement


class _MeshOptions(ctypes.Structure):
    _fields_ = [("cell_size", ctypes.c_double), ("crease_angle", ctypes.c_double),
                ("max_cells", ctypes.c_uint64), ("threads", ctypes.c_int32),
                ("refine_steps", ctypes.c_int32), ("bounds_lo", ctypes.c_double * 3),
                ("bounds_hi", ctypes.c_double * 3), ("bounds_valid", ctypes.c_int32),
                ("max_depth", ctypes.c_int32), ("tolerance", ctypes.c_double),
                ("max_triangles", ctypes.c_uint64)]


class _MeshStats(ctypes.Structure):
    _fields_ = [("vertex_count", ctypes.c_uint64), ("triangle_count", ctypes.c_uint64),
                ("crease_segment_count", ctypes.c_uint64), ("active_cells", ctypes.c_uint64),
                ("boundary_edges", ctypes.c_uint64), ("nonmanifold_edges", ctypes.c_uint64),
                ("components", ctypes.c_uint64), ("cavities", ctypes.c_uint64), ("volume", ctypes.c_double),
                ("area", ctypes.c_double), ("cavity_volume", ctypes.c_double),
                ("centroid", ctypes.c_double * 3), ("bounds_lo", ctypes.c_double * 3),
                ("bounds_hi", ctypes.c_double * 3), ("cell_size", ctypes.c_double),
                ("seconds", ctypes.c_double), ("max_deviation", ctypes.c_double), ("tolerance", ctypes.c_double),
                ("refined_cells", ctypes.c_uint64), ("balanced_cells", ctypes.c_uint64),
                ("flagged_triangles", ctypes.c_uint64), ("max_level", ctypes.c_uint32),
                ("passes", ctypes.c_uint32)]


class _SimplifyOptions(ctypes.Structure):
    _fields_ = [("tolerance", ctypes.c_double), ("threads", ctypes.c_int32), ("reserved", ctypes.c_int32)]


class _FieldReport(ctypes.Structure):
    _fields_ = [("bound", ctypes.c_double), ("value_lo", ctypes.c_double), ("value_hi", ctypes.c_double),
                ("gradient_min", ctypes.c_double), ("gradient_median", ctypes.c_double),
                ("gradient_max", ctypes.c_double), ("quality", ctypes.c_double), ("boxes", ctypes.c_uint64),
                ("samples", ctypes.c_uint64), ("ok", ctypes.c_int32), ("failed_node", ctypes.c_uint32),
                ("failed_lo", ctypes.c_double * 3), ("failed_hi", ctypes.c_double * 3),
                ("message", ctypes.c_char * 512)]


class _GraphHandle(ctypes.Structure):
    """Opaque rsdf_graph."""


class _MeshHandle(ctypes.Structure):
    """Opaque rsdf_mesh."""


_GRAPH = ctypes.POINTER(_GraphHandle)
_MESH = ctypes.POINTER(_MeshHandle)
_DOUBLES = ctypes.POINTER(ctypes.c_double)
_FLOATS = ctypes.POINTER(ctypes.c_float)

# (restype, argtypes) of every function in radsdf.h.
_PROTOTYPES = {
    "rsdf_abi_version": (ctypes.c_uint32, []),
    "rsdf_kernel_version": (ctypes.c_char_p, []),
    "rsdf_last_error": (ctypes.c_char_p, []),
    "rsdf_graph_create": (_GRAPH, [ctypes.c_char_p, ctypes.c_size_t]),
    "rsdf_graph_free": (None, [_GRAPH]),
    "rsdf_graph_node_count": (ctypes.c_uint32, [_GRAPH]),
    "rsdf_node_dimension": (ctypes.c_int, [_GRAPH, ctypes.c_uint32]),
    "rsdf_bounds": (ctypes.c_int, [_GRAPH, ctypes.c_uint32, _DOUBLES, _DOUBLES, _DOUBLES]),
    "rsdf_evaluate": (ctypes.c_int, [_GRAPH, ctypes.c_uint32, _DOUBLES, ctypes.c_uint64, _DOUBLES,
                                     ctypes.c_int32]),
    "rsdf_gradient": (ctypes.c_int, [_GRAPH, ctypes.c_uint32, _DOUBLES, ctypes.c_uint64, ctypes.c_double,
                                     _DOUBLES, ctypes.c_int32]),
    "rsdf_raycast": (ctypes.c_int, [_GRAPH, ctypes.c_uint32, _DOUBLES, _DOUBLES, ctypes.c_uint64,
                                    ctypes.c_double, ctypes.c_double, ctypes.c_int32,
                                    ctypes.POINTER(ctypes.c_uint8), _DOUBLES, ctypes.c_int32]),
    "rsdf_mesh_build": (_MESH, [_GRAPH, ctypes.c_uint32, ctypes.POINTER(_MeshOptions)]),
    "rsdf_mesh_free": (None, [_MESH]),
    "rsdf_mesh_get_stats": (None, [_MESH, ctypes.POINTER(_MeshStats)]),
    "rsdf_mesh_positions": (_FLOATS, [_MESH]),
    "rsdf_mesh_normals": (_FLOATS, [_MESH]),
    "rsdf_mesh_triangles": (ctypes.POINTER(ctypes.c_uint32), [_MESH]),
    "rsdf_mesh_creases": (_FLOATS, [_MESH]),
    "rsdf_mesh_simplify": (_MESH, [_GRAPH, ctypes.c_uint32, _MESH, ctypes.POINTER(_SimplifyOptions)]),
    "rsdf_field_analyze": (ctypes.c_int, [_GRAPH, ctypes.c_uint32, _DOUBLES, _DOUBLES, ctypes.c_uint32,
                                          ctypes.POINTER(_FieldReport)]),
}


def _library_path():
    """RAD_SDF_LIBRARY, else the library next to this module, else the
    repository's SwiftPM build products, release before debug."""
    configured = os.environ.get(_LIBRARY_VARIABLE)
    if configured:
        return Path(configured)
    here = Path(__file__).resolve().parent
    build = here.parent / ".build"
    candidates = [here / _LIBRARY_NAME]
    for configuration in ("release", "debug"):
        # Both SwiftPM build systems link .build/<configuration> to their
        # products; the default one writes them to out/Products/<Configuration>.
        candidates += [build / configuration / _LIBRARY_NAME,
                       build / "out" / "Products" / configuration.capitalize() / _LIBRARY_NAME]
    for candidate in candidates:
        if candidate.is_file():
            return candidate
    raise SDFError(f"The native SDF kernel {_LIBRARY_NAME} is missing (searched "
                   f"{', '.join(map(str, candidates))}); set {_LIBRARY_VARIABLE} to its path.")


def _load_library():
    path = _library_path()
    try:
        library = ctypes.CDLL(str(path))
    except OSError as error:
        raise SDFError(f"The native SDF kernel {path} could not be loaded: {error}") from None
    for name, (restype, argtypes) in _PROTOTYPES.items():
        function = getattr(library, name, None)
        if function is None:
            raise SDFError(f"The native SDF kernel {path} does not export {name}.")
        function.restype = restype
        function.argtypes = argtypes
    version = library.rsdf_abi_version()
    if version != _ABI_VERSION:
        raise SDFError(f"The native SDF kernel {path} has ABI version {version}; "
                       f"this module needs version {_ABI_VERSION}.")
    return library


_library = None
_library_failure = None
_library_lock = threading.Lock()


def _native():
    """The loaded kernel library. Loading happens on first use, so building
    shapes works without it; failures are remembered and re-raised."""
    global _library, _library_failure
    library = _library
    if library is not None:
        return library
    with _library_lock:
        if _library is None and _library_failure is None:
            try:
                _library = _load_library()
            except SDFError as error:
                _library_failure = str(error)
        if _library_failure is not None:
            raise SDFError(_library_failure)
        return _library


def kernel_version():
    """Version string of the native kernel (loads it on first use)."""
    return (_native().rsdf_kernel_version() or b"").decode("utf-8", "replace")


def _kernel_error(operation):
    """SDFError for a failed kernel call, with the kernel's message for this thread."""
    message = _native().rsdf_last_error()
    detail = message.decode("utf-8", "replace") if message else "the kernel gave no reason"
    return SDFError(f"{operation} failed: {detail}")


def _check(status, operation):
    if status != 0:
        raise _kernel_error(operation)


def _doubles(array):
    return array.ctypes.data_as(_DOUBLES)


def _point_batch(values, dimension, name):
    """Points as a contiguous (N, dimension) float64 array, the leading shape
    to restore on results, and whether a single point was given."""
    try:
        array = np.asarray(values, dtype=np.float64)
    except (TypeError, ValueError):
        raise SDFError(f"{name} must be numbers.") from None
    if array.ndim == 0 or array.shape[-1] != dimension:
        raise SDFError(f"{name} must have shape (..., {dimension}).")
    flat = np.ascontiguousarray(array.reshape(-1, dimension))
    if not np.isfinite(flat).all():
        raise SDFError(f"{name} must be finite.")
    return flat, array.shape[:-1], array.ndim == 1


def _copied(pointer, count, dtype):
    """A numpy copy of `count` values of a native mesh buffer."""
    if count == 0:
        return np.empty(0, dtype=dtype)
    if not pointer:
        raise SDFError("Meshing failed: the kernel returned no buffer for a nonempty mesh.")
    return np.ctypeslib.as_array(pointer, shape=(count,)).astype(dtype, copy=True)


class Graph:
    """Shapes compiled into one native kernel graph, shared subgraphs once.

    Query methods take one of the shapes the graph was built from, or a raw
    node index; a graph loaded with from_bytes() takes node indices. Use the
    graph as a context manager, or call close(), to free its native memory
    deterministically; otherwise it is freed when the object is collected.
    Queries may run on several threads at once; close() lets running queries
    finish (the last one frees the graph) and makes later queries fail.
    """
    __slots__ = ("roots", "_targets", "_handle", "_release", "_lock", "_calls", "_closing", "__weakref__")

    def __init__(self, shapes):
        shapes = _root_list(shapes)
        data, roots, order = _encode(shapes, with_order=True)
        try:
            self._load(data, "Building the shape graph")
        except SDFError as error:
            raise SDFError(_describe_field_nodes(str(error), order)) from None
        self.roots = tuple(roots)
        self._targets = {id(shape): (shape, index) for shape, index in zip(shapes, roots)}

    @classmethod
    def from_bytes(cls, data, roots):
        """A graph decoded from RSDF version 1 bytes (any bytes-like buffer),
        such as a build's model.rsdf. `roots` lists node indices, such as
        the build's partRoots; each must name a node of the graph. They
        become `graph.roots`."""
        if not isinstance(data, bytes):
            try:
                data = memoryview(data).tobytes()
            except TypeError:
                raise SDFError(f"RSDF data must be bytes, got {type(data).__name__}.") from None
        if isinstance(roots, (str, bytes)):
            raise SDFError("roots must be a list of node indices.")
        try:
            roots = list(roots)
        except TypeError:
            raise SDFError(f"roots must be a list of node indices, got {type(roots).__name__}.") from None
        graph = cls.__new__(cls)
        graph._load(data, "Loading the RSDF graph")
        graph._targets = {}
        try:
            last = graph.node_count - 1
            graph.roots = tuple(_whole(root, "root node index", 0, last,
                                       f"root node index {root} is not a node of the graph (0 to {last}).")
                                for root in roots)
        except SDFError:
            graph.close()
            raise
        return graph

    def _load(self, data, operation):
        library = _native()
        handle = library.rsdf_graph_create(data, len(data))
        if not handle:
            raise _kernel_error(operation)
        self._handle = handle
        self._lock = threading.Lock()
        self._calls = 0  # native calls running on any thread
        self._closing = False
        self._release = weakref.finalize(self, library.rsdf_graph_free, handle)

    def __enter__(self):
        return self

    def __exit__(self, *exception):
        self.close()

    def close(self):
        """Free the native graph; later queries raise SDFError. Queries
        already running on other threads finish first, and the last of them
        frees the graph."""
        with self._lock:
            self._closing = True
            if self._calls:
                return
        self._release()

    @property
    def closed(self):
        return self._closing or not self._release.alive

    @property
    def node_count(self):
        with self._live() as handle:
            return _native().rsdf_graph_node_count(handle)

    @contextlib.contextmanager
    def _live(self):
        """The native handle, kept alive until the block ends: ctypes releases
        the GIL during calls, so close() on another thread must wait."""
        with self._lock:
            if self._closing or not self._release.alive:
                raise SDFError("The graph has been closed.")
            self._calls += 1
        try:
            yield self._handle
        finally:
            with self._lock:
                self._calls -= 1
                free = self._closing and self._calls == 0
            if free:
                self._release()

    def _target(self, target):
        """(node index, dimension) of a root shape or field, or a raw node index."""
        node = _node_of(target)
        if node is not None:
            entry = self._targets.get(id(node))
            if entry is None or entry[0] is not node:
                raise SDFError("The shape is not one of this graph's roots.")
            return entry[1], node.dimension
        index = _whole(target, "node index", 0, self.node_count - 1)
        with self._live() as handle:
            dimension = _native().rsdf_node_dimension(handle, index)
            if dimension not in (2, 3):
                raise _kernel_error("Reading the node dimension")
        return index, dimension

    def _solid_target(self, target, operation):
        node, dimension = self._target(target)
        if dimension != 3:
            raise SDFError(f"{operation} needs a 3D solid; extrude or revolve the profile first.")
        return node

    def analyze_field(self, target, bounds=None):
        """verify_field's dictionary for an IMPLICIT node of the graph (over
        the box it analyses at load when `bounds` is None, so the bound is
        its L) or a field root (over `bounds` as implicit() analyses it).
        Node references in a failure message are the graph's node indices."""
        node, _ = self._target(target)
        if bounds is None:
            lo = hi = None
        else:
            box = _field_box(bounds, "analyze_field")
            analysis_lo, analysis_hi = _analysis_box(box.lo, box.hi)
            lo, hi = (ctypes.c_double * 3)(*analysis_lo), (ctypes.c_double * 3)(*analysis_hi)
        report = _FieldReport()
        with self._live() as handle:
            _check(_native().rsdf_field_analyze(handle, node, lo, hi, 0, ctypes.byref(report)),
                   "Analyzing the field")
        return _field_report(report)

    def _bounds_and_lipschitz(self, target):
        node, dimension = self._target(target)
        lo, hi, lipschitz = (ctypes.c_double * 3)(), (ctypes.c_double * 3)(), ctypes.c_double()
        with self._live() as handle:
            _check(_native().rsdf_bounds(handle, node, lo, hi, ctypes.byref(lipschitz)), "Computing bounds")
        return Bounds(lo[:dimension], hi[:dimension]), lipschitz.value

    def bounds(self, target):
        """Conservative bounds of `target` (components may be infinite)."""
        return self._bounds_and_lipschitz(target)[0]

    def lipschitz(self, target):
        """A Lipschitz bound of `target`'s field."""
        return self._bounds_and_lipschitz(target)[1]

    def evaluate(self, target, points, *, threads=0):
        """Signed distances of `target` at points (..., dimension); a float for one point."""
        node, dimension = self._target(target)
        flat, leading, single = _point_batch(points, dimension, "points")
        threads = _threads(threads)
        distances = np.empty(len(flat))
        if len(flat):
            with self._live() as handle:
                _check(_native().rsdf_evaluate(handle, node, _doubles(flat), len(flat), _doubles(distances),
                                               threads), "Evaluating the shape")
        return float(distances[0]) if single else distances.reshape(leading)

    def gradient(self, target, points, eps=None, *, threads=0):
        """Unit outward normals of a solid at points (..., 3); see SDF3.gradient."""
        node = self._solid_target(target, "gradient")
        flat, leading, single = _point_batch(points, 3, "points")
        eps = 0.0 if eps is None else _positive(eps, "gradient eps")
        threads = _threads(threads)
        normals = np.empty((len(flat), 3))
        if len(flat):
            with self._live() as handle:
                _check(_native().rsdf_gradient(handle, node, _doubles(flat), len(flat), eps, _doubles(normals),
                                               threads), "Computing gradients")
        return normals[0] if single else normals.reshape(leading + (3,))

    def raycast(self, target, origins, directions, max_distance=1e5, tolerance=1e-4, max_steps=512, *,
                threads=0):
        """Sphere-trace rays against a solid; see SDF3.raycast."""
        node = self._solid_target(target, "raycast")
        try:
            origins, directions = np.broadcast_arrays(np.asarray(origins, dtype=np.float64),
                                                      np.asarray(directions, dtype=np.float64))
        except (TypeError, ValueError):
            raise SDFError("ray origins and directions must be (..., 3) arrays that broadcast together.") from None
        flat_origins, leading, single = _point_batch(origins, 3, "ray origins")
        flat_directions = _point_batch(directions, 3, "ray directions")[0]
        # Scaled by their largest component (the kernel normalises them):
        # squaring components beyond about 1e154 would overflow.
        scales = np.max(np.abs(flat_directions), axis=1) if len(flat_directions) else np.zeros(0)
        if np.any(scales == 0):
            raise SDFError("ray directions must not be zero.")
        flat_directions = np.ascontiguousarray(flat_directions / scales[:, None])
        max_distance = _positive(max_distance, "max_distance")
        tolerance = _positive(tolerance, "tolerance")
        max_steps = _whole(max_steps, "max_steps", 1, _INT32_MAX)
        threads = _threads(threads)
        count = len(flat_origins)
        hits = np.zeros(count, dtype=np.uint8)
        distances = np.zeros(count)
        if count:
            with self._live() as handle:
                _check(_native().rsdf_raycast(handle, node, _doubles(flat_origins), _doubles(flat_directions),
                                              count, max_distance, tolerance, max_steps,
                                              hits.ctypes.data_as(ctypes.POINTER(ctypes.c_uint8)),
                                              _doubles(distances), threads), "Casting rays")
        hits = hits.astype(bool)
        if single:
            return bool(hits[0]), float(distances[0])
        return hits.reshape(leading), distances.reshape(leading)

    def mesh(self, target, cell_size=None, crease_angle=35.0, max_cells=None, threads=0, bounds=None,
             simplify=None, *, tolerance=None, max_depth=3, max_triangles=None):
        """A closed triangle mesh of a solid; see SDF3.mesh. A simplified
        mesh's stats add `unsimplified_triangles` and `simplify_tolerance`."""
        simplify = None if simplify is None else _nonnegative(simplify, "simplify")
        return self.native_mesh(target, cell_size, crease_angle, max_cells, threads, bounds, tolerance=tolerance,
                                max_depth=max_depth, max_triangles=max_triangles).mesh(simplify)

    def native_mesh(self, target, cell_size=None, crease_angle=35.0, max_cells=None, threads=0, bounds=None,
                    *, tolerance=None, max_depth=3, max_triangles=None):
        """The kernel's mesh of a solid (options as in SDF3.mesh), kept native
        as a NativeMesh: its stats are known before NativeMesh.mesh()
        simplifies and copies it, so callers can check sizes first."""
        node = self._solid_target(target, "mesh")
        if bounds is None:
            region = None
            extent = self.bounds(target)
            if not extent.finite:
                raise SDFError("mesh needs finite bounds: intersect unbounded shapes with a bounded "
                               "solid or pass bounds=(lo, hi).")
        else:
            region = extent = _as_bounds(bounds)
            if region.lo.shape != (3,) or not region.finite or np.any(region.hi <= region.lo):
                raise SDFError("mesh bounds must be finite 3D corners with hi > lo.")
        crease_angle = _finite(crease_angle, "crease_angle")
        if not 0.0 <= crease_angle <= 180.0:
            raise SDFError("crease_angle must be between 0 and 180 degrees.")
        options = _MeshOptions(
            cell_size=auto_cell_size(extent) if cell_size is None else _positive(cell_size, "cell_size"),
            # The kernel reads 0 as its default (35 degrees); 0 here means every edge.
            crease_angle=max(crease_angle, _MIN_CREASE_ANGLE),
            max_cells=0 if max_cells is None else _whole(max_cells, "max_cells", 1, 2**63),
            threads=_threads(threads), refine_steps=_REFINE_STEPS,
            # The kernel reads 0 as its default depth (3) and a negative depth as none.
            max_depth=_whole(max_depth, "max_depth", 0, _MAX_DEPTH) or -1,
            tolerance=0.0 if tolerance is None else _positive(tolerance, "tolerance"),
            max_triangles=0 if max_triangles is None else _whole(max_triangles, "max_triangles", 1, 2**63))
        if region is not None:
            options.bounds_lo[:] = region.lo.tolist()
            options.bounds_hi[:] = region.hi.tolist()
            options.bounds_valid = 1
        with self._live() as graph:
            handle = _native().rsdf_mesh_build(graph, node, ctypes.byref(options))
            if not handle:
                raise _kernel_error("Meshing")
        return NativeMesh(self, node, handle, options.threads)


class NativeMesh:
    """A mesh the kernel holds (Graph.native_mesh) with its `stats` dict
    (rsdf_mesh_stats fields). mesh() copies it out, optionally simplified,
    and frees it; close() frees it unused. Use it from one thread."""
    __slots__ = ("stats", "_graph", "_node", "_handle", "_threads", "_release", "__weakref__")

    def __init__(self, graph, node, handle, threads):
        library = _native()
        self._graph, self._node, self._handle, self._threads = graph, node, handle, threads
        self._release = weakref.finalize(self, library.rsdf_mesh_free, handle)
        self.stats = _native_stats(handle)

    def __enter__(self):
        return self

    def __exit__(self, *exception):
        self.close()

    def __repr__(self):
        return f"<NativeMesh {self.stats['triangle_count']} triangles{' (freed)' if self.closed else ''}>"

    @property
    def closed(self):
        return not self._release.alive

    def close(self):
        """Free the native mesh (idempotent)."""
        self._release()

    def mesh(self, simplify=None):
        """The mesh as a Mesh, simplified within `simplify` mm first when
        given (None or 0 keeps it; see SDF3.mesh), which adds
        `unsimplified_triangles` and `simplify_tolerance` to its stats.
        Frees the native mesh."""
        tolerance = None if simplify is None else _nonnegative(simplify, "simplify")
        if self.closed:
            raise SDFError("The native mesh has been freed.")
        library = _native()
        handle = self._handle
        if tolerance:
            try:
                with self._graph._live() as graph:
                    handle = library.rsdf_mesh_simplify(
                        graph, self._node, self._handle,
                        ctypes.byref(_SimplifyOptions(tolerance=tolerance, threads=self._threads)))
                    if not handle:
                        raise _kernel_error("Simplifying the mesh")
            finally:
                self.close()  # the simplified mesh replaces it
        try:
            result = _native_copy(handle)
        finally:
            if tolerance:
                library.rsdf_mesh_free(handle)
            else:
                self.close()
        if tolerance:
            result.stats["unsimplified_triangles"] = self.stats["triangle_count"]
            result.stats["simplify_tolerance"] = tolerance
        return result


def _native_stats(handle):
    """rsdf_mesh_stats of a native mesh as a dict (arrays as lists)."""
    stats = _MeshStats()
    _native().rsdf_mesh_get_stats(handle, ctypes.byref(stats))
    summary = {}
    for name, _ in _MeshStats._fields_:
        value = getattr(stats, name)
        summary[name] = list(value) if isinstance(value, ctypes.Array) else value
    return summary


def _native_copy(handle):
    """A Mesh copied from a native mesh, which stays allocated."""
    library = _native()
    summary = _native_stats(handle)
    vertices, triangle_count = summary["vertex_count"], summary["triangle_count"]
    positions = _copied(library.rsdf_mesh_positions(handle), vertices * 3, np.float32)
    normals = _copied(library.rsdf_mesh_normals(handle), vertices * 3, np.float32)
    triangles = _copied(library.rsdf_mesh_triangles(handle), triangle_count * 3, np.uint32)
    creases = _copied(library.rsdf_mesh_creases(handle), summary["crease_segment_count"] * 6, np.float32)
    return Mesh(positions.reshape(-1, 3), normals.reshape(-1, 3), triangles.reshape(-1, 3),
                creases.reshape(-1, 2, 3), summary)


# Shapes reuse the native graphs of their most recent queries. Entries hold
# their shape, so an identity key stays valid while cached; an evicted graph
# is freed once no running query still references it.
_GRAPH_CACHE_SIZE = 8
_graph_cache = collections.OrderedDict()
_graph_cache_lock = threading.Lock()


def _graph_for(shape):
    key = id(shape)
    with _graph_cache_lock:
        graph = _graph_cache.get(key)
        if graph is not None:
            _graph_cache.move_to_end(key)
            return graph
    graph = Graph(shape)
    with _graph_cache_lock:
        _graph_cache[key] = graph
        while len(_graph_cache) > _GRAPH_CACHE_SIZE:
            _graph_cache.popitem(last=False)
    return graph


def _release_graphs():
    """Drop the cached graphs of shape queries; each is freed once no running
    query still uses it. A build calls this after its program has run."""
    with _graph_cache_lock:
        graphs = list(_graph_cache.values())
        _graph_cache.clear()
    for graph in graphs:
        graph.close()


def auto_cell_size(bounds, cells_across=200, minimum=0.02):
    """A meshing cell size: the largest extent of `bounds` (a Bounds or
    (lo, hi)) divided by `cells_across`, but at least `minimum` mm. The
    minimum never coarsens the grid below 32 cells across, so tiny shapes
    still mesh."""
    extent = _as_bounds(bounds)
    if not extent.finite:
        raise SDFError("auto_cell_size needs finite bounds.")
    cells = _positive(cells_across, "cells_across")
    minimum = _nonnegative(minimum, "minimum")
    largest = float(np.max(extent.size))
    size = max(largest / cells, min(minimum, largest / 32))
    if size <= 0:
        raise SDFError("auto_cell_size needs bounds with a nonzero extent.")
    return size


# ======================================================================== meshes

_STL_HEADER = b"RAD binary STL, millimetres".ljust(80, b" ")
_STL_RECORD = np.dtype([("normal", "<f4", (3,)), ("vertices", "<f4", (3, 3)), ("attribute", "<u2")])
_SUMMED_STATS = ("active_cells", "boundary_edges", "nonmanifold_edges", "components", "cavities", "cavity_volume",
                 "seconds")


def _mesh_array(values, dtype, tail, name):
    """`values` as a contiguous array of shape (N, *tail) and type `dtype`."""
    try:
        array = np.asarray(values)
    except (TypeError, ValueError):
        array = None
    if array is not None and array.size == 0:
        array = array.reshape((0, *tail))
    if array is None or array.shape[1:] != tail or array.ndim != len(tail) + 1:
        raise SDFError(f"Mesh {name} must have shape (N, {', '.join(map(str, tail))}).")
    if np.dtype(dtype).kind == "u":
        if array.size and (array.dtype.kind not in "iu" or array.min() < 0):
            raise SDFError(f"Mesh {name} must be nonnegative integers.")
    elif array.size and not np.isfinite(array).all():
        raise SDFError(f"Mesh {name} must be finite.")
    return np.ascontiguousarray(array, dtype=dtype)


def _measure(positions, triangles):
    """Counts, enclosed volume, area, centroid and bounds of a mesh, in float64."""
    points = positions.astype(np.float64)
    if len(points):
        lo, hi = points.min(axis=0), points.max(axis=0)
    else:
        lo = hi = np.zeros(3)
    # Sums about a local origin stay well conditioned far from the world origin.
    origin = (lo + hi) / 2
    a, b, c = (points[triangles[:, corner]] - origin for corner in range(3))
    area = 0.5 * float(np.linalg.norm(np.cross(b - a, c - a), axis=1).sum())
    six_volumes = np.einsum("ij,ij->i", a, np.cross(b, c))  # signed tetrahedra with the origin
    volume = float(six_volumes.sum()) / 6.0
    centroid = origin
    if volume != 0.0:
        centroid = origin + (six_volumes[:, None] * (a + b + c)).sum(axis=0) / (24.0 * volume)
    return {"vertex_count": len(points), "triangle_count": len(triangles), "volume": volume,
            "area": area, "centroid": centroid.tolist(), "bounds_lo": lo.tolist(), "bounds_hi": hi.tolist()}


class Mesh:
    """A triangle mesh: float32 positions and unit normals (V, 3), outward
    counter-clockwise uint32 triangles (T, 3), float32 crease segments
    (S, 2, 3) and a stats dict (the kernel's rsdf_mesh_stats fields; meshes
    built here measure counts, volume, area, centroid and bounds)."""
    __slots__ = ("positions", "normals", "triangles", "creases", "stats")

    def __init__(self, positions, normals, triangles, creases=None, stats=None):
        self.positions = _mesh_array(positions, np.float32, (3,), "positions")
        self.normals = _mesh_array(normals, np.float32, (3,), "normals")
        self.triangles = _mesh_array(triangles, np.uint32, (3,), "triangles")
        self.creases = _mesh_array(np.empty((0, 2, 3)) if creases is None else creases, np.float32, (2, 3),
                                   "creases")
        if len(self.normals) != len(self.positions):
            raise SDFError("Mesh normals must have one row per position.")
        if self.triangles.size and int(self.triangles.max()) >= len(self.positions):
            raise SDFError("Mesh triangles must index existing positions.")
        if stats is None:
            stats = _measure(self.positions, self.triangles)
            stats["crease_segment_count"] = len(self.creases)
        self.stats = dict(stats)

    def __repr__(self):
        return f"<Mesh {len(self.positions)} vertices, {len(self.triangles)} triangles>"

    def save_stl(self, path):
        """Write binary STL (millimetres) with facet normals from the triangle winding."""
        corners = self.positions.astype(np.float64)[self.triangles]
        normals = np.cross(corners[:, 1] - corners[:, 0], corners[:, 2] - corners[:, 0])
        lengths = np.linalg.norm(normals, axis=1, keepdims=True)
        normals = np.divide(normals, lengths, out=np.zeros_like(normals), where=lengths > 0)
        records = np.zeros(len(corners), dtype=_STL_RECORD)
        records["normal"] = normals
        records["vertices"] = corners
        with open(path, "wb") as file:
            file.write(_STL_HEADER)
            file.write(struct.pack("<I", len(records)))
            file.write(records.tobytes())

    def transformed(self, matrix):
        """This mesh mapped by a 4x4 (or 3x4) affine matrix. Normals follow the
        inverse transpose, and reflections reverse the triangle winding so
        triangles stay outward; stats measurements are recomputed."""
        linear, offset = _affine(matrix, "Mesh.transformed")
        determinant = float(np.linalg.det(linear))
        if abs(determinant) < 1e-12:
            raise SDFError("Mesh.transformed needs a nonsingular matrix.")
        positions = self.positions.astype(np.float64) @ linear.T + offset
        normals = self.normals.astype(np.float64) @ np.linalg.inv(linear)
        lengths = np.linalg.norm(normals, axis=1, keepdims=True)
        normals = np.divide(normals, lengths, out=np.zeros_like(normals), where=lengths > 0)
        triangles = self.triangles[:, ::-1] if determinant < 0 else self.triangles.copy()
        creases = self.creases.astype(np.float64) @ linear.T + offset
        mesh = Mesh(positions, normals, triangles, creases, self.stats)
        mesh.stats.update(_measure(mesh.positions, mesh.triangles))
        if "cavity_volume" in mesh.stats:  # cavities keep their count
            mesh.stats["cavity_volume"] = float(mesh.stats["cavity_volume"]) * abs(determinant)
        return mesh


def merge_meshes(meshes):
    """One mesh holding every input mesh as separate shells. Counts, volume
    and area add up; the centroid is volume weighted and bounds are joined."""
    try:
        meshes = list(meshes)
    except TypeError:
        raise SDFError("merge_meshes takes a list of meshes.") from None
    if not meshes:
        raise SDFError("merge_meshes needs at least one mesh.")
    if not all(isinstance(mesh, Mesh) for mesh in meshes):
        raise SDFError("merge_meshes takes Mesh objects.")
    sizes = [len(mesh.positions) for mesh in meshes]
    if sum(sizes) > 2**32:
        raise SDFError("merge_meshes would exceed 2**32 vertices.")
    offsets = np.cumsum([0, *sizes[:-1]])
    positions = np.concatenate([mesh.positions for mesh in meshes])
    triangles = np.concatenate([mesh.triangles + np.uint32(offset) for mesh, offset in zip(meshes, offsets)])
    creases = np.concatenate([mesh.creases for mesh in meshes])
    stats = _measure(positions, triangles)
    stats["crease_segment_count"] = len(creases)
    for key in _SUMMED_STATS:
        if all(key in mesh.stats for mesh in meshes):
            stats[key] = sum(mesh.stats[key] for mesh in meshes)
    if all("cell_size" in mesh.stats for mesh in meshes):
        stats["cell_size"] = max(mesh.stats["cell_size"] for mesh in meshes)
    return Mesh(positions, np.concatenate([mesh.normals for mesh in meshes]), triangles, creases, stats)


# ================================================================ triangle mesh solids
#
# External triangle meshes (STL, OBJ) become MESH nodes: the exact signed
# distance to a closed, consistently oriented, manifold triangle mesh
# (SDF-DESIGN.md). The readers weld vertices and mend zero-area triangles that
# close T-junctions; the kernel validates the mesh when it is loaded, which
# mesh_solid does at once.

__all__ += ["mesh_solid", "load_stl", "load_obj"]

_MAX_MESH_TRIANGLES = 2_000_000
_MAX_MESH_VERTICES = 2_000_000
_MAX_MESH_SOURCE_BYTES = 512 * 1024 * 1024
_MAX_OBJ_VERTICES = 6_000_000
_WELD_TOLERANCE = 1e-6  # of the mesh's largest extent
# The neighbouring weld cells that follow a cell in lexicographic order.
_FORWARD_CELLS = np.array([(dx, dy, dz) for dx in (0, 1) for dy in (-1, 0, 1) for dz in (-1, 0, 1)
                           if (dx, dy, dz) > (0, 0, 0)], dtype=np.int64)
_CELL_BITS = 21
_CELL_MASK = (1 << _CELL_BITS) - 1
_STL_SOLID = re.compile(rb"\s*solid\b", re.IGNORECASE)
# An ASCII STL facet; separators are any white space (LF, CR LF or CR lines).
_STL_FACET_RECORD = re.compile(
    rb"(?<![a-z])facet\s+(?:normal\s+\S+\s+\S+\s+\S+\s+)?outer\s+loop"
    + rb"\s+vertex\s+(\S+)\s+(\S+)\s+(\S+)" * 3 + rb"\s+endloop\s+endfacet(?![a-z])", re.IGNORECASE)
_STL_END_FACET = re.compile(rb"endfacet", re.IGNORECASE)
_STL_VERTEX_LINE = re.compile(rb"(?<![a-z])vertex\s+\S+\s+\S+\s+\S+", re.IGNORECASE)
_STL_TEXT_CHUNK = 16 * 1024 * 1024
# A triangle whose doubled area is at most this fraction of its longest edge
# squared is degenerate (the kernel's rule).
_FLAT_TRIANGLE = 1e-12
_MESH_PREFIX = re.compile(r"^.*?node \d+ \(MESH\): ")


def mesh_solid(vertices, triangles, *, weld=True):
    """A solid bounded by a closed triangle mesh (exact distance to its facets).

    vertices: (N, 3) positions in mm. triangles: (M, 3) vertex indices,
    counter-clockwise seen from outside; a mesh facing inward as a whole is
    turned outward. The mesh must be closed, consistently oriented and
    manifold (every edge joins exactly two triangles), without degenerate
    triangles. Separate bodies may overlap or touch (the solid is their
    union); a shell facing inward within another's solid is a cavity. With
    `weld` (default), vertices within 1e-6 of the mesh's largest extent merge
    first, triangles that collapse are dropped and zero-area triangles that
    close T-junctions are mended. Curved originals stay faceted.
    """
    vertices = _mesh_vertices(vertices)
    triangles = _mesh_triangles(triangles, len(vertices))
    vertices, triangles = _cleaned(vertices, triangles) if weld else _compact(vertices, triangles)
    if len(triangles) > _MAX_MESH_TRIANGLES or len(vertices) > _MAX_MESH_VERTICES:
        raise SDFError(f"mesh_solid: at most {_MAX_MESH_TRIANGLES:,} triangles and {_MAX_MESH_VERTICES:,} "
                       f"vertices are supported (got {len(triangles):,} and {len(vertices):,}).")
    if len(triangles) < 4:
        raise SDFError(f"mesh_solid: a closed mesh needs at least four triangles (got {len(triangles)}).")
    params = np.empty(2 + 3 * len(vertices) + 3 * len(triangles), dtype="<f8")
    params[0], params[1] = len(vertices), len(triangles)
    params[2:2 + 3 * len(vertices)] = vertices.ravel()
    params[2 + 3 * len(vertices):] = triangles.ravel()
    params.flags.writeable = False
    node = SDF3.__new__(SDF3)
    object.__setattr__(node, "_opcode", Opcode.MESH)
    object.__setattr__(node, "_params", params)
    object.__setattr__(node, "_children", ())
    try:
        _graph_for(node)  # the kernel validates the mesh; queries of the solid reuse this graph
    except SDFError as error:
        message = str(error)
        if _MESH_PREFIX.match(message):
            raise SDFError("mesh_solid: " + _MESH_PREFIX.sub("", message, count=1)) from None
        raise
    return node


def load_stl(source):
    """Read a binary or ASCII STL file (a path, or the file's bytes).

    Returns (vertices, triangles): float64 (N, 3) positions and int64 (M, 3)
    vertex indices for mesh_solid, with vertices within 1e-6 of the mesh's
    largest extent welded, triangles that collapse dropped and zero-area
    triangles that close T-junctions mended. STL has no units; RAD reads
    millimetres.
    """
    data = _mesh_source(source, "load_stl")
    corners = _stl_corners(data)
    if not len(corners):
        raise SDFError("load_stl: the file has no triangles.")
    if not np.isfinite(corners).all():
        raise SDFError("load_stl: the file has coordinates that are not finite numbers.")
    return _cleaned(*_shared_corners(corners))


def _stl_corners(data):
    """Triangle corners (T, 3, 3) of a binary or ASCII STL.

    A binary STL is 84 bytes of header and count plus 50 bytes per triangle.
    A file of exactly that size is binary; otherwise a file starting with
    'solid' (after an optional UTF-8 byte order mark) that holds facets is
    ASCII; otherwise a binary file may carry bytes after its triangles, or a
    triangle count of 0 left by writers that stream their triangles.
    """
    count = struct.unpack_from("<I", data, 80)[0] if len(data) >= 84 else None
    if count is not None and len(data) == 84 + 50 * count:
        return _binary_stl(data, count)
    corners = _ascii_stl(data)
    if corners is not None and len(corners):
        return corners
    if count is not None:
        available, extra = divmod(len(data) - 84, 50)
        if count == 0 and available and not extra:
            return _binary_stl(data, available)
        if 0 < count <= available:
            return _binary_stl(data, count)
        if count > available and corners is None:
            raise SDFError(f"load_stl: the binary STL lists {count:,} triangles but holds only {available:,}; "
                           "the file is truncated.")
    if corners is not None:
        return corners  # an ASCII STL without facets
    raise SDFError("load_stl: the file is neither a binary STL (its size does not match its triangle count) "
                   "nor an ASCII STL ('solid' followed by facets).")


def _binary_stl(data, count):
    if count > _MAX_MESH_TRIANGLES:
        raise SDFError(f"load_stl: the file has {count:,} triangles; at most {_MAX_MESH_TRIANGLES:,} are supported.")
    return np.ascontiguousarray(np.frombuffer(data, dtype=_STL_RECORD, count=count, offset=84)["vertices"])


def _shared_corners(corners):
    """Vertices (float64) and triangles of triangle corners (T, 3, 3), with
    bitwise equal corners shared: STL repeats every shared vertex, so the
    weld then works on far fewer vertices."""
    rows = np.ascontiguousarray(corners.reshape(-1, 3))
    if rows.dtype == np.float32:
        bits = rows.view(np.uint32).astype(np.uint64)
        keys = [bits[:, 0] << np.uint64(32) | bits[:, 1], bits[:, 2]]
    else:
        bits = rows.view(np.uint64)
        keys = [bits[:, 0], bits[:, 1], bits[:, 2]]
    order = np.lexsort(keys[::-1])
    keys = [key[order] for key in keys]
    first = np.ones(len(order), dtype=bool)
    first[1:] = np.logical_or.reduce([key[1:] != key[:-1] for key in keys])
    inverse = np.empty(len(order), dtype=np.int64)
    inverse[order] = np.cumsum(first) - 1
    return rows[order[first]].astype(np.float64), inverse.reshape(-1, 3)


def load_obj(source):
    """Read the polygons of an OBJ file (a path, or the file's bytes).

    Returns (vertices, triangles) as load_stl does. `v` statements may carry
    a weight (`v x y z w`: coordinates are divided by it) or a colour (`v x y
    z r g b`, with an optional fourth colour value), which is ignored; faces
    may use negative (relative) indices and v/vt/vn references, and polygons
    with more than three corners are split into fans from their first corner
    (so they should be convex). Other statements are ignored. Lines end with
    LF, CR LF or CR.
    """
    data = _mesh_source(source, "load_obj")
    try:
        text = data.decode("utf-8-sig")
    except UnicodeDecodeError:
        text = data.decode("latin-1")
    vertices, corners, sizes = [], [], []
    for number, line in enumerate(text.split("\n") if "\n" in text else text.split("\r"), 1):
        if "#" in line:
            line = line.split("#", 1)[0]
        fields = line.split()
        if not fields:
            continue
        if fields[0] == "v":
            if len(fields) not in (4, 5, 7, 8):
                raise SDFError(f"load_obj line {number}: a vertex needs three coordinates and an optional weight "
                               "or colour.")
            try:
                point = [float(field) for field in fields[1:]]
            except ValueError:
                raise SDFError(f"load_obj line {number}: vertex coordinates must be numbers.") from None
            if len(point) > 4:
                point = point[:3]  # a colour (r g b, or r g b a)
            if len(point) == 4:
                if point[3] == 0:
                    raise SDFError(f"load_obj line {number}: a vertex weight must not be zero.")
                point = [value / point[3] for value in point[:3]]
            vertices.append(point)
            if len(vertices) > _MAX_OBJ_VERTICES:
                raise SDFError(f"load_obj: the file has more than {_MAX_OBJ_VERTICES:,} vertices.")
        elif fields[0] == "f":
            if len(fields) < 4:
                raise SDFError(f"load_obj line {number}: a face needs at least three vertices.")
            count = len(vertices)
            for field in fields[1:]:
                try:
                    index = int(field.split("/", 1)[0])
                except ValueError:
                    raise SDFError(f"load_obj line {number}: {field!r} is not a vertex reference.") from None
                index = index - 1 if index > 0 else count + index if index < 0 else -1
                if not 0 <= index < count:
                    raise SDFError(f"load_obj line {number}: {field!r} refers to a vertex that is not defined "
                                   "before the face.")
                corners.append(index)
            sizes.append(len(fields) - 1)
    if not sizes:
        raise SDFError("load_obj: the file has no faces.")
    sizes = np.array(sizes, dtype=np.int64)
    corners = np.array(corners, dtype=np.int64)
    fans = sizes - 2
    if int(fans.sum()) > _MAX_MESH_TRIANGLES:
        raise SDFError(f"load_obj: the faces make {int(fans.sum()):,} triangles; at most "
                       f"{_MAX_MESH_TRIANGLES:,} are supported.")
    starts = np.repeat(np.cumsum(sizes) - sizes, fans)
    steps = np.arange(int(fans.sum())) - np.repeat(np.cumsum(fans) - fans, fans)
    triangles = np.stack([corners[starts], corners[starts + 1 + steps], corners[starts + 2 + steps]], axis=1)
    vertices = np.array(vertices, dtype=np.float64).reshape(-1, 3)
    if not np.isfinite(vertices).all():
        raise SDFError("load_obj: the file has coordinates that are not finite numbers.")
    return _cleaned(vertices, triangles)


def _mesh_source(source, reader):
    """The bytes of a mesh file given as a path or as bytes."""
    if isinstance(source, (bytes, bytearray, memoryview)):
        data = bytes(source)
    elif isinstance(source, (str, os.PathLike)):
        try:
            with open(source, "rb") as file:
                data = file.read(_MAX_MESH_SOURCE_BYTES + 1)
        except OSError as error:
            raise SDFError(f"{reader} could not read {os.fspath(source)!r}: {error.strerror or error}.") from None
    else:
        raise SDFError(f"{reader} takes a file path or the file's bytes.")
    if len(data) > _MAX_MESH_SOURCE_BYTES:
        raise SDFError(f"{reader}: mesh files are limited to 512 MB.")
    if not data:
        raise SDFError(f"{reader}: the file is empty.")
    return data


def _ascii_stl(data):
    """Triangle corners (T, 3, 3) of an ASCII STL (none when it holds no
    facets), or None when the file does not start with 'solid'. Every facet
    needs exactly three vertices between 'outer loop' and 'endloop'. The text
    is read in blocks of whole facets, so parsing needs little memory beyond
    the result."""
    start = 3 if data.startswith(b"\xef\xbb\xbf") else 0
    if not _STL_SOLID.match(data, start, start + 1024):
        return None
    blocks, facets, position = [], 0, start
    while position < len(data):
        end = min(position + _STL_TEXT_CHUNK, len(data))
        if end < len(data):
            found = _STL_END_FACET.search(data, end)
            end = found.end() if found else len(data)
        block = data[position:end]
        records = _STL_FACET_RECORD.findall(block)
        # Every vertex line must belong to a well-formed facet (the plain
        # count is exact unless keywords are capitalized or a name has them).
        if block.count(b"vertex") != 3 * len(records) and len(_STL_VERTEX_LINE.findall(block)) != 3 * len(records):
            raise SDFError("load_stl: every ASCII STL facet must have exactly three vertex lines between "
                           "'outer loop' and 'endloop'.")
        facets += len(records)
        if facets > _MAX_MESH_TRIANGLES:
            raise SDFError(f"load_stl: the file has more than {_MAX_MESH_TRIANGLES:,} triangles; at most "
                           f"{_MAX_MESH_TRIANGLES:,} are supported.")
        if records:
            try:
                blocks.append(np.fromiter(map(float, itertools.chain.from_iterable(records)), dtype=np.float64,
                                          count=9 * len(records)))
            except ValueError:
                raise SDFError("load_stl: ASCII STL vertex coordinates must be numbers.") from None
        position = end
    return np.concatenate(blocks).reshape(-1, 3, 3) if blocks else np.zeros((0, 3, 3))


def _mesh_vertices(values):
    try:
        array = np.asarray(values, dtype=np.float64)
    except (TypeError, ValueError):
        array = None
    if array is None or array.ndim != 2 or array.shape[1] != 3 or not np.isfinite(array).all():
        raise SDFError("mesh_solid vertices must be an (N, 3) array of finite numbers.")
    return np.ascontiguousarray(array)


def _mesh_triangles(values, vertex_count):
    try:
        array = np.asarray(values)
    except (TypeError, ValueError):
        array = None
    message = "mesh_solid triangles must be an (M, 3) array of vertex indices."
    if array is None or array.ndim != 2 or array.shape[1] != 3 or array.dtype.kind not in "iuf":
        raise SDFError(message)
    if array.dtype.kind == "f":
        if not np.isfinite(array).all() or not np.array_equal(array, np.floor(array)):
            raise SDFError(message)
    if len(array) > 3 * _MAX_MESH_TRIANGLES:
        raise SDFError(f"mesh_solid: at most {_MAX_MESH_TRIANGLES:,} triangles are supported.")
    if array.size and (array.min() < 0 or array.max() >= vertex_count):
        raise SDFError(f"mesh_solid triangles must index the {vertex_count} vertices (0 to {vertex_count - 1}).")
    return np.ascontiguousarray(array, dtype=np.int64)


def _compact(vertices, triangles):
    """Drop vertices that no triangle uses, keeping their order."""
    used = np.zeros(len(vertices), dtype=bool)
    used[triangles.ravel()] = True
    if used.all():
        return vertices, triangles
    return vertices[used], (np.cumsum(used) - 1)[triangles]


def _weld(vertices, triangles):
    """Merge vertices whose coordinates differ by at most 1e-6 of the mesh's
    largest extent (chains merge together), drop triangles that collapse and
    unused vertices. A merged vertex keeps the position of its first use;
    vertices are numbered by first use."""
    vertices, triangles = _compact(vertices, triangles)
    if not len(triangles):
        return vertices, triangles
    lo = vertices.min(axis=0)
    tolerance = _WELD_TOLERANCE * float(np.max(vertices.max(axis=0) - lo))
    cells = (np.floor((vertices - lo) / tolerance).astype(np.int64) if tolerance > 0
             else np.zeros(vertices.shape, dtype=np.int64))
    keys = cells[:, 0] << 2 * _CELL_BITS | cells[:, 1] << _CELL_BITS | cells[:, 2]
    order = np.argsort(keys, kind="stable")
    ordered_keys = keys[order]
    starts = np.flatnonzero(np.concatenate(([True], ordered_keys[1:] != ordered_keys[:-1])))
    cell_keys = ordered_keys[starts]
    cell = np.empty(len(vertices), dtype=np.int64)
    cell[order] = np.repeat(np.arange(len(starts)), np.diff(np.append(starts, len(order))))
    ordered = vertices[order]
    box_lo = np.minimum.reduceat(ordered, starts, axis=0)
    box_hi = np.maximum.reduceat(ordered, starts, axis=0)
    # Vertices within the tolerance share a cell or lie in neighbouring cells.
    coordinates = np.stack([cell_keys >> 2 * _CELL_BITS, cell_keys >> _CELL_BITS & _CELL_MASK,
                            cell_keys & _CELL_MASK], axis=1)
    first, second = [], []
    for offset in _FORWARD_CELLS:
        neighbour = coordinates + offset
        wanted = neighbour[:, 0] << 2 * _CELL_BITS | neighbour[:, 1] << _CELL_BITS | neighbour[:, 2]
        position = np.searchsorted(cell_keys, wanted)
        found = np.all(neighbour >= 0, axis=1) & (position < len(cell_keys))
        found[found] = cell_keys[position[found]] == wanted[found]
        a = np.flatnonzero(found)
        b = position[a]
        gap = np.max(np.maximum(box_lo[b] - box_hi[a], box_lo[a] - box_hi[b]), axis=1)
        first.append(a[gap <= tolerance])
        second.append(b[gap <= tolerance])
    # Connected cells take their smallest label. Every round hooks the larger
    # root of each joined pair onto the smaller, then points every cell at
    # its root, so the roots of a connected set at least halve per round.
    labels = np.arange(len(cell_keys))
    a, b = np.concatenate(first), np.concatenate(second)
    while len(a):
        root_a, root_b = labels[a], labels[b]
        joined = root_a != root_b
        a, b, root_a, root_b = a[joined], b[joined], root_a[joined], root_b[joined]
        if not len(a):
            break
        np.minimum.at(labels, np.maximum(root_a, root_b), np.minimum(root_a, root_b))
        while True:
            jumped = labels[labels]
            if np.array_equal(jumped, labels):
                break
            labels = jumped
    flat = triangles.ravel()
    cluster = labels[cell[flat]]
    first_use = np.full(len(cell_keys), len(flat), dtype=np.int64)
    np.minimum.at(first_use, cluster, np.arange(len(flat)))
    used = np.flatnonzero(first_use < len(flat))
    ranked = used[np.argsort(first_use[used], kind="stable")]  # clusters in order of first use
    rank = np.empty(len(cell_keys), dtype=np.int64)
    rank[ranked] = np.arange(len(ranked))
    welded = rank[cluster].reshape(-1, 3)
    positions = vertices[flat[first_use[ranked]]]
    keep = (welded[:, 0] != welded[:, 1]) & (welded[:, 1] != welded[:, 2]) & (welded[:, 2] != welded[:, 0])
    return _compact(positions, welded[keep])


def _cleaned(vertices, triangles):
    """Welded vertices (see _weld) with zero-area T-junction triangles mended."""
    return _close_slivers(*_weld(vertices, triangles))


def _flat_triangles(vertices, triangles):
    """Whether each triangle is degenerate by the kernel's rule (its doubled
    area at most 1e-12 times its longest edge squared)."""
    a, b, c = (vertices[triangles[:, k]] for k in range(3))
    area = np.linalg.norm(np.cross(b - a, c - a), axis=1)
    longest = np.max([np.einsum("ij,ij->i", e, e) for e in (b - a, c - a, c - b)], axis=0)
    return ~(area > _FLAT_TRIANGLE * longest)


def _close_slivers(vertices, triangles):
    """Mend zero-area triangles that close T-junctions, as exporters write
    where one side of an edge has a vertex on it and the other does not: the
    triangle across the sliver's longest edge is split at the sliver's middle
    vertex, and the sliver is dropped. Slivers without exactly one such
    neighbour are kept (the kernel reports them)."""
    for _ in range(16):  # a few rounds mend runs of slivers along one edge
        flat = _flat_triangles(vertices, triangles) if len(triangles) else np.zeros(0, dtype=bool)
        slivers = np.flatnonzero(flat)
        if not len(slivers):
            break
        corners = triangles[slivers]
        following = np.roll(corners, -1, axis=1)
        lengths = np.einsum("ijk,ijk->ij", vertices[following] - vertices[corners],
                            vertices[following] - vertices[corners])
        edge = np.argmax(lengths, axis=1)  # the sliver's longest edge runs corner k -> k + 1
        rows = np.arange(len(slivers))
        start, end = corners[rows, edge], corners[rows, (edge + 1) % 3]
        middle = corners[rows, (edge + 2) % 3]
        count = len(vertices)
        keys = (triangles * count + np.roll(triangles, -1, axis=1)).ravel()
        order = np.argsort(keys, kind="stable")
        ordered = keys[order]
        wanted = end * count + start  # the neighbour runs end -> start
        first = np.searchsorted(ordered, wanted, "left")
        found = np.searchsorted(ordered, wanted, "right") - first == 1
        use = order[np.minimum(first, len(order) - 1)]
        neighbour, slot = use // 3, use % 3
        ok = found & ~flat[neighbour]
        ok &= np.bincount(neighbour[ok], minlength=len(triangles))[neighbour] == 1  # one sliver per neighbour
        if not ok.any():
            break
        apex = triangles[neighbour[ok], (slot[ok] + 2) % 3]
        split = np.concatenate([np.stack([end[ok], middle[ok], apex], axis=1),
                                np.stack([middle[ok], start[ok], apex], axis=1)])
        keep = np.ones(len(triangles), dtype=bool)
        keep[slivers[ok]] = False
        keep[neighbour[ok]] = False
        triangles = np.concatenate([triangles[keep], split])
    return vertices, triangles
