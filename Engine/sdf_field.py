"""GPU field programs: the build's field.bin (SDF-DESIGN.md, "GPU field programs").

The kernel compiles every part into a sparse grid of bricks whose tapes the
app's Metal evaluator runs per pixel. This module binds the kernel's
rsdf_gpu_* functions (radsdf_gpu.h) with ctypes on the library radsdf loads,
and publishes the program beside mesh.bin.
"""

import ctypes
import time
from pathlib import Path

import radsdf

FILE_NAME = "field.bin"
BRICK_CELLS = 8           # top-level brick edge in preview cells
MIN_BRICK_CELLS = 0.5     # smallest refined brick
MARGIN_CELLS = 0.1        # tapes stay exact this many preview cells outside their brick
FORMAT_VERSION = 2


class _Options(ctypes.Structure):
    _fields_ = [("brick_size", ctypes.c_double), ("margin", ctypes.c_double), ("min_brick_size", ctypes.c_double),
                ("max_words", ctypes.c_uint64), ("max_tape_instructions", ctypes.c_uint32),
                ("refine_instructions", ctypes.c_uint32), ("threads", ctypes.c_int32), ("reserved", ctypes.c_int32)]


class _PartInfo(ctypes.Structure):
    _fields_ = [("status", ctypes.c_int32), ("grid", ctypes.c_uint32 * 3), ("bricks", ctypes.c_uint64),
                ("band_bricks", ctypes.c_uint64), ("tapes", ctypes.c_uint64), ("words", ctypes.c_uint64),
                ("instructions", ctypes.c_uint64), ("max_instructions", ctypes.c_uint32),
                ("value_registers", ctypes.c_uint32), ("point_registers", ctypes.c_uint32),
                ("levels", ctypes.c_uint32), ("cells", ctypes.c_uint64), ("seconds", ctypes.c_double),
                ("message", ctypes.c_char * 256)]


class _ProgramHandle(ctypes.Structure):
    """Opaque rsdf_gpu_program."""


_PROGRAM = ctypes.POINTER(_ProgramHandle)
_PROTOTYPES = {
    "rsdf_gpu_program_build": (_PROGRAM, [radsdf._GRAPH, ctypes.POINTER(ctypes.c_uint32), ctypes.c_uint32,
                                          ctypes.POINTER(_Options)]),
    "rsdf_gpu_program_free": (None, [_PROGRAM]),
    "rsdf_gpu_program_words": (ctypes.POINTER(ctypes.c_uint32), [_PROGRAM, ctypes.POINTER(ctypes.c_uint64)]),
    "rsdf_gpu_program_part": (ctypes.c_int, [_PROGRAM, ctypes.c_uint32, ctypes.POINTER(_PartInfo)]),
    "rsdf_gpu_program_evaluate": (ctypes.c_int, [_PROGRAM, ctypes.c_uint32, radsdf._DOUBLES, ctypes.c_uint64,
                                                 ctypes.c_int32, radsdf._DOUBLES]),
}
_bound = None


def _library():
    global _bound
    library = radsdf._native()
    if _bound is not library:
        for name, (restype, argtypes) in _PROTOTYPES.items():
            function = getattr(library, name, None)
            if function is None:
                raise radsdf.SDFError(f"The native SDF kernel does not export {name}.")
            function.restype = restype
            function.argtypes = argtypes
        _bound = library
    return library


class FieldProgram:
    """A compiled program of a graph's parts. `words` (numpy uint32) are the
    field.bin contents; `parts` describes each part in node order. The graph
    must stay open while evaluate() is used."""

    def __init__(self, graph, nodes, brick_size, margin, threads=0, max_words=0, max_instructions=0,
                 min_brick_size=0.0, refine_instructions=0):
        import numpy as np
        library = _library()
        nodes = [int(node) for node in nodes]
        array = (ctypes.c_uint32 * len(nodes))(*nodes)
        options = _Options(float(brick_size), float(margin), float(min_brick_size), int(max_words),
                           int(max_instructions), int(refine_instructions), int(threads), 0)
        self.graph = graph
        with graph._live() as handle:
            started = time.perf_counter()
            pointer = library.rsdf_gpu_program_build(handle, array, len(nodes), ctypes.byref(options))
            self.seconds = time.perf_counter() - started
        if not pointer:
            raise radsdf._kernel_error("Compiling the GPU field program")
        self._handle = pointer
        count = ctypes.c_uint64()
        words = library.rsdf_gpu_program_words(pointer, ctypes.byref(count))
        self.words = np.ctypeslib.as_array(words, shape=(count.value,)).copy() if count.value else np.zeros(0, np.uint32)
        self.brick_size = float(brick_size)
        self.margin = float(margin)
        self.parts = []
        for index in range(len(nodes)):
            info = _PartInfo()
            if library.rsdf_gpu_program_part(pointer, index, ctypes.byref(info)) != 0:
                raise radsdf._kernel_error("Reading the GPU field program")
            record = {"status": "field" if info.status == 0 else "meshOnly", "grid": list(info.grid),
                      "bricks": info.bricks, "bandBricks": info.band_bricks, "tapes": info.tapes,
                      "words": info.words, "instructions": info.instructions,
                      "maxInstructions": info.max_instructions, "valueRegisters": info.value_registers,
                      "pointRegisters": info.point_registers, "levels": info.levels, "cells": info.cells,
                      "seconds": info.seconds}
            if info.status != 0:
                record["reason"] = info.message.decode("utf-8", "replace")
            self.parts.append(record)

    def close(self):
        if getattr(self, "_handle", None):
            _library().rsdf_gpu_program_free(self._handle)
            self._handle = None

    def __del__(self):
        try:
            self.close()
        except Exception:
            pass

    def evaluate(self, part, points, mode=0):
        """Reference float64 values (mode 0: as the GPU does; 1: unpruned)."""
        import numpy as np
        points = np.ascontiguousarray(points, dtype=np.float64).reshape(-1, 3)
        values = np.empty(len(points), dtype=np.float64)
        with self.graph._live():
            status = _library().rsdf_gpu_program_evaluate(
                self._handle, int(part), points.ctypes.data_as(radsdf._DOUBLES), len(points), int(mode),
                values.ctypes.data_as(radsdf._DOUBLES))
        if status != 0:
            raise radsdf._kernel_error("Evaluating the GPU field program")
        return values


def build(graph, cell, threads):
    """The design's FieldProgram at the preview cell size, or None (with a
    reason) when the kernel cannot compile it; a build never fails here."""
    try:
        program = FieldProgram(graph, graph.roots, BRICK_CELLS * cell, MARGIN_CELLS * cell, threads,
                               min_brick_size=MIN_BRICK_CELLS * cell)
    except radsdf.SDFError as error:
        return None, str(error)
    program.close()  # the words and part records are copied
    return program, None


def publish(program, failure, output):
    """Write field.bin (temporary name, then renamed) and return the
    result.json description: {path, words, version, brickSize, margin,
    buildSeconds, parts: [{solid, status, reason?, ...}]}, or {error}."""
    if program is None:
        return {"error": failure or "unavailable"}
    output = Path(output)
    staged = output / (FILE_NAME + ".tmp")
    try:
        staged.write_bytes(program.words.astype("<u4").tobytes())
        staged.replace(output / FILE_NAME)
    finally:
        staged.unlink(missing_ok=True)
    parts = []
    for solid, record in enumerate(program.parts):
        entry = {"solid": solid, "status": record["status"]}
        if record["status"] == "field":
            entry.update(bandBricks=record["bandBricks"], tapes=record["tapes"], words=record["words"],
                         maxInstructions=record["maxInstructions"])
        else:
            entry["reason"] = record["reason"]
        parts.append(entry)
    return {"path": FILE_NAME, "version": FORMAT_VERSION, "words": int(len(program.words)),
            "brickSize": program.brick_size, "margin": program.margin, "buildSeconds": program.seconds,
            "parts": parts}
