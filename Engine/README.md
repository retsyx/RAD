# CAD worker

The app passes a complete Python design program to a local worker. The program
builds a signed distance field (SDF) graph with `radsdf` and `radsdf_assembly`;
the worker serializes that graph to RAD's binary RSDF format and RAD's native
kernel (`libRADSDF.dylib`, built from `Sources/RADSDFKernel`, C API in
`include/radsdf.h`) evaluates and meshes it. [SDF-DESIGN.md](../SDF-DESIGN.md)
specifies the RSDF format, the kernel's algorithms, the program API and this
protocol, and records the design decisions; this file summarizes the worker's
interfaces.

| File | Role |
| --- | --- |
| `worker.py` | CLI, prepared build workers (`--await-build`), the serving worker (`--serve`) and error reports |
| `sdf_build.py` | Builds: runs the program, normalizes `result`, meshes the parts and their proxies, publishes the outputs |
| `sdf_ops.py` | Native operations on a build's graph: measure, clearance, print_orient, export, validate, probe |
| `radsdf.py` | Modeling API and the kernel's ctypes bindings |
| `radsdf_assembly.py` | `Assembly`, `Part`, `Placement`, `Motion` and coupling helpers |
| `assembly_motion.py` | Motion definition validation and pose records |
| `agent_server.py`, `agent-tools.json` | Agent tool interface |
| `script.sb` | Sandbox profile for builds |

## Runtime

For development, `Scripts/setup.sh` prepares the pinned standalone Python and
NumPy payload under `.build/cad-runtime/arm64`, and
`swift build --product RADSDF` builds the kernel. The app provisions its own
versioned runtime under `~/Library/Application Support/RAD/Runtimes` from that
payload and does not depend on the checkout. The packaged app carries the
kernel as `Contents/Frameworks/libRADSDF.dylib`.

Provisioning warms NumPy and `radsdf` once in the private installed runtime,
generating bytecode caches and evaluating a sphere with the kernel before
marking it ready. Existing installs without the current warmup marker are
warmed offline on their next use. Ordinary CAD workers read those caches while
keeping bytecode writes disabled; caches are not included in the downloadable
payload.

Each process holds a shared lock on its runtime's `.in-use` file and refreshes
its `.last-used` marker. After the current runtime is ready, a low-priority pass
removes superseded runtimes except the most recently used previous one, so
alternating between two app builds does not reinstall either. A runtime any
running process holds is never removed. Removal renames under the installation
lock and deletes afterward; the next installation finishes an interrupted one.

The worker loads the kernel from `RAD_SDF_LIBRARY`. The app passes the
kernel's real path in that variable and to the sandbox profile: the bundle's
copy, or in development `RAD_SDF_LIBRARY`, else the checkout's SwiftPM
products, release before debug. Run the worker as a script (not with
`python -I`), so its own folder is importable:

```sh
swift build --disable-sandbox --arch arm64 --product RADSDF -c release
export RAD_SDF_LIBRARY="$PWD/.build/release/libRADSDF.dylib"
.build/cad-runtime/arm64/expanded/python/bin/python3 Engine/worker.py \
  --request Examples/mounting-bracket.json --output /tmp/rad-bracket --threads 8
.build/cad-runtime/arm64/expanded/python/bin/python3 -m unittest discover -s Engine/tests -v
```

## Builds

`worker.py --request R --output O [--threads N]` runs one
build. A request contains `title`, `summary`, `units` (`"mm"`), a nonempty
`script` of at most 200,000 UTF-8 bytes, and optionally `print: true`.
`inputAssets` records (`{id, name, format: stl|obj, path: "input-%03d.<format>"}`)
are bound into the validation receipt, and programs read them as
`imported_assets[id]`: each STL or OBJ file becomes a radsdf mesh solid (a MESH
node, the exact signed distance to its closed triangle mesh) on first use, one
solid per distinct file within a build (`id in imported_assets` reads none).
An input that is not a valid closed mesh fails the build with a message naming
it.
All dimensions are millimeters, angles degrees.

A build validates the request, loads the kernel (a missing or incompatible
kernel is a runtime failure), computes the receipt, runs the program, normalizes
`result` into physical parts, encodes their graph once, checks every part's
conservative bounds (empty, unbounded or beyond ±1,000,000 mm parts are design
errors that name the part), meshes every part at one preview cell size, and
publishes its outputs, each written to a temporary name and renamed:

- `model.rsdf`: the RSDF graph of every part, each part's root already placed in
  world coordinates. It is the build's native artifact.
- `mesh.bin`: little-endian sections `positions` and `normals` (Float32 × 3, unit
  normals: SDF gradients, split at creases), `triangles` (UInt32 × 3, outward
  counter-clockwise, contiguous per part), `edges` (Float32 × 6 crease segments
  whose endpoints are bit-identical copies of the same part's vertex
  positions), and `triangleSolids` and `edgeSolids` (Int32 part owners);
  then, when any part has a proxy, `proxyPositions` (Float32 × 3),
  `proxyTriangles` (UInt32 × 3 indices into `proxyPositions`, outward
  counter-clockwise, grouped by part in ascending solid order) and
  `proxySolids` (Int32 part owners). Float32 matches the attributes the app
  uploads to the GPU, and the app copies each section straight into its
  packed array.
- Proxies, for exact per-pixel faces: each part with a GPU field program
  gets a closed, outward mesh of its offset surface f = δ (δ = 0.5 preview
  cells), meshed adaptively at two preview cells within δ / 3 from the
  design's graph with one appended OFFSET node per part (`model.rsdf` keeps
  the design's nodes only), simplified within 0.6 δ of the offset surface
  (the GPU shades every proxy fragment, so few large triangles are cheaper),
  its coincident vertices merged. The mesher's folds are repaired by edge
  flips and by vertex moves projected back onto f = δ. Every vertex, seven
  samples of every triangle (edge midpoints, centroid, three interior
  points) and a grid of samples at most δ / 2 apart on longer triangles have
  f > 0.1 δ and above 16 float32 steps of the largest coordinate times the
  slope bound; no triangle is flipped (non-degenerate, facing against the
  gradient at its centroid and corners while f decreases through it along
  its normal). The part's `shell` = 1.1 × max(δ, largest sampled f) + that
  float32 margin: the app's march misses only once f rises above it. Parts
  share 1,000,000 proxy triangles by their preview triangle counts; a proxy
  over its share is meshed once at twice the cell, one failing a check once
  at half the tolerance, and otherwise the part has none. Proxies never fail
  a build.
- `field.bin`: the GPU field program of every part (`sdf_field.py`,
  SDF-DESIGN.md "GPU field programs"): little-endian 32-bit words that the
  app uploads as one Metal buffer and evaluates per pixel. Top bricks are 8
  preview cells (refined down to half a cell where tapes are long), tapes
  are exact 0.1 cell beyond their bricks. A part over the program's limits
  is left to mesh rendering; a compile failure never fails the build.
- `result.json`, written last: `buildID` (a random UUID), `artifactPath`
  (absolute path of `model.rsdf`), `partRoots` (RSDF node of each part),
  `bounds`, `volume`, `area`, `solids`, `parts`, `partStats`
  (`[{solid, volume, area, bounds, centroid, triangles, components,
  cavities, maxDeviation}]`: separate pieces, sealed internal voids and a
  bound of the distance in mm from the part's triangles to its exact
  surface), `mesh` (`{cellSize, boundaryEdges, nonmanifoldEdges,
  maxDeviation}`: the base cell and the parts' largest bound), optional
  `assemblyMotion`,
  `printVersion`, `printDown`, `referenceSolids` and `customFields`,
  `validatedHash`, `fieldProgram` (`{path: "field.bin", version, words,
  brickSize, margin, buildSeconds, parts: [{solid, status: "field" |
  "meshOnly", reason?, bandBricks?, tapes?, words?, maxInstructions?}]}`, or
  `{error}` when the kernel could not compile it), `proxy` (`{offset,
  cellSize, tolerance, buildSeconds, parts: [{solid, status: "proxy",
  triangles, shell, minimum, maxDeviation, cellSize?} | {solid, status:
  "none", reason}]}`: δ, the proxy cell and tolerance, and per part its
  shell, smallest sampled f, the mesher's deviation bound before
  simplification and a coarsened cell; or `{error}` without a field program
  or proxy graph),
  `buildSeconds`, `threads`,
  `runtime` (`{python, numpy, kernel}`), `warnings` and `meshBuffers` (the
  section offsets and value counts of `mesh.bin`). `customFields` lists the
  first 32 `implicit()` shapes (custom fields) as `{solids, line?, bounds,
  slopeBound, quality}`: the parts that read the shape, the program line that
  made it, its bounds, the slope bound the kernel proved and divides the field
  by, and median sampled slope / bound (1 for a true distance); more shapes
  add a warning.

Bounds are the exact extremes of the Float32 vertex positions; volumes, areas,
centroids and piece counts come from the kernel's mesh statistics. The preview
base cell size is the design's largest extent over 256, coarsened (with a
warning) until the predicted unrefined mesh fits three quarters of one million
triangles, one million vertices and 250,000 crease segments; cells then halve
up to three times where triangles stray from the surface by more than a
twentieth of the cell, within the rest of the triangle budget (each part gets
the room earlier parts left, shared with later ones by their predicted
sizes). A part whose mesh is empty, open or encloses no volume
is a design error naming the part and suggesting thicker features. Non-manifold
edges and parts with several disconnected pieces add warnings (sealed cavities
are voids inside a piece, not pieces). Failed builds
leave no outputs besides `error.json`. The CLI prints
`{"status": "complete", "solids": N}` on success.

A request with `verifyFields: true` (rad_verify) runs the program and, instead
of loading and meshing the design, analyses the field of each of the first 64
`implicit()` shapes as its load would (`radsdf.verify_field`), so a field
without a finite slope bound is reported rather than failing. It publishes
only `result.json`: `solids`, `parts`, `fields` (`{solids, bounds, line?, ok}`
plus `slopeBound`, `valueRange`, `gradient {min, median, max}`, `quality` and
`boxes`, or `message` and `region` when no bound exists), `fieldCount`,
`buildSeconds`, `threads` and `runtime`. A field without a finite slope bound
fails an ordinary build with a design error naming the part, the
`implicit()` line, the operation and the region.

Once the app has decoded a build, it deletes everything in the build folder
except `model.rsdf`, which stays private to the session and is never a design
asset. It keeps display data in memory for the few most recently displayed
revisions; older ones replay their program when shown again.

Failures exit nonzero and publish a bounded `error.json`
(`{formatVersion: 1, stage, failureKind, message}`): `failureKind` is `design`
when changing the program can fix the failure and `runtime` when the runtime,
kernel or file access failed. Meshing failures report `stage:
"preview_meshing"` and the `cellSize`, failures of a part's bounds `stage:
"part_bounds"`, and inputs that are not valid solids `stage: "input_import"`.
The app shows the message and gives
design failures to the agent's repair loop; a replay that fails is reported and
not displayed.

## Design programs

A design's `script` is ordinary Python. Every public `radsdf` and
`radsdf_assembly` name is preloaded (`sphere`, `box`, `cylinder`, `circle`,
`polygon`, `union`, `Assembly`, `Motion`, …), with the module itself as `sdf`,
`math`, `np` (NumPy), `rad` and `imported_assets`. Programs can use functions,
loops and NumPy; there is no operation whitelist. The final geometry is
assigned to `result`:

```python
body = box((40, 30, 6), radius=1, center=(0, 0, 3)) - cylinder(3, 10, center=(12, 0, 3))
post = circle(4).extrude(16).translate((0, 0, 6))
result = Assembly("Bracket")
result.add(body | post, id="base", name="Base", color="#808080")
result.add(sphere(3, center=(30, 0, 9)), id="knob", name="Knob")
```

- A single 3D shape is one part (`solid/0`, "Part 1"); a list or tuple of 3D
  shapes is one part per shape (`solid/<i>`, "Part <i+1>"); an `Assembly` is a
  named hierarchy.
- Each physical part is exactly one solid index and one rigid body, numbered in
  parent-before-child insertion order. A part whose shape has several
  disconnected pieces is still one body, with a warning.
- `Assembly.add(shape, id=None, name=None, color=None, print_down=None, at=None,
  reference=None)` adds a part, `folder(name, …)` a child assembly and
  `add_assembly(assembly)` grafts one. `color` is `(r, g, b)` in 0..1 or
  `"#RRGGBB"`; colors, `print_down` and `reference` pass to descendants and
  attach to physical parts only. `at` is a rigid placement (a 4×4 matrix or
  `Placement(translate=…, rotate=(angle, axis))`) composed from the root down.

- `polygon(points)` checks its outline in the kernel when it is called: one
  that crosses, touches or overlaps itself raises `SDFError` at that line,
  naming the points (numbered from 0 as given), and the build fails as a
  design error.
- Custom fields: `x`, `y`, `z` are symbolic coordinates; arithmetic, `sin`,
  `cos`, `exp`, `log`, `sqrt`, `atan2`, `minimum`, `maximum` (with
  `smooth=k`, the blends of the booleans), `clamp`, `length`, `triwave`,
  `shape.field` and `shape.at(fx, fy, fz)` record a
  native expression (field nodes, opcodes 200–223) instead of computing
  numbers; with numbers these functions return numbers. `implicit(field,
  bounds)` is the only way a field becomes a solid: the kernel proves a slope
  bound L over the bounds when the graph loads and uses the field divided by
  L. `verify_field(field, bounds)` returns that proof. A `result` (or part)
  that is a field is a design error.

Programs are not a Python-language security boundary. The app runs them in a
macOS process sandbox (below); apply equivalent isolation when running the
worker outside RAD with untrusted programs.

## Parts and identities

`parts` is a recursive array of `{id, name, solidIndices, children, color?}`
nodes. Part leaves use `children: null` and own one solid in `solidIndices`;
folders use a `children` array and empty `solidIndices`, so empty arrays
identify folders rather than leaves. A folder's full solid set is the union of
its descendants'. Every solid has exactly one leaf owner. An unnamed root
assembly without an id is only a container: its children become the top level.

A part's `id` is its stable identity. An explicit `id` is used verbatim,
including existing user-created folder IDs, and must be unique. Without one, a
globally unique name becomes `name:<name>`; repeated names use a qualified
`path:` ID excluding the outer root, and the outer container uses `root`. A
generated ID that collides with an explicit one receives a `generated:` prefix.
Parts whose id starts with `user-folder:` or `folder:` stay folders, with the
physical part as their `body:` leaf. Explicit IDs preserve user placements
through renames, reordering and reparenting.

## Moving assemblies

`result.motion` holds the motion definition, written with the `Motion` builder
(`ground`, `connector`, `fixed`, `revolute`, `slider`, `couple`) and the coupling
helpers `gear`, `rack`, `screw`, `cam` and `geneva`, or as a dictionary with the
same schema. Only the result assembly's motion is read. The worker validates
and normalizes it as the preview's `assemblyMotion`; the app solves the rigid
poses without rebuilding geometry.

`grounded` contains physical part IDs; `connectors` contains
`{id, part, origin, xAxis, zAxis}` frames; and `joints` contains
`{id, name, type, parent, child, position, limits?}` records whose `parent` and
`child` reference connector IDs. Connector origins and unit perpendicular X/Z
axes use the built design's world coordinates, including every `at` placement
(`frame="part"` in the builder gives part coordinates, which the build maps to
world coordinates). Omitted axes default to world X and Z; the implied Y axis
is Z × X. Joint and connector IDs are unique within their lists. Groups cannot
act as physical bodies.

Joint types are `fixed`, `revolute` and `slider`. A zero-position mate aligns
its connector frames; a revolute position rotates about the parent connector's
Z axis in degrees, and a slider position translates along that axis in mm.
`position` defaults to zero, and optional `[lower, upper]` limits must contain
it. Fixed joints accept only zero position and no limits. Frames need not
coincide in the construction pose. Closed loops and multiple grounded parts
are allowed, with at least one grounded physical part in each connected joint
component. Definitions are bounded to 1,024 connectors and 256 joints.

Optional `couplings` contain `{id, type, driver, follower, ...}` records that
make the follower joint's coordinate a function of its movable driver joint. A
`linear` coupling sets follower = `ratio` × driver + `offset` (nonzero ratio,
offset default zero). A `table` coupling interpolates linearly between
`samples` `[[driver, follower], ...]` with strictly increasing driver values.
With a positive `period`, samples span at most one period from the first driver
sample, the relation repeats every period, and each period adds `advance`
(default zero) to the follower; a last sample at first + period must equal the
first follower value + advance. Without a period, the driver's limits must lie
within the sample range. Each joint follows at most one coupling, chains are
acyclic, and definitions are bounded to 256 couplings, 4,096 samples each and
65,536 in total. Followers are not free solver variables: the worker derives
their normalized `position`, rejecting a contradicting authored one, and the
app resolves them from their drivers before limit checks, closure and grabs.
Both implementations are tested against the golden cases in
`tests/assembly-coupling-cases.json`. Closed loops and grabs act on each
follower's independent root driver; supplied follower coordinates are
recomputed.

Viewport grabs cache the picked point, camera projection and joint graph for
the gesture. Inverse solving runs off the main actor and accepts poses only
after joint closure validation. One background worker coalesces pointer targets;
release drains the final target, while cancellation retains the last accepted
pose. Only the resulting joint coordinates enter the revision's saved state.

## 3D printing

A program may read `rad.print`, which is true only when the worker builds the
design's 3D printing version (`"print": true` in the build request, which is
part of the validation receipt). The preview reports `printVersion` when the
program read it; otherwise its ordinary geometry also serves for printing.
`print_down=(x, y, z)` pins a part's outward world direction that faces the
build plate, inherited by descendants and reported per solid as `printDown`.
`reference=True` marks reference parts, such as a circuit board inside its
case: shown, measured and checked for clearance, but never printed. The flag
is inherited like colors (a part's own `False` opts out), reported as the
ascending solid indices `referenceSolids`, and reference parts get no
`printDown` record. The app requires a printing version to keep the ordinary
build's physical parts, solid ownership and reference parts, and validates it
before accepting an agent design.

The app orients each printed part once per oriented geometry for the session
(the `print_orient` operation below; reference parts are never requested) and
packs the parts on Z=0 with 50 mm clearance
(`PrintLayout`). The 3D printing version view shows the printing version in
built coordinates; the 3D printing layout view orients and packs every printed
part of the displayed geometry once and leaves reference parts out, and hiding
parts never repacks. Exports for 3D printing use the printing version, pack
only the exported printed parts (reference parts are left out; a selection of
reference parts alone fails) and send the placements as `solidTransforms`, so
the display and exports share geometry,
orientations and the packing rule. Selected surface points belong to the build
where they were picked: the other build of the design and its printing version
shows none, and switching back restores them. Individual-part export writes one
STL per part into a complete folder named from the current parts tree, swapped
in atomically after staging; placement is shared across those files. Both
export options are remembered as app-wide preferences.

## Native operations

Runtime, measurement, clearance, printing-orientation, export, validation and
probe requests go to one long-lived `worker.py --serve` process per app engine.
It reads one JSON line per request (`{id, request, output, threads}`) on stdin
and answers `{id, status, message?}` on stdout; results and any `error.json`
stay in the request's output folder. The serving worker never runs design
programs, which always run in their own sandboxed process. The app stops it
after two idle minutes, and when a canceled request is still running two
seconds later.

Each request stages an APFS clone of a build's `model.rsdf` as `source.rsdf`
and carries `artifact: "source.rsdf"`, `artifactIdentity` (the build's own
`model.rsdf` path) and `partRoots`. The worker checks the file strictly before
the kernel decodes it (a regular file with one link, opened without following
links, a valid header, node count and roots) and keeps the two most recently
used graphs by `artifactIdentity`. Fields an operation does not define are
refused. Results are plain JSON in `result.json`.

Requests that place parts accept optional `solidTransforms: [{solid, matrix}]`.
Each matrix is a row-major 4×4 proper rigid world-space delta from the built
pose, acting on column vectors, with translation at indices 3, 7 and 11.
Omitted solids keep their built pose. Points and normals of single parts are in
built coordinates; pair results are posed.

- `{"operation": "runtime"}` → `{runtime}`.
- `measure`: `entities` lists one or two `{kind: "part", solid}` or
  `{kind: "point", solid, point}` items (points in built coordinates) →
  `{metrics, closestPoints?, points?}`. Points snap to their part's surface by
  gradient steps and return in request order as `{solid, point, normal}`. Two
  items add `pair:distance`, `pair:delta`, `pair:first` and `pair:second`
  metrics and `closestPoints`: the minimum distance between two parts, or from
  a point to a part. Point coordinates, point-to-point distances, angles and
  part totals are computed by the app from the selection and `partStats`.
- `clearance`: `pairs` (1–8 pairs of two disjoint, nonempty solid-index groups)
  and `poses` (1–361 sparse `solidTransforms` lists) →
  `{samples: [[{distance, closestPoints, overlapVolume?}]], tolerance}`, by pose
  and pair. A distance within the 1e-6 mm `tolerance` is contact (zero) and
  adds the interference volume, integrated on an adaptive grid, which separates
  touching from overlapping parts.
- `print_orient`: optional `solidIndices` and pinned `printDown` directions →
  `{orientations: [{solid, matrix, bounds}]}`: a rigid world-space delta from
  the built part to its printing orientation and the oriented part's bounds. A
  pin replaces the search. Otherwise candidates (large planar regions facing
  down, inertia axes, world axes and sphere directions) are ranked by the 45°
  overhang area on a sampling mesh, preferring poses whose volume centroid lies
  inside their bed contact; a quarter turn keeps each footprint at least as wide
  as deep. Parts placed by a translation of shared geometry share the work.
- `export`: optional `solidIndices`, `individualParts` (groups of solids, one
  file each), `partPresentation` (names and sRGB colors by solid index),
  `solidTransforms`, `cellSize` and `tolerance` → `model.3mf` or `part-%04d.3mf`
  and `{solids, triangles, cellSize, tolerance, meshedTriangles, maxDeviation,
  partFiles?}`. The 3MF stores separate closed meshes in millimeters, each
  with an sRGB color from the 3MF Materials and Properties Extension. Parts
  are re-meshed from a base cell of half the design's default preview cell
  size (its largest extent over 256), clamped to 0.05–0.5 mm, or `cellSize`,
  coarsened so the unrefined mesh fits three quarters of 4,000,000 meshed
  triangles; cells then halve up to three times until the triangles lie
  within 0.02 mm of the surface (within the budget), and the meshes are
  simplified within `tolerance` mm (default 0.01; at most a quarter of the
  cell; 0 keeps the full mesh), keeping sharp edges and corners
  (SDF-DESIGN.md, "Mesh simplification"). Every part is meshed before any is
  simplified. `maxDeviation` is the largest bound (mm) of the exported
  triangles' distance from the surface.
- `validate`: optional `cellSize` → `{ok, cellSize, parts: [{solid, closed,
  boundaryEdges, nonmanifoldEdges, components, cavities, previewCavities,
  volume, previewVolume, volumeChange}], warnings}`. Every part is re-meshed
  from half the preview cell size (or `cellSize`), refined as the preview is;
  open or non-manifold surfaces,
  parts that split or vanish (pieces compared without sealed cavities), sealed
  cavities that open or close, and volume changes above 2% are problems and
  clear `ok`.
- `probe`: optional `solids`, up to 65,536 world `points`, optional
  `directions` (one per point) and `maxDistance` → `{solids, samples}`, where
  `samples[i][j]` is `{distance, normal}` of point i for part `solids[j]`; with
  directions it adds `hit`, `hitDistance` (the traced distance, capped at
  `maxDistance` on a miss) and, on a hit, `hitPoint`.

Distances come from the parts' signed distance fields, seeded by sampling-mesh
vertices and refined to surface points along the field gradients, because the
fields of booleans are only bounds of the distance away from their surfaces.
Measured values are numeric, not exact geometric facts.

## Prepared build workers

Preview builds usually run in a sandboxed worker started before they are
requested. `worker.py --await-build <fifo> --output <folder>` loads Python,
NumPy, the modeling modules and the kernel, then reads one build's options
(`threads`) from the private FIFO the app created in that build
folder, after the app has staged `request.json` there. Each process still runs
exactly one program under the same sandbox profile, launched and reaped by its
own thread. The app prepares the next worker when a build finishes or the agent
calls `rad_runtime`, and stops an unused one after five minutes; the worker
exits when RAD does.

## Agent tool interface

`agent_server.py --workspace <private-task-directory>` is RAD's stdlib-only
stdio MCP adapter. The native app connects it for each agent invocation and
serves requests through private descriptor-confined filesystem IPC. Its tools
are `rad_runtime`, `rad_request_permission`, `rad_versions`, `rad_version`,
`rad_build`, `rad_inspect`, `rad_get_part_colors`, `rad_set_part_colors`,
`rad_measure`, `rad_validate`, `rad_verify`, `rad_render`, `rad_check_motion`,
`rad_screenshot`, `rad_export_3mf` and `rad_import`, all implemented in Swift
(`CADAgentTools`) on this worker's builds and native operations.
`rad_runtime` stages the agent-neutral modeling guide (the end of
`Sources/RAD/Resources/model-instructions.md`) in the task workspace.

`agent-tools.json` is the canonical CAD tool schema catalog shared by this
adapter and RAD's native external-agent interface. Each tool's `host` object
tells the native app how to authorize, route and audit its calls: the external
capability it requires, image results, permission prompts, recorded program
files and assigned export paths. Agents never see it. When enabled in Settings,
the app also serves MCP Streamable HTTP at `http://127.0.0.1:<port>/mcp`
(default port 47522). The external interface uses per-client credentials,
explicit design/task IDs, an isolated worker per task, binary file uploads,
and opaque task file IDs. `rad_build.programID` refers to an uploaded program;
other file arguments refer to opaque IDs rather than host paths. Authenticated
downloads expose task results without a shared filesystem. See the root
[External agents guide](../README.md#external-agents) for connection and workflow
details.

Tool arguments and results use the world coordinates of the artifact's assembly
pose. `rad_render` accepts optional `jointPositions` overrides and a
`drivingJoint` whose requested coordinate remains fixed while passive joints
satisfy closed loops; it solves a private pose for that image and returns the
effective joint coordinates without changing the candidate or live design.
Summary metadata identifies construction bounds as `builtBounds` while an
assembly is posed. `rad_check_motion` solves evenly spaced poses of one
independent joint, carrying closed-loop continuity from the last reachable
pose, sends only the paired solids' transforms to `clearance`, and reports
reachable samples, per-pair minimum clearance, contact and interference
positions, and the joints that moved. PNG images use RAD's native Metal
renderer and are returned as MCP image blocks for visual review. Build and
export outputs stay in the task workspace until final commit. Stop cancels
native work and releases waiting MCP calls. The built-in adapter does not
install a listener or global MCP configuration. The optional external listener
exists only while enabled in the running app and binds exclusively to
127.0.0.1. Source files and audit snapshots remain separate from the bounded
initial prompt.

External builds remain speculative until `rad_commit`. Its expected state token
protects newer design state, and its idempotency key is saved with a revision
receipt so retries do not create duplicate revisions. The app reuses the
matching validated candidate, preserves the existing part-organization and
color rules, and publishes through the same coordinated autosave writer.
External task audits retain the initiating request and public client identity,
with model and effort only when supplied. They do not invent chat messages or
agent turns. Request text, candidate programs and source snapshots use the
existing shared hash asset table; derived media stays transient.

External tools are user-managed. The bundled CAD baseline needs no extra install.
Before an optional download or installation, the agent calls
`rad_request_permission` with a specific operation and reason. The tool returns
a decision only; it never installs or executes anything. After an allowed
decision, the agent performs the operation in an isolated temporary environment
inside its task workspace. Each new proposed operation must query the current
global Mac-local policy, even after a previous always-allow or always-deny
result. Saved chat/audit decisions are not grants. Settings can change
Ask/Always Allow/Always Deny at any time; policy changes affect future or
pending requests without terminating commands already running. Denied/canceled
requests authorize nothing; do not rephrase or retry the same operation without
new user direction. The shared runtime, global settings, original files and
paths outside the workspace stay protected. The final CAD program must replay
with bundled dependencies only; task environments disappear after the turn.
Permission requests and results use the ordinary tool audit. All design
geometry stays in the complete replayable program; renders, exports and native
graphs remain temporary. See the main README for the optional Tool Activity
pane, which is hidden by default.

## Limits, sandbox and process runner

Designs are limited to 4,096 parts, 8,192 part-tree nodes nested at most 32
deep, and coordinates within ±1,000,000 mm. A design's RSDF graph is limited to
512 MB and 4,000,000 nodes (SDF-DESIGN.md lists the format's other limits). The
preview is capped at one million vertices and triangles and 250,000 crease
segments. These output limits bound native resource use; programs can use the
whole modeling API without an operation whitelist.

The macOS app starts design programs through `sandbox-exec` using `script.sb`.
This denies networking and process creation, allows reads only from the
installed runtime, worker, kernel library, system libraries and build
directory, maps only the runtime, kernel and system libraries as executable,
and allows writes only within that build directory. The worker receives a small
environment allowlist without ambient API keys; `RAD_SDF_LIBRARY` and the
profile's `KERNEL` parameter carry the kernel's real path, since sandbox rules
match literal real paths. Runtime paths are provided as profile parameters
rather than embedded user paths. The sandbox fails closed if its tool or profile
is unavailable.

The native process runner enforces a 10-minute deadline and cancellation with
a hard-kill fallback. Python model files, including stderr logs, have a hard
512 MB (512,000,000-byte) per-file limit, which 3MF exports and native graphs
share. On completion the app accepts only bounded regular output files, rejects
symbolic links and hardlinks, verifies the private `model.rsdf` path and header,
and reads only a bounded diagnostic tail on failure.

FeatureScript and OpenSCAD are not executed by this worker. Importing such a
design is a translation into a radsdf program, with the original source retained
by the app; a completed translation must describe its fidelity accurately.

## Validation receipts

Every build returns `validatedHash`, SHA-256 over length-prefixed chunks of the
validation rules tag (the rules version, the kernel version and every preview
meshing parameter), the runtime versions, the construction (units, program,
ordered input identities and formats, and the printing flag) and each input's
exact bytes. Titles, summaries and input display names are excluded, matching
validated candidate reuse at final commit. Saved revisions carry the receipt;
an SDF build has no separate verification to skip, so builds do not take one.
This is an accidental-change detector, not an authenticity mechanism or
proof that Python is deterministic. Meshes, statistics and native graphs remain
session-only.

## Tests

`Engine/tests` covers the kernel through its C API (`test_sdf_kernel_native.py`),
the bindings and modeling API (`test_radsdf.py`), builds (`test_sdf_build.py`,
`test_worker.py`, `test_preview_result.py`), native operations (measurements,
clearance, printing, 3MF export, validation and probes), motion and couplings,
the serving and prepared workers, the agent tool transport and release
tooling. The kernel tests need `RAD_SDF_LIBRARY` or a built kernel in the
checkout. Opt-in macOS integration tests check builds inside the real native
sandbox and its denial of external reads/writes, sockets and subprocesses. Run
these outside another sandbox:

```sh
RAD_RUN_SANDBOX_TESTS=1 .build/cad-runtime/arm64/expanded/python/bin/python3 -m unittest discover -s Engine/tests -v
```

`test_gpu_program.py` checks GPU field programs against the kernel: unpruned
tapes and every brick's pruned tape bitwise, far values as signed distance
bounds, determinism, limits and the build's `field.bin` (with
`RAD_RUN_SANITIZER_FUZZ=1` also under AddressSanitizer, UBSan and TSan).
`Scripts/field_gpu_validate.py` builds designs and compares the app's Metal
evaluator (`Scripts/field-gpu-check`) with the kernel: distances, signs,
normals, ray hits, and the cost of a frame.

The Swift suite's opt-in engine tests (`RAD_RUN_ENGINE_TESTS=1`) run real builds
and every native operation through the app's `EngineService`;
`EngineProtocolTests` compares the worker's raw results field by field with the
protocol above.

`requirements.txt` pins the top-level tested engine dependency (NumPy). The
full tested Python 3.14 / Apple Silicon environment is in
`requirements-lock.txt`; third-party licenses remain applicable.
