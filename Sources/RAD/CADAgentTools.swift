import AppKit
import CryptoKit
import CoreFoundation
import Darwin
import Foundation

/// RAD-owned operations available to the agent for one writable task workspace.
/// Builds remain speculative until the final program is committed by AppState.
@MainActor
final class CADAgentTools {
    private struct Artifact {
        var id: String
        var model: ModelSpec
        var preview: PreviewMesh?
        var programURL: URL
        var metadataURL: URL
        var versionID: UUID? = nil
        var parts: [PartNode] = []
        var partLayout: PartLayout? = nil
        var partColors: [String: PartColor] = [:]
        var colorChanges: [CADPartColorChange] = []
        /// Built with `rad.print` true: the program's 3D printing version.
        var print = false
        /// Printing orientations found for this artifact's own native geometry.
        var printOrientations: [Int: PrintOrientation] = [:]
    }
    private let workspace: CADAgentWorkspace
    private let python: String
    private let threads: Int
    private let engine: EngineService
    private let permissionController: AgentPermissionController
    private let current: ModelSpec?
    private let versions: ProjectVersionContext
    private weak var viewport: CADSceneView?
    private weak var permissionWindow: NSWindow?
    private var permissionWindowWasBound: Bool
    private let hasPermissionViewport: Bool
    private var hasExplicitPermissionWindow: Bool
    private let hiddenSolids: Set<Int>
    private let partLayout: PartLayout?
    private let initialPartColors: [String: PartColor]
    /// Candidate previews kept per task besides the current design's.
    static let retainedCandidatePreviews = 4
    private var artifacts: [String: Artifact] = [:]
    /// Creation order decides which matching construction is newest.
    private var artifactOrder: [String] = []
    /// Build recency decides which candidates keep their previews.
    private var previewOrder: [String] = []
    private var inputs: [String: CADInputAsset] = [:]
    /// SHA-256 digests of the geometry this task exported. Such files are
    /// generated design geometry and never become imported inputs.
    private var generatedGeometry = Set<String>()
    private var versionInputSizes: [String: Int] = [:]
    private var versionTasks: [UUID: Task<CADAgentVersionFiles.Prepared, Error>] = [:]
    private var runtime: CADRuntime?
    private var permissionRequest: (id: UUID, task: Task<AgentPermissionDecision, Error>)?
    /// A built-in design task starts without network access in its sandbox; an
    /// allowed decision enables it for the rest of that task. External clients
    /// own their own network and leave this unset.
    var onPermissionAllowed: (@MainActor (_ operation: String) -> Void)?
    private var cancelled = false

    init(directory: URL, python: String, threads: Int, current: ModelSpec?, currentPreview: PreviewMesh?,
         engine: EngineService, viewport: CADSceneView? = nil, hiddenSolids: Set<Int> = [],
         solidColors: [Int: PartColor] = [:],
         referenceAssets: [CADInputAsset] = [], versions: ProjectVersionContext = .empty,
         partLayout: PartLayout? = nil, partColors: [String: PartColor] = [:],
         permissionController: AgentPermissionController? = nil, permissionWindow: NSWindow? = nil) throws {
        workspace = try CADAgentWorkspace(directory: directory)
        self.python = python
        self.threads = threads
        self.current = current
        self.versions = versions
        self.engine = engine
        self.permissionController = permissionController ?? .shared
        self.viewport = viewport
        self.permissionWindow = permissionWindow ?? viewport?.window
        permissionWindowWasBound = self.permissionWindow != nil
        hasPermissionViewport = viewport != nil
        hasExplicitPermissionWindow = permissionWindow != nil
        self.hiddenSolids = hiddenSolids
        self.partLayout = partLayout
        let currentParts = partLayout?.nodes ?? Self.parts(in: currentPreview)
        if !currentParts.isEmpty {
            let count = currentPreview?.solids ?? (currentParts.flatMap(\.flattened).flatMap(\.solidIndices).max().map { $0 + 1 } ?? 0)
            try ProjectStore.validatePartHierarchy(currentParts, solidCount: count)
        }
        let physicalIDs = Self.physicalIDs(in: currentParts)
        guard partColors.count <= 4096, partColors.allSatisfy({ physicalIDs.contains($0.key) && $0.value.isValid }),
              solidColors.allSatisfy({ $0.key >= 0 && $0.key < (currentPreview?.solids ?? Int.max) && $0.value.isValid }) else {
            throw RADError.message("The current design has invalid part colors.")
        }
        var currentColors = Self.completePartColors(in: currentParts, colors: partColors)
        for node in currentParts.flatMap(\.flattened) where partColors[node.id] == nil && !node.solidIndices.isEmpty {
            let legacy = node.solidIndices.compactMap { solidColors[$0] }
            if legacy.count == node.solidIndices.count, let color = legacy.first, legacy.allSatisfy({ $0 == color }) {
                currentColors[node.id] = color
            }
        }
        self.initialPartColors = currentColors
        runtime = currentPreview?.runtime
        for asset in current?.inputAssets ?? [] {
            guard inputs[asset.id] == nil else { throw RADError.message("Imported geometry has duplicate asset identities.") }
            inputs[asset.id] = asset
        }
        // A queued selection may refer to an earlier revision's original
        // external inputs. Keep these available for exact origin replay without
        // changing the current design.
        try registerReferenceAssets(referenceAssets)
        if let current {
            // CodexFileContext owns creation of context/. Native candidate
            // references live under artifacts/ for both MCP transports.
            let programURL = try workspace.write(Data(current.script.utf8), path: "artifacts/current/program.py")
            artifacts["current"] = Artifact(id: "current", model: current, preview: currentPreview,
                programURL: programURL,
                metadataURL: workspace.root.appendingPathComponent("artifacts/current/metadata.json"),
                parts: currentParts, partLayout: partLayout, partColors: currentColors)
            if currentPreview != nil, let artifact = artifacts["current"] {
                _ = try workspace.write(JSONSerialization.data(withJSONObject: summary(artifact), options: [.sortedKeys]),
                                        path: "artifacts/current/metadata.json")
            }
        }
    }

    func registerReferenceAssets(_ assets: [CADInputAsset]) throws {
        guard assets.count <= 256 else { throw RADError.message("The selection has too many reference inputs.") }
        var merged = inputs
        for asset in assets {
            if let existing = merged[asset.id] {
                guard existing.format == asset.format, existing.data == asset.data else {
                    throw RADError.message("Selection reference inputs have conflicting geometry identities.")
                }
            } else { merged[asset.id] = asset }
        }
        guard merged.count <= 8_192 else { throw RADError.message("The selection has too many reference inputs.") }
        var bytes = 0
        for asset in merged.values {
            guard asset.data.count <= ProjectStore.maximumFileBytes - bytes else {
                throw RADError.message("The selection's reference inputs exceed the design's 1,024 MB storage limit.")
            }
            bytes += asset.data.count
        }
        inputs = merged
    }

    func resolveAssets(ids: [String]) throws -> [CADInputAsset] {
        try resolveAssets(ids: ids, available: inputs)
    }

    static let generatedImportRefusal = "rad_import registers only pre-existing external or reference meshes. This file is geometry generated for this design (an export, render or artifact of this task, or an STL written by RAD); keep generated geometry in the complete replayable program instead."

    /// Build failures of `result = imported_assets[id]` that leave the mesh a
    /// valid solid: its program ran, and only its stand-alone preview failed.
    static let importPreviewStages: Set<String> = ["preview_meshing", "part_bounds"]

    /// radsdf's Mesh.save_stl and earlier RAD STL exports use this header.
    static func isGeneratedMesh(_ data: Data) -> Bool {
        data.count >= 84 && data.prefix(14) == Data("RAD binary STL".utf8)
    }

    /// Makes an imported input available to later builds and the final response.
    private func register(importedInput asset: CADInputAsset) throws {
        if let existing = inputs[asset.id] {
            guard existing.format == asset.format, existing.data == asset.data else {
                throw RADError.message("Imported inputs have conflicting geometry identities.")
            }
            return
        }
        let bytes = inputs.values.reduce(0) { $0 + $1.data.count }
        guard inputs.count < 8_192, asset.data.count <= ProjectStore.maximumFileBytes - bytes else {
            throw RADError.message("Imported inputs exceed the design's 1,024 MB storage limit.")
        }
        inputs[asset.id] = asset
    }

    private func resolveAssets(ids: [String], available: [String: CADInputAsset]) throws -> [CADInputAsset] {
        guard ids.count <= 256, Set(ids).count == ids.count else {
            throw RADError.message("Choose at most 256 distinct imported geometry inputs.")
        }
        let assets = try ids.map { id in
            guard let asset = available[id] else { throw RADError.message("Imported geometry input ‘\(id)’ is unavailable in this build's input context.") }
            return asset
        }
        var total = 0
        for asset in assets {
            guard asset.data.count <= ProjectStore.maximumFileBytes - total else {
                throw RADError.message("Imported geometry inputs exceed the design's 1,024 MB storage limit.")
            }
            total += asset.data.count
        }
        return assets
    }

    /// A matching candidate built and validated during this task.
    func validatedPreview(for model: ModelSpec) -> PreviewMesh? {
        validatedPreview(for: model, print: false)
    }

    /// A matching 3D printing version built during this task, if any.
    func validatedPrintPreview(for model: ModelSpec) -> PreviewMesh? {
        validatedPreview(for: model, print: true)
    }

    private func validatedPreview(for model: ModelSpec, print: Bool) -> PreviewMesh? {
        for id in Array(artifactOrder.reversed()) + ["current"] {
            guard let artifact = artifacts[id], artifact.print == print, constructionMatches(artifact.model, model),
                  let preview = artifact.preview, Self.hasNativeArtifact(preview) else { continue }
            return preview
        }
        return nil
    }

    /// A build's RSDF graph still exists, so native operations can use it.
    private static func hasNativeArtifact(_ preview: PreviewMesh) -> Bool {
        preview.artifactPath.map { FileManager.default.fileExists(atPath: $0) } ?? false
    }

    /// A program that reads `rad.print` must also build a 3D printing version
    /// with the design's parts. Reuses this task's matching printing build.
    func validatedPrintVersion(for model: ModelSpec, design: PreviewMesh) async throws -> PreviewMesh? {
        guard design.printVersion == true else { return nil }
        let printing: PreviewMesh
        if let cached = validatedPrintPreview(for: model) { printing = cached }
        else { printing = try await engine.buildPreview(model, print: true, python: python, threads: threads) }
        try checkCancellation()
        try PrintLayout.validateParts(of: printing, matching: design)
        return printing
    }

    /// Rebuilds a committable candidate whose display cache was evicted, then
    /// applies the same checks as `validatedCandidate`.
    func committableCandidate(artifactID: String) async throws -> (model: ModelSpec, preview: PreviewMesh, colorChanges: [CADPartColorChange]) {
        try checkCancellation()
        if let artifact = artifacts[artifactID], artifact.versionID == nil, !artifact.print,
           !(artifact.preview.map(Self.hasNativeArtifact) ?? false) {
            _ = try await build(artifact.model, id: artifactID)
        }
        let candidate = try validatedCandidate(artifactID: artifactID)
        do { _ = try await validatedPrintVersion(for: candidate.model, design: candidate.preview) }
        catch let failure as EngineBuildFailure { throw RADError.message(EngineFailureDiagnostic.repairText(for: failure)) }
        return candidate
    }

    /// External agents commit a named candidate, never arbitrary source paired
    /// with another candidate's geometry. This boundary applies the same final
    /// replay checks as an internal Codex response.
    func validatedCandidate(artifactID: String) throws -> (model: ModelSpec, preview: PreviewMesh, colorChanges: [CADPartColorChange]) {
        try checkCancellation()
        guard let artifact = artifacts[artifactID], let preview = artifact.preview, Self.hasNativeArtifact(preview) else {
            throw RADError.message("Build and validate this CAD candidate before committing it.")
        }
        guard artifact.versionID == nil else {
            throw RADError.message("Historical builds use read-only saved input bindings. Register any required original reference files and build the final complete CAD program without versionID before committing it.")
        }
        guard !artifact.print else {
            throw RADError.message("Commit the program's ordinary build, not its 3D printing version. RAD derives the printing version from the same program.")
        }
        let model = artifact.model
        try ProjectStore.validateModel(model)
        guard constructionMatches(artifact.model, model) else {
            throw RADError.message("The final CAD program no longer matches its validated geometry. Build it again before committing.")
        }
        // Construction matching permits geometry reuse, but speculative color
        // patches belong to one exact artifact. A later matching candidate must
        // never replace the colors reviewed on the explicitly committed one.
        return (model, preview, artifact.colorChanges)
    }

    /// Only the chosen construction's speculative color patch crosses the
    /// successful task boundary. Other candidates and history remain isolated.
    func partColorChanges(for model: ModelSpec?) -> [CADPartColorChange] {
        guard !cancelled else { return [] }
        guard let model else { return artifacts["current"]?.colorChanges ?? [] }
        for id in Array(artifactOrder.reversed()) + ["current"] {
            guard let artifact = artifacts[id], constructionMatches(artifact.model, model) else { continue }
            guard artifact.versionID != nil else { return artifact.colorChanges }
            // Choosing a historical construction must also restore the exact
            // current colors that were reviewed.
            var changes: [String: CADPartColorChange] = [:]
            for id in Self.physicalIDs(in: artifact.parts) where artifact.partColors[id] != initialPartColors[id] {
                changes[id] = CADPartColorChange(partID: id, color: artifact.partColors[id])
            }
            for change in artifact.colorChanges { changes[change.partID] = change }
            return changes.values.sorted { $0.partID < $1.partID }
        }
        return []
    }

    func cancel() {
        cancelled = true
        versionTasks.values.forEach { $0.cancel() }
        permissionRequest?.task.cancel()
        engine.cancel()
    }

    func bindPermissionWindow(_ window: NSWindow) throws {
        try checkCancellation()
        guard !hasExplicitPermissionWindow || permissionWindow === window else {
            throw RADError.message("This task's permission request belongs to a different design window.")
        }
        if let owningViewportWindow = viewport?.window, owningViewportWindow !== window {
            throw RADError.message("Use this task's owning design window for permission requests.")
        }
        permissionWindow = window
        permissionWindowWasBound = true
        hasExplicitPermissionWindow = true
    }

    func handle(name: String, arguments: [String: Any]) async throws -> CADAgentToolResult {
        try checkCancellation()
        guard try AgentToolCatalog.native()[name] != nil else { throw RADError.message("Unknown RAD tool ‘\(name)’.") }
        switch name {
        case "rad_runtime":
            try allowed(arguments, [])
            // The agent calls this before constructing geometry; load the CAD
            // runtime for its first build meanwhile.
            engine.prepareBuildWorker(python: python)
            if runtime == nil { runtime = try await engine.runtimeInfo(python: python, threads: threads) }
            let guide = try workspace.write(Data(try Self.modelingGuide().utf8), path: "context/radsdf-guide.md")
            return try result(["pythonExecutable": python, "runtime": try json(runtime!), "threads": threads,
                               "workspace": workspace.root.path, "apiGuidePath": guide.path,
                               "programInterface": "A Python program using the radsdf SDF modeling API assigns its final solid, list of solids or Assembly to result. Every public radsdf name (sphere, box, cylinder, circle, rect, polygon, union, difference, Assembly, Motion, ...), the module as sdf, math, np, rad (rad.print) and imported_assets are preloaded. Millimetres and degrees; distances are negative inside.",
                               "capabilities": ["3D primitives", "2D profiles", "extrude", "tapered extrude", "revolve", "loft", "booleans", "smooth blends", "shell", "offset", "twist", "elongate", "mirror", "linear, grid and polar arrays", "involute gears", "assemblies", "motion and couplings", "3D printing version", "custom fields (implicit)", "SDF probes", "numeric measurements", "PNG rendering", "3MF export", "STL and OBJ mesh import"],
                               "externalTools": "External tools are user-managed. Before each new optional download or dependency installation, call rad_request_permission with the proposed operation and reason, even after a previous always-allow/always-deny result; Settings can change at any time. Proceed only when the current result has allowed true. That tool returns a decision only; you execute any approved operation in an isolated environment inside this task workspace, using RAD's Python as the base for a helper environment if needed. Never change RAD's shared runtime, global settings, original sources or files outside the task workspace. Final authored geometry must replay using only RAD's bundled CAD dependencies and registered reference inputs. rad_import is only for pre-existing external or reference meshes and faithful format conversions of user-provided inputs; never import or otherwise freeze geometry generated for this design, by RAD (such as a 3MF export) or by external tools.",
                               "apiHelp": "The selected runtime is ready to use; no CAD environment bootstrap is needed. Read apiGuidePath for the complete modeling guide and run programs with rad_build.",
                               "inputAssets": inputs.values.sorted { $0.id < $1.id }.map { ["assetID": $0.id, "name": $0.name, "format": $0.format] }])
        case "rad_request_permission":
            try allowed(arguments, ["operation", "reason"])
            let operation = try permissionText("operation", arguments, maximum: 200)
            let reason = try permissionText("reason", arguments, maximum: 2_000)
            guard permissionRequest == nil else { throw RADError.message("A permission request is already pending for this task.") }
            let controller = permissionController
            let owner: NSWindow?
            if hasExplicitPermissionWindow {
                guard let window = permissionWindow else { throw CancellationError() }
                owner = window
            } else if hasPermissionViewport {
                if !permissionWindowWasBound, let window = viewport?.window {
                    permissionWindow = window
                    permissionWindowWasBound = true
                }
                guard let window = permissionWindow, viewport?.window === window else { throw CancellationError() }
                owner = window
            } else { owner = nil }
            let id = UUID()
            let grantsNetwork = onPermissionAllowed != nil
            let consequence = grantsNetwork ? "Allowing also enables network access for the rest of this design task." : nil
            let request = Task { try await controller.request(operation: operation, reason: reason, consequence: consequence, window: owner) }
            permissionRequest = (id, request)
            defer { if permissionRequest?.id == id { permissionRequest = nil } }
            let decision = try await withTaskCancellationHandler {
                try await request.value
            } onCancel: {
                request.cancel()
            }
            try checkCancellation()
            var values: [String: Any] = ["allowed": decision.allowed, "source": decision.source.rawValue,
                               "rememberedPolicy": decision.rememberedPolicy.map { $0.rawValue as Any } ?? NSNull(),
                               "scope": "task_workspace", "workspace": workspace.root.path,
                               "executionPerformed": false,
                               "nextStep": decision.allowed
                                   ? "Permission granted for this proposed operation inside this task workspace. Execute it yourself; RAD has not run anything. Use an isolated local environment for optional installs, keep final CAD replay dependent only on RAD's bundled dependencies, and query permission again for each new operation."
                                   : "Permission denied. Do not perform the proposed operation or rephrase/retry it without new user direction. Continue with existing or bundled capabilities, or explain the limitation."]
            if decision.allowed, grantsNetwork, let grant = onPermissionAllowed {
                // Grant before the agent can act on this decision.
                grant(operation)
                values["networkAccess"] = "enabled for the rest of this task"
            }
            return try result(values)
        case "rad_versions":
            try allowed(arguments, ["offset", "limit"])
            let offset = try integer("offset", arguments, fallback: 0, range: 0...1_000_000)
            let limit = try integer("limit", arguments, fallback: 50, range: 1...100)
            let page = Array(versions.versions.dropFirst(offset).prefix(limit))
            return try result(["offset": offset, "total": versions.versions.count,
                               "items": page.map(CADAgentVersionFiles.metadata),
                               "hasMore": offset + page.count < versions.versions.count])
        case "rad_version":
            try allowed(arguments, ["reference"])
            let reference = try string("reference", arguments, required: true)!.trimmingCharacters(in: .whitespacesAndNewlines)
            guard !reference.isEmpty else { throw RADError.message("Choose a saved version by its versionID, vN label or exact custom name.") }
            let matches = versions.matches(reference: reference)
            guard !matches.isEmpty else { throw RADError.message("The saved version ‘\(reference)’ is unavailable. Use rad_versions to list this task's saved versions.") }
            guard matches.count == 1 else {
                return try result(["status": "ambiguous", "reference": reference,
                                   "candidates": matches.map(CADAgentVersionFiles.metadata),
                                   "nextStep": "Choose a candidate's stable versionID; no version was opened or rebuilt."])
            }
            return try await stageVersion(matches[0])
        case "rad_build":
            try allowed(arguments, ["programPath", "title", "summary", "assetIDs", "versionID", "print"])
            let printing = try bool("print", arguments) ?? false
            let version = try string("versionID", arguments).map { try versions.version(id: $0) }
            let origin = version?.model ?? current
            let path = try string("programPath", arguments, required: true)!
            let bytes = try workspace.read(path, maximumBytes: 200_000)
            guard let script = String(data: bytes, encoding: .utf8), !script.trimmingCharacters(in: .whitespacesAndNewlines).isEmpty else {
                throw RADError.message("The CAD program must be nonempty UTF-8 Python text.")
            }
            let ids = try strings("assetIDs", arguments) ?? (origin?.inputAssets ?? []).map(\.id)
            let selected: [CADInputAsset]
            if let version {
                let assets = version.model.inputAssets ?? []
                try ProjectStore.validateInputAssets(assets)
                selected = try resolveAssets(ids: ids, available: Dictionary(uniqueKeysWithValues: assets.map { ($0.id, $0) }))
            } else { selected = try resolveAssets(ids: ids) }
            let model = ModelSpec(title: try boundedText("title", arguments, fallback: version?.model.title ?? "Untitled design", maximum: 160),
                                  summary: try boundedText("summary", arguments, fallback: version?.model.summary ?? "CAD program built and validated by RAD.", maximum: 6_000),
                                  units: "mm", script: script, inputAssets: selected.isEmpty ? nil : selected)
            let artifact = try await build(model, versionID: version?.id, print: printing)
            return try result(summary(artifact))
        case "rad_import":
            try allowed(arguments, ["path", "name", "producerPaths"])
            let path = try string("path", arguments, required: true)!
            let producerPaths = try (strings("producerPaths", arguments) ?? []).map { try workspace.url($0).path }
            let url = try workspace.url(path)
            // The workspace's volume may ignore case, so its generated folders do too.
            let relative = String(url.path.dropFirst(workspace.root.path.count + 1)).lowercased()
            guard !["artifacts/", "exports/", "renders/"].contains(where: relative.hasPrefix) else {
                throw RADError.message(Self.generatedImportRefusal)
            }
            let format = url.pathExtension.lowercased()
            guard CADInputAsset.supportedFormats.contains(format) else {
                throw RADError.message("rad_import reads STL (.stl) or OBJ (.obj) triangle meshes from the task workspace. Original external meshes in other formats can be converted to STL after asking permission for any tool the conversion needs.")
            }
            let data = try workspace.read(path, maximumBytes: ProjectStore.maximumAssetBytes)
            let digest = SHA256.hexDigest(data)
            guard !generatedGeometry.contains(digest), !Self.isGeneratedMesh(data) else {
                throw RADError.message(Self.generatedImportRefusal)
            }
            let id = "asset-" + digest
            if let existing = inputs[id], existing.format != format {
                throw RADError.message("These bytes are already registered as the \(existing.format.uppercased()) input \(id).")
            }
            let asset = CADInputAsset(id: id, name: try boundedText("name", arguments, fallback: url.lastPathComponent, maximum: 255),
                                      format: format, data: data)
            try ProjectStore.validateInputAssets([asset])
            let binding = String(decoding: try JSONEncoder().encode(id), as: UTF8.self)
            let model = ModelSpec(title: "Imported geometry", summary: "Validated \(format.uppercased()) input mesh.", units: "mm",
                                  script: "result = imported_assets[\(binding)]\n", inputAssets: [asset])
            var info: [String: Any]
            do {
                info = summary(try await build(model))
            } catch let failure as EngineBuildFailure where failure.kind == .design
                        && Self.importPreviewStages.contains(failure.stage ?? "") {
                // The mesh decoded as a valid solid (its program ran); only a
                // preview of it alone failed, such as a sheet thinner than the
                // preview cells of its own extent. It still works in builds.
                info = ["warnings": ["The mesh is a valid solid, but a preview of it alone failed, so no artifact was built: "
                    + EngineFailureDiagnostic.repairText(for: failure)
                    + " Use it within a larger part or as a cutter, and check those builds."]]
            }
            try register(importedInput: asset)
            info["assetID"] = id
            info["name"] = asset.name
            info["format"] = format
            info["byteCount"] = data.count
            info["fidelity"] = "A closed triangle mesh, used as the exact signed distance to its facets. Smooth surfaces are not recovered: curved originals stay faceted, and RAD's previews, measurements and 3MF exports re-mesh the facets at their resolution."
            info["programBinding"] = "imported_assets[assetID]"
            info["producerPaths"] = producerPaths
            info["provenance"] = producerPaths.isEmpty
                ? "No producer files were specified. Optional producerPaths can retain source-conversion code and dependency descriptions for a pre-existing reference input in the task audit."
                : "Producer file references supplied for the saved task audit."
            return try result(info)
        case "rad_get_part_colors":
            try allowed(arguments, ["artifactID", "offset", "limit"])
            let artifact = try colorArtifact(arguments)
            let offset = try integer("offset", arguments, fallback: 0, range: 0...1_000_000)
            let limit = try integer("limit", arguments, fallback: 100, range: 1...500)
            return try result(colorPage(artifact, offset: offset, limit: limit))
        case "rad_set_part_colors":
            try allowed(arguments, ["artifactID", "changes"])
            var artifact = try colorArtifact(arguments)
            let changes = try colorChanges(arguments)
            let normalized = try PartColorState.normalized(changes: changes, in: artifact.parts)
            let updated = try PartColorState.applyingPhysical(changes: normalized, to: artifact.partColors, in: artifact.parts)
            var staged = Dictionary(uniqueKeysWithValues: artifact.colorChanges.map { ($0.partID, $0) })
            for change in normalized { staged[change.partID] = change }
            artifact.partColors = Self.completePartColors(in: artifact.parts, colors: updated)
            artifact.colorChanges = staged.values.sorted { $0.partID < $1.partID }
            artifacts[artifact.id] = artifact
            var info = try colorPage(artifact, offset: 0, limit: 100)
            info["staged"] = true
            info["changedPartCount"] = normalized.count
            info["nextStep"] = "Colors are staged for this artifact only. Use rad_render to review them when needed. A successful final response applies only the current design's patch or the final replacement program's matching artifact patch."
            return try result(info)
        case "rad_inspect":
            try allowed(arguments, ["artifactID", "kind", "solidIndices", "partIDs", "offset", "limit",
                                    "points", "directions", "maxDistance"])
            let kind = try string("kind", arguments) ?? "summary"
            guard ["summary", "parts", "motion", "print", "probe"].contains(kind) else {
                throw RADError.message("Inspect summary, parts, motion, print or probe.")
            }
            guard kind == "probe" || ["points", "directions", "maxDistance"].allSatisfy({ arguments[$0] == nil }) else {
                throw RADError.message("points, directions and maxDistance apply only to kind probe.")
            }
            var artifact = try await artifact(arguments)
            let preview = artifact.preview!
            let solids = try selectedSolids(arguments, artifact: artifact, preview: preview)
            let offset = try integer("offset", arguments, fallback: 0, range: 0...1_000_000)
            let limit = try integer("limit", arguments, fallback: 100, range: 1...500)
            let space = (preview.assemblyPose?.solidTransforms.isEmpty ?? true) ? "builtGeometry" : "currentAssemblyPose"
            switch kind {
            case "summary":
                return try result(summary(artifact))
            case "parts":
                guard let stats = preview.partStats else { throw RADError.message("The design has no part statistics.") }
                let selected = stats.filter { solids?.contains($0.solidIndex) ?? true }
                let page = CADMeasurements.posedParts(Array(selected.dropFirst(offset).prefix(limit)),
                                                      transforms: preview.assemblyPose?.solidTransforms ?? [:])
                let items = try partItems(page, artifact: artifact)
                return try result(["artifactID": artifact.id, "kind": kind, "offset": offset,
                                   "total": selected.count, "items": items, "hasMore": offset + items.count < selected.count,
                                   "coordinateSpace": space])
            case "motion":
                return try result(motionInfo(artifact))
            case "print":
                var info: [String: Any] = ["artifactID": artifact.id, "kind": kind, "print": artifact.print,
                                           "definesPrintVersion": preview.printVersion == true]
                let owners = Self.owners(in: artifact.parts)
                info["printDown"] = (preview.printDown ?? []).map { pin -> [String: Any] in
                    var item: [String: Any] = ["solidIndex": pin.solid, "direction": pin.direction]
                    if let owner = owners[pin.solid] { item["partID"] = owner.id; item["name"] = owner.name }
                    return item
                }
                let requested = solids ?? Set(0..<preview.solids)
                let placed = PrintLayout.printedSolids(requested, of: preview)
                let references = requested.subtracting(placed).sorted()
                if !references.isEmpty {
                    info["referenceParts"] = references.map { solid -> [String: Any] in
                        var item: [String: Any] = ["solidIndex": solid]
                        if let owner = owners[solid] { item["partID"] = owner.id; item["name"] = owner.name }
                        return item
                    }
                    info["referencePartsNote"] = "Reference parts (reference=True) are shown, measured and checked for clearance with the design, but never printed: the printing layout and exports for printing leave them out."
                }
                guard artifact.print || preview.printVersion != true else {
                    info["layoutAvailability"] = "This program reads rad.print. Build its 3D printing version with rad_build print=true and inspect that artifact for its printing layout."
                    return try result(info)
                }
                let transforms = try await printLayoutTransforms(&artifact, preview: preview, solids: placed)
                let stats = (preview.partStats ?? []).filter { placed.contains($0.solidIndex) }
                let posed = CADMeasurements.posedParts(stats, transforms: transforms)
                let page = Array(posed.dropFirst(offset).prefix(limit))
                info["layout"] = page.map { part -> [String: Any] in
                    var item: [String: Any] = ["solidIndex": part.solidIndex, "bounds": part.bounds,
                                               "transform": transforms[part.solidIndex] ?? AssemblyPose.identityTransform]
                    if let owner = owners[part.solidIndex] { item["partID"] = owner.id; item["name"] = owner.name }
                    return item
                }
                if !posed.isEmpty {
                    info["plateBounds"] = (0..<3).map { axis in posed.map { $0.bounds[axis] }.min()! }
                        + (0..<3).map { axis in posed.map { $0.bounds[axis + 3] }.max()! }
                }
                info["transformLayout"] = "row-major 4x4 rigid placement from built coordinates onto the build plate"
                info["offset"] = offset; info["total"] = posed.count; info["hasMore"] = offset + page.count < posed.count
                info["plate"] = "Parts are oriented to minimise unsupported overhangs and packed on the build plate at z = 0 with \(Int(PrintLayout.gap)) mm gaps."
                return try result(info)
            default:
                let points = try vectors("points", arguments, maximum: 256)
                guard let points, !points.isEmpty else { throw RADError.message("Probe at least one world point.") }
                let directions = try vectors("directions", arguments, maximum: 256)
                if let directions {
                    guard directions.count == points.count, directions.allSatisfy({ $0.contains { $0 != 0 } }) else {
                        throw RADError.message("Give one nonzero ray direction per probed point.")
                    }
                }
                let maxDistance = try positive("maxDistance", arguments, maximum: 1_000_000)
                let probed = (solids ?? Set(0..<preview.solids)).sorted()
                guard points.count * probed.count <= 8_192 else {
                    throw RADError.message("Probe at most 8,192 point-part samples; choose fewer points or parts.")
                }
                let samples = try await probeParts(preview, solids: probed, points: points, directions: directions,
                                                   maxDistance: maxDistance)
                let owners = Self.owners(in: artifact.parts)
                let rows = points.indices.map { row -> [String: Any] in
                    var item: [String: Any] = ["point": points[row]]
                    if let directions { item["direction"] = directions[row] }
                    item["parts"] = probed.map { solid -> [String: Any] in
                        let sample = samples[row][solid]!
                        var value: [String: Any] = ["solidIndex": solid, "distance": sample.distance,
                                                    "inside": sample.distance < 0, "normal": sample.normal]
                        if let owner = owners[solid] { value["partID"] = owner.id }
                        if let hit = sample.hit {
                            value["hit"] = hit
                            value["hitDistance"] = sample.hitDistance ?? 0
                            if let point = sample.hitPoint { value["hitPoint"] = point }
                        }
                        return value
                    }
                    return item
                }
                return try result(["artifactID": artifact.id, "kind": kind, "coordinateSpace": space, "solidIndices": probed,
                                   "points": rows, "units": "mm",
                                   "note": "distance is the signed distance field value: negative inside, 0 on the surface. Smooth blends, booleans and offsets make it a lower bound of the true distance away from the surface, and implicit() parts report f / slopeBound, smaller still where distanceQuality is below 1; rays trace to the actual surface."])
            }
        case "rad_measure":
            try allowed(arguments, ["artifactID", "partIDs", "solidIndices", "points"])
            let requested = try measuredPoints(arguments)
            let artifact = try await artifact(arguments)
            let preview = artifact.preview!
            let solids = try selectedSolids(arguments, artifact: artifact, preview: preview) ?? []
            guard !solids.isEmpty || !requested.isEmpty else {
                throw RADError.message("Measure at least one part or surface point.")
            }
            let owners = Self.owners(in: artifact.parts)
            let transforms = preview.assemblyPose?.solidTransforms ?? [:]
            var pointSolids = try requested.map { item -> Int? in
                guard let id = item.partID else { return nil }
                guard let node = artifact.parts.flatMap(\.flattened).first(where: { $0.id == id }),
                      node.allSolidIndices.count == 1, let solid = node.allSolidIndices.first else {
                    throw RADError.message("‘\(id)’ is not a physical part of this artifact. Points snap to one physical part.")
                }
                return solid
            }
            for index in pointSolids.indices where pointSolids[index] == nil {
                pointSolids[index] = try await nearestPart(to: requested[index].point, in: preview)
            }
            // Snap every point to its part's surface in built coordinates.
            var points: [CADSurfacePoint] = []
            for (index, item) in requested.enumerated() {
                let solid = pointSolids[index]!
                let local = transforms[solid].map { Self.unposed(item.point, by: $0) } ?? item.point
                let snapped = try await engine.measure(preview, entities: [.point(solid: solid, point: local)],
                                                       solidTransforms: nil, python: python, threads: threads)
                try checkCancellation()
                guard let point = snapped.points?.first, point.isValid else {
                    throw RADError.message("The point could not be placed on the surface of its part.")
                }
                points.append(point)
            }
            guard Set(points).count == points.count else {
                throw RADError.message("Two measured points snap to the same surface point. Choose distinct points.")
            }
            let report = try await engine.measure(preview, points: points, solids: solids, python: python, threads: threads)
            try checkCancellation()
            var info = try json(report)
            info["artifactID"] = artifact.id
            info["coordinateSpace"] = transforms.isEmpty ? "builtGeometry" : "currentAssemblyPose"
            info["units"] = "mm"
            info["points"] = points.enumerated().map { index, point -> [String: Any] in
                let posed = CADMeasurements.posedPoint(point, transforms: transforms)
                var item: [String: Any] = ["requested": requested[index].point, "point": posed.position,
                                           "normal": posed.normal, "builtPoint": point.point, "solidIndex": point.solidIndex,
                                           "snapDistance": zip(posed.position, requested[index].point).map { ($0 - $1) * ($0 - $1) }.reduce(0, +).squareRoot()]
                if let owner = owners[point.solidIndex] { item["partID"] = owner.id; item["name"] = owner.name }
                return item
            }
            info["parts"] = solids.sorted().map { solid -> [String: Any] in
                var item: [String: Any] = ["solidIndex": solid]
                if let owner = owners[solid] { item["partID"] = owner.id; item["name"] = owner.name }
                return item
            }
            return try result(info)
        case "rad_validate":
            try allowed(arguments, ["artifactID", "cellSize"])
            let cellSize = try positive("cellSize", arguments, maximum: 1_000)
            let artifact = try await artifact(arguments)
            let preview = artifact.preview!
            let validation = try await engine.validate(preview, cellSize: cellSize, python: python, threads: threads)
            try checkCancellation()
            let owners = Self.owners(in: artifact.parts)
            var info: [String: Any] = ["artifactID": artifact.id, "valid": validation.ok, "cellSize": validation.cellSize,
                                       "warnings": validation.warnings, "units": "mm"]
            if let cell = preview.meshStatistics?.cellSize { info["previewCellSize"] = cell }
            info["parts"] = validation.parts.map { part -> [String: Any] in
                var item: [String: Any] = ["solidIndex": part.solid, "closed": part.closed, "boundaryEdges": part.boundaryEdges,
                                           "nonmanifoldEdges": part.nonmanifoldEdges, "pieces": part.components,
                                           "cavities": part.cavities ?? 0, "volume": part.volume,
                                           "previewVolume": part.previewVolume, "volumeChange": part.volumeChange]
                if let cavities = part.previewCavities { item["previewCavities"] = cavities }
                if let owner = owners[part.solid] { item["partID"] = owner.id; item["name"] = owner.name }
                return item
            }
            info["checks"] = ["program ran and every part has a finite, nonempty SDF",
                              "every part meshed closed at the preview resolution with positive volume",
                              "every part re-meshed at the validation cell size: closed surface, no boundary or non-manifold edges",
                              "the same number of pieces and sealed cavities as the preview mesh",
                              "volume within 2% of the preview mesh"]
            if !validation.ok {
                info["nextStep"] = "Thicken features thinner than about two preview cells, separate touching parts or surfaces, remove slivers, or close sealed gaps that thin (such as clearance between bodies unioned into one part; add bodies that move as separate parts), then rebuild and validate again."
            }
            return try result(info)
        case "rad_verify":
            try allowed(arguments, ["artifactID", "programPath", "assetIDs", "print"])
            let model: ModelSpec
            let printing: Bool
            var parts: [PartNode]? = nil
            var info: [String: Any] = ["units": "mm"]
            if let path = try string("programPath", arguments) {
                guard arguments["artifactID"] == nil else {
                    throw RADError.message("Verify either an artifact or a program file, not both.")
                }
                let bytes = try workspace.read(path, maximumBytes: 200_000)
                guard let script = String(data: bytes, encoding: .utf8), !script.trimmingCharacters(in: .whitespacesAndNewlines).isEmpty else {
                    throw RADError.message("The CAD program must be nonempty UTF-8 Python text.")
                }
                let selected = try resolveAssets(ids: try strings("assetIDs", arguments) ?? (current?.inputAssets ?? []).map(\.id))
                model = ModelSpec(title: "Custom field verification", summary: "Custom fields verified by RAD.", units: "mm",
                                  script: script, inputAssets: selected.isEmpty ? nil : selected)
                printing = try bool("print", arguments) ?? false
                info["programPath"] = try workspace.url(path).path
            } else {
                guard arguments["assetIDs"] == nil, arguments["print"] == nil else {
                    throw RADError.message("assetIDs and print apply when verifying a program file; an artifact keeps its own inputs and build.")
                }
                let id = try string("artifactID", arguments) ?? "current"
                if let artifact = artifacts[id] {
                    model = artifact.model
                    printing = artifact.print
                    if !artifact.parts.isEmpty { parts = artifact.parts }
                } else if id == "current", let current {
                    model = current
                    printing = false
                } else {
                    throw RADError.message(id == "current" ? "There is no current design. Build a CAD program first." : "The requested CAD artifact is unavailable in this task.")
                }
                info["artifactID"] = id
            }
            let verification = try await engine.verifyFields(model, print: printing, python: python, threads: threads)
            try checkCancellation()
            let owners = Self.owners(in: parts ?? verification.parts ?? [])
            info["fieldCount"] = verification.fieldCount
            let complete = verification.fields.count == verification.fieldCount
            info["verified"] = complete && verification.fields.allSatisfy(\.ok)
            info["fields"] = verification.fields.map { field -> [String: Any] in
                var item: [String: Any] = ["ok": field.ok, "bounds": field.bounds,
                                           "parts": field.solids.map { solid -> [String: Any] in
                                               var part: [String: Any] = ["solidIndex": solid]
                                               if let owner = owners[solid] { part["partID"] = owner.id; part["name"] = owner.name }
                                               return part
                                           }]
                if let line = field.line { item["line"] = line }
                if let bound = field.slopeBound { item["slopeBound"] = bound }
                if let range = field.valueRange { item["valueRange"] = range }
                if let gradient = field.gradient {
                    item["gradientLength"] = ["min": gradient.min, "median": gradient.median, "max": gradient.max]
                }
                if let quality = field.quality { item["distanceQuality"] = quality }
                if let boxes = field.boxes { item["boxes"] = boxes }
                if let message = field.message { item["problem"] = message }
                if let region = field.region { item["region"] = region }
                return item
            }
            if !complete {
                info["fieldsTruncated"] = "Only the first \(verification.fields.count) of \(verification.fieldCount) implicit() shapes were verified, so verified is false; rad_build proves every field when the design loads and fails on any without a bound."
            }
            if verification.fieldCount == 0 {
                info["note"] = "The program has no implicit() shapes."
            } else {
                info["interpretation"] = "slopeBound is the proven upper bound of |grad f| over the bounds; inside the bounds the solid's field is max(f / slopeBound, signed distance to the bounds box), and outside them hypot(f(c) / slopeBound, d) where f(c) > 0, else d (c the nearest point of the box, d the distance to it): a distance bound. Outward offsets, shells, smooth blends and clearance pockets that reach past the bounds bulge past the requested surface there (by up to about 0.4 of the reach for a true distance, more where distanceQuality is lower), so make the bounds enclose the solid by more than that reach divided by distanceQuality. distanceQuality is the median sampled gradient length / slopeBound: 1 for a true distance. Below 1 the solid's field underestimates distances, so its outward offsets, shells, smooth blend radii and clearance pockets come out larger than requested (by up to about 1/distanceQuality), inward offsets remove more, and probed distances are smaller than true distances; to compensate, multiply requested distances by about distanceQuality, or build the thickness into the field. A field without a finite slope bound fails rad_build: change the operation named in problem within region, or shrink the bounds."
            }
            return try result(info)
        case "rad_render":
            try allowed(arguments, ["artifactID", "view", "projection", "display", "width", "height", "solidIndices",
                                    "yawDegrees", "elevationDegrees", "rollDegrees", "cameraPosition", "cameraTarget", "zoom",
                                    "jointPositions", "drivingJoint", "printLayout"])
            let printLayout = try bool("printLayout", arguments) ?? false
            let view = try string("view", arguments) ?? "isometric"
            guard ["isometric", "front", "back", "top", "bottom", "left", "right"].contains(view),
                  let projection = ViewportProjectionMode(rawValue: try string("projection", arguments) ?? "isometric"),
                  let display = ViewportDisplayMode(rawValue: try string("display", arguments) ?? "edges") else {
                throw RADError.message("Choose a standard camera view, perspective/isometric projection, and a supported rendering mode.")
            }
            let width = try integer("width", arguments, fallback: 1024, range: 256...2048)
            let height = try integer("height", arguments, fallback: 768, range: 256...2048)
            let camera = try renderCamera(arguments)
            let jointPositions = try renderJointPositions(arguments)
            let drivingJoint = try drivingJoint(arguments, jointPositions: jointPositions)
            var artifact = try await artifact(arguments)
            var preview = artifact.preview!
            // The printing layout leaves reference parts out.
            var omitted = Set<Int>()
            if printLayout {
                guard jointPositions == nil else {
                    throw RADError.message("Render printLayout without jointPositions; the printing layout replaces assembly poses.")
                }
                guard artifact.print || preview.printVersion != true else {
                    throw RADError.message("This program reads rad.print. Build its 3D printing version with rad_build print=true and render that artifact with printLayout.")
                }
                omitted = Set(preview.referenceSolids ?? [])
                preview.assemblyPose = AssemblyPose(positions: [:],
                    solidTransforms: try await printLayoutTransforms(&artifact, preview: preview,
                        solids: Set(0..<preview.solids).subtracting(omitted)))
            } else if let jointPositions {
                preview.assemblyPose = try await solvedPose(artifact, preview: preview, jointPositions: jointPositions,
                                                            driving: drivingJoint)
            }
            let selected = try indices(arguments, preview: preview, optional: true)
            let hidden = (selected.map { Set(0..<preview.solids).subtracting($0) } ?? []).union(omitted)
            guard hidden.count < preview.solids else {
                throw RADError.message("The printing layout leaves out reference parts, and every requested part is a reference part.")
            }
            let captured = try await CADAgentRenderer.renderWithMetadata(preview: preview, view: view, projection: projection,
                display: display, hiddenSolids: hidden, solidColors: renderColors(for: artifact), width: width, height: height, camera: camera)
            try checkCancellation()
            let url = try workspace.write(captured.png, path: "renders/\(UUID().uuidString).png")
            var metadata = captured.metadata
            metadata.merge(["artifactID": artifact.id, "view": view, "projection": projection.rawValue,
                            "display": display.rawValue, "width": width, "height": height,
                            "zoom": camera.zoom, "customCamera": camera.isCustom,
                            "source": "RAD offscreen renderer"]) { _, supplied in supplied }
            metadata["cameraParameters"] = arguments.filter {
                ["yawDegrees", "elevationDegrees", "rollDegrees", "cameraPosition", "cameraTarget", "zoom"].contains($0.key)
            }
            if printLayout {
                metadata["printLayout"] = true
                if !omitted.isEmpty { metadata["referenceSolidsLeftOut"] = omitted.sorted() }
            } else if let pose = preview.assemblyPose { metadata["jointPositions"] = pose.positions }
            return try imageResult(url, metadata: metadata)
        case "rad_check_motion":
            try allowed(arguments, ["artifactID", "joint", "from", "to", "steps", "pairs", "jointPositions", "requiredClearance"])
            let jointID = try string("joint", arguments, required: true)!
            let from = try finite("from", arguments), to = try finite("to", arguments)
            let steps = try integer("steps", arguments, fallback: 73, range: 2...361)
            let required = try nonnegative("requiredClearance", arguments, maximum: 1_000) ?? 0
            guard let rawPairs = arguments["pairs"] as? [Any], (1...8).contains(rawPairs.count) else {
                throw RADError.message("Check between 1 and 8 pairs of physical part IDs.")
            }
            let pairIDs = try rawPairs.map { raw -> [String] in
                guard let pair = raw as? [Any], pair.count == 2,
                      let ids = pair as? [String], ids.allSatisfy({ !$0.isEmpty && $0.utf8.count <= 1024 }), ids[0] != ids[1] else {
                    throw RADError.message("Each clearance pair requires two different physical part IDs.")
                }
                return ids
            }
            let overrides = try renderJointPositions(arguments) ?? [:]
            let artifact = try await artifact(arguments)
            let preview = artifact.preview!
            guard let motion = preview.assemblyMotion else { throw RADError.message("The design has no assembly joints to move.") }
            guard let joint = motion.joints.first(where: { $0.id == jointID }), joint.isMovable else {
                throw RADError.message("Sweep an existing movable joint of this design.")
            }
            try motion.requireIndependent(jointID)
            for id in overrides.keys.sorted() { try motion.requireIndependent(id) }
            let owners = Dictionary(artifact.parts.flatMap(\.flattened).filter { !$0.isGroup && !$0.solidIndices.isEmpty }
                .map { ($0.id, $0) }, uniquingKeysWith: { first, _ in first })
            let pairs = try pairIDs.map { ids -> [[Int]] in
                try ids.map { id in
                    guard let owner = owners[id] else { throw RADError.message("‘\(id)’ is not a physical part of this design.") }
                    return owner.solidIndices.sorted()
                }
            }
            guard pairs.allSatisfy({ Set($0[0]).isDisjoint(with: $0[1]) }) else {
                throw RADError.message("Each clearance pair requires two different physical parts.")
            }
            let involved = Set(pairs.flatMap { $0.flatMap { $0 } })
            let parts = artifact.parts
            let start = (preview.assemblyPose?.positions ?? [:]).merging(overrides) { _, requested in requested }
            let solving = Task.detached(priority: .userInitiated) { () throws -> [AssemblyPose?] in
                var hint = start, poses = [AssemblyPose?]()
                for step in 0..<steps {
                    try Task.checkCancellation()
                    hint[jointID] = from + (to - from) * Double(step) / Double(steps - 1)
                    do {
                        let pose = try motion.solve(parts: parts, positions: hint, driving: jointID)
                        // Carry closure continuity from the last reachable pose.
                        hint = pose.positions; poses.append(pose)
                    } catch is CancellationError { throw CancellationError() }
                    catch { poses.append(nil) }
                }
                return poses
            }
            let poses = try await withTaskCancellationHandler(operation: { try await solving.value }, onCancel: { solving.cancel() })
            try checkCancellation()
            let reachable = poses.compactMap { $0 }
            guard !reachable.isEmpty else {
                throw RADError.message("\(joint.name) cannot reach any position in this range within the design's joint constraints and motion limits.")
            }
            let measured = try await engine.clearance(preview, pairs: pairs,
                poses: reachable.map { $0.solidTransforms.filter { involved.contains($0.key) } }, python: python, threads: threads)
            try checkCancellation()
            let positions = (0..<steps).map { from + (to - from) * Double($0) / Double(steps - 1) }
            // Report only joints that actually move during the sweep.
            let moving = motion.joints.map(\.id).filter { id in
                let values = reachable.compactMap { $0.positions[id] }
                return values.contains { abs($0 - values[0]) > 1e-9 }
            }
            var samples = [[String: Any]](), unreachable = [Double]()
            var cursor = 0
            for (index, pose) in poses.enumerated() {
                guard let pose else { unreachable.append(positions[index]); continue }
                let clearance = measured[cursor]; cursor += 1
                samples.append(["position": positions[index],
                                "jointPositions": Dictionary(uniqueKeysWithValues: moving.map { ($0, pose.positions[$0] ?? 0) }),
                                "distances": clearance.map(\.distance),
                                "overlapVolumes": clearance.map { item -> Any in
                                    guard item.distance <= Self.contactTolerance else { return 0.0 }
                                    return item.overlapVolume.map { $0 as Any } ?? NSNull()
                                }])
            }
            let reachedPositions = poses.indices.filter { poses[$0] != nil }.map { positions[$0] }
            let contact = Self.contactTolerance
            let summaries = pairIDs.indices.map { pair -> [String: Any] in
                let values = measured.map { $0[pair] }
                let minimum = values.indices.min { values[$0].distance < values[$1].distance }!
                let interfering = values.indices.filter { values[$0].distance <= contact && (values[$0].overlapVolume ?? 1) > contact }
                let touching = values.indices.filter { values[$0].distance <= contact && (values[$0].overlapVolume ?? 1) <= contact }
                let tight = values.indices.filter { values[$0].distance > contact && values[$0].distance < required }
                var summary: [String: Any] = [
                    "parts": pairIDs[pair], "minimumDistance": values[minimum].distance,
                    "minimumAt": reachedPositions[minimum], "closestPoints": values[minimum].closestPoints,
                    "interferencePositions": interfering.map { reachedPositions[$0] },
                    "contactPositions": touching.map { reachedPositions[$0] },
                    "tightPositions": tight.map { reachedPositions[$0] },
                    "meetsRequiredClearance": values[minimum].distance > contact && values[minimum].distance >= required,
                    "maximumOverlapVolume": interfering.compactMap { values[$0].overlapVolume }.max() ?? 0]
                if values.contains(where: { $0.distance <= contact && $0.overlapVolume == nil }) {
                    summary["unresolvedOverlap"] = true
                }
                return summary
            }
            return try result(["artifactID": artifact.id, "joint": jointID, "positionUnit": joint.unit, "from": from, "to": to,
                               "steps": steps, "pairs": summaries, "samples": samples, "unreachablePositions": unreachable,
                               "requiredClearance": required, "contactTolerance": contact, "units": "mm"])
        case "rad_screenshot":
            try allowed(arguments, [])
            if let viewport {
                let captured = try await CADAgentRenderer.captureWithMetadata(view: viewport)
                try checkCancellation()
                let url = try workspace.write(captured.png, path: "renders/\(UUID().uuidString).png")
                var metadata = Self.identifiedSelection(captured.metadata, parts: artifacts["current"]?.parts ?? [])
                metadata["source"] = "Current RAD viewport"
                metadata["artifactID"] = "current"
                return try imageResult(url, metadata: metadata)
            }
            let artifact = try await artifact([:])
            let png = try await CADAgentRenderer.renderPNG(preview: artifact.preview!, hiddenSolids: hiddenSolids, solidColors: renderColors(for: artifact))
            try checkCancellation()
            let url = try workspace.write(png, path: "renders/\(UUID().uuidString).png")
            return try imageResult(url, metadata: ["source": "RAD offscreen fallback; no live viewport is attached", "artifactID": artifact.id])
        case "rad_export_3mf":
            try allowed(arguments, ["artifactID", "path", "solidIndices", "partIDs", "individualParts", "jointPositions",
                                    "drivingJoint", "printLayout"])
            let individual = try bool("individualParts", arguments) ?? false
            let printLayout = try bool("printLayout", arguments) ?? false
            let jointPositions = try renderJointPositions(arguments)
            let drivingJoint = try drivingJoint(arguments, jointPositions: jointPositions)
            guard !printLayout || jointPositions == nil else {
                throw RADError.message("Export printLayout without jointPositions; the printing layout replaces assembly poses.")
            }
            let requestedID = try string("artifactID", arguments) ?? "current"
            let previous = artifacts[requestedID]
            var artifact = try await artifact(arguments)
            let preview = artifact.preview!
            var indices = try indices(arguments, preview: preview, optional: true)
            if let requested = indices, let previous, previous.preview == nil, !previous.parts.isEmpty {
                indices = try Self.remappedExportIndices(requested, from: previous.parts, to: artifact.parts)
            }
            if let parts = try partSolids(arguments, artifact: artifact, preview: preview) {
                indices = (indices ?? []).union(parts)
            }
            // Nil keeps the artifact's own assembly pose.
            var transforms: [Int: [Double]]?
            var placement = (preview.assemblyPose?.solidTransforms.isEmpty ?? true) ? "builtGeometry" : "currentAssemblyPose"
            var leftOut: [Int] = []
            if printLayout {
                guard artifact.print || preview.printVersion != true else {
                    throw RADError.message("This program reads rad.print. Build its 3D printing version with rad_build print=true and export that artifact with printLayout.")
                }
                // Reference parts are never printed.
                let requested = indices ?? Set(0..<preview.solids)
                let printed = PrintLayout.printedSolids(requested, of: preview)
                guard !printed.isEmpty else {
                    throw RADError.message("The printing layout leaves out reference parts, and every requested part is a reference part.")
                }
                leftOut = requested.subtracting(printed).sorted()
                if !leftOut.isEmpty { indices = printed }
                transforms = try await printLayoutTransforms(&artifact, preview: preview, solids: printed)
                placement = "printLayout"
            } else if let jointPositions {
                transforms = try await solvedPose(artifact, preview: preview, jointPositions: jointPositions,
                                                  driving: drivingJoint).solidTransforms
                placement = "jointPositions"
            }
            let partPresentation = ThreeMFExport.presentation(in: artifact.parts, overrides: artifact.partColors)
            var info: [String: Any] = ["artifactID": artifact.id, "solidIndices": (indices ?? Set(0..<preview.solids)).sorted(),
                                       "placement": placement, "format": "3MF", "units": "mm"]
            if !leftOut.isEmpty { info["referenceSolidsLeftOut"] = leftOut }
            if individual {
                let parts = ThreeMFExport.parts(in: artifact.parts, selected: indices, solidCount: preview.solids)
                let files = try await engine.export3MFParts(preview, parts: parts.map(\.solidIndices),
                                                            partPresentation: partPresentation, solidTransforms: transforms,
                                                            python: python, threads: threads)
                try checkCancellation()
                var folder = try string("path", arguments) ?? "exports/\(artifact.id)-parts"
                if folder.lowercased().hasSuffix(".3mf") { folder.removeLast(4) }
                let owners = Self.owners(in: artifact.parts)
                generatedGeometry.formUnion(files.map { SHA256.hexDigest($0) })
                info["files"] = try zip(parts, files).map { part, data -> [String: Any] in
                    let url = try workspace.write(data, path: folder + "/" + part.fileName)
                    var item: [String: Any] = ["path": url.path, "name": part.name, "solidIndices": part.solidIndices,
                                               "byteCount": data.count]
                    if let owner = part.solidIndices.first.flatMap({ owners[$0] }) { item["partID"] = owner.id }
                    return item
                }
                info["byteCount"] = files.reduce(0) { $0 + $1.count }
            } else {
                let data = try await engine.export3MF(preview, solidIndices: indices,
                                                      partPresentation: partPresentation, solidTransforms: transforms,
                                                      python: python, threads: threads)
                try checkCancellation()
                generatedGeometry.insert(SHA256.hexDigest(data))
                let url = try workspace.write(data, path: try string("path", arguments) ?? "exports/\(artifact.id).3mf")
                info["path"] = url.path
                info["byteCount"] = data.count
            }
            return try result(info)
        default: throw RADError.message("Unknown RAD tool ‘\(name)’.")
        }
    }

    private func stageVersion(_ version: ProjectVersionContext.Version) async throws -> CADAgentToolResult {
        let workspace = self.workspace
        let prepared = try await versionWork { try CADAgentVersionFiles.prepare(version, workspace: workspace) }
        try checkCancellation()
        var sizes = versionInputSizes
        for (path, count) in prepared.inputSizes { sizes[path] = count }
        var total = 0
        for count in sizes.values {
            guard count <= ProjectStore.maximumFileBytes - total else {
                throw RADError.message("Saved version inputs exceed the design's 1,024 MB storage limit.")
            }
            total += count
        }
        versionInputSizes = sizes
        _ = try await versionWork {
            try CADAgentVersionFiles.write(prepared, workspace: workspace)
            return prepared
        }
        try checkCancellation()
        guard let object = try JSONSerialization.jsonObject(with: prepared.payload) as? [String: Any] else {
            throw RADError.message("The saved version metadata is unavailable.")
        }
        return try result(object)
    }

    private func versionWork(_ operation: @escaping @Sendable () throws -> CADAgentVersionFiles.Prepared) async throws -> CADAgentVersionFiles.Prepared {
        let id = UUID()
        let task = Task.detached(priority: .userInitiated, operation: operation)
        versionTasks[id] = task
        defer { versionTasks.removeValue(forKey: id) }
        return try await withTaskCancellationHandler(operation: { try await task.value }, onCancel: { task.cancel() })
    }

    private func build(_ model: ModelSpec, id: String? = nil, versionID: UUID? = nil,
                       print: Bool = false) async throws -> Artifact {
        let id = id ?? UUID().uuidString
        var preview: PreviewMesh
        if let cached = validatedPreview(for: model, print: print) { preview = cached }
        else { preview = try await engine.buildPreview(model, print: print, python: python, threads: threads) }
        try checkCancellation()
        if print {
            // The printing version must keep the ordinary build's parts.
            let design: PreviewMesh
            if let cached = validatedPreview(for: model) { design = cached }
            else { design = try await build(model, versionID: versionID).preview! }
            do { try PrintLayout.validateParts(of: preview, matching: design) }
            catch let failure as EngineBuildFailure { throw RADError.message(EngineFailureDiagnostic.repairText(for: failure)) }
        }
        runtime = preview.runtime
        let programURL = try workspace.write(Data(model.script.utf8), path: "artifacts/\(id)/program.py")
        let metadataURL = workspace.root.appendingPathComponent("artifacts/\(id)/metadata.json")
        let defaults = Self.parts(in: preview)
        let previous = artifacts[id]
        let version = versionID.flatMap { identity in versions.versions.first { $0.id == identity } }
        let layout = previous?.partLayout ?? version?.partLayout ?? (versionID == nil ? partLayout : nil)
            ?? previous.flatMap { $0.parts.isEmpty ? nil : PartLayout(nodes: $0.parts) }
        let resolvedLayout = layout.map {
            $0.remappingGenericParts(to: defaults).reconciled(with: defaults,
                allowingPlacementChanges: Set(model.reorganizePartIDs ?? []))
        } ?? PartLayout(nodes: defaults)
        let nodes = resolvedLayout.nodes
        if let version, let positions = version.jointPositions,
           constructionMatches(version.model, model), let motion = preview.assemblyMotion {
            let solving = Task.detached(priority: .userInitiated) { try motion.solve(parts: nodes, positions: positions) }
            preview.assemblyPose = try await withTaskCancellationHandler(operation: { try await solving.value }, onCancel: { solving.cancel() })
            try checkCancellation()
        }
        let inherited = previous?.partColors ?? (versionID != nil ? version?.partColors ?? [:]
            : artifacts["current"]?.partColors ?? initialPartColors)
        let physicalIDs = Self.physicalIDs(in: nodes)
        let patch = previous?.colorChanges ?? (versionID == nil ? artifacts["current"]?.colorChanges ?? [] : [])
        let artifact = Artifact(id: id, model: model, preview: preview, programURL: programURL, metadataURL: metadataURL,
            versionID: versionID, parts: nodes, partLayout: resolvedLayout,
            partColors: Self.completePartColors(in: nodes, colors: inherited),
            colorChanges: patch.filter { physicalIDs.contains($0.partID) }, print: print)
        _ = try workspace.write(JSONSerialization.data(withJSONObject: summary(artifact), options: [.sortedKeys, .prettyPrinted]), path: "artifacts/\(id)/metadata.json")
        artifacts[id] = artifact
        if !artifactOrder.contains(id) { artifactOrder.append(id) }
        // Retain recipes for replay while bounding heavyweight display caches.
        // An evicted candidate rebuilds when inspected or committed, and then
        // counts as the most recently built one.
        previewOrder.removeAll { $0 == id }
        previewOrder.append(id)
        for old in previewOrder.dropLast(Self.retainedCandidatePreviews) where old != "current" { artifacts[old]?.preview = nil }
        previewOrder = previewOrder.filter { artifacts[$0]?.preview != nil }
        return artifact
    }

    private static func parts(in preview: PreviewMesh?) -> [PartNode] {
        guard let preview else { return [] }
        return preview.parts ?? (0..<preview.solids).map {
            PartNode(id: "solid/\($0)", name: "Part \($0 + 1)", solidIndices: [$0])
        }
    }

    private static func physicalIDs(in nodes: [PartNode]) -> Set<String> {
        Set(nodes.flatMap(\.flattened).filter { !$0.solidIndices.isEmpty }.map(\.id))
    }

    /// Color lookup can expose saved ownership before a cold replay. Keep an
    /// export subset attached to those physical identities if native order changes.
    private static func remappedExportIndices(_ requested: Set<Int>, from previous: [PartNode],
                                             to current: [PartNode]) throws -> Set<Int> {
        let owners = Dictionary(uniqueKeysWithValues: current.flatMap(\.flattened).filter {
            !$0.solidIndices.isEmpty
        }.map { ($0.id, $0.solidIndices) })
        var result = Set<Int>()
        for part in previous.flatMap(\.flattened) where !Set(part.solidIndices).isDisjoint(with: requested) {
            guard let indices = owners[part.id], indices.count == part.solidIndices.count else {
                throw RADError.message("Part ownership changed while rebuilding this artifact. Inspect its current parts before exporting a subset.")
            }
            for (old, new) in zip(part.solidIndices, indices) where requested.contains(old) { result.insert(new) }
        }
        guard result.count == requested.count else {
            throw RADError.message("The selected export parts are no longer available. Inspect the artifact's current parts.")
        }
        return result
    }

    /// Distances at most this far apart are contact, as the worker reports.
    static let contactTolerance = 1e-6
    /// Build summaries list at most this many parts' statistics.
    static let summaryParts = 64

    /// The agent-neutral radsdf modeling guide: the end of the model
    /// instructions from its heading, staged for every agent by rad_runtime.
    static func modelingGuide() throws -> String {
        guard let url = Bundle.module.url(forResource: "model-instructions", withExtension: "md", subdirectory: "Resources"),
              let text = try? String(contentsOf: url, encoding: .utf8),
              let start = text.range(of: "\n" + modelingGuideHeading + "\n") else {
            throw RADError.message("RAD's radsdf modeling guide is unavailable.")
        }
        return String(text[text.index(after: start.lowerBound)...])
    }
    static let modelingGuideHeading = "RADSDF MODELING GUIDE"

    /// The physical solid owning each solid index, first in tree order.
    private static func owners(in nodes: [PartNode]) -> [Int: PartNode] {
        var owners: [Int: PartNode] = [:]
        for node in nodes.flatMap(\.flattened) {
            for solid in node.solidIndices where owners[solid] == nil { owners[solid] = node }
        }
        return owners
    }

    /// Solids named by stable part or folder IDs; folders select their physical parts.
    private func partSolids(_ values: [String: Any], artifact: Artifact, preview: PreviewMesh) throws -> Set<Int>? {
        guard let ids = try strings("partIDs", values) else { return nil }
        guard !ids.isEmpty else { throw RADError.message("Choose at least one part ID.") }
        let nodes = Dictionary(artifact.parts.flatMap(\.flattened).map { ($0.id, $0) }, uniquingKeysWith: { first, _ in first })
        var solids = Set<Int>()
        for id in ids {
            guard let node = nodes[id], !node.allSolidIndices.isEmpty else {
                throw RADError.message("‘\(id)’ is not a part or folder of this artifact. Use rad_inspect kind=parts or rad_get_part_colors for its part IDs.")
            }
            solids.formUnion(node.allSolidIndices)
        }
        guard solids.allSatisfy({ (0..<preview.solids).contains($0) }) else {
            throw RADError.message("The artifact's parts do not match its solids. Rebuild it.")
        }
        return solids
    }

    /// The union of `solidIndices` and `partIDs`, or nil when neither is given.
    private func selectedSolids(_ values: [String: Any], artifact: Artifact, preview: PreviewMesh) throws -> Set<Int>? {
        let indices = try indices(values, preview: preview, optional: true)
        guard let parts = try partSolids(values, artifact: artifact, preview: preview) else { return indices }
        return (indices ?? []).union(parts)
    }

    /// Part statistics with their stable identities and current colors.
    private func partItems(_ parts: [CADPart], artifact: Artifact) throws -> [[String: Any]] {
        let owners = Self.owners(in: artifact.parts)
        let references = Set(artifact.preview?.referenceSolids ?? [])
        return try parts.map { part in
            var item: [String: Any] = ["solidIndex": part.solidIndex, "volume": part.volume, "area": part.area,
                                       "bounds": part.bounds, "centroid": part.centroid, "triangles": part.triangles,
                                       "pieces": part.components]
            if let cavities = part.cavities, cavities > 0 { item["cavities"] = cavities }
            if references.contains(part.solidIndex) { item["reference"] = true }
            if let owner = owners[part.solidIndex] {
                item["partID"] = owner.id; item["name"] = owner.name
                item.merge(try colorInfo(owner, in: artifact)) { _, color in color }
            }
            return item
        }
    }

    private func motionInfo(_ artifact: Artifact) throws -> [String: Any] {
        let preview = artifact.preview!
        var info: [String: Any] = ["artifactID": artifact.id, "kind": "motion"]
        guard let motion = preview.assemblyMotion else {
            info["hasMotion"] = false
            info["nextStep"] = "This artifact defines no assembly motion. Assign a Motion to the result Assembly's motion to add joints."
            return info
        }
        let positions = preview.assemblyPose?.positions ?? Dictionary(uniqueKeysWithValues: motion.joints.map { ($0.id, $0.position) })
        info["hasMotion"] = true
        info["assemblyMotion"] = try json(motion)
        info["jointPositions"] = positions
        info["joints"] = motion.joints.map { joint -> [String: Any] in
            var item: [String: Any] = ["id": joint.id, "name": joint.name, "type": joint.type.rawValue,
                                       "unit": joint.type == .revolute ? "degrees" : "mm",
                                       "position": positions[joint.id] ?? joint.position, "movable": joint.isMovable]
            if let limits = joint.limits { item["limits"] = limits }
            if let driver = motion.driver(of: joint.id) { item["drivenBy"] = driver.id }
            else { item["independent"] = joint.isMovable }
            return item
        }
        info["coordinateSpace"] = "Connector frames use the built design's world coordinates."
        return info
    }

    /// Every requested solid's printing orientation, packed on the build
    /// plate. Orientations are found once per artifact build.
    private func printLayoutTransforms(_ artifact: inout Artifact, preview: PreviewMesh, solids: Set<Int>) async throws -> [Int: [Double]] {
        let missing = solids.subtracting(artifact.printOrientations.keys)
        if !missing.isEmpty {
            let found = try await engine.printOrientations(preview, solids: missing, python: python, threads: threads)
            try checkCancellation()
            artifact.printOrientations.merge(found) { current, _ in current }
            if artifacts[artifact.id]?.preview?.buildID == preview.buildID { artifacts[artifact.id] = artifact }
        }
        return PrintLayout.transforms(for: solids, orientations: artifact.printOrientations)
    }

    /// The artifact's pose with independent joint overrides, solved off the main actor.
    private func solvedPose(_ artifact: Artifact, preview: PreviewMesh, jointPositions: [String: Double],
                            driving: String?) async throws -> AssemblyPose {
        guard let motion = preview.assemblyMotion else {
            throw RADError.message("The design has no assembly joints to position.")
        }
        for id in jointPositions.keys.sorted() { try motion.requireIndependent(id) }
        let parts = artifact.parts
        let positions = (preview.assemblyPose?.positions ?? [:]).merging(jointPositions) { _, requested in requested }
        let solving = Task.detached(priority: .userInitiated) {
            try motion.solve(parts: parts, positions: positions, driving: driving)
        }
        let pose = try await withTaskCancellationHandler(operation: { try await solving.value }, onCancel: { solving.cancel() })
        try checkCancellation()
        return pose
    }

    private func drivingJoint(_ values: [String: Any], jointPositions: [String: Double]?) throws -> String? {
        guard let drivingJoint = try string("drivingJoint", values) else { return nil }
        guard drivingJoint.utf8.count <= 1024, jointPositions?[drivingJoint] != nil else {
            throw RADError.message("The assembly drivingJoint must name a joint supplied in jointPositions.")
        }
        return drivingJoint
    }

    /// Capture metadata with the selected parts' identities, and the selected
    /// points' parts and world positions as displayed (posed by the capture's
    /// own solid transforms), ready for rad_measure.
    static func identifiedSelection(_ metadata: [String: Any], parts: [PartNode]) -> [String: Any] {
        var metadata = metadata
        let owners = owners(in: parts)
        func numbers(_ value: Any?, count: Int) -> [Double]? {
            guard let values = value as? [NSNumber], values.count == count else { return nil }
            return values.map(\.doubleValue)
        }
        var transforms: [Int: [Double]] = [:]
        for record in metadata["solidTransforms"] as? [[String: Any]] ?? [] {
            if let solid = record["solid"] as? Int, let matrix = numbers(record["matrix"], count: 16) { transforms[solid] = matrix }
        }
        if let solids = metadata["selectedSolidIndices"] as? [Int] {
            metadata["selectedParts"] = solids.map { solid -> [String: Any] in
                var item: [String: Any] = ["solidIndex": solid]
                if let owner = owners[solid] { item["partID"] = owner.id; item["name"] = owner.name }
                return item
            }
        }
        if let points = metadata["selectedPoints"] as? [[String: Any]] {
            metadata["selectedPoints"] = points.map { point -> [String: Any] in
                var item = point
                guard let solid = point["solidIndex"] as? Int, let local = numbers(point["point"], count: 3),
                      let normal = numbers(point["normal"], count: 3) else { return item }
                let matrix = transforms[solid] ?? AssemblyPose.identityTransform
                item["worldPoint"] = AssemblySolidTransform.point(local, by: matrix)
                item["worldNormal"] = AssemblySolidTransform.point(normal, by: matrix, direction: true)
                if let owner = owners[solid] { item["partID"] = owner.id; item["name"] = owner.name }
                return item
            }
        }
        if let edges = metadata["selectedEdges"] as? [[String: Any]] {
            metadata["selectedEdges"] = edges.map { edge in
                var item = edge
                guard let solid = edge["solidIndex"] as? Int, let owner = owners[solid] else { return item }
                item["partID"] = owner.id
                item["name"] = owner.name
                return item
            }
        }
        return metadata
    }

    /// Probe the parts' SDFs at world points of the artifact's pose. Each
    /// part is sampled in its built coordinates, with one worker request per
    /// distinct placement; normals and ray hits are posed back.
    private func probeParts(_ preview: PreviewMesh, solids: [Int], points: [[Double]], directions: [[Double]]?,
                            maxDistance: Double?) async throws -> [[Int: EngineService.ProbeSample]] {
        let transforms = preview.assemblyPose?.solidTransforms ?? [:]
        var groups: [[Double]: [Int]] = [:]
        for solid in solids { groups[transforms[solid] ?? AssemblyPose.identityTransform, default: []].append(solid) }
        var samples = Array(repeating: [Int: EngineService.ProbeSample](), count: points.count)
        for (matrix, members) in groups.sorted(by: { $0.value[0] < $1.value[0] }) {
            let moved = matrix != AssemblyPose.identityTransform
            let local = moved ? points.map { Self.unposed($0, by: matrix) } : points
            let rays = directions.map { list in moved ? list.map { Self.unposed($0, by: matrix, direction: true) } : list }
            let probe = try await engine.probe(preview, solids: Set(members), points: local, directions: rays,
                                               maxDistance: maxDistance, python: python, threads: threads)
            try checkCancellation()
            for (row, values) in probe.samples.enumerated() {
                for (column, solid) in probe.solids.enumerated() {
                    var sample = values[column]
                    if moved {
                        sample.normal = AssemblySolidTransform.point(sample.normal, by: matrix, direction: true)
                        sample.hitPoint = sample.hitPoint.map { AssemblySolidTransform.point($0, by: matrix) }
                    }
                    samples[row][solid] = sample
                }
            }
        }
        return samples
    }

    /// The part whose surface is nearest a world point, among the parts whose
    /// posed preview bounds lie within a small margin of it.
    private func nearestPart(to point: [Double], in preview: PreviewMesh) async throws -> Int {
        let margin = max(1, 4 * (preview.meshStatistics?.cellSize ?? 0))
        let stats = CADMeasurements.posedParts(preview.partStats ?? [], transforms: preview.assemblyPose?.solidTransforms ?? [:])
        let candidates = stats.filter { part in
            part.bounds.count == 6 && (0..<3).allSatisfy { part.bounds[$0] - margin <= point[$0] && point[$0] <= part.bounds[$0 + 3] + margin }
        }.map(\.solidIndex).sorted()
        guard !candidates.isEmpty else {
            throw RADError.message("The point \(point) is not near any part. Give a point on a part's surface, or its partID.")
        }
        guard candidates.count > 1 else { return candidates[0] }
        let samples = try await probeParts(preview, solids: candidates, points: [point], directions: nil, maxDistance: nil)[0]
        return candidates.min { abs(samples[$0]!.distance) < abs(samples[$1]!.distance) }!
    }

    private func measuredPoints(_ values: [String: Any]) throws -> [(point: [Double], partID: String?)] {
        guard let raw = values["points"] else { return [] }
        guard let list = raw as? [Any], (1...3).contains(list.count) else {
            throw RADError.message("Measure between one and three surface points.")
        }
        return try list.map { entry in
            guard let item = entry as? [String: Any] else {
                throw RADError.message("Each measured point is an object with point and an optional partID.")
            }
            try allowed(item, ["point", "partID"])
            guard let raw = item["point"], let point = try vectors("point", ["point": [raw]], maximum: 1)?.first else {
                throw RADError.message("Each measured point needs point: three world millimetre coordinates.")
            }
            return (point, try string("partID", item))
        }
    }

    /// An array of [x, y, z] vectors within the design limits.
    private func vectors(_ key: String, _ values: [String: Any], maximum: Int) throws -> [[Double]]? {
        guard let raw = values[key] else { return nil }
        let invalid = RADError.message("‘\(key)’ must list at most \(maximum) points of three finite millimetre coordinates within ±1,000,000.")
        guard let list = raw as? [Any], !list.isEmpty, list.count <= maximum else { throw invalid }
        return try list.map { entry in
            guard let vector = entry as? [Any], vector.count == 3 else { throw invalid }
            return try vector.map { value in
                guard let number = value as? NSNumber, CFGetTypeID(number) != CFBooleanGetTypeID(),
                      number.doubleValue.isFinite, abs(number.doubleValue) <= 1_000_000 else { throw invalid }
                return number.doubleValue
            }
        }
    }

    private func positive(_ key: String, _ values: [String: Any], maximum: Double) throws -> Double? {
        guard let value = try nonnegative(key, values, maximum: maximum) else { return nil }
        guard value > 0 else { throw RADError.message("The RAD tool argument ‘\(key)’ must be positive.") }
        return value
    }

    private func nonnegative(_ key: String, _ values: [String: Any], maximum: Double) throws -> Double? {
        guard let raw = values[key] else { return nil }
        guard let number = raw as? NSNumber, CFGetTypeID(number) != CFBooleanGetTypeID(), number.doubleValue.isFinite,
              number.doubleValue >= 0, number.doubleValue <= maximum else {
            throw RADError.message("The RAD tool argument ‘\(key)’ must be a number from 0 to \(Int(maximum)).")
        }
        return number.doubleValue
    }

    /// A posed world point (or direction) in its part's built coordinates:
    /// the inverse of a rigid row-major placement.
    private static func unposed(_ point: [Double], by matrix: [Double], direction: Bool = false) -> [Double] {
        guard point.count == 3, matrix.count == 16 else { return point }
        let vector = direction ? point : [point[0] - matrix[3], point[1] - matrix[7], point[2] - matrix[11]]
        return (0..<3).map { column in
            matrix[column] * vector[0] + matrix[4 + column] * vector[1] + matrix[8 + column] * vector[2]
        }
    }

    private func colorArtifact(_ arguments: [String: Any]) throws -> Artifact {
        let id = try string("artifactID", arguments) ?? "current"
        guard let artifact = artifacts[id] else {
            throw RADError.message(id == "current" ? "There is no current design with parts to color."
                : "The requested CAD artifact is unavailable in this task.")
        }
        return artifact
    }

    private func colorChanges(_ arguments: [String: Any]) throws -> [CADPartColorChange] {
        guard let values = arguments["changes"] as? [[String: Any]], !values.isEmpty, values.count <= 256 else {
            throw RADError.message("Provide between 1 and 256 part color assignments.")
        }
        return try values.map { value in
            try allowed(value, ["partID", "color"])
            let id = try string("partID", value, required: true)!
            guard let raw = value["color"] else { throw RADError.message("Each part color assignment requires color, or null to use the neutral default finish.") }
            if raw is NSNull { return CADPartColorChange(partID: id, color: nil) }
            guard let components = raw as? [String: Any], Set(components.keys) == ["red", "green", "blue"] else {
                throw RADError.message("Part colors require red, green and blue sRGB components between 0 and 1.")
            }
            func component(_ name: String) throws -> Double {
                guard let number = components[name] as? NSNumber, CFGetTypeID(number) != CFBooleanGetTypeID(),
                      number.doubleValue.isFinite, (0...1).contains(number.doubleValue) else {
                    throw RADError.message("Part color components must be finite sRGB numbers between 0 and 1.")
                }
                return number.doubleValue
            }
            return CADPartColorChange(partID: id, color: PartColor(red: try component("red"),
                green: try component("green"), blue: try component("blue")))
        }
    }

    private static func completePartColors(in nodes: [PartNode], colors: [String: PartColor]) -> [String: PartColor] {
        Dictionary(uniqueKeysWithValues: nodes.flatMap(\.flattened).filter { !$0.solidIndices.isEmpty }.map { node in
            (node.id, colors[node.id] ?? node.color ?? PartColorState.defaultFinish)
        })
    }

    private func colorInfo(_ node: PartNode, in artifact: Artifact) throws -> [String: Any] {
        ["color": try json(artifact.partColors[node.id] ?? PartColorState.defaultFinish)]
    }

    private func colorPage(_ artifact: Artifact, offset: Int, limit: Int) throws -> [String: Any] {
        var nodes: [(PartNode, String?)] = []
        func append(_ children: [PartNode], parent: String?) {
            for node in children { nodes.append((node, parent)); append(node.children ?? [], parent: node.id) }
        }
        append(artifact.parts, parent: nil)
        let page = try nodes.dropFirst(offset).prefix(limit).map { node, parent -> [String: Any] in
            var item: [String: Any] = ["partID": node.id, "name": node.name, "isGroup": node.isGroup,
                                      "solidIndices": node.solidIndices]
            if let parent { item["parentID"] = parent }
            if !node.solidIndices.isEmpty { item.merge(try colorInfo(node, in: artifact)) { _, color in color } }
            return item
        }
        return ["artifactID": artifact.id, "offset": offset, "total": nodes.count, "items": page,
                "hasMore": offset + page.count < nodes.count, "staged": !artifact.colorChanges.isEmpty]
    }

    private func renderColors(for artifact: Artifact) -> [Int: PartColor] {
        PartColorState.solidColors(in: artifact.parts, overrides: artifact.partColors)
    }

    private func artifact(_ arguments: [String: Any]) async throws -> Artifact {
        let id = try string("artifactID", arguments) ?? "current"
        if let artifact = artifacts[id] {
            return artifact.preview == nil
                ? try await build(artifact.model, id: id, versionID: artifact.versionID, print: artifact.print) : artifact
        }
        if id == "current", let current { return try await build(current, id: "current") }
        throw RADError.message(id == "current" ? "There is no current design. Build a CAD program first." : "The requested CAD artifact is unavailable in this task.")
    }

    private func summary(_ artifact: Artifact) -> [String: Any] {
        let preview = artifact.preview!
        var result: [String: Any] = ["artifactID": artifact.id, "units": "mm", "programPath": artifact.programURL.path,
                                    "metadataPath": artifact.metadataURL.path, "solids": preview.solids,
                                    "volume": preview.volume, "surfaceArea": preview.area, "triangles": preview.triangleCount,
                                    "buildSeconds": preview.buildSeconds, "threads": preview.threads,
                                    "assetIDs": (artifact.model.inputAssets ?? []).map(\.id), "warnings": preview.warnings ?? []]
        if let cellSize = preview.meshStatistics?.cellSize { result["meshCellSize"] = cellSize }
        if !(preview.assemblyPose?.solidTransforms.isEmpty ?? true) {
            result["builtBounds"] = preview.bounds; result["builtSize"] = preview.size
            result["boundsAvailability"] = "Use rad_inspect kind=parts for part bounds at the current assembly pose."
        } else {
            result["bounds"] = preview.bounds; result["size"] = preview.size
        }
        if let stats = preview.partStats {
            let posed = CADMeasurements.posedParts(Array(stats.prefix(Self.summaryParts)),
                                                   transforms: preview.assemblyPose?.solidTransforms ?? [:])
            result["parts"] = (try? partItems(posed, artifact: artifact)) ?? []
            if stats.contains(where: { ($0.cavities ?? 0) > 0 }) {
                result["cavitiesNote"] = "cavities counts sealed voids inside a part, which are not separate pieces; printed, they trap powder, resin or support material."
            }
            if stats.count > Self.summaryParts {
                result["partsTruncated"] = "The first \(Self.summaryParts) of \(stats.count) parts; page the rest with rad_inspect kind=parts."
            }
        }
        if let motion = preview.assemblyMotion, let definition = try? json(motion) {
            result["assemblyMotion"] = definition
            result["jointPositions"] = preview.assemblyPose?.positions ?? Dictionary(uniqueKeysWithValues: motion.joints.map { ($0.id, $0.position) })
        }
        if let runtime = preview.runtime, let info = try? json(runtime) { result["runtime"] = info }
        if let versionID = artifact.versionID { result["versionID"] = versionID.uuidString }
        if artifact.print { result["print"] = true }
        else if preview.printVersion == true { result["definesPrintVersion"] = true }
        if let pins = preview.printDown, !pins.isEmpty {
            result["printDown"] = pins.map { ["solidIndex": $0.solid, "direction": $0.direction] as [String: Any] }
        }
        if let references = preview.referenceSolids, !references.isEmpty {
            result["referenceSolids"] = references
        }
        if let fields = preview.customFields, !fields.isEmpty {
            let owners = Self.owners(in: artifact.parts)
            result["customFields"] = fields.map { field -> [String: Any] in
                var item: [String: Any] = ["solidIndices": field.solids, "bounds": field.bounds,
                                           "slopeBound": field.slopeBound, "distanceQuality": field.quality]
                let names = field.solids.compactMap { owners[$0]?.name }
                if !names.isEmpty { item["parts"] = names }
                if let line = field.line { item["line"] = line }
                return item
            }
            result["customFieldsNote"] = "Each implicit() solid divides its field by slopeBound. distanceQuality below 1 makes its offsets, shells, blends and clearances larger than requested (by up to about 1/distanceQuality) and its probed distances smaller; rad_verify reports value ranges, gradient statistics and failures."
        }
        return result
    }

    private func constructionMatches(_ first: ModelSpec, _ second: ModelSpec) -> Bool {
        guard first.units == second.units, first.script == second.script else { return false }
        // Programs have the full Python API and may iterate imported_assets.
        // Its insertion order is therefore part of the construction contract.
        let a = first.inputAssets ?? []
        let b = second.inputAssets ?? []
        return a.count == b.count && zip(a, b).allSatisfy { $0.id == $1.id && $0.format == $1.format && $0.data == $1.data }
    }

    private func imageResult(_ url: URL, metadata: [String: Any]) throws -> CADAgentToolResult {
        // Verify the renderer wrote a regular workspace file before the MCP
        // adapter reads it. The audited call saves these exact bytes with its
        // activity event.
        let bytes = try workspace.read(url.path, maximumBytes: 32 * 1024 * 1024)
        var info = metadata
        info["path"] = url.path
        info["byteCount"] = bytes.count
        info["sha256"] = SHA256.hexDigest(bytes)
        return try CADAgentToolResult(metadata: info, image: .init(path: url.path, data: bytes))
    }

    private func result(_ object: [String: Any]) throws -> CADAgentToolResult {
        try CADAgentToolResult(metadata: object)
    }
    private func json<T: Encodable>(_ value: T) throws -> [String: Any] {
        try JSONSerialization.jsonObject(with: JSONEncoder().encode(value)) as! [String: Any]
    }
    private func checkCancellation() throws {
        if cancelled { throw CancellationError() }
        try Task.checkCancellation()
    }
    private func allowed(_ values: [String: Any], _ keys: Set<String>) throws {
        guard Set(values.keys).isSubset(of: keys) else { throw RADError.message("The RAD tool request contains unsupported arguments.") }
    }
    private func string(_ key: String, _ values: [String: Any], required: Bool = false) throws -> String? {
        guard let value = values[key] else {
            if required { throw RADError.message("The RAD tool requires ‘\(key)’.") }
            return nil
        }
        guard let value = value as? String, !value.isEmpty, value.utf8.count <= 8_192,
              value.unicodeScalars.allSatisfy({ !CharacterSet.controlCharacters.contains($0) }) else {
            throw RADError.message("The RAD tool argument ‘\(key)’ must be nonempty text.")
        }
        return value
    }
    private func bool(_ key: String, _ values: [String: Any]) throws -> Bool? {
        guard let value = values[key] else { return nil }
        guard let number = value as? NSNumber, CFGetTypeID(number) == CFBooleanGetTypeID() else {
            throw RADError.message("The RAD tool argument ‘\(key)’ must be true or false.")
        }
        return number.boolValue
    }
    private func strings(_ key: String, _ values: [String: Any]) throws -> [String]? {
        guard let value = values[key] else { return nil }
        guard let list = value as? [String], list.count <= 256,
              list.allSatisfy({ !$0.isEmpty && $0.utf8.count <= 1_024 }) else {
            throw RADError.message("The RAD tool argument ‘\(key)’ must be an array of identities.")
        }
        return list
    }
    private func boundedText(_ key: String, _ values: [String: Any], fallback: String, maximum: Int) throws -> String {
        guard let value = (values[key] ?? fallback) as? String,
              !value.trimmingCharacters(in: .whitespacesAndNewlines).isEmpty,
              value.unicodeScalars.count <= maximum, !value.contains("\0") else {
            throw RADError.message("‘\(key)’ must be nonempty text of at most \(maximum) characters.")
        }
        return value
    }
    private func permissionText(_ key: String, _ values: [String: Any], maximum: Int) throws -> String {
        let value = try string(key, values, required: true)!.trimmingCharacters(in: .whitespacesAndNewlines)
        guard !value.isEmpty, value.unicodeScalars.count <= maximum else {
            throw RADError.message("‘\(key)’ must be nonempty text of at most \(maximum) characters.")
        }
        return value
    }
    private func integer(_ key: String, _ values: [String: Any], fallback: Int, range: ClosedRange<Int>) throws -> Int {
        guard let raw = values[key] else { return fallback }
        guard let number = raw as? NSNumber, CFGetTypeID(number) != CFBooleanGetTypeID(),
              number.doubleValue == Double(number.intValue), range.contains(number.intValue) else {
            throw RADError.message("The RAD tool argument ‘\(key)’ is outside its supported range.")
        }
        return number.intValue
    }
    private func finite(_ key: String, _ values: [String: Any]) throws -> Double {
        guard let number = values[key] as? NSNumber, CFGetTypeID(number) != CFBooleanGetTypeID(),
              number.doubleValue.isFinite, abs(number.doubleValue) <= 1e9 else {
            throw RADError.message("The RAD tool requires ‘\(key)’ as a finite number.")
        }
        return number.doubleValue
    }

    private func renderCamera(_ values: [String: Any]) throws -> CADAgentCamera {
        func number(_ raw: Any, key: String) throws -> Double {
            guard let value = raw as? NSNumber, CFGetTypeID(value) != CFBooleanGetTypeID(), value.doubleValue.isFinite else {
                throw RADError.message("The camera argument ‘\(key)’ must be a finite number.")
            }
            return value.doubleValue
        }
        func scalar(_ key: String) throws -> Double? {
            try values[key].map { try number($0, key: key) }
        }
        func point(_ key: String) throws -> [Double]? {
            guard let raw = values[key] else { return nil }
            guard let list = raw as? [Any], list.count == 3 else {
                throw RADError.message("The camera argument ‘\(key)’ must contain three millimetre coordinates.")
            }
            return try list.map { try number($0, key: key) }
        }
        var camera = CADAgentCamera()
        camera.yawDegrees = try scalar("yawDegrees")
        camera.elevationDegrees = try scalar("elevationDegrees")
        camera.rollDegrees = try scalar("rollDegrees") ?? 0
        camera.position = try point("cameraPosition")
        camera.target = try point("cameraTarget")
        camera.zoom = try scalar("zoom") ?? 1
        try camera.validate()
        return camera
    }
    private func renderJointPositions(_ values: [String: Any]) throws -> [String: Double]? {
        guard let raw = values["jointPositions"] else { return nil }
        guard let positions = raw as? [String: Any], positions.count <= 256 else {
            throw RADError.message("Assembly jointPositions must map at most 256 joint IDs to finite positions.")
        }
        var result = [String: Double]()
        for (id, raw) in positions {
            guard !id.isEmpty, id.utf8.count <= 1024,
                  !id.unicodeScalars.contains(where: CharacterSet.controlCharacters.contains),
                  let value = raw as? NSNumber, CFGetTypeID(value) != CFBooleanGetTypeID(),
                  value.doubleValue.isFinite, abs(value.doubleValue) <= 1e9 else {
                throw RADError.message("Assembly jointPositions must map joint IDs to finite numeric positions.")
            }
            result[id] = value.doubleValue
        }
        return result
    }
    private func indices(_ values: [String: Any], preview: PreviewMesh, optional: Bool, allowEmpty: Bool = false) throws -> Set<Int>? {
        guard let raw = values["solidIndices"] else { return optional ? nil : [] }
        guard let numbers = raw as? [NSNumber], (allowEmpty || !numbers.isEmpty), numbers.count <= preview.solids else {
            throw RADError.message("Choose at least one available solid index.")
        }
        var indices = Set<Int>()
        for number in numbers {
            guard CFGetTypeID(number) != CFBooleanGetTypeID(), number.doubleValue == Double(number.intValue),
                  (0..<preview.solids).contains(number.intValue), indices.insert(number.intValue).inserted else {
                throw RADError.message("The selected parts contain invalid or duplicate solid indices.")
            }
        }
        return indices
    }
}

/// Descriptor-relative I/O keeps the native host inside this task's workspace,
/// even if an agent replaces a parent directory while a tool is reading/writing.
struct CADAgentWorkspace: Sendable {
    let root: URL
    private let suppliedRoot: URL
    init(directory: URL) throws {
        suppliedRoot = directory.standardizedFileURL
        root = try EngineArtifact.canonicalDirectory(directory)
    }

    func url(_ path: String) throws -> URL {
        guard !path.isEmpty, path.unicodeScalars.allSatisfy({ !CharacterSet.controlCharacters.contains($0) }) else {
            throw RADError.message("Use a valid file path inside the task workspace.")
        }
        var url = (path.hasPrefix("/") ? URL(fileURLWithPath: path) : root.appendingPathComponent(path)).standardizedFileURL
        if url.path.hasPrefix(suppliedRoot.path + "/") {
            url = root.appendingPathComponent(String(url.path.dropFirst(suppliedRoot.path.count + 1)))
        }
        guard url.path.hasPrefix(root.path + "/"), url.path != root.path,
              !url.lastPathComponent.isEmpty else { throw RADError.message("Use a file path inside the task workspace.") }
        return url
    }

    func read(_ path: String, maximumBytes: Int, allowEmpty: Bool = false) throws -> Data {
        let url = try url(path)
        let parent = try parentDescriptor(url, create: false)
        defer { Darwin.close(parent) }
        let descriptor = Darwin.openat(parent, url.lastPathComponent, O_RDONLY | O_NOFOLLOW | O_NONBLOCK | O_CLOEXEC)
        guard descriptor >= 0 else { throw RADError.message("The workspace file is missing or is a symbolic link.") }
        let handle = FileHandle(fileDescriptor: descriptor, closeOnDealloc: true)
        defer { try? handle.close() }
        var info = stat()
        guard Darwin.fstat(descriptor, &info) == 0, info.st_mode & mode_t(S_IFMT) == mode_t(S_IFREG),
              info.st_nlink == 1, info.st_size >= (allowEmpty ? 0 : 1), info.st_size <= maximumBytes else {
            throw RADError.message("Use a bounded regular workspace file without symbolic or hard links.")
        }
        var data = Data()
        while let chunk = try handle.read(upToCount: min(1_048_576, maximumBytes + 1 - data.count)), !chunk.isEmpty {
            data.append(chunk)
            guard data.count <= maximumBytes else { throw RADError.message("The workspace file exceeds its size limit.") }
        }
        return data
    }

    /// Inspect bounded output metadata without reading a potentially large
    /// external reference or 3MF export into the UI process a second time.
    func fileSize(_ path: String, maximumBytes: Int, allowEmpty: Bool = false) throws -> Int {
        let url = try url(path)
        let parent = try parentDescriptor(url, create: false)
        defer { Darwin.close(parent) }
        let descriptor = Darwin.openat(parent, url.lastPathComponent, O_RDONLY | O_NOFOLLOW | O_NONBLOCK | O_CLOEXEC)
        guard descriptor >= 0 else { throw RADError.message("The workspace file is missing or is a symbolic link.") }
        defer { Darwin.close(descriptor) }
        var info = stat()
        guard Darwin.fstat(descriptor, &info) == 0, info.st_mode & mode_t(S_IFMT) == mode_t(S_IFREG),
              info.st_nlink == 1, info.st_size >= (allowEmpty ? 0 : 1), info.st_size <= maximumBytes else {
            throw RADError.message("Use a bounded regular workspace file without symbolic or hard links.")
        }
        return Int(info.st_size)
    }

    func removeFile(_ path: String) throws {
        let url = try url(path)
        let parent = try parentDescriptor(url, create: false)
        defer { Darwin.close(parent) }
        guard Darwin.unlinkat(parent, url.lastPathComponent, 0) == 0 || errno == ENOENT else {
            throw RADError.message("Could not remove the temporary task file.")
        }
    }

    @discardableResult
    func write(_ data: Data, path: String) throws -> URL {
        let url = try url(path)
        let parent = try parentDescriptor(url, create: true)
        defer { Darwin.close(parent) }
        let temporary = ".rad-\(UUID().uuidString)"
        let descriptor = Darwin.openat(parent, temporary, O_WRONLY | O_CREAT | O_EXCL | O_NOFOLLOW | O_CLOEXEC, mode_t(0o600))
        guard descriptor >= 0 else { throw RADError.message("Could not create a private workspace output.") }
        defer { Darwin.unlinkat(parent, temporary, 0) }
        let handle = FileHandle(fileDescriptor: descriptor, closeOnDealloc: true)
        try handle.write(contentsOf: data)
        try handle.close()
        guard Darwin.renameat(parent, temporary, parent, url.lastPathComponent) == 0 else {
            throw RADError.message("Could not finish writing the workspace output.")
        }
        return url
    }

    private func parentDescriptor(_ url: URL, create: Bool) throws -> Int32 {
        let relative = String(url.path.dropFirst(root.path.count + 1))
        let components = relative.split(separator: "/").dropLast().map(String.init)
        var descriptor = Darwin.open(root.path, O_RDONLY | O_DIRECTORY | O_NOFOLLOW | O_CLOEXEC)
        guard descriptor >= 0 else { throw RADError.message("The task workspace is unavailable.") }
        for component in components {
            if create { _ = Darwin.mkdirat(descriptor, component, mode_t(0o700)) }
            let next = Darwin.openat(descriptor, component, O_RDONLY | O_DIRECTORY | O_NOFOLLOW | O_CLOEXEC)
            Darwin.close(descriptor)
            guard next >= 0 else { throw RADError.message("Workspace output directories must not contain symbolic links.") }
            descriptor = next
        }
        return descriptor
    }
}
