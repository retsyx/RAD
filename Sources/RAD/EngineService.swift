import Foundation
import os
import Darwin

enum EngineExportError: LocalizedError {
    case nativeArtifactUnavailable

    var errorDescription: String? {
        "The native geometry is unavailable. Rebuild the design and try again."
    }
}

/// A sandboxed CAD program run that failed. `kind` records why, so callers
/// act on the failure itself rather than on the wording of its message.
struct EngineBuildFailure: LocalizedError {
    enum Kind: Sendable, Equatable {
        /// The worker ran the program and reported that the design failed: a
        /// program error, invalid geometry or failed validation.
        case design
        /// The CAD runtime, its native libraries or its file access failed.
        case runtime
        /// The macOS sandbox could not start the program.
        case sandbox
        /// The program exceeded the engine's time limit.
        case timeout
        /// The engine stopped without reporting why.
        case engine
    }

    let kind: Kind
    /// The user-facing description.
    let message: String
    /// The worker's own report of the failure, when it made one.
    let report: String?
    /// The build stage the worker reported for the failure (error.json
    /// `stage`), such as "input_import" or "preview_meshing".
    let stage: String?

    var errorDescription: String? { message }

    /// Only a design the worker ran and rejected can be fixed by changing its program.
    var isRepairable: Bool { kind == .design }

    init(kind: Kind, message: String, report: String? = nil, stage: String? = nil) {
        self.kind = kind; self.message = message; self.report = report; self.stage = stage
    }

    /// Classify a worker that exited unsuccessfully while performing `action`.
    init(action: String, sandboxed: Bool, diagnostic: EngineFailureDiagnostic.Diagnostic) {
        let text = diagnostic.text
        if text.contains("No module named") {
            self.init(kind: .runtime, message: "RAD’s CAD runtime is incomplete. Reinstall RAD and try again.\n\n\(text.suffix(1_200))",
                      report: diagnostic.report)
        } else if sandboxed, text.contains("sandbox_apply") || text.contains("sandbox-exec:") {
            self.init(kind: .sandbox, message: "The macOS sandbox could not start this design. Restart RAD outside an enclosing sandbox and try again.\n\n\(text.suffix(1_200))",
                      report: diagnostic.report)
        } else {
            // The worker states whether the design or its runtime failed.
            let kind: Kind = switch diagnostic.kind {
            case "design": .design
            case "runtime": .runtime
            default: .engine
            }
            self.init(kind: kind, message: "The CAD engine could not \(action).\n\n\(text)", report: diagnostic.report,
                      stage: diagnostic.stage)
        }
    }
}

/// Worker failures remain bounded data from the private build directory.
/// A separate report keeps native validation details clear of stderr noise.
enum EngineFailureDiagnostic {
    struct Diagnostic: Sendable {
        /// The bounded failure text shown to the user.
        var text: String
        /// The worker's own report, when it made one.
        var report: String?
        /// The worker's classification of a reported failure: "design" or "runtime".
        var kind: String?
        /// The build stage of a reported failure.
        var stage: String? = nil
    }

    private struct Report: Decodable {
        var formatVersion: Int
        var message: String
        var failureKind: String?
        var stage: String?
    }

    private static let marker = "RAD CAD engine:"

    static func read(in folder: URL, stderr: String) -> Diagnostic {
        if let data = try? EngineArtifact.read(folder.appendingPathComponent("error.json"), in: folder,
                                               maximumBytes: 65_536),
           let report = try? JSONDecoder().decode(Report.self, from: data),
           report.formatVersion == 1, report.message.utf8.count <= 48_000,
           !report.message.trimmingCharacters(in: .whitespacesAndNewlines).isEmpty {
            return Diagnostic(text: "\(marker) \(report.message)",
                              report: report.message.trimmingCharacters(in: .whitespacesAndNewlines), kind: report.failureKind,
                              stage: report.stage)
        }
        // Without a report file, the worker's last words on stderr are its report.
        let text = String(stderr.suffix(3_000))
        let report = text.range(of: marker, options: .backwards).map {
            String(text[$0.upperBound...]).trimmingCharacters(in: .whitespacesAndNewlines)
        }
        return Diagnostic(text: text, report: report)
    }

    static func message(in folder: URL, stderr: String) -> String { read(in: folder, stderr: stderr).text }

    /// The failure as the agent and conversation see it: the worker's own
    /// report when it made one.
    static func repairText(for failure: Error) -> String {
        if let failure = failure as? EngineBuildFailure { return String((failure.report ?? failure.message).prefix(16_000)) }
        let description = failure.localizedDescription
        let diagnostic: String
        if let found = description.range(of: marker, options: .backwards) {
            diagnostic = String(description[found.upperBound...]).trimmingCharacters(in: .whitespacesAndNewlines)
        } else {
            diagnostic = description
        }
        return String(diagnostic.prefix(16_000))
    }
}

/// A separate process keeps the UI responsive and contains native-kernel failures.
/// Generated CAD programs run under a restricted macOS sandbox.
@MainActor
final class EngineService {
    nonisolated static let log = Logger(subsystem: "studio.rad.cad", category: "EngineService")
    private var job: EngineProcessJob?
    private var generation = UUID()
    private let temporaryDirectory: URL
    /// A cached 3MF export: the staged graph and the whole export request,
    /// so every request field selects its own bytes.
    private struct ThreeMFExportKey: Hashable {
        var artifactPath: String
        var request: EngineWorkerRequest.Export
    }

    /// Measurements, clearance checks, exports and runtime inspection share
    /// one long-lived worker; builds keep their own sandboxed process.
    let native: CADNativeWorker

    /// A sandboxed build worker started before this engine's next build, with
    /// the CAD runtime already loaded. It serves exactly one build.
    private struct PreparedBuildWorker {
        let python: String
        let artifacts: EngineBuildArtifacts
        let start: URL
        let job: EngineProcessJob
    }
    private var preparedWorker: PreparedBuildWorker?
    private var isPreparingWorker = false
    private var preparedWorkerExpiry: Task<Void, Never>?
    /// Builds served by a prepared worker, for regression tests.
    private(set) var preparedBuilds = 0
    var hasPreparedBuildWorker: Bool { preparedWorker != nil }

    init(temporaryDirectory: URL = FileManager.default.temporaryDirectory) {
        self.temporaryDirectory = temporaryDirectory
        native = CADNativeWorker(temporaryDirectory: temporaryDirectory)
    }

    deinit {
        job?.cancel()
        preparedWorker?.job.discard()
    }

    static var defaultPython: String {
        CADRuntimeManager.defaultPythonPath
    }

    /// Build the program's SDF graph and preview meshes. The worker reports
    /// the receipt of every build.
    /// `print` builds the program's 3D printing version (`rad.print`).
    func buildPreview(_ spec: ModelSpec, print: Bool = false, python: String, threads: Int) async throws -> PreviewMesh {
        let output = try await run(python: python, threads: threads, scriptMode: true) { folder in
            try Self.stageBuild(spec, print: print, verifyFields: false, in: folder)
        }
        return try await readPreview(output)
    }

    /// Writes a program request and its imported inputs into a build folder.
    nonisolated private static func stageBuild(_ spec: ModelSpec, print: Bool, verifyFields: Bool, in folder: URL) throws {
        var staged: [EngineWorkerRequest.InputAsset] = []
        if let assets = spec.inputAssets, !assets.isEmpty {
            guard assets.count <= 256, Set(assets.map(\.id)).count == assets.count else {
                throw RADError.message("The design contains too many or duplicate imported geometry inputs.")
            }
            for (index, asset) in assets.enumerated() {
                guard CADInputAsset.supportedFormats.contains(asset.format), !asset.data.isEmpty,
                      asset.data.count <= ProjectStore.maximumAssetBytes else {
                    throw RADError.message("An imported geometry input is invalid or exceeds 512 MB.")
                }
                let name = String(format: "input-%03d", index) + ".\(asset.format)"
                try asset.data.write(to: folder.appendingPathComponent(name), options: .atomic)
                staged.append(.init(id: asset.id, name: asset.name, format: asset.format, path: name))
            }
        }
        try EngineWorkerRequest.build(title: spec.title, summary: spec.summary, units: spec.units, script: spec.script,
                                      inputAssets: staged, print: print, verifyFields: verifyFields).write(to: folder)
    }

    /// The custom fields (implicit() shapes) of a program as its build would
    /// prove them, without meshing: one record per shape, at most 64.
    struct FieldVerification: Decodable, Sendable, Equatable {
        /// rad_verify analyses at most this many implicit() shapes (the
        /// worker's `MAX_VERIFIED_FIELDS`).
        static let maximumFields = 64
        /// The verification result carries the design's part tree (up to
        /// 8,192 nodes with 1,024-byte ids and 160-character names, escaped).
        static let maximumResultBytes = 64_000_000
        struct Field: Decodable, Sendable, Equatable {
            struct Gradient: Decodable, Sendable, Equatable {
                var min: Double
                var median: Double
                var max: Double
            }
            var solids: [Int]
            var line: Int?
            var bounds: [Double]
            var ok: Bool
            /// The proven upper bound of |grad f| over the bounds.
            var slopeBound: Double?
            var valueRange: [Double]?
            var gradient: Gradient?
            /// Median sampled |grad f| / slopeBound.
            var quality: Double?
            var boxes: Int?
            /// Without a finite slope bound: what fails, and where.
            var message: String?
            var region: [Double]?
        }
        var solids: Int
        var parts: [PartNode]?
        var fields: [Field]
        /// Every implicit() shape of the design, including any beyond `fields`.
        var fieldCount: Int
    }

    /// Run a program's custom field verification (rad_verify) in the build sandbox.
    func verifyFields(_ spec: ModelSpec, print: Bool = false, python: String, threads: Int) async throws -> FieldVerification {
        let output = try await run(python: python, threads: threads, scriptMode: true,
                                   action: "verify this design's custom fields") { folder in
            try Self.stageBuild(spec, print: print, verifyFields: true, in: folder)
        }
        let folder = output.folder
        let data = try await Task.detached(priority: .userInitiated) {
            try EngineArtifact.read(folder.appendingPathComponent("result.json"), in: folder,
                                    maximumBytes: FieldVerification.maximumResultBytes)
        }.value
        guard output.token == generation, !Task.isCancelled else { throw CancellationError() }
        let result = try JSONDecoder().decode(FieldVerification.self, from: data)
        guard (1...4_096).contains(result.solids), (0...4_000_000).contains(result.fieldCount),
              result.fields.count <= min(FieldVerification.maximumFields, result.fieldCount),
              result.fields.allSatisfy({ field in
                  !field.solids.isEmpty && field.solids.allSatisfy { (0..<result.solids).contains($0) } &&
                  field.bounds.count == 6 && field.bounds.allSatisfy(\.isFinite) &&
                  (field.ok ? field.slopeBound.map { $0.isFinite && $0 >= 0 } == true : field.message?.isEmpty == false)
              }) else {
            throw RADError.message("The CAD engine returned an invalid custom field report.")
        }
        return result
    }

    func runtimeInfo(python: String, threads: Int) async throws -> CADRuntime {
        let output = try await runNative(python: python, threads: threads, action: "inspect the CAD runtime") { folder in
            try EngineWorkerRequest.runtime.write(to: folder)
        }
        let data = try await Task.detached(priority: .userInitiated) {
            try EngineArtifact.read(output.folder.appendingPathComponent("result.json"), in: output.folder, maximumBytes: 65_536)
        }.value
        struct RuntimeResult: Decodable { var runtime: CADRuntime }
        return try JSONDecoder().decode(RuntimeResult.self, from: data).runtime
    }

    /// Export separately colored meshes from the build's SDF graph without
    /// replaying the program. Cache by geometry, subset, placement and presentation.
    func export3MF(_ preview: PreviewMesh, solidIndices: Set<Int>? = nil,
                   partPresentation: [Int: ThreeMFExport.Presentation]? = nil,
                   solidTransforms: [Int: [Double]]? = nil, python: String, threads: Int) async throws -> Data {
        defer { withExtendedLifetime(preview.buildArtifacts) {} }
        try Task.checkCancellation()
        let transforms = try AssemblySolidTransform.validatedRecords(solidTransforms ?? preview.assemblyPose?.solidTransforms ?? [:], solidCount: preview.solids)
        if let solidIndices {
            guard !solidIndices.isEmpty, solidIndices.allSatisfy({ $0 >= 0 && $0 < preview.solids }) else {
                throw RADError.message("Select at least one available part to export.")
            }
        }
        let selection: [Int]? = solidIndices.flatMap { $0.count == preview.solids ? nil : $0.sorted() }
        let source = try nativeSource(preview)
        let presentation = partPresentation ?? ThreeMFExport.presentation(in: preview.parts ?? [])
        let request = EngineWorkerRequest.Export(solidIndices: selection,
            partPresentation: presentation.filter { solidIndices?.contains($0.key) ?? true }, solidTransforms: transforms)
        let cacheKey = ThreeMFExportKey(artifactPath: source.url.path, request: request)
        if let cached = ThreeMFExportCache.shared.data(for: cacheKey) { return cached[0] }
        let output = try await runNative(python: python, threads: threads, action: "export this design") { folder in
            try EngineWorkerRequest.export(Self.stageNative(source, into: folder), request).write(to: folder)
        }
        let data = try await Task.detached(priority: .userInitiated) {
            let data = try EngineArtifact.read(output.folder.appendingPathComponent("model.3mf"), in: output.folder,
                                               maximumBytes: ProjectStore.maximumAssetBytes)
            try Self.validate3MF(data)
            return data
        }.value
        try Task.checkCancellation()
        ThreeMFExportCache.shared.store([data], for: cacheKey, owner: source.artifacts.folder)
        return data
    }

    /// Export all requested named parts in one worker, preserving their order.
    /// Placement is shared across files, so importing them together
    /// reconstructs the same layout as a combined export.
    func export3MFParts(_ preview: PreviewMesh, parts: [[Int]],
                        partPresentation: [Int: ThreeMFExport.Presentation]? = nil,
                        solidTransforms: [Int: [Double]]? = nil, python: String, threads: Int) async throws -> [Data] {
        defer { withExtendedLifetime(preview.buildArtifacts) {} }
        try Task.checkCancellation()
        let transforms = try AssemblySolidTransform.validatedRecords(solidTransforms ?? preview.assemblyPose?.solidTransforms ?? [:], solidCount: preview.solids)
        let indices = parts.flatMap { $0 }
        guard !parts.isEmpty, parts.count <= preview.solids, parts.allSatisfy({ !$0.isEmpty }),
              indices.count <= preview.solids, Set(indices).count == indices.count,
              indices.allSatisfy({ (0..<preview.solids).contains($0) }) else {
            throw RADError.message("Select distinct available parts to export individually.")
        }
        let parts = parts.map { $0.sorted() }
        let selection = indices.count == preview.solids ? nil : indices.sorted()
        let source = try nativeSource(preview)
        let presentation = partPresentation ?? ThreeMFExport.presentation(in: preview.parts ?? [])
        let included = Set(indices)
        let request = EngineWorkerRequest.Export(solidIndices: selection,
            partPresentation: presentation.filter { included.contains($0.key) },
            solidTransforms: transforms, individualParts: parts)
        let cacheKey = ThreeMFExportKey(artifactPath: source.url.path, request: request)
        if let cached = ThreeMFExportCache.shared.data(for: cacheKey) { return cached }
        let output = try await runNative(python: python, threads: threads, action: "export these design parts") { folder in
            try EngineWorkerRequest.export(Self.stageNative(source, into: folder), request).write(to: folder)
        }
        struct ExportedPart: Decodable { var path: String; var solidIndices: [Int] }
        struct ExportedParts: Decodable { var partFiles: [ExportedPart] }
        let data = try await Task.detached(priority: .userInitiated) {
            let metadata = try EngineArtifact.read(output.folder.appendingPathComponent("result.json"), in: output.folder,
                                                   maximumBytes: 1_000_000)
            let files = try JSONDecoder().decode(ExportedParts.self, from: metadata).partFiles
            guard files.count == parts.count else { throw RADError.message("The CAD engine returned an incomplete parts export.") }
            var result: [Data] = [], total = 0
            for (index, file) in files.enumerated() {
                try Task.checkCancellation()
                guard file.path == String(format: "part-%04d.3mf", index), file.solidIndices == parts[index] else {
                    throw RADError.message("The CAD engine returned an invalid parts export.")
                }
                let bytes = try EngineArtifact.read(output.folder.appendingPathComponent(file.path), in: output.folder,
                                                    maximumBytes: ProjectStore.maximumAssetBytes - total)
                try Self.validate3MF(bytes)
                total += bytes.count
                result.append(bytes)
            }
            return result
        }.value
        try Task.checkCancellation()
        ThreeMFExportCache.shared.store(data, for: cacheKey, owner: source.artifacts.folder)
        return data
    }

    /// Printing orientations for the requested solids of an already built
    /// native solid, honoring the program's pinned directions. No replay or
    /// meshing; placement on the bed is `PrintLayout`'s. Reference parts are
    /// never printed, so they cannot be requested.
    func printOrientations(_ preview: PreviewMesh, solids: Set<Int>, python: String,
                           threads: Int) async throws -> [Int: PrintOrientation] {
        defer { withExtendedLifetime(preview.buildArtifacts) {} }
        try Task.checkCancellation()
        guard !solids.isEmpty else { return [:] }
        guard solids.allSatisfy({ (0..<preview.solids).contains($0) }) else {
            throw RADError.message("Choose available parts to orient for 3D printing.")
        }
        guard PrintLayout.printedSolids(solids, of: preview) == solids else {
            throw RADError.message("Reference parts are not printed, so they have no printing orientation.")
        }
        try PrintLayout.validatePins(preview)
        let requested = solids.sorted()
        let selection: [Int]? = requested.count == preview.solids ? nil : requested
        let pins = preview.printDown ?? []
        let source = try nativeSource(preview)
        let output = try await runNative(python: python, threads: threads, action: "orient this design for 3D printing") { folder in
            try EngineWorkerRequest.printOrientations(Self.stageNative(source, into: folder), solidIndices: selection,
                                                      printDown: pins).write(to: folder)
        }
        struct Report: Decodable { var orientations: [PrintOrientation] }
        let solidCount = preview.solids
        let orientations = try await Task.detached(priority: .userInitiated) {
            let data = try EngineArtifact.read(output.folder.appendingPathComponent("result.json"), in: output.folder, maximumBytes: 8_000_000)
            return try PrintLayout.validatedOrientations(JSONDecoder().decode(Report.self, from: data).orientations,
                                                         requested: requested, solidCount: solidCount)
        }.value
        try Task.checkCancellation()
        return orientations
    }

    /// Point and part measurements are computed locally from the selection
    /// and the build's statistics, posed with the current placements. Two
    /// parts, or a point and a part, add the worker's minimum distance and
    /// closest points; the worker's surface snap then replaces the point's
    /// display-mesh coordinates and normal.
    func measure(_ preview: PreviewMesh, points: [CADSurfacePoint] = [], solids: Set<Int>,
                 solidTransforms: [Int: [Double]]? = nil, python: String, threads: Int) async throws -> CADMeasurementReport {
        let transforms = solidTransforms ?? preview.assemblyPose?.solidTransforms ?? [:]
        var report = try CADMeasurements.report(preview, points: points, solids: solids, solidTransforms: transforms)
        guard let entities = CADMeasurements.workerEntities(points: points, solids: solids) else { return report }
        let pair = try await measure(preview, entities: entities, solidTransforms: transforms, python: python, threads: threads)
        for (index, snapped) in (pair.points ?? []).enumerated() where index < points.count {
            let posed = CADMeasurements.posedPoint(snapped, transforms: transforms)
            for metric in report.metrics.indices where report.metrics[metric].id.hasPrefix(CADMeasurements.pointPrefix(index)) {
                if report.metrics[metric].id.hasSuffix(":position") { report.metrics[metric].values = posed.position }
                if report.metrics[metric].id.hasSuffix(":normal") { report.metrics[metric].values = posed.normal }
            }
        }
        report.metrics += pair.metrics
        report.closestPoints = pair.closestPoints
        return report
    }

    struct Measurement: Decodable, Sendable {
        var metrics: [CADMeasurementMetric]
        var closestPoints: [[Double]]?
        /// The point entities in request order, snapped to their part's
        /// surface in built coordinates with unit SDF normals.
        var points: [CADSurfacePoint]?
    }

    /// The worker's measurement of one or two parts or surface points, from
    /// the build's SDF at the given rigid placements.
    func measure(_ preview: PreviewMesh, entities: [EngineWorkerRequest.MeasureEntity], solidTransforms: [Int: [Double]]?,
                 python: String, threads: Int) async throws -> Measurement {
        guard (1...2).contains(entities.count), entities.allSatisfy({ (0..<preview.solids).contains($0.solid) }),
              entities.allSatisfy({ entity in
                  guard case .point(_, let point) = entity else { return true }
                  return point.count == 3 && point.allSatisfy { $0.isFinite && abs($0) <= 1_000_000 }
              }) else {
            throw RADError.message("Measure one or two available parts or surface points.")
        }
        let records = try solidTransforms.map { try AssemblySolidTransform.validatedRecords($0, solidCount: preview.solids) }
        let source = try nativeSource(preview)
        let output = try await runNative(python: python, threads: threads, action: "measure these items") { folder in
            try EngineWorkerRequest.measure(Self.stageNative(source, into: folder), entities: entities,
                                            solidTransforms: records).write(to: folder)
        }
        let requested = entities.compactMap { entity -> Int? in
            guard case .point(let solid, _) = entity else { return nil }
            return solid
        }
        let measurement = try await Task.detached(priority: .userInitiated) {
            let data = try EngineArtifact.read(output.folder.appendingPathComponent("result.json"), in: output.folder, maximumBytes: 32_000_000)
            var result = try JSONDecoder().decode(Measurement.self, from: data)
            // Gradients are unit length up to rounding; normalize before use.
            result.points = result.points?.map { point in
                var point = point
                let length = sqrt(point.normal.reduce(0) { $0 + $1 * $1 })
                if point.normal.count == 3, length.isFinite, length > 0.5, length < 2 { point.normal = point.normal.map { $0 / length } }
                return point
            }
            guard result.metrics.allSatisfy({ !$0.values.isEmpty && $0.values.allSatisfy(\.isFinite) }),
                  result.closestPoints.map({ $0.count == 2 && $0.allSatisfy({ $0.count == 3 && $0.allSatisfy(\.isFinite) }) }) ?? true,
                  (result.points ?? []).map(\.solidIndex) == requested,
                  (result.points ?? []).allSatisfy(\.isValid) else {
                throw RADError.message("The CAD engine returned an invalid measurement.")
            }
            return result
        }.value
        try Task.checkCancellation()
        return measurement
    }

    struct Clearance: Decodable, Sendable, Equatable {
        /// Minimum distance between the parts' surfaces; zero for touching or interfering parts.
        var distance: Double
        /// Common volume of interfering parts, when the worker integrated it.
        var overlapVolume: Double?
        var closestPoints: [[Double]]
    }

    /// Pairwise clearance at each supplied rigid pose, from the build's SDF.
    /// `pairs` lists two disjoint solid-index groups per pair; the result is
    /// indexed by pose and then by pair.
    func clearance(_ preview: PreviewMesh, pairs: [[[Int]]], poses: [[Int: [Double]]],
                   python: String, threads: Int) async throws -> [[Clearance]] {
        let records = try poses.map { try AssemblySolidTransform.validatedRecords($0, solidCount: preview.solids) }
        let source = try nativeSource(preview)
        let output = try await runNative(python: python, threads: threads, action: "check this motion") { folder in
            try EngineWorkerRequest.clearance(Self.stageNative(source, into: folder), pairs: pairs, poses: records).write(to: folder)
        }
        struct Report: Decodable { var samples: [[Clearance]] }
        let report = try await Task.detached(priority: .userInitiated) {
            let data = try EngineArtifact.read(output.folder.appendingPathComponent("result.json"), in: output.folder, maximumBytes: 32_000_000)
            let report = try JSONDecoder().decode(Report.self, from: data)
            guard report.samples.count == poses.count, report.samples.allSatisfy({ sample in
                sample.count == pairs.count && sample.allSatisfy { item in
                    item.distance.isFinite && item.distance >= 0 && (item.overlapVolume.map { $0.isFinite && $0 >= 0 } ?? true)
                        && item.closestPoints.count == 2 && item.closestPoints.allSatisfy { $0.count == 3 && $0.allSatisfy(\.isFinite) }
                }
            }) else { throw RADError.message("The CAD engine returned an invalid clearance report.") }
            return report
        }.value
        try Task.checkCancellation()
        return report.samples
    }

    /// Surface quality of every part re-meshed finer than its preview.
    struct Validation: Decodable, Sendable, Equatable {
        struct Part: Decodable, Sendable, Equatable {
            var solid: Int
            var closed: Bool
            var boundaryEdges: Int
            var nonmanifoldEdges: Int
            /// Separate pieces (outward surfaces); sealed voids are `cavities`.
            var components: Int
            var cavities: Int? = nil
            var previewCavities: Int? = nil
            var volume: Double
            var previewVolume: Double
            /// Relative change from the preview volume.
            var volumeChange: Double
        }
        var ok: Bool
        var cellSize: Double
        var parts: [Part]
        var warnings: [String]
    }

    /// Re-mesh every part at `cellSize` (nil: half the preview cell size) and
    /// report open, non-manifold, split or vanishing surfaces, cavities that
    /// open or close, and volume changes.
    func validate(_ preview: PreviewMesh, cellSize: Double? = nil, python: String, threads: Int) async throws -> Validation {
        if let cellSize { guard cellSize.isFinite, cellSize > 0 else { throw RADError.message("The validation cell size must be positive.") } }
        let source = try nativeSource(preview)
        let output = try await runNative(python: python, threads: threads, action: "validate this design") { folder in
            try EngineWorkerRequest.validate(Self.stageNative(source, into: folder), cellSize: cellSize).write(to: folder)
        }
        let solidCount = preview.solids
        let validation = try await Task.detached(priority: .userInitiated) {
            let data = try EngineArtifact.read(output.folder.appendingPathComponent("result.json"), in: output.folder, maximumBytes: 8_000_000)
            let result = try JSONDecoder().decode(Validation.self, from: data)
            guard result.cellSize.isFinite, result.cellSize > 0, result.parts.map(\.solid) == Array(0..<solidCount),
                  result.parts.allSatisfy({ part in
                      part.boundaryEdges >= 0 && part.nonmanifoldEdges >= 0 && part.components >= 0 && (part.cavities ?? 0) >= 0 &&
                      (part.previewCavities ?? 0) >= 0 && [part.volume, part.previewVolume, part.volumeChange].allSatisfy(\.isFinite)
                  }),
                  result.warnings.count <= 256, result.warnings.allSatisfy({ $0.utf8.count <= 4_096 }) else {
                throw RADError.message("The CAD engine returned an invalid validation report.")
            }
            return result
        }.value
        try Task.checkCancellation()
        return validation
    }

    /// The SDF of one part at one probed point.
    struct ProbeSample: Decodable, Sendable, Equatable {
        /// Signed distance, negative inside.
        var distance: Double
        var normal: [Double]
        /// Ray results, present when directions were probed.
        var hit: Bool?
        var hitDistance: Double?
        var hitPoint: [Double]?
    }

    struct Probe: Decodable, Sendable, Equatable {
        var solids: [Int]
        /// Indexed by point, then by `solids`.
        var samples: [[ProbeSample]]
    }

    /// Signed distances and unit normals of the requested parts at points of
    /// the built design; with directions, also the ray hits from each point.
    func probe(_ preview: PreviewMesh, solids: Set<Int>? = nil, points: [[Double]], directions: [[Double]]? = nil,
               maxDistance: Double? = nil, python: String, threads: Int) async throws -> Probe {
        func vector(_ value: [Double]) -> Bool { value.count == 3 && value.allSatisfy { $0.isFinite && abs($0) <= 1e9 } }
        guard (1...65_536).contains(points.count), points.allSatisfy(vector),
              directions.map({ $0.count == points.count && $0.allSatisfy { vector($0) && $0.contains { $0 != 0 } } }) ?? true,
              maxDistance.map({ $0.isFinite && $0 > 0 }) ?? true,
              solids.map({ !$0.isEmpty && $0.allSatisfy { (0..<preview.solids).contains($0) } }) ?? true else {
            throw RADError.message("Probe 1 to 65,536 finite points of available parts, with one nonzero direction per point.")
        }
        let requested = solids.map { $0.sorted() }
        let source = try nativeSource(preview)
        let output = try await runNative(python: python, threads: threads, action: "probe this design") { folder in
            try EngineWorkerRequest.probe(Self.stageNative(source, into: folder), solids: requested, points: points,
                                          directions: directions, maxDistance: maxDistance).write(to: folder)
        }
        let expectedSolids = requested ?? Array(0..<preview.solids), rays = directions != nil
        let probe = try await Task.detached(priority: .userInitiated) {
            let data = try EngineArtifact.read(output.folder.appendingPathComponent("result.json"), in: output.folder, maximumBytes: 256_000_000)
            let result = try JSONDecoder().decode(Probe.self, from: data)
            guard result.solids == expectedSolids, result.samples.count == points.count,
                  result.samples.allSatisfy({ row in
                      row.count == expectedSolids.count && row.allSatisfy { sample in
                          sample.distance.isFinite && sample.normal.count == 3 && sample.normal.allSatisfy(\.isFinite) &&
                          (sample.hit != nil) == rays && (sample.hitDistance.map { $0.isFinite && $0 >= 0 } ?? !rays) &&
                          (sample.hitPoint.map { $0.count == 3 && $0.allSatisfy(\.isFinite) } ?? (sample.hit != true))
                      }
                  }) else {
                throw RADError.message("The CAD engine returned an invalid probe report.")
            }
            return result
        }.value
        try Task.checkCancellation()
        return probe
    }

    /// A build's own RSDF graph, identified by its unique session path.
    private struct NativeSource: Sendable {
        let url: URL
        let partRoots: [Int]
        let artifacts: EngineBuildArtifacts
    }

    private func nativeSource(_ preview: PreviewMesh) throws -> NativeSource {
        guard let path = preview.artifactPath, let roots = preview.partRoots, roots.count == preview.solids,
              let artifacts = preview.buildArtifacts,
              path == artifacts.folder.appendingPathComponent("model.rsdf").path else { throw EngineExportError.nativeArtifactUnavailable }
        return NativeSource(url: URL(fileURLWithPath: path), partRoots: roots, artifacts: artifacts)
    }

    /// Stage a build's graph into a request and return the request's reference
    /// to it. An APFS clone shares the file's blocks, so staging neither reads
    /// a large graph into memory nor duplicates it on disk.
    nonisolated private static func stageNative(_ source: NativeSource, into folder: URL) throws -> EngineWorkerRequest.NativeSource {
        try withExtendedLifetime(source.artifacts) {
            guard let header = try? EngineArtifact.boundaries(source.url, in: source.url.deletingLastPathComponent(),
                                                              headerBytes: artifactMarker.count),
                  header.header == artifactMarker,
                  copyfile(source.url.path, folder.appendingPathComponent("source.rsdf").path, nil,
                           copyfile_flags_t(COPYFILE_CLONE)) == 0 else {
                throw EngineExportError.nativeArtifactUnavailable
            }
        }
        return EngineWorkerRequest.NativeSource(identity: source.url.path, partRoots: source.partRoots)
    }

    /// The RSDF graph format's leading bytes.
    nonisolated static let artifactMarker = Data("RSDF".utf8)

    /// Native requests run on a long-lived worker. They never cancel a build
    /// running on this engine, and a build never cancels them.
    private func runNative(python: String, threads: Int, action: String,
                           prepare: @escaping @Sendable (URL) throws -> Void) async throws -> RunResult {
        try Task.checkCancellation()
        let artifacts = try EngineBuildArtifacts.create(in: temporaryDirectory, token: UUID())
        let folder = artifacts.folder
        try await Task.detached(priority: .userInitiated) { try prepare(folder) }.value
        try Task.checkCancellation()
        let python = try await CADRuntimeManager.resolvePython(python)
        let reply: CADNativeWorker.Reply
        do { reply = try await native.perform(folder: folder, threads: max(1, min(256, threads)), python: python) }
        catch is CancellationError { throw CancellationError() }
        catch { throw RADError.message("The CAD engine could not \(action).\n\n\(error.localizedDescription)") }
        try Task.checkCancellation()
        guard reply.status == "complete" else {
            let diagnostic = EngineFailureDiagnostic.message(in: folder, stderr: reply.message ?? "")
            if diagnostic.contains("No module named") {
                throw RADError.message("RAD’s CAD runtime is incomplete. Reinstall RAD and try again.\n\n\(diagnostic.suffix(1_200))")
            }
            throw RADError.message("The CAD engine could not \(action).\n\n\(diagnostic)")
        }
        return RunResult(artifacts: artifacts, token: UUID())
    }

    /// Check the worker's ZIP structure without inflating its mesh XML.
    /// The worker validates the mesh and writes three entries to a seekable file.
    nonisolated static func validate3MF(_ data: Data) throws {
        let invalid = RADError.message("The exported 3MF is invalid or exceeds 512 MB.")
        let entryNames = Set(["[Content_Types].xml", "_rels/.rels", "3D/3dmodel.model"])
        guard data.count >= 22, data.count <= ProjectStore.maximumAssetBytes else { throw invalid }
        try data.withUnsafeBytes { raw in
            func u16(_ offset: Int) -> Int {
                Int(UInt16(littleEndian: raw.loadUnaligned(fromByteOffset: offset, as: UInt16.self)))
            }
            func u32(_ offset: Int) -> Int {
                Int(UInt32(littleEndian: raw.loadUnaligned(fromByteOffset: offset, as: UInt32.self)))
            }

            let end = raw.count - 22
            guard u32(end) == 0x06054B50, u32(end + 4) == 0, u16(end + 20) == 0,
                  u16(end + 8) == entryNames.count, u16(end + 10) == entryNames.count else { throw invalid }
            let central = u32(end + 16)
            guard central + u32(end + 12) == end else { throw invalid }

            var cursor = central
            var remaining = entryNames
            var previousEnd = 0
            for _ in entryNames {
                guard cursor + 46 <= end, u32(cursor) == 0x02014B50 else { throw invalid }
                let nameLength = u16(cursor + 28), compressedSize = u32(cursor + 20)
                let nameStart = cursor + 46, nameEnd = nameStart + nameLength
                let next = nameEnd + u16(cursor + 30) + u16(cursor + 32)
                guard next <= end,
                      remaining.remove(String(decoding: raw[nameStart..<nameEnd], as: UTF8.self)) != nil else { throw invalid }

                let local = u32(cursor + 42)
                guard local == previousEnd, local + 30 <= central, u32(local) == 0x04034B50,
                      u16(local + 6) == 0, u16(cursor + 8) == 0,
                      u16(local + 8) == u16(cursor + 10), [0, 8].contains(u16(local + 8)),
                      u32(local + 14) == u32(cursor + 16), u32(local + 18) == compressedSize, compressedSize > 0,
                      u32(local + 22) == u32(cursor + 24), u32(local + 22) > 0,
                      u16(local + 26) == nameLength else { throw invalid }
                let contents = local + 30 + nameLength + u16(local + 28)
                previousEnd = contents + compressedSize
                guard previousEnd <= central,
                      raw[(local + 30)..<(local + 30 + nameLength)].elementsEqual(raw[nameStart..<nameEnd]) else { throw invalid }
                cursor = next
            }
            guard cursor == end, previousEnd == central else { throw invalid }
        }
    }

    struct RunResult: Sendable {
        var artifacts: EngineBuildArtifacts
        var token: UUID
        var folder: URL { artifacts.folder }
    }

    func run(python: String, threads: Int, scriptMode: Bool,
             action: String = "build this design",
             prepare: @escaping @Sendable (URL) throws -> Void) async throws -> RunResult {
        try Task.checkCancellation()
        cancel()
        let token = UUID()
        generation = token
        // Program builds of the same runtime use the prepared worker.
        let prepared = scriptMode ? takePreparedWorker(python: python) : nil
        let artifacts: EngineBuildArtifacts
        do { artifacts = try prepared?.artifacts ?? EngineBuildArtifacts.create(in: temporaryDirectory, token: token) }
        catch { prepared?.job.discard(); throw error }
        let folder = artifacts.folder
        defer { withExtendedLifetime(artifacts) {} }
        let request = folder.appendingPathComponent("request.json")
        do {
            try await Task.detached(priority: .userInitiated) { try prepare(folder) }.value
            guard token == generation, !Task.isCancelled else { throw CancellationError() }
        } catch {
            prepared?.job.discard()
            throw error
        }
        let workerThreads = max(1, min(256, threads))
        var task: EngineProcessJob?
        if let prepared {
            let data = try JSONSerialization.data(withJSONObject: ["threads": workerThreads])
            do {
                try await Task.detached(priority: .userInitiated) { try prepared.job.begin(options: data, at: prepared.start) }.value
                task = prepared.job
                preparedBuilds += 1
            } catch {
                // Its process ended while waiting; launch this build normally.
                prepared.job.discard()
            }
        }
        if task == nil {
            let python = try await CADRuntimeManager.resolvePython(python)
            guard token == generation, !Task.isCancelled else { throw CancellationError() }
            let arguments = ["--request", request.path, "--output", folder.path, "--threads", String(workerThreads)]
            task = try Self.workerJob(python: python, sandboxed: scriptMode, arguments: arguments, artifacts: artifacts)
        }
        let running = task!
        job = running
        defer { if job === running { job = nil } }
        let status = try await withTaskCancellationHandler {
            try await running.run()
        } onCancel: {
            running.cancel()
        }
        if scriptMode { prepareBuildWorker(python: python) }
        guard token == generation, !Task.isCancelled else { throw CancellationError() }
        job = nil
        guard status == 0 else {
            let logData = try? EngineArtifact.tail(folder.appendingPathComponent("stderr.log"), in: folder, maximumBytes: 65_536)
            let log = logData.map { String(decoding: $0, as: UTF8.self) } ?? ""
            throw EngineBuildFailure(action: action, sandboxed: scriptMode,
                                     diagnostic: EngineFailureDiagnostic.read(in: folder, stderr: log))
        }
        return RunResult(artifacts: artifacts, token: token)
    }

    /// A worker process for `arguments`, in the program sandbox when `sandboxed`.
    private static func workerJob(python: String, sandboxed: Bool, arguments: [String],
                                  artifacts: EngineBuildArtifacts) throws -> EngineProcessJob {
        let folder = artifacts.folder
        let launch = try workerLaunch(python: python)
        let worker = launch.worker, runtime = launch.runtime, actualRuntime = launch.actualRuntime, interpreter = launch.interpreter
        let executable: URL
        let launchArguments: [String]
        if sandboxed {
            executable = URL(fileURLWithPath: "/usr/bin/sandbox-exec")
            let profile = worker.deletingLastPathComponent().appendingPathComponent("script.sb")
            guard FileManager.default.isExecutableFile(atPath: executable.path), FileManager.default.fileExists(atPath: profile.path) else {
                throw RADError.message("RAD could not start its CAD execution sandbox. Reinstall RAD and try again.")
            }
            let parameters = [
                ("RUNTIME", realDirectory(runtime.deletingLastPathComponent().deletingLastPathComponent()).path),
                ("PYTHON_BASE", pythonBaseDirectory(runtime: runtime, actualRuntime: actualRuntime).path),
                ("WORKER", worker.deletingLastPathComponent().path),
                ("OUTPUT", folder.path),
                ("PYTHON", interpreter.path), ("ENTRY_PYTHON", runtime.path),
                ("KERNEL", launch.kernel.path)
            ]
            launchArguments = parameters.flatMap { ["-D", "\($0.0)=\($0.1)"] } + ["-f", profile.path, interpreter.path, worker.path] + arguments
        } else {
            executable = runtime
            launchArguments = [worker.path] + arguments
        }
        let environment = try workerEnvironment(folder: folder, launch: launch)
        return try EngineProcessJob(executable: executable, arguments: launchArguments, artifacts: artifacts, environment: environment)
    }

    /// Start a sandboxed worker for this engine's next program build, so the
    /// CAD runtime is loaded before that build is requested. Each worker
    /// still runs one program; an unused one stops after five minutes.
    func prepareBuildWorker(python: String) {
        guard preparedWorker == nil, !isPreparingWorker else { return }
        isPreparingWorker = true
        Task { [weak self] in
            guard let self else { return }
            defer { isPreparingWorker = false }
            do {
                let resolved = try await CADRuntimeManager.resolvePython(python)
                let artifacts = try EngineBuildArtifacts.create(in: temporaryDirectory, token: UUID())
                let start = artifacts.folder.appendingPathComponent("start")
                guard mkfifo(start.path, 0o600) == 0 else { return }
                let job = try Self.workerJob(python: resolved, sandboxed: true,
                                             arguments: ["--await-build", start.path, "--output", artifacts.folder.path],
                                             artifacts: artifacts)
                try await Task.detached(priority: .utility) { try job.start() }.value
                preparedWorker?.job.discard()
                preparedWorker = PreparedBuildWorker(python: python, artifacts: artifacts, start: start, job: job)
                preparedWorkerExpiry?.cancel()
                preparedWorkerExpiry = Task { [weak self] in
                    try? await Task.sleep(for: .seconds(300))
                    guard !Task.isCancelled, let self else { return }
                    preparedWorker?.job.discard()
                    preparedWorker = nil
                }
            } catch {
                return
            }
        }
    }

    private func takePreparedWorker(python: String) -> PreparedBuildWorker? {
        guard let worker = preparedWorker else { return nil }
        preparedWorker = nil
        preparedWorkerExpiry?.cancel(); preparedWorkerExpiry = nil
        guard worker.python == python, worker.job.isRunning else { worker.job.discard(); return nil }
        return worker
    }

    func readPreview(_ output: RunResult) async throws -> PreviewMesh {
        let folder = output.folder, artifacts = output.artifacts
        let resultURL = folder.appendingPathComponent("result.json")
        let decoding = Task.detached(priority: .userInitiated) {
            let data = try EngineArtifact.read(resultURL, in: folder, maximumBytes: 256_000_000)
            var result = try JSONDecoder().decode(PreviewMesh.self, from: data)
            guard let buffers = try PreviewMeshBuffers.manifest(in: data) else {
                throw RADError.message("The CAD engine returned an incomplete design preview.")
            }
            try PreviewMeshBuffers.read(buffers, in: folder, into: &result)
            guard result.vertexNormals != nil else {
                throw RADError.message("The CAD engine returned an incomplete design preview.")
            }
            try ProjectStore.validatePreview(result)
            try PrintLayout.validatePins(result)
            try PrintLayout.validateReferences(result)
            guard UUID(uuidString: result.buildID) != nil, result.partRoots != nil, result.partStats != nil,
                  result.meshStatistics != nil, result.warnings != nil, result.runtime != nil,
                  result.triangleSolids != nil, result.edgeSolids != nil else {
                throw RADError.message("The CAD engine returned an incomplete design preview.")
            }
            if let motion = result.assemblyMotion {
                result.assemblyPose = try motion.solve(parts: result.parts ?? [])
            }
            // Generated code cannot redirect the app to read an unrelated file.
            guard result.artifactPath == folder.appendingPathComponent("model.rsdf").path else {
                throw RADError.message("CAD engine returned an unexpected native geometry path.")
            }
            let graph = try EngineArtifact.boundaries(folder.appendingPathComponent("model.rsdf"), in: folder,
                                                      headerBytes: EngineService.artifactMarker.count)
            guard graph.header == EngineService.artifactMarker else {
                throw RADError.message("CAD engine did not return native SDF geometry.")
            }
            // A missing or invalid field program leaves the parts to mesh
            // rendering; an invalid one is logged.
            do {
                result.fieldProgram = try FieldProgram.read(resultJSON: data, in: folder, solids: result.solids)
            } catch {
                EngineService.log.error("Ignoring the build's field program: \(error.localizedDescription, privacy: .public)")
            }
            artifacts.discardAll(except: ["model.rsdf"])
            return result
        }
        var mesh = try await withTaskCancellationHandler { try await decoding.value }
            onCancel: { decoding.cancel() }
        guard output.token == generation, !Task.isCancelled else { throw CancellationError() }
        mesh.buildArtifacts = output.artifacts
        return mesh
    }

    func cancel() {
        generation = UUID()
        job?.cancel()
        job = nil
    }

    /// The worker script, interpreter and native kernel for a resolved runtime.
    struct WorkerLaunch: Sendable {
        let worker: URL
        let runtime: URL
        let actualRuntime: URL
        let interpreter: URL
        /// The real path of libRADSDF.dylib, which the worker loads through ctypes.
        let kernel: URL
    }

    nonisolated static func workerLaunch(python: String) throws -> WorkerLaunch {
        guard let worker = ResourcePaths.realFile(ResourcePaths.root.appendingPathComponent("Engine/worker.py")) else {
            throw RADError.message("RAD’s CAD engine is missing. Reinstall RAD and try again.")
        }
        let requested = URL(fileURLWithPath: (python as NSString).expandingTildeInPath).standardizedFileURL
        // The sandbox profile matches real paths literally. Foundation's path
        // standardization drops /private, as in /private/tmp, so keep the entry
        // interpreter's name inside its real directory.
        let runtime = realDirectory(requested.deletingLastPathComponent()).appendingPathComponent(requested.lastPathComponent)
        guard FileManager.default.isExecutableFile(atPath: runtime.path) else {
            throw RADError.message("RAD’s CAD runtime is missing. Restart RAD and try again.")
        }
        let actualRuntime = ResourcePaths.realFile(runtime) ?? runtime.resolvingSymlinksInPath()
        // Homebrew/framework bin/python is a launcher that spawns Python.app.
        // Start the interpreter directly so generated code can never fork.
        let frameworkInterpreter = actualRuntime.deletingLastPathComponent().deletingLastPathComponent()
            .appendingPathComponent("Resources/Python.app/Contents/MacOS/Python")
        let interpreter = FileManager.default.isExecutableFile(atPath: frameworkInterpreter.path) ? frameworkInterpreter : actualRuntime
        guard let kernel = ResourcePaths.kernelLibrary else {
            throw RADError.message("RAD’s CAD kernel is missing. Reinstall RAD and try again.")
        }
        return WorkerLaunch(worker: worker, runtime: runtime, actualRuntime: actualRuntime, interpreter: interpreter, kernel: kernel)
    }

    /// An allowlist avoids passing ambient credentials or model-service keys
    /// to the worker. Runtime/cache locations are explicit and local.
    nonisolated static func workerEnvironment(folder: URL, launch: WorkerLaunch) throws -> [String: String] {
        let temporary = folder.appendingPathComponent("tmp")
        try FileManager.default.createDirectory(at: temporary, withIntermediateDirectories: true)
        var environment = [
            "PATH": "/usr/bin:/bin", "LANG": "en_US.UTF-8", "LC_ALL": "C",
            "HOME": FileManager.default.homeDirectoryForCurrentUser.path,
            "PYTHONUNBUFFERED": "1", "PYTHONNOUSERSITE": "1", "PYTHONDONTWRITEBYTECODE": "1",
            "XDG_CACHE_HOME": folder.appendingPathComponent(".cache").path,
            "TMPDIR": temporary.path, "RAD_MODEL_OUTPUT": folder.path, "RAD_SDF_LIBRARY": launch.kernel.path
        ]
        if launch.interpreter != launch.actualRuntime { environment["__PYVENV_LAUNCHER__"] = launch.runtime.path }
        return environment
    }

    private static func pythonBaseDirectory(runtime: URL, actualRuntime: URL) -> URL {
        let environment = runtime.deletingLastPathComponent().deletingLastPathComponent()
        if let configuration = try? String(contentsOf: environment.appendingPathComponent("pyvenv.cfg"), encoding: .utf8),
           let home = configuration.components(separatedBy: .newlines).first(where: { $0.hasPrefix("home = ") }) {
            return realDirectory(URL(fileURLWithPath: String(home.dropFirst(7))).deletingLastPathComponent())
        }
        return actualRuntime.deletingLastPathComponent().deletingLastPathComponent()
    }

    nonisolated private static func realDirectory(_ url: URL) -> URL {
        (try? EngineArtifact.canonicalDirectory(url)) ?? url.resolvingSymlinksInPath()
    }
}

/// Generated Python can write only the scratch directory, but an output could
/// still be a symlink or hardlink. Never follow either into the user's files.
enum EngineArtifact {
    static func canonicalDirectory(_ url: URL) throws -> URL {
        var buffer = [CChar](repeating: 0, count: Int(PATH_MAX))
        guard Darwin.realpath(url.path, &buffer) != nil else { throw RADError.message("The CAD build directory could not be resolved.") }
        return URL(fileURLWithPath: String(cString: buffer), isDirectory: true)
    }

    static func read(_ url: URL, in folder: URL, maximumBytes: Int) throws -> Data {
        let (file, size) = try openRegular(url, in: folder)
        defer { try? file.close() }
        guard size > 0, size <= maximumBytes else { throw RADError.message("CAD engine design asset exceeds its size limit or is empty.") }
        var data = Data()
        while let chunk = try file.read(upToCount: min(1_048_576, maximumBytes + 1 - data.count)), !chunk.isEmpty {
            data.append(chunk)
            guard data.count <= maximumBytes else { throw RADError.message("CAD engine design asset exceeds its size limit.") }
        }
        return data
    }

    static func tail(_ url: URL, in folder: URL, maximumBytes: Int) throws -> Data {
        let (file, size) = try openRegular(url, in: folder)
        defer { try? file.close() }
        try file.seek(toOffset: UInt64(max(0, size - Int64(maximumBytes))))
        return try file.read(upToCount: maximumBytes) ?? Data()
    }

    /// Inspect format markers without retaining the native geometry payload.
    /// Both ranges come from the same safely opened descriptor, including when
    /// a short file's header and footer overlap.
    static func boundaries(_ url: URL, in folder: URL, headerBytes: Int, footerBytes: Int = 0) throws
        -> (size: Int, header: Data, footer: Data) {
        let (file, size) = try openRegular(url, in: folder)
        defer { try? file.close() }
        guard size > 0, headerBytes >= 0, footerBytes >= 0 else {
            throw RADError.message("CAD engine design asset is empty or has invalid marker ranges.")
        }
        let headerCount = min(headerBytes, Int(size)), footerCount = min(footerBytes, Int(size))
        let header = try file.read(upToCount: headerCount) ?? Data()
        try file.seek(toOffset: UInt64(size - Int64(footerCount)))
        let footer = footerCount == 0 ? Data() : try file.read(upToCount: footerCount) ?? Data()
        var finalInfo = stat()
        guard header.count == headerCount, footer.count == footerCount,
              Darwin.fstat(file.fileDescriptor, &finalInfo) == 0, finalInfo.st_size == size else {
            throw RADError.message("The CAD engine design asset changed while it was being checked.")
        }
        return (Int(size), header, footer)
    }

    static func openRegular(_ url: URL, in folder: URL) throws -> (FileHandle, Int64) {
        guard url.deletingLastPathComponent().path == folder.path,
              try canonicalDirectory(url.deletingLastPathComponent()).path == folder.path else {
            throw RADError.message("CAD engine returned an unsafe design asset path.")
        }
        let descriptor = Darwin.open(url.path, O_RDONLY | O_NOFOLLOW | O_NONBLOCK | O_CLOEXEC)
        guard descriptor >= 0 else { throw RADError.message("CAD engine design asset is missing or is a symbolic link.") }
        let file = FileHandle(fileDescriptor: descriptor, closeOnDealloc: true)
        var info = stat()
        guard Darwin.fstat(descriptor, &info) == 0,
              info.st_mode & mode_t(S_IFMT) == mode_t(S_IFREG), info.st_nlink == 1,
              info.st_size >= 0, info.st_size <= ProjectStore.maximumAssetBytes else {
            try? file.close()
            throw RADError.message("CAD engine design asset must be a regular file with one link and a supported size.")
        }
        return (file, info.st_size)
    }
}

/// Owns process lifecycle away from the main actor, including launch races,
/// deadline enforcement and a hard-kill fallback for native geometry calls.
private final class EngineProcessJob: @unchecked Sendable {
    private let process = Process()
    private let lock = NSLock()
    private var cancelled = false
    private var finished = false
    private var launched = false
    private var timedOut = false
    private let directory: URL
    private let artifacts: EngineBuildArtifacts
    private let workerInput: FileHandle
    private var stderr: FileHandle?

    init(executable: URL, arguments: [String], artifacts: EngineBuildArtifacts, environment: [String: String]) throws {
        self.artifacts = artifacts
        directory = artifacts.folder
        workerInput = try artifacts.workerInput()
        process.executableURL = executable
        process.arguments = arguments
        process.currentDirectoryURL = artifacts.folder
        process.environment = environment
        process.standardInput = workerInput
        process.standardOutput = FileHandle.nullDevice
    }

    /// Launch the process; `run` launches it when this was not called first.
    /// Foundation delivers a process's termination on its launching thread's
    /// run loop, so one thread launches the process and reaps it, however
    /// long a prepared worker waits for its build.
    func start() throws {
        let log = directory.appendingPathComponent("stderr.log")
        FileManager.default.createFile(atPath: log.path, contents: nil)
        let handle = try FileHandle(forWritingTo: log)
        lock.lock()
        defer { lock.unlock() }
        guard !cancelled else { finished = true; try? handle.close(); throw CancellationError() }
        stderr = handle
        process.standardError = handle
        let started = DispatchSemaphore(value: 0)
        var failure: Error?
        let reaper = Thread { [process] in
            do { try process.run() } catch { failure = error; started.signal(); return }
            started.signal()
            process.waitUntilExit()
        }
        reaper.name = "RAD CAD worker"
        reaper.start()
        started.wait()
        if let failure { finished = true; try? handle.close(); throw failure }
        launched = true
    }

    var isRunning: Bool { lock.withLock { launched && !finished && process.isRunning } }

    /// Stop a launched worker that no build awaits. Its directory, which its
    /// process keeps leased, stays owned until that process has exited.
    func discard() {
        cancel()
        DispatchQueue.global(qos: .utility).async { [self] in
            while lock.withLock({ launched && process.isRunning }) { Thread.sleep(forTimeInterval: 0.03) }
            lock.withLock { finished = true }
            try? workerInput.close(); try? stderr?.close()
        }
    }

    /// Deliver a prepared worker's build options through its private FIFO,
    /// waiting while it finishes loading the CAD runtime.
    func begin(options: Data, at start: URL) throws {
        guard options.count <= Int(PIPE_BUF) else { throw RADError.message("The CAD build options are too large.") }
        var descriptor: Int32 = -1
        while descriptor < 0 {
            descriptor = Darwin.open(start.path, O_WRONLY | O_NONBLOCK | O_NOFOLLOW | O_CLOEXEC)
            // ENXIO: the worker has not opened its end yet.
            if descriptor < 0 {
                guard errno == ENXIO, isRunning, !lock.withLock({ cancelled }) else {
                    throw RADError.message("The prepared CAD worker stopped before its build.")
                }
                Thread.sleep(forTimeInterval: 0.005)
            }
        }
        defer { Darwin.close(descriptor) }
        let written = options.withUnsafeBytes { Darwin.write(descriptor, $0.baseAddress, $0.count) }
        guard written == options.count else { throw RADError.message("The prepared CAD worker did not accept its build.") }
    }

    func run() async throws -> Int32 {
        try await withCheckedThrowingContinuation { continuation in
            DispatchQueue.global(qos: .userInitiated).async { [self] in
                defer { try? workerInput.close(); try? stderr?.close() }
                do {
                    if !lock.withLock({ launched }) { try start() }
                    let deadline = Date().addingTimeInterval(600)
                    while process.isRunning {
                        if Date() >= deadline {
                            lock.lock(); timedOut = true; lock.unlock()
                            cancel()
                        }
                        Thread.sleep(forTimeInterval: 0.03)
                    }
                    lock.lock()
                    finished = true
                    let didCancel = cancelled, didTimeOut = timedOut
                    lock.unlock()
                    if didTimeOut {
                        throw EngineBuildFailure(kind: .timeout, message: "The CAD engine exceeded its 10-minute limit. Simplify the design and try again.")
                    }
                    if didCancel { throw CancellationError() }
                    continuation.resume(returning: process.terminationStatus)
                } catch { continuation.resume(throwing: error) }
            }
        }
    }

    func cancel() {
        lock.lock()
        cancelled = true
        let running = launched && !finished && process.isRunning
        let pid = running ? process.processIdentifier : 0
        if running { process.terminate() }
        lock.unlock()
        if pid > 0 {
            DispatchQueue.global(qos: .utility).asyncAfter(deadline: .now() + 1) { [self] in
                lock.lock()
                if !finished && process.isRunning { Darwin.kill(pid, SIGKILL) }
                lock.unlock()
            }
        }
    }
}
