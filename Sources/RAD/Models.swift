import Foundation

struct ModelSpec: Codable, Sendable, Equatable {
    var title: String
    var summary: String
    var units: String
    var script: String
    /// Existing user placements may change only when explicitly requested.
    var reorganizePartIDs: [String]? = nil
    /// Portable inputs required by imported_assets in the saved CAD program.
    var inputAssets: [CADInputAsset]? = nil

    static let bracket = ModelSpec(
        title: "Mounting bracket",
        summary: "An 80 × 50 mm mounting bracket with a 40 mm upright, four 6 mm mounting holes, and a 16 mm cable opening. All dimensions are in millimeters.",
        units: "mm",
        script: """
        base = box((80, 50, 6), center=(0, 0, 3))
        wall = box((80, 6, 40), center=(0, 22, 20))
        holes = [cylinder(3, 10, center=(x, y, 3)) for x, y in [(-28, -14), (28, -14), (-28, 8), (28, 8)]]
        opening = cylinder(8, 12, center=(0, 22, 25), axis=(0, 1, 0))
        result = difference(union(base, wall), *holes, opening)
        """
    )

    private enum CodingKeys: String, CodingKey {
        case title, summary, units, script, reorganizePartIDs, inputAssets
    }
}

extension ModelSpec {
    init(from decoder: Decoder) throws {
        enum UnsupportedFields: String, CodingKey { case nodes, result }
        let unsupported = try decoder.container(keyedBy: UnsupportedFields.self)
        guard !unsupported.contains(.nodes), !unsupported.contains(.result) else {
            throw RADError.message("Feature graphs are unsupported. A design must contain a complete CAD program.")
        }
        let values = try decoder.container(keyedBy: CodingKeys.self)
        title = try values.decode(String.self, forKey: .title)
        summary = try values.decode(String.self, forKey: .summary)
        units = try values.decode(String.self, forKey: .units)
        script = try values.decode(String.self, forKey: .script)
        reorganizePartIDs = try values.decodeIfPresent([String].self, forKey: .reorganizePartIDs)
        inputAssets = try values.decodeIfPresent([CADInputAsset].self, forKey: .inputAssets)
    }
}

struct PartNode: Codable, Sendable, Identifiable, Equatable {
    var id: String
    var name: String
    var solidIndices: [Int]
    var children: [PartNode]? = nil
    /// Opaque sRGB color for this node's own solids.
    var color: PartColor? = nil

    var isGroup: Bool { children != nil }
    var allSolidIndices: Set<Int> {
        Set(solidIndices).union((children ?? []).flatMap { $0.allSolidIndices })
    }
    var allNodeIDs: Set<String> { Set([id]).union((children ?? []).flatMap { $0.allNodeIDs }) }
    var flattened: [PartNode] { [self] + (children ?? []).flatMap(\.flattened) }
}

/// Versions that define a build's behaviour: Python, numpy and the native SDF kernel.
struct CADRuntime: Codable, Sendable, Equatable {
    var python: String
    var numpy: String
    var kernel: String
}

/// One implicit() shape of a build: the parts that read it, the program
/// line that made it, its bounds, the upper bound of the field's slope the
/// kernel proved and divides the field by, and median slope / bound (1 for
/// a true distance; below 1 the solid's offsets, shells and blends come out
/// larger than requested).
struct CADCustomField: Codable, Sendable, Equatable {
    /// The build summary analyses at most this many implicit() shapes
    /// (the worker's `MAX_SUMMARY_FIELDS`).
    static let maximumSummaryFields = 32
    var solids: [Int]
    var line: Int? = nil
    var bounds: [Double]
    var slopeBound: Double
    var quality: Double
}

/// Display geometry for one build. Bulk arrays are packed: SceneKit uploads
/// Float32 attributes, and the worker publishes them as binary sections beside
/// its JSON metadata (see `PreviewMeshBuffers`). They are never encoded.
struct PreviewMesh: Codable, Sendable {
    /// Mesher statistics of one build's preview resolution.
    struct MeshStatistics: Codable, Sendable, Equatable {
        /// Edge length of the dual contouring grid's base cells, in millimeters.
        var cellSize: Double
        var boundaryEdges: Int
        var nonmanifoldEdges: Int
        /// The largest part's bound of the distance from its triangles to the exact surface, in millimeters.
        var maxDeviation: Double? = nil
    }

    /// Vertex coordinates, xyz per vertex.
    var positions: [Float] = []
    /// Vertex indices, three per triangle, outward counter-clockwise.
    var triangleIndices: [UInt32] = []
    /// Crease segments, start xyz then end xyz per segment.
    var edgePoints: [Float] = []
    /// The build's unique session identity.
    var buildID: String
    /// The build's private RSDF graph (`model.rsdf`), the source of every
    /// native operation. Never stored in a project.
    var artifactPath: String? = nil
    /// RSDF node index of each part's root, by solid index.
    var partRoots: [Int]? = nil
    var bounds: [Double]
    var volume: Double
    var area: Double
    var solids: Int
    var buildSeconds: Double
    var threads: Int
    /// Solid index per triangle.
    var triangleSolids: [Int]? = nil
    /// Solid index per crease segment.
    var edgeSolids: [Int]? = nil
    var parts: [PartNode]? = nil
    /// Mesh volume, area, bounds and centroid of each part in built coordinates.
    var partStats: [CADPart]? = nil
    var meshStatistics: MeshStatistics? = nil
    var runtime: CADRuntime? = nil
    /// Mesher findings that do not fail the build, such as a coarsened preview.
    var warnings: [String]? = nil

    /// Unit shading normals, xyz per vertex: SDF gradients, split at creases.
    /// Derived worker output only; project archives never retain display
    /// buffers. Synthetic previews may omit these.
    var vertexNormals: [Float]? = nil

    /// Successful worker validation receipt, bound to complete replay inputs.
    var validatedHash: String? = nil

    /// Reconstructed from the complete CAD program; never a replacement for it.
    var assemblyMotion: AssemblyMotion? = nil
    /// Derived rigid placements relative to the built native geometry.
    var assemblyPose: AssemblyPose? = nil

    /// The program reads `rad.print`, so its 3D printing version is a
    /// separate build. Otherwise this geometry also serves for printing.
    var printVersion: Bool? = nil
    /// Program-pinned outward directions that face the build plate, by solid.
    var printDown: [PrintDownPin]? = nil
    /// Sorted solid indices of reference parts: shown, measured and checked
    /// for clearance, but never laid out or exported for 3D printing.
    var referenceSolids: [Int]? = nil
    /// The build's implicit() shapes (custom fields), at most 32.
    var customFields: [CADCustomField]? = nil

    /// The build's GPU field program (field.bin), validated; nil when the
    /// build has none. Session data like the display buffers.
    var fieldProgram: FieldProgram? = nil
    /// Exact-face proxies: result.json `proxy` and the mesh.bin proxy
    /// sections (xyz per vertex, three outward counter-clockwise indices and
    /// one solid per triangle, grouped by solid). Session data.
    var proxy: CADFieldProxyManifest? = nil
    var proxyPositions: [Float]? = nil
    var proxyTriangleIndices: [UInt32]? = nil
    var proxySolids: [Int]? = nil

    /// Keeps private native files alive across copies and history navigation.
    /// Decoding derived worker JSON never grants ownership of arbitrary paths.
    var buildArtifacts: EngineBuildArtifacts? = nil

    /// Session-only viewport buffers, converted once and shared by every copy.
    var preparedDisplay = CADViewportPreparedMesh.Shared()

    private enum CodingKeys: String, CodingKey {
        case buildID, artifactPath, partRoots, bounds, volume, area, solids, parts, partStats
        case meshStatistics = "mesh"
        case buildSeconds, threads, runtime, warnings, validatedHash, assemblyMotion, printVersion, printDown
        case referenceSolids, customFields
    }

    var vertexCount: Int { positions.count / 3 }
    var triangleCount: Int { triangleIndices.count / 3 }
    var edgeSegmentCount: Int { edgePoints.count / 6 }

    func position(_ index: Int) -> SIMD3<Float> {
        let start = index * 3
        return SIMD3(positions[start], positions[start + 1], positions[start + 2])
    }

    var hasValidNormals: Bool {
        guard let vertexNormals, vertexNormals.count == positions.count else { return false }
        for start in stride(from: 0, to: vertexNormals.count, by: 3) {
            let x = vertexNormals[start], y = vertexNormals[start + 1], z = vertexNormals[start + 2]
            guard x.isFinite, y.isFinite, z.isFinite, abs(x * x + y * y + z * z - 1) <= 0.0001 else { return false }
        }
        return true
    }

    var size: [Double] {
        guard bounds.count == 6 else { return [0, 0, 0] }
        return (0..<3).map { bounds[$0 + 3] - bounds[$0] }
    }
}

struct ChatMessage: Codable, Sendable, Identifiable, Equatable {
    var id = UUID()
    var role: String
    var content: String
    var date = Date()
    var attachments: [SourceAttachment]? = nil
    /// Assistant updates, conversational replies, and validated CAD closeouts.
    var phase: String? = nil
    /// Immutable user selection captured when this request was submitted.
    var selection: ProjectSelectionContext? = nil
}

/// A submitted follow-up awaiting its own turn; distinct from sent chat history.
struct ProjectQueuedRequest: Codable, Sendable, Identifiable, Equatable {
    var id = UUID()
    var content: String
    var date = Date()
    var attachments: [SourceAttachment]? = nil
    /// Capture the choice at enqueue time without changing later composer choices.
    var codexSettings: CodexProjectSettings? = nil
    /// Exact user-reference paths already snapshotted, including aliases that
    /// may disappear or be retargeted before this request is dispatched.
    var capturedSourcePaths: [String]? = nil
    /// Retained while queued, edited or sent as same-turn information.
    var selection: ProjectSelectionContext? = nil
}

struct ProjectRevision: Codable, Sendable, Identifiable {
    var id = UUID()
    var model: ModelSpec
    var date = Date()
    var preview: PreviewMesh? = nil
    var partLayout: PartLayout? = nil
    var hiddenSolids: [Int]? = nil
    /// Stable default hierarchy, independent of the regenerable display mesh.
    var parts: [PartNode]? = nil
    /// Versions used to validate the saved modeling instructions.
    var runtime: CADRuntime? = nil
    /// The successful conversation turn that produced this validated model.
    var userMessageID: UUID? = nil
    var responseMessageID: UUID? = nil
    /// User-selected opaque colors, addressed by stable individual part IDs.
    var partColors: [String: PartColor]? = nil
    /// Optional user label; unnamed revisions display their stable history position.
    var name: String? = nil
    /// Reuse expensive validation only while the exact replay inputs match.
    var validatedHash: String? = nil
    /// Joint coordinates are presentation state; the program defines the mates.
    var jointPositions: [String: Double]? = nil
}

/// Revision labels are document metadata, not filenames or generated model titles.
enum ProjectRevisionNaming {
    static let maximumNameBytes = 512

    static func userName(_ entered: String) throws -> String? {
        guard entered.unicodeScalars.allSatisfy({
            !CharacterSet.controlCharacters.contains($0) && !CharacterSet.newlines.contains($0)
        }) else { throw RADError.message("Version names cannot contain line breaks or control characters.") }
        let name = entered.trimmingCharacters(in: .whitespacesAndNewlines)
        guard name.utf8.count <= maximumNameBytes else {
            throw RADError.message("The version name is too long.")
        }
        return name.isEmpty ? nil : name
    }

    static func validateSavedName(_ name: String) throws {
        guard try userName(name) == name else {
            throw RADError.message("The saved version name is empty or contains surrounding whitespace.")
        }
    }
}

struct RADProject: Codable, Sendable {
    var formatVersion = 1
    var messages: [ChatMessage]
    var revisions: [ProjectRevision]
    var selectedRevisionID: UUID?
    /// Stable identity for autosave naming, independent of title and revisions.
    /// Identity remains stable across renames and model revisions.
    var projectID: UUID? = nil
    /// Model and reasoning choices travel with this document, not app settings.
    var codexSettings: CodexProjectSettings? = nil
    /// An explicit user name overrides generated design titles for this project.
    var projectName: String? = nil
    /// Grid visibility belongs to the project; absent values use the default.
    var showGrid: Bool? = nil
    /// The 3D printing views are local to this design; absent means off.
    var printVersionView: Bool? = nil
    var printLayoutView: Bool? = nil
    /// Projection and surface rendering choices are local to this design.
    var projectionMode: ViewportProjectionMode? = nil
    var displayMode: ViewportDisplayMode? = nil
    /// Most recently chosen part colors, newest first, local to this design.
    var recentPartColors: [PartColor]? = nil
    /// Tool and command activity belongs to this project and survives reopening.
    var taskAudits: [ProjectTaskAudit]? = nil
    /// Pending turns and the current unsent composer travel with this project.
    var queuedRequests: [ProjectQueuedRequest]? = nil
    var draftPrompt: String? = nil
    var draftAttachments: [SourceAttachment]? = nil
    var queuePaused: Bool? = nil
    /// Preferred share of the sidebar height reserved for Parts, local to this project.
    var partsPaneFraction: Double = 0.75
    /// Session-only file baseline for conflict detection, including first save.
    var fileOrigin: ProjectFileOrigin? = nil
    /// Optional Finder presentation image. Package storage saves it separately;
    /// it is never part of conversation, task activity, or CAD replay inputs.
    var thumbnailData: Data? = nil

    private enum CodingKeys: String, CodingKey {
        case formatVersion, messages, revisions, selectedRevisionID, projectID, codexSettings, projectName
        case showGrid, printVersionView, printLayoutView, projectionMode, displayMode, recentPartColors, taskAudits, queuedRequests
        case draftPrompt, draftAttachments, queuePaused, partsPaneFraction
    }
}

enum RADError: LocalizedError {
    case message(String)
    var errorDescription: String? {
        switch self { case .message(let message): return message }
    }
}

enum ResourcePaths {
    static var root: URL {
        if let path = ProcessInfo.processInfo.environment["RAD_ROOT"] { return URL(fileURLWithPath: path) }
        // Packaged app keeps its engine alongside the resource bundle.
        let packaged = Bundle.main.bundleURL.appendingPathComponent("Contents/Resources")
        if Bundle.main.bundleURL.pathExtension == "app" { return packaged }
        if FileManager.default.fileExists(atPath: packaged.appendingPathComponent("Engine/worker.py").path) { return packaged }
        return URL(fileURLWithPath: FileManager.default.currentDirectoryPath)
    }

    static func resource(_ name: String) -> URL? {
        Bundle.module.url(forResource: name, withExtension: nil, subdirectory: "Resources")
    }

    static let kernelLibraryName = "libRADSDF.dylib"

    /// The native SDF kernel CAD workers load, as a real path: the sandbox
    /// profile matches it literally. A packaged app uses only its own signed
    /// copy; development runs honor RAD_SDF_LIBRARY, else the checkout's
    /// SwiftPM products, release before debug.
    static var kernelLibrary: URL? {
        if Bundle.main.bundleURL.pathExtension == "app" {
            return realFile(Bundle.main.bundleURL.appendingPathComponent("Contents/Frameworks/\(kernelLibraryName)"))
        }
        if let configured = ProcessInfo.processInfo.environment["RAD_SDF_LIBRARY"], !configured.isEmpty {
            return realFile(URL(fileURLWithPath: configured))
        }
        let build = root.appendingPathComponent(".build")
        for products in ["release", "out/Products/Release", "debug", "out/Products/Debug"] {
            if let library = realFile(build.appendingPathComponent(products).appendingPathComponent(kernelLibraryName)) {
                return library
            }
        }
        return nil
    }

    /// An existing regular file's real path. Unlike `resolvingSymlinksInPath`,
    /// this keeps the `/private` prefix that sandbox path rules see.
    static func realFile(_ url: URL) -> URL? {
        var buffer = [CChar](repeating: 0, count: Int(PATH_MAX))
        guard Darwin.realpath(url.path, &buffer) != nil else { return nil }
        let resolved = URL(fileURLWithPath: String(cString: buffer))
        var directory: ObjCBool = false
        guard FileManager.default.fileExists(atPath: resolved.path, isDirectory: &directory), !directory.boolValue else { return nil }
        return resolved
    }
}
