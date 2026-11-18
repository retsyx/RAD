import AppKit
import CryptoKit
import Foundation

struct ProjectAgentCandidate {
    let model: ModelSpec
    let preview: PreviewMesh
    let colorChanges: [CADPartColorChange]
}

/// A native tool result for an external client. Its metadata names task files
/// by opaque IDs, never workspace paths, and `files` describes every file
/// handle the task has issued.
struct ProjectAgentToolResult {
    let result: CADAgentToolResult
    let files: [[String: Any]]
}

/// An external agent owns its files, CAD worker and speculative candidates.
/// Document routing, authorization and revision publication remain with RAD.
@MainActor
final class ProjectAgentSession {
    static let maximumUploadBytes = 64 * 1024 * 1024
    static let maximumFileBytes = 512 * 1024 * 1024
    static let maximumStoredBytes = 512 * 1024 * 1024
    static let maximumFiles = 2_048

    private struct File {
        let id: String
        let path: String
        let name: String
        let byteCount: Int
        let uploaded: Bool
        /// SHA-256 of an uploaded mesh file, compared with generated geometry.
        var contentDigest: String? = nil

        var isGeometry: Bool { CADInputAsset.supportedFormats.contains(URL(fileURLWithPath: path).pathExtension.lowercased()) }
    }

    let id: UUID
    let workspaceURL: URL
    private let workspace: CADAgentWorkspace
    private let tools: CADAgentTools
    private let onActivity: @Sendable (ProjectActivityEvent) -> Void
    private var files: [String: File] = [:]
    private var fileIDsByPath: [String: String] = [:]
    private var storedBytes = 0
    private var pendingUploadBytes = 0
    private var pendingUploadCount = 0
    /// Digests of mesh files RAD generated and the client downloaded. Such
    /// files never become imported reference inputs when uploaded again.
    private var generatedGeometryDigests = Set<String>()
    private(set) var isBusy = false
    private(set) var isCancelled = false

    init(id: UUID = UUID(), python: String, threads: Int, current: ModelSpec?, currentPreview: PreviewMesh?,
         versions: ProjectVersionContext = .empty, partLayout: PartLayout? = nil,
         partColors: [String: PartColor] = [:], viewport: CADSceneView? = nil,
         permissionWindow: NSWindow? = nil, hiddenSolids: Set<Int> = [],
         solidColors: [Int: PartColor] = [:],
         onActivity: @escaping @Sendable (ProjectActivityEvent) -> Void = { _ in }) throws {
        self.id = id
        self.onActivity = onActivity
        let folder = FileManager.default.temporaryDirectory.appendingPathComponent("RAD-Agent-\(id.uuidString)", isDirectory: true)
        try FileManager.default.createDirectory(at: folder, withIntermediateDirectories: false,
                                                attributes: [.posixPermissions: 0o700])
        do {
            workspace = try CADAgentWorkspace(directory: folder)
            workspaceURL = workspace.root
            tools = try CADAgentTools(directory: folder, python: python, threads: max(1, threads),
                current: current, currentPreview: currentPreview, engine: EngineService(), viewport: viewport,
                hiddenSolids: hiddenSolids, solidColors: solidColors, versions: versions,
                partLayout: partLayout, partColors: partColors, permissionWindow: permissionWindow)
        } catch {
            try? FileManager.default.removeItem(at: folder)
            throw error
        }
    }

    deinit { try? FileManager.default.removeItem(at: workspaceURL) }

    func upload(data: Data, name: String) throws -> [String: Any] {
        try checkAvailable()
        guard !data.isEmpty, data.count <= Self.maximumUploadBytes else {
            throw RADError.message("Upload a nonempty task file of at most 64 MB.")
        }
        let name = try Self.filename(name)
        try checkStorage(adding: data.count)
        let identity = UUID().uuidString
        let url = try workspace.write(data, path: "uploads/\(identity)/\(name)")
        return descriptor(try register(url.path, uploaded: true, identity: identity,
                                       contentDigest: Self.geometryDigest(data, name: name)))
    }

    /// HTTP transfers perform large writes and hashing away from AppKit. Keep
    /// synchronous counterparts for small in-process callers and test fixtures.
    func uploadFile(data: Data, name: String) async throws -> [String: Any] {
        try checkAvailable()
        guard !data.isEmpty, data.count <= Self.maximumUploadBytes else {
            throw RADError.message("Upload a nonempty task file of at most 64 MB.")
        }
        let name = try Self.filename(name)
        try checkStorage(adding: data.count)
        pendingUploadBytes += data.count; pendingUploadCount += 1
        defer { pendingUploadBytes -= data.count; pendingUploadCount -= 1 }
        let identity = UUID().uuidString, workspace = workspace
        let relative = "uploads/\(identity)/\(name)"
        var published = false
        defer { if !published { try? workspace.removeFile(relative) } }
        let prepared = Task.detached(priority: .userInitiated) {
            try Task.checkCancellation()
            return (try workspace.write(data, path: relative).path, Self.geometryDigest(data, name: name))
        }
        let (path, digest) = try await withTaskCancellationHandler {
            try await prepared.value
        } onCancel: { prepared.cancel() }
        try checkAvailable()
        // The pending reservation already includes this file. Release it just
        // while atomically publishing the uploaded handle on the main actor.
        pendingUploadBytes -= data.count; pendingUploadCount -= 1
        defer { pendingUploadBytes += data.count; pendingUploadCount += 1 }
        let result = descriptor(try register(path, uploaded: true, identity: identity, contentDigest: digest))
        published = true
        return result
    }

    func download(fileID: String) throws -> (data: Data, name: String) {
        try checkAvailable()
        let file = try file(fileID)
        var data = try workspace.read(file.path, maximumBytes: Self.maximumFileBytes, allowEmpty: true)
        if isGenerated(file) { generatedGeometryDigests.insert(SHA256.hexDigest(data)) }
        // Native metadata files contain local references. Present their JSON
        // through the same opaque handles as tool metadata. Original uploads
        // and programs are always downloaded byte-for-byte.
        if !file.uploaded, URL(fileURLWithPath: file.path).pathExtension == "json", data.count <= 2_000_000,
           let object = try? JSONSerialization.jsonObject(with: data) {
            data = try JSONSerialization.data(withJSONObject: externalized(object), options: [.sortedKeys, .withoutEscapingSlashes])
        }
        return (data, file.name)
    }

    func downloadFile(fileID: String) async throws -> (data: Data, name: String) {
        try checkAvailable()
        let file = try file(fileID), workspace = workspace
        let maximumBytes = Self.maximumFileBytes, generated = isGenerated(file)
        let prepared = Task.detached(priority: .userInitiated) {
            try Task.checkCancellation()
            let bytes = try workspace.read(file.path, maximumBytes: maximumBytes, allowEmpty: true)
            return (bytes, generated ? SHA256.hexDigest(bytes) : nil)
        }
        var (bytes, digest) = try await withTaskCancellationHandler {
            try await prepared.value
        } onCancel: { prepared.cancel() }
        try checkAvailable()
        if let digest { generatedGeometryDigests.insert(digest) }
        if !file.uploaded, URL(fileURLWithPath: file.path).pathExtension == "json", bytes.count <= 2_000_000,
           let object = try? JSONSerialization.jsonObject(with: bytes) {
            bytes = try JSONSerialization.data(withJSONObject: externalized(object), options: [.sortedKeys, .withoutEscapingSlashes])
        }
        return (bytes, file.name)
    }

    func candidate(artifactID: String) async throws -> ProjectAgentCandidate {
        try checkAvailable()
        guard !isBusy else { throw RADError.message("Wait for this task's running CAD operation before committing.") }
        isBusy = true
        defer { isBusy = false }
        let value = try await tools.committableCandidate(artifactID: artifactID)
        return ProjectAgentCandidate(model: value.model, preview: value.preview, colorChanges: value.colorChanges)
    }

    func bindPermissionWindow(_ window: NSWindow) throws {
        try checkAvailable()
        try tools.bindPermissionWindow(window)
    }

    func call(name: String, arguments: [String: Any]) async throws -> ProjectAgentToolResult {
        try checkAvailable()
        guard !isBusy else { throw RADError.message("A CAD operation is already running in this task.") }
        isBusy = true
        defer { isBusy = false }
        let tool = try AgentToolCatalog.native()[name]
        let call = CADAgentToolCall(tools: tools, workspace: workspace, onActivity: onActivity)
        // The activity records the native result, with its workspace paths.
        let native = try await call.perform(id: "tool:\(UUID().uuidString)", name: name, arguments: arguments,
            prepare: { try resolvedArguments(tool: tool, arguments: arguments) },
            checkAvailable: { try checkAvailable() })
        let result = try CADAgentToolResult(metadata: externalized(native.metadata), image: native.image, isError: native.isError)
        return ProjectAgentToolResult(result: result, files: files.values.sorted { $0.id < $1.id }.map(descriptor))
    }

    func cancel() {
        guard !isCancelled else { return }
        isCancelled = true
        tools.cancel()
        files.removeAll(); fileIDsByPath.removeAll(); storedBytes = 0
        try? FileManager.default.removeItem(at: workspaceURL)
    }

    private func resolvedArguments(tool: AgentTool?, arguments: [String: Any]) throws -> [String: Any] {
        var values = arguments
        guard values["taskID"] == nil else { throw RADError.message("Route this request to its chosen task before executing it.") }
        if let program = values.removeValue(forKey: "programID") {
            guard values["programPath"] == nil else { throw RADError.message("Provide one program file ID.") }
            values["programPath"] = program
        }
        let export = tool?.host.exportPath
        if let export {
            guard values[export] == nil else { throw RADError.message("RAD assigns an export file ID; do not provide an output path.") }
        }
        for key in Array(values.keys) where key == "path" || key.hasSuffix("Path") || key.hasSuffix("Paths") {
            if key.hasSuffix("Paths") {
                guard let ids = values[key] as? [String], ids.count <= 64 else {
                    throw RADError.message("Provide an array of at most 64 task file IDs.")
                }
                values[key] = try ids.map { try file($0).path }
            } else {
                guard let identity = values[key] as? String else { throw RADError.message("Provide a task file ID for \(key).") }
                let staged = try file(identity)
                if tool?.host.references.contains(key) == true, !isOriginalReference(staged) {
                    throw RADError.message("Import original external reference files uploaded to this task. Geometry RAD generated, such as an STL export, cannot become a reference input.")
                }
                values[key] = staged.path
            }
        }
        if let export {
            // Every download handle denotes one immutable export, including
            // exports of the same artifact before and after later edits.
            values[export] = "exports/\(UUID().uuidString).stl"
        }
        return values
    }

    private func externalized(_ object: [String: Any]) throws -> [String: Any] {
        try externalized(object as Any) as? [String: Any] ?? [:]
    }

    private func externalized(_ value: Any) throws -> Any {
        if let object = value as? [String: Any] {
            var transformed: [String: Any] = [:]
            for (key, child) in object {
                if key == "workspace" || key == "pythonExecutable" { continue }
                if key == "externalTools" {
                    transformed[key] = "External tools are user-managed. Request a current allowed decision from rad_request_permission before each new optional download or dependency installation. The permission tool executes nothing. Use an isolated temporary environment for approved client-side work and upload required original reference files. Never modify RAD's shared CAD runtime or original inputs. Final CAD programs must replay using bundled dependencies and explicitly registered original references only. Generated design geometry must remain in the complete replayable program."
                    continue
                }
                transformed[key == "programPath" ? "programID" : key] = try externalized(child)
            }
            return transformed
        }
        if let list = value as? [Any] { return try list.map(externalized) }
        if let path = value as? String, path.hasPrefix(workspace.root.path + "/") {
            return try register(path, uploaded: false).id
        }
        return value
    }

    /// A mesh RAD wrote into this task (an export) rather than a client upload
    /// or a saved design's original input.
    private func isGenerated(_ file: File) -> Bool {
        !file.uploaded && file.isGeometry && !file.path.hasPrefix(workspace.root.path + "/context/versions/inputs/")
    }

    private func isOriginalReference(_ file: File) -> Bool {
        if file.path.hasPrefix(workspace.root.path + "/context/versions/inputs/") { return true }
        return file.uploaded && file.contentDigest.map { !generatedGeometryDigests.contains($0) } != false
    }

    private func register(_ path: String, uploaded: Bool, identity: String = UUID().uuidString,
                          contentDigest: String? = nil) throws -> File {
        let url = try workspace.url(path)
        if let id = fileIDsByPath[url.path], let file = files[id] { return file }
        // Descriptor-confined inspection checks every parent and rejects links
        // without loading a large export into the UI process again.
        let byteCount = try workspace.fileSize(url.path, maximumBytes: Self.maximumFileBytes, allowEmpty: true)
        try checkStorage(adding: byteCount)
        let file = File(id: identity, path: url.path, name: url.lastPathComponent, byteCount: byteCount, uploaded: uploaded,
                        contentDigest: contentDigest)
        files[file.id] = file; fileIDsByPath[file.path] = file.id; storedBytes += file.byteCount
        return file
    }

    private func file(_ identity: String) throws -> File {
        guard UUID(uuidString: identity) != nil, let file = files[identity] else {
            throw RADError.message("This file ID is unavailable in the chosen task. Upload or look up the file first.")
        }
        return file
    }

    private func descriptor(_ file: File) -> [String: Any] {
        var value: [String: Any] = ["fileID": file.id, "name": file.name]
        // Downloaded native JSON substitutes opaque handles for paths, so its
        // wire size is computed by the HTTP response after that substitution.
        if file.uploaded || URL(fileURLWithPath: file.path).pathExtension != "json" {
            value["byteCount"] = file.byteCount
        }
        return value
    }

    private func checkStorage(adding bytes: Int) throws {
        guard files.count + pendingUploadCount < Self.maximumFiles,
              bytes <= Self.maximumStoredBytes - storedBytes - pendingUploadBytes else {
            throw RADError.message("This task's temporary file storage limit has been reached. Start a new task.")
        }
    }

    private func checkAvailable() throws {
        if isCancelled { throw CancellationError() }
        try Task.checkCancellation()
    }

    private static func filename(_ name: String) throws -> String {
        guard !name.isEmpty, name.utf8.count <= 255, name != ".", name != "..",
              !name.contains("/"), !name.contains("\\"),
              name.unicodeScalars.allSatisfy({ !CharacterSet.controlCharacters.contains($0) }) else {
            throw RADError.message("Provide a filename without directories or control characters.")
        }
        return name
    }

    nonisolated private static func geometryDigest(_ data: Data, name: String) -> String? {
        CADInputAsset.supportedFormats.contains(URL(fileURLWithPath: name).pathExtension.lowercased()) ? SHA256.hexDigest(data) : nil
    }
}
