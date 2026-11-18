import Darwin
import Foundation
import XCTest
@testable import RAD

/// Hosts RAD's external MCP server for an agent that drives the CAD tools from a
/// shell, through the same tools, tasks, commits and autosave writer an external
/// client of the app uses, without windows. It runs only when
/// RAD_AGENT_HARNESS_DIR names a directory, which holds all of its state:
///
/// - `preferences/` a private defaults suite: server settings and the digest of
///   the harness client's credential
/// - `runtimes/` the CAD runtime, provisioned from the checkout's prepared payload
/// - `designs/` the save folder of every design the agent creates
/// - `harness.json` endpoint, bearer token, port, pid and designs directory
/// - `open-request.json` the `.rad` design inside this directory that the next
///   rad_project_open chooses, standing in for the user's open panel choice
/// - `stop` created to end the session; SIGTERM and SIGINT also end it
/// - `designs.json` each design's title and file, written after the final saves
/// - `harness.log` this session's log, written as it happens
///
/// RAD_AGENT_HARNESS_PORT (default 47631) and RAD_AGENT_HARNESS_SECONDS (default
/// 14400) set the loopback port and the longest session.
final class AgentHarnessTests: XCTestCase {
    @MainActor
    func testHostsExternalAgentServerUntilStopped() async throws {
        let environment = ProcessInfo.processInfo.environment
        guard let path = environment["RAD_AGENT_HARNESS_DIR"], !path.isEmpty else {
            throw XCTSkip("Set RAD_AGENT_HARNESS_DIR to host the external agent server for shell-driven modeling.")
        }
        let host = try AgentHarnessHost(directory: URL(fileURLWithPath: path, isDirectory: true), environment: environment)
        let problems = try await host.run()
        XCTAssertTrue(problems.isEmpty, problems.joined(separator: "\n"))
    }
}

@MainActor
private final class AgentHarnessHost {
    private static let reservedPorts: Set<Int> = [47512, 47522]
    private let directory: URL
    private let designs: URL
    private let runtimes: URL
    private let stopFile: URL
    private let openRequest: URL
    private let logFile: URL
    private let port: Int
    private let seconds: Double
    private let defaults: UserDefaults
    private var coordinator: ProjectWindowCoordinator?
    private var sessions: [ProjectWindowSession] = []
    private var signalSources: [DispatchSourceSignal] = []
    private var stopReason: String?

    init(directory: URL, environment: [String: String]) throws {
        let manager = FileManager.default
        try manager.createDirectory(at: directory, withIntermediateDirectories: true, attributes: [.posixPermissions: 0o700])
        let root = try EngineArtifact.canonicalDirectory(directory)
        self.directory = root
        designs = root.appendingPathComponent("designs", isDirectory: true)
        runtimes = root.appendingPathComponent("runtimes", isDirectory: true)
        stopFile = root.appendingPathComponent("stop")
        openRequest = root.appendingPathComponent("open-request.json")
        logFile = root.appendingPathComponent("harness.log")
        guard let port = Int(environment["RAD_AGENT_HARNESS_PORT"] ?? "47631"), (1024...65535).contains(port),
              !Self.reservedPorts.contains(port) else {
            throw RADError.message("RAD_AGENT_HARNESS_PORT must be within 1024-65535, excluding 47512 and 47522.")
        }
        guard let seconds = Double(environment["RAD_AGENT_HARNESS_SECONDS"] ?? "14400"), seconds.isFinite, seconds > 0 else {
            throw RADError.message("RAD_AGENT_HARNESS_SECONDS must be a positive number of seconds.")
        }
        self.port = port
        self.seconds = seconds
        for folder in [designs, runtimes, root.appendingPathComponent("preferences", isDirectory: true)] {
            try manager.createDirectory(at: folder, withIntermediateDirectories: true, attributes: [.posixPermissions: 0o700])
        }
        // An earlier session's stop request, handoff and results never apply to this one.
        for name in ["stop", "harness.json", "designs.json", "open-request.json", "harness.log"] {
            try? manager.removeItem(at: root.appendingPathComponent(name))
        }
        // A suite named by an absolute path keeps its property list in this directory.
        guard let defaults = UserDefaults(suiteName: root.appendingPathComponent("preferences/RAD-agent-harness").path) else {
            throw RADError.message("The harness preferences could not be created.")
        }
        self.defaults = defaults
    }

    func run() async throws -> [String] {
        installSignalHandlers()
        defer { removeSignalHandlers() }
        guard let kernel = ResourcePaths.kernelLibrary else {
            throw RADError.message("Build the kernel first: swift build --arch arm64 --product RADSDF -c release")
        }
        let payload = ResourcePaths.root.appendingPathComponent(".build/cad-runtime/\(CADRuntimePayload.hostArchitecture)", isDirectory: true)
        log("Provisioning the CAD runtime from \(payload.path) into \(runtimes.path)")
        let python = try await CADRuntimeManager(payloadDirectory: payload, installationRoot: runtimes).ensureReady()
        let designs = designs
        let makeState: @MainActor () -> AppState = {
            AppState(autosaveStore: ProjectAutosaveStore(rootDirectory: designs), python: python)
        }
        let coordinator = ProjectWindowCoordinator(loadProject: { project, url in
            let state = makeState()
            state.loadProject(project, from: url)
            return state
        }, createProjectState: makeState)
        self.coordinator = coordinator
        // The active window is a placeholder that never holds a design, so no
        // design records itself as the app's recent design in standard defaults.
        let placeholder = makeState()
        coordinator.register(ProjectWindowSession(state: placeholder), id: UUID(), sourceURL: nil)
        coordinator.onReopenProjectWindow = { [weak self] request in self?.openWindow(request, state: makeState()) }

        let server = ExternalAgentServer(defaults: defaults, coordinator: coordinator)
        defer { server.stop() }
        server.chooseProject = { [weak self] _ in self?.chooseRequestedDesign() }
        for client in server.access.clients { server.access.revoke(client.id) }
        let issued = try server.access.create(name: "Agent harness", capabilities: Set(ExternalAgentCapability.allCases))
        server.preferredPort = port
        server.enabled = true
        let endpoint = try await waitForEndpoint(server)
        let started = Date(), deadline = started.addingTimeInterval(seconds)
        let threads = placeholder.threads, cores = ProcessInfo.processInfo.activeProcessorCount
        var info: [String: Any] = ["endpoint": endpoint, "token": issued.token, "port": port, "pid": Int(getpid()),
            "designsDirectory": designs.path, "runtimesDirectory": runtimes.path, "python": python, "kernel": kernel.path,
            "threads": threads, "externalTaskThreads": min(threads, max(1, cores / 2)),
            "stopFile": stopFile.path, "openRequestFile": openRequest.path,
            "startedAt": Self.timestamp(started), "endsBy": Self.timestamp(deadline)]
        try writePrivateJSON(info, to: "harness.json")
        log("Serving \(endpoint) for up to \(Int(seconds)) s; kernel \(kernel.path)")

        while stopReason == nil {
            if FileManager.default.fileExists(atPath: stopFile.path) { stopReason = "stop file" }
            else if Date() >= deadline { stopReason = "time limit" }
            else {
                do { try await Task.sleep(for: .milliseconds(250)) } catch { stopReason = "canceled" }
            }
        }
        log("Stopping (\(stopReason ?? "")): ending tasks and saving designs")
        server.stop()
        server.access.revoke(issued.client.id)
        var problems: [String] = [], listed: [[String: Any]] = []
        var seen = Set<ObjectIdentifier>()
        for state in sessions.map(\.state) where seen.insert(ObjectIdentifier(state)).inserted {
            await state.prepareToQuit()
            var entry: [String: Any] = ["title": state.title, "name": state.projectDisplayName,
                "projectID": state.projectDocumentID.uuidString, "versions": state.revisions.count,
                "file": state.projectURL.map { ((try? EngineArtifact.canonicalDirectory($0)) ?? $0).path } ?? NSNull()]
            if let id = state.selectedRevisionID { entry["selectedRevisionID"] = id.uuidString }
            if let status = state.saveStatus, status.hasPrefix("Not saved:") {
                entry["error"] = status
                problems.append("\(state.title): \(status)")
            }
            listed.append(entry)
        }
        try writePrivateJSON(listed, to: "designs.json")
        info.removeValue(forKey: "token")
        info["stoppedAt"] = Self.timestamp(Date())
        info["stopReason"] = stopReason ?? ""
        try writePrivateJSON(info, to: "harness.json")
        ProjectPackage.clearSessionCache()
        log("Saved \(listed.count) design(s); wrote \(directory.appendingPathComponent("designs.json").path)")
        return problems
    }

    /// Stands in for the native window a new or opened design receives.
    private func openWindow(_ request: ProjectWindowRequest, state: AppState) {
        guard let coordinator else { return }
        let session = ProjectWindowSession(state: state)
        // A created design is already retained for this request; registration adopts it.
        let starts = coordinator.register(session, id: request.id, sourceURL: request.fileURL)
        sessions.append(session)
        if starts { session.state.start(restoringRecent: false, opening: request.fileURL) }
        log(request.fileURL.map { "Opening \($0.path)" } ?? "Created design \(session.state.projectDocumentID.uuidString)")
    }

    /// rad_project_open's choice: the design named in open-request.json, when it
    /// is a `.rad` package inside the harness directory that no window holds.
    private func chooseRequestedDesign() -> URL? {
        defer { try? FileManager.default.removeItem(at: openRequest) }
        guard let data = try? Data(contentsOf: openRequest),
              let path = (try? JSONSerialization.jsonObject(with: data) as? [String: Any])?["path"] as? String else {
            log("rad_project_open canceled: no open-request.json")
            return nil
        }
        guard let real = try? EngineArtifact.canonicalDirectory(URL(fileURLWithPath: path, isDirectory: true)),
              real.pathExtension == "rad", real.path.hasPrefix(directory.path + "/"),
              (try? real.resourceValues(forKeys: [.isDirectoryKey]))?.isDirectory == true else {
            log("rad_project_open canceled: \(path) is not a .rad design inside \(directory.path)")
            return nil
        }
        let url = ProjectFileOrigin.canonicalURL(real)
        // Opening a design that is already open would focus and activate its window.
        let open = sessions.contains { session in
            [session.sourceURL, session.state.projectURL].contains { $0.map(ProjectFileOrigin.canonicalURL) == url }
        }
        guard !open else {
            log("rad_project_open canceled: \(url.lastPathComponent) is already open")
            return nil
        }
        return url
    }

    private func waitForEndpoint(_ server: ExternalAgentServer) async throws -> String {
        for _ in 0..<400 {
            if server.isRunning, let endpoint = server.endpoint { return endpoint }
            if let error = server.error { throw RADError.message(error) }
            try await Task.sleep(for: .milliseconds(25))
        }
        throw RADError.message("The external agent server did not start on port \(port).")
    }

    private func installSignalHandlers() {
        for number in [SIGTERM, SIGINT] {
            signal(number, SIG_IGN)
            let source = DispatchSource.makeSignalSource(signal: number, queue: .main)
            source.setEventHandler { [weak self] in
                MainActor.assumeIsolated {
                    guard let self, self.stopReason == nil else { return }
                    self.stopReason = "signal \(number)"
                }
            }
            source.resume()
            signalSources.append(source)
        }
    }

    private func removeSignalHandlers() {
        for source in signalSources { source.cancel() }
        signalSources.removeAll()
        signal(SIGTERM, SIG_DFL)
        signal(SIGINT, SIG_DFL)
    }

    /// Written owner-only and renamed into place: harness.json holds the token.
    private func writePrivateJSON(_ value: Any, to name: String) throws {
        let data = try JSONSerialization.data(withJSONObject: value, options: [.prettyPrinted, .sortedKeys, .withoutEscapingSlashes])
        let target = directory.appendingPathComponent(name)
        let staging = directory.appendingPathComponent(".\(name).\(UUID().uuidString)")
        guard FileManager.default.createFile(atPath: staging.path, contents: data, attributes: [.posixPermissions: 0o600]),
              Darwin.rename(staging.path, target.path) == 0 else {
            try? FileManager.default.removeItem(at: staging)
            throw RADError.message("Could not write \(target.path).")
        }
    }

    /// swift test relays a test's output only when it finishes, so the
    /// session also keeps its own log.
    private func log(_ message: String) {
        let line = "[agent-harness] \(Self.timestamp(Date())) \(message)\n"
        print(line, terminator: "")
        fflush(stdout)
        if let handle = try? FileHandle(forWritingTo: logFile) {
            _ = try? handle.seekToEnd()
            try? handle.write(contentsOf: Data(line.utf8))
            try? handle.close()
        } else {
            FileManager.default.createFile(atPath: logFile.path, contents: Data(line.utf8), attributes: [.posixPermissions: 0o600])
        }
    }

    private static func timestamp(_ date: Date) -> String {
        ISO8601DateFormatter.string(from: date, timeZone: .current, formatOptions: [.withInternetDateTime])
    }
}
