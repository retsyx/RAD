import Foundation
import XCTest
@testable import RAD

/// This opt-in test uses the installed, signed-in Codex CLI. It works in an
/// empty temporary task, never opens or changes a user's RAD document.
final class CodexAgentIntegrationTests: XCTestCase {
    @MainActor
    func testInstalledCodexCanBuildInspectMeasureValidateRenderAndExportThroughRAD() async throws {
        guard ProcessInfo.processInfo.environment["RAD_RUN_CODEX_AGENT_TESTS"] == "1" else {
            throw XCTSkip("Set RAD_RUN_CODEX_AGENT_TESTS=1 for the signed-in Codex/MCP/native CAD workflow.")
        }
        let executable = ProcessInfo.processInfo.environment["RAD_TEST_CODEX_PATH"] ?? CodexService.findExecutable() ?? ""
        guard !executable.isEmpty else { throw XCTSkip("Install and sign in to Codex before running this integration check.") }
        let model = ProcessInfo.processInfo.environment["RAD_TEST_CODEX_MODEL"] ?? ""
        let capture = AgentIntegrationCapture()
        let service = CodexService()
        let result = try await service.generate(prompt: """
        This is a small RAD tool integration verification in an empty design.
        Create one solid: a circular loft from radius 5 mm at Z=0 to radius 3 mm at Z=12,
        with a concentric radius 1 mm through-hole. Use RAD's radsdf modeling API.
        Call rad_runtime; write a complete .py recipe; call rad_build; call rad_inspect with
        kind probe to check the wall between the hole and the outside, and rad_measure for
        the part; call rad_validate; call rad_render and review the returned image; call
        rad_export_3mf to a path in the task workspace. Finish with the schema JSON, the exact program you built,
        assetIDs=[] and a short summary mentioning what the rendered shape shows.
        Do not install packages or inspect unrelated files. Do not rebuild the CAD runtime.
        """, current: nil, history: [], executable: executable, model: model,
            onProgress: { _ in }, onMessage: { _ in }, onActivity: { capture.append($0) },
            python: EngineService.defaultPython, threads: 4)
        guard case .ready(let spec) = result else { return XCTFail("Expected a completed CAD design.") }
        let preview = try XCTUnwrap(service.validatedPreview(for: spec), "The final recipe must reuse geometry validated during the agent turn.")
        XCTAssertEqual(preview.solids, 1)
        XCTAssertEqual(preview.volume, 184 * .pi, accuracy: 184 * .pi * 0.01)
        let completed = Set(capture.events.filter { $0.category == "tool" && $0.status == "completed" }.map(\.title))
        for tool in ["rad_runtime", "rad_build", "rad_inspect", "rad_measure", "rad_validate", "rad_render", "rad_export_3mf"] {
            XCTAssertTrue(completed.contains(tool), "The live Codex turn did not successfully use \(tool).")
        }
        XCTAssertTrue(capture.events.contains { $0.title == "rad_render" && $0.output?.contains("image_path") == true })
        XCTAssertTrue(capture.events.contains { $0.title == "rad_build" && $0.files?.contains(where: { $0.data == Data(spec.script.utf8) }) == true })
        XCTAssertFalse(capture.events.contains { $0.output?.contains("\"data\":") == true && $0.output?.contains("image/png") == true })
        print("Live Codex verified RAD tools: \(completed.sorted().joined(separator: ", ")); \(preview.solids) solid, \(preview.volume) mm³; validated native geometry reused.")
        try? FileManager.default.removeItem(at: URL(fileURLWithPath: preview.buildID).deletingLastPathComponent())
    }
}

private final class AgentIntegrationCapture: @unchecked Sendable {
    private let lock = NSLock()
    private var stored = [String: ProjectActivityEvent]()
    private var order = [String]()
    func append(_ event: ProjectActivityEvent) {
        lock.lock(); defer { lock.unlock() }
        if stored[event.id] == nil { order.append(event.id) }
        stored[event.id] = event
    }
    var events: [ProjectActivityEvent] {
        lock.lock(); defer { lock.unlock() }
        return order.compactMap { stored[$0] }
    }
}
