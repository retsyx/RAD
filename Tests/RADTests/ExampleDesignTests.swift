import Foundation
import XCTest
@testable import RAD

/// The shipped example designs (`Examples/<name>.rad`) are written from their
/// programs (`Examples/*.json`) by the app's own build, revision and Save New
/// code, so each holds exactly one validated revision that opens in the app.
final class ExampleDesignTests: XCTestCase {
    struct Example {
        /// The design's name and package filename.
        let design: String
        /// Its program in `Examples`, a model request (title, summary, units, script).
        let program: String
    }

    static let examples = [
        Example(design: "Rubber duck", program: "rubber-duck.json"),
        Example(design: "HDMI board case", program: "hdmi-board-case.json"),
        Example(design: "Geneva drive", program: "geneva-drive.json"),
    ]

    static var folder: URL {
        URL(fileURLWithPath: #filePath).deletingLastPathComponent().deletingLastPathComponent()
            .deletingLastPathComponent().appendingPathComponent("Examples", isDirectory: true)
    }

    static func spec(_ example: Example) throws -> ModelSpec {
        try JSONDecoder().decode(ModelSpec.self, from: Data(contentsOf: folder.appendingPathComponent(example.program)))
    }

    static func package(_ example: Example) -> URL {
        folder.appendingPathComponent("\(example.design).rad", isDirectory: true)
    }

    /// Regenerates the example designs after their programs change:
    /// `RAD_WRITE_EXAMPLE_DESIGNS=<empty folder> RAD_RUN_ENGINE_TESTS=1 swift test --filter
    /// ExampleDesignTests/testWriteExampleDesigns`, then replace `Examples/*.rad` with the written packages.
    @MainActor
    func testWriteExampleDesigns() async throws {
        guard let path = ProcessInfo.processInfo.environment["RAD_WRITE_EXAMPLE_DESIGNS"] else {
            throw XCTSkip("Set RAD_WRITE_EXAMPLE_DESIGNS to an empty folder to write the example designs.")
        }
        let output = URL(fileURLWithPath: path, isDirectory: true)
        XCTAssertEqual(try FileManager.default.contentsOfDirectory(atPath: output.path), [])
        let scratch = FileManager.default.temporaryDirectory.appendingPathComponent("RAD-example-store-\(UUID().uuidString)")
        defer { try? FileManager.default.removeItem(at: scratch) }
        let store = ProjectAutosaveStore(rootDirectory: scratch)
        for example in Self.examples {
            let spec = try Self.spec(example)
            let preview = try await EngineService().buildPreview(spec, python: EngineService.defaultPython, threads: 4)
            XCTAssertEqual(preview.warnings, [], example.design)
            // The revision a successful design request appends, as a new design shows it.
            let parts = try XCTUnwrap(preview.parts)
            let revision = ProjectRevision(model: spec, preview: preview, partLayout: PartLayout(nodes: parts), hiddenSolids: [],
                                           parts: parts, runtime: preview.runtime, validatedHash: preview.validatedHash,
                                           jointPositions: preview.assemblyPose?.positions)
            var project = RADProject(messages: [], revisions: [revision], selectedRevisionID: revision.id, projectID: UUID(),
                                     showGrid: true, projectionMode: .perspective, displayMode: .edges)
            project.thumbnailData = try await ProjectThumbnail.render(project: project, preview: preview)
            // Save New keeps only that revision, its parts organization and the view choices.
            var design = try SaveNewProject.snapshot(from: project)
            design.projectName = example.design
            try await store.writeCopy(design, to: output.appendingPathComponent("\(example.design).rad", isDirectory: true))
        }
    }

    /// Each example opens through the package reader, replays its program to
    /// the saved parts and receipt, and opens in a design window's state.
    @MainActor
    func testExampleDesignsOpenAndReplayTheirPrograms() async throws {
        guard ProcessInfo.processInfo.environment["RAD_RUN_ENGINE_TESTS"] == "1" else {
            throw XCTSkip("Set RAD_RUN_ENGINE_TESTS=1 to replay the example designs.")
        }
        let root = FileManager.default.temporaryDirectory.appendingPathComponent("RAD-examples-\(UUID().uuidString)")
        try FileManager.default.createDirectory(at: root, withIntermediateDirectories: false)
        defer { try? FileManager.default.removeItem(at: root) }
        var previews: [String: PreviewMesh] = [:]
        for example in Self.examples {
            ProjectPackage.clearSessionCache()
            let saved = try ProjectStore.read(from: Self.package(example))
            let revision = try XCTUnwrap(saved.revisions.first)
            let preview = try await EngineService().buildPreview(revision.model, python: EngineService.defaultPython, threads: 4,
                                                                 validatedHash: revision.validatedHash)
            XCTAssertEqual(preview.parts, revision.parts, example.design)
            XCTAssertEqual(preview.warnings, [], example.design)
            if preview.runtime == revision.runtime {
                XCTAssertEqual(preview.validatedHash, revision.validatedHash, "\(example.design) replays to its receipt")
            }
            XCTAssertEqual(preview.assemblyPose?.positions, revision.jointPositions, example.design)
            previews[example.design] = preview

            // A design window opens a copy; opening never needs the original program file.
            let copy = root.appendingPathComponent("\(example.design).rad", isDirectory: true)
            try FileManager.default.copyItem(at: Self.package(example), to: copy)
            ProjectPackage.clearSessionCache()
            let state = AppState(autosaveStore: ProjectAutosaveStore(rootDirectory: root), python: EngineService.defaultPython)
            defer { state.cancel() }
            state.loadProject(try ProjectStore.read(from: copy), from: copy)
            let deadline = ContinuousClock.now + .seconds(60)
            while state.mesh == nil || state.isWorking || state.isSolvingAssembly, ContinuousClock.now < deadline {
                try await Task.sleep(for: .milliseconds(25))
            }
            XCTAssertNil(state.error, example.design)
            XCTAssertEqual(state.mesh?.solids, preview.solids, example.design)
            XCTAssertEqual(state.partTree, revision.partLayout?.nodes, example.design)
            XCTAssertEqual(state.title, example.design)
        }

        let duck = try XCTUnwrap(previews["Rubber duck"])
        XCTAssertEqual(duck.solids, 6)
        XCTAssertTrue(duck.parts?.flatMap(\.flattened).filter { !$0.solidIndices.isEmpty }.allSatisfy { $0.color != nil } == true)
        let board = try XCTUnwrap(previews["HDMI board case"])
        let pcb = try XCTUnwrap(board.parts?.flatMap(\.flattened).first { $0.id == "pcb" })
        XCTAssertEqual(board.referenceSolids, pcb.allSolidIndices.sorted(), "The board and its components are reference parts")
        XCTAssertEqual(board.referenceSolids?.count, 7)
        XCTAssertEqual(board.printDown?.map(\.solid), [0, 1])
        let geneva = try XCTUnwrap(previews["Geneva drive"])
        XCTAssertEqual(geneva.assemblyMotion?.joints.map(\.id), ["crank", "star"])
        XCTAssertEqual(geneva.assemblyMotion?.couplings.map(\.id), ["geneva"])
    }
}
