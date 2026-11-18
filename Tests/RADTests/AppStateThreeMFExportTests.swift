import Foundation
import XCTest
@testable import RAD

final class AppStateThreeMFExportTests: XCTestCase {
    @MainActor
    func testApprovedIndividualExportReplacesTheCompleteExistingFolderWithPlacedParts() async throws {
        try requireRuntime()
        let fixture = try await Fixture()
        defer { fixture.cleanUp() }
        let parts = try XCTUnwrap(fixture.preview.parts)
        var layout = PartLayout(nodes: parts)
        try layout.rename("first", to: "Base")
        let revision = ProjectRevision(model: fixture.model, preview: fixture.preview, partLayout: layout, parts: parts,
            partColors: ["first": .init(red: 1, green: 0.5, blue: 0)])
        fixture.state.revisions = [revision]
        fixture.state.selectedRevisionID = revision.id
        XCTAssertTrue(fixture.state.loadStoredRevision())
        let native = URL(fileURLWithPath: try XCTUnwrap(fixture.preview.artifactPath))
        let nativeBefore = try Data(contentsOf: native)
        let folder = fixture.directory.appendingPathComponent("Existing individual export", isDirectory: true)
        try FileManager.default.createDirectory(at: folder, withIntermediateDirectories: false)
        try Data("previous 3MF".utf8).write(to: folder.appendingPathComponent("Base.3mf"))
        try Data("obsolete part".utf8).write(to: folder.appendingPathComponent("Obsolete.3mf"))
        fixture.state.export3MF(to: folder, for3DPrinting: true, exportIndividualParts: true, replaceExisting: true)
        try await finish(fixture.state, expecting: folder)
        XCTAssertEqual(Set(try FileManager.default.contentsOfDirectory(atPath: folder.path)), ["Base.3mf", "Second.3mf"])
        let base = try EngineThreeMFExportTests.summary(Data(contentsOf: folder.appendingPathComponent("Base.3mf")))
        let second = try EngineThreeMFExportTests.summary(Data(contentsOf: folder.appendingPathComponent("Second.3mf")))
        XCTAssertEqual(base.colors, ["Base": "#FF8000"])
        XCTAssertEqual(second.colors, ["Second": "#21C7E5"])
        XCTAssertEqual(base.volume, 24, accuracy: 24 * 0.02)
        XCTAssertEqual(second.volume, 30, accuracy: 30 * 0.02)
        XCTAssertEqual(base.bounds[2], 0, accuracy: 0.05)
        XCTAssertEqual(second.bounds[2], 0, accuracy: 0.05)
        let a = base.bounds, b = second.bounds
        XCTAssertGreaterThan(max(b[0] - a[3], a[0] - b[3], b[1] - a[4], a[1] - b[4]), 0, "Placed parts must not overlap on the bed.")
        XCTAssertEqual(try Data(contentsOf: native), nativeBefore)
        XCTAssertEqual(fixture.state.revisions.map(\.id), [revision.id])
    }

    @MainActor
    func testFullAndSubsetExportsUseTheBuiltCoordinates() async throws {
        try requireRuntime()
        let fixture = try await Fixture()
        defer { fixture.cleanUp() }
        let parts = try XCTUnwrap(fixture.preview.parts)
        let revision = ProjectRevision(model: fixture.model, preview: fixture.preview, partLayout: PartLayout(nodes: parts), parts: parts)
        fixture.state.revisions = [revision]
        fixture.state.selectedRevisionID = revision.id
        XCTAssertTrue(fixture.state.loadStoredRevision())
        XCTAssertTrue(fixture.state.canExport)
        let full = fixture.directory.appendingPathComponent("design.3mf")
        try await export(fixture.state, to: full)
        let whole = try EngineThreeMFExportTests.summary(Data(contentsOf: full))
        XCTAssertEqual(whole.volume, 54, accuracy: 54 * 0.02)
        XCTAssertEqual(whole.colors, ["First": "#BE413D", "Second": "#21C7E5"])
        for (actual, expected) in zip(whole.bounds, [0.0, 0, 0, 25, 9, 12]) { XCTAssertEqual(actual, expected, accuracy: 0.05) }
        XCTAssertEqual(fixture.state.mesh?.artifactPath, fixture.preview.artifactPath,
                       "A retained host-owned preview can be exported by another engine without replaying its program.")
        let second = try XCTUnwrap(fixture.state.partTree.flatMap(\.flattened).first { $0.id == "second" })
        fixture.state.setPartColor(.init(red: 0, green: 1, blue: 0.5), for: "second")
        let subset = fixture.directory.appendingPathComponent("selected.3mf")
        try await export(fixture.state, to: subset, solidIndices: Set(second.solidIndices))
        try assertPart(at: [22.5, 8, 10.5], volume: 30, bounds: [20, 7, 9, 25, 9, 12],
                       color: ["Second": "#00FF80"], in: Data(contentsOf: subset))
        XCTAssertEqual(fixture.state.revisions.map(\.id), [revision.id])
    }

    @MainActor
    func testSubsetReplayRemapsTheCapturedPartByStableIDAndKeepsItsNameAndColor() async throws {
        try requireRuntime()
        let fixture = try await Fixture()
        defer { fixture.cleanUp() }
        let generated = try XCTUnwrap(fixture.preview.parts)
        func permuted(_ nodes: [PartNode]) -> [PartNode] {
            nodes.map { node in
                var value = node
                value.solidIndices = node.solidIndices.map { fixture.preview.solids - 1 - $0 }
                if let children = node.children { value.children = permuted(children) }
                return value
            }
        }
        let savedParts = permuted(generated)
        let oldSecond = try XCTUnwrap(savedParts.flatMap(\.flattened).first { $0.id == "second" })
        let freshSecond = try XCTUnwrap(generated.flatMap(\.flattened).first { $0.id == "second" })
        XCTAssertNotEqual(oldSecond.solidIndices, freshSecond.solidIndices)
        var layout = PartLayout(nodes: savedParts)
        try layout.rename("second", to: "Renamed lid")
        // A saved preview without session geometry forces a replay.
        let cached = PreviewMesh(vertices: [], triangles: [], edges: [], bounds: fixture.preview.bounds,
            volume: fixture.preview.volume, area: fixture.preview.area, solids: fixture.preview.solids,
            buildSeconds: 0, threads: 2, buildID: UUID().uuidString, parts: savedParts)
        let revision = ProjectRevision(model: fixture.model, preview: cached, partLayout: layout, parts: savedParts,
            partColors: ["second": .init(red: 0.5, green: 0, blue: 1)])
        fixture.state.revisions = [revision]
        fixture.state.selectedRevisionID = revision.id
        XCTAssertTrue(fixture.state.loadStoredRevision())
        let folder = fixture.directory.appendingPathComponent("Selected parts", isDirectory: true)
        fixture.state.export3MF(to: folder, solidIndices: Set(oldSecond.solidIndices), exportIndividualParts: true)
        try await finish(fixture.state, expecting: folder)
        XCTAssertEqual(try FileManager.default.contentsOfDirectory(atPath: folder.path), ["Renamed lid.3mf"])
        try assertPart(at: [22.5, 8, 10.5], volume: 30, bounds: [20, 7, 9, 25, 9, 12],
                       color: ["Renamed lid": "#8000FF"],
                       in: Data(contentsOf: folder.appendingPathComponent("Renamed lid.3mf")))
        let remapped = try XCTUnwrap(fixture.state.partTree.flatMap(\.flattened).first { $0.id == "second" })
        XCTAssertEqual(remapped.solidIndices, freshSecond.solidIndices)
        XCTAssertEqual(fixture.state.revisions.map(\.id), [revision.id])
    }

    @MainActor
    func testAgentExportsColdCurrentOwnershipAndIndependentCandidateSubset() async throws {
        try requireRuntime()
        let fixture = try await Fixture()
        defer { fixture.cleanUp() }
        let generated = try XCTUnwrap(fixture.preview.parts)
        func reversedOwnership(_ nodes: [PartNode]) -> [PartNode] {
            nodes.map { node in
                var value = node
                value.solidIndices = node.solidIndices.map { fixture.preview.solids - 1 - $0 }
                if let children = node.children { value.children = reversedOwnership(children) }
                return value
            }
        }
        let tools = try CADAgentTools(directory: fixture.directory, python: EngineService.defaultPython, threads: 2,
            current: fixture.model, currentPreview: nil, engine: fixture.engine,
            partLayout: PartLayout(nodes: reversedOwnership(generated)),
            partColors: ["first": .init(red: 1, green: 0.5, blue: 0)])
        defer { tools.cancel() }
        func exported(_ artifactID: String, path: String, indices: [Int]? = nil) async throws -> Data {
            var arguments: [String: Any] = ["artifactID": artifactID, "path": path]
            if let indices { arguments["solidIndices"] = indices }
            let info = try await tools.handle(name: "rad_export_3mf", arguments: arguments).metadata
            return try Data(contentsOf: URL(fileURLWithPath: try XCTUnwrap(info["path"] as? String)))
        }
        // Without a current preview, export reconciles old saved ownership
        // against the recipe's freshly generated order.
        let oldFirst = try XCTUnwrap(reversedOwnership(generated).flatMap(\.flattened).first { $0.id == "first" })
        try assertPart(at: [1, 1.5, 2], volume: 24, bounds: [0, 0, 0, 2, 3, 4],
                       color: ["First": "#FF8000"],
                       in: await exported("current", path: "exports/agent-cold-subset.3mf", indices: oldFirst.solidIndices))
        let whole = try EngineThreeMFExportTests.summary(await exported("current", path: "exports/agent-current.3mf"))
        XCTAssertEqual(whole.volume, 54, accuracy: 54 * 0.02)

        let candidate = """
        result = Assembly("Parts", id="parts")
        result.add(box((5, 2, 3), center=(22.5, 8, 10.5)), id="second", name="Second")
        result.add(box((2, 3, 4), center=(1, 1.5, 2)), id="first", name="First")
        """
        try Data(candidate.utf8).write(to: fixture.directory.appendingPathComponent("candidate.py"))
        let built = try await tools.handle(name: "rad_build", arguments: ["programPath": "candidate.py"]).metadata
        let artifactID = try XCTUnwrap(built["artifactID"] as? String)
        _ = try await tools.handle(name: "rad_set_part_colors", arguments: ["artifactID": artifactID,
            "changes": [["partID": "first", "color": ["red": 162.0 / 255, "green": 71.0 / 255, "blue": 205.0 / 255]]]])
        try assertPart(at: [1, 1.5, 2], volume: 24, bounds: [0, 0, 0, 2, 3, 4],
                       color: ["First": "#A247CD"], in: await exported(artifactID, path: "exports/agent-candidate.3mf", indices: [1]))
    }

    @MainActor
    private func export(_ state: AppState, to url: URL, solidIndices: Set<Int>? = nil) async throws {
        state.export3MF(to: url, solidIndices: solidIndices)
        XCTAssertTrue(state.isWorking)
        try await finish(state, expecting: url)
    }

    @MainActor
    private func finish(_ state: AppState, expecting url: URL) async throws {
        let deadline = ContinuousClock.now + .seconds(60)
        while state.isWorking, ContinuousClock.now < deadline { try await Task.sleep(for: .milliseconds(10)) }
        XCTAssertFalse(state.isWorking, "The 3MF export must finish.")
        if state.isWorking { state.cancel() }
        XCTAssertNil(state.error)
        guard FileManager.default.fileExists(atPath: url.path) else {
            throw RADError.message("No 3MF was exported: \(state.error ?? state.activity)")
        }
    }

    private func assertPart(at center: [Double], volume: Double, bounds: [Double], color: [String: String], in data: Data,
                            file: StaticString = #filePath, line: UInt = #line) throws {
        let part = try EngineThreeMFExportTests.summary(data)
        for (actual, expected) in zip(part.centroid, center) { XCTAssertEqual(actual, expected, accuracy: 0.05, file: file, line: line) }
        for (actual, expected) in zip(part.bounds, bounds) { XCTAssertEqual(actual, expected, accuracy: 0.05, file: file, line: line) }
        XCTAssertEqual(part.volume, volume, accuracy: volume * 0.02, file: file, line: line)
        XCTAssertEqual(part.colors, color, file: file, line: line)
    }

    private func requireRuntime() throws {
        guard ProcessInfo.processInfo.environment["RAD_RUN_ENGINE_TESTS"] == "1" else {
            throw XCTSkip("Set RAD_RUN_ENGINE_TESTS=1 for AppState 3MF export and replay regression tests.")
        }
    }

    @MainActor
    private struct Fixture {
        let directory: URL
        let model: ModelSpec
        let preview: PreviewMesh
        let state: AppState
        let engine: EngineService
        let oldThreads: Any?

        init() async throws {
            directory = FileManager.default.temporaryDirectory.appendingPathComponent("RAD-AppState-3MF-\(UUID().uuidString)")
            try FileManager.default.createDirectory(at: directory, withIntermediateDirectories: true)
            model = ModelSpec(title: "3MF export fixture", summary: "Two boxes exported as separate shells.", units: "mm", script: """
            result = Assembly("Parts", id="parts", color="#BE413D")
            result.add(box((2, 3, 4), center=(1, 1.5, 2)), id="first", name="First")
            result.add(box((5, 2, 3), center=(2.5, 1, 1.5)), id="second", name="Second", color="#21C7E5", at=Placement(translate=(20, 7, 9)))
            """)
            engine = EngineService()
            preview = try await engine.buildPreview(model, python: EngineService.defaultPython, threads: 2)
            state = AppState(autosaveStore: ProjectAutosaveStore(rootDirectory: directory.appendingPathComponent("autosaves")))
            oldThreads = UserDefaults.standard.object(forKey: "workerThreads")
            UserDefaults.standard.set(2, forKey: "workerThreads")
        }

        func cleanUp() {
            state.cancel()
            engine.cancel()
            if let oldThreads { UserDefaults.standard.set(oldThreads, forKey: "workerThreads") }
            else { UserDefaults.standard.removeObject(forKey: "workerThreads") }
            try? FileManager.default.removeItem(at: directory)
        }
    }
}
