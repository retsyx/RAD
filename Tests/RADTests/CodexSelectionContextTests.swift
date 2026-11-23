import Foundation
import XCTest
@testable import RAD

final class CodexSelectionContextTests: XCTestCase {
    @MainActor
    func testRequestReferencesImmutableSelectionAndOlderOriginRecipeWithoutEmbeddingContents() async throws {
        let selection = target()
        let origin = originModel()
        let fixture = try FakeCodex(responseJSON: readyResponse(), delay: 0.02)
        defer { fixture.remove() }
        _ = try await CodexService().generate(prompt: "Round the selected point", current: .bracket, history: [],
            executable: fixture.executable.path, model: "", onProgress: { _ in },
            selection: selection, selectionModel: origin)
        let request = try text(try fixture.turnRequest())
        let path = try reference("REQUEST SELECTION FILE:", in: request)
        let file = try JSONDecoder().decode(CodexSelectionFile.self, from: fixture.capturedData(at: path))
        XCTAssertEqual(file.selection, selection)
        XCTAssertEqual(file.originRecipeStatus, "available")
        XCTAssertFalse(file.originMatchesCurrent)
        XCTAssertTrue(request.contains("1 selected parts and 1 selected points"))
        XCTAssertTrue(request.contains("never reuse them blindly"))
        XCTAssertFalse(request.contains("SELECTION_NAME_STAYS_LOCAL"))
        XCTAssertFalse(request.contains("ORIGIN_PROGRAM_STAYS_LOCAL"))
        XCTAssertFalse(request.contains("ORIGINAL_MESH_STAYS_LOCAL"))
        let originReference = try XCTUnwrap(file.originModel)
        let stagedOrigin = try JSONDecoder().decode(ModelSpec.self, from: fixture.capturedData(at: originReference.readPath))
        XCTAssertEqual(stagedOrigin.script, origin.script)
        XCTAssertNil(stagedOrigin.inputAssets)
        let manifestPath = try XCTUnwrap(file.originGeometryInputsManifestPath)
        let inputs = try XCTUnwrap(JSONSerialization.jsonObject(with: fixture.capturedData(at: manifestPath)) as? [[String: String]])
        XCTAssertEqual(inputs.count, 1)
        for input in inputs {
            let data = try fixture.capturedData(at: XCTUnwrap(input["readPath"]))
            XCTAssertEqual(input["programBinding"], "imported_assets[assetID]")
            XCTAssertEqual(data, origin.inputAssets?.first?.data)
            XCTAssertFalse(request.contains(data.base64EncodedString()))
        }
        try fixture.assertJobWasCleanedUp()
    }

    func testHistoricalSelectionsSurviveArchiveAndShareOriginRecipeAndExactInputs() throws {
        let root = try temporary()
        defer { try? FileManager.default.removeItem(at: root) }
        let origin = originModel()
        let revision = ProjectRevision(model: origin)
        let selection = target(revisionID: revision.id)
        let messages = [ChatMessage(role: "user", content: "First target", selection: selection),
                        ChatMessage(role: "user", content: "Use that target again", selection: selection)]
        let project = RADProject(messages: messages, revisions: [revision], selectedRevisionID: revision.id)
        let reopened = try ProjectStore.decode(ProjectStore.encode(project))
        let files = try CodexFileContext.prepare(current: .bracket, history: reopened.messages, attachments: [], directory: root,
            selectionModelsByRevision: Dictionary(uniqueKeysWithValues: reopened.revisions.map { ($0.id, $0.model) }))
        let manifest = try JSONDecoder().decode([CodexSelectionHistoryReference].self,
            from: Data(contentsOf: URL(fileURLWithPath: XCTUnwrap(files.selectionManifestPath))))
        XCTAssertEqual(manifest.map(\.messageID), messages.map(\.id))
        XCTAssertEqual(manifest.map(\.revisionID), [revision.id, revision.id])
        let staged = try manifest.map {
            try JSONDecoder().decode(CodexSelectionFile.self, from: Data(contentsOf: URL(fileURLWithPath: $0.selectionPath)))
        }
        XCTAssertEqual(staged.map(\.selection), [selection, selection])
        XCTAssertEqual(Set(staged.compactMap { $0.originModel?.readPath }).count, 1,
                       "One revision's complete recipe is staged once even when several messages select it.")
        let assetsFolder = root.appendingPathComponent("context/selection-assets")
        XCTAssertEqual(try FileManager.default.contentsOfDirectory(atPath: assetsFolder.path).count, 1)
        let recipeURL = URL(fileURLWithPath: try XCTUnwrap(staged[0].originModel?.readPath))
        XCTAssertEqual(try JSONDecoder().decode(ModelSpec.self, from: Data(contentsOf: recipeURL)).script, origin.script)
        XCTAssertEqual(try permissions(recipeURL), 0o600)
        XCTAssertEqual(try permissions(recipeURL.deletingLastPathComponent()), 0o700)
    }

    func testMissingHistoricalOriginRemainsExplicitlyUnavailableWithItsPointsIntact() throws {
        let root = try temporary()
        defer { try? FileManager.default.removeItem(at: root) }
        let selection = target()
        let message = ChatMessage(role: "user", content: "That selected point", selection: selection)
        let files = try CodexFileContext.prepare(current: .bracket, history: [message], attachments: [], directory: root)
        let manifest = try JSONDecoder().decode([CodexSelectionHistoryReference].self,
            from: Data(contentsOf: URL(fileURLWithPath: XCTUnwrap(files.selectionManifestPath))))
        let file = try JSONDecoder().decode(CodexSelectionFile.self,
            from: Data(contentsOf: URL(fileURLWithPath: XCTUnwrap(manifest.first?.selectionPath))))
        XCTAssertEqual(file.selection, selection)
        XCTAssertNil(file.originModel)
        XCTAssertEqual(file.originRecipeStatus, "unavailable")
        XCTAssertNil(file.originGeometryInputsManifestPath)
        XCTAssertFalse(file.originMatchesCurrent)
    }

    @MainActor
    func testSameTurnSteeringUsesUniqueSelectionFilesAndPreservesEarlierSnapshot() async throws {
        let fixture = try FakeCodex(responseJSON: readyResponse(), delay: 2)
        defer { fixture.remove() }
        let service = CodexService()
        defer { service.cancel() }
        let generation = Task { @MainActor in
            try await service.generate(prompt: "Refine the case", current: nil, history: [],
                executable: fixture.executable.path, model: "", onProgress: { _ in })
        }
        try await waitUntil { service.canSteer }
        let first = target()
        let second = ProjectSelectionContext(revisionID: first.revisionID, parts: [], points: [],
            edges: [ProjectSelectedEdge(partID: "lid", name: "SELECTION_NAME_STAYS_LOCAL", path: ["Case", "Lid"],
                solidIndex: 0, closed: false, length: 4, landmarks: [[0, 0, 0], [2, 0, 0], [4, 0, 0]])])
        let turn = try XCTUnwrap(service.currentTurnID)
        let firstAccepted = try await service.steer(prompt: "This point", selection: first, selectionModel: originModel())
        let secondAccepted = try await service.steer(prompt: "Also this edge", selection: second, selectionModel: .bracket)
        XCTAssertEqual(firstAccepted, turn)
        XCTAssertEqual(secondAccepted, turn)
        _ = try await generation.value
        let calls = try fixture.rpcRequests().filter { $0["method"] as? String == "turn/steer" }
        XCTAssertEqual(calls.count, 2)
        var paths: [String] = []
        for (call, expected) in zip(calls, [first, second]) {
            let params = try XCTUnwrap(call["params"] as? [String: Any])
            XCTAssertEqual(params["expectedTurnId"] as? String, turn)
            let request = try text(params)
            let path = try reference("REQUEST SELECTION FILE:", in: request)
            paths.append(path)
            let file = try JSONDecoder().decode(CodexSelectionFile.self, from: fixture.capturedData(at: path))
            XCTAssertEqual(file.selection, expected)
            XCTAssertFalse(file.originMatchesCurrent)
            XCTAssertFalse(request.contains("SELECTION_NAME_STAYS_LOCAL"))
            XCTAssertFalse(request.contains("ORIGIN_PROGRAM_STAYS_LOCAL"))
        }
        XCTAssertEqual(Set(paths).count, 2)
        XCTAssertNotEqual(URL(fileURLWithPath: paths[0]).deletingLastPathComponent(),
                          URL(fileURLWithPath: paths[1]).deletingLastPathComponent())
        XCTAssertEqual(try fixture.rpcRequests().filter { $0["method"] as? String == "turn/start" }.count, 1)
        try fixture.assertJobWasCleanedUp()
    }

    @MainActor
    func testEmptySelectionDoesNotCreateASelectionReferenceOrChangeTurnInputKinds() async throws {
        let fixture = try FakeCodex(responseJSON: readyResponse(), delay: 0.02)
        defer { fixture.remove() }
        let empty = ProjectSelectionContext(revisionID: UUID(), parts: [], points: [])
        _ = try await CodexService().generate(prompt: "Create a block", current: nil, history: [],
            executable: fixture.executable.path, model: "", onProgress: { _ in }, selection: empty, selectionModel: .bracket)
        let params = try fixture.turnRequest()
        let request = try text(params)
        XCTAssertFalse(request.contains("REQUEST SELECTION FILE:\n"))
        XCTAssertFalse(request.contains("SAVED SELECTION REFERENCE MANIFEST FILE:\n"))
        let inputs = try XCTUnwrap(params["input"] as? [[String: Any]])
        XCTAssertEqual(inputs.compactMap { $0["type"] as? String }, ["text"])
    }

    private func target(revisionID: UUID = UUID()) -> ProjectSelectionContext {
        ProjectSelectionContext(revisionID: revisionID, capturedAt: Date(timeIntervalSinceReferenceDate: 123),
            parts: [ProjectSelectedPart(partID: "case-lid", name: "SELECTION_NAME_STAYS_LOCAL", path: ["Case", "Lid"], solidIndex: 0)],
            points: [ProjectSelectedPoint(partID: "case-lid", name: "SELECTION_NAME_STAYS_LOCAL", path: ["Case", "Lid"],
                solidIndex: 0, point: [1, 2, 3], normal: [0, 0.6, 0.8])])
    }

    private func originModel() -> ModelSpec {
        ModelSpec(title: "Origin case", summary: "Selection origin", units: "mm",
            script: "# ORIGIN_PROGRAM_STAYS_LOCAL\nresult = imported_assets['reference-input']\n",
            inputAssets: [CADInputAsset(id: "reference-input", name: "reference.stl", format: "stl", data: Data("ORIGINAL_MESH_STAYS_LOCAL".utf8))])
    }

    private func text(_ params: [String: Any]) throws -> String {
        let inputs = try XCTUnwrap(params["input"] as? [[String: Any]])
        return try XCTUnwrap(inputs.first { $0["type"] as? String == "text" }?["text"] as? String)
    }

    private func reference(_ marker: String, in request: String) throws -> String {
        let range = try XCTUnwrap(request.range(of: marker + "\n"))
        let line = try XCTUnwrap(request[range.upperBound...].split(separator: "\n", maxSplits: 1).first)
        return try JSONDecoder().decode(String.self, from: Data(line.utf8))
    }

    private func readyResponse() throws -> String {
        let model = ModelSpec.bracket
        let response: [String: Any] = ["status": "ready", "message": "Design ready.", "assetIDs": [], "model": [
            "title": model.title, "summary": model.summary, "units": model.units, "script": model.script]]
        return String(decoding: try JSONSerialization.data(withJSONObject: response), as: UTF8.self)
    }

    private func temporary() throws -> URL {
        let root = FileManager.default.temporaryDirectory.appendingPathComponent("RAD-selection-context-\(UUID())", isDirectory: true)
        try FileManager.default.createDirectory(at: root, withIntermediateDirectories: false)
        return root
    }

    private func permissions(_ url: URL) throws -> Int {
        let attributes = try FileManager.default.attributesOfItem(atPath: url.path)
        return try XCTUnwrap(attributes[.posixPermissions] as? NSNumber).intValue & 0o777
    }

    @MainActor
    private func waitUntil(_ predicate: () -> Bool) async throws {
        let deadline = ContinuousClock.now + .seconds(5)
        while !predicate(), ContinuousClock.now < deadline { try await Task.sleep(for: .milliseconds(10)) }
        XCTAssertTrue(predicate())
    }
}
