import Foundation
import XCTest
@testable import RAD

final class AppStateSelectionContextTests: XCTestCase {
    @MainActor
    func testPartSelectionWhilePreviewLoadsIsCapturedWithoutStartingAnotherBuild() async throws {
        let fixture = try Fixture()
        defer { fixture.cleanUp() }
        let state = fixture.state()
        let revision = try installGeometry(in: state)
        state.mesh = nil
        state.isWorking = true
        state.selectPartNodes(["case"])
        state.prompt = "Change these selected parts once loading finishes"
        state.submit()
        let captured = try XCTUnwrap(state.queuedRequests.first?.selection)
        XCTAssertEqual(captured.revisionID, revision.id)
        XCTAssertEqual(captured.parts.map(\.partID), ["base", "lid"])
        XCTAssertEqual(captured.parts.map(\.path), [["Case", "Base"], ["Case", "Lid"]])
        XCTAssertTrue(captured.parts.allSatisfy { $0.anchor == nil })
        XCTAssertTrue(captured.points.isEmpty)
        XCTAssertTrue(state.messages.isEmpty)
        XCTAssertTrue(state.taskAudits.isEmpty)
        state.cancel()
        await state.prepareToQuit()
    }

    @MainActor
    func testImmediateRequestCapturesWholeFolderPartsBeforeLiveSelectionChanges() async throws {
        let fixture = try Fixture()
        defer { fixture.cleanUp() }
        let state = fixture.state()
        let revision = try installGeometry(in: state)
        state.selectPartNodes(["case"])
        state.prompt = "Round the selected enclosure parts"
        state.submit()
        let message = try XCTUnwrap(state.messages.first(where: { $0.role == "user" }))
        let captured = try XCTUnwrap(message.selection)
        XCTAssertEqual(captured.revisionID, revision.id)
        XCTAssertEqual(captured.parts.map(\.partID), ["base", "lid"])
        XCTAssertEqual(captured.parts.map(\.solidIndex), [0, 1])
        XCTAssertEqual(captured.parts.map(\.path), [["Case", "Base"], ["Case", "Lid"]])
        XCTAssertTrue(captured.points.isEmpty)
        XCTAssertEqual(captured.parts[0].anchor?.volume, 16)
        XCTAssertEqual(captured.parts[1].anchor?.bounds, [10, 0, 0, 14, 4, 1])

        state.clearSelections()
        state.selectPartNodes(["lid"])
        XCTAssertEqual(state.messages.first(where: { $0.id == message.id })?.selection, captured,
                       "A submitted request must retain its original selection rather than the live tree selection.")
        state.cancel() // Cancel before yielding: neither Codex nor a CAD worker starts.
        await state.prepareToQuit()
        let saved = try ProjectStore.read(from: XCTUnwrap(state.projectURL))
        XCTAssertEqual(saved.messages.first(where: { $0.id == message.id })?.selection, captured)
    }

    @MainActor
    func testQueuedSelectionSurvivesEditingReorderingArchiveReopenAndDispatch() async throws {
        let fixture = try Fixture()
        defer { fixture.cleanUp() }
        let state = fixture.state()
        let revision = try installGeometry(in: state)
        state.isWorking = true
        state.selectPartNodes(["case"])
        state.prompt = "Round these selected parts"
        state.submit()
        let first = try XCTUnwrap(state.queuedRequests.first)
        let firstSelection = try XCTUnwrap(first.selection)
        XCTAssertEqual(firstSelection.revisionID, revision.id)
        XCTAssertEqual(firstSelection.parts.map(\.partID), ["base", "lid"])

        state.clearSelections()
        state.selectPartNodes(["lid"])
        state.prompt = "Use the selected lid"
        state.submit()
        let second = try XCTUnwrap(state.queuedRequests.last)
        let secondSelection = try XCTUnwrap(second.selection)
        XCTAssertEqual(secondSelection.parts.map(\.partID), ["lid"])

        state.clearSelections()
        state.selectPartNodes(["lid"])
        state.prompt = "Keep this unsent draft"
        state.beginEditingQueuedRequest(id: first.id)
        state.prompt = "Round these selected parts with a 2 mm radius"
        XCTAssertTrue(state.commitEditingQueuedRequest())
        XCTAssertEqual(state.prompt, "Keep this unsent draft")
        XCTAssertEqual(state.queuedRequests.first?.selection, firstSelection,
                       "Editing text in the composer must not recapture a different live selection.")
        state.beginEditingQueuedRequest(id: second.id)
        state.prompt = "Discard this text edit"
        state.selectPartNodes(["base"])
        state.cancelEditingQueuedRequest()
        XCTAssertEqual(state.queuedRequests.last?.selection, secondSelection)
        state.moveQueuedRequest(id: second.id, beforeID: first.id)
        XCTAssertEqual(state.queuedRequests.map(\.id), [second.id, first.id])
        state.clearSelections()
        let expected = state.queuedRequests
        XCTAssertTrue(state.messages.isEmpty)
        XCTAssertTrue(state.taskAudits.isEmpty)

        state.cancel()
        await state.prepareToQuit()
        let url = try XCTUnwrap(state.projectURL)
        let saved = try ProjectStore.read(from: url)
        XCTAssertEqual(saved.queuedRequests, expected)
        let reopened = fixture.state("Reopened")
        reopened.loadProject(saved, from: url)
        reopened.cancel() // Stop revision replay before yielding to a worker.
        XCTAssertEqual(reopened.queuedRequests, expected)
        XCTAssertTrue(reopened.queuePaused)
        XCTAssertTrue(reopened.selectedPoints.isEmpty)
        XCTAssertTrue(reopened.selectedSolids.isEmpty)
        reopened.resumeQueue()
        let dispatched = try XCTUnwrap(reopened.messages.first(where: { $0.role == "user" }))
        XCTAssertEqual(dispatched.content, second.content)
        XCTAssertEqual(dispatched.selection, secondSelection,
                       "Dispatch must use the saved request snapshot even though no live topology is loaded.")
        XCTAssertEqual(reopened.queuedRequests.map(\.id), [first.id])
        XCTAssertEqual(reopened.queuedRequests.first?.selection, firstSelection)
        reopened.cancel()
        await reopened.prepareToQuit()
    }

    @MainActor
    func testSelectionSnapshotsRemainIndependentAcrossOpenProjects() async throws {
        let fixture = try Fixture()
        defer { fixture.cleanUp() }
        let first = fixture.state("First"), second = fixture.state("Second")
        let firstRevision = try installGeometry(in: first)
        let secondRevision = try installGeometry(in: second)
        first.isWorking = true; second.isWorking = true
        first.selectPartNodes(["base"])
        second.selectPartNodes(["lid"])
        first.prompt = "Change the base in the first design"; first.submit()
        second.prompt = "Change the lid in the second design"; second.submit()
        let firstSelection = try XCTUnwrap(first.queuedRequests.first?.selection)
        let secondSelection = try XCTUnwrap(second.queuedRequests.first?.selection)
        XCTAssertEqual(firstSelection.revisionID, firstRevision.id)
        XCTAssertEqual(secondSelection.revisionID, secondRevision.id)
        XCTAssertNotEqual(firstSelection.revisionID, secondSelection.revisionID)
        XCTAssertEqual(firstSelection.parts.map(\.partID), ["base"])
        XCTAssertTrue(firstSelection.points.isEmpty)
        XCTAssertEqual(secondSelection.parts.map(\.partID), ["lid"])
        XCTAssertTrue(secondSelection.points.isEmpty)
        first.clearSelections()
        first.newProject()
        XCTAssertTrue(first.queuedRequests.isEmpty)
        XCTAssertEqual(second.queuedRequests.first?.selection, secondSelection)
        second.cancel()
        await first.prepareToQuit(); await second.prepareToQuit()
        let secondSaved = try ProjectStore.read(from: XCTUnwrap(second.projectURL))
        XCTAssertEqual(secondSaved.queuedRequests?.first?.selection, secondSelection)
    }

    @MainActor
    func testRequestCapturesSelectedPointsInOrderWithoutSelectingTheirParts() async throws {
        let fixture = try Fixture()
        defer { fixture.cleanUp() }
        let state = fixture.state()
        let revision = try installGeometry(in: state)
        let lid = CADSurfacePoint(solidIndex: 1, point: [12, 1, 0], normal: [0, 0, 1])
        let base = CADSurfacePoint(solidIndex: 0, point: [1, 1, 0], normal: [0, 0, 1])
        state.selectPoints([lid, base, lid, CADSurfacePoint(solidIndex: 7, point: [0, 0, 0], normal: [0, 0, 1])])
        XCTAssertEqual(state.selectedPoints, [lid, base], "Duplicates and points on missing parts are dropped.")
        XCTAssertEqual(state.featureSelectedSolids, [0, 1])
        XCTAssertTrue(state.selectedSolids.isEmpty)
        XCTAssertEqual(state.selectedPartSolids, [0, 1], "Hiding and coloring include the points' parts.")
        XCTAssertFalse(state.canExportSelectedParts, "Point picks never add their parts to part exports.")
        state.isWorking = true
        state.prompt = "Drill a hole at each selected point"
        state.submit()
        let captured = try XCTUnwrap(state.queuedRequests.first?.selection)
        XCTAssertEqual(captured.revisionID, revision.id)
        XCTAssertTrue(captured.parts.isEmpty)
        XCTAssertEqual(captured.points.map(\.partID), ["lid", "base"])
        XCTAssertEqual(captured.points.map(\.path), [["Case", "Lid"], ["Case", "Base"]])
        XCTAssertEqual(captured.points.map(\.point), [[12, 1, 0], [1, 1, 0]])
        XCTAssertEqual(captured.points.map(\.normal), [[0, 0, 1], [0, 0, 1]])
        XCTAssertNoThrow(try captured.validate())

        state.togglePartVisibility([1])
        XCTAssertEqual(state.selectedPoints, [base], "Hiding a part prunes its points.")
        XCTAssertEqual(state.queuedRequests.first?.selection, captured)
        state.cancel()
        await state.prepareToQuit()
        let saved = try ProjectStore.read(from: XCTUnwrap(state.projectURL))
        XCTAssertEqual(saved.queuedRequests?.first?.selection, captured)
    }

    @MainActor
    func testNoSelectionDoesNotInventRequestContext() async throws {
        let fixture = try Fixture()
        defer { fixture.cleanUp() }
        let state = fixture.state()
        _ = try installGeometry(in: state)
        state.isWorking = true
        state.prompt = "Add a new unrelated part"
        state.submit()
        XCTAssertNil(state.queuedRequests.first?.selection)
        state.cancel()
        await state.prepareToQuit()
    }

    @MainActor
    private func installGeometry(in state: AppState) throws -> ProjectRevision {
        let parts = [PartNode(id: "case", name: "Case", solidIndices: [], children: [
            PartNode(id: "base", name: "Base", solidIndices: [0]),
            PartNode(id: "lid", name: "Lid", solidIndices: [1])
        ])]
        let vertices = [[0.0, 0, 0], [4, 0, 0], [0, 4, 0], [10, 0, 0], [14, 0, 0], [10, 4, 0]]
        let stats = [CADPart(solidIndex: 0, area: 48, volume: 16, bounds: [0, 0, 0, 4, 4, 1], centroid: [2, 2, 0.5], triangles: 1, components: 1),
                     CADPart(solidIndex: 1, area: 48, volume: 16, bounds: [10, 0, 0, 14, 4, 1], centroid: [12, 2, 0.5], triangles: 1, components: 1)]
        let preview = PreviewMesh(vertices: vertices, triangles: [[0, 1, 2], [3, 4, 5]],
            edges: [[0, 0, 0, 4, 0, 0], [10, 0, 0, 14, 0, 0]], bounds: [0, 0, 0, 14, 4, 1],
            volume: 32, area: 96, solids: 2, buildSeconds: 0, threads: 1,
            buildID: UUID().uuidString, triangleSolids: [0, 1], edgeSolids: [0, 1], parts: parts, partStats: stats)
        try ProjectStore.validatePreview(preview)
        let revision = ProjectRevision(model: .bracket, preview: preview, parts: parts)
        state.revisions = [revision]
        state.selectedRevisionID = revision.id
        state.mesh = preview
        return revision
    }

    @MainActor
    private struct Fixture {
        let directory: URL
        private let originalCodex: Any?

        init() throws {
            directory = FileManager.default.temporaryDirectory.appendingPathComponent("RAD-selection-state-\(UUID())", isDirectory: true)
            try FileManager.default.createDirectory(at: directory, withIntermediateDirectories: false)
            originalCodex = UserDefaults.standard.object(forKey: "codexPath")
            UserDefaults.standard.set(directory.appendingPathComponent("missing-codex").path, forKey: "codexPath")
        }

        func state(_ name: String = "Original") -> AppState {
            let state = AppState(autosaveStore: ProjectAutosaveStore(rootDirectory: directory.appendingPathComponent(name)))
            state.newProject()
            return state
        }

        func cleanUp() {
            if let originalCodex { UserDefaults.standard.set(originalCodex, forKey: "codexPath") }
            else { UserDefaults.standard.removeObject(forKey: "codexPath") }
            try? FileManager.default.removeItem(at: directory)
        }
    }
}
