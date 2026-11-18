import Foundation
import XCTest
@testable import RAD

final class AppStatePointSelectionTests: XCTestCase {
    @MainActor
    func testPartRangeTreatsPointOnlySelectionAsUnselectedAndPreservesPoints() throws {
        let state = try fixture()
        defer { state.clearSelections() }
        let points = [point(1, 0)]
        state.selectPoints(points)
        state.selectPartNodes(["base"])
        state.togglePartNode("board", extendingRange: true)
        XCTAssertEqual(state.selectedSolids, [0, 1, 2], "A point-selected lid must not stop the range before base.")
        XCTAssertEqual(state.selectedPoints, points)

        state.togglePartNode("lid")
        state.togglePartNode("base", extendingRange: true)
        XCTAssertEqual(state.selectedSolids, [2], "The point-selected lid is an unselected boundary.")
        XCTAssertEqual(state.selectedPoints, points)
    }

    @MainActor
    func testTreeAndWholePartSelectionPreserveOrderedPointsAndCanonicalTreeSelection() throws {
        let state = try fixture()
        defer { state.clearSelections() }
        let points = [point(2, 0), point(0, 0), point(1, 0)]
        state.selectPoints(points)
        state.selectPartNodes(["case"])
        XCTAssertEqual(state.selectedPoints, points)
        XCTAssertEqual(state.selectedSolids, [0, 1])
        XCTAssertEqual(state.selectedPartNodeIDs, ["case"])

        state.selectParts([2])
        XCTAssertEqual(state.selectedPoints, points)
        XCTAssertEqual(state.selectedSolids, [2])
        XCTAssertEqual(state.selectedPartNodeIDs, ["board"])
        XCTAssertEqual(state.measurementReport?.solidIndices, [2])
        XCTAssertEqual(try values(state, "parts:count"), [1])

        state.selectParts([1], extending: true)
        XCTAssertEqual(state.selectedSolids, [1, 2])
        XCTAssertEqual(state.selectedPartNodeIDs, ["lid", "board"])
        XCTAssertEqual(state.selectedPoints, points)
        state.selectParts([2], extending: true)
        XCTAssertEqual(state.selectedSolids, [1])
        XCTAssertEqual(state.selectedPartNodeIDs, ["lid"])
        XCTAssertEqual(state.selectedPoints, points)

        state.selectPartNodes(["case"], extending: true)
        XCTAssertEqual(state.selectedSolids, [0, 1])
        XCTAssertEqual(state.selectedPartNodeIDs, ["case"])
        XCTAssertEqual(state.selectedPoints, points)
        state.selectPartNodes(["case"], extending: true)
        XCTAssertTrue(state.selectedSolids.isEmpty)
        XCTAssertTrue(state.selectedPartNodeIDs.isEmpty)
        XCTAssertEqual(state.selectedPoints, points, "Deselecting a folder must retain its selected points.")

        state.selectParts([])
        XCTAssertEqual(state.selectedPoints, points, "Clearing whole-part selection must retain point selection.")
        XCTAssertEqual(state.measurementReport?.points, points)
        XCTAssertEqual(state.measurementReport?.solidIndices, [])
        XCTAssertEqual(try values(state, "triangle:sides").count, 3, "Three points measure their triangle.")
    }

    @MainActor
    func testPointTogglesPreserveWholePartsAndSelectedFolderIdentity() throws {
        let state = try fixture()
        defer { state.clearSelections() }
        state.selectPartNodes(["case"])
        let face = point(2, 0), vertex = point(0, 0), edge = point(1, 0)
        let ordered = [face, vertex, edge]
        state.selectPoints(ordered + [face])
        XCTAssertEqual(state.selectedPoints, ordered)
        XCTAssertEqual(state.selectedSolids, [0, 1])
        XCTAssertEqual(state.selectedPartNodeIDs, ["case"])
        XCTAssertEqual(state.selectedPartSolids, [0, 1, 2])
        XCTAssertEqual(state.measurementReport?.solidIndices, [0, 1])
        XCTAssertEqual(try values(state, "parts:count"), [2])

        // The viewport supplies the ordered result after each toggle.
        state.selectPoints([face, edge])
        state.selectPoints([face, edge, vertex])
        XCTAssertEqual(state.selectedPoints, [face, edge, vertex])
        XCTAssertEqual(state.selectedSolids, [0, 1])
        XCTAssertEqual(state.selectedPartNodeIDs, ["case"])
        state.selectPoints([])
        XCTAssertTrue(state.selectedPoints.isEmpty)
        XCTAssertEqual(state.selectedSolids, [0, 1], "Deselecting all points must retain whole-part selection.")
        XCTAssertEqual(state.selectedPartNodeIDs, ["case"])
        XCTAssertEqual(state.measurementReport?.solidIndices, [0, 1])

        state.selectPoints(ordered)
        state.clearSelections()
        XCTAssertTrue(state.selectedPoints.isEmpty)
        XCTAssertTrue(state.selectedSolids.isEmpty)
        XCTAssertTrue(state.selectedPartNodeIDs.isEmpty)
        XCTAssertNil(state.measurementReport)
    }

    @MainActor
    func testPointSelectionIsPartialAndItsOwnerIsColoredButNeverExported() throws {
        let state = try fixture()
        state.selectPoints([point(0, 0)])
        XCTAssertTrue(state.selectedSolids.isEmpty, "A point must not highlight its entire owner")
        XCTAssertTrue(state.selectedPartNodeIDs.isEmpty, "A point selection must not look like a whole tree row selection")
        XCTAssertEqual(state.selectedPartSolids, [0])
        XCTAssertTrue(state.canExport)
        XCTAssertFalse(state.canExportSelectedParts, "Point picks do not add their parts to part exports")

        let color = PartColor(red: 0.2, green: 0.5, blue: 0.8)
        state.setPartColor(color, for: "base")
        XCTAssertEqual(state.solidColors, [0: color])
        XCTAssertEqual(state.selectedPoints, [point(0, 0)], "Color changes retain the exact selected point")
        state.selectParts([2])
        XCTAssertTrue(state.canExportSelectedParts)
        state.clearSelections()
        XCTAssertFalse(state.canExportSelectedParts)
    }

    @MainActor
    func testHideSelectedUsesTheUnionOfWholePartsAndPointOwnersAndClearsFeedback() throws {
        let state = try fixture()
        state.selectPoints([point(0, 0), point(0, 1)])
        state.selectParts([2])
        XCTAssertEqual(state.selectedPartSolids, [0, 2])
        state.hideSelectedParts()
        XCTAssertEqual(state.hiddenSolids, [0, 2])
        XCTAssertEqual(state.revisions.first?.hiddenSolids, [0, 2])
        XCTAssertTrue(state.selectedPoints.isEmpty)
        XCTAssertTrue(state.selectedSolids.isEmpty)
        XCTAssertTrue(state.selectedPartNodeIDs.isEmpty)
        XCTAssertFalse(state.hasMeasurementSelection)
        XCTAssertNil(state.measurementReport)
        XCTAssertNil(state.measurementError)
        XCTAssertFalse(state.isMeasuring)
        state.showAllParts()
        XCTAssertTrue(state.hiddenSolids.isEmpty)
        XCTAssertEqual(state.revisions.first?.hiddenSolids, [])
    }

    @MainActor
    func testVisibilityToggleRemovesOnlyPointsOwnedByTheHiddenParts() throws {
        let state = try fixture()
        let a = point(0, 0), b = point(1, 0), c = point(2, 0)
        state.selectPoints([c, b, a])
        state.togglePartVisibility([1])
        XCTAssertEqual(state.selectedPoints, [c, a])
        XCTAssertEqual(state.pointSelectedSolids, [0, 2])
        XCTAssertEqual(state.hiddenSolids, [1])
        XCTAssertNil(state.measurementError)
        XCTAssertFalse(state.isMeasuring)
        state.togglePartVisibility([1])
        XCTAssertEqual(state.selectedPoints, [c, a], "Showing a part must not reselect a discarded point")
        state.selectPoints([b])
        XCTAssertEqual(state.selectedPoints, [b])
        state.togglePartVisibility([1])
        state.selectPoints([b, a])
        XCTAssertEqual(state.selectedPoints, [a], "Points on hidden parts cannot be selected")
        state.clearSelections()
    }

    @MainActor
    func testMeshReplacementAndHistoryNavigationDiscardSessionPointSelection() throws {
        let state = try fixture()
        state.selectPoints([point(0, 0)])
        var replacement = try XCTUnwrap(state.mesh)
        replacement.artifactPath = "/session/replacement/model.rsdf"
        state.mesh = replacement
        XCTAssertTrue(state.selectedPoints.isEmpty, "A replaced preview must not carry points onto different geometry")
        XCTAssertNil(state.measurementReport)
        state.selectPoints([point(0, 0)])
        replacement.buildID = "replacement-preview"
        state.mesh = replacement
        XCTAssertTrue(state.selectedPoints.isEmpty)
        XCTAssertNil(state.measurementReport)

        let first = try XCTUnwrap(state.revisions.first)
        let second = ProjectRevision(model: .bracket, preview: preview(path: "history-second"), parts: parts())
        state.revisions.append(second)
        state.selectPoints([point(1, 0)])
        state.selectRevision(second.id)
        XCTAssertEqual(state.mesh?.buildID, "history-second")
        XCTAssertTrue(state.selectedPoints.isEmpty)
        XCTAssertNil(state.measurementReport)
        state.selectPoints([point(2, 0)])
        state.selectRevision(first.id)
        XCTAssertTrue(state.selectedPoints.isEmpty)
        XCTAssertNil(state.measurementReport)
    }

    @MainActor
    func testNewAndLoadedProjectsClearPointsAndCancelPreviousMeasurements() async throws {
        let state = try fixture()
        state.selectPoints([point(0, 0)])
        state.selectParts([1])
        XCTAssertTrue(state.isMeasuring, "A point and a part wait for the worker's minimum distance.")
        XCTAssertEqual(try values(state, "point1:position"), point(0, 0).point, "Local values show at once.")
        state.newProject()
        XCTAssertTrue(state.selectedPoints.isEmpty)
        XCTAssertNil(state.measurementReport)
        XCTAssertFalse(state.isMeasuring)
        await state.prepareToQuit()
        XCTAssertNil(state.projectURL)
        let directory = FileManager.default.temporaryDirectory.appendingPathComponent("RAD-point-reset-\(UUID())", isDirectory: true)
        try FileManager.default.createDirectory(at: directory, withIntermediateDirectories: false)
        defer { try? FileManager.default.removeItem(at: directory) }

        let incomingRevision = ProjectRevision(model: .bracket, preview: preview(path: "loaded-preview"), parts: parts())
        let incoming = RADProject(messages: [], revisions: [incomingRevision], selectedRevisionID: incomingRevision.id,
                                  projectID: UUID(), projectName: "Incoming")
        let incomingURL = directory.appendingPathComponent("Incoming.rad")
        try ProjectStore.write(incoming, to: incomingURL)
        state.loadProject(incoming, from: incomingURL)
        state.selectPoints([point(0, 0)])
        state.selectParts([0])
        XCTAssertNotNil(state.measurementReport)
        let empty = RADProject(messages: [], revisions: [], selectedRevisionID: nil, projectID: UUID(), projectName: "Empty")
        let emptyURL = directory.appendingPathComponent("Empty.rad")
        try ProjectStore.write(empty, to: emptyURL)
        state.loadProject(empty, from: emptyURL)
        try await Task.sleep(for: .milliseconds(200))
        XCTAssertTrue(state.selectedPoints.isEmpty)
        XCTAssertNil(state.measurementReport)
        XCTAssertNil(state.measurementError)
        XCTAssertFalse(state.isMeasuring)
        await state.prepareToQuit()
    }

    /// A point on the top face of the fixture part `owner`.
    private func point(_ owner: Int, _ index: Int) -> CADSurfacePoint {
        CADSurfacePoint(solidIndex: owner, point: [Double(owner * 5) + 0.25 * Double(index + 1), 0.5, 1], normal: [0, 0, 1])
    }

    @MainActor
    private func values(_ state: AppState, _ identity: String) throws -> [Double] {
        try XCTUnwrap(state.measurementReport?.metrics.first { $0.id == identity }, identity).values
    }

    @MainActor
    private func fixture() throws -> AppState {
        let root = FileManager.default.temporaryDirectory.appendingPathComponent("RAD-point-state-\(UUID())", isDirectory: true)
        try FileManager.default.createDirectory(at: root, withIntermediateDirectories: false)
        addTeardownBlock { try? FileManager.default.removeItem(at: root) }
        let state = AppState(autosaveStore: ProjectAutosaveStore(rootDirectory: root))
        let revision = ProjectRevision(model: .bracket, preview: preview(path: "initial-preview"), parts: parts())
        state.revisions = [revision]
        state.selectedRevisionID = revision.id
        XCTAssertTrue(state.loadStoredRevision())
        return state
    }

    private func parts() -> [PartNode] {
        [PartNode(id: "case", name: "Case", solidIndices: [], children: [
            PartNode(id: "base", name: "Base", solidIndices: [0]), PartNode(id: "lid", name: "Lid", solidIndices: [1])]),
         PartNode(id: "board", name: "Board", solidIndices: [2])]
    }

    private func preview(path: String) -> PreviewMesh {
        let owners = (0..<3).map { owner in
            CADPart(solidIndex: owner, area: 6, volume: 1,
                    bounds: [Double(owner * 5), 0, 0, Double(owner * 5 + 1), 1, 1], centroid: [Double(owner * 5) + 0.5, 0.5, 0.5],
                    triangles: 1, components: 1)
        }
        return PreviewMesh(vertices: [[0, 0, 0], [1, 0, 0], [0, 1, 0]], triangles: [[0, 1, 2], [0, 1, 2], [0, 1, 2]], edges: [],
                           bounds: [0, 0, 0, 11, 5, 1], volume: 3, area: 18, solids: 3,
                           buildSeconds: 0, threads: 1, buildID: path, triangleSolids: [0, 1, 2], edgeSolids: [], parts: parts(),
                           partStats: owners)
    }
}
