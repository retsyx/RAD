import Foundation
import XCTest
@testable import RAD

final class ProjectSelectionContextTests: XCTestCase {
    func testSelectedFoldersExpandToNamedPartsInDeterministicSolidOrder() throws {
        let selection = try XCTUnwrap(ProjectSelectionContext.capture(revisionID: UUID(), preview: preview(),
            nodes: hierarchy(), selectedSolids: [1], selectedNodeIDs: ["case"]))
        XCTAssertEqual(selection.parts.map(\.solidIndex), [0, 1])
        XCTAssertEqual(selection.parts.map(\.partID), ["base", "lid"])
        XCTAssertEqual(selection.parts.map(\.path), [["Case", "Base"], ["Case", "Lid"]])
        XCTAssertEqual(selection.parts[0].anchor?.bounds, [0, 0, 0, 4, 4, 4])
        XCTAssertEqual(selection.parts[0].anchor?.center, [2, 2, 2])
        XCTAssertEqual(selection.parts[1].anchor?.volume, 24)
        XCTAssertTrue(selection.points.isEmpty)
        XCTAssertNil(ProjectSelectionContext.capture(revisionID: UUID(), preview: preview(), nodes: hierarchy(),
            selectedSolids: [], selectedNodeIDs: []))
    }

    func testCaptureIsIndependentOfSubsequentGeometryAndHierarchyChanges() throws {
        var mesh = preview()
        var nodes = hierarchy()
        let selection = try XCTUnwrap(ProjectSelectionContext.capture(revisionID: UUID(), preview: mesh,
            nodes: nodes, selectedSolids: [0], selectedNodeIDs: []))
        mesh.partStats?[0].bounds = [100, 100, 100, 200, 200, 200]
        nodes[0].name = "Changed folder"
        nodes[0].children?[0].name = "Changed part"
        XCTAssertEqual(selection.parts[0].name, "Base")
        XCTAssertEqual(selection.parts[0].path, ["Case", "Base"])
        XCTAssertEqual(selection.parts[0].anchor?.bounds, [0, 0, 0, 4, 4, 4])
    }

    func testNamedPartSelectionSurvivesWhileNativeGeometryIsStillLoading() throws {
        let selection = try XCTUnwrap(ProjectSelectionContext.capture(revisionID: UUID(), preview: nil,
            nodes: hierarchy(), selectedSolids: [], selectedNodeIDs: ["case"]))
        XCTAssertEqual(selection.parts.map(\.solidIndex), [0, 1])
        XCTAssertEqual(selection.parts.map(\.partID), ["base", "lid"])
        XCTAssertEqual(selection.parts.map(\.path), [["Case", "Base"], ["Case", "Lid"]])
        XCTAssertTrue(selection.parts.allSatisfy { $0.anchor == nil }, "An unloaded preview must not introduce guessed geometry anchors.")
        XCTAssertNoThrow(try selection.validate())
    }

    func testChatAndQueuedSelectionsRoundTripAsIntentWithoutDerivedGeometryAssets() throws {
        let captured = try XCTUnwrap(ProjectSelectionContext.capture(revisionID: UUID(), preview: preview(),
            nodes: hierarchy(), selectedSolids: [1], selectedNodeIDs: []))
        let user = ChatMessage(role: "user", content: "Round the selected lid", selection: captured)
        let queue = ProjectQueuedRequest(content: "Move the selected lid", selection: captured)
        let project = RADProject(messages: [user], revisions: [], selectedRevisionID: nil, queuedRequests: [queue])
        let bytes = try ProjectStore.encode(project)
        let restored = try ProjectStore.decode(bytes)
        XCTAssertEqual(restored.messages, [user])
        XCTAssertEqual(restored.queuedRequests, [queue])
        XCTAssertEqual(restored.messages[0].selection?.revisionID, captured.revisionID,
                       "Save New may preserve a prompt whose originating revision is outside the copied history.")
        let manifest = try XCTUnwrap(JSONSerialization.jsonObject(with: ProjectZIP.decode(bytes)) as? [String: Any])
        XCTAssertTrue(try XCTUnwrap(manifest["assets"] as? [String: String]).isEmpty,
                      "Selection intent must not introduce programs, exports, images or meshes.")
        let json = String(decoding: try ProjectZIP.decode(bytes), as: UTF8.self)
        for forbidden in ["triangles", "partStats", "preview", "buildID", "artifactPath"] {
            XCTAssertFalse(json.contains("\"\(forbidden)\""), "The saved selected-target anchors must stay compact.")
        }
    }

    func testInvalidSelectionsCannotEnterAProjectArchive() throws {
        let original = try XCTUnwrap(ProjectSelectionContext.capture(revisionID: UUID(), preview: preview(),
            nodes: hierarchy(), selectedSolids: [0], selectedNodeIDs: []))
        var invalid = original
        invalid.parts[0].solidIndex = -1
        XCTAssertThrowsError(try invalid.validate())
        invalid = original
        invalid.parts.append(invalid.parts[0])
        XCTAssertThrowsError(try invalid.validate())
        invalid = original
        invalid.parts[0].anchor?.center = [Double.nan, 1, 2]
        XCTAssertThrowsError(try invalid.validate())
        invalid = original
        invalid.parts[0].path = ["folder\nname"]
        XCTAssertThrowsError(try invalid.validate())
        invalid = original
        invalid.parts[0].anchor?.bounds = [4, 0, 0, 0, 1, 1]
        XCTAssertThrowsError(try invalid.validate())
        let message = ChatMessage(role: "assistant", content: "Done", selection: original)
        XCTAssertThrowsError(try ProjectStore.encode(RADProject(messages: [message], revisions: [], selectedRevisionID: nil)))
        let queue = ProjectQueuedRequest(content: "Apply the change", selection: invalid)
        XCTAssertThrowsError(try ProjectStore.encode(RADProject(messages: [], revisions: [], selectedRevisionID: nil, queuedRequests: [queue])))
    }

    func testPointsKeepSelectionOrderPartIdentityAndBuiltCoordinatesUnderAPose() throws {
        let lid = CADSurfacePoint(solidIndex: 1, point: [2, 2, 6], normal: [0, 0, 1])
        let base = CADSurfacePoint(solidIndex: 0, point: [4, 1, 1], normal: [1, 0, 0])
        // The lid is turned and lifted: points stay in built coordinates, the
        // whole-part anchor's centroid is posed and its built bounds dropped.
        let pose = AssemblyPose(positions: ["hinge": 90], solidTransforms: [1: [0, -1, 0, 0, 1, 0, 0, 0, 0, 0, 1, 10, 0, 0, 0, 1]])
        let selection = try XCTUnwrap(ProjectSelectionContext.capture(revisionID: UUID(), preview: preview(), nodes: hierarchy(),
            selectedSolids: [1], selectedNodeIDs: [], selectedPoints: [lid, base, lid, CADSurfacePoint(solidIndex: 5, point: [0, 0, 0], normal: [0, 0, 1])],
            assemblyPose: pose))
        XCTAssertEqual(selection.points.map(\.partID), ["lid", "base"], "Duplicates and points on missing parts are dropped.")
        XCTAssertEqual(selection.points.map(\.path), [["Case", "Lid"], ["Case", "Base"]])
        XCTAssertEqual(selection.points.map(\.point), [[2, 2, 6], [4, 1, 1]])
        XCTAssertEqual(selection.points.map(\.normal), [[0, 0, 1], [1, 0, 0]])
        XCTAssertEqual(selection.parts.map(\.partID), ["lid"])
        XCTAssertEqual(selection.parts[0].anchor?.center, [-2, 2, 15])
        XCTAssertNil(selection.parts[0].anchor?.bounds)
        XCTAssertEqual(selection.jointPositions, ["hinge": 90])
        XCTAssertNoThrow(try selection.validate())

        let pointsOnly = try XCTUnwrap(ProjectSelectionContext.capture(revisionID: UUID(), preview: preview(), nodes: hierarchy(),
            selectedSolids: [], selectedNodeIDs: [], selectedPoints: [base]))
        XCTAssertTrue(pointsOnly.parts.isEmpty, "A point's part is not a whole-part selection.")
        XCTAssertEqual(pointsOnly.points.count, 1)
        let message = ChatMessage(role: "user", content: "Drill here", selection: pointsOnly)
        let restored = try ProjectStore.decode(ProjectStore.encode(RADProject(messages: [message], revisions: [], selectedRevisionID: nil)))
        XCTAssertEqual(restored.messages[0].selection, pointsOnly)
        let json = String(decoding: try JSONEncoder().encode(pointsOnly), as: UTF8.self)
        XCTAssertFalse(json.contains("\"features\""))
    }

    func testInvalidPointsCannotEnterAProjectArchive() throws {
        let point = CADSurfacePoint(solidIndex: 0, point: [4, 1, 1], normal: [1, 0, 0])
        let original = try XCTUnwrap(ProjectSelectionContext.capture(revisionID: UUID(), preview: preview(), nodes: hierarchy(),
            selectedSolids: [], selectedNodeIDs: [], selectedPoints: [point]))
        XCTAssertNoThrow(try original.validate())
        var invalid = original
        invalid.points.append(invalid.points[0])
        XCTAssertThrowsError(try invalid.validate(), "Duplicate points")
        let changes: [(inout ProjectSelectedPoint) -> Void] = [
            { $0.point = [1, .infinity, 0] }, { $0.point = [1, 2] }, { $0.point = [2_000_000, 0, 0] },
            { $0.normal = [0, 0, 0] }, { $0.normal = [0, 3, 4] },
            { $0.solidIndex = -1 }, { $0.partID = "" }, { $0.path = [] }
        ]
        for change in changes {
            invalid = original
            change(&invalid.points[0])
            XCTAssertThrowsError(try invalid.validate())
            let queue = ProjectQueuedRequest(content: "Apply the change", selection: invalid)
            XCTAssertThrowsError(try ProjectStore.encode(RADProject(messages: [], revisions: [], selectedRevisionID: nil,
                                                                    queuedRequests: [queue])))
        }
        XCTAssertThrowsError(try ProjectSelectionContext(revisionID: UUID(), parts: [], points: []).validate(),
                             "An empty selection is not a selection.")
    }

    private func hierarchy() -> [PartNode] {
        [PartNode(id: "case", name: "Case", solidIndices: [], children: [
            PartNode(id: "lid", name: "Lid", solidIndices: [1]),
            PartNode(id: "base", name: "Base", solidIndices: [0])
        ])]
    }

    private func preview() -> PreviewMesh {
        var mesh = PreviewMesh(vertices: [], triangles: [], edges: [], bounds: [0, 0, 0, 4, 4, 6], volume: 88,
            area: 152, solids: 2, buildSeconds: 0, threads: 1, buildID: UUID().uuidString)
        mesh.partStats = [
            CADPart(solidIndex: 0, area: 96, volume: 64, bounds: [0, 0, 0, 4, 4, 4], centroid: [2, 2, 2], triangles: 12, components: 1),
            CADPart(solidIndex: 1, area: 56, volume: 24, bounds: [0, 0, 4, 4, 4, 6], centroid: [2, 2, 5], triangles: 12, components: 1)
        ]
        return mesh
    }
}
