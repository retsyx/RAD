import AppKit
import Metal
import SceneKit
import simd
import XCTest
@testable import RAD

final class ViewportPointTests: XCTestCase {
    private let origin = CADSurfacePoint(solidIndex: 0, point: [0, 0, 0], normal: [0, 0, 1])
    private let corner = CADSurfacePoint(solidIndex: 0, point: [6, 0, 0], normal: [0, 0, 1])

    func testNearestSelectedMarkerWinsWithinTheClickTolerance() {
        let screen: (CADSurfacePoint) -> SIMD2<Double>? = { SIMD2($0.point[0], $0.point[1]) }
        let selection = [corner, origin]
        XCTAssertEqual(ViewportPointPicking.marker(at: [4, 0], in: selection, screen: screen), 0)
        XCTAssertNil(ViewportPointPicking.marker(at: [6 + ViewportPointPicking.markerTolerance + 0.5, 0],
                                                in: selection, screen: screen))
        XCTAssertNil(ViewportPointPicking.marker(at: [6, 0], in: selection, screen: { _ in nil }),
                     "A marker behind the eye is not clickable.")
    }

    @MainActor
    func testClicksSelectPartLocalPointsOnAPosedPartAndAMarkerClickRemovesItsPoint() async throws {
        let view = try makeView()
        let moved: [Int: [Double]] = [0: [1, 0, 0, 4, 0, 1, 0, 2, 0, 0, 1, 0, 0, 0, 0, 1]]
        view.update(mesh: mesh(), mode: .edges, grid: false, command: ViewportCommand(.top),
                    selectedSolids: [], hiddenSolids: [], solidTransforms: moved)
        try await waitForPreparation(view)
        var selections = [[CADSurfacePoint]]()
        var partChanges = 0
        view.onPointSelectionChanged = { selections.append($0) }
        view.onSelectionChanged = { _ in partChanges += 1 }

        // The part's local origin is at world (4, 2, 0).
        let target = point([4, 2, 0], in: view)
        let picked = try XCTUnwrap(view.surfacePoint(at: target))
        XCTAssertEqual(picked.solidIndex, 0)
        for axis in 0..<3 { XCTAssertEqual(picked.point[axis], 0, accuracy: 1e-3, "Points are in built part coordinates.") }
        XCTAssertEqual(picked.normal[2], 1, accuracy: 1e-6)
        XCTAssertTrue(picked.isValid)

        view.mouseDown(with: try click(at: target, modifiers: .shift))
        XCTAssertEqual(selections.count, 1)
        let first = try XCTUnwrap(selections.last?.first)
        view.mouseDown(with: try click(at: point([9, 7, 0], in: view), modifiers: .shift))
        let second = try XCTUnwrap(selections.last?.last)
        XCTAssertEqual(selections.last?.count, 2)
        XCTAssertEqual(second.point[0], 5, accuracy: 1e-3)
        XCTAssertEqual(second.point[1], 5, accuracy: 1e-3)
        XCTAssertEqual(view.makeFrame(width: 640, height: 480).seeds.map(\.channel), [.selectedVertex, .selectedVertex])

        view.mouseDown(with: try click(at: NSPoint(x: target.x + 3, y: target.y - 2)))
        XCTAssertEqual(selections.last, [second], "Clicking near a selected marker removes that point.")
        XCTAssertFalse(selections.last?.contains(first) ?? true)
        XCTAssertEqual(partChanges, 0, "Point clicks never select whole parts.")
        XCTAssertNil(view.surfacePoint(at: NSPoint(x: 4, y: 4)), "Empty space has no surface point.")
    }

    @MainActor
    func testEmptySingleClickPreservesSelectionAndDoubleClickClearsEverythingWithoutFit() async throws {
        let view = try makeView()
        view.update(mesh: mesh(), mode: .edges, grid: false, command: ViewportCommand(.front),
                    selectedSolids: [0], hiddenSolids: [], selectedPoints: [origin])
        try await waitForPreparation(view)
        var partSelections = [Set<Int>]()
        var pointSelections = [[CADSurfacePoint]]()
        var clearCount = 0
        view.onSelectionChanged = { partSelections.append($0) }
        view.onPointSelectionChanged = { pointSelections.append($0) }
        view.onDeselectAll = { clearCount += 1 }
        let camera = try XCTUnwrap(view.pointOfView)
        let before = camera.simdTransform
        let empty = NSPoint(x: 4, y: 4)
        view.mouseDown(with: try click(at: empty))
        XCTAssertTrue(partSelections.isEmpty)
        XCTAssertTrue(pointSelections.isEmpty)
        view.mouseDown(with: try click(at: empty, count: 2))
        XCTAssertEqual(partSelections, [[]])
        XCTAssertEqual(pointSelections, [[]])
        XCTAssertEqual(clearCount, 1)
        for column in 0..<4 {
            XCTAssertEqual(camera.simdTransform[column], before[column], "Double-clicking empty space must not fit or move the camera.")
        }
    }

    @MainActor
    func testEscapeAndOptionClickKeepTheirPartAndDeselectRoles() async throws {
        let view = try makeView()
        view.update(mesh: mesh(), mode: .edges, grid: false, command: ViewportCommand(.top),
                    selectedSolids: [], hiddenSolids: [], selectedPoints: [origin])
        try await waitForPreparation(view)
        var parts = [Set<Int>](), points = [[CADSurfacePoint]]()
        view.onSelectionChanged = { parts.append($0) }
        view.onPointSelectionChanged = { points.append($0) }
        let center = point([3, 3, 0], in: view)
        view.mouseDown(with: try click(at: center, modifiers: .option))
        XCTAssertEqual(parts, [[0]], "Option-click toggles the whole part.")
        XCTAssertTrue(points.isEmpty, "Option-click leaves points selected.")
        let escape = try XCTUnwrap(NSEvent.keyEvent(with: .keyDown, location: .zero, modifierFlags: [], timestamp: 0,
            windowNumber: 0, context: nil, characters: "\u{1b}", charactersIgnoringModifiers: "\u{1b}", isARepeat: false, keyCode: 53))
        view.keyDown(with: escape)
        XCTAssertEqual(parts.last, [])
        XCTAssertEqual(points.last, [])
        XCTAssertEqual(view.makeFrame(width: 640, height: 480).seeds.count, 0)
    }

    @MainActor
    func testRightClickContextMenuPreservesPointAndPartSelections() async throws {
        let view = try makeView()
        view.update(mesh: mesh(), mode: .edges, grid: false, command: ViewportCommand(.top),
                    selectedSolids: [0], hiddenSolids: [], selectedPoints: [origin])
        try await waitForPreparation(view)
        var partChanges = 0
        var pointChanges = 0
        view.onSelectionChanged = { _ in partChanges += 1 }
        view.onPointSelectionChanged = { _ in pointChanges += 1 }
        let rightClick = try XCTUnwrap(NSEvent.mouseEvent(with: .rightMouseDown,
            location: NSPoint(x: view.bounds.midX, y: view.bounds.midY), modifierFlags: [], timestamp: 0,
            windowNumber: 0, context: nil, eventNumber: 1, clickCount: 1, pressure: 1))
        // This is the menu preparation used by rightMouseUp, without starting
        // AppKit's blocking menu-tracking loop in the test runner.
        let menu = try XCTUnwrap(view.menu(for: rightClick))
        XCTAssertEqual(partChanges, 0)
        XCTAssertEqual(pointChanges, 0)
        XCTAssertTrue(try XCTUnwrap(menu.item(withTitle: "Hide Selected Parts")).isEnabled)
        XCTAssertTrue(try XCTUnwrap(menu.item(withTitle: "Deselect All")).isEnabled)

        view.update(mesh: mesh(), mode: .edges, grid: false, command: nil,
                    selectedSolids: [], hiddenSolids: [], selectedPoints: [origin])
        let pointMenu = try XCTUnwrap(view.menu(for: rightClick))
        XCTAssertTrue(try XCTUnwrap(pointMenu.item(withTitle: "Hide Selected Parts")).isEnabled,
                      "Selected points allow hiding their owning parts without selecting the whole part.")
        XCTAssertTrue(try XCTUnwrap(pointMenu.item(withTitle: "Deselect All")).isEnabled)
        let material = try XCTUnwrap(node("solid:0", in: view).geometry?.firstMaterial)
        let color = try XCTUnwrap((material.diffuse.contents as? NSColor)?.usingColorSpace(.sRGB))
        XCTAssertEqual(color.redComponent, 0.70, accuracy: 1e-6, "Opening the menu must not apply whole-part selection styling.")
        XCTAssertEqual(partChanges, 0)
        XCTAssertEqual(pointChanges, 0)
        view.update(mesh: mesh(), mode: .edges, grid: false, command: nil,
                    selectedSolids: [], hiddenSolids: [0], selectedPoints: [origin])
        XCTAssertFalse(try XCTUnwrap(view.menu(for: rightClick)?.item(withTitle: "Hide Selected Parts")).isEnabled)
    }

    @MainActor
    func testSelectedMarkersFollowPartMotionAndHiddenPartsDrawNone() async throws {
        let view = try makeView()
        let moved: [Int: [Double]] = [0: [0, -1, 0, 3, 1, 0, 0, -2, 0, 0, 1, 1, 0, 0, 0, 1]]
        view.update(mesh: mesh(), mode: .edges, grid: false, command: ViewportCommand(.top),
                    selectedSolids: [], hiddenSolids: [], selectedPoints: [corner, origin, corner], solidTransforms: moved)
        try await waitForPreparation(view)
        let frame = view.makeFrame(width: 640, height: 480)
        XCTAssertEqual(frame.seeds.map(\.channel), [.selectedVertex, .selectedVertex], "Duplicate points draw one marker.")
        XCTAssertTrue(frame.verticesEnabled)
        let expected = ViewportRigidTransform.sceneMatrix(try XCTUnwrap(ViewportRigidTransform.matrix(moved[0]!)))
        for seeds in frame.seeds {
            for column in 0..<4 { XCTAssertEqual(seeds.model[column], expected[column], "Markers follow their part's placement.") }
        }
        XCTAssertEqual(view.cachedPointHighlightCount, 2)
        view.update(mesh: mesh(), mode: .edges, grid: false, command: nil,
                    selectedSolids: [], hiddenSolids: [0], selectedPoints: [corner, origin], solidTransforms: moved)
        XCTAssertTrue(view.makeFrame(width: 640, height: 480).seeds.isEmpty, "Hidden parts draw no markers.")
        XCTAssertNil(view.screenPosition(of: origin))
    }

    @MainActor
    func testHoverAndClickUseOneWholeCreaseAndLeavePointSelectionAvailable() async throws {
        let view = try makeView()
        var preview = mesh()
        preview.edgePoints = [-10, -10, 0, 0, -10, 0, 0, -10, 0, 10, -10, 0]
        preview.edgeSolids = [0, 0]
        view.update(mesh: preview, mode: .edges, grid: false, command: ViewportCommand(.top),
                    selectedSolids: [], hiddenSolids: [])
        try await waitForPreparation(view)
        let first = point([-7, -10, 0], in: view)
        let second = point([7, -10, 0], in: view)
        var edges = [[CADCreaseEdge]](), points = [[CADSurfacePoint]]()
        view.onEdgeSelectionChanged = { edges.append($0) }
        view.onPointSelectionChanged = { points.append($0) }
        let move = try XCTUnwrap(NSEvent.mouseEvent(with: .mouseMoved, location: first, modifierFlags: [],
            timestamp: 0, windowNumber: 0, context: nil, eventNumber: 1, clickCount: 0, pressure: 0))
        view.mouseMoved(with: move)
        let hover = try XCTUnwrap(view.makeFrame(width: 640, height: 480).seeds.first)
        XCTAssertEqual(hover.channel, .hoveredEdge)
        XCTAssertEqual(hover.buffer.vertexCount, 4, "Both segments highlight from either end.")
        XCTAssertTrue(edges.isEmpty)
        view.mouseDown(with: try click(at: first))
        let edge = try XCTUnwrap(edges.last?.first)
        XCTAssertEqual(edge.length, 20, accuracy: 1e-5)
        XCTAssertTrue(points.isEmpty)
        view.mouseMoved(with: move)
        let selected = view.makeFrame(width: 640, height: 480)
        XCTAssertEqual(selected.seeds.map(\.channel), [.selectedEdge])
        XCTAssertTrue(selected.seeds.first?.buffer === hover.buffer)
        XCTAssertFalse(selected.verticesEnabled)
        let metadata = try XCTUnwrap((view.agentCaptureMetadata["selectedEdges"] as? [[String: Any]])?.first)
        XCTAssertEqual(metadata["length"] as? Double, edge.length)
        XCTAssertEqual((metadata["landmarks"] as? [[Double]])?.count, 3)

        view.mouseDown(with: try click(at: point([0, 0, 0], in: view), modifiers: .shift))
        XCTAssertEqual(points.last?.count, 1)
        view.mouseDown(with: try click(at: second))
        XCTAssertEqual(edges.last, [], "Clicking another segment toggles the same edge off.")
        XCTAssertEqual(points.last?.count, 1)
        let marker = CADSurfacePoint(solidIndex: 0, point: [-7, -10, 0], normal: [0, 0, 1])
        view.update(mesh: preview, mode: .edges, grid: false, command: nil, selectedSolids: [], hiddenSolids: [],
                    selectedPoints: [marker], selectedEdges: [edge])
        view.mouseDown(with: try click(at: first))
        XCTAssertEqual(points.last, [], "An existing point marker takes precedence over its edge.")
        XCTAssertEqual(view.makeFrame(width: 640, height: 480).seeds.map(\.channel), [.selectedEdge])
    }

    @MainActor
    func testCreasePickingHonorsOcclusionVisibilityPartPlacementAndDisplayMode() async throws {
        let view = try makeView()
        let preview = PreviewMesh(vertices: [[-10,-10,0], [10,-10,0], [10,10,0], [-10,10,0],
                                              [-3,-3,1], [3,-3,1], [3,3,1], [-3,3,1]],
            triangles: [[0,1,2], [0,2,3], [4,5,6], [4,6,7]], edges: [[-10,0,0,10,0,0]],
            bounds: [-10,-10,0,10,10,1], volume: 1, area: 436, solids: 2, buildSeconds: 0, threads: 1,
            buildID: "covered-edge", triangleSolids: [0,0,1,1], edgeSolids: [0])
        for projection in [ViewportProjectionMode.perspective, .isometric] {
            view.update(mesh: preview, mode: .edges, grid: false, command: ViewportCommand(.top),
                        selectedSolids: [], hiddenSolids: [], projection: projection)
            try await waitForPreparation(view)
            let center = point([0, 0, 0], in: view)
            guard case .patch(let cover) = view.pickFeature(at: center) else {
                return XCTFail("The front part hides the rear crease.")
            }
            XCTAssertEqual(cover.solidIndex, 1)
            guard case .edge(let edge) = view.pickFeature(at: point([8, 0, 0], in: view)) else {
                return XCTFail("The exposed portion of the same crease is selectable.")
            }
            view.update(mesh: preview, mode: .shaded, grid: false, command: nil, selectedSolids: [],
                        hiddenSolids: [], projection: projection, selectedEdges: [edge])
            XCTAssertEqual(view.pickFeature(at: center), .edge(edge), "The selected overlay can be toggled through the cover.")
            view.update(mesh: preview, mode: .edges, grid: false, command: nil, selectedSolids: [],
                        hiddenSolids: [1], projection: projection, selectedEdges: [edge])
            XCTAssertEqual(view.pickFeature(at: center), .edge(edge))
            let moved: [Int: [Double]] = [0: [1,0,0,0, 0,1,0,5, 0,0,1,0, 0,0,0,1]]
            view.update(mesh: preview, mode: .edges, grid: false, command: nil, selectedSolids: [],
                        hiddenSolids: [1], projection: projection, selectedEdges: [edge], solidTransforms: moved)
            let posed = point([0, 5, 0], in: view)
            XCTAssertNotEqual(view.pickFeature(at: center), .edge(edge))
            XCTAssertEqual(view.pickFeature(at: posed), .edge(edge))
            let frame = view.makeFrame(width: 640, height: 480)
            XCTAssertEqual(frame.seeds.first?.model.columns.3.y, 5)
            view.update(mesh: preview, mode: .shaded, grid: false, command: nil, selectedSolids: [],
                        hiddenSolids: [1], projection: projection)
            if case .edge = view.pickFeature(at: center) { XCTFail("Shaded mode hides the ordinary crease lines.") }
            view.update(mesh: preview, mode: .wireframe, grid: false, command: nil, selectedSolids: [],
                        hiddenSolids: [], projection: projection)
            XCTAssertEqual(view.pickFeature(at: center), .edge(edge), "Wireframe exposes lines behind faces.")
            view.update(mesh: preview, mode: .edges, grid: false, command: nil, selectedSolids: [],
                        hiddenSolids: [0], projection: projection, selectedEdges: [edge])
            XCTAssertTrue(view.makeFrame(width: 640, height: 480).seeds.isEmpty)
            XCTAssertNil(view.pickFeature(at: point([8, 0, 0], in: view)))
        }
    }

    @MainActor
    func testShiftSwitchesStationaryHoverAndPicksAPointEvenBesideACrease() async throws {
        let view = try makeView()
        let preview = mesh()
        view.update(mesh: preview, mode: .edges, grid: false, command: ViewportCommand(.top),
                    selectedSolids: [], hiddenSolids: [])
        try await waitForPreparation(view)
        let rim = point([0, -10, 0], in: view)
        let interior = point([0, 0, 0], in: view)
        let nearEdge = NSPoint(x: rim.x, y: rim.y + (interior.y > rim.y ? 2 : -2))
        guard case .edge = view.pickFeature(at: nearEdge) else { return XCTFail("Expected the ordinary crease.") }
        func flags(_ flags: NSEvent.ModifierFlags) throws -> NSEvent {
            try XCTUnwrap(NSEvent.keyEvent(with: .flagsChanged, location: interior, modifierFlags: flags, timestamp: 0,
                windowNumber: 0, context: nil, characters: "", charactersIgnoringModifiers: "", isARepeat: false, keyCode: 56))
        }
        func move(_ point: NSPoint) throws {
            view.mouseMoved(with: try XCTUnwrap(NSEvent.mouseEvent(with: .mouseMoved, location: point, modifierFlags: [],
                timestamp: 0, windowNumber: 0, context: nil, eventNumber: 1, clickCount: 0, pressure: 0)))
        }
        try move(interior)
        XCTAssertEqual(view.makeFrame(width: 640, height: 480).batches.flatMap(\.masks).map(\.selected), [false])
        view.flagsChanged(with: try flags(.shift))
        var frame = view.makeFrame(width: 640, height: 480)
        XCTAssertEqual(frame.seeds.map(\.channel), [.hoveredVertex])
        XCTAssertTrue(frame.batches.isEmpty)
        view.flagsChanged(with: try flags([]))
        frame = view.makeFrame(width: 640, height: 480)
        XCTAssertTrue(frame.seeds.isEmpty)
        XCTAssertEqual(frame.batches.flatMap(\.masks).map(\.selected), [false])

        var points = [CADSurfacePoint](), patches = [CADSurfacePatch]()
        view.onPointSelectionChanged = { points = $0 }
        view.onPatchSelectionChanged = { patches = $0 }
        view.mouseDown(with: try click(at: nearEdge, modifiers: .shift))
        XCTAssertEqual(points.count, 1)
        XCTAssertTrue(patches.isEmpty)
        view.mouseDown(with: try click(at: interior))
        let patch = try XCTUnwrap(patches.first)
        let moved = [0: [1.0,0,0,3, 0,1,0,6, 0,0,1,4, 0,0,0,1]]
        view.update(mesh: preview, mode: .shaded, grid: false, command: nil, selectedSolids: [], hiddenSolids: [],
                    selectedPatches: [patch], solidTransforms: moved)
        XCTAssertEqual(view.pickFeature(at: point([3, 6, 4], in: view)), .patch(patch))
        XCTAssertEqual(view.makeFrame(width: 640, height: 480).batches.first?.masks.first?.model.columns.3, [3,6,4,1])
        view.update(mesh: preview, mode: .wireframe, grid: false, command: nil, selectedSolids: [], hiddenSolids: [])
        XCTAssertNil(view.pickFeature(at: interior), "Wireframe has no filled face to select.")
        guard case .point = view.pickFeature(at: interior, precisePoint: true) else {
            return XCTFail("An explicit point pick still works in wireframe.")
        }
        view.update(mesh: preview, mode: .edges, grid: false, command: nil, selectedSolids: [], hiddenSolids: [0],
                    selectedPatches: [patch])
        XCTAssertNil(view.pickFeature(at: interior))
        XCTAssertTrue(view.makeFrame(width: 640, height: 480).batches.isEmpty)
    }

    @MainActor
    private func makeView() throws -> CADSceneView {
        _ = try XCTUnwrap(MTLCreateSystemDefaultDevice())
        let view = CADSceneView(frame: NSRect(x: 0, y: 0, width: 640, height: 480))
        view.layout()
        return view
    }

    @MainActor
    private func point(_ position: [Double], in view: CADSceneView) -> NSPoint {
        let projected = view.projectPoint(SCNVector3(position[0], position[1], position[2]))
        return NSPoint(x: CGFloat(projected.x), y: CGFloat(projected.y))
    }

    @MainActor
    private func click(at point: NSPoint, count: Int = 1, modifiers: NSEvent.ModifierFlags = []) throws -> NSEvent {
        try XCTUnwrap(NSEvent.mouseEvent(with: .leftMouseDown, location: point, modifierFlags: modifiers, timestamp: 0,
                                       windowNumber: 0, context: nil, eventNumber: 1, clickCount: count, pressure: 1))
    }

    @MainActor
    private func waitForPreparation(_ view: CADSceneView) async throws {
        let deadline = Date().addingTimeInterval(5)
        while view.isPreparingResources, Date() < deadline { try await Task.sleep(nanoseconds: 5_000_000) }
        XCTAssertFalse(view.isPreparingResources)
    }

    @MainActor
    private func node(_ name: String, in view: CADSceneView) throws -> SCNNode {
        try XCTUnwrap(view.scene.rootNode.childNode(withName: name, recursively: true))
    }

    private func mesh() -> PreviewMesh {
        PreviewMesh(vertices: [[-10, -10, 0], [10, -10, 0], [10, 10, 0], [-10, 10, 0]],
                    triangles: [[0, 1, 2], [0, 2, 3]],
                    edges: [[-10, -10, 0, 10, -10, 0]],
                    bounds: [-10, -10, 0, 10, 10, 0], volume: 1, area: 400, solids: 1,
                    buildSeconds: 0, threads: 1, buildID: "viewport-surface-points",
                    triangleSolids: [0, 0], edgeSolids: [0])
    }
}
