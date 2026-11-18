import AppKit
import Metal
import SceneKit
import simd
import XCTest
@testable import RAD

final class ViewportPointTests: XCTestCase {
    private let origin = CADSurfacePoint(solidIndex: 0, point: [0, 0, 0], normal: [0, 0, 1])
    private let corner = CADSurfacePoint(solidIndex: 0, point: [6, 0, 0], normal: [0, 0, 1])

    func testAMarkerClickRemovesTheNearestSelectedPointAndOtherClicksAddTheHitInOrder() {
        // Screen positions in points equal the points' X and Y here.
        let screen: (CADSurfacePoint) -> SIMD2<Double>? = { SIMD2($0.point[0], $0.point[1]) }
        var selection = ViewportPointPicking.toggling(at: [6, 0], hit: corner, in: [], screen: screen)
        selection = ViewportPointPicking.toggling(at: [30, 0], hit: origin, in: selection, screen: screen)
        XCTAssertEqual(selection, [corner, origin], "The selection keeps click order.")

        let near = CADSurfacePoint(solidIndex: 0, point: [4, 0, 0], normal: [0, 0, 1])
        XCTAssertEqual(ViewportPointPicking.marker(at: [4, 0], in: selection, screen: screen), 0,
                       "The nearest marker within the tolerance wins.")
        XCTAssertEqual(ViewportPointPicking.toggling(at: [4, 0], hit: near, in: selection, screen: screen), [origin],
                       "A click on a marker removes its point instead of adding the hit beneath it.")
        let far = CADSurfacePoint(solidIndex: 0, point: [20, 0, 0], normal: [0, 0, 1])
        XCTAssertEqual(ViewportPointPicking.toggling(at: [20, 0], hit: far, in: selection, screen: screen), [corner, origin, far])
        XCTAssertEqual(ViewportPointPicking.toggling(at: [6 + ViewportPointPicking.markerTolerance + 0.5, 0], hit: nil,
                                                     in: selection, screen: screen), selection,
                       "A miss beyond every marker changes nothing.")
        XCTAssertNil(ViewportPointPicking.marker(at: [6, 0], in: selection, screen: { _ in nil }),
                     "Markers without a screen position, such as those behind the eye, are not clickable.")
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

        view.mouseDown(with: try click(at: target))
        XCTAssertEqual(selections.count, 1)
        let first = try XCTUnwrap(selections.last?.first)
        view.mouseDown(with: try click(at: point([9, 7, 0], in: view)))
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
