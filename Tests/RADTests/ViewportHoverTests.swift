import AppKit
import Metal
import SceneKit
import XCTest
@testable import RAD

final class ViewportHoverTests: XCTestCase {
    private let selected = CADSurfacePoint(solidIndex: 0, point: [-5, -5, 0], normal: [0, 0, 1])

    @MainActor
    func testTrackingIncludesMovementAndBoundaryEvents() throws {
        let view = try makeView()
        view.updateTrackingAreas()
        XCTAssertTrue(view.trackingAreas.contains {
            $0.options.contains(.mouseMoved) && $0.options.contains(.mouseEnteredAndExited) &&
            $0.options.contains(.inVisibleRect)
        })
    }

    @MainActor
    func testHoverShowsTheClickablePointMarkerWithoutChangingSelection() async throws {
        let view = try await readyView()
        var partChanges = 0, pointChanges = 0, clears = 0
        view.onSelectionChanged = { _ in partChanges += 1 }
        view.onPointSelectionChanged = { _ in pointChanges += 1 }
        view.onDeselectAll = { clears += 1 }

        view.mouseEntered(with: try event(.mouseEntered, at: point([0, 0, 0], in: view)))
        XCTAssertEqual(channels(in: view), [.hoveredVertex])
        view.mouseMoved(with: try event(.mouseMoved, at: point([5, 5, 0], in: view)))
        XCTAssertEqual(channels(in: view), [.hoveredVertex])
        XCTAssertEqual(view.cachedPointHighlightCount, 1, "Only the active hover marker keeps a buffer.")
        view.mouseMoved(with: try event(.mouseMoved, at: NSPoint(x: 2, y: 2)))
        XCTAssertEqual(channels(in: view), [], "Empty-space hover clears the marker.")

        view.mouseMoved(with: try event(.mouseMoved, at: point([1, 1, 0], in: view)))
        XCTAssertEqual(channels(in: view), [.hoveredVertex])
        view.mouseDragged(with: try event(.leftMouseDragged, at: NSPoint(x: 2, y: 2)))
        XCTAssertEqual(channels(in: view), [], "A button-held drag must not leave a hover marker behind.")
        view.mouseMoved(with: try event(.mouseMoved, at: point([1, 1, 0], in: view)))
        view.mouseExited(with: try event(.mouseExited, at: NSPoint(x: -1, y: -1)))
        XCTAssertEqual(channels(in: view), [])
        XCTAssertEqual(partChanges, 0)
        XCTAssertEqual(pointChanges, 0, "Preselection must not become a measurement selection.")
        XCTAssertEqual(clears, 0)
        XCTAssertEqual(view.cachedPointHighlightCount, 0)
    }

    @MainActor
    func testSelectedMarkersTakePrecedenceAndHoverOnAMarkerShowsNoNewPoint() async throws {
        let view = try makeView()
        let preview = mesh()
        view.update(mesh: preview, mode: .edges, grid: false, command: ViewportCommand(.top),
                    selectedSolids: [], hiddenSolids: [], selectedPoints: [selected])
        try await waitForPreparation(view)
        XCTAssertEqual(channels(in: view), [.selectedVertex])
        view.mouseMoved(with: try event(.mouseMoved, at: point(selected.point, in: view)))
        XCTAssertEqual(channels(in: view), [.selectedVertex], "A click there removes the point, so nothing new is previewed.")
        view.mouseMoved(with: try event(.mouseMoved, at: point([5, 5, 0], in: view)))
        XCTAssertEqual(channels(in: view), [.selectedVertex, .hoveredVertex])
        view.mouseExited(with: try event(.mouseExited, at: NSPoint(x: -1, y: -1)))
        XCTAssertEqual(channels(in: view), [.selectedVertex], "Exit clears only preselection.")

        view.mouseMoved(with: try event(.mouseMoved, at: point([5, 5, 0], in: view)))
        view.update(mesh: preview, mode: .edges, grid: false, command: nil,
                    selectedSolids: [0], hiddenSolids: [], selectedPoints: [selected])
        XCTAssertEqual(channels(in: view), [.selectedVertex, .hoveredVertex], "Selected parts still preview points on them.")
        XCTAssertEqual(view.makeFrame(width: 640, height: 480).batches.flatMap(\.masks).filter(\.selected).count, 1)
    }

    @MainActor
    func testHidingMovingAndReplacingGeometryCannotRestoreAStaleHover() async throws {
        let view = try await readyView()
        view.mouseMoved(with: try event(.mouseMoved, at: point([0, 0, 0], in: view)))
        XCTAssertEqual(channels(in: view), [.hoveredVertex])
        view.update(mesh: mesh(), mode: .edges, grid: false, command: nil, selectedSolids: [], hiddenSolids: [0])
        XCTAssertEqual(channels(in: view), [])
        view.update(mesh: mesh(), mode: .edges, grid: false, command: nil, selectedSolids: [], hiddenSolids: [])
        XCTAssertEqual(channels(in: view), [], "Showing a part must not restore an old hover.")

        view.mouseMoved(with: try event(.mouseMoved, at: point([0, 0, 0], in: view)))
        XCTAssertEqual(channels(in: view), [.hoveredVertex])
        view.update(mesh: mesh(), mode: .edges, grid: false, command: nil, selectedSolids: [], hiddenSolids: [],
                    solidTransforms: [0: [1, 0, 0, 2, 0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1]])
        XCTAssertEqual(channels(in: view), [], "Moving a part clears the hover over its old position.")

        view.mouseMoved(with: try event(.mouseMoved, at: point([2, 0, 0], in: view)))
        XCTAssertEqual(channels(in: view), [.hoveredVertex])
        var replacement = mesh()
        replacement.buildID = "viewport-hover-replacement"
        view.update(mesh: replacement, mode: .edges, grid: false, command: nil, selectedSolids: [], hiddenSolids: [])
        XCTAssertEqual(view.cachedPointHighlightCount, 0)
        view.update(mesh: nil, mode: .edges, grid: false, command: nil, selectedSolids: [], hiddenSolids: [])
        XCTAssertEqual(channels(in: view), [])
        XCTAssertNil(view.surfacePoint(at: NSPoint(x: 320, y: 240)))
    }

    @MainActor
    func testCameraAndViewportChangesClearHoverAndRefreshPickingCoordinates() async throws {
        let view = try await readyView()
        let preview = mesh()
        view.mouseMoved(with: try event(.mouseMoved, at: point([0, 0, 0], in: view)))
        XCTAssertEqual(channels(in: view), [.hoveredVertex])
        view.update(mesh: preview, mode: .edges, grid: false, command: ViewportCommand(.front),
                    selectedSolids: [], hiddenSolids: [])
        XCTAssertEqual(channels(in: view), [])
        view.update(mesh: preview, mode: .edges, grid: false, command: ViewportCommand(.top),
                    selectedSolids: [], hiddenSolids: [])
        view.mouseMoved(with: try event(.mouseMoved, at: point([0, 0, 0], in: view)))
        XCTAssertEqual(channels(in: view), [.hoveredVertex])
        view.setFrameSize(NSSize(width: 800, height: 600))
        view.layout()
        XCTAssertEqual(channels(in: view), [])
        let target = point([-8, -8, 0], in: view)
        let picked = try XCTUnwrap(view.surfacePoint(at: target), "A resized viewport picks with its new projection.")
        XCTAssertEqual(picked.point[0], -8, accuracy: 1e-3)
        XCTAssertEqual(picked.point[1], -8, accuracy: 1e-3)
        view.mouseMoved(with: try event(.mouseMoved, at: target))
        XCTAssertEqual(channels(in: view), [.hoveredVertex])
    }

    @MainActor
    func testWindowResignationAndDetachmentClearOnlyHover() async throws {
        let view = try makeView()
        let window = NSWindow(contentRect: view.frame, styleMask: [.borderless], backing: .buffered, defer: false)
        window.contentView = view
        view.update(mesh: mesh(), mode: .edges, grid: false, command: ViewportCommand(.top),
                    selectedSolids: [], hiddenSolids: [], selectedPoints: [selected])
        try await waitForPreparation(view)
        let hover = point([6, 6, 0], in: view)
        view.mouseMoved(with: try event(.mouseMoved, at: hover))
        XCTAssertEqual(channels(in: view), [.selectedVertex, .hoveredVertex])
        NotificationCenter.default.post(name: NSWindow.didResignKeyNotification, object: window)
        XCTAssertEqual(channels(in: view), [.selectedVertex])
        view.mouseMoved(with: try event(.mouseMoved, at: hover))
        XCTAssertEqual(channels(in: view), [.selectedVertex, .hoveredVertex])
        XCTAssertTrue(view.resignFirstResponder())
        XCTAssertEqual(channels(in: view), [.selectedVertex], "Moving keyboard focus to the composer must clear preselection too.")
        view.mouseMoved(with: try event(.mouseMoved, at: hover))
        XCTAssertEqual(channels(in: view), [.selectedVertex, .hoveredVertex])
        view.removeFromSuperview()
        XCTAssertEqual(channels(in: view), [.selectedVertex])
    }

    @MainActor
    func testExploringTheSurfaceRetainsOnlyActiveMarkerBuffers() async throws {
        let view = try makeView()
        view.update(mesh: mesh(), mode: .edges, grid: false, command: ViewportCommand(.top),
                    selectedSolids: [], hiddenSolids: [], selectedPoints: [selected])
        try await waitForPreparation(view)
        for index in 0..<81 {
            let position = [Double(index % 9) * 2 - 8, Double(index / 9) * 2 - 8, 0]
            view.mouseMoved(with: try event(.mouseMoved, at: point(position, in: view)))
        }
        XCTAssertLessThanOrEqual(view.cachedPointHighlightCount, 2,
                                 "Exploring the surface must not retain a marker for every point ever hovered.")
        XCTAssertEqual(channels(in: view).first, .selectedVertex, "The selected marker is retained.")
    }

    @MainActor
    private func readyView() async throws -> CADSceneView {
        let view = try makeView()
        view.update(mesh: mesh(), mode: .edges, grid: false, command: ViewportCommand(.top),
                    selectedSolids: [], hiddenSolids: [])
        try await waitForPreparation(view)
        return view
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
    private func event(_ type: NSEvent.EventType, at point: NSPoint) throws -> NSEvent {
        if type == .mouseEntered || type == .mouseExited {
            return try XCTUnwrap(NSEvent.enterExitEvent(with: type, location: point, modifierFlags: [], timestamp: 0,
                windowNumber: 0, context: nil, eventNumber: 1, trackingNumber: 0, userData: nil))
        }
        return try XCTUnwrap(NSEvent.mouseEvent(with: type, location: point, modifierFlags: [], timestamp: 0,
            windowNumber: 0, context: nil, eventNumber: 1, clickCount: 0, pressure: 0))
    }

    @MainActor
    private func waitForPreparation(_ view: CADSceneView) async throws {
        let deadline = Date().addingTimeInterval(5)
        while view.isPreparingResources, Date() < deadline { try await Task.sleep(for: .milliseconds(5)) }
        XCTAssertFalse(view.isPreparingResources)
    }

    /// Seed channels the next frame draws for point markers.
    @MainActor
    private func channels(in view: CADSceneView) -> [CADViewportFrame.SeedChannel] {
        view.makeFrame(width: 640, height: 480).seeds.map(\.channel)
    }

    private func mesh() -> PreviewMesh {
        PreviewMesh(vertices: [[-10, -10, 0], [10, -10, 0], [10, 10, 0], [-10, 10, 0]],
                    triangles: [[0, 1, 2], [0, 2, 3]], edges: [[-10, -10, 0, 10, -10, 0]],
                    bounds: [-10, -10, 0, 10, 10, 0], volume: 1, area: 400, solids: 1,
                    buildSeconds: 0, threads: 1, buildID: "viewport-hover-points",
                    triangleSolids: [0, 0], edgeSolids: [0])
    }
}
