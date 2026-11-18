import AppKit
import Metal
import SceneKit
import simd
import XCTest
@testable import RAD

final class ViewportAssemblyMotionTests: XCTestCase {
    private let moved: [Double] = [0, -1, 0, 3, 1, 0, 0, -2, 0, 0, 1, 0, 0, 0, 0, 1]

    func testRigidBoundaryRejectsScaleReflectionAndNonfiniteMatrices() throws {
        let matrix = try XCTUnwrap(ViewportRigidTransform.matrix(moved))
        XCTAssertEqual(matrix * SIMD4(2, -4, 0, 1), SIMD4(7, 0, 0, 1))
        XCTAssertEqual(ViewportRigidTransform.bounds([2, -4, 0, 10, 4, 0], matrix: matrix), [-1, 0, 0, 7, 8, 0])
        let quarterTurn = [0.0, -1, 0, 3, 1, 0, 0, -2, 0, 0, 1, 1, 0, 0, 0, 1]
        XCTAssertEqual(ViewportRigidTransform.bounds([2, -4, 0, 10, 4, 0],
            matrix: try XCTUnwrap(ViewportRigidTransform.matrix(quarterTurn))), [-1, 0, 1, 7, 8, 1])
        for (index, value) in [(0, 2.0), (0, -1.0), (3, Double.nan), (12, 1.0), (15, 0.0)] {
            var invalid = moved
            invalid[index] = value
            XCTAssertNil(ViewportRigidTransform.matrix(invalid))
        }
        XCTAssertNil(ViewportRigidTransform.matrix(Array(moved.dropLast())))
    }

    @MainActor
    func testMotionPreservesPreparedBuffersCameraAndHighlightPoses() async throws {
        let view = try makeView()
        let preview = mesh()
        view.update(mesh: preview, mode: .edges, grid: false, command: ViewportCommand(.top),
                    selectedSolids: [1], hiddenSolids: [])
        try await ready(view)
        let surface = try node("solid:1", in: view)
        let ordinaryEdges = try node("edges:1", in: view)
        let geometry = try XCTUnwrap(surface.geometry)
        let material = try XCTUnwrap(geometry.firstMaterial)
        let edgeGeometry = ordinaryEdges.geometry
        let camera = try XCTUnwrap(view.pointOfView).simdTransform
        view.update(mesh: preview, mode: .edges, grid: false, command: nil,
                    selectedSolids: [1], hiddenSolids: [], solidTransforms: [1: moved])
        XCTAssertFalse(view.isPreparingResources, "Moving rigid parts must reuse their uploaded drawing resources.")
        XCTAssertTrue(surface.geometry === geometry)
        XCTAssertTrue(surface.geometry?.firstMaterial === material)
        XCTAssertTrue(ordinaryEdges.geometry === edgeGeometry)
        XCTAssertEqual(try XCTUnwrap(view.pointOfView).simdTransform, camera)
        for node in [surface, ordinaryEdges] {
            XCTAssertEqual(node.simdTransform.columns.3, SIMD4(3, -2, 0, 1))
        }
        let masks = view.makeFrame(width: 640, height: 480).batches.flatMap(\.masks)
        XCTAssertEqual(masks.count, 1)
        for mask in masks {
            XCTAssertEqual(mask.model.columns.3, SIMD4(3, -2, 0, 1), "Part outline masks must use the body's current pose.")
        }
        XCTAssertEqual(try node("solid:0", in: view).simdTransform, matrix_identity_float4x4)
        view.update(mesh: preview, mode: .edges, grid: false, command: nil, selectedSolids: [1], hiddenSolids: [])
        XCTAssertEqual(surface.simdTransform, matrix_identity_float4x4, "Clearing a pose restores the built geometry.")
        XCTAssertTrue(surface.geometry === geometry)
    }

    @MainActor
    func testPickingAndIndependentCaptureFollowMotionInBothProjections() async throws {
        for projection in ViewportProjectionMode.allCases {
            let view = try makeView()
            let preview = mesh()
            view.update(mesh: preview, mode: .edges, grid: false, command: ViewportCommand(.top),
                        selectedSolids: [], hiddenSolids: [], projection: projection)
            try await ready(view)
            func picked(_ point: [Double]) throws -> Set<Int>? {
                var selected: Set<Int>?
                view.onSelectionChanged = { selected = $0 }
                let location = view.convert(projected(point, in: view), to: nil)
                view.mouseDown(with: try XCTUnwrap(NSEvent.mouseEvent(with: .leftMouseDown, location: location,
                    modifierFlags: .option, timestamp: 0, windowNumber: 0, context: nil, eventNumber: 1, clickCount: 1, pressure: 1)))
                return selected
            }
            XCTAssertEqual(try picked([6, -2, 0]), [1])
            XCTAssertNil(try picked([0, 6, 0]))
            let before = try await CADAgentRenderer.capturePNG(view: view)
            view.update(mesh: preview, mode: .edges, grid: false, command: nil,
                        selectedSolids: [], hiddenSolids: [], projection: projection, solidTransforms: [1: moved])
            XCTAssertNil(try picked([6, -2, 0]))
            XCTAssertEqual(try picked([0, 6, 0]), [1])
            let captured = try await CADAgentRenderer.captureWithMetadata(view: view)
            XCTAssertNotEqual(captured.png, before, "A detached capture must clone the current body poses.")
            let poses = try XCTUnwrap(captured.metadata["solidTransforms"] as? [[String: Any]])
            XCTAssertEqual(poses.count, 1)
            XCTAssertEqual(poses[0]["solid"] as? Int, 1)
            XCTAssertEqual(poses[0]["matrix"] as? [Double], moved)
            view.update(mesh: preview, mode: .edges, grid: false, command: nil,
                        selectedSolids: [], hiddenSolids: [1], projection: projection, solidTransforms: [1: moved])
            XCTAssertNil(try picked([0, 6, 0]))
        }
    }

    @MainActor
    func testIndependentPosedRenderFitsTheVisiblePart() async throws {
        var preview = mesh()
        preview.assemblyPose = AssemblyPose(positions: [:], solidTransforms: [1: moved])
        let captured = try await CADAgentRenderer.renderWithMetadata(preview: preview, view: "top",
            projection: .isometric, hiddenSolids: [0], width: 320, height: 240)
        let target = try XCTUnwrap(captured.metadata["cameraTarget"] as? [Double])
        XCTAssertEqual(target, [3, 4, 0], "Independent images should fit the visible posed body, excluding hidden bodies.")
    }

    @MainActor
    func testOmittedSolidsAreNeitherDrawnPickedNorFramed() async throws {
        let view = try makeView()
        let preview = mesh()
        view.update(mesh: preview, mode: .edges, grid: false, command: ViewportCommand(.top),
                    selectedSolids: [], hiddenSolids: [])
        try await ready(view)
        func target() throws -> [Double] { try XCTUnwrap(view.agentCaptureMetadata["cameraTarget"] as? [Double]) }
        // Hidden parts stay in the framed bounds; omitted ones, such as reference
        // parts on the printing plate, do not.
        view.update(mesh: preview, mode: .edges, grid: false, command: ViewportCommand(.fit),
                    selectedSolids: [], hiddenSolids: [0], solidTransforms: [1: moved])
        XCTAssertEqual(try target()[0], -1.5, accuracy: 1e-9)
        XCTAssertEqual(try target()[1], 2, accuracy: 1e-9)
        view.update(mesh: preview, mode: .edges, grid: false, command: ViewportCommand(.fit),
                    selectedSolids: [], hiddenSolids: [], omittedSolids: [0], solidTransforms: [1: moved])
        XCTAssertEqual(try target()[0], 3, accuracy: 1e-9)
        XCTAssertEqual(try target()[1], 4, accuracy: 1e-9)
        XCTAssertTrue(try node("solid:0", in: view).isHidden)
        XCTAssertFalse(try node("solid:1", in: view).isHidden)
        XCTAssertEqual(view.agentCaptureMetadata["hiddenSolidIndices"] as? [Int], [0])
        var selected: Set<Int>?
        view.onSelectionChanged = { selected = $0 }
        let location = view.convert(projected([-6, 0, 0], in: view), to: nil)
        view.mouseDown(with: try XCTUnwrap(NSEvent.mouseEvent(with: .leftMouseDown, location: location,
            modifierFlags: .option, timestamp: 0, windowNumber: 0, context: nil, eventNumber: 1, clickCount: 1, pressure: 1)))
        XCTAssertNil(selected, "Omitted solids cannot be picked")
        let rightClick = try XCTUnwrap(NSEvent.mouseEvent(with: .rightMouseDown,
            location: NSPoint(x: view.bounds.midX, y: view.bounds.midY), modifierFlags: [], timestamp: 0,
            windowNumber: 0, context: nil, eventNumber: 2, clickCount: 1, pressure: 1))
        XCTAssertNil(view.menu(for: rightClick)?.item(withTitle: "Show All Parts"),
                     "Omitted solids are not hidden parts the user can show")
        view.update(mesh: preview, mode: .edges, grid: false, command: nil,
                    selectedSolids: [], hiddenSolids: [1], omittedSolids: [0], solidTransforms: [1: moved])
        XCTAssertNotNil(view.menu(for: rightClick)?.item(withTitle: "Show All Parts"))
        // Leaving the omission frames and shows every part again.
        view.update(mesh: preview, mode: .edges, grid: false, command: ViewportCommand(.fit),
                    selectedSolids: [], hiddenSolids: [])
        XCTAssertEqual(try target()[0], 0, accuracy: 1e-9)
        XCTAssertEqual(try target()[1], 0, accuracy: 1e-9)
        XCTAssertFalse(try node("solid:0", in: view).isHidden)
    }

    @MainActor
    func testMotionAfterIdleProducesFreshLiveFramesAndReturnsToIdle() async throws {
        guard !NSScreen.screens.isEmpty else { throw XCTSkip("A window server is required for live assembly motion.") }
        let app = NSApplication.shared
        let policy = app.activationPolicy()
        _ = app.setActivationPolicy(.regular)
        defer { _ = app.setActivationPolicy(policy) }
        app.finishLaunching()
        let view = try makeView()
        let frames = ViewportRenderCounter(view)
        let window = NSWindow(contentRect: view.frame, styleMask: [.titled], backing: .buffered, defer: false)
        window.title = "RAD assembly motion test"
        window.isReleasedWhenClosed = false
        window.contentView = view
        defer {
            window.orderOut(nil)
            window.contentView = nil
            window.close()
        }
        let preview = mesh()
        view.update(mesh: preview, mode: .edges, grid: false, command: ViewportCommand(.top),
                    selectedSolids: [], hiddenSolids: [])
        app.activate(ignoringOtherApps: true)
        window.makeKeyAndOrderFront(nil)
        window.displayIfNeeded()
        try await ready(view)
        try await waitUntil { frames.count > 0 }
        try await idle(view, frames: frames)
        let previous = frames.count
        let camera = try XCTUnwrap(view.pointOfView).simdTransform
        view.update(mesh: preview, mode: .edges, grid: false, command: nil,
                    selectedSolids: [], hiddenSolids: [], solidTransforms: [1: moved])
        try await waitUntil { frames.count > previous }
        try await idle(view, frames: frames)
        XCTAssertEqual(try XCTUnwrap(view.pointOfView).simdTransform, camera)
        XCTAssertFalse(view.hasPendingFrames)
        let idleCount = frames.count
        view.update(mesh: preview, mode: .edges, grid: false, command: nil,
                    selectedSolids: [], hiddenSolids: [], solidTransforms: [1: moved])
        try await Task.sleep(for: .milliseconds(100))
        XCTAssertFalse(view.hasPendingFrames, "An unchanged pose must not draw again.")
        XCTAssertEqual(frames.count, idleCount)
    }

    @MainActor
    private func makeView() throws -> CADSceneView {
        _ = try XCTUnwrap(MTLCreateSystemDefaultDevice())
        return CADSceneView(frame: NSRect(x: 50, y: 50, width: 640, height: 480))
    }

    @MainActor
    private func ready(_ view: CADSceneView) async throws {
        try await waitUntil { !view.isPreparingResources }
    }

    @MainActor
    private func idle(_ view: CADSceneView, frames: ViewportRenderCounter) async throws {
        let deadline = ContinuousClock.now + .seconds(10)
        var previous = frames.count
        var lastChange = ContinuousClock.now
        while ContinuousClock.now < deadline {
            try await Task.sleep(for: .milliseconds(25))
            let current = frames.count
            if current != previous { previous = current; lastChange = ContinuousClock.now }
            if !view.hasPendingFrames, ContinuousClock.now - lastChange >= .milliseconds(250) { return }
        }
        XCTFail("The moving assembly viewport should become idle after rendering its latest pose.")
    }

    @MainActor
    private func waitUntil(_ predicate: () -> Bool) async throws {
        let deadline = ContinuousClock.now + .seconds(10)
        while !predicate(), ContinuousClock.now < deadline { try await Task.sleep(for: .milliseconds(10)) }
        XCTAssertTrue(predicate())
    }

    @MainActor
    private func projected(_ point: [Double], in view: CADSceneView) -> NSPoint {
        let value = view.projectPoint(SCNVector3(point[0], point[1], point[2]))
        return NSPoint(x: CGFloat(value.x), y: CGFloat(value.y))
    }

    @MainActor
    private func node(_ name: String, in view: CADSceneView) throws -> SCNNode {
        try XCTUnwrap(view.scene.rootNode.childNode(withName: name, recursively: true))
    }

    private func mesh() -> PreviewMesh {
        PreviewMesh(vertices: [[-10, -4, 0], [-2, -4, 0], [-2, 4, 0], [-10, 4, 0],
                                           [2, -4, 0], [10, -4, 0], [10, 4, 0], [2, 4, 0]],
            triangles: [[0, 1, 2], [0, 2, 3], [4, 5, 6], [4, 6, 7]],
            edges: [[-10, -4, 0, -2, -4, 0], [2, -4, 0, 10, -4, 0]],
            bounds: [-10, -4, 0, 10, 4, 0], volume: 1, area: 128, solids: 2,
            buildSeconds: 0, threads: 1, buildID: "viewport-assembly-motion", triangleSolids: [0, 0, 1, 1], edgeSolids: [0, 1])
    }
}


