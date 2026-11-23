import AppKit
import Metal
import SceneKit
import XCTest
@testable import RAD

/// Point markers render through the unoccluded overlay in the live view and
/// in independent captures.
final class ViewportPointMarkerTests: XCTestCase {
    @MainActor
    func testHoveredAndSelectedPointMarkersRenderDistinctlyAndReuseTheirBuffer() async throws {
        let view = try makeView()
        let preview = plane()
        let colors = [0: PartColor(red: 0.38, green: 0.67, blue: 0.82)]
        view.update(mesh: preview, mode: .edges, grid: false, command: ViewportCommand(.top),
                    selectedSolids: [], hiddenSolids: [], projection: .isometric, solidColors: colors)
        view.layout()
        try await ready(view)
        let target = projected([2, 3, 0], in: view)
        view.mouseMoved(with: try mouse(.mouseMoved, at: target))
        let hovered = try XCTUnwrap(seeds(in: view).first)
        XCTAssertEqual(seeds(in: view).count, 1)
        XCTAssertEqual(hovered.channel, .hoveredVertex)
        XCTAssertEqual(hovered.buffer.primitive, .point)
        XCTAssertEqual(hovered.buffer.vertexCount, 1, "A screen-space circle needs only the point's center.")
        let hover = try await CADAgentRenderer.capturePNG(view: view)
        XCTAssertGreaterThan(try markerPixels(hover, around: target, in: view), 35,
                             "A cloned capture must contain the hovered marker, including its screen-size uniforms.")

        let picked = try XCTUnwrap(view.surfacePoint(at: target))
        view.update(mesh: preview, mode: .edges, grid: false, command: nil, selectedSolids: [], hiddenSolids: [],
                    projection: .isometric, solidColors: colors, selectedPoints: [picked])
        try await ready(view)
        XCTAssertTrue(seeds(in: view).first?.buffer === hovered.buffer, "Selecting the hovered point reuses its marker.")
        XCTAssertEqual(seeds(in: view).map(\.channel), [.selectedVertex])
        let selected = try await CADAgentRenderer.capturePNG(view: view)
        XCTAssertGreaterThan(try markerPixels(selected, around: target, in: view), 45)
        XCTAssertNotEqual(hover, selected, "Hover and selection must be visually distinguishable in a rendered image.")

        view.setFrameSize(NSSize(width: 800, height: 600))
        view.layout()
        XCTAssertEqual(view.makeFrame(width: 1_600, height: 1_200).referenceHeight, 600,
                       "Screen-point widths scale from the view's logical height to output pixels.")
        view.update(mesh: preview, mode: .edges, grid: false, command: nil, selectedSolids: [], hiddenSolids: [0],
                    projection: .isometric, solidColors: colors, selectedPoints: [picked])
        XCTAssertTrue(seeds(in: view).isEmpty)
        XCTAssertNil(view.surfacePoint(at: projected([2, 3, 0], in: view)),
                     "Hidden parts are excluded from picking as well as rendering.")
    }

    @MainActor
    private func seeds(in view: CADSceneView) -> [CADViewportFrame.Seeds] {
        view.makeFrame(width: Int(view.bounds.width), height: Int(view.bounds.height)).seeds
    }

    /// Gold pixels within 10 points of a view point in a captured PNG.
    @MainActor
    private func markerPixels(_ data: Data, around point: NSPoint, in view: CADSceneView) throws -> Int {
        let image = try XCTUnwrap(NSBitmapImageRep(data: data))
        let scale = CGFloat(image.pixelsWide) / view.bounds.width
        let centerX = Int((point.x * scale).rounded())
        let centerY = image.pixelsHigh - 1 - Int((point.y * scale).rounded())
        let radius = Int((10 * scale).rounded())
        var count = 0
        for y in max(0, centerY - radius)...min(image.pixelsHigh - 1, centerY + radius) {
            for x in max(0, centerX - radius)...min(image.pixelsWide - 1, centerX + radius) {
                let c = try XCTUnwrap(image.colorAt(x: x, y: y)?.usingColorSpace(.sRGB))
                if c.redComponent > 0.5, c.redComponent > c.blueComponent + 0.12,
                   c.greenComponent > c.blueComponent + 0.08 { count += 1 }
            }
        }
        return count
    }

    @MainActor
    private func makeView() throws -> CADSceneView {
        _ = try XCTUnwrap(MTLCreateSystemDefaultDevice())
        return CADSceneView(frame: NSRect(x: 0, y: 0, width: 512, height: 512))
    }

    @MainActor
    private func ready(_ view: CADSceneView) async throws {
        let deadline = ContinuousClock.now + .seconds(10)
        while view.isPreparingResources, ContinuousClock.now < deadline { try await Task.sleep(for: .milliseconds(5)) }
        XCTAssertFalse(view.isPreparingResources)
    }

    @MainActor
    private func projected(_ position: [Double], in view: CADSceneView) -> NSPoint {
        let point = view.projectPoint(SCNVector3(position[0], position[1], position[2]))
        return NSPoint(x: CGFloat(point.x), y: CGFloat(point.y))
    }

    @MainActor
    private func mouse(_ type: NSEvent.EventType, at point: NSPoint) throws -> NSEvent {
        try XCTUnwrap(NSEvent.mouseEvent(with: type, location: point, modifierFlags: .shift, timestamp: 0,
            windowNumber: 0, context: nil, eventNumber: 1, clickCount: 0, pressure: 0))
    }

    private func plane() -> PreviewMesh {
        PreviewMesh(vertices: [[-10, -10, 0], [10, -10, 0], [10, 10, 0], [-10, 10, 0]],
                    triangles: [[0, 1, 2], [0, 2, 3]], edges: [[-10, -10, 0, 10, -10, 0]], bounds: [-10, -10, 0, 10, 10, 0],
                    volume: 0, area: 400, solids: 1, buildSeconds: 0, threads: 1,
                    buildID: "viewport-point-markers", triangleSolids: [0, 0], edgeSolids: [0])
    }
}
