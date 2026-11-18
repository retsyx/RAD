import AppKit
import Metal
import SceneKit
import XCTest
@testable import RAD

final class ViewportNearestHitTests: XCTestCase {
    @MainActor
    func testLapelInteriorAlwaysPicksNearestOwnerAcrossNodeOrderAndProjection() async throws {
        let orders = [[0, 1, 2], [1, 0, 2], [2, 0, 1], [2, 1, 0]]
        let visibility: [Set<Int>] = [[], [2], [1]]
        for projection in ViewportProjectionMode.allCases {
            let preview = panels()
            let view = try await preparedView(preview, projection: projection)
            let nodes = try (0..<3).map { index in
                try XCTUnwrap(view.scene.rootNode.childNode(withName: "solid:\(index)", recursively: true))
            }
            let parent = try XCTUnwrap(nodes[0].parent)
            for order in orders {
                nodes.forEach { $0.removeFromParentNode() }
                order.forEach { parent.addChildNode(nodes[$0]) }
                SCNTransaction.flush()
                for hidden in visibility {
                    let existingPart = hidden.contains(2) ? 0 : 2
                    view.update(mesh: preview, mode: .edges, grid: false, command: nil,
                                selectedSolids: [existingPart], hiddenSolids: hidden, projection: projection)
                    SCNTransaction.flush()
                    let projected = view.projectPoint(SCNVector3(1.1, -0.15, 2.2))
                    let point = NSPoint(x: CGFloat(projected.x), y: CGFloat(projected.y))
                    XCTAssertTrue(view.bounds.contains(point))
                    let expected = hidden.contains(1) ? 0 : 1
                    let context = "projection=\(projection.rawValue), order=\(order), hidden=\(hidden.sorted())"
                    XCTAssertEqual(view.surfacePoint(at: point)?.solidIndex, expected,
                                   "Surface points come from the same nearest visible owner. \(context)")
                    var partSelections = [Set<Int>]()
                    view.onSelectionChanged = { partSelections.append($0) }
                    view.mouseDown(with: try click(at: point, modifiers: .option))
                    XCTAssertEqual(partSelections, [[existingPart, expected]],
                        "The visible lapel must win even when coat geometry was submitted first; hidden owners must still be filtered. \(context)")
                    view.mouseDown(with: try click(at: point, modifiers: .option))
                    XCTAssertEqual(partSelections, [[existingPart, expected], [existingPart]],
                                   "A second Option-click must deselect the owner. \(context)")
                }
            }
        }
    }

    private func click(at point: NSPoint, modifiers: NSEvent.ModifierFlags = []) throws -> NSEvent {
        try XCTUnwrap(NSEvent.mouseEvent(with: .leftMouseDown, location: point, modifierFlags: modifiers,
            timestamp: 0, windowNumber: 0, context: nil, eventNumber: 1, clickCount: 1, pressure: 1))
    }

    @MainActor
    private func preparedView(_ preview: PreviewMesh, projection: ViewportProjectionMode,
                              size: NSSize = NSSize(width: 640, height: 480)) async throws -> CADSceneView {
        _ = try XCTUnwrap(MTLCreateSystemDefaultDevice())
        let view = CADSceneView(frame: NSRect(origin: .zero, size: size))
        view.update(mesh: preview, mode: .edges, grid: false, command: ViewportCommand(.front),
                    selectedSolids: [], hiddenSolids: [], projection: projection)
        view.layout()
        let deadline = ContinuousClock.now + .seconds(10)
        while view.isPreparingResources, ContinuousClock.now < deadline { try await Task.sleep(for: .milliseconds(5)) }
        XCTAssertFalse(view.isPreparingResources)
        SCNTransaction.flush()
        return view
    }

    private func panels() -> PreviewMesh {
        // The coat lies behind a small inset lapel, matching the real design's
        // front (-Y) direction. A separate off-axis part exercises the branch
        // that must query all intersections when an unrelated part is hidden.
        let quads: [[[Double]]] = [
            [[-12,0,-10],[12,0,-10],[12,0,10],[-12,0,10]],
            [[-4,-0.15,-6],[4,-0.15,-6],[4,-0.15,6],[-4,-0.15,6]],
            [[28,0,-2],[32,0,-2],[32,0,2],[28,0,2]]
        ]
        var triangles = [[Int]](), edges = [[Double]](), triangleSolids = [Int](), edgeSolids = [Int]()
        var parts = [CADPart]()
        for (solid, quad) in quads.enumerated() {
            let base = solid * 4
            triangles += [[base, base + 1, base + 2], [base, base + 2, base + 3]]
            triangleSolids += [solid, solid]
            for index in 0..<4 {
                edges.append(quad[index] + quad[(index + 1) % 4])
                edgeSolids.append(solid)
            }
            let center = zip(quad[0], quad[2]).map { ($0 + $1) / 2 }
            parts.append(CADPart(solidIndex: solid, area: 1, volume: 1,
                                 bounds: [quad[0][0],quad[0][1],quad[0][2],quad[2][0],quad[2][1],quad[2][2]], centroid: center, triangles: 1, components: 1))
        }
        return PreviewMesh(vertices: quads.flatMap { $0 }, triangles: triangles, edges: edges,
            bounds: [-12,-0.15,-10,32,0,10], volume: 3, area: 3, solids: 3, buildSeconds: 0, threads: 1,
            buildID: "nearest-lapel-preview", triangleSolids: triangleSolids, edgeSolids: edgeSolids, partStats: parts)
    }
}
