import XCTest
import simd
@testable import RAD

final class CADCreasePickingTests: XCTestCase {
    func testCornerAndBranchEndCurvesAndReversedDuplicateIsIgnored() throws {
        // Straight from 0 to 2 via 1, then a right angle up to a
        // three-way junction. The duplicated first segment is reversed.
        let coordinates: [Float] = [
            1,0,0, 0,0,0, 2,0,0, 1,0,0, 2,0,0, 2,1,0,
            2,2,0, 2,1,0, 2,1,0, 3,1,0, 0,0,0, 1,0,0
        ]
        let edges = try CADCreaseEdge.group(points: coordinates, segments: Array(0..<6), solid: 4)
        XCTAssertEqual(edges.count, 4)
        XCTAssertEqual(edges.map(\.length).sorted(), [1, 1, 1, 2])
        XCTAssertEqual(edges.map(\.index), [0, 1, 2, 3])
        XCTAssertTrue(edges.allSatisfy { $0.solidIndex == 4 && $0.isValid })
        let longest = try XCTUnwrap(edges.max { $0.length < $1.length })
        XCTAssertEqual(longest.points[1], [1, 0, 0])
    }

    func testClosedRimKeepsAllSegmentsAcrossShuffledDirections() throws {
        let count = 24
        let points = (0..<count).map { i in
            let angle = Float(i) * 2 * .pi / Float(count)
            return [cos(angle), sin(angle), 0]
        }
        var segments = [Float]()
        for n in 0..<count {
            let a = (n * 5) % count, b = (a + 1) % count
            segments += points[n.isMultiple(of: 2) ? b : a] + points[n.isMultiple(of: 2) ? a : b]
        }
        let edges = try CADCreaseEdge.group(points: segments, segments: Array(0..<count), solid: 2)
        XCTAssertEqual(edges.count, 1)
        let rim = try XCTUnwrap(edges.first)
        XCTAssertEqual(rim.points.count, count + 1)
        XCTAssertEqual(rim.points.first, rim.points.last)
        XCTAssertEqual(rim.length, 2 * Double(count) * sin(.pi / Double(count)), accuracy: 1e-5)
    }

    private func camera(orthographic: Bool) -> CameraRig.Projection {
        CameraRig.Projection(position: .zero, right: SIMD3(1,0,0), up: SIMD3(0,1,0), backward: SIMD3(0,0,1),
                             tangent: 1, orthographic: orthographic, orthographicScale: 1, width: 200, height: 200)
    }

    func testPerspectiveDepthAndVisibilityCheckedAtClosestPointOnSegment() throws {
        let sloping = CADCreaseEdge(solidIndex: 0, index: 0, points: [[-1, 0, -2], [5, 0, -10]])
        let front = CADCreaseEdge(solidIndex: 1, index: 0, points: [[-0.5, 0, -1], [0.5, 0, -1]])
        let projection = CADCreaseProjection(edges: [sloping, front], projection: camera(orthographic: false),
                                             placements: [:], hidden: [])
        var checked = [Int]()
        let edge = projection.pick(at: SIMD2(100, 103), radius: 4) { edge, screen, depth in
            checked.append(edge.solidIndex)
            XCTAssertEqual(screen, SIMD2(100, 100))
            if edge == front {
                XCTAssertEqual(depth, 1)
                return false
            }
            XCTAssertEqual(depth, 10.0 / 3, accuracy: 1e-7)
            return true
        }
        XCTAssertEqual(checked, [1, 0])
        XCTAssertEqual(edge, sloping)
        XCTAssertNil(projection.pick(at: SIMD2(100, 105), radius: 4) { _, _, _ in true })
    }

    func testPlacementVisibilityAndEyePlaneClipping() throws {
        let line = [[-1.0, 0, 1], [1, 0, -3]]
        let edge = CADCreaseEdge(solidIndex: 0, index: 0, points: line)
        let hidden = CADCreaseEdge(solidIndex: 1, index: 0, points: line)
        var placement = matrix_identity_double4x4
        placement.columns.3.y = 0.5
        let projection = CADCreaseProjection(edges: [hidden, edge], projection: camera(orthographic: true),
                                             placements: [0: placement], hidden: [1])
        XCTAssertNil(projection.pick(at: SIMD2(150, 100), radius: 3) { _, _, _ in true })
        XCTAssertNil(projection.pick(at: SIMD2(20, 150), radius: 3) { _, _, _ in true })
        XCTAssertEqual(projection.pick(at: SIMD2(150, 150), radius: 3) { _, screen, depth in
            XCTAssertEqual(screen.x, 150, accuracy: 1e-8)
            XCTAssertEqual(screen.y, 150, accuracy: 1e-8)
            XCTAssertEqual(depth, 2, accuracy: 1e-8)
            return true
        }, edge)

        let perspective = CADCreaseProjection(edges: [edge], projection: camera(orthographic: false), placements: [:], hidden: [])
        XCTAssertEqual(perspective.pick(at: SIMD2(10, 100), radius: 2) { _, _, depth in depth > 0 }, edge)
        XCTAssertNil(perspective.pick(at: SIMD2(180, 100), radius: 2) { _, _, _ in true })
    }
}
