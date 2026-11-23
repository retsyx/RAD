import XCTest
import simd
@testable import RAD

final class CADSurfacePatchTests: XCTestCase {
    func testWeldedCornersCreaseBarrierAndPartLocalFaceIndices() throws {
        // Two identical squares in overlapping parts. Each triangle has its own
        // shading vertices. Only the second part has a crease on its diagonal.
        let square = [[0.0,0,0], [1,0,0], [1,1,0], [0,0,0], [1,1,0], [0,1,0]]
        let preview = PreviewMesh(vertices: square + square,
            triangles: [[0,1,2], [6,7,8], [3,4,5], [9,10,11]], edges: [[0,0,0,1,1,0]],
            bounds: [0,0,0,1,1,0], volume: 0, area: 2, solids: 2,
            buildSeconds: 0, threads: 1, buildID: "patch-test", triangleSolids: [0,1,0,1], edgeSolids: [1])
        let prepared = try XCTUnwrap(try CADViewportPreparedMesh.prepareShared(preview))
        let patches = prepared.surfacePatches
        let squarePatch = try XCTUnwrap(patches.patch(solid: 0, faceIndex: 1))
        XCTAssertEqual(squarePatch, patches.patch(solid: 0, faceIndex: 0))
        XCTAssertEqual(squarePatch.area, 1, accuracy: 1e-12)
        XCTAssertEqual(squarePatch.centroid, [0.5, 0.5, 0])
        XCTAssertEqual(squarePatch.bounds, [0, 0, 0, 1, 1, 0])
        XCTAssertEqual(squarePatch.normal, [0, 0, 1])
        XCTAssertTrue(squarePatch.isValid)
        let first = try XCTUnwrap(patches.patch(solid: 1, faceIndex: 0))
        let second = try XCTUnwrap(patches.patch(solid: 1, faceIndex: 1))
        XCTAssertNotEqual(first, second)
        XCTAssertEqual(first.area, 0.5, accuracy: 1e-12)
        XCTAssertEqual(second.area, 0.5, accuracy: 1e-12)
        func indices(_ patch: CADSurfacePatch) throws -> [Int32] {
            let data = try XCTUnwrap(patches.triangleIndices(for: patch))
            return data.withUnsafeBytes { Array($0.bindMemory(to: Int32.self)) }
        }
        XCTAssertEqual(try indices(squarePatch), [0,1,2,3,4,5])
        XCTAssertEqual(try indices(first), [6,7,8])
        XCTAssertEqual(try indices(second), [9,10,11])
        XCTAssertNil(patches.patch(solid: 1, faceIndex: 2))
        XCTAssertEqual(try XCTUnwrap(CADViewportPreparedMesh.shared(for: preview)).surfacePatches.patch(solid: 1, faceIndex: 1), second)

        var ambiguous = preview
        ambiguous.edgeSolids = nil
        let unknownOwner = try XCTUnwrap(try CADViewportPreparedMesh.prepare(ambiguous))
        XCTAssertNil(unknownOwner.surfacePatches.patch(solid: 0, faceIndex: 0))
        XCTAssertNil(unknownOwner.surfacePatches.patch(solid: 1, faceIndex: 0))
    }

    func testNonManifoldSideAndPointOnlyContactDoNotBridgeTriangles() throws {
        let positions: [Float] = [0,0,0, 1,0,0, 0,1,0, 0,0,1, 1,1,1, -1,1,0, -1,0,0]
        let patches = try CADSurfacePatches(positions: positions, groups: [0: [0,1,2, 1,0,3, 0,1,4, 0,5,6, 0,0,1]],
                                            edgePoints: [], edgeGroups: [:])
        let selected = try (0..<4).map { face in try XCTUnwrap(patches.patch(solid: 0, faceIndex: face)) }
        XCTAssertEqual(Set(selected).count, 4)
        XCTAssertNil(patches.patch(solid: 0, faceIndex: 4))
    }

    func testWorkerPatchesOnSharpAndTangentSurfaces() async throws {
        guard ProcessInfo.processInfo.environment["RAD_RUN_ENGINE_TESTS"] == "1" else {
            throw XCTSkip("Set RAD_RUN_ENGINE_TESTS=1 to check patches on native SDF primitives.")
        }
        let spec = ModelSpec(title: "Surface patches", summary: "Sharp and tangent surfaces", units: "mm",
                             script: """
                             result = [box(10, center=(-16,0,0)), cylinder(5,10),
                                       sphere(5, center=(16,0,0)), box(10, radius=2, center=(32,0,0))]
                             """)
        let preview = try await EngineService().buildPreview(spec, python: EngineService.defaultPython, threads: 2)
        let prepared = try XCTUnwrap(try CADViewportPreparedMesh.prepare(preview))
        var all = [Set<CADSurfacePatch>]()
        for solid in 0..<4 {
            let count = try XCTUnwrap(prepared.groups[solid]).count / 3
            let patches = Set(try (0..<count).map { try XCTUnwrap(prepared.surfacePatches.patch(solid: solid, faceIndex: $0)) })
            XCTAssertTrue(patches.allSatisfy(\.isValid))
            all.append(patches)
        }
        XCTAssertEqual(all.map(\.count), [6, 3, 1, 1])
        let sphere = try XCTUnwrap(all[2].first)
        XCTAssertEqual(sphere.centroid[0], 16, accuracy: 0.05)
        XCTAssertEqual(simd_distance(SIMD3<Double>(sphere.point), SIMD3(16, 0, 0)), 5, accuracy: 0.1)
    }
}
