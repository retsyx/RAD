import Foundation
import XCTest
@testable import RAD

final class CADMeasurementTests: XCTestCase {
    func testSelectedPartTotalsUseOriginalCoordinatesAndVolumeWeightedCentroid() throws {
        var mesh = preview()
        mesh.solids = 2
        mesh.partStats?.append(CADPart(solidIndex: 1, area: 6, volume: 1, bounds: [10, 0, 0, 11, 1, 1], centroid: [10.5, 0.5, 0.5],
                                       triangles: 12, components: 1))
        let report = try CADMeasurements.report(mesh, solids: [0, 1])
        XCTAssertEqual(report.solidIndices, [0, 1])
        XCTAssertEqual(try values(report, "parts:count"), [2])
        XCTAssertEqual(try values(report, "parts:volume"), [61])
        XCTAssertEqual(try values(report, "parts:area"), [100])
        XCTAssertEqual(try values(report, "parts:size"), [11, 4, 5])
        let centroid = try values(report, "parts:centroid")
        XCTAssertEqual(centroid[0], (1.5 * 60 + 10.5) / 61, accuracy: 1e-9)
        XCTAssertEqual(centroid[1], (2 * 60 + 0.5) / 61, accuracy: 1e-9)
        XCTAssertEqual(centroid[2], (2.5 * 60 + 0.5) / 61, accuracy: 1e-9)
    }

    func testPosedPartsMoveTheirCentroidAndEncloseTheirTransformedBounds() {
        let part = CADPart(solidIndex: 0, area: 94, volume: 60, bounds: [0, 0, 0, 3, 4, 5], centroid: [1.5, 2, 2.5],
                           triangles: 12, components: 1)
        let k = 0.5.squareRoot()
        // 45° about Z, then up by 10.
        let matrix = [k, -k, 0, 0, k, k, 0, 0, 0, 0, 1, 10, 0, 0, 0, 1]
        let posed = CADMeasurements.posedParts([part], transforms: [0: matrix])[0]
        XCTAssertEqual(posed.volume, 60)
        XCTAssertEqual(posed.area, 94)
        for (actual, expected) in zip(posed.centroid, [(1.5 - 2) * k, (1.5 + 2) * k, 12.5]) { XCTAssertEqual(actual, expected, accuracy: 1e-12) }
        for (actual, expected) in zip(posed.bounds, [-4 * k, 0, 10, 3 * k, 7 * k, 15]) { XCTAssertEqual(actual, expected, accuracy: 1e-12) }
        XCTAssertEqual(CADMeasurements.posedParts([part], transforms: [1: matrix]), [part], "Other parts' poses do not move it.")
    }

    func testInvalidSelectionsMissingStatisticsAndZeroVolumeNeverProduceMeasurements() throws {
        XCTAssertThrowsError(try CADMeasurements.report(preview(), solids: [1]))
        XCTAssertThrowsError(try CADMeasurements.report(preview(), solids: []))
        var mesh = preview()
        mesh.partStats?[0].volume = 0
        XCTAssertThrowsError(try CADMeasurements.report(mesh, solids: [0]))
        mesh.partStats = nil
        XCTAssertThrowsError(try CADMeasurements.report(mesh, solids: [0]))
    }

    func testOnePointReportsItsPosedCoordinatesAndNormal() throws {
        let point = CADSurfacePoint(solidIndex: 0, point: [1, 2, 3], normal: [1, 0, 0])
        let built = try CADMeasurements.report(preview(), points: [point], solids: [])
        XCTAssertEqual(built.title, "Surface point")
        XCTAssertEqual(built.points, [point])
        XCTAssertEqual(built.solidIndices, [])
        XCTAssertEqual(built.metrics.map(\.id), ["point1:position", "point1:normal"])
        XCTAssertEqual(built.metrics.map(\.label), ["Coordinates X, Y, Z", "Normal"])
        XCTAssertEqual(try values(built, "point1:position"), [1, 2, 3])
        XCTAssertEqual(try values(built, "point1:normal"), [1, 0, 0])

        // A quarter turn about Z, then up by 10: coordinates move, the normal turns.
        let posed = try CADMeasurements.report(preview(), points: [point], solids: [],
                                               solidTransforms: [0: [0, -1, 0, 0, 1, 0, 0, 0, 0, 0, 1, 10, 0, 0, 0, 1]])
        XCTAssertEqual(try values(posed, "point1:position"), [-2, 1, 13])
        XCTAssertEqual(try values(posed, "point1:normal"), [0, 1, 0])
        XCTAssertNil(posed.closestPoints)
    }

    func testTwoPointsReportDistanceAndDeltasAndThreePointsTheirTriangle() throws {
        let a = CADSurfacePoint(solidIndex: 0, point: [3, 0, 0], normal: [0, 0, 1])
        let b = CADSurfacePoint(solidIndex: 0, point: [0, 0, 0], normal: [0, 0, 1])
        let c = CADSurfacePoint(solidIndex: 0, point: [0, 4, 0], normal: [0, 0, 1])
        let pair = try CADMeasurements.report(preview(), points: [a, b], solids: [])
        XCTAssertEqual(pair.title, "Measurements")
        XCTAssertEqual(try values(pair, "pair:distance"), [3])
        XCTAssertEqual(try values(pair, "pair:delta"), [-3, 0, 0])
        XCTAssertEqual(pair.closestPoints, [[3, 0, 0], [0, 0, 0]])
        XCTAssertEqual(pair.metrics.first { $0.id == "point2:position" }?.label, "Point 2 · Coordinates X, Y, Z")

        let triangle = try CADMeasurements.report(preview(), points: [a, b, c], solids: [])
        XCTAssertEqual(try values(triangle, "triangle:angle")[0], 90, accuracy: 1e-12)
        XCTAssertEqual(try values(triangle, "triangle:area"), [6])
        XCTAssertEqual(try values(triangle, "triangle:perimeter"), [12])
        XCTAssertEqual(try values(triangle, "triangle:sides"), [3, 4, 5])
        XCTAssertNil(triangle.metrics.first { $0.id.hasPrefix("pair:") })

        let four = try CADMeasurements.report(preview(), points: [a, b, c, CADSurfacePoint(solidIndex: 0, point: [1, 1, 0], normal: [0, 0, 1])],
                                              solids: [])
        XCTAssertEqual(four.metrics.filter { $0.id.hasSuffix(":position") }.count, 4)
        XCTAssertNil(four.metrics.first { $0.id.hasPrefix("triangle:") || $0.id.hasPrefix("pair:") })
    }

    func testPointsAndPartsCombineAndOnlyPartPairsUseTheWorker() throws {
        let point = CADSurfacePoint(solidIndex: 0, point: [1, 2, 3], normal: [0, 0, 1])
        let report = try CADMeasurements.report(preview(), points: [point], solids: [0])
        XCTAssertEqual(try values(report, "parts:volume"), [60])
        XCTAssertEqual(report.metrics.first { $0.id == "point1:position" }?.label, "Point 1 · Coordinates X, Y, Z")
        XCTAssertEqual(CADMeasurements.workerEntities(points: [point], solids: [0]),
                       [.point(solid: 0, point: [1, 2, 3]), .part(solid: 0)])
        XCTAssertEqual(CADMeasurements.workerEntities(points: [], solids: [3, 1]), [.part(solid: 1), .part(solid: 3)])
        XCTAssertNil(CADMeasurements.workerEntities(points: [point, point], solids: []), "Point pairs are measured locally.")
        XCTAssertNil(CADMeasurements.workerEntities(points: [point], solids: []))
        XCTAssertNil(CADMeasurements.workerEntities(points: [point], solids: [0, 1]))
    }

    func testInvalidPointsNeverProduceMeasurements() {
        let valid = CADSurfacePoint(solidIndex: 0, point: [1, 2, 3], normal: [0, 0, 1])
        XCTAssertThrowsError(try CADMeasurements.report(preview(), points: [], solids: []))
        XCTAssertThrowsError(try CADMeasurements.report(preview(), points: [valid, valid], solids: []), "Duplicate points")
        for invalid in [CADSurfacePoint(solidIndex: 1, point: [1, 2, 3], normal: [0, 0, 1]),
                        CADSurfacePoint(solidIndex: 0, point: [1, .nan, 3], normal: [0, 0, 1]),
                        CADSurfacePoint(solidIndex: 0, point: [1, 2], normal: [0, 0, 1]),
                        CADSurfacePoint(solidIndex: 0, point: [2e6, 0, 0], normal: [0, 0, 1]),
                        CADSurfacePoint(solidIndex: 0, point: [1, 2, 3], normal: [0, 0, 2])] {
            XCTAssertThrowsError(try CADMeasurements.report(preview(), points: [invalid], solids: []), "\(invalid)")
        }
    }

    func testSurfacePointsEncodeWithTheWorkerPointShape() throws {
        let point = CADSurfacePoint(solidIndex: 2, point: [1, -0.5, 3], normal: [0, -1, 0])
        let object = try XCTUnwrap(JSONSerialization.jsonObject(with: JSONEncoder().encode(point)) as? [String: Any])
        XCTAssertEqual(Set(object.keys), ["solid", "point", "normal"])
        XCTAssertEqual(try JSONDecoder().decode(CADSurfacePoint.self, from: Data(#"{"solid":2,"point":[1,-0.5,3],"normal":[0,-1,0]}"#.utf8)),
                       point)
    }

    func testPreviewsAndMeasurementsAreNeverEmbeddedInProjectArchives() throws {
        let revision = ProjectRevision(model: .bracket, preview: preview())
        let project = RADProject(messages: [], revisions: [revision], selectedRevisionID: revision.id, projectID: UUID())
        let compressed = try ProjectStore.encode(project)
        let manifest = String(decoding: try ProjectZIP.decode(compressed), as: UTF8.self)
        for forbidden in ["partStats", "partRoots", "buildID", "artifactPath", "closestPoints", "meshBuffers"] {
            XCTAssertFalse(manifest.contains("\"\(forbidden)\""), forbidden)
        }
        XCTAssertNil(try ProjectStore.decode(compressed).revisions[0].preview)
    }

    func testMeasurementFormattingKeepsAreaVolumeAnglesAndVectorsDistinct() {
        XCTAssertEqual(CADMeasurementMetric(id: "area", label: "Area", values: [12.5], unit: .area).formattedValue, "12.5 mm²")
        XCTAssertEqual(CADMeasurementMetric(id: "volume", label: "Volume", values: [6], unit: .volume).formattedValue, "6 mm³")
        XCTAssertEqual(CADMeasurementMetric(id: "angle", label: "Angle", values: [90], unit: .angle).formattedValue, "90°")
        XCTAssertEqual(CADMeasurementMetric(id: "vector", label: "Point", values: [-0.00000001, 3, 4], unit: .length).formattedValue, "0, 3, 4 mm")
    }

    private func values(_ report: CADMeasurementReport, _ key: String) throws -> [Double] {
        try XCTUnwrap(report.metrics.first { $0.id == key }, key).values
    }

    private func preview() -> PreviewMesh {
        PreviewMesh(vertices: [[0, 0, 0], [3, 0, 0], [3, 4, 0]], triangles: [[0, 1, 2]], edges: [[0, 0, 0, 3, 0, 0]],
                    bounds: [-2, -2, 0, 5, 5, 5], volume: 60, area: 94, solids: 1,
                    buildSeconds: 0, threads: 1, buildID: UUID().uuidString, triangleSolids: [0], edgeSolids: [0],
                    partStats: [CADPart(solidIndex: 0, area: 94, volume: 60, bounds: [0, 0, 0, 3, 4, 5], centroid: [1.5, 2, 2.5],
                                        triangles: 1, components: 1)])
    }
}
