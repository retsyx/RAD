import Foundation
import XCTest
@testable import RAD

final class EngineMeasurementTests: XCTestCase {
    @MainActor
    func testPairMeasurementAndSelectedExportUseTheSessionGraph() async throws {
        guard ProcessInfo.processInfo.environment["RAD_RUN_ENGINE_TESTS"] == "1" else {
            throw XCTSkip("Set RAD_RUN_ENGINE_TESTS=1 to verify native measurement and selected STL export.")
        }
        let service = EngineService()
        let spec = ModelSpec(title: "Measurement assembly", summary: "A sphere and a separated block.", units: "mm", script: """
        result = Assembly("Assembly", id="assembly")
        result.add(sphere(2), id="sphere", name="Sphere")
        result.add(box(1, center=(10.5, 0.5, 0.5)), id="block", name="Block")
        """)
        let preview = try await service.buildPreview(spec, python: EngineService.defaultPython, threads: 4)
        let graph = try XCTUnwrap(preview.artifactPath)
        let report = try await service.measure(preview, solids: [0, 1], python: EngineService.defaultPython, threads: 4)
        XCTAssertEqual(try XCTUnwrap(report.metrics.first { $0.id == "pair:distance" }?.values.first), 8, accuracy: 0.01)
        XCTAssertEqual(report.closestPoints?.count, 2)
        XCTAssertEqual(try XCTUnwrap(report.metrics.first { $0.id == "parts:count" }?.values.first), 2)
        let point = try await service.measure(preview, entities: [.point(solid: 1, point: [10, 0.5, 0.5]), .part(solid: 0)],
                                              solidTransforms: nil, python: EngineService.defaultPython, threads: 4)
        XCTAssertEqual(try XCTUnwrap(point.metrics.first { $0.id == "pair:distance" }?.values.first), 100.5.squareRoot() - 2, accuracy: 0.05)
        let snappedPoint = try XCTUnwrap(point.points?.first)
        XCTAssertEqual(point.points?.count, 1)
        XCTAssertEqual(snappedPoint.solidIndex, 1)
        XCTAssertEqual(snappedPoint.point[0], 10, accuracy: 0.01)
        XCTAssertEqual(snappedPoint.normal[0], -1, accuracy: 0.01)

        // A display-mesh pick just inside the block's face: the worker snaps it
        // to the surface, and the report shows the snapped point and normal.
        let picked = CADSurfacePoint(solidIndex: 1, point: [10.02, 0.5, 0.5], normal: [-1, 0, 0])
        let combined = try await service.measure(preview, points: [picked], solids: [0], python: EngineService.defaultPython, threads: 4)
        XCTAssertEqual(combined.points, [picked])
        let position = try XCTUnwrap(combined.metrics.first { $0.id == "point1:position" }?.values)
        XCTAssertEqual(position[0], 10, accuracy: 0.01)
        XCTAssertEqual(position[1], 0.5, accuracy: 0.01)
        let normal = try XCTUnwrap(combined.metrics.first { $0.id == "point1:normal" }?.values)
        XCTAssertEqual(normal[0], -1, accuracy: 0.01)
        XCTAssertEqual(try XCTUnwrap(combined.metrics.first { $0.id == "pair:distance" }?.values.first), 100.5.squareRoot() - 2, accuracy: 0.05)
        XCTAssertEqual(try XCTUnwrap(combined.metrics.first { $0.id == "parts:count" }?.values.first), 1)
        XCTAssertEqual(combined.closestPoints?.count, 2)
        let selected = try await service.exportSTL(preview, solidIndices: [1], python: EngineService.defaultPython, threads: 4)
        let exported = try EngineSTLExportTests.summary(selected)
        XCTAssertEqual(exported.volume, 1, accuracy: 0.02)
        for (actual, expected) in zip(exported.bounds, [10.0, 0, 0, 11, 1, 1]) { XCTAssertEqual(actual, expected, accuracy: 0.02) }
        try FileManager.default.removeItem(atPath: graph)
        let cached = try await service.exportSTL(preview, solidIndices: [1], python: "/nonexistent/python", threads: 1)
        XCTAssertEqual(cached, selected)
    }

    @MainActor
    func testPartTotalsArePosedLocallyWithoutAWorker() throws {
        var preview = PreviewMesh(vertices: [[0, 0, 0], [1, 0, 0], [0, 1, 0]], triangles: [[0, 1, 2], [0, 1, 2]], edges: [],
                                  bounds: [0, 0, 0, 12, 4, 6], volume: 56, area: 112, solids: 2, buildSeconds: 0, threads: 1,
                                  buildID: UUID().uuidString, triangleSolids: [0, 1])
        preview.partStats = [
            CADPart(solidIndex: 0, area: 24, volume: 8, bounds: [0, 0, 0, 2, 2, 2], centroid: [1, 1, 1], triangles: 1, components: 1),
            CADPart(solidIndex: 1, area: 88, volume: 48, bounds: [10, 0, 0, 12, 4, 6], centroid: [11, 2, 3], triangles: 1, components: 1)
        ]
        // A quarter turn about Z through the origin.
        let turn: [Double] = [0, -1, 0, 0, 1, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1]
        let report = try CADMeasurements.report(preview, solids: [1], solidTransforms: [1: turn])
        func values(_ id: String) throws -> [Double] { try XCTUnwrap(report.metrics.first { $0.id == id }?.values) }
        XCTAssertEqual(try values("parts:volume"), [48])
        XCTAssertEqual(try values("parts:area"), [88])
        XCTAssertEqual(try values("parts:centroid"), [-2, 11, 3])
        XCTAssertEqual(try values("parts:minimum"), [-4, 10, 0])
        XCTAssertEqual(try values("parts:maximum"), [0, 12, 6])
        let both = try CADMeasurements.report(preview, solids: [0, 1])
        XCTAssertEqual(both.metrics.first { $0.id == "parts:volume" }?.values, [56])
        let expected: [Double] = [(8 + 48 * 11) / 56, (8 + 48 * 2) / 56, (8 + 48 * 3) / 56]
        XCTAssertEqual(both.metrics.first { $0.id == "parts:centroid" }?.values, expected)
        preview.partStats = nil
        XCTAssertThrowsError(try CADMeasurements.report(preview, solids: [0]))
    }

    @MainActor
    func testInvalidSelectedExportFailsBeforeAnyNativeProcess() async throws {
        let preview = PreviewMesh(vertices: [], triangles: [], edges: [], bounds: [0, 0, 0, 1, 1, 1], volume: 1,
                                  area: 6, solids: 1, buildSeconds: 0, threads: 1, buildID: UUID().uuidString)
        let selections: [Set<Int>] = [[], [-1], [1]]
        for selection in selections {
            do {
                _ = try await EngineService().exportSTL(preview, solidIndices: selection, python: "/nonexistent/python", threads: 1)
                XCTFail("Invalid selections must be rejected before resolving a runtime or reading an artifact")
            } catch {
                XCTAssertEqual(error.localizedDescription, "Select at least one available part to export.")
            }
        }
    }

    @MainActor
    func testInvalidMeasureEntitiesFailBeforeAnyNativeProcess() async throws {
        let preview = PreviewMesh(vertices: [], triangles: [], edges: [], bounds: [0, 0, 0, 1, 1, 1], volume: 1,
                                  area: 6, solids: 1, buildSeconds: 0, threads: 1, buildID: UUID().uuidString)
        let invalid: [[EngineWorkerRequest.MeasureEntity]] = [[], [.part(solid: 1)], [.part(solid: 0), .part(solid: 0), .part(solid: 0)],
                                                               [.point(solid: 0, point: [0, 0])], [.point(solid: 0, point: [.nan, 0, 0])]]
        for entities in invalid {
            do {
                _ = try await EngineService().measure(preview, entities: entities, solidTransforms: nil,
                                                      python: "/nonexistent/python", threads: 1)
                XCTFail("Invalid entities must be rejected before reading an artifact")
            } catch {
                XCTAssertEqual(error.localizedDescription, "Measure one or two available parts or surface points.")
            }
        }
    }
}
