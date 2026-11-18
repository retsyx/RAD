import Foundation
import XCTest
@testable import RAD

final class EngineSTLExportTests: XCTestCase {
    @MainActor
    func testInvalidSelectionsFailBeforeReadingGeometryOrStartingPython() async throws {
        let preview = unavailablePreview()
        for solids in [Set<Int>(), [-1], [2]] {
            do {
                _ = try await EngineService().exportSTL(preview, solidIndices: solids, python: "/missing/python", threads: 1)
                XCTFail("An invalid subset must not start an export.")
            } catch {
                XCTAssertEqual(error.localizedDescription, "Select at least one available part to export.")
            }
        }
        do {
            _ = try await EngineService().exportSTL(preview, python: "/missing/python", threads: 1)
            XCTFail("Exports need the build's own SDF graph.")
        } catch EngineExportError.nativeArtifactUnavailable {}
    }

    @MainActor
    func testIndividualExportsRejectInvalidPartitionsBeforeReadingNativeGeometry() async throws {
        let preview = unavailablePreview()
        for parts in [[], [[]], [[0], [0]], [[0, 0]], [[-1]], [[2]], [[0, 1], [1]]] {
            do {
                _ = try await EngineService().exportSTLParts(preview, parts: parts, python: "/missing/python", threads: 1)
                XCTFail("Invalid individual part ownership must not start a worker.")
            } catch {
                XCTAssertEqual(error.localizedDescription, "Select distinct available parts to export individually.")
            }
        }
        do {
            _ = try await EngineService().exportSTLParts(preview, parts: [[0]], python: "/missing/python", threads: 1)
            XCTFail("Individual export requires the build's own SDF graph.")
        } catch EngineExportError.nativeArtifactUnavailable {}
    }

    func testBinarySTLValidationRequiresExactLengthAndFiniteValues() throws {
        let triangle: [[Float]] = [[0, 0, 1], [0, 0, 0], [1, 0, 0], [0, 1, 0]]
        let valid = Self.stl([triangle, triangle])
        XCTAssertNoThrow(try EngineService.validateSTL(valid))
        XCTAssertThrowsError(try EngineService.validateSTL(valid.dropLast()))
        XCTAssertThrowsError(try EngineService.validateSTL(Self.stl([])))
        XCTAssertThrowsError(try EngineService.validateSTL(Self.stl([[[0, 0, 1], [0, 0, .nan], [1, 0, 0], [0, 1, 0]]])))
        XCTAssertThrowsError(try EngineService.validateSTL(Data(count: 84)))
    }

    @MainActor
    func testFullSubsetAndPlacedExportsUseTheirOwnCacheEntriesWithoutRebuilding() async throws {
        try requireRuntime()
        let service = EngineService()
        let preview = try await service.buildPreview(assembly(), python: EngineService.defaultPython, threads: 2)
        let native = URL(fileURLWithPath: try XCTUnwrap(preview.artifactPath))
        let nativeBefore = try Data(contentsOf: native)
        let full = try Self.summary(await service.exportSTL(preview, python: EngineService.defaultPython, threads: 2))
        XCTAssertEqual(full.volume, 54, accuracy: 54 * 0.02)
        for (actual, expected) in zip(full.bounds, [0.0, 0, 0, 25, 9, 12]) { XCTAssertEqual(actual, expected, accuracy: 0.05) }
        let subsetBytes = try await service.exportSTL(preview, solidIndices: [1], python: EngineService.defaultPython, threads: 2)
        let subset = try Self.summary(subsetBytes)
        XCTAssertEqual(subset.volume, 30, accuracy: 30 * 0.02)
        for (actual, expected) in zip(subset.bounds, [20.0, 7, 9, 25, 9, 12]) { XCTAssertEqual(actual, expected, accuracy: 0.05) }

        let orientations = try await service.printOrientations(preview, solids: [0, 1], python: EngineService.defaultPython, threads: 2)
        let plate = PrintLayout.transforms(for: [0, 1], orientations: orientations)
        let arrangedBytes = try await service.exportSTL(preview, solidTransforms: plate, python: EngineService.defaultPython, threads: 2)
        let arranged = try Self.summary(arrangedBytes)
        XCTAssertEqual(arranged.volume, full.volume, accuracy: full.volume * 0.02)
        XCTAssertEqual(arranged.bounds[2], 0, accuracy: 0.05, "Printing placement puts every part on the bed.")
        XCTAssertEqual(try Data(contentsOf: native), nativeBefore, "Exports must not mutate or replay the session geometry.")

        // Cached variants need neither the graph nor a runtime.
        try FileManager.default.removeItem(at: native)
        let cachedSubset = try await service.exportSTL(preview, solidIndices: [1], python: "/missing/python", threads: 1)
        XCTAssertEqual(cachedSubset, subsetBytes)
        let cachedArranged = try await service.exportSTL(preview, solidTransforms: plate, python: "/missing/python", threads: 1)
        XCTAssertEqual(cachedArranged, arrangedBytes)
        do {
            _ = try await service.exportSTL(preview, solidIndices: [0], python: "/missing/python", threads: 1)
            XCTFail("Another subset must not reuse a cached variant.")
        } catch EngineExportError.nativeArtifactUnavailable {}
    }

    @MainActor
    func testIndividualExportsPreserveOrderGeometryAndSeparateCache() async throws {
        try requireRuntime()
        let service = EngineService()
        let preview = try await service.buildPreview(assembly(), python: EngineService.defaultPython, threads: 2)
        let native = URL(fileURLWithPath: try XCTUnwrap(preview.artifactPath))
        let files = try await service.exportSTLParts(preview, parts: [[1], [0]], python: EngineService.defaultPython, threads: 2)
        XCTAssertEqual(files.count, 2)
        let second = try Self.summary(files[0]), first = try Self.summary(files[1])
        XCTAssertEqual(second.volume, 30, accuracy: 30 * 0.02)
        XCTAssertEqual(first.volume, 24, accuracy: 24 * 0.02)
        for (actual, expected) in zip(first.bounds, [0.0, 0, 0, 2, 3, 4]) { XCTAssertEqual(actual, expected, accuracy: 0.05) }
        try FileManager.default.removeItem(at: native)
        let cached = try await service.exportSTLParts(preview, parts: [[1], [0]], python: "/missing/python", threads: 1)
        XCTAssertEqual(cached, files)
        do {
            _ = try await service.exportSTLParts(preview, parts: [[0], [1]], python: "/missing/python", threads: 1)
            XCTFail("A differently ordered partition must not return mismatched cached files.")
        } catch EngineExportError.nativeArtifactUnavailable {}
    }

    /// Triangle count, bounds, enclosed volume and volume centroid of a binary STL.
    static func summary(_ data: Data) throws -> (triangles: Int, bounds: [Double], volume: Double, centroid: [Double]) {
        try EngineService.validateSTL(data)
        return data.withUnsafeBytes { raw in
            let count = Int(raw.loadUnaligned(fromByteOffset: 80, as: UInt32.self))
            var low = [Double](repeating: .infinity, count: 3), high = [Double](repeating: -.infinity, count: 3)
            var volume = 0.0, moment = [0.0, 0, 0]
            for triangle in 0..<count {
                let base = 84 + 50 * triangle
                let corners = (1...3).map { vertex in
                    (0..<3).map { Double(raw.loadUnaligned(fromByteOffset: base + 12 * vertex + 4 * $0, as: Float.self)) }
                }
                for corner in corners {
                    for axis in 0..<3 { low[axis] = min(low[axis], corner[axis]); high[axis] = max(high[axis], corner[axis]) }
                }
                let a = corners[0], b = corners[1], c = corners[2]
                let signed = (a[0] * (b[1] * c[2] - b[2] * c[1]) - a[1] * (b[0] * c[2] - b[2] * c[0])
                              + a[2] * (b[0] * c[1] - b[1] * c[0])) / 6
                volume += signed
                for axis in 0..<3 { moment[axis] += signed * (a[axis] + b[axis] + c[axis]) / 4 }
            }
            return (count, low + high, volume, moment.map { volume == 0 ? 0 : $0 / volume })
        }
    }

    /// A binary STL with one normal and three vertices per triangle.
    static func stl(_ triangles: [[[Float]]]) -> Data {
        var data = Data(count: 80)
        var count = UInt32(triangles.count).littleEndian
        data.append(Data(bytes: &count, count: 4))
        for triangle in triangles {
            for var value in triangle.flatMap({ $0 }) { data.append(Data(bytes: &value, count: 4)) }
            data.append(Data(count: 2))
        }
        return data
    }

    private func requireRuntime() throws {
        guard ProcessInfo.processInfo.environment["RAD_RUN_ENGINE_TESTS"] == "1" else {
            throw XCTSkip("Set RAD_RUN_ENGINE_TESTS=1 for STL export and cache tests.")
        }
    }

    private func unavailablePreview() -> PreviewMesh {
        PreviewMesh(vertices: [], triangles: [], edges: [], bounds: [0,0,0,1,1,1], volume: 1,
                    area: 6, solids: 2, buildSeconds: 0, threads: 1,
                    buildID: UUID().uuidString, artifactPath: "/unregistered/model.rsdf", partRoots: [0, 1])
    }

    private func assembly() -> ModelSpec {
        ModelSpec(title: "STL assembly", summary: "Two boxes exported as separate shells.", units: "mm", script: """
        result = Assembly("Parts", id="parts")
        result.add(box((2, 3, 4), center=(1, 1.5, 2)), id="first", name="First")
        result.add(box((5, 2, 3), center=(2.5, 1, 1.5)), id="second", name="Second", at=Placement(translate=(20, 7, 9)))
        """)
    }
}
