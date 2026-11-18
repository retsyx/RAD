import Foundation
import XCTest
@testable import RAD

final class EngineThreeMFExportTests: XCTestCase {
    @MainActor
    func testInvalidSelectionsFailBeforeReadingGeometryOrStartingPython() async throws {
        let preview = unavailablePreview()
        for solids in [Set<Int>(), [-1], [2]] {
            do {
                _ = try await EngineService().export3MF(preview, solidIndices: solids, python: "/missing/python", threads: 1)
                XCTFail("An invalid subset must not start an export.")
            } catch {
                XCTAssertEqual(error.localizedDescription, "Select at least one available part to export.")
            }
        }
        do {
            _ = try await EngineService().export3MF(preview, python: "/missing/python", threads: 1)
            XCTFail("Exports need the build's own SDF graph.")
        } catch EngineExportError.nativeArtifactUnavailable {}
    }

    @MainActor
    func testIndividualExportsRejectInvalidPartitionsBeforeReadingNativeGeometry() async throws {
        let preview = unavailablePreview()
        for parts in [[], [[]], [[0], [0]], [[0, 0]], [[-1]], [[2]], [[0, 1], [1]]] {
            do {
                _ = try await EngineService().export3MFParts(preview, parts: parts, python: "/missing/python", threads: 1)
                XCTFail("Invalid individual part ownership must not start a worker.")
            } catch {
                XCTAssertEqual(error.localizedDescription, "Select distinct available parts to export individually.")
            }
        }
        do {
            _ = try await EngineService().export3MFParts(preview, parts: [[0]], python: "/missing/python", threads: 1)
            XCTFail("Individual export requires the build's own SDF graph.")
        } catch EngineExportError.nativeArtifactUnavailable {}
    }

    func test3MFValidationRejectsIncompleteArchives() {
        XCTAssertThrowsError(try EngineService.validate3MF(Data()))
        XCTAssertThrowsError(try EngineService.validate3MF(Data([0x50, 0x4b, 0x03, 0x04]) + Data(count: 32)))
    }

    @MainActor
    func testFullSubsetAndPlacedExportsUseTheirOwnCacheEntriesWithoutRebuilding() async throws {
        try requireRuntime()
        let service = EngineService()
        let preview = try await service.buildPreview(assembly(), python: EngineService.defaultPython, threads: 2)
        let native = URL(fileURLWithPath: try XCTUnwrap(preview.artifactPath))
        let nativeBefore = try Data(contentsOf: native)
        let fullBytes = try await service.export3MF(preview, python: EngineService.defaultPython, threads: 2)
        XCTAssertNoThrow(try EngineService.validate3MF(fullBytes))
        XCTAssertThrowsError(try EngineService.validate3MF(fullBytes.dropLast()))
        let full = try Self.summary(fullBytes)
        XCTAssertEqual(full.colors, ["First": "#12AB56", "Second": "#E37A21"])
        XCTAssertEqual(full.volume, 54, accuracy: 54 * 0.02)
        for (actual, expected) in zip(full.bounds, [0.0, 0, 0, 25, 9, 12]) { XCTAssertEqual(actual, expected, accuracy: 0.05) }
        let subsetBytes = try await service.export3MF(preview, solidIndices: [1], python: EngineService.defaultPython, threads: 2)
        let subset = try Self.summary(subsetBytes)
        XCTAssertEqual(subset.volume, 30, accuracy: 30 * 0.02)
        XCTAssertEqual(subset.colors, ["Second": "#E37A21"])
        for (actual, expected) in zip(subset.bounds, [20.0, 7, 9, 25, 9, 12]) { XCTAssertEqual(actual, expected, accuracy: 0.05) }

        let orientations = try await service.printOrientations(preview, solids: [0, 1], python: EngineService.defaultPython, threads: 2)
        let plate = PrintLayout.transforms(for: [0, 1], orientations: orientations)
        let arrangedBytes = try await service.export3MF(preview, solidTransforms: plate, python: EngineService.defaultPython, threads: 2)
        let arranged = try Self.summary(arrangedBytes)
        XCTAssertEqual(arranged.volume, full.volume, accuracy: full.volume * 0.02)
        XCTAssertEqual(arranged.bounds[2], 0, accuracy: 0.05, "Printing placement puts every part on the bed.")
        XCTAssertEqual(try Data(contentsOf: native), nativeBefore, "Exports must not mutate or replay the session geometry.")

        let recolored = ThreeMFExport.presentation(in: try XCTUnwrap(preview.parts),
            overrides: ["second": PartColor(red: 0, green: 0.5, blue: 1)])
        let recoloredBytes = try await service.export3MF(preview, solidIndices: [1], partPresentation: recolored,
            python: EngineService.defaultPython, threads: 2)
        let newSubset = try Self.summary(recoloredBytes)
        XCTAssertEqual(newSubset.colors, ["Second": "#0080FF"])
        XCTAssertEqual(newSubset.bounds, subset.bounds)
        XCTAssertEqual(newSubset.volume, subset.volume)

        // Cached variants need neither the graph nor a runtime.
        try FileManager.default.removeItem(at: native)
        let cachedSubset = try await service.export3MF(preview, solidIndices: [1], python: "/missing/python", threads: 1)
        XCTAssertEqual(cachedSubset, subsetBytes)
        let cachedRecolored = try await service.export3MF(preview, solidIndices: [1], partPresentation: recolored,
            python: "/missing/python", threads: 1)
        XCTAssertEqual(cachedRecolored, recoloredBytes)
        let cachedArranged = try await service.export3MF(preview, solidTransforms: plate, python: "/missing/python", threads: 1)
        XCTAssertEqual(cachedArranged, arrangedBytes)
        do {
            _ = try await service.export3MF(preview, solidIndices: [0], python: "/missing/python", threads: 1)
            XCTFail("Another subset must not reuse a cached variant.")
        } catch EngineExportError.nativeArtifactUnavailable {}
    }

    @MainActor
    func testIndividualExportsPreserveOrderGeometryAndSeparateCache() async throws {
        try requireRuntime()
        let service = EngineService()
        let preview = try await service.buildPreview(assembly(), python: EngineService.defaultPython, threads: 2)
        let native = URL(fileURLWithPath: try XCTUnwrap(preview.artifactPath))
        let files = try await service.export3MFParts(preview, parts: [[1], [0]], python: EngineService.defaultPython, threads: 2)
        XCTAssertEqual(files.count, 2)
        let second = try Self.summary(files[0]), first = try Self.summary(files[1])
        XCTAssertEqual(second.volume, 30, accuracy: 30 * 0.02)
        XCTAssertEqual(first.volume, 24, accuracy: 24 * 0.02)
        for (actual, expected) in zip(first.bounds, [0.0, 0, 0, 2, 3, 4]) { XCTAssertEqual(actual, expected, accuracy: 0.05) }
        try FileManager.default.removeItem(at: native)
        let cached = try await service.export3MFParts(preview, parts: [[1], [0]], python: "/missing/python", threads: 1)
        XCTAssertEqual(cached, files)
        do {
            _ = try await service.export3MFParts(preview, parts: [[0], [1]], python: "/missing/python", threads: 1)
            XCTFail("A differently ordered partition must not return mismatched cached files.")
        } catch EngineExportError.nativeArtifactUnavailable {}
    }

    /// Read the exported ZIP and model using system archive and XML readers.
    static func summary(_ data: Data) throws -> (
        triangles: Int, bounds: [Double], volume: Double, centroid: [Double], colors: [String: String]
    ) {
        let file = FileManager.default.temporaryDirectory.appendingPathComponent("RAD-\(UUID()).3mf")
        try data.write(to: file, options: .withoutOverwriting)
        defer { try? FileManager.default.removeItem(at: file) }

        let unzip = Process()
        unzip.executableURL = URL(fileURLWithPath: "/usr/bin/unzip")
        unzip.arguments = ["-p", file.path, "3D/3dmodel.model"]
        let contents = Pipe()
        unzip.standardOutput = contents
        try unzip.run()
        let xml = try XCTUnwrap(contents.fileHandleForReading.readToEnd())
        unzip.waitUntilExit()
        XCTAssertEqual(unzip.terminationStatus, 0)

        let model = try XCTUnwrap(XMLDocument(data: xml, options: []).rootElement())
        XCTAssertEqual(model.uri, "http://schemas.microsoft.com/3dmanufacturing/core/2015/02")
        XCTAssertEqual(model.attribute(forName: "unit")?.stringValue, "millimeter")
        let resources = try XCTUnwrap(model.elements(forName: "resources").first)
        let build = try XCTUnwrap(model.elements(forName: "build").first)
        func attribute(_ name: String, _ element: XMLElement) throws -> String {
            try XCTUnwrap(element.attribute(forName: name)?.stringValue)
        }
        func byID(_ elements: [XMLElement]) throws -> [String: XMLElement] {
            try Dictionary(uniqueKeysWithValues: elements.map { (try attribute("id", $0), $0) })
        }
        let materialNamespace = "http://schemas.microsoft.com/3dmanufacturing/material/2015/02"
        let groups = try byID(resources.elements(forLocalName: "colorgroup", uri: materialNamespace))
        let objects = try byID(resources.elements(forName: "object"))

        var low = [Double](repeating: .infinity, count: 3), high = [Double](repeating: -.infinity, count: 3)
        var volume = 0.0, moment = [0.0, 0, 0], count = 0
        var colors: [String: String] = [:]
        for item in build.elements(forName: "item") {
            XCTAssertNil(item.attribute(forName: "transform"), "Geometry already carries its placement.")
            let object = try XCTUnwrap(objects[try attribute("objectid", item)])
            let group = try XCTUnwrap(groups[try attribute("pid", object)])
            let index = try XCTUnwrap(Int(attribute("pindex", object)))
            let entries = group.elements(forLocalName: "color", uri: materialNamespace)
            let entry = try XCTUnwrap(entries.indices.contains(index) ? entries[index] : nil)
            colors[try attribute("name", object)] = try attribute("color", entry)
            let mesh = try XCTUnwrap(object.elements(forName: "mesh").first)
            let vertices = try XCTUnwrap(mesh.elements(forName: "vertices").first)
            let triangles = try XCTUnwrap(mesh.elements(forName: "triangles").first)
            let points = try vertices.elements(forName: "vertex").map { vertex in
                try ["x", "y", "z"].map { try XCTUnwrap(Double(attribute($0, vertex))) }
            }
            for triangle in triangles.elements(forName: "triangle") {
                let corners = try ["v1", "v2", "v3"].map { name in
                    let index = try XCTUnwrap(Int(attribute(name, triangle)))
                    return try XCTUnwrap(points.indices.contains(index) ? points[index] : nil)
                }
                for corner in corners {
                    for axis in 0..<3 {
                        low[axis] = min(low[axis], corner[axis])
                        high[axis] = max(high[axis], corner[axis])
                    }
                }
                let a = corners[0], b = corners[1], c = corners[2]
                let signed = (a[0] * (b[1] * c[2] - b[2] * c[1]) - a[1] * (b[0] * c[2] - b[2] * c[0])
                              + a[2] * (b[0] * c[1] - b[1] * c[0])) / 6
                count += 1
                volume += signed
                for axis in 0..<3 { moment[axis] += signed * (a[axis] + b[axis] + c[axis]) / 4 }
            }
        }
        XCTAssertGreaterThan(count, 0)
        return (count, low + high, volume, moment.map { volume == 0 ? 0 : $0 / volume }, colors)
    }

    private func requireRuntime() throws {
        guard ProcessInfo.processInfo.environment["RAD_RUN_ENGINE_TESTS"] == "1" else {
            throw XCTSkip("Set RAD_RUN_ENGINE_TESTS=1 for 3MF export and cache tests.")
        }
    }

    private func unavailablePreview() -> PreviewMesh {
        PreviewMesh(vertices: [], triangles: [], edges: [], bounds: [0,0,0,1,1,1], volume: 1,
                    area: 6, solids: 2, buildSeconds: 0, threads: 1,
                    buildID: UUID().uuidString, artifactPath: "/unregistered/model.rsdf", partRoots: [0, 1])
    }

    private func assembly() -> ModelSpec {
        ModelSpec(title: "3MF assembly", summary: "Two boxes exported as separate shells.", units: "mm", script: """
        result = Assembly("Parts", id="parts", color="#12AB56")
        result.add(box((2, 3, 4), center=(1, 1.5, 2)), id="first", name="First")
        result.add(box((5, 2, 3), center=(2.5, 1, 1.5)), id="second", name="Second", color="#E37A21", at=Placement(translate=(20, 7, 9)))
        """)
    }
}
