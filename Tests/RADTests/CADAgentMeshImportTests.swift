import CryptoKit
import Foundation
import XCTest
@testable import RAD

/// rad_import registers user-provided STL and OBJ meshes as design inputs
/// that programs read as imported_assets[assetID] (SDF-DESIGN.md, MESH).
final class CADAgentMeshImportTests: XCTestCase {
    @MainActor
    func testImportArgumentsAndGeneratedGeometryFailBeforeNativeWork() async throws {
        let root = temporary()
        defer { try? FileManager.default.removeItem(at: root) }
        let tools = try CADAgentTools(directory: root, python: "/missing/python", threads: 1, current: nil,
                                      currentPreview: nil, engine: EngineService())
        let cube = MeshImportFixture.cubeSTL(size: 10)
        try cube.write(to: root.appendingPathComponent("part.step"))
        for folder in ["exports", "Renders", "ARTIFACTS/a"] {
            try FileManager.default.createDirectory(at: root.appendingPathComponent(folder), withIntermediateDirectories: true)
            try cube.write(to: root.appendingPathComponent("\(folder)/copy.stl"))
        }
        try cube.write(to: root.appendingPathComponent("part.stl"))
        // An STL that RAD wrote (an export of any task, or Mesh.save_stl) is
        // generated geometry wherever it is.
        try MeshImportFixture.boxSTL([10, 10, 10], header: "RAD binary STL, millimetres").write(to: root.appendingPathComponent("earlier.stl"))
        let failures: [([String: Any], String)] = [
            (["path": "part.step"], "STL (.stl) or OBJ (.obj)"),
            (["path": "exports/copy.stl"], "generated for this design"),
            (["path": "Exports/copy.stl"], "generated for this design"),
            (["path": "Renders/copy.stl"], "generated for this design"),
            (["path": "ARTIFACTS/a/copy.stl"], "generated for this design"),
            (["path": "earlier.stl"], "an STL written by RAD"),
            (["path": "missing.stl"], "missing"),
            (["path": "part.stl", "format": "stl"], "unsupported arguments"),
            (["path": "part.stl", "name": "folder/part.stl"], "invalid ID, filename"),
            (["path": "../outside.stl"], "inside the task workspace"),
            ([:], "requires ‘path’"),
        ]
        for (arguments, message) in failures {
            do {
                _ = try await tools.handle(name: "rad_import", arguments: arguments)
                XCTFail("rad_import \(arguments) must fail")
            } catch {
                XCTAssertTrue(error.localizedDescription.contains(message), "\(arguments): \(error.localizedDescription)")
            }
        }
    }

    func testCatalogDeclaresImportReferencesAndProducerAudit() throws {
        let tool = try XCTUnwrap(AgentToolCatalog.native()["rad_import"])
        XCTAssertEqual(tool.host.capability, .build)
        XCTAssertEqual(tool.host.references, ["path"])
        XCTAssertEqual(tool.host.sources, ["producerPaths"])
        XCTAssertTrue(tool.description.contains("imported_assets[assetID]"))
        XCTAssertTrue(tool.description.contains("Never import geometry generated for this design"))
        let external = try XCTUnwrap(AgentToolCatalog.external()["rad_import"])
        let properties = try XCTUnwrap(external.inputSchema["properties"] as? [String: Any])
        XCTAssertTrue((properties["path"] as? [String: Any])?["description"] as? String == "Opaque file ID returned by a task upload or a previous tool result; filesystem paths are not accepted.")
        XCTAssertTrue(external.description.contains("rad_commit"))
    }

    @MainActor
    func testImportedMeshesBuildAsInputsAndGeneratedExportsCannotReturn() async throws {
        guard ProcessInfo.processInfo.environment["RAD_RUN_ENGINE_TESTS"] == "1" else {
            throw XCTSkip("Set RAD_RUN_ENGINE_TESTS=1 to import meshes with the native kernel.")
        }
        let root = temporary()
        defer { try? FileManager.default.removeItem(at: root) }
        let cube = MeshImportFixture.cubeSTL(size: 10)
        try cube.write(to: root.appendingPathComponent("cube.stl"))
        try MeshImportFixture.cubeSTL(size: 10, open: true).write(to: root.appendingPathComponent("open.stl"))
        try Data(MeshImportFixture.cubeOBJ(size: 4).utf8).write(to: root.appendingPathComponent("block.obj"))
        try Data("# conversion notes\n".utf8).write(to: root.appendingPathComponent("convert.py"))
        let tools = try CADAgentTools(directory: root, python: EngineService.defaultPython, threads: 2,
                                      current: nil, currentPreview: nil, engine: EngineService())
        let imported = try await tools.handle(name: "rad_import", arguments: ["path": "cube.stl", "producerPaths": ["convert.py"]]).metadata
        let assetID = try XCTUnwrap(imported["assetID"] as? String)
        XCTAssertEqual(assetID, "asset-" + SHA256.hexDigest(cube))
        XCTAssertEqual(imported["format"] as? String, "stl")
        XCTAssertEqual(imported["name"] as? String, "cube.stl")
        XCTAssertEqual(imported["byteCount"] as? Int, cube.count)
        XCTAssertEqual(imported["programBinding"] as? String, "imported_assets[assetID]")
        XCTAssertTrue((imported["fidelity"] as? String)?.contains("faceted") == true)
        XCTAssertEqual((imported["producerPaths"] as? [String])?.map { URL(fileURLWithPath: $0).lastPathComponent }, ["convert.py"])
        XCTAssertEqual(try XCTUnwrap(imported["volume"] as? Double), 1000, accuracy: 5)
        let runtime = try await tools.handle(name: "rad_runtime", arguments: [:]).metadata
        XCTAssertEqual((runtime["inputAssets"] as? [[String: Any]])?.compactMap { $0["assetID"] as? String }, [assetID])

        let program = "result = imported_assets[\"\(assetID)\"] | box((30, 4, 4))\n"
        try Data(program.utf8).write(to: root.appendingPathComponent("union.py"))
        do {
            _ = try await tools.handle(name: "rad_build", arguments: ["programPath": "union.py"])
            XCTFail("A build reads only the inputs it selects; the design has none yet.")
        } catch { XCTAssertTrue(error.localizedDescription.contains("not available"), error.localizedDescription) }
        let built = try await tools.handle(name: "rad_build", arguments: ["programPath": "union.py", "assetIDs": [assetID]]).metadata
        let artifactID = try XCTUnwrap(built["artifactID"] as? String)
        XCTAssertEqual(try XCTUnwrap(built["volume"] as? Double), 1000 + 480 - 160, accuracy: 8)
        XCTAssertEqual(built["assetIDs"] as? [String], [assetID])
        let candidate = try await tools.committableCandidate(artifactID: artifactID)
        XCTAssertEqual(candidate.model.inputAssets?.map(\.id), [assetID])
        XCTAssertEqual(candidate.model.inputAssets?.first?.data, cube)

        // Generated geometry never becomes an input, even renamed elsewhere.
        let export = try await tools.handle(name: "rad_export_3mf", arguments: ["artifactID": artifactID]).metadata
        let exported = try Data(contentsOf: URL(fileURLWithPath: try XCTUnwrap(export["path"] as? String)))
        try exported.write(to: root.appendingPathComponent("renamed.stl"))
        do {
            _ = try await tools.handle(name: "rad_import", arguments: ["path": "renamed.stl"])
            XCTFail("An exported 3MF must not become an imported input.")
        } catch { XCTAssertTrue(error.localizedDescription.contains("generated for this design"), error.localizedDescription) }

        do {
            _ = try await tools.handle(name: "rad_import", arguments: ["path": "open.stl", "name": "Open lid"])
            XCTFail("An open mesh is not a solid.")
        } catch {
            XCTAssertTrue(error.localizedDescription.contains("'Open lid'"), error.localizedDescription)
            XCTAssertTrue(error.localizedDescription.contains("not a closed"), error.localizedDescription)
        }
        let obj = try await tools.handle(name: "rad_import", arguments: ["path": "block.obj", "name": "Block"]).metadata
        XCTAssertEqual(obj["format"] as? String, "obj")
        XCTAssertEqual(try XCTUnwrap(obj["volume"] as? Double), 64, accuracy: 1)
    }

    @MainActor
    func testThinMeshesRegisterWhenOnlyTheirStandAlonePreviewFails() async throws {
        guard ProcessInfo.processInfo.environment["RAD_RUN_ENGINE_TESTS"] == "1" else {
            throw XCTSkip("Set RAD_RUN_ENGINE_TESTS=1 to import meshes with the native kernel.")
        }
        let root = temporary()
        defer { try? FileManager.default.removeItem(at: root) }
        // A sheet far thinner than the preview cells of its own extent.
        let sheet = MeshImportFixture.boxSTL([300, 300, 0.5])
        try sheet.write(to: root.appendingPathComponent("sheet.stl"))
        let tools = try CADAgentTools(directory: root, python: EngineService.defaultPython, threads: 2,
                                      current: nil, currentPreview: nil, engine: EngineService())
        let imported = try await tools.handle(name: "rad_import", arguments: ["path": "sheet.stl"]).metadata
        let assetID = try XCTUnwrap(imported["assetID"] as? String)
        XCTAssertNil(imported["artifactID"])
        let warning = try XCTUnwrap((imported["warnings"] as? [String])?.first)
        XCTAssertTrue(warning.contains("valid solid") && warning.contains("preview cell size"), warning)
        XCTAssertEqual(try tools.resolveAssets(ids: [assetID]).map(\.data), [sheet])
        let program = "result = box((40, 40, 20)) - imported_assets[\"\(assetID)\"].offset(0.3)\n"
        try Data(program.utf8).write(to: root.appendingPathComponent("slot.py"))
        let built = try await tools.handle(name: "rad_build", arguments: ["programPath": "slot.py", "assetIDs": [assetID]]).metadata
        XCTAssertEqual(try XCTUnwrap(built["volume"] as? Double), 40 * 40 * 20 - 40 * 40 * 1.1, accuracy: 40)
        // An invalid mesh is still refused, whatever its size.
        try MeshImportFixture.boxSTL([300, 300, 0.5], open: true).write(to: root.appendingPathComponent("open.stl"))
        do {
            _ = try await tools.handle(name: "rad_import", arguments: ["path": "open.stl"])
            XCTFail("An open mesh is not a solid.")
        } catch { XCTAssertTrue(error.localizedDescription.contains("not a closed"), error.localizedDescription) }
    }

    @MainActor
    func testHistoricalBuildUsesItsOwnOriginalMeshAndConflictingAssetIDsWithoutChangingCurrent() async throws {
        guard ProcessInfo.processInfo.environment["RAD_RUN_ENGINE_TESTS"] == "1" else {
            throw XCTSkip("Set RAD_RUN_ENGINE_TESTS=1 for historical replay of imported meshes.")
        }
        let root = temporary()
        defer { try? FileManager.default.removeItem(at: root) }
        let oldAsset = CADInputAsset(id: "shared", name: "old.stl", format: "stl", data: MeshImportFixture.cubeSTL(size: 2))
        let currentAsset = CADInputAsset(id: "shared", name: "current.stl", format: "stl", data: MeshImportFixture.cubeSTL(size: 5))
        let program = """
        result = Assembly('Fixture')
        result.add(imported_assets['shared'], id='reference', name='Reference')

        """
        let historical = ProjectRevision(model: ModelSpec(title: "Historical", summary: "A saved design.", units: "mm",
                                                          script: program, inputAssets: [oldAsset]))
        let current = ModelSpec(title: "Current", summary: "A saved design.", units: "mm", script: program, inputAssets: [currentAsset])
        let tools = try CADAgentTools(directory: root, python: EngineService.defaultPython, threads: 2, current: current,
                                      currentPreview: nil, engine: EngineService(),
                                      versions: ProjectVersionContext(revisions: [historical], selectedRevisionID: nil))
        let lookedUp = try await tools.handle(name: "rad_version", arguments: ["reference": "v1"]).metadata
        let programPath = try XCTUnwrap(lookedUp["programPath"] as? String)
        let oldBuild = try await tools.handle(name: "rad_build", arguments: ["programPath": programPath,
                                                                              "versionID": historical.id.uuidString]).metadata
        XCTAssertEqual(oldBuild["versionID"] as? String, historical.id.uuidString)
        XCTAssertEqual(try XCTUnwrap(oldBuild["volume"] as? Double), 8, accuracy: 0.08)
        XCTAssertEqual(try tools.resolveAssets(ids: ["shared"]), [currentAsset])
        let currentBuild = try await tools.handle(name: "rad_build", arguments: ["programPath": programPath]).metadata
        XCTAssertNil(currentBuild["versionID"])
        XCTAssertEqual(try XCTUnwrap(currentBuild["volume"] as? Double), 125, accuracy: 1.25)
        XCTAssertEqual(try XCTUnwrap(tools.validatedPreview(for: historical.model)?.volume), 8, accuracy: 0.08)
        XCTAssertEqual(try XCTUnwrap(tools.validatedPreview(for: current)?.volume), 125, accuracy: 1.25)
    }

    private func temporary() -> URL {
        let url = FileManager.default.temporaryDirectory.appendingPathComponent("RAD-mesh-import-test-\(UUID().uuidString)")
        try! FileManager.default.createDirectory(at: url, withIntermediateDirectories: true)
        return url
    }
}

/// Closed (or deliberately open) cube meshes centred on the origin.
enum MeshImportFixture {
    static let corners: [[Float]] = [[-1, -1, -1], [-1, -1, 1], [-1, 1, -1], [-1, 1, 1],
                                     [1, -1, -1], [1, -1, 1], [1, 1, -1], [1, 1, 1]]
    /// Outward counter-clockwise quads (vertex index = 4 x + 2 y + z).
    static let quads: [[Int]] = [[0, 1, 3, 2], [4, 6, 7, 5], [0, 4, 5, 1], [2, 3, 7, 6], [0, 2, 6, 4], [1, 5, 7, 3]]

    static func cubeSTL(size: Float, open: Bool = false) -> Data {
        boxSTL([size, size, size], open: open)
    }

    /// A box of the given extents (x, y, z) centred on the origin.
    static func boxSTL(_ extents: [Float], open: Bool = false, header: String = "fixture cube") -> Data {
        var triangles = quads.flatMap { [[$0[0], $0[1], $0[2]], [$0[0], $0[2], $0[3]]] }
        if open { triangles.removeFirst() }
        var data = Data(header.utf8)
        data.append(Data(repeating: 0x20, count: 80 - data.count))
        withUnsafeBytes(of: UInt32(triangles.count).littleEndian) { data.append(contentsOf: $0) }
        for triangle in triangles {
            var record = [Float](repeating: 0, count: 3)
            for index in triangle { record += zip(corners[index], extents).map { $0 * $1 / 2 } }
            for value in record { withUnsafeBytes(of: value.bitPattern.littleEndian) { data.append(contentsOf: $0) } }
            data.append(contentsOf: [0, 0])
        }
        return data
    }

    static func cubeOBJ(size: Float) -> String {
        corners.map { "v \($0[0] * size / 2) \($0[1] * size / 2) \($0[2] * size / 2)\n" }.joined()
            + quads.map { "f " + $0.map { String($0 + 1) }.joined(separator: " ") + "\n" }.joined()
    }
}
