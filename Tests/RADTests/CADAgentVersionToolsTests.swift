import Foundation
import XCTest
@testable import RAD

final class CADAgentVersionToolsTests: XCTestCase {
    @MainActor
    func testVersionListingIsPaginatedMetadataOnlyWithoutStartingCADOrStagingFiles() async throws {
        let root = try temporary()
        defer { try? FileManager.default.removeItem(at: root) }
        let revisions = (1...105).map { index in
            ProjectRevision(model: model("Design \(index)", script: "private_program_\(index)"),
                            name: index == 3 ? "Rounded" : nil)
        }
        let tools = try makeTools(root, versions: ProjectVersionContext(revisions: revisions, selectedRevisionID: revisions[2].id))
        let first = try content(try await tools.handle(name: "rad_versions", arguments: [:]))
        let items = try XCTUnwrap(first["items"] as? [[String: Any]])
        XCTAssertEqual(first["total"] as? Int, 105)
        XCTAssertEqual(items.count, 50)
        XCTAssertEqual(first["hasMore"] as? Bool, true)
        XCTAssertEqual(items[2]["version"] as? String, "v3")
        XCTAssertEqual(items[2]["name"] as? String, "Rounded")
        XCTAssertEqual(items[2]["versionID"] as? String, revisions[2].id.uuidString)
        XCTAssertEqual(items[2]["isCurrent"] as? Bool, true)
        XCTAssertNotNil(items[2]["date"] as? String)
        let second = try content(try await tools.handle(name: "rad_versions", arguments: ["offset": 100, "limit": 100]))
        XCTAssertEqual((second["items"] as? [[String: Any]])?.count, 5)
        XCTAssertEqual(second["hasMore"] as? Bool, false)
        XCTAssertFalse(String(describing: first).contains("private_program"))
        XCTAssertTrue(try FileManager.default.contentsOfDirectory(atPath: root.path).isEmpty)
        for arguments: [String: Any] in [["offset": -1], ["offset": true], ["limit": 0], ["limit": 101], ["unexpected": 1]] {
            await assertFailure(tools, "rad_versions", arguments)
        }
        let empty = try makeTools(root)
        let emptyList = try content(try await empty.handle(name: "rad_versions", arguments: [:]))
        XCTAssertEqual(emptyList["total"] as? Int, 0)
    }

    @MainActor
    func testLookupUsesStableIDsAndExactNamesAndReturnsAmbiguousCandidatesWithoutGuessing() async throws {
        let root = try temporary()
        defer { try? FileManager.default.removeItem(at: root) }
        let first = ProjectRevision(model: model("First"), name: "Rounded")
        let second = ProjectRevision(model: model("Second"), name: "Rounded")
        let third = ProjectRevision(model: model("Third"), name: "Named third")
        let fourth = ProjectRevision(model: model("Fourth"), name: "v3")
        let fifth = ProjectRevision(model: model("Fifth"), name: first.id.uuidString)
        let tools = try makeTools(root, versions: ProjectVersionContext(revisions: [first, second, third, fourth, fifth], selectedRevisionID: third.id))
        for reference in ["Rounded", "v3"] {
            let result = try content(try await tools.handle(name: "rad_version", arguments: ["reference": reference]))
            XCTAssertEqual(result["status"] as? String, "ambiguous")
            XCTAssertEqual((result["candidates"] as? [[String: Any]])?.count, 2)
            XCTAssertNil(result["programPath"])
        }
        XCTAssertTrue(try FileManager.default.contentsOfDirectory(atPath: root.path).isEmpty)
        let explicit = try content(try await tools.handle(name: "rad_version", arguments: ["reference": first.id.uuidString.lowercased()]))
        XCTAssertEqual(explicit["versionID"] as? String, first.id.uuidString, "Stable UUID takes precedence over a colliding custom name")
        let named = try content(try await tools.handle(name: "rad_version", arguments: ["reference": "  Named third  "]))
        XCTAssertEqual(named["versionID"] as? String, third.id.uuidString)
        XCTAssertEqual(named["isCurrent"] as? Bool, true)
        let ordinal = try content(try await tools.handle(name: "rad_version", arguments: ["reference": "v2"]))
        XCTAssertEqual(ordinal["versionID"] as? String, second.id.uuidString)
        for reference in ["rounded", "v0", "v99", "v03", "", "   ", UUID().uuidString] {
            await assertFailure(tools, "rad_version", ["reference": reference])
        }
        await assertFailure(tools, "rad_version", ["reference": 3])
        await assertFailure(tools, "rad_version", ["reference": "v1", "unexpected": true])
    }

    @MainActor
    func testLookupStagesCompleteImmutableRecipesAndOnlyRequiredOriginalInputs() async throws {
        let root = try temporary()
        defer { try? FileManager.default.removeItem(at: root) }
        let source = mesh("historical original")
        let asset = CADInputAsset(id: "shared", name: "reference.obj", format: "obj", data: source)
        let program = "# complete historical program\n" + String(repeating: "# source must remain a file reference\n", count: 300) + "result = imported_assets['shared']\n"
        let part = PartNode(id: "part", name: "Original part", solidIndices: [0])
        let preview = PreviewMesh(vertices: [[987654, 0, 0]], triangles: [], edges: [], bounds: [0,0,0,1,1,1],
                                  volume: 1, area: 6, solids: 1, buildSeconds: 0, threads: 1,
                                  buildID: "/private/derived-preview", parts: [part], artifactPath: "/private/derived-cache/model.rsdf")
        var revision = ProjectRevision(model: model("Historical", script: program, assets: [asset]),
                                       preview: preview,
                                       partLayout: PartLayout(nodes: [part]), hiddenSolids: [0],
                                       runtime: CADRuntime(python: "3.12", numpy: "2.6", kernel: "7.8"), name: "Before")
        revision.jointPositions = ["hinge": 42]
        let context = ProjectVersionContext(revisions: [revision], selectedRevisionID: revision.id)
        revision.name = "After"
        revision.model.script = "mutated later"
        revision.model.inputAssets?[0].data = mesh("mutated later")
        let currentAsset = CADInputAsset(id: "shared", name: "current.stl", format: "stl", data: mesh("current original"))
        let current = model("Current", assets: [currentAsset])
        let tools = try makeTools(root, current: current, versions: context)
        let output = try content(try await tools.handle(name: "rad_version", arguments: ["reference": "v1"]))
        XCTAssertEqual(output["name"] as? String, "Before")
        XCTAssertEqual(output["jointPositions"] as? [String: Double], ["hinge": 42])
        XCTAssertEqual(try read(output, "programPath"), Data(program.utf8))
        let saved = try JSONDecoder().decode(ModelSpec.self, from: read(output, "modelPath"))
        XCTAssertEqual(saved.script, program)
        XCTAssertNil(saved.inputAssets)
        let manifest = try object(read(output, "inputsManifestPath"))
        let stagedAsset = try XCTUnwrap((manifest["inputAssets"] as? [[String: Any]])?.first)
        XCTAssertEqual(stagedAsset["assetID"] as? String, "shared")
        XCTAssertEqual(try read(stagedAsset, "path"), source)
        XCTAssertEqual(stagedAsset["programBinding"] as? String, "imported_assets[assetID]")
        XCTAssertEqual(try tools.resolveAssets(ids: ["shared"]), [currentAsset], "Historical IDs never overwrite the current registry")
        XCTAssertEqual((output["runtime"] as? [String: Any])?["numpy"] as? String, "2.6")
        XCTAssertNotNil(output["partsLayoutPath"] as? String)
        XCTAssertNotNil(output["partsPath"] as? String)
        let outputText = String(decoding: try JSONSerialization.data(withJSONObject: output), as: UTF8.self)
        for forbidden in ["source must remain", "987654", "derived-preview", "derived-cache", "never stage", source.base64EncodedString()] {
            XCTAssertFalse(outputText.contains(forbidden))
        }
        let allFiles = try FileManager.default.subpathsOfDirectory(atPath: root.path).filter { !$0.hasSuffix("/") }
        XCTAssertFalse(allFiles.contains { $0.contains("rsdf") || $0.contains("preview") })
        let permissions = try FileManager.default.attributesOfItem(atPath: XCTUnwrap(output["programPath"] as? String))[.posixPermissions] as? NSNumber
        XCTAssertEqual(permissions?.intValue, 0o600)
    }

    @MainActor
    func testRepeatedLookupReusesPathsAndRestoresModifiedFilesFromTheTaskSnapshot() async throws {
        let root = try temporary()
        defer { try? FileManager.default.removeItem(at: root) }
        let asset = CADInputAsset(id: "input", name: "original.obj", format: "obj", data: Data("original external mesh bytes".utf8))
        let first = ProjectRevision(model: model("First", assets: [asset]))
        let second = ProjectRevision(model: model("Second", assets: [asset]))
        let tools = try makeTools(root, versions: ProjectVersionContext(revisions: [first, second], selectedRevisionID: second.id))
        let original = try content(try await tools.handle(name: "rad_version", arguments: ["reference": "v1"]))
        let originalAsset = try XCTUnwrap((original["inputAssets"] as? [[String: Any]])?.first)
        let originalManifest = try read(original, "inputsManifestPath")
        for path in [try XCTUnwrap(original["programPath"] as? String), try XCTUnwrap(original["inputsManifestPath"] as? String), try XCTUnwrap(originalAsset["path"] as? String)] {
            try Data("agent modified this file".utf8).write(to: URL(fileURLWithPath: path))
        }
        let repeated = try content(try await tools.handle(name: "rad_version", arguments: ["reference": first.id.uuidString]))
        XCTAssertEqual(repeated["programPath"] as? String, original["programPath"] as? String)
        XCTAssertEqual(try read(repeated, "programPath"), Data(first.model.script.utf8))
        XCTAssertEqual(try read(repeated, "inputsManifestPath"), originalManifest)
        XCTAssertEqual(try read(originalAsset, "path"), asset.data)
        let other = try content(try await tools.handle(name: "rad_version", arguments: ["reference": "v2"]))
        let otherAsset = try XCTUnwrap((other["inputAssets"] as? [[String: Any]])?.first)
        XCTAssertEqual(otherAsset["path"] as? String, originalAsset["path"] as? String)
        let inputFolder = root.appendingPathComponent("context/versions/inputs")
        XCTAssertEqual(try FileManager.default.contentsOfDirectory(atPath: inputFolder.path).count, 1)
    }

    @MainActor
    func testHistoryStagingCannotFollowWorkspaceDirectoryLinksAndBuildContextsDoNotFallBack() async throws {
        let root = try temporary(), outside = try temporary()
        defer { try? FileManager.default.removeItem(at: root); try? FileManager.default.removeItem(at: outside) }
        let historical = CADInputAsset(id: "old", name: "old.obj", format: "obj", data: Data("old".utf8))
        let currentAsset = CADInputAsset(id: "current", name: "current.obj", format: "obj", data: Data("current".utf8))
        let revision = ProjectRevision(model: model("Old", assets: [historical]))
        let tools = try makeTools(root, current: model("Current", assets: [currentAsset]),
                                  versions: ProjectVersionContext(revisions: [revision], selectedRevisionID: nil))
        try FileManager.default.createDirectory(at: root.appendingPathComponent("context"), withIntermediateDirectories: true)
        try FileManager.default.createSymbolicLink(at: root.appendingPathComponent("context/versions"), withDestinationURL: outside)
        await assertFailure(tools, "rad_version", ["reference": "v1"])
        XCTAssertTrue(try FileManager.default.contentsOfDirectory(atPath: outside.path).isEmpty)
        try Data("result=box((1, 1, 1), center=(1 / 2, 1 / 2, 1 / 2))".utf8).write(to: root.appendingPathComponent("program.py"))
        await assertFailure(tools, "rad_build", ["programPath": "program.py", "versionID": revision.id.uuidString, "assetIDs": ["current"]])
        await assertFailure(tools, "rad_build", ["programPath": "program.py", "versionID": "v1"])
        XCTAssertThrowsError(try tools.resolveAssets(ids: ["old"]))
        XCTAssertEqual(try tools.resolveAssets(ids: ["current"]), [currentAsset])
        tools.cancel()
        do {
            _ = try await tools.handle(name: "rad_versions", arguments: [:])
            XCTFail("Cancelled history access must stop")
        } catch is CancellationError {} catch { XCTFail("Unexpected cancellation error: \(error)") }
    }

    private func model(_ title: String, script: String = "result = box(1)", assets: [CADInputAsset]? = nil) -> ModelSpec {
        ModelSpec(title: title, summary: "A saved design.", units: "mm", script: script, inputAssets: assets)
    }

    @MainActor
    private func makeTools(_ root: URL, current: ModelSpec? = nil, versions: ProjectVersionContext = .empty) throws -> CADAgentTools {
        try CADAgentTools(directory: root, python: "/missing/python", threads: 1, current: current,
                          currentPreview: nil, engine: EngineService(), versions: versions)
    }

    @MainActor
    private func assertFailure(_ tools: CADAgentTools, _ name: String, _ arguments: [String: Any], file: StaticString = #filePath, line: UInt = #line) async {
        do {
            _ = try await tools.handle(name: name, arguments: arguments)
            XCTFail("Invalid history access must fail", file: file, line: line)
        } catch {
            XCTAssertFalse(error.localizedDescription.contains("Python runtime not found"), "Arguments must fail before starting CAD", file: file, line: line)
        }
    }

    private func content(_ result: CADAgentToolResult) throws -> [String: Any] {
        result.metadata
    }
    private func object(_ data: Data) throws -> [String: Any] {
        try XCTUnwrap(JSONSerialization.jsonObject(with: data) as? [String: Any])
    }
    private func read(_ object: [String: Any], _ key: String) throws -> Data {
        try Data(contentsOf: URL(fileURLWithPath: XCTUnwrap(object[key] as? String)))
    }
    private func mesh(_ marker: String) -> Data {
        Data("# \(marker)\nv 0 0 0\nv 1 0 0\nv 0 1 0\nv 0 0 1\nf 1 3 2\nf 1 2 4\nf 1 4 3\nf 2 3 4\n".utf8)
    }
    private func temporary() throws -> URL {
        let root = FileManager.default.temporaryDirectory.appendingPathComponent("RAD-version-tools-\(UUID().uuidString)")
        try FileManager.default.createDirectory(at: root, withIntermediateDirectories: true)
        return root
    }
}
