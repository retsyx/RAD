import AppKit
import Foundation
import XCTest
@testable import RAD

final class ProjectAgentSessionTests: XCTestCase {
    @MainActor
    func testUploadsAreImmutableTaskScopedAndRejectFilesystemPaths() async throws {
        let first = try session(), second = try session()
        defer { first.cancel(); second.cancel() }
        let original = Data("result = box((1, 2, 3), center=(1 / 2, 2 / 2, 3 / 2))\n".utf8)
        let descriptor = try await first.uploadFile(data: original, name: "program.py")
        let identity = try XCTUnwrap(descriptor["fileID"] as? String)
        let downloaded = try await first.downloadFile(fileID: identity)
        XCTAssertEqual(downloaded.data, original)
        XCTAssertThrowsError(try second.download(fileID: identity))
        XCTAssertThrowsError(try first.download(fileID: "/etc/passwd"))
        XCTAssertThrowsError(try first.upload(data: original, name: "../program.py"))
        XCTAssertThrowsError(try first.upload(data: original, name: "bad\\program.py"))
        XCTAssertThrowsError(try first.upload(data: original, name: "bad\nprogram.py"))
        XCTAssertThrowsError(try first.upload(data: Data(), name: "empty.py"))
        do {
            _ = try await first.call(name: "rad_build", arguments: ["programID": "/etc/passwd"])
            XCTFail("An external tool must accept only known task file IDs.")
        } catch { XCTAssertFalse(error.localizedDescription.contains("Python runtime")) }
        XCTAssertEqual(try first.download(fileID: identity).data, original)
        let folder = first.workspaceURL
        first.cancel()
        XCTAssertFalse(FileManager.default.fileExists(atPath: folder.path))
        XCTAssertThrowsError(try first.download(fileID: identity))
        do {
            _ = try await first.candidate(artifactID: "current")
            XCTFail("A canceled task must not provide a commit candidate.")
        } catch {}
    }

    @MainActor
    func testCurrentCandidateUsesItsExactProgramAndSeparateDownloadHandles() async throws {
        let native = FileManager.default.temporaryDirectory.appendingPathComponent("RAD-Agent-Native-\(UUID().uuidString).rsdf")
        try EngineService.artifactMarker.write(to: native)
        defer { try? FileManager.default.removeItem(at: native) }
        let model = ModelSpec(title: "Current", summary: "A validated design.", units: "mm", script: "result = box((1, 1, 1), center=(1 / 2, 1 / 2, 1 / 2))\n")
        let preview = PreviewMesh(vertices: [], triangles: [], edges: [], bounds: [0, 0, 0, 1, 1, 1], volume: 1,
            area: 6, solids: 1, buildSeconds: 0, threads: 1,
            buildID: native.deletingPathExtension().path, artifactPath: native.path)
        let task = try ProjectAgentSession(python: "/missing/python", threads: 1, current: model, currentPreview: preview)
        defer { task.cancel() }
        let result = try await task.call(name: "rad_inspect", arguments: [:])
        let values = try metadata(result)
        let programID = try XCTUnwrap(values["programID"] as? String)
        XCTAssertNil(values["programPath"])
        XCTAssertNotNil(UUID(uuidString: programID))
        XCTAssertEqual(try task.download(fileID: programID).data, Data(model.script.utf8))
        let metadataID = try XCTUnwrap(values["metadataPath"] as? String)
        let bytes = try task.download(fileID: metadataID).data
        XCTAssertFalse(String(decoding: bytes, as: UTF8.self).contains(task.workspaceURL.path))
        let currentCandidate = try await task.candidate(artifactID: "current")
        XCTAssertEqual(currentCandidate.model, model)
        XCTAssertEqual(currentCandidate.preview.artifactPath, native.path)
        do {
            _ = try await task.call(name: "rad_import", arguments: ["path": programID])
            XCTFail("Only uploaded original files can become reference inputs.")
        } catch { XCTAssertTrue(error.localizedDescription.contains("Import original external reference files uploaded to this task")) }
        task.cancel()
        XCTAssertTrue(FileManager.default.fileExists(atPath: native.path), "Committed native geometry belongs to its preview, not the task workspace.")
    }

    func testExternalCatalogAddsRoutingAndOpaqueFilesWithoutChangingNativeCatalog() throws {
        let native = try AgentToolCatalog.native()
        let external = try AgentToolCatalog.external()
        XCTAssertEqual(native.tools.map(\.name), external.tools.map(\.name))
        let nativeSchema = try XCTUnwrap(native["rad_build"]).inputSchema
        let schema = try XCTUnwrap(external["rad_build"]).inputSchema
        XCTAssertTrue((nativeSchema["required"] as? [String] ?? []).contains("programPath"))
        XCTAssertEqual(Set(schema["required"] as? [String] ?? []), ["taskID", "programID"])
        for tool in external.tools {
            XCTAssertTrue((tool.inputSchema["required"] as? [String] ?? []).contains("taskID"))
            XCTAssertEqual(Set(tool.descriptor.keys), ["name", "description", "inputSchema", "annotations"])
            if let export = tool.host.exportPath {
                XCTAssertNil((tool.inputSchema["properties"] as? [String: Any])?[export])
            }
        }
        XCTAssertEqual(Set(native.tools.filter { $0.host.exportPath != nil }.map(\.name)), ["rad_export_3mf"])
    }

    @MainActor
    func testExplicitCandidateKeepsItsOwnColorsWhenMatchingProgramsShareNativeGeometry() async throws {
        let native = FileManager.default.temporaryDirectory.appendingPathComponent("RAD-Agent-Color-\(UUID().uuidString).rsdf")
        try EngineService.artifactMarker.write(to: native)
        defer { try? FileManager.default.removeItem(at: native) }
        let model = ModelSpec(title: "Current", summary: "Validated geometry shared by candidates.", units: "mm",
                              script: "result = box((1, 1, 1), center=(1 / 2, 1 / 2, 1 / 2))\n")
        let parts = [PartNode(id: "body", name: "Body", solidIndices: [0])]
        let preview = PreviewMesh(vertices: [], triangles: [], edges: [], bounds: [0, 0, 0, 1, 1, 1], volume: 1,
            area: 6, solids: 1, buildSeconds: 0, threads: 1,
            buildID: native.deletingPathExtension().path, parts: parts, artifactPath: native.path)
        let task = try ProjectAgentSession(python: "/missing/python", threads: 1, current: model, currentPreview: preview)
        defer { task.cancel() }
        let program = try task.upload(data: Data(model.script.utf8), name: "same-program.py")
        let programID = try XCTUnwrap(program["fileID"] as? String)
        let first = try metadata(try await task.call(name: "rad_build", arguments: ["programID": programID, "title": "First"]))
        let second = try metadata(try await task.call(name: "rad_build", arguments: ["programID": programID, "title": "Second"]))
        let firstID = try XCTUnwrap(first["artifactID"] as? String)
        let secondID = try XCTUnwrap(second["artifactID"] as? String)
        let red = PartColor(red: 1, green: 0, blue: 0), blue = PartColor(red: 0, green: 0, blue: 1)
        let green = PartColor(red: 0, green: 1, blue: 0)
        for (id, color) in [(firstID, red), (secondID, blue), ("current", green)] {
            _ = try await task.call(name: "rad_set_part_colors", arguments: ["artifactID": id, "changes": [
                ["partID": "body", "color": ["red": color.red, "green": color.green, "blue": color.blue]]
            ]])
        }
        let firstCandidate = try await task.candidate(artifactID: firstID)
        let secondCandidate = try await task.candidate(artifactID: secondID)
        let currentCandidate = try await task.candidate(artifactID: "current")
        XCTAssertEqual(firstCandidate.colorChanges, [CADPartColorChange(partID: "body", color: red)])
        XCTAssertEqual(secondCandidate.colorChanges, [CADPartColorChange(partID: "body", color: blue)])
        XCTAssertEqual(currentCandidate.colorChanges, [CADPartColorChange(partID: "body", color: green)])
        XCTAssertEqual(firstCandidate.preview.artifactPath, secondCandidate.preview.artifactPath,
                       "The regression must cover distinct reviewed colors on reused native geometry.")
    }

    @MainActor
    func testEvictedCandidatePreviewRebuildsWhenCommitted() async throws {
        guard ProcessInfo.processInfo.environment["RAD_RUN_ENGINE_TESTS"] == "1" else {
            throw XCTSkip("Set RAD_RUN_ENGINE_TESTS=1 for the native external CAD task workflow.")
        }
        let task = try ProjectAgentSession(python: ProcessInfo.processInfo.environment["RAD_CAD_PYTHON"] ?? EngineService.defaultPython,
            threads: 2, current: nil, currentPreview: nil, versions: .empty)
        defer { task.cancel() }
        var artifactIDs = [String]()
        for size in 1...(CADAgentTools.retainedCandidatePreviews + 1) {
            let source = Data("result = box((\(size), 1, 1), center=(\(size) / 2, 0.5, 0.5))\n".utf8)
            let uploaded = try await task.uploadFile(data: source, name: "box\(size).py")
            let fileID = try XCTUnwrap(uploaded["fileID"] as? String)
            let built = try metadata(try await task.call(name: "rad_build", arguments: ["programID": fileID, "title": "Box \(size)"]))
            artifactIDs.append(try XCTUnwrap(built["artifactID"] as? String))
        }
        let candidate = try await task.candidate(artifactID: artifactIDs[0])
        XCTAssertEqual(candidate.preview.volume, 1, accuracy: 1e-8)
        XCTAssertTrue(FileManager.default.fileExists(atPath: try XCTUnwrap(candidate.preview.artifactPath)))
    }

    @MainActor
    func testExternalNativeBuildCanCommitAndSurvivesTaskCleanup() async throws {
        guard ProcessInfo.processInfo.environment["RAD_RUN_ENGINE_TESTS"] == "1" else {
            throw XCTSkip("Set RAD_RUN_ENGINE_TESTS=1 for the native external CAD task workflow.")
        }
        let source = Data("result = box((2, 3, 4), center=(2 / 2, 3 / 2, 4 / 2))\n".utf8)
        let historical = ProjectRevision(model: ModelSpec(title: "Historical box", summary: "Saved complete recipe.",
            units: "mm", script: String(decoding: source, as: UTF8.self)))
        let task = try ProjectAgentSession(python: ProcessInfo.processInfo.environment["RAD_CAD_PYTHON"] ?? EngineService.defaultPython,
            threads: 2, current: nil, currentPreview: nil,
            versions: ProjectVersionContext(revisions: [historical], selectedRevisionID: nil))
        defer { task.cancel() }
        let uploaded = try await task.uploadFile(data: source, name: "box.py")
        let fileID = try XCTUnwrap(uploaded["fileID"] as? String)
        let built = try metadata(try await task.call(name: "rad_build", arguments: ["programID": fileID, "title": "External box"]))
        let artifactID = try XCTUnwrap(built["artifactID"] as? String)
        let candidate = try await task.candidate(artifactID: artifactID)
        XCTAssertEqual(candidate.model.script, String(decoding: source, as: UTF8.self))
        XCTAssertEqual(candidate.preview.volume, 24, accuracy: 1e-3)
        let replay = try metadata(try await task.call(name: "rad_build", arguments: ["programID": fileID, "versionID": historical.id.uuidString]))
        let historicalArtifactID = try XCTUnwrap(replay["artifactID"] as? String)
        do {
            _ = try await task.candidate(artifactID: historicalArtifactID)
            XCTFail("A historical build must not be committable.")
        } catch {
            XCTAssertTrue(error.localizedDescription.contains("without versionID"))
        }
        let export = try metadata(try await task.call(name: "rad_export_3mf", arguments: ["artifactID": artifactID]))
        let modelFileID = try XCTUnwrap(export["path"] as? String)
        let downloaded = try await task.downloadFile(fileID: modelFileID)
        let bytes = downloaded.data
        XCTAssertEqual(try EngineThreeMFExportTests.summary(bytes).volume, 24, accuracy: 0.05)
        let secondExport = try metadata(try await task.call(name: "rad_export_3mf", arguments: ["artifactID": artifactID]))
        XCTAssertNotEqual(secondExport["path"] as? String, modelFileID)
        let originalDownload = try await task.downloadFile(fileID: modelFileID)
        XCTAssertEqual(originalDownload.data, bytes)
        // A downloaded export never becomes a reference input, even renamed and uploaded again.
        do {
            _ = try await task.call(name: "rad_import", arguments: ["path": modelFileID])
            XCTFail("A generated 3MF export must not become an imported input.")
        } catch { XCTAssertTrue(error.localizedDescription.contains("cannot become a reference input")) }
        let reuploaded = try await task.uploadFile(data: bytes, name: "renamed-reference.stl")
        do {
            _ = try await task.call(name: "rad_import", arguments: ["path": try XCTUnwrap(reuploaded["fileID"] as? String)])
            XCTFail("Renaming and uploading generated geometry again must not make it an original reference.")
        } catch { XCTAssertTrue(error.localizedDescription.contains("cannot become a reference input")) }
        // An uploaded original mesh is imported, built with and committed with its input.
        let reference = try await task.uploadFile(data: MeshImportFixture.cubeSTL(size: 10), name: "fixture.stl")
        let imported = try metadata(try await task.call(name: "rad_import",
            arguments: ["path": try XCTUnwrap(reference["fileID"] as? String), "name": "Fixture"]))
        let assetID = try XCTUnwrap(imported["assetID"] as? String)
        let cutSource = try await task.uploadFile(data: Data("result = imported_assets[\"\(assetID)\"] - box(4)\n".utf8), name: "cut.py")
        let cut = try metadata(try await task.call(name: "rad_build",
            arguments: ["programID": try XCTUnwrap(cutSource["fileID"] as? String), "assetIDs": [assetID]]))
        let cutCandidate = try await task.candidate(artifactID: try XCTUnwrap(cut["artifactID"] as? String))
        XCTAssertEqual(cutCandidate.model.inputAssets?.map(\.name), ["Fixture"])
        XCTAssertEqual(cutCandidate.preview.volume, 1000 - 64, accuracy: 6)
        // Individual part exports name one 3MF handle per part.
        let parts = try metadata(try await task.call(name: "rad_export_3mf", arguments: ["artifactID": artifactID, "individualParts": true]))
        let files = try XCTUnwrap(parts["files"] as? [[String: Any]])
        XCTAssertEqual(files.count, 1)
        let partID = try XCTUnwrap(files.first?["path"] as? String)
        XCTAssertEqual(try task.download(fileID: partID).data.count, try XCTUnwrap(files.first?["byteCount"] as? Int))
        task.cancel()
        XCTAssertTrue(FileManager.default.fileExists(atPath: try XCTUnwrap(candidate.preview.artifactPath)))
    }

    @MainActor
    private func session() throws -> ProjectAgentSession {
        try ProjectAgentSession(python: "/missing/python", threads: 1, current: nil, currentPreview: nil)
    }

    private func metadata(_ output: ProjectAgentToolResult) throws -> [String: Any] {
        output.result.metadata
    }
}
