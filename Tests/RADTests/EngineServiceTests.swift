import Foundation
import XCTest
@testable import RAD

/// Opt-in tests need the local CAD runtime and permission to start macOS's
/// sandbox, so ordinary swift test remains independent of native dependencies.
final class EngineServiceTests: XCTestCase {
    @MainActor
    func testCandidateValidationSurvivesFinalMetadataAndProjectRoundtrip() async throws {
        try requireRuntime()
        let candidate = ModelSpec(title: "Candidate", summary: "Build-time description", units: "mm",
                                  script: "result = box((2, 3, 4))")
        let engine = EngineService()
        let built = try await engine.buildPreview(candidate, python: EngineService.defaultPython, threads: 2)
        let hash = try XCTUnwrap(built.validatedHash)
        var final = candidate
        final.title = "Final design title"
        final.summary = "A different final description after reviewing the validated candidate."
        let revision = ProjectRevision(model: final, validatedHash: hash)
        let project = RADProject(messages: [], revisions: [revision], selectedRevisionID: revision.id)
        let saved = try ProjectStore.decode(ProjectStore.encode(project)).revisions[0]
        let replay = try await engine.buildPreview(saved.model, python: EngineService.defaultPython, threads: 2)
        XCTAssertEqual(replay.validatedHash, hash,
                       "Changing only final presentation metadata must not invalidate the reviewed candidate.")
        XCTAssertEqual(replay.volume, built.volume)
        XCTAssertEqual(replay.bounds, built.bounds)
        XCTAssertNotEqual(replay.buildID, built.buildID, "Every build has its own session identity.")
    }

    @MainActor
    func testPreparedBuildWorkersServeLaterBuildsInsideTheSandbox() async throws {
        try requireRuntime()
        let canary = FileManager.default.temporaryDirectory.appendingPathComponent("RAD-private-canary-\(UUID().uuidString).txt")
        try Data("private canary".utf8).write(to: canary)
        defer { try? FileManager.default.removeItem(at: canary) }
        let probe = ModelSpec(title: "Isolated CAD program", summary: "Sandbox verification box.", units: "mm",
                              script: try Self.sandboxProbe(canary))
        let engine = EngineService()
        func prepared() async throws {
            let deadline = ContinuousClock.now + .seconds(60)
            while !engine.hasPreparedBuildWorker, ContinuousClock.now < deadline { try await Task.sleep(for: .milliseconds(20)) }
            XCTAssertTrue(engine.hasPreparedBuildWorker)
        }
        engine.prepareBuildWorker(python: EngineService.defaultPython)
        try await prepared()
        let first = try await engine.buildPreview(probe, python: EngineService.defaultPython, threads: 2)
        XCTAssertEqual(engine.preparedBuilds, 1)
        XCTAssertEqual(first.volume, 1_000, accuracy: 10)
        try await prepared()
        let second = try await engine.buildPreview(probe, python: EngineService.defaultPython, threads: 2)
        XCTAssertEqual(engine.preparedBuilds, 2, "Each finished build prepares the next build's worker.")
        XCTAssertEqual(second.validatedHash, first.validatedHash)
        let folder = URL(fileURLWithPath: try XCTUnwrap(second.artifactPath)).deletingLastPathComponent()
        XCTAssertEqual(Set(try FileManager.default.contentsOfDirectory(atPath: folder.path)), ["model.rsdf", ".rad-build-lease"])
    }

    @MainActor
    func testDecodedBuildRetainsOnlyItsSDFGraph() async throws {
        try requireRuntime()
        let spec = ModelSpec(title: "Scratch cleanup", summary: "Requests and results are temporary.", units: "mm",
                             script: "result = box((2, 3, 4))")
        let preview = try await EngineService().buildPreview(spec, python: EngineService.defaultPython, threads: 2)
        let folder = URL(fileURLWithPath: try XCTUnwrap(preview.artifactPath)).deletingLastPathComponent()
        let names = Set(try FileManager.default.contentsOfDirectory(atPath: folder.path))
        XCTAssertEqual(names, ["model.rsdf", ".rad-build-lease"],
                       "Requests, results, logs and display buffers are not reused after decoding")
    }

    @MainActor
    func testBuildPublishesPartsMeshStatisticsAndItsGraph() async throws {
        try requireRuntime()
        let spec = ModelSpec(title: "Case", summary: "A plate and a post.", units: "mm", script: """
        result = Assembly("Case", id="case")
        result.add(box((7, 8, 2), center=(3.5, 4, 1)), id="base", name="Base")
        result.add(cylinder(2, 3, center=(20, 5, 3.5)), id="post", name="Post")
        """)
        let preview = try await EngineService().buildPreview(spec, python: EngineService.defaultPython, threads: 4)
        XCTAssertNotNil(UUID(uuidString: preview.buildID))
        let graph = URL(fileURLWithPath: try XCTUnwrap(preview.artifactPath))
        XCTAssertEqual(graph.lastPathComponent, "model.rsdf")
        XCTAssertEqual(try Data(contentsOf: graph).prefix(4), Data("RSDF".utf8))
        XCTAssertEqual(preview.partRoots?.count, 2)
        XCTAssertEqual(preview.parts?.first?.id, "case")
        XCTAssertEqual(preview.normals?.count, preview.vertices.count, "Every vertex has a unit shading normal.")
        XCTAssertTrue(preview.hasValidNormals)
        XCTAssertEqual(Set(preview.triangleSolids ?? []), [0, 1])
        let stats = try XCTUnwrap(preview.partStats)
        XCTAssertEqual(stats.map(\.solidIndex), [0, 1])
        XCTAssertEqual(stats[0].volume, 112, accuracy: 112 * 0.01)
        XCTAssertEqual(stats[1].volume, .pi * 12, accuracy: .pi * 12 * 0.02)
        XCTAssertEqual(stats.map(\.triangles).reduce(0, +), preview.triangleCount)
        XCTAssertGreaterThan(try XCTUnwrap(preview.meshStatistics).cellSize, 0)
        XCTAssertEqual(preview.meshStatistics?.boundaryEdges, 0)
        XCTAssertNotNil(preview.warnings)
        let runtime = try XCTUnwrap(preview.runtime)
        XCTAssertFalse(runtime.numpy.isEmpty)
        XCTAssertFalse(runtime.kernel.isEmpty)
    }

    @MainActor
    func testProgramErrorsAreRepairableDesignFailures() async throws {
        try requireRuntime()
        let spec = ModelSpec(title: "Broken", summary: "A program that raises.", units: "mm",
                             script: "result = box(1)\nraise ValueError('the slot is wider than the plate')")
        do {
            _ = try await EngineService().buildPreview(spec, python: EngineService.defaultPython, threads: 2)
            XCTFail("A failing program must not produce a preview.")
        } catch let failure as EngineBuildFailure {
            XCTAssertEqual(failure.kind, .design)
            XCTAssertTrue(EngineFailureDiagnostic.repairText(for: failure).contains("the slot is wider than the plate"))
        }
    }

    @MainActor
    func testPreviewExports3MFAndReusesSessionBytes() async throws {
        try requireRuntime()
        let spec = ModelSpec(title: "3MF assembly", summary: "A plate and a post.", units: "mm", script: """
        result = Assembly("Case", id="case")
        result.add(box((7, 8, 2), center=(3.5, 4, 1)), id="base", name="Base")
        result.add(cylinder(2, 3, center=(20, 5, 3.5)), id="post", name="Post")
        """)
        let service = EngineService()
        let preview = try await service.buildPreview(spec, python: EngineService.defaultPython, threads: 4)
        let graph = URL(fileURLWithPath: try XCTUnwrap(preview.artifactPath))
        let data = try await service.export3MF(preview, python: EngineService.defaultPython, threads: 4)
        let summary = try EngineThreeMFExportTests.summary(data)
        XCTAssertEqual(summary.volume, preview.volume, accuracy: preview.volume * 0.02)
        for (actual, expected) in zip(summary.bounds, preview.bounds) { XCTAssertEqual(actual, expected, accuracy: 0.1) }
        // A repeated export needs neither the graph nor a runtime.
        try FileManager.default.removeItem(at: graph)
        let cached = try await service.export3MF(preview, python: "/nonexistent/python", threads: 1)
        XCTAssertEqual(cached, data)
        let shared = try await EngineService().export3MF(preview, python: "/nonexistent/python", threads: 1)
        XCTAssertEqual(shared, data, "Every engine shares one bounded export cache.")
    }

    @MainActor
    func testExportRejectsUnregisteredNativeArtifactBeforeLaunchingPython() async throws {
        let preview = PreviewMesh(vertices: [], triangles: [], edges: [], bounds: [0,0,0,1,1,1], volume: 1,
                                  area: 1, solids: 1, buildSeconds: 0, threads: 1,
                                  buildID: UUID().uuidString, artifactPath: "/unrelated/model.rsdf", partRoots: [0])
        do {
            _ = try await EngineService().export3MF(preview, python: "/nonexistent/python", threads: 1)
            XCTFail("An unregistered artifact must never be read or sent to Python")
        } catch EngineExportError.nativeArtifactUnavailable {
            // AppState can rebuild an evicted model instead of losing export.
        }
    }

    func testWorkerLaunchUsesTheRealRuntimePathsTheSandboxProfileMatches() throws {
        try XCTSkipIf(ResourcePaths.kernelLibrary == nil, "Build the RADSDF kernel to resolve worker launches.")
        // The temporary directory lies under /var, a symbolic link to /private/var.
        let root = FileManager.default.temporaryDirectory.appendingPathComponent("RAD-launch-test-\(UUID().uuidString)", isDirectory: true)
        let bin = root.appendingPathComponent("venv/bin", isDirectory: true)
        try FileManager.default.createDirectory(at: bin, withIntermediateDirectories: true)
        defer { try? FileManager.default.removeItem(at: root) }
        let python = bin.appendingPathComponent("python")
        XCTAssertTrue(FileManager.default.createFile(atPath: python.path, contents: Data("#!/bin/sh\n".utf8),
                                                     attributes: [.posixPermissions: 0o755]))
        let launch = try EngineService.workerLaunch(python: python.path)
        let real = try EngineArtifact.canonicalDirectory(bin).appendingPathComponent("python").path
        XCTAssertEqual(launch.runtime.path, real)
        XCTAssertEqual(launch.actualRuntime.path, real)
        XCTAssertEqual(launch.interpreter.path, real)
    }

    @MainActor
    func testSavedReceiptReplaysTheSameMesh() async throws {
        try requireRuntime()
        let engine = EngineService()
        for radius in [1.0, 3.0] {
            let spec = ModelSpec(title: "Rounded slot", summary: "A ball with a rounded slot.", units: "mm", script: """
            slot = rect((15, 8.5), radius=\(radius)).extrude(200, center=True)
            result = sphere(50) - slot
            """)
            let preview = try await engine.buildPreview(spec, python: EngineService.defaultPython, threads: 2)
            XCTAssertEqual(preview.solids, 1)
            XCTAssertGreaterThan(preview.volume, 0)
            let receipt = try XCTUnwrap(preview.validatedHash)
            let revision = ProjectRevision(model: spec, validatedHash: receipt)
            let project = RADProject(messages: [], revisions: [revision], selectedRevisionID: revision.id)
            let saved = try XCTUnwrap(ProjectStore.decode(ProjectStore.encode(project)).revisions.first)
            XCTAssertNil(saved.preview)
            let replay = try await engine.buildPreview(saved.model, python: EngineService.defaultPython, threads: 4)
            XCTAssertEqual(replay.validatedHash, receipt)
            XCTAssertEqual(replay.volume, preview.volume, "Meshing is deterministic for any thread count.")
            XCTAssertEqual(replay.bounds, preview.bounds)
            XCTAssertEqual(replay.positions, preview.positions)
        }
    }

    @MainActor
    func testScriptSandboxBlocksUnrelatedFilesAndProcessExecution() async throws {
        try requireRuntime()
        let canary = FileManager.default.temporaryDirectory.appendingPathComponent("RAD-private-canary-\(UUID().uuidString).txt")
        try Data("private canary".utf8).write(to: canary)
        defer { try? FileManager.default.removeItem(at: canary) }
        let spec = ModelSpec(title: "Isolated CAD program", summary: "Sandbox verification box.", units: "mm",
                             script: try Self.sandboxProbe(canary))
        let mesh = try await EngineService().buildPreview(spec, python: EngineService.defaultPython, threads: 2)
        XCTAssertEqual(mesh.solids, 1)
        XCTAssertEqual(mesh.volume, 1_000, accuracy: 10)
    }

    /// A program that fails unless the sandbox denies an unrelated file, a
    /// subprocess and the network; it then builds a 10 mm cube.
    private static func sandboxProbe(_ canary: URL) throws -> String {
        // JSON quoting is also valid for this generated Python string literal.
        let encoder = JSONEncoder(); encoder.outputFormatting = [.withoutEscapingSlashes]
        let literal = String(decoding: try encoder.encode(canary.path), as: UTF8.self)
        return """
        import os, socket, subprocess
        try:
            open(\(literal)).read()
        except PermissionError:
            pass
        else:
            raise RuntimeError('unrelated file was readable')
        try:
            subprocess.run(['/bin/echo', 'should not run'], check=True)
        except (PermissionError, OSError):
            pass
        else:
            raise RuntimeError('external process was executable')
        sock = None
        try:
            sock = socket.socket()
            sock.connect(('127.0.0.1', 9))
        except PermissionError:
            pass
        except OSError as error:
            if error.errno not in (1, 13):
                raise RuntimeError('network was not sandbox denied: ' + str(error))
        else:
            raise RuntimeError('network was allowed')
        finally:
            if sock is not None:
                sock.close()
        assert not any(key in os.environ for key in ['OPENAI_API_KEY', 'CODEX_API_KEY', 'AWS_SECRET_ACCESS_KEY'])
        result = box(10)
        """
    }

    private func requireRuntime() throws {
        guard ProcessInfo.processInfo.environment["RAD_RUN_ENGINE_TESTS"] == "1" else {
            throw XCTSkip("Set RAD_RUN_ENGINE_TESTS=1 to verify the installed native CAD runtime and macOS sandbox.")
        }
    }
}
