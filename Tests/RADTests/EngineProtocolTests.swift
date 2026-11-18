import Foundation
import XCTest
@testable import RAD

/// End-to-end cross-check of the app against the SDF worker (SDF-DESIGN.md,
/// "Worker ↔ app protocol"): one assembly with folders, colors, a printing
/// pin, a printing version and a revolute joint is built by the real worker,
/// then every native operation runs through `EngineService`. A raw run of the
/// worker compares each result's fields with the protocol the Swift decoders
/// implement, so a renamed or missing optional field cannot decode silently.
final class EngineProtocolTests: XCTestCase {
    static let program = """
    base = box((40, 30, 6), center=(0, 0, 3)) | cylinder(4, 16, center=(0, 0, 8))
    if rad.print:
        base = base - cylinder(1.5, 10, center=(12, 0, 3))
    arm = box((50, 12, 5), center=(20, 0, 18.5)) - cylinder(4.2, 18, center=(0, 0, 18))
    knob = sphere(3, center=(60, 0, 18.5))
    result = Assembly("Crosscheck")
    result.add(base, id="base", name="Base", color="#808080", print_down=(0, 0, -1))
    arms = result.folder("Arms", id="arms", color=(0.9, 0.2, 0.1))
    arms.add(arm, id="arm", name="Arm")
    arms.add(knob, id="knob", name="Knob", color="#112233")
    motion = Motion()
    motion.ground("base", "knob")
    motion.connector("base-pivot", "base", (0, 0, 16))
    motion.connector("arm-pivot", "arm", (0, 0, 16))
    motion.revolute("hinge", "base-pivot", "arm-pivot", name="Arm angle", limits=(-120, 120))
    result.motion = motion
    """

    /// Analytic volumes: the base plate and post, the arm with its 4.2 mm
    /// hole, the knob sphere, and the printing version's 1.5 mm hole.
    static let baseVolume = 40.0 * 30 * 6 + Double.pi * 16 * 10
    static let armVolume = 50.0 * 12 * 5 - Double.pi * 4.2 * 4.2 * 5
    static let knobVolume = 4.0 / 3 * Double.pi * 27
    static let printHoleVolume = Double.pi * 1.5 * 1.5 * 6

    static var spec: ModelSpec {
        ModelSpec(title: "Protocol cross-check", summary: "An arm on a hinged base beside a knob.", units: "mm", script: program)
    }

    private func requireRuntime() throws {
        guard ProcessInfo.processInfo.environment["RAD_RUN_ENGINE_TESTS"] == "1" else {
            throw XCTSkip("Set RAD_RUN_ENGINE_TESTS=1 to cross-check the app against the SDF worker.")
        }
    }

    private func folder() throws -> URL {
        let url = FileManager.default.temporaryDirectory.appendingPathComponent("RAD-protocol-\(UUID())", isDirectory: true)
        try FileManager.default.createDirectory(at: url, withIntermediateDirectories: false)
        addTeardownBlock { try? FileManager.default.removeItem(at: url) }
        return url
    }

    private static func rotationZ(_ degrees: Double) -> [Double] {
        let c = cos(degrees * .pi / 180), s = sin(degrees * .pi / 180)
        return [c, -s, 0, 0, s, c, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1]
    }

    private static func translation(_ x: Double, _ y: Double, _ z: Double) -> [Double] {
        [1, 0, 0, x, 0, 1, 0, y, 0, 0, 1, z, 0, 0, 0, 1]
    }

    private func assertVector(_ actual: [Double]?, _ expected: [Double], accuracy: Double,
                              _ message: String = "", file: StaticString = #filePath, line: UInt = #line) {
        guard let actual, actual.count == expected.count else {
            return XCTFail("Expected \(expected), got \(String(describing: actual)). \(message)", file: file, line: line)
        }
        for (value, wanted) in zip(actual, expected) {
            XCTAssertEqual(value, wanted, accuracy: accuracy, "\(actual) vs \(expected). \(message)", file: file, line: line)
        }
    }

    private func metric(_ report: CADMeasurementReport, _ id: String, file: StaticString = #filePath, line: UInt = #line) -> [Double]? {
        let values = report.metrics.first { $0.id == id }?.values
        if values == nil { XCTFail("Missing metric \(id) in \(report.metrics.map(\.id))", file: file, line: line) }
        return values
    }

    // MARK: Builds

    @MainActor
    func testBuildDecodesEveryPreviewFieldForTheDesignAndItsPrintingVersion() async throws {
        try requireRuntime()
        let engine = EngineService(), python = EngineService.defaultPython
        let preview = try await engine.buildPreview(Self.spec, python: python, threads: 2)
        XCTAssertNotNil(UUID(uuidString: preview.buildID))
        XCTAssertEqual(preview.solids, 3)
        XCTAssertEqual(preview.threads, 2)
        XCTAssertEqual(preview.validatedHash?.count, 64)
        XCTAssertEqual(preview.warnings, [])
        let artifact = try XCTUnwrap(preview.artifactPath)
        XCTAssertTrue(artifact.hasSuffix("/model.rsdf"))
        XCTAssertEqual(try Data(contentsOf: URL(fileURLWithPath: artifact)).prefix(4), Data("RSDF".utf8))
        XCTAssertEqual(preview.partRoots?.count, 3)

        // Mesher statistics: about 256 cells across the 83 mm design, closed and manifold.
        let mesh = try XCTUnwrap(preview.meshStatistics)
        XCTAssertEqual(mesh.cellSize, 83.0 / 256, accuracy: 0.01)
        XCTAssertEqual(mesh.boundaryEdges, 0)
        XCTAssertEqual(mesh.nonmanifoldEdges, 0)

        // Per-part statistics in built coordinates.
        let stats = try XCTUnwrap(preview.partStats)
        XCTAssertEqual(stats.map(\.solidIndex), [0, 1, 2])
        for (part, volume) in zip(stats, [Self.baseVolume, Self.armVolume, Self.knobVolume]) {
            XCTAssertEqual(part.volume, volume, accuracy: volume * 0.01, "solid \(part.solidIndex)")
            XCTAssertEqual(part.components, 1)
            XCTAssertGreaterThan(part.triangles, 0)
        }
        assertVector(stats[1].bounds, [-5, -6, 16, 45, 6, 21], accuracy: 0.05)
        assertVector(stats[2].centroid, [60, 0, 18.5], accuracy: 0.01)
        XCTAssertEqual(preview.volume, stats.reduce(0) { $0 + $1.volume }, accuracy: 1e-6)
        XCTAssertEqual(preview.area, stats.reduce(0) { $0 + $1.area }, accuracy: 1e-6)
        assertVector(preview.bounds, [-20, -15, 0, 63, 15, 21.5], accuracy: 0.05)

        // Display buffers: unit normals, every triangle and crease owned by a part.
        XCTAssertTrue(preview.hasValidNormals)
        XCTAssertEqual(preview.triangleSolids?.count, preview.triangleCount)
        XCTAssertEqual(Set(preview.triangleSolids ?? []), [0, 1, 2])
        XCTAssertEqual(preview.edgeSolids?.count, preview.edgeSegmentCount)
        XCTAssertGreaterThan(preview.edgeSegmentCount, 0, "Box edges and hole rims are creases.")
        for solid in 0..<3 {
            let owned = zip(preview.triangleSolids ?? [], 0...).filter { $0.0 == solid }.count
            XCTAssertEqual(owned, stats[solid].triangles)
        }

        // Hierarchy, colors (folders pass theirs on), printing pins and motion.
        let root = try XCTUnwrap(preview.parts?.first)
        XCTAssertEqual(root.name, "Crosscheck")
        let nodes = Dictionary(uniqueKeysWithValues: root.flattened.map { ($0.id, $0) })
        XCTAssertEqual(nodes["base"]?.solidIndices, [0])
        XCTAssertEqual(nodes["arms"]?.children?.map(\.id), ["arm", "knob"])
        XCTAssertNil(nodes["arms"]?.color)
        XCTAssertEqual(nodes["base"]?.color, PartColor(red: 128.0 / 255, green: 128.0 / 255, blue: 128.0 / 255))
        XCTAssertEqual(nodes["arm"]?.color, PartColor(red: 0.9, green: 0.2, blue: 0.1))
        XCTAssertEqual(nodes["knob"]?.color, PartColor(red: 0x11 / 255.0, green: 0x22 / 255.0, blue: 0x33 / 255.0))
        XCTAssertEqual(preview.printVersion, true)
        XCTAssertEqual(preview.printDown, [PrintDownPin(solid: 0, direction: [0, 0, -1])])
        let motion = try XCTUnwrap(preview.assemblyMotion)
        XCTAssertEqual(motion.grounded, ["base", "knob"])
        XCTAssertEqual(motion.joints.map(\.id), ["hinge"])
        XCTAssertEqual(motion.joints.first?.limits, [-120, 120])
        XCTAssertEqual(preview.assemblyPose?.positions["hinge"], 0)
        let turned = try motion.solve(parts: preview.parts ?? [], positions: ["hinge": 90])
        let arm = try XCTUnwrap(turned.solidTransforms[1])
        assertVector(AssemblySolidTransform.point([40, 0, 21], by: arm).map(abs), [0, 40, 21], accuracy: 1e-6)

        // The printing version keeps the parts and changes only the base.
        let printing = try await engine.buildPreview(Self.spec, print: true, python: python, threads: 2)
        XCTAssertNotEqual(printing.buildID, preview.buildID)
        XCTAssertNotEqual(printing.validatedHash, preview.validatedHash)
        XCTAssertNoThrow(try PrintLayout.validateParts(of: printing, matching: preview))
        let printed = try XCTUnwrap(printing.partStats)
        XCTAssertEqual(printed[0].volume, Self.baseVolume - Self.printHoleVolume, accuracy: Self.baseVolume * 0.01)
        XCTAssertEqual(printed[0].volume - stats[0].volume, -Self.printHoleVolume, accuracy: Self.printHoleVolume * 0.1)
        XCTAssertEqual(printed[1].volume, stats[1].volume, accuracy: 1e-9, "Unchanged parts mesh identically.")

        let runtime = try await engine.runtimeInfo(python: python, threads: 2)
        XCTAssertEqual(runtime, preview.runtime)
        XCTAssertFalse(runtime.kernel.isEmpty)
    }

    // MARK: Native operations

    @MainActor
    func testEveryNativeOperationAnswersThroughTheEngine() async throws {
        try requireRuntime()
        let engine = EngineService(), python = EngineService.defaultPython
        let preview = try await engine.buildPreview(Self.spec, python: python, threads: 2)
        let motion = try XCTUnwrap(preview.assemblyMotion)
        let turned = try motion.solve(parts: preview.parts ?? [], positions: ["hinge": 90]).solidTransforms

        // measure: two parts. The base's nearest point is its plate's top edge.
        let parts = try await engine.measure(preview, solids: [0, 2], python: python, threads: 2)
        XCTAssertEqual(metric(parts, "pair:distance")?.first ?? 0, (1600.0 + 12.5 * 12.5).squareRoot() - 3, accuracy: 0.01)
        XCTAssertEqual(metric(parts, "parts:count"), [2])
        XCTAssertEqual(metric(parts, "parts:volume")?.first ?? 0, Self.baseVolume + Self.knobVolume, accuracy: 80)
        XCTAssertEqual(parts.closestPoints?.count, 2)

        // measure: a picked point and a part at the hinge's quarter turn. The
        // worker snaps the point (built coordinates) and the app poses it.
        let picked = CADSurfacePoint(solidIndex: 1, point: [40, 0.2, 21.05], normal: [0, 0, 1])
        let posed = try await engine.measure(preview, points: [picked], solids: [2], solidTransforms: turned,
                                             python: python, threads: 2)
        let snapped = AssemblySolidTransform.point([40, 0.2, 21], by: turned[1]!)
        let toKnob = zip(snapped, [60.0, 0, 18.5]).map { $0 - $1 }
        XCTAssertEqual(metric(posed, "pair:distance")?.first ?? 0,
                       toKnob.reduce(0) { $0 + $1 * $1 }.squareRoot() - 3, accuracy: 0.002)
        assertVector(metric(posed, "point1:position"), snapped, accuracy: 1e-4)
        assertVector(metric(posed, "point1:normal"), [0, 0, 1], accuracy: 1e-6)
        assertVector(posed.closestPoints?.first, snapped, accuracy: 1e-4)
        assertVector(metric(posed, "pair:first"), snapped, accuracy: 1e-4)

        // measure: one point alone only snaps (the agent's point measurement).
        let single = try await engine.measure(preview, entities: [.point(solid: 1, point: [40, 0.2, 21.05])],
                                              solidTransforms: nil, python: python, threads: 2)
        XCTAssertEqual(single.metrics.count, 0)
        XCTAssertNil(single.closestPoints)
        assertVector(single.points?.first?.point, [40, 0.2, 21], accuracy: 1e-6)
        XCTAssertEqual(single.points?.first?.solidIndex, 1)

        // clearance: the arm turns around the post 0.2 mm away; the shifted knob interferes.
        let pairs = [[[0], [1]], [[1], [2]]]
        let samples = try await engine.clearance(preview, pairs: pairs, poses: [[:], turned, [2: Self.translation(-14, 0, 0)]],
                                                 python: python, threads: 2)
        XCTAssertEqual(samples.count, 3)
        for pose in samples { XCTAssertEqual(pose[0].distance, 0.2, accuracy: 0.002) }
        XCTAssertEqual(samples[0][1].distance, 12, accuracy: 0.002)
        XCTAssertNil(samples[0][1].overlapVolume)
        XCTAssertEqual(samples[1][1].distance, 51, accuracy: 0.002)
        XCTAssertEqual(samples[2][1].distance, 0)
        // A spherical cap 2 mm high of the 3 mm knob lies inside the arm.
        XCTAssertEqual(try XCTUnwrap(samples[2][1].overlapVolume), .pi * 4 * 7 / 3, accuracy: 1.5)

        // print_orient: the pinned base stays upright; the layout rests every part on the bed.
        let orientations = try await engine.printOrientations(preview, solids: [0, 1, 2], python: python, threads: 2)
        XCTAssertEqual(orientations.keys.sorted(), [0, 1, 2])
        assertVector(Array(try XCTUnwrap(orientations[0]).matrix.prefix(11)), [1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1], accuracy: 1e-9)
        let knob = try XCTUnwrap(orientations[2]).bounds
        assertVector((0..<3).map { knob[$0 + 3] - knob[$0] }, [6, 6, 6], accuracy: 0.05)
        let layout = PrintLayout.transforms(for: [0, 1, 2], orientations: orientations)
        XCTAssertEqual(layout.keys.sorted(), [0, 1, 2])
        for part in CADMeasurements.posedParts(try XCTUnwrap(preview.partStats), transforms: layout) {
            XCTAssertEqual(part.bounds[2], 0, accuracy: 0.05, "solid \(part.solidIndex) rests on the bed")
        }

        // export: whole, posed subset, printing layout and one file per part.
        let whole = try EngineSTLExportTests.summary(await engine.exportSTL(preview, python: python, threads: 2))
        XCTAssertEqual(whole.volume, preview.volume, accuracy: preview.volume * 0.01)
        let swung = try EngineSTLExportTests.summary(await engine.exportSTL(preview, solidIndices: [1], solidTransforms: turned,
                                                                            python: python, threads: 2))
        XCTAssertEqual(swung.volume, Self.armVolume, accuracy: Self.armVolume * 0.01)
        assertVector([swung.bounds[0], swung.bounds[3]], [-6, 6], accuracy: 0.05, "The arm exports at its pose.")
        let plate = try EngineSTLExportTests.summary(await engine.exportSTL(preview, solidTransforms: layout, python: python, threads: 2))
        XCTAssertEqual(plate.bounds[2], 0, accuracy: 0.05)
        let files = try await engine.exportSTLParts(preview, parts: [[0], [1, 2]], python: python, threads: 2)
        XCTAssertEqual(files.count, 2)
        XCTAssertEqual(try EngineSTLExportTests.summary(files[0]).volume, Self.baseVolume, accuracy: Self.baseVolume * 0.01)
        XCTAssertEqual(try EngineSTLExportTests.summary(files[1]).volume, Self.armVolume + Self.knobVolume,
                       accuracy: (Self.armVolume + Self.knobVolume) * 0.01)

        // validate: half the preview cell by default, or the requested cell.
        let validation = try await engine.validate(preview, python: python, threads: 2)
        XCTAssertTrue(validation.ok, "\(validation)")
        XCTAssertEqual(validation.cellSize, try XCTUnwrap(preview.meshStatistics).cellSize / 2, accuracy: 1e-12)
        XCTAssertEqual(validation.parts.map(\.previewVolume), preview.partStats?.map(\.volume))
        XCTAssertTrue(validation.parts.allSatisfy { $0.closed && $0.components == 1 && abs($0.volumeChange) < 0.02 })
        let coarse = try await engine.validate(preview, cellSize: 0.5, python: python, threads: 2)
        XCTAssertEqual(coarse.cellSize, 0.5)

        // probe: signed distances and normals of every part; rays report hits.
        let probe = try await engine.probe(preview, points: [[0, 0, 3], [60, 0, 30]], python: python, threads: 2)
        XCTAssertEqual(probe.solids, [0, 1, 2])
        XCTAssertEqual(probe.samples[0][0].distance, -3, accuracy: 1e-6)
        XCTAssertEqual(probe.samples[1][2].distance, 8.5, accuracy: 1e-6)
        assertVector(probe.samples[1][2].normal, [0, 0, 1], accuracy: 1e-4)
        XCTAssertNil(probe.samples[0][0].hit)
        let rays = try await engine.probe(preview, solids: [2], points: [[60, 0, 30], [60, 30, 30]],
                                          directions: [[0, 0, -1], [0, 0, -1]], maxDistance: 100, python: python, threads: 2)
        XCTAssertEqual(rays.solids, [2])
        XCTAssertEqual(rays.samples[0][0].hit, true)
        XCTAssertEqual(try XCTUnwrap(rays.samples[0][0].hitDistance), 8.5, accuracy: 1e-4)
        assertVector(rays.samples[0][0].hitPoint, [60, 0, 21.5], accuracy: 1e-4)
        XCTAssertEqual(rays.samples[1][0].hit, false)
        XCTAssertNil(rays.samples[1][0].hitPoint)
        XCTAssertEqual(try XCTUnwrap(rays.samples[1][0].hitDistance), 100, accuracy: 1e-9)
    }

    // MARK: Raw protocol fields

    /// Run the worker as the app does, outside the program sandbox, and
    /// return its parsed result.json (or error.json when it fails).
    private func runWorker(_ request: EngineWorkerRequest, python: String, graph: URL? = nil,
                           expectFailure: Bool = false) throws -> [String: Any] {
        // Results are read from the folder's real path, as the app reads them.
        let output = try EngineArtifact.canonicalDirectory(try folder())
        if let graph { try FileManager.default.copyItem(at: graph, to: output.appendingPathComponent("source.rsdf")) }
        try request.write(to: output)
        let launch = try EngineService.workerLaunch(python: python)
        let process = Process()
        process.executableURL = launch.runtime
        process.arguments = [launch.worker.path, "--request", output.appendingPathComponent("request.json").path,
                             "--output", output.path, "--threads", "2"]
        process.environment = try EngineService.workerEnvironment(folder: output, launch: launch)
        process.currentDirectoryURL = output
        process.standardOutput = FileHandle.nullDevice
        process.standardError = FileHandle.nullDevice
        try process.run()
        process.waitUntilExit()
        XCTAssertEqual(process.terminationStatus == 0, !expectFailure)
        let name = expectFailure ? "error.json" : "result.json"
        let data = try Data(contentsOf: output.appendingPathComponent(name))
        if !expectFailure, request.isBuild {
            // The raw result still decodes with the app's preview decoder.
            var decoded = try JSONDecoder().decode(PreviewMesh.self, from: data)
            try PreviewMeshBuffers.read(try XCTUnwrap(PreviewMeshBuffers.manifest(in: data)), in: output, into: &decoded)
            XCTAssertNoThrow(try ProjectStore.validatePreview(decoded))
        }
        return try XCTUnwrap(JSONSerialization.jsonObject(with: data) as? [String: Any])
    }

    private func keys(_ value: Any?) -> Set<String> { Set((value as? [String: Any])?.keys.map { $0 } ?? []) }
    private func first(_ value: Any?) -> Any? { (value as? [Any])?.first }
    /// The field names of each item of a list of lists.
    private func table(_ value: Any?) -> [[Set<String>]] { ((value as? [[Any]]) ?? []).map { $0.map { keys($0) } } }

    func testRawWorkerResultsCarryExactlyTheProtocolFields() async throws {
        try requireRuntime()
        let python = try await CADRuntimeManager.resolvePython(EngineService.defaultPython)
        func build(_ script: String) -> EngineWorkerRequest {
            .build(title: "Protocol fields", summary: "Raw worker results.", units: "mm", script: script, inputAssets: [])
        }
        let result = try runWorker(build(Self.program), python: python)
        let common: Set<String> = ["buildID", "artifactPath", "partRoots", "bounds", "volume", "area", "solids", "parts",
                                   "partStats", "mesh", "validatedHash", "buildSeconds", "threads", "runtime", "warnings",
                                   "meshBuffers"]
        XCTAssertEqual(keys(result), common.union(["assemblyMotion", "printVersion", "printDown"]))
        XCTAssertEqual(keys(first(result["partStats"])),
                       ["solid", "volume", "area", "bounds", "centroid", "triangles", "components"])
        XCTAssertEqual(keys(result["mesh"]), ["cellSize", "boundaryEdges", "nonmanifoldEdges"])
        XCTAssertEqual(keys(result["runtime"]), ["python", "numpy", "kernel"])
        XCTAssertEqual(keys(first(result["printDown"])), ["solid", "direction"])
        XCTAssertEqual(keys(result["assemblyMotion"]), ["grounded", "connectors", "joints", "couplings"])
        XCTAssertEqual(keys(result["meshBuffers"]), ["path", "sections"])
        XCTAssertEqual(keys((result["meshBuffers"] as? [String: Any])?["sections"]), PreviewMeshBuffers.sectionNames)
        let root = first(result["parts"]) as? [String: Any]
        XCTAssertEqual(keys(root), ["id", "name", "solidIndices", "children"])
        XCTAssertEqual(keys(first(root?["children"])), ["id", "name", "solidIndices", "children", "color"])
        XCTAssertEqual(keys(first(root?["children"]).flatMap { ($0 as? [String: Any])?["color"] }), ["red", "green", "blue"])
        // Optional facts are omitted when the program does not define them.
        let plain = try runWorker(build("result = [box(10), sphere(4, center=(20, 0, 0))]"), python: python)
        XCTAssertEqual(keys(plain), common)
        let referenced = try runWorker(build("""
            result = Assembly()
            result.add(box(10), id="case")
            result.add(sphere(4, center=(20, 0, 0)), id="board", reference=True)
            """), python: python)
        XCTAssertEqual(keys(referenced), common.union(["referenceSolids"]))
        XCTAssertEqual(referenced["referenceSolids"] as? [Int], [1])

        let artifact = try XCTUnwrap(result["artifactPath"] as? String)
        let source = EngineWorkerRequest.NativeSource(identity: artifact, partRoots: try XCTUnwrap(result["partRoots"] as? [Int]))
        let graph = URL(fileURLWithPath: artifact)
        func native(_ request: EngineWorkerRequest) throws -> [String: Any] { try runWorker(request, python: python, graph: graph) }

        XCTAssertEqual(keys(try runWorker(.runtime, python: python)), ["runtime"])
        let pair = try native(.measure(source, entities: [.point(solid: 1, point: [40, 0, 21.1]), .part(solid: 2)],
                                       solidTransforms: nil))
        XCTAssertEqual(keys(pair), ["metrics", "closestPoints", "points"])
        XCTAssertEqual(keys(first(pair["metrics"])), ["id", "label", "values", "unit"])
        XCTAssertEqual(keys(first(pair["points"])), ["solid", "point", "normal"])
        XCTAssertEqual(keys(try native(.measure(source, entities: [.part(solid: 0), .part(solid: 2)], solidTransforms: nil))),
                       ["metrics", "closestPoints"])
        let clearance = try native(.clearance(source, pairs: [[[1], [2]]], poses: [[], [
            AssemblySolidTransform(solid: 2, matrix: Self.translation(-14, 0, 0))]]))
        XCTAssertEqual(keys(clearance), ["samples", "tolerance"])
        XCTAssertEqual(table(clearance["samples"]), [[["distance", "closestPoints"]],
                                                     [["distance", "closestPoints", "overlapVolume"]]])
        let orientations = try native(.printOrientations(source, solidIndices: nil, printDown: []))
        XCTAssertEqual(keys(orientations), ["orientations"])
        XCTAssertEqual(keys(first(orientations["orientations"])), ["solid", "matrix", "bounds"])
        XCTAssertEqual(keys(try native(.export(source, .init(solidIndices: [2])))),
                       ["solids", "triangles", "cellSize", "tolerance", "meshedTriangles"])
        let files = try native(.export(source, .init(solidIndices: nil, individualParts: [[0], [1, 2]])))
        XCTAssertEqual(keys(files), ["solids", "triangles", "cellSize", "tolerance", "meshedTriangles", "partFiles"])
        XCTAssertEqual(keys(first(files["partFiles"])), ["path", "solidIndices"])
        let validation = try native(.validate(source, cellSize: nil))
        XCTAssertEqual(keys(validation), ["ok", "cellSize", "parts", "warnings"])
        XCTAssertEqual(keys(first(validation["parts"])), ["solid", "closed", "boundaryEdges", "nonmanifoldEdges",
                                                          "components", "volume", "previewVolume", "volumeChange"])
        let probe = try native(.probe(source, solids: nil, points: [[0, 0, 3]], directions: nil, maxDistance: nil))
        XCTAssertEqual(keys(probe), ["solids", "samples"])
        XCTAssertEqual(keys(first(first(probe["samples"]))), ["distance", "normal"])
        let rays = try native(.probe(source, solids: [2], points: [[60, 0, 30], [60, 30, 30]],
                                     directions: [[0, 0, -1], [0, 0, -1]], maxDistance: nil))
        XCTAssertEqual(table(rays["samples"]), [[["distance", "normal", "hit", "hitDistance", "hitPoint"]],
                                                [["distance", "normal", "hit", "hitDistance"]]])

        // Failure reports: the worker's classification, and the meshing stage's cell size.
        let program = try runWorker(build("result = undefined_name"), python: python, expectFailure: true)
        XCTAssertEqual(keys(program), ["formatVersion", "stage", "failureKind", "message"])
        XCTAssertEqual(program["failureKind"] as? String, "design")
        let thin = try runWorker(build("result = [box(100), box((10, 10, 0.01), center=(200, 0, 0))]"), python: python,
                                 expectFailure: true)
        XCTAssertEqual(keys(thin), ["formatVersion", "stage", "failureKind", "message", "cellSize"])
        XCTAssertEqual(thin["stage"] as? String, "preview_meshing")
    }
}

private extension EngineWorkerRequest {
    var isBuild: Bool { if case .build = self { true } else { false } }
}
