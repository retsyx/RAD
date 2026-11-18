import AppKit
import Metal
import XCTest
@testable import RAD

final class CADAgentAssemblyMotionTests: XCTestCase {
    @MainActor
    func testMalformedPoseArgumentsFailBeforeBuildingGeometry() async throws {
        let root = try directory()
        defer { try? FileManager.default.removeItem(at: root) }
        let tools = try CADAgentTools(directory: root, python: "/missing/python", threads: 1,
            current: nil, currentPreview: nil, engine: EngineService())
        let invalid: [[String: Any]] = [
            ["jointPositions": [1, 2]], ["jointPositions": ["hinge": true]],
            ["jointPositions": ["hinge": "30"]], ["jointPositions": ["hinge": Double.nan]],
            ["jointPositions": ["hinge": 1e10]], ["jointPositions": ["": 30]],
            ["drivingJoint": "hinge"], ["jointPositions": ["hinge": 30], "drivingJoint": "other"]
        ]
        for arguments in invalid {
            do { _ = try await tools.handle(name: "rad_render", arguments: arguments); XCTFail("Invalid pose accepted") }
            catch { XCTAssertTrue(error.localizedDescription.lowercased().contains("joint")) }
        }
    }

    @MainActor
    func testIndependentRenderReportsRequestedPoseWithoutChangingArtifact() async throws {
        guard MTLCreateSystemDefaultDevice() != nil else { throw XCTSkip("Metal is unavailable") }
        let root = try directory()
        defer { try? FileManager.default.removeItem(at: root) }
        let preview = try mesh()
        let tools = try CADAgentTools(directory: root, python: "/missing/python", threads: 1,
            current: .bracket, currentPreview: preview, engine: EngineService())
        let arguments: [String: Any] = ["view": "top", "width": 320, "height": 256]
        let originalResult = try await tools.handle(name: "rad_render", arguments: arguments)
        let original = try content(originalResult)
        let originalPNG = try XCTUnwrap(originalResult.image?.data)
        var positioned = arguments
        positioned["jointPositions"] = ["hinge": 90]
        positioned["drivingJoint"] = "hinge"
        let renderedResult = try await tools.handle(name: "rad_render", arguments: positioned)
        let rendered = try content(renderedResult)
        let positionedPNG = try XCTUnwrap(renderedResult.image?.data)
        XCTAssertEqual((original["jointPositions"] as? [String: Double])?["hinge"], 0)
        XCTAssertEqual((rendered["jointPositions"] as? [String: Double])?["hinge"], 90)
        XCTAssertNotEqual(originalPNG, positionedPNG)
        let summary = try content(try await tools.handle(name: "rad_inspect", arguments: [:]))
        XCTAssertEqual((summary["jointPositions"] as? [String: Double])?["hinge"], 0)
        XCTAssertEqual(summary["bounds"] as? [Double], preview.bounds)
        XCTAssertNil(summary["builtBounds"])
        XCTAssertEqual(preview.assemblyPose?.positions["hinge"], 0)
        XCTAssertTrue(preview.assemblyPose?.solidTransforms.isEmpty ?? false)
    }

    @MainActor
    func testMovedSummaryNamesConstructionBoundsAndExposesAssemblyDefinition() async throws {
        let root = try directory()
        defer { try? FileManager.default.removeItem(at: root) }
        var preview = try mesh()
        preview.assemblyPose = try XCTUnwrap(preview.assemblyMotion).solve(parts: try XCTUnwrap(preview.parts), positions: ["hinge": 45])
        let tools = try CADAgentTools(directory: root, python: "/missing/python", threads: 1,
            current: .bracket, currentPreview: preview, engine: EngineService())
        let summary = try content(try await tools.handle(name: "rad_inspect", arguments: [:]))
        XCTAssertNotNil(summary["assemblyMotion"])
        XCTAssertEqual((summary["jointPositions"] as? [String: Double])?["hinge"], 45)
        XCTAssertNil(summary["bounds"])
        XCTAssertNil(summary["size"])
        XCTAssertEqual(summary["builtBounds"] as? [Double], preview.bounds)
        XCTAssertEqual(summary["builtSize"] as? [Double], preview.size)
    }

    @MainActor
    func testNativePosedInspectionMeasurementAndExportUsePosedGeometry() async throws {
        guard ProcessInfo.processInfo.environment["RAD_RUN_ENGINE_TESTS"] == "1" else {
            throw XCTSkip("Set RAD_RUN_ENGINE_TESTS=1 for native assembly tool integration.")
        }
        let root = try directory()
        defer { try? FileManager.default.removeItem(at: root) }
        let program = """
        motion={'grounded':['base'],'connectors':[
            {'id':'base-pin','part':'base','origin':[0,0,0],'xAxis':[1,0,0],'zAxis':[0,1,0]},
            {'id':'arm-pin','part':'arm','origin':[0,0,0],'xAxis':[1,0,0],'zAxis':[0,1,0]}],
            'joints':[{'id':'hinge','name':'Hinge','type':'revolute','parent':'base-pin','child':'arm-pin','position':45}]}
        result = Assembly("Mechanism")
        result.add(box((2, 2, 1), center=(-3, -3, -0.5)), id="base", name="Base")
        result.add(cylinder(2, 5, center=(10, 0, 2.5)), id="arm", name="Rotor")
        result.motion = motion
        """
        let model = ModelSpec(title: "Mechanism", summary: "Revolute assembly", units: "mm", script: program)
        let engine = EngineService()
        let preview = try await engine.buildPreview(model, python: EngineService.defaultPython, threads: 2)
        let tools = try CADAgentTools(directory: root, python: EngineService.defaultPython, threads: 2,
            current: model, currentPreview: preview, engine: engine,
            partColors: ["arm": .init(red: 0.1, green: 0.8, blue: 0.2)])
        let inspected = try content(try await tools.handle(name: "rad_inspect", arguments: ["kind": "parts", "solidIndices": [1], "limit": 1]))
        XCTAssertEqual(inspected["coordinateSpace"] as? String, "currentAssemblyPose")
        let part = try XCTUnwrap((inspected["items"] as? [[String: Any]])?.first)
        XCTAssertEqual(part["partID"] as? String, "arm")
        XCTAssertEqual(part["name"] as? String, "Rotor")
        XCTAssertEqual((part["color"] as? [String: Double])?["green"], 0.8)
        let k = sqrt(0.5)
        let bounds = [8*k, -2, -12*k, 17*k, 2, -3*k]
        assertVector(try XCTUnwrap(part["bounds"] as? [Double]), bounds)

        let measured = try content(try await tools.handle(name: "rad_measure", arguments: ["solidIndices": [1]]))
        let metrics = try XCTUnwrap(measured["metrics"] as? [[String: Any]])
        let centroid = try XCTUnwrap(metrics.first { $0["id"] as? String == "parts:centroid" }?["values"] as? [Double])
        assertVector(centroid, [12.5*k, 0, -7.5*k])
        let exported = try content(try await tools.handle(name: "rad_export_stl", arguments: ["solidIndices": [1]]))
        let data = try Data(contentsOf: URL(fileURLWithPath: XCTUnwrap(exported["path"] as? String)))
        assertVector(try EngineSTLExportTests.summary(data).bounds, bounds)
        XCTAssertEqual(preview.assemblyPose?.positions["hinge"], 45)

        var revision = ProjectRevision(model: model, parts: preview.parts)
        revision.jointPositions = ["hinge": 90]
        let historyRoot = root.appendingPathComponent("history")
        try FileManager.default.createDirectory(at: historyRoot, withIntermediateDirectories: false)
        let history = try CADAgentTools(directory: historyRoot, python: EngineService.defaultPython, threads: 2,
            current: nil, currentPreview: nil, engine: EngineService(),
            versions: ProjectVersionContext(revisions: [revision], selectedRevisionID: nil))
        let saved = try content(try await history.handle(name: "rad_version", arguments: ["reference": "v1"]))
        let rebuilt = try content(try await history.handle(name: "rad_build", arguments: [
            "programPath": XCTUnwrap(saved["programPath"] as? String), "versionID": revision.id.uuidString]))
        XCTAssertEqual((rebuilt["jointPositions"] as? [String: Double])?["hinge"], 90)
        let historicalParts = try content(try await history.handle(name: "rad_inspect", arguments: [
            "artifactID": XCTUnwrap(rebuilt["artifactID"] as? String), "kind": "parts", "solidIndices": [1]]))
        let historicalPart = try XCTUnwrap((historicalParts["items"] as? [[String: Any]])?.first)
        assertVector(try XCTUnwrap(historicalPart["bounds"] as? [Double]), [0, -2, -12, 5, 2, -8])
        XCTAssertEqual(preview.assemblyPose?.positions["hinge"], 45)
    }

    @MainActor
    func testCoupledFollowersAndInvalidSweepsFailBeforeNativeWork() async throws {
        let root = try directory()
        defer { try? FileManager.default.removeItem(at: root) }
        var preview = try mesh()
        let parts = (preview.parts ?? []) + [PartNode(id: "wheel", name: "Wheel", solidIndices: [2])]
        let motion = AssemblyMotion(grounded: ["base"], connectors: [
            .init(id: "base-pin", part: "base", origin: [0, 0, 0]), .init(id: "arm-pin", part: "arm", origin: [0, 0, 0]),
            .init(id: "base-wheel", part: "base", origin: [0, 0, 0]), .init(id: "wheel-pin", part: "wheel", origin: [0, 0, 0])
        ], joints: [
            .init(id: "hinge", name: "Arm", type: .revolute, parent: "base-pin", child: "arm-pin", limits: [-180, 180]),
            .init(id: "spin", name: "Wheel", type: .revolute, parent: "base-wheel", child: "wheel-pin")
        ], couplings: [.linear(id: "belt", driver: "hinge", follower: "spin", ratio: 2)])
        preview.parts = parts; preview.solids = 3
        preview.assemblyMotion = motion
        preview.assemblyPose = try motion.solve(parts: parts)
        let tools = try CADAgentTools(directory: root, python: "/missing/python", threads: 1,
            current: .bracket, currentPreview: preview, engine: EngineService())
        let summary = try content(try await tools.handle(name: "rad_inspect", arguments: [:]))
        let definition = try XCTUnwrap(summary["assemblyMotion"] as? [String: Any])
        XCTAssertEqual((definition["couplings"] as? [[String: Any]])?.first?["follower"] as? String, "spin")
        let sweep: [String: Any] = ["joint": "hinge", "from": 0, "to": 90, "pairs": [["arm", "wheel"]]]
        let invalid: [(String, [String: Any], String)] = [
            ("rad_render", ["jointPositions": ["spin": 10]], "follows Arm"),
            ("rad_check_motion", sweep.merging(["joint": "spin"]) { $1 }, "follows Arm"),
            ("rad_check_motion", sweep.merging(["jointPositions": ["spin": 1]]) { $1 }, "follows Arm"),
            ("rad_check_motion", sweep.merging(["joint": "missing"]) { $1 }, "movable joint"),
            ("rad_check_motion", sweep.filter { $0.key != "joint" }, "joint"),
            ("rad_check_motion", sweep.merging(["from": Double.nan]) { $1 }, "from"),
            ("rad_check_motion", sweep.merging(["to": true]) { $1 }, "to"),
            ("rad_check_motion", sweep.merging(["steps": 1]) { $1 }, "range"),
            ("rad_check_motion", sweep.merging(["steps": 362]) { $1 }, "range"),
            ("rad_check_motion", sweep.merging(["pairs": []]) { $1 }, "pairs"),
            ("rad_check_motion", sweep.merging(["pairs": [["arm"]]]) { $1 }, "two different"),
            ("rad_check_motion", sweep.merging(["pairs": [["arm", "arm"]]]) { $1 }, "two different"),
            ("rad_check_motion", sweep.merging(["pairs": [["arm", "missing"]]]) { $1 }, "not a physical part"),
            ("rad_check_motion", sweep.merging(["extra": 1]) { $1 }, "unsupported")
        ]
        for (tool, arguments, expected) in invalid {
            do {
                _ = try await tools.handle(name: tool, arguments: arguments)
                XCTFail("\(tool) accepted \(arguments)")
            } catch {
                XCTAssertTrue(error.localizedDescription.contains(expected), "\(error.localizedDescription) lacks \(expected)")
            }
        }
    }

    @MainActor
    func testMotionSweepReportsClearanceContactInterferenceAndCoupledJoints() async throws {
        guard ProcessInfo.processInfo.environment["RAD_RUN_ENGINE_TESTS"] == "1" else {
            throw XCTSkip("Set RAD_RUN_ENGINE_TESTS=1 for native assembly tool integration.")
        }
        let root = try directory()
        defer { try? FileManager.default.removeItem(at: root) }
        let program = """
        motion={'grounded':['post'],'connectors':[
            {'id':'post-pin','part':'post','origin':[0,0,0]},{'id':'arm-pin','part':'arm','origin':[0,0,0]},
            {'id':'post-wheel','part':'post','origin':[0,-30,0]},{'id':'wheel-pin','part':'wheel','origin':[0,-30,0]}],
            'joints':[{'id':'hinge','name':'Arm','type':'revolute','parent':'post-pin','child':'arm-pin','limits':[0,90]},
                      {'id':'spin','name':'Wheel','type':'revolute','parent':'post-wheel','child':'wheel-pin'}],
            'couplings':[{'id':'belt','type':'linear','driver':'hinge','follower':'spin','ratio':-2}]}
        result = Assembly("Mechanism")
        result.add(box((2, 2, 2), center=(11, 6, 1)), id="post", name="Post")
        result.add(box((20, 2, 2), center=(10, 0, 1)), id="arm", name="Arm")
        result.add(cylinder(3, 2, center=(0, -30, 1)), id="wheel", name="Wheel")
        result.motion = motion
        """
        let model = ModelSpec(title: "Mechanism", summary: "Coupled sweep", units: "mm", script: program)
        let engine = EngineService()
        let preview = try await engine.buildPreview(model, python: EngineService.defaultPython, threads: 2)
        XCTAssertEqual(preview.assemblyMotion?.couplings.first?.ratio, -2)
        let tools = try CADAgentTools(directory: root, python: EngineService.defaultPython, threads: 2,
            current: model, currentPreview: preview, engine: engine)
        let report = try content(try await tools.handle(name: "rad_check_motion", arguments: [
            "joint": "hinge", "from": 0, "to": 90, "steps": 19, "pairs": [["arm", "post"], ["arm", "wheel"]]]))
        XCTAssertEqual(report["positionUnit"] as? String, "°")
        XCTAssertEqual(report["unreachablePositions"] as? [Double], [])
        let samples = try XCTUnwrap(report["samples"] as? [[String: Any]])
        XCTAssertEqual(samples.count, 19)
        let first = try XCTUnwrap(samples.first), last = try XCTUnwrap(samples.last)
        XCTAssertEqual(try XCTUnwrap(first["distances"] as? [Double])[0], 4, accuracy: 0.05)
        XCTAssertEqual(try XCTUnwrap(last["distances"] as? [Double])[0], 9, accuracy: 0.05)
        XCTAssertEqual(try XCTUnwrap((last["jointPositions"] as? [String: Double])?["spin"]), -180, accuracy: 1e-9)
        let pairs = try XCTUnwrap(report["pairs"] as? [[String: Any]])
        XCTAssertEqual(pairs[0]["parts"] as? [String], ["arm", "post"])
        XCTAssertEqual(try XCTUnwrap(pairs[0]["minimumDistance"] as? Double), 0, accuracy: 1e-9)
        let interference = try XCTUnwrap(pairs[0]["interferencePositions"] as? [Double])
        XCTAssertFalse(interference.isEmpty)
        XCTAssertTrue(interference.allSatisfy { (10...50).contains($0) }, "\(interference)")
        XCTAssertGreaterThan(try XCTUnwrap(pairs[0]["maximumOverlapVolume"] as? Double), 0)
        XCTAssertEqual(try XCTUnwrap(pairs[1]["minimumDistance"] as? Double), 26, accuracy: 0.05)
        XCTAssertEqual(pairs[1]["interferencePositions"] as? [Double], [])
        XCTAssertEqual(preview.assemblyPose?.positions["hinge"], 0)
    }

    /// Mesh-resolution comparison: values come from the SDF preview or export mesh.
    private func assertVector(_ actual: [Double], _ expected: [Double], file: StaticString = #filePath, line: UInt = #line) {
        XCTAssertEqual(actual.count, expected.count, file: file, line: line)
        for (a, b) in zip(actual, expected) { XCTAssertEqual(a, b, accuracy: 0.05, file: file, line: line) }
    }

    private func content(_ result: CADAgentToolResult) throws -> [String: Any] {
        result.metadata
    }

    private func directory() throws -> URL {
        let root = FileManager.default.temporaryDirectory.appendingPathComponent("RAD-agent-motion-\(UUID())")
        try FileManager.default.createDirectory(at: root, withIntermediateDirectories: false)
        return root
    }

    private func mesh() throws -> PreviewMesh {
        let parts = [PartNode(id: "base", name: "Base", solidIndices: [0]), PartNode(id: "arm", name: "Arm", solidIndices: [1])]
        let motion = AssemblyMotion(grounded: ["base"], connectors: [
            .init(id: "base-pin", part: "base", origin: [0, 0, 0]),
            .init(id: "arm-pin", part: "arm", origin: [0, 0, 0])],
            joints: [.init(id: "hinge", type: .revolute, parent: "base-pin", child: "arm-pin", limits: [-180, 180])])
        let base: [[Double]] = [[-1,-1,-1],[1,-1,-1],[1,1,-1],[-1,1,-1],[-1,-1,1],[1,-1,1],[1,1,1],[-1,1,1]]
        let arm: [[Double]] = [[3,-0.5,-0.5],[9,-0.5,-0.5],[9,0.5,-0.5],[3,0.5,-0.5],[3,-0.5,0.5],[9,-0.5,0.5],[9,0.5,0.5],[3,0.5,0.5]]
        let triangles = [[0,2,1],[0,3,2],[4,5,6],[4,6,7],[0,1,5],[0,5,4],[1,2,6],[1,6,5],[2,3,7],[2,7,6],[3,0,4],[3,4,7]]
        var preview = PreviewMesh(vertices: base + arm, triangles: triangles + triangles.map { $0.map { $0 + 8 } },
            edges: [], bounds: [-1,-1,-1,9,1,1], volume: 14, area: 50, solids: 2,
            buildSeconds: 0, threads: 1, buildID: "/session/assembly-motion", triangleSolids: Array(repeating: 0, count: 12) + Array(repeating: 1, count: 12), parts: parts)
        preview.assemblyMotion = motion
        preview.assemblyPose = try motion.solve(parts: parts)
        return preview
    }
}
