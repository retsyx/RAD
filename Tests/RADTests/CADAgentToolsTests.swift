import AppKit
import Foundation
import XCTest
@testable import RAD

final class CADAgentToolsTests: XCTestCase {
    func testWorkspaceRejectsTraversalLinksAndKeepsAtomicWritesInsideItsRoot() throws {
        let root = temporary(), outside = temporary()
        defer { try? FileManager.default.removeItem(at: root); try? FileManager.default.removeItem(at: outside) }
        let workspace = try CADAgentWorkspace(directory: root)
        let original = outside.appendingPathComponent("original.txt")
        try Data("unchanged".utf8).write(to: original)
        XCTAssertThrowsError(try workspace.url("../escape.txt"))
        XCTAssertThrowsError(try workspace.read(original.path, maximumBytes: 100))
        XCTAssertThrowsError(try workspace.url("bad\0path"))
        let symlink = root.appendingPathComponent("symlink.txt")
        try FileManager.default.createSymbolicLink(at: symlink, withDestinationURL: original)
        XCTAssertThrowsError(try workspace.read("symlink.txt", maximumBytes: 100))
        let hardlink = root.appendingPathComponent("hardlink.txt")
        try FileManager.default.linkItem(at: original, to: hardlink)
        XCTAssertThrowsError(try workspace.read("hardlink.txt", maximumBytes: 100))
        try FileManager.default.createSymbolicLink(at: root.appendingPathComponent("linked"), withDestinationURL: outside)
        XCTAssertThrowsError(try workspace.write(Data("unsafe".utf8), path: "linked/changed.txt"))
        // Atomic output replaces a workspace link itself; it never follows the
        // link into the external source and changes its bytes.
        _ = try workspace.write(Data("workspace output".utf8), path: "symlink.txt")
        XCTAssertEqual(try Data(contentsOf: original), Data("unchanged".utf8))
        XCTAssertEqual(try workspace.read("symlink.txt", maximumBytes: 100), Data("workspace output".utf8))
        let nested = try workspace.write(Data("nested".utf8), path: "artifacts/a/metadata.txt")
        XCTAssertTrue(nested.path.hasPrefix(workspace.root.path + "/"))
        XCTAssertEqual(try workspace.read(nested.path, maximumBytes: 100), Data("nested".utf8))
        _ = try workspace.write(Data(), path: "requirements.txt")
        XCTAssertThrowsError(try workspace.read("requirements.txt", maximumBytes: 100))
        XCTAssertEqual(try workspace.read("requirements.txt", maximumBytes: 100, allowEmpty: true), Data())
    }

    @MainActor
    func testToolArgumentsFailBeforeRuntimeAndFinalReuseRequiresMatchingConstruction() async throws {
        let root = temporary()
        defer { try? FileManager.default.removeItem(at: root) }
        let native = root.appendingPathComponent("model.rsdf")
        try Data("RSDF".utf8).write(to: native)
        let first = CADInputAsset(id: "input", name: "input.obj", format: "obj", data: Data("source".utf8))
        let model = ModelSpec(title: "Current", summary: "Current shape", units: "mm", script: "result=imported_assets['input']", inputAssets: [first])
        let preview = PreviewMesh(vertices: [], triangles: [], edges: [], bounds: [0,0,0,1,1,1], volume: 1,
                                  area: 6, solids: 1, buildSeconds: 0, threads: 1,
                                  buildID: root.appendingPathComponent("build").path, artifactPath: native.path)
        let tools = try CADAgentTools(directory: root, python: "/missing/python", threads: 1, current: model,
                                      currentPreview: preview, engine: EngineService())
        let summary = try content(try await tools.handle(name: "rad_inspect", arguments: [:]))
        XCTAssertEqual(summary["artifactID"] as? String, "current")
        XCTAssertEqual(summary["solids"] as? Int, 1)
        XCTAssertEqual(try tools.resolveAssets(ids: ["input"]), [first])
        XCTAssertThrowsError(try tools.resolveAssets(ids: ["missing"]))
        XCTAssertThrowsError(try tools.resolveAssets(ids: ["input", "input"]))
        var renamed = model; renamed.title = "Renamed"; renamed.summary = "New prose"
        XCTAssertNotNil(tools.validatedPreview(for: renamed), "Metadata edits need no extra CAD build")
        renamed.script = "result=box((2, 2, 2), center=(2 / 2, 2 / 2, 2 / 2))"
        XCTAssertNil(tools.validatedPreview(for: renamed))
        for (name, args) in [("unknown", [:]), ("rad_build", ["programPath": "/outside/file.py"]),
                             ("rad_runtime", ["unexpected": true]), ("rad_render", ["width": true]) ] as [(String,[String:Any])] {
            do { _ = try await tools.handle(name: name, arguments: args); XCTFail("Invalid tool call must fail") }
            catch { XCTAssertFalse(error.localizedDescription.contains("Python runtime not found")) }
        }
        tools.cancel()
        do { _ = try await tools.handle(name: "rad_inspect", arguments: [:]); XCTFail("Stopped tools must not run") }
        catch is CancellationError {} catch { XCTFail("Unexpected cancellation error: \(error)") }
    }

    @MainActor
    func testPreviewReusePreservesTheInsertionOrderAvailableToPythonPrograms() throws {
        let root = temporary()
        defer { try? FileManager.default.removeItem(at: root) }
        let native = root.appendingPathComponent("model.rsdf")
        try Data("RSDF".utf8).write(to: native)
        let first = CADInputAsset(id: "first", name: "one.obj", format: "obj", data: Data("first input".utf8))
        let second = CADInputAsset(id: "second", name: "two.obj", format: "obj", data: Data("second input".utf8))
        let model = ModelSpec(title: "First input", summary: "Iterates complete Python input dictionary.", units: "mm",
                              script: "result = next(iter(imported_assets.values()))", inputAssets: [first, second])
        let preview = PreviewMesh(vertices: [], triangles: [], edges: [], bounds: [0,0,0,1,1,1], volume: 1,
                                  area: 6, solids: 1, buildSeconds: 0, threads: 1,
                                  buildID: root.appendingPathComponent("build").path, artifactPath: native.path)
        let tools = try CADAgentTools(directory: root, python: "/missing/python", threads: 1, current: model,
                                      currentPreview: preview, engine: EngineService())
        XCTAssertNotNil(tools.validatedPreview(for: model))
        var reordered = model
        reordered.inputAssets = [second, first]
        XCTAssertNil(tools.validatedPreview(for: reordered), "The same program can select a different shape after dictionary insertion order changes.")
        var renamed = model
        renamed.title = "Metadata change"
        renamed.inputAssets?[0].name = "renamed source.obj"
        XCTAssertNotNil(tools.validatedPreview(for: renamed), "Provenance names and prose do not affect the injected geometry.")
    }

    @MainActor
    func testModelingGuideDocumentsTheCompleteProgramInterface() throws {
        let guide = try CADAgentTools.modelingGuide()
        XCTAssertTrue(guide.hasPrefix(CADAgentTools.modelingGuideHeading + "\n"))
        for marker in ["RESPONSE CONTRACT", "FINAL RESPONSE", "TASK WORKSPACE:", "assetIDs\":[]"] {
            XCTAssertFalse(guide.contains(marker), "The staged guide is agent-neutral: \(marker)")
        }
        for forbidden in ["CadQuery", "OpenCascade", "OCP", "STEP", "BREP"] {
            XCTAssertFalse(guide.contains(forbidden), forbidden)
        }
        for imported in ["Imported meshes", "imported_assets[assetID]", "rad_import", "mesh_solid", "load_stl", "load_obj"] {
            XCTAssertTrue(guide.contains(imported), "The modeling guide must document imported meshes: \(imported)")
        }
        // Every exported program name and the public shape, assembly and
        // motion methods are documented with their names.
        var names: [String] = []
        for module in ["radsdf", "radsdf_assembly"] {
            let source = try String(contentsOf: ResourcePaths.root.appendingPathComponent("Engine/\(module).py"), encoding: .utf8)
            let list = try XCTUnwrap(source.range(of: "__all__ = [").map { source[$0.upperBound...] }?.prefix { $0 != "]" })
            names += list.split(separator: ",").map { $0.trimmingCharacters(in: CharacterSet(charactersIn: " \n\"")) }.filter { !$0.isEmpty }
        }
        XCTAssertGreaterThan(names.count, 40)
        names += ["translate", "moved", "rotate", "rotate_x", "rotate_y", "rotate_z", "orient", "scale", "transform", "mirror",
                  "mirrored_copy", "offset", "shell", "cut", "elongate", "twist", "linear_array", "grid_array", "polar_array",
                  "extrude", "extrude_between", "revolve", "loft_to", "outline", "bounds", "evaluate", "contains", "lipschitz",
                  "add", "folder", "add_assembly", "parts", "motion", "from_matrix", "ground", "connector", "fixed", "revolute",
                  "slider", "couple", "definition", "print_down", "rad.print", "smooth=", "crank_at_center"]
        for name in names {
            XCTAssertTrue(guide.contains(name), "The modeling guide must document \(name).")
        }
    }

    @MainActor
    func testInspectMeasureValidateAndExportArgumentsFailBeforeNativeWork() async throws {
        let root = temporary()
        defer { try? FileManager.default.removeItem(at: root) }
        let parts = [PartNode(id: "case", name: "Case", solidIndices: [], children: [
            PartNode(id: "base", name: "Base", solidIndices: [0]), PartNode(id: "lid", name: "Lid", solidIndices: [1])])]
        let stats = [CADPart(solidIndex: 0, area: 52, volume: 24, bounds: [0, 0, 0, 2, 3, 4], centroid: [1, 1.5, 2], triangles: 12, components: 1),
                     CADPart(solidIndex: 1, area: 6, volume: 1, bounds: [5, 0, 0, 6, 1, 1], centroid: [5.5, 0.5, 0.5], triangles: 12, components: 1)]
        let preview = PreviewMesh(vertices: [], triangles: [], edges: [], bounds: [0, 0, 0, 6, 3, 4], volume: 25, area: 58,
                                  solids: 2, buildSeconds: 0, threads: 1, buildID: "synthetic", parts: parts, partStats: stats)
        let model = ModelSpec(title: "Case", summary: "Two parts", units: "mm", script: "result = [box(1), box(1)]")
        let tools = try CADAgentTools(directory: root, python: "/missing/python", threads: 1, current: model,
                                      currentPreview: preview, engine: EngineService())
        let summary = try await tools.handle(name: "rad_inspect", arguments: [:]).metadata
        XCTAssertEqual((summary["parts"] as? [[String: Any]])?.compactMap { $0["partID"] as? String }, ["base", "lid"])
        let page = try await tools.handle(name: "rad_inspect", arguments: ["kind": "parts", "partIDs": ["lid"]]).metadata
        let item = try XCTUnwrap((page["items"] as? [[String: Any]])?.first)
        XCTAssertEqual(page["total"] as? Int, 1)
        XCTAssertEqual(item["solidIndex"] as? Int, 1)
        XCTAssertEqual(item["name"] as? String, "Lid")
        XCTAssertEqual(item["pieces"] as? Int, 1)
        let folder = try await tools.handle(name: "rad_inspect", arguments: ["kind": "parts", "partIDs": ["case"], "solidIndices": [0]]).metadata
        XCTAssertEqual(folder["total"] as? Int, 2, "A folder selects its physical parts.")
        let motion = try await tools.handle(name: "rad_inspect", arguments: ["kind": "motion"]).metadata
        XCTAssertEqual(motion["hasMotion"] as? Bool, false)
        let failures: [(String, [String: Any], String)] = [
            ("rad_inspect", ["kind": "vertices"], "Inspect summary, parts, motion, print or probe"),
            ("rad_inspect", ["kind": "parts", "points": [[0, 0, 0]]], "apply only to kind probe"),
            ("rad_inspect", ["kind": "parts", "partIDs": ["missing"]], "not a part or folder"),
            ("rad_inspect", ["kind": "probe"], "Probe at least one world point"),
            ("rad_inspect", ["kind": "probe", "points": [[0, 0]]], "three finite millimetre coordinates"),
            ("rad_inspect", ["kind": "probe", "points": [[0, 0, 0]], "directions": [[0, 0, 0]]], "nonzero ray direction"),
            ("rad_inspect", ["kind": "probe", "points": [[0, 0, 0]], "maxDistance": -1], "‘maxDistance’"),
            ("rad_measure", [:], "Measure at least one part or surface point"),
            ("rad_measure", ["points": [["point": [0, 0, 0]], ["point": [1, 0, 0]], ["point": [2, 0, 0]], ["point": [3, 0, 0]]]], "between one and three"),
            ("rad_measure", ["points": [["point": [0, 0, 0], "solid": 0]]], "unsupported arguments"),
            ("rad_measure", ["points": [["point": [0, 0, 0], "partID": "case"]]], "Points snap to one physical part"),
            ("rad_measure", ["points": [["point": [500, 500, 500]]]], "is not near any part"),
            ("rad_validate", ["cellSize": 0], "must be positive"),
            ("rad_verify", ["artifactID": "current", "programPath": "program.py"], "not both"),
            ("rad_verify", ["assetIDs": [String]()], "apply when verifying a program file"),
            ("rad_verify", ["artifactID": "missing"], "unavailable in this task"),
            ("rad_verify", ["cellSize": 1], "unsupported arguments"),
            ("rad_check_motion", ["joint": "j", "from": 0, "to": 1, "pairs": [["base", "lid"]], "requiredClearance": -1], "‘requiredClearance’"),
            ("rad_export_stl", ["printLayout": true, "jointPositions": ["j": 1]], "printLayout without jointPositions"),
            ("rad_export_stl", ["jointPositions": ["j": 1]], "no assembly joints"),
            ("rad_import", ["path": "part.step"], "STL (.stl) or OBJ (.obj)")
        ]
        for (name, arguments, message) in failures {
            do {
                _ = try await tools.handle(name: name, arguments: arguments)
                XCTFail("\(name) \(arguments) must fail")
            } catch {
                XCTAssertTrue(error.localizedDescription.contains(message), "\(name): \(error.localizedDescription)")
            }
        }
    }

    @MainActor
    func testScreenshotSelectionNamesPartsAndPosesPointsForMeasurement() throws {
        let parts = [PartNode(id: "case", name: "Case", solidIndices: [], children: [
            PartNode(id: "base", name: "Base", solidIndices: [0]), PartNode(id: "lid", name: "Lid", solidIndices: [1])])]
        let metadata: [String: Any] = [
            "selectedSolidIndices": [0],
            "solidTransforms": [["solid": 1, "matrix": [0, -1, 0, 10, 1, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1]]],
            "selectedPoints": [["solidIndex": 1, "point": [1.0, 2.0, 3.0], "normal": [1.0, 0.0, 0.0]],
                               ["solidIndex": 0, "point": [4, 5, 6], "normal": [0, 0, 1]]]]
        let identified = CADAgentTools.identifiedSelection(metadata, parts: parts)
        XCTAssertEqual((identified["selectedParts"] as? [[String: Any]])?.first?["partID"] as? String, "base")
        let points = try XCTUnwrap(identified["selectedPoints"] as? [[String: Any]])
        XCTAssertEqual(points[0]["partID"] as? String, "lid")
        XCTAssertEqual(points[0]["point"] as? [Double], [1, 2, 3], "Built coordinates are kept.")
        XCTAssertEqual(points[0]["worldPoint"] as? [Double], [8, 1, 3])
        XCTAssertEqual(points[0]["worldNormal"] as? [Double], [0, 1, 0])
        XCTAssertEqual(points[1]["worldPoint"] as? [Double], [4, 5, 6])
        XCTAssertEqual(points[1]["name"] as? String, "Base")
    }

    @MainActor
    func testRuntimeStagesTheModelingGuideInTheWorkspace() throws {
        let guide = try CADAgentTools.modelingGuide()
        let instructions = try String(contentsOf: try XCTUnwrap(ResourcePaths.resource("model-instructions.md")), encoding: .utf8)
        XCTAssertTrue(instructions.hasSuffix(guide), "The guide is the end of the model instructions.")
        XCTAssertTrue(instructions.contains("RESPONSE CONTRACT"))
        XCTAssertLessThan(instructions.utf8.count, CodexService.maximumRequestBytes / 4)
    }

    @MainActor
    func testNativeAgentBuildInspectProbeMeasureValidateRenderAndExport() async throws {
        guard ProcessInfo.processInfo.environment["RAD_RUN_ENGINE_TESTS"] == "1" else {
            throw XCTSkip("Set RAD_RUN_ENGINE_TESTS=1 to exercise the complete native agent CAD tool interface.")
        }
        let root = temporary()
        defer { try? FileManager.default.removeItem(at: root) }
        let program = """
        plate = difference(box((20, 10, 4), center=(0, 0, 2)), cylinder(2, 10, center=(5, 0, 2)))
        result = Assembly('Test', id='test')
        result.add(plate, id='plate', name='Plate', color='#336699')
        result.add(box(2, center=(0, 0, 10)), id='block', name='Block', print_down=(0, 0, 1))
        """
        try Data(program.utf8).write(to: root.appendingPathComponent("program.py"))
        let engine = EngineService()
        let tools = try CADAgentTools(directory: root, python: EngineService.defaultPython, threads: 4,
                                      current: nil, currentPreview: nil, engine: engine)
        let runtime = try await tools.handle(name: "rad_runtime", arguments: [:]).metadata
        XCTAssertEqual(runtime["pythonExecutable"] as? String, EngineService.defaultPython)
        XCTAssertNotNil((runtime["runtime"] as? [String: Any])?["numpy"])
        XCTAssertNotNil((runtime["runtime"] as? [String: Any])?["kernel"])
        let guide = try String(contentsOfFile: try XCTUnwrap(runtime["apiGuidePath"] as? String), encoding: .utf8)
        XCTAssertEqual(guide, try CADAgentTools.modelingGuide())

        let built = try await tools.handle(name: "rad_build", arguments: ["programPath": "program.py", "title": "Plate and block"]).metadata
        let id = try XCTUnwrap(built["artifactID"] as? String)
        XCTAssertEqual(built["solids"] as? Int, 2)
        XCTAssertEqual(built["warnings"] as? [String], [])
        XCTAssertGreaterThan(try XCTUnwrap(built["meshCellSize"] as? Double), 0)
        let builtParts = try XCTUnwrap(built["parts"] as? [[String: Any]])
        XCTAssertEqual(builtParts.compactMap { $0["partID"] as? String }, ["plate", "block"])
        XCTAssertEqual(try XCTUnwrap(builtParts[1]["volume"] as? Double), 8, accuracy: 1e-3)
        XCTAssertNotNil(tools.validatedPreview(for: ModelSpec(title: "Final", summary: "Final", units: "mm", script: program)))

        let print = try await tools.handle(name: "rad_inspect", arguments: ["artifactID": id, "kind": "print"]).metadata
        XCTAssertEqual((print["layout"] as? [[String: Any]])?.count, 2)
        XCTAssertEqual(try XCTUnwrap((print["plateBounds"] as? [Double])?[2]), 0, accuracy: 0.05)
        XCTAssertEqual((print["printDown"] as? [[String: Any]])?.first?["partID"] as? String, "block")

        // Probe the plate: inside, in the hole and a ray from inside the
        // plate to the hole wall at x = 3.
        let probe = try await tools.handle(name: "rad_inspect", arguments: [
            "artifactID": id, "kind": "probe", "partIDs": ["plate"],
            "points": [[-5, 0, 2], [5, 0, 2]], "directions": [[1, 0, 0], [0, 0, 1]]]).metadata
        let rows = try XCTUnwrap(probe["points"] as? [[String: Any]])
        let inside = try XCTUnwrap((rows[0]["parts"] as? [[String: Any]])?.first)
        XCTAssertEqual(try XCTUnwrap(inside["distance"] as? Double), -2, accuracy: 1e-6)
        XCTAssertEqual(inside["partID"] as? String, "plate")
        XCTAssertEqual(try XCTUnwrap(inside["hitDistance"] as? Double), 8, accuracy: 1e-3)
        let hole = try XCTUnwrap((rows[1]["parts"] as? [[String: Any]])?.first)
        XCTAssertEqual(try XCTUnwrap(hole["distance"] as? Double), 2, accuracy: 1e-6)
        XCTAssertEqual(hole["hit"] as? Bool, false, "A ray up the through-hole meets nothing.")

        // A plate face point snaps in place; the nearest part is found without a partID.
        let pair = try await tools.handle(name: "rad_measure", arguments: [
            "artifactID": id, "points": [["point": [10, 0, 2]]], "partIDs": ["block"]]).metadata
        let metrics = try XCTUnwrap(pair["metrics"] as? [[String: Any]])
        let distance = try XCTUnwrap((metrics.first { $0["id"] as? String == "pair:distance" }?["values"] as? [Double])?.first)
        XCTAssertEqual(distance, sqrt(130), accuracy: 1e-3)
        let snapped = try XCTUnwrap((pair["points"] as? [[String: Any]])?.first)
        XCTAssertEqual(snapped["partID"] as? String, "plate")
        XCTAssertEqual(try XCTUnwrap(snapped["snapDistance"] as? Double), 0, accuracy: 1e-6)
        let across = try await tools.handle(name: "rad_measure", arguments: [
            "artifactID": id, "points": [["point": [10, 0, 2]], ["point": [-10, 0, 2], "partID": "plate"]]]).metadata
        let span = try XCTUnwrap(((across["metrics"] as? [[String: Any]])?.first { $0["id"] as? String == "pair:distance" })?["values"] as? [Double])
        XCTAssertEqual(span[0], 20, accuracy: 1e-6)

        let valid = try await tools.handle(name: "rad_validate", arguments: ["artifactID": id]).metadata
        XCTAssertEqual(valid["valid"] as? Bool, true)
        XCTAssertEqual((valid["parts"] as? [[String: Any]])?.count, 2)

        let image = try await tools.handle(name: "rad_render", arguments: ["artifactID": id, "view": "front", "width": 400, "height": 300])
        let png = try Data(contentsOf: URL(fileURLWithPath: try XCTUnwrap(image.image?.path)))
        XCTAssertEqual(image.image?.data, png)
        XCTAssertEqual(try XCTUnwrap(NSBitmapImageRep(data: png)).pixelsWide, 400)

        let exported = try await tools.handle(name: "rad_export_stl", arguments: ["artifactID": id, "partIDs": ["block"], "path": "exports/block.stl"]).metadata
        let exportedPath = try XCTUnwrap(exported["path"] as? String)
        XCTAssertTrue(exportedPath.hasPrefix(try EngineArtifact.canonicalDirectory(root).path + "/"))
        let block = try EngineSTLExportTests.summary(Data(contentsOf: URL(fileURLWithPath: exportedPath)))
        XCTAssertEqual(block.volume, 8, accuracy: 1e-3)
        XCTAssertEqual(block.bounds[2], 9, accuracy: 1e-3)
        let individual = try await tools.handle(name: "rad_export_stl", arguments: ["artifactID": id, "individualParts": true]).metadata
        let files = try XCTUnwrap(individual["files"] as? [[String: Any]])
        XCTAssertEqual(files.compactMap { ($0["path"] as? String).map { URL(fileURLWithPath: $0).lastPathComponent } }, ["Plate.stl", "Block.stl"])
        XCTAssertEqual(files.compactMap { $0["partID"] as? String }, ["plate", "block"])
        let plate = try XCTUnwrap(files.first?["path"] as? String)
        XCTAssertEqual(try EngineSTLExportTests.summary(Data(contentsOf: URL(fileURLWithPath: plate))).volume, 800 - 16 * .pi, accuracy: 1)
        let layout = try await tools.handle(name: "rad_export_stl", arguments: ["artifactID": id, "printLayout": true]).metadata
        XCTAssertEqual(layout["placement"] as? String, "printLayout")
        let plateLayout = try EngineSTLExportTests.summary(Data(contentsOf: URL(fileURLWithPath: try XCTUnwrap(layout["path"] as? String))))
        XCTAssertEqual(plateLayout.bounds[2], 0, accuracy: 0.05)
    }

    @MainActor
    func testAgentBuildsOnlyCandidatesThatPassValidation() async throws {
        guard ProcessInfo.processInfo.environment["RAD_RUN_ENGINE_TESTS"] == "1" else {
            throw XCTSkip("Set RAD_RUN_ENGINE_TESTS=1 to exercise candidate validation.")
        }
        let root = temporary()
        defer { try? FileManager.default.removeItem(at: root) }
        let empty = "result = intersection(box(2), box(2, center=(5, 0, 0)))\n"
        try Data(empty.utf8).write(to: root.appendingPathComponent("empty.py"))
        try Data("result = box((2, 3, 4), center=(1, 1.5, 2))\n".utf8).write(to: root.appendingPathComponent("valid.py"))
        let tools = try CADAgentTools(directory: root, python: EngineService.defaultPython, threads: 2,
                                      current: nil, currentPreview: nil, engine: EngineService())
        do {
            _ = try await tools.handle(name: "rad_build", arguments: ["programPath": "empty.py", "title": "Empty"])
            XCTFail("An empty part is not built for review")
        } catch { XCTAssertTrue(error.localizedDescription.lowercased().contains("empty"), error.localizedDescription) }
        XCTAssertNil(tools.validatedPreview(for: ModelSpec(title: "Empty", summary: "Empty", units: "mm", script: empty)))
        let built = try await tools.handle(name: "rad_build", arguments: ["programPath": "valid.py", "title": "Valid"])
        let valid = try XCTUnwrap(built.metadata["artifactID"] as? String)
        let passed = try await tools.handle(name: "rad_validate", arguments: ["artifactID": valid, "cellSize": 0.05])
        XCTAssertEqual(passed.metadata["valid"] as? Bool, true)
        XCTAssertEqual(passed.metadata["cellSize"] as? Double, 0.05)
        let committed = try await tools.committableCandidate(artifactID: valid)
        XCTAssertNotNil(committed.preview.validatedHash, "A committed candidate carries its validation receipt.")
    }

    @MainActor
    func testCustomFieldsReportTheirProvenSlopeBoundAndFailingRegions() async throws {
        guard ProcessInfo.processInfo.environment["RAD_RUN_ENGINE_TESTS"] == "1" else {
            throw XCTSkip("Set RAD_RUN_ENGINE_TESTS=1 to build and verify custom fields.")
        }
        let root = temporary()
        defer { try? FileManager.default.removeItem(at: root) }
        let gyroid = """
            k = 2 * math.pi / 10.0
            g = sin(k*x)*cos(k*y) + sin(k*y)*cos(k*z) + sin(k*z)*cos(k*x)
            lattice = implicit(abs(g) - 0.35, bounds=((-20, -20, -20), (20, 20, 20)))
            result = Assembly("Lattice")
            result.add(intersection(box(40), lattice), id="lattice", name="Lattice")
            """
        try Data(gyroid.utf8).write(to: root.appendingPathComponent("gyroid.py"))
        try Data("result = implicit(1 / x, ((-1, -1, -1), (1, 1, 1)))\n".utf8).write(to: root.appendingPathComponent("pole.py"))
        let tools = try CADAgentTools(directory: root, python: EngineService.defaultPython, threads: 2,
                                      current: nil, currentPreview: nil, engine: EngineService())
        // A program is verified before it is built.
        let unbuilt = try await tools.handle(name: "rad_verify", arguments: ["programPath": "gyroid.py"]).metadata
        XCTAssertEqual(unbuilt["verified"] as? Bool, true)
        XCTAssertEqual(unbuilt["fieldCount"] as? Int, 1)
        let field = try XCTUnwrap((unbuilt["fields"] as? [[String: Any]])?.first)
        let bound = try XCTUnwrap(field["slopeBound"] as? Double)
        // The gyroid's steepest slope is sqrt(3) * k; the proof may not go below it.
        XCTAssertGreaterThanOrEqual(bound, 3.squareRoot() * 2 * .pi / 10)
        XCTAssertLessThan(bound, 1.6)
        XCTAssertEqual(field["line"] as? Int, 3)
        XCTAssertEqual((field["parts"] as? [[String: Any]])?.first?["partID"] as? String, "lattice")
        XCTAssertTrue((0.3...1).contains(try XCTUnwrap(field["distanceQuality"] as? Double)))
        XCTAssertEqual((field["gradientLength"] as? [String: Double])?.keys.sorted(), ["max", "median", "min"])

        let built = try await tools.handle(name: "rad_build", arguments: ["programPath": "gyroid.py", "title": "Lattice"]).metadata
        let summary = try XCTUnwrap((built["customFields"] as? [[String: Any]])?.first)
        XCTAssertEqual(summary["slopeBound"] as? Double, bound, "The solid divides its field by the bound rad_verify proves.")
        XCTAssertEqual(summary["parts"] as? [String], ["Lattice"])
        XCTAssertEqual(summary["line"] as? Int, 3)
        XCTAssertNotNil(built["customFieldsNote"])
        let artifact = try XCTUnwrap(built["artifactID"] as? String)
        let verified = try await tools.handle(name: "rad_verify", arguments: ["artifactID": artifact]).metadata
        XCTAssertEqual((verified["fields"] as? [[String: Any]])?.first?["slopeBound"] as? Double, bound)

        do {
            _ = try await tools.handle(name: "rad_build", arguments: ["programPath": "pole.py"])
            XCTFail("A field without a finite slope bound does not build.")
        } catch {
            for detail in ["implicit() at line 1", "a division (/)", "divisor may be 0", "rad_verify"] {
                XCTAssertTrue(error.localizedDescription.contains(detail), error.localizedDescription)
            }
        }
        let pole = try await tools.handle(name: "rad_verify", arguments: ["programPath": "pole.py"]).metadata
        XCTAssertEqual(pole["verified"] as? Bool, false)
        let failed = try XCTUnwrap((pole["fields"] as? [[String: Any]])?.first)
        XCTAssertEqual(failed["ok"] as? Bool, false)
        XCTAssertNil(failed["slopeBound"])
        XCTAssertTrue((failed["problem"] as? String)?.contains("the divisor may be 0") == true)
        let region = try XCTUnwrap(failed["region"] as? [Double])
        XCTAssertLessThanOrEqual(region[0], 0)
        XCTAssertGreaterThanOrEqual(region[3], 0, "The failing region holds the plane x = 0.")

        // Fields beyond the 64 verified ones leave the design unverified.
        let beads = (0..<70).map { index -> String in
            let pole = index == 69 ? " + 0.1 / (x - \(index * 3))" : ""
            return "parts.append(implicit(length(x - \(index * 3), y, z) - 1\(pole), ((\(index * 3 - 2), -2, -2), (\(index * 3 + 2), 2, 2))))"
        }
        try Data((["parts = []"] + beads + ["result = union(parts)"]).joined(separator: "\n").utf8)
            .write(to: root.appendingPathComponent("beads.py"))
        let many = try await tools.handle(name: "rad_verify", arguments: ["programPath": "beads.py"]).metadata
        XCTAssertEqual(many["fieldCount"] as? Int, 70)
        XCTAssertEqual((many["fields"] as? [[String: Any]])?.count, EngineService.FieldVerification.maximumFields)
        XCTAssertEqual((many["fields"] as? [[String: Any]])?.allSatisfy { $0["ok"] as? Bool == true }, true)
        XCTAssertEqual(many["verified"] as? Bool, false, "The unverified 70th field has no bound.")
        XCTAssertTrue((many["fieldsTruncated"] as? String)?.contains("first 64 of 70") == true)
    }

    @MainActor
    func testSealedCavitiesAreReportedApartFromPieces() async throws {
        guard ProcessInfo.processInfo.environment["RAD_RUN_ENGINE_TESTS"] == "1" else {
            throw XCTSkip("Set RAD_RUN_ENGINE_TESTS=1 to build designs.")
        }
        let root = temporary()
        defer { try? FileManager.default.removeItem(at: root) }
        try Data("result = box(10) - sphere(3)\n".utf8).write(to: root.appendingPathComponent("hollow.py"))
        let tools = try CADAgentTools(directory: root, python: EngineService.defaultPython, threads: 2,
                                      current: nil, currentPreview: nil, engine: EngineService())
        let built = try await tools.handle(name: "rad_build", arguments: ["programPath": "hollow.py", "title": "Hollow"]).metadata
        let part = try XCTUnwrap((built["parts"] as? [[String: Any]])?.first)
        XCTAssertEqual(part["pieces"] as? Int, 1)
        XCTAssertEqual(part["cavities"] as? Int, 1)
        XCTAssertNotNil(built["cavitiesNote"])
        XCTAssertEqual((built["warnings"] as? [String])?.filter { $0.contains("pieces") } ?? [], [])
        let artifact = try XCTUnwrap(built["artifactID"] as? String)
        let validated = try await tools.handle(name: "rad_validate", arguments: ["artifactID": artifact]).metadata
        XCTAssertEqual(validated["valid"] as? Bool, true)
        let checked = try XCTUnwrap((validated["parts"] as? [[String: Any]])?.first)
        XCTAssertEqual(checked["pieces"] as? Int, 1)
        XCTAssertEqual(checked["cavities"] as? Int, 1)
    }

    private func content(_ result: CADAgentToolResult) throws -> [String: Any] {
        result.metadata
    }
    private func temporary() -> URL {
        let url = FileManager.default.temporaryDirectory.appendingPathComponent("RAD-agent-test-\(UUID().uuidString)")
        try! FileManager.default.createDirectory(at: url, withIntermediateDirectories: true)
        return url
    }
}
