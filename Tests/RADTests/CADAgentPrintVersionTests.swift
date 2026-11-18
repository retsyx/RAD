import AppKit
import Metal
import XCTest
@testable import RAD

final class CADAgentPrintVersionTests: XCTestCase {
    private static let program = """
    plate = box((40, 20, 6)) - cylinder(3, 3, center=(0, 0, 1.5)) - cylinder(1.6, 10)
    if rad.print:
        plate = plate | cylinder(1.6, 0.4, center=(0, 0, -0.2))
    peg = cylinder(3, 15, center=(30, 0, 7.5))
    result = Assembly("Kit", id="kit")
    result.add(plate, id="plate", name="Plate", print_down=(0, 0, 1))
    result.add(peg, id="peg", name="Peg")
    """

    private static let splitWhenPrinting = """
    result = Assembly("Kit", id="kit")
    result.add(box((10, 10, 4), center=(5, 5, 2)), id="base", name="Base")
    if rad.print:
        result.add(box(2, center=(21, 1, 1)), id="brim", name="Brim")
    """

    private static let failsWhenPrinting = """
    if rad.print:
        raise ValueError("bridge layer is too thin")
    result = Assembly("Kit", id="kit")
    result.add(box((10, 10, 4), center=(5, 5, 2)), id="base", name="Base")
    """

    /// A case around a reference circuit board, with a printed button cap
    /// that opts out of its folder's reference flag.
    private static let enclosure = """
    shell = box((40, 30, 12), center=(0, 0, 6)) - box((36, 26, 12), center=(0, 0, 8))
    result = Assembly("Enclosure", id="enclosure")
    result.add(shell, id="case", name="Case", print_down=(0, 0, -1))
    pcb = result.folder("PCB", id="pcb", reference=True)
    pcb.add(box((30, 20, 1.6), center=(0, 0, 4)), id="board", name="Board")
    pcb.add(cylinder(2, 3, center=(10, 0, 6.3)), id="cap", name="Button cap", reference=False)
    """

    @MainActor
    func testReferencePartsAreReportedAndLeftOutOfThePrintingLayout() async throws {
        guard ProcessInfo.processInfo.environment["RAD_RUN_ENGINE_TESTS"] == "1" else {
            throw XCTSkip("Set RAD_RUN_ENGINE_TESTS=1 for 3D printing version agent tools.")
        }
        let root = FileManager.default.temporaryDirectory.appendingPathComponent("RAD-agent-reference-\(UUID().uuidString)")
        try FileManager.default.createDirectory(at: root, withIntermediateDirectories: false)
        defer { try? FileManager.default.removeItem(at: root) }
        try Data(Self.enclosure.utf8).write(to: root.appendingPathComponent("enclosure.py"))
        let tools = try CADAgentTools(directory: root, python: EngineService.defaultPython, threads: 2,
            current: nil, currentPreview: nil, engine: EngineService())
        defer { tools.cancel() }

        let built = try await tools.handle(name: "rad_build", arguments: ["programPath": "enclosure.py"]).metadata
        XCTAssertEqual(built["referenceSolids"] as? [Int], [1])
        let id = try XCTUnwrap(built["artifactID"] as? String)
        let parts = try XCTUnwrap(built["parts"] as? [[String: Any]])
        XCTAssertEqual(parts.map { $0["partID"] as? String }, ["case", "board", "cap"])
        XCTAssertEqual(parts.map { $0["reference"] as? Bool }, [nil, true, nil])

        let print = try await tools.handle(name: "rad_inspect", arguments: ["artifactID": id, "kind": "print"]).metadata
        XCTAssertEqual((print["layout"] as? [[String: Any]])?.compactMap { $0["solidIndex"] as? Int }, [0, 2])
        XCTAssertEqual((print["referenceParts"] as? [[String: Any]])?.compactMap { $0["partID"] as? String }, ["board"])
        XCTAssertEqual(print["total"] as? Int, 2)

        let exported = try await tools.handle(name: "rad_export_3mf", arguments: ["artifactID": id, "printLayout": true]).metadata
        XCTAssertEqual(exported["solidIndices"] as? [Int], [0, 2])
        XCTAssertEqual(exported["referenceSolidsLeftOut"] as? [Int], [1])
        do {
            _ = try await tools.handle(name: "rad_export_3mf", arguments: ["artifactID": id, "printLayout": true,
                                                                           "partIDs": ["board"]])
            XCTFail("Reference parts are never printed")
        } catch { XCTAssertTrue(error.localizedDescription.contains("every requested part is a reference part")) }
        let ordinary = try await tools.handle(name: "rad_export_3mf", arguments: ["artifactID": id, "partIDs": ["board"]]).metadata
        XCTAssertEqual(ordinary["solidIndices"] as? [Int], [1], "Ordinary exports include reference parts")
        XCTAssertNil(ordinary["referenceSolidsLeftOut"])

        if MTLCreateSystemDefaultDevice() != nil {
            let layout = try await tools.handle(name: "rad_render", arguments: ["artifactID": id, "printLayout": true,
                                                                                "width": 320, "height": 256])
            XCTAssertEqual(layout.metadata["referenceSolidsLeftOut"] as? [Int], [1])
            do {
                _ = try await tools.handle(name: "rad_render", arguments: ["artifactID": id, "printLayout": true,
                                                                           "solidIndices": [1], "width": 320, "height": 256])
                XCTFail("The printing layout has no reference parts to show")
            } catch { XCTAssertTrue(error.localizedDescription.contains("every requested part is a reference part")) }
        }
    }

    @MainActor
    func testPrintBuildsLayoutRendersAndCommitChecksUseTheSameProgram() async throws {
        guard ProcessInfo.processInfo.environment["RAD_RUN_ENGINE_TESTS"] == "1" else {
            throw XCTSkip("Set RAD_RUN_ENGINE_TESTS=1 for 3D printing version agent tools.")
        }
        let root = FileManager.default.temporaryDirectory.appendingPathComponent("RAD-agent-print-\(UUID().uuidString)")
        try FileManager.default.createDirectory(at: root, withIntermediateDirectories: false)
        defer { try? FileManager.default.removeItem(at: root) }
        for (name, program) in [("kit.py", Self.program), ("split.py", Self.splitWhenPrinting), ("fails.py", Self.failsWhenPrinting)] {
            try Data(program.utf8).write(to: root.appendingPathComponent(name))
        }
        let tools = try CADAgentTools(directory: root, python: EngineService.defaultPython, threads: 2,
            current: nil, currentPreview: nil, engine: EngineService())
        defer { tools.cancel() }

        let ordinary = try await tools.handle(name: "rad_build", arguments: ["programPath": "kit.py"]).metadata
        XCTAssertEqual(ordinary["definesPrintVersion"] as? Bool, true)
        XCTAssertNil(ordinary["print"])
        let pins = try XCTUnwrap(ordinary["printDown"] as? [[String: Any]])
        XCTAssertEqual(pins.count, 1)
        XCTAssertEqual(pins.first?["direction"] as? [Double], [0, 0, 1])
        let ordinaryID = try XCTUnwrap(ordinary["artifactID"] as? String)

        let printing = try await tools.handle(name: "rad_build", arguments: ["programPath": "kit.py", "print": true]).metadata
        XCTAssertEqual(printing["print"] as? Bool, true)
        let printingID = try XCTUnwrap(printing["artifactID"] as? String)
        XCTAssertNotEqual(printingID, ordinaryID)
        let added = try XCTUnwrap(printing["volume"] as? Double) - XCTUnwrap(ordinary["volume"] as? Double)
        // The bridge fills the bore from z = -0.4 to 0, two preview cells thick.
        XCTAssertEqual(added, Double.pi * 1.6 * 1.6 * 0.4, accuracy: 0.05)
        do {
            _ = try await tools.handle(name: "rad_build", arguments: ["programPath": "kit.py", "print": "yes"])
            XCTFail("The printing option is a boolean")
        } catch { XCTAssertTrue(error.localizedDescription.contains("true or false")) }

        do {
            _ = try await tools.handle(name: "rad_render", arguments: ["artifactID": ordinaryID, "printLayout": true])
            XCTFail("A program with a printing version renders its layout from the printing artifact")
        } catch { XCTAssertTrue(error.localizedDescription.contains("print=true")) }
        do {
            _ = try await tools.handle(name: "rad_render", arguments: ["artifactID": printingID, "printLayout": true,
                                                                       "jointPositions": ["hinge": 1]])
            XCTFail("Printing layouts replace assembly poses")
        } catch { XCTAssertTrue(error.localizedDescription.contains("jointPositions")) }
        if MTLCreateSystemDefaultDevice() != nil {
            let layout = try await tools.handle(name: "rad_render", arguments: ["artifactID": printingID, "printLayout": true,
                                                                                "width": 320, "height": 256])
            XCTAssertEqual(layout.metadata["printLayout"] as? Bool, true)
            let built = try await tools.handle(name: "rad_render", arguments: ["artifactID": printingID,
                                                                               "width": 320, "height": 256])
            XCTAssertNotEqual(layout.image?.data, built.image?.data)
        }

        let model = ModelSpec(title: "Untitled design", summary: "CAD program built and validated by RAD.", units: "mm", script: Self.program)
        XCTAssertNotNil(tools.validatedPrintPreview(for: model))
        do {
            _ = try await tools.committableCandidate(artifactID: printingID)
            XCTFail("Only the ordinary build is committed")
        } catch { XCTAssertTrue(error.localizedDescription.contains("ordinary build")) }
        let candidate = try await tools.committableCandidate(artifactID: ordinaryID)
        XCTAssertEqual(candidate.model.script, Self.program)

        do {
            _ = try await tools.handle(name: "rad_build", arguments: ["programPath": "split.py", "print": true])
            XCTFail("A printing version cannot add parts")
        } catch { XCTAssertTrue(error.localizedDescription.contains("brim")) }

        let fails = try await tools.handle(name: "rad_build", arguments: ["programPath": "fails.py"]).metadata
        do {
            _ = try await tools.committableCandidate(artifactID: try XCTUnwrap(fails["artifactID"] as? String))
            XCTFail("A failing printing version blocks the commit")
        } catch { XCTAssertTrue(error.localizedDescription.contains("too thin")) }
    }
}
