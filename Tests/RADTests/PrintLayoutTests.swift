import Foundation
import XCTest
@testable import RAD

final class PrintLayoutTests: XCTestCase {
    private let identity: [Double] = [1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1]

    private func orientation(_ solid: Int, _ bounds: [Double], matrix: [Double]? = nil) -> PrintOrientation {
        PrintOrientation(solid: solid, matrix: matrix ?? identity, bounds: bounds)
    }

    private func placedBounds(_ item: PrintOrientation, _ matrix: [Double]) -> [Double] {
        // Packing only translates the oriented solid.
        let shift = [matrix[3] - item.matrix[3], matrix[7] - item.matrix[7], matrix[11] - item.matrix[11]]
        return (0..<6).map { item.bounds[$0] + shift[$0 % 3] }
    }

    func testPackingGroundsAndSeparatesEveryFootprintAndKeepsOrientation() throws {
        let quarter: [Double] = [0, -1, 0, 4, 1, 0, 0, -2, 0, 0, 1, 7, 0, 0, 0, 1]
        let items = [orientation(0, [-5, -2, -1, 5, 2, 3]),
                     orientation(1, [10, 10, 4, 40, 25, 9], matrix: quarter),
                     orientation(2, [0, 0, 0, 3, 3, 3]),
                     orientation(3, [-100, -50, -7, -60, -45, -1])]
        let orientations = Dictionary(uniqueKeysWithValues: items.map { ($0.solid, $0) })
        let transforms = PrintLayout.transforms(for: [0, 1, 2, 3], orientations: orientations)
        XCTAssertEqual(Set(transforms.keys), [0, 1, 2, 3])
        let boxes = try items.map { item in placedBounds(item, try XCTUnwrap(transforms[item.solid])) }
        for (item, box) in zip(items, boxes) {
            let matrix = try XCTUnwrap(transforms[item.solid])
            XCTAssertEqual(Array(matrix[0..<3]), Array(item.matrix[0..<3]))
            XCTAssertEqual(Array(matrix[4..<7]), Array(item.matrix[4..<7]))
            XCTAssertEqual(Array(matrix[8..<11]), Array(item.matrix[8..<11]))
            XCTAssertEqual(box[2], 0, accuracy: 1e-12)
            XCTAssertGreaterThanOrEqual(box[0], -1e-12)
            XCTAssertGreaterThanOrEqual(box[1], -1e-12)
        }
        for first in boxes.indices {
            for second in boxes.indices where second > first {
                let a = boxes[first], b = boxes[second]
                XCTAssertGreaterThanOrEqual(max(b[0] - a[3], a[0] - b[3], b[1] - a[4], a[1] - b[4]), PrintLayout.gap - 1e-9)
            }
        }
    }

    func testSubsetPacksFromTheOriginAndOmitsOtherSolids() throws {
        let items = [orientation(0, [0, 0, 0, 30, 20, 5]), orientation(1, [8, 9, 2, 12, 11, 4])]
        let orientations = Dictionary(uniqueKeysWithValues: items.map { ($0.solid, $0) })
        let subset = PrintLayout.transforms(for: [1], orientations: orientations)
        XCTAssertEqual(Array(subset.keys), [1])
        XCTAssertEqual(placedBounds(items[1], try XCTUnwrap(subset[1])), [0, 0, 0, 4, 2, 2])
        let full = PrintLayout.transforms(for: [0, 1], orientations: orientations)
        XCTAssertNotEqual(full[1], subset[1], "A subset export packs only its own parts")
        XCTAssertTrue(PrintLayout.transforms(for: [5], orientations: orientations).isEmpty)
    }

    func testWorkerOrientationsMustMatchTheRequestAndBeRigid() throws {
        let valid = orientation(1, [0, 0, 0, 1, 1, 1])
        XCTAssertEqual(try PrintLayout.validatedOrientations([valid], requested: [1], solidCount: 2), [1: valid])
        for records in [[orientation(0, [0, 0, 0, 1, 1, 1])], [valid, valid],
                        [orientation(1, [0, 0, 0, 1, 1])], [orientation(1, [2, 0, 0, 1, 1, 1])],
                        [orientation(1, [0, 0, 0, 1, 1, 1], matrix: [2, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1])]] {
            XCTAssertThrowsError(try PrintLayout.validatedOrientations(records, requested: [1], solidCount: 2))
        }
    }

    func testPinnedDirectionsMustBeUnitVectorsForDistinctSolids() {
        var preview = mesh(parts: nil, solids: 2)
        preview.printDown = [PrintDownPin(solid: 1, direction: [0, 0, -1])]
        XCTAssertNoThrow(try PrintLayout.validatePins(preview))
        for pins in [[PrintDownPin(solid: 2, direction: [0, 0, 1])], [PrintDownPin(solid: 0, direction: [0, 0, 2])],
                     [PrintDownPin(solid: 0, direction: [0, 1])],
                     [PrintDownPin(solid: 0, direction: [0, 0, 1]), PrintDownPin(solid: 0, direction: [0, 0, -1])]] {
            preview.printDown = pins
            XCTAssertThrowsError(try PrintLayout.validatePins(preview))
        }
    }

    func testPrintingVersionMustKeepTheDesignsPartsAndSolids() throws {
        let parts = [PartNode(id: "kit", name: "Kit", solidIndices: [], children: [
            PartNode(id: "plate", name: "Plate", solidIndices: [0]), PartNode(id: "peg", name: "Peg", solidIndices: [1])])]
        let design = mesh(parts: parts, solids: 2)
        var renamed = parts
        renamed[0].children?[0].name = "Printed plate"
        XCTAssertNoThrow(try PrintLayout.validateParts(of: mesh(parts: renamed, solids: 2), matching: design),
                         "Names and hierarchy are organization, not part identity")
        var swapped = parts
        swapped[0].children?[0].solidIndices = [1]
        swapped[0].children?[1].solidIndices = [0]
        var replaced = parts
        replaced[0].children?[1].id = "pin"
        let split = [PartNode(id: "kit", name: "Kit", solidIndices: [], children: [
            PartNode(id: "plate", name: "Plate", solidIndices: [0, 2]), PartNode(id: "peg", name: "Peg", solidIndices: [1])])]
        for (candidate, solids) in [(swapped, 2), (replaced, 2), (split, 3)] {
            XCTAssertThrowsError(try PrintLayout.validateParts(of: mesh(parts: candidate, solids: solids), matching: design)) { error in
                guard let failure = error as? EngineBuildFailure else { return XCTFail("Expected a design failure") }
                XCTAssertTrue(failure.isRepairable)
                XCTAssertTrue(failure.report?.contains("rad_id") == true)
                XCTAssertFalse(failure.message.contains("rad_"), "The user-facing message must not expose program terms")
            }
        }
        XCTAssertNoThrow(try PrintLayout.validateParts(of: mesh(parts: nil, solids: 3), matching: mesh(parts: nil, solids: 3)))
        XCTAssertThrowsError(try PrintLayout.validateParts(of: mesh(parts: nil, solids: 2), matching: mesh(parts: nil, solids: 3)))
    }

    func testReferencePartsAreValidatedAndNeverPrinted() async throws {
        let json = #"{"buildID": "b", "bounds": [0, 0, 0, 1, 1, 1], "volume": 1, "area": 6, "solids": 3, "buildSeconds": 0, "threads": 1, "referenceSolids": [1, 2]}"#
        var preview = try JSONDecoder().decode(PreviewMesh.self, from: Data(json.utf8))
        XCTAssertEqual(preview.referenceSolids, [1, 2])
        XCTAssertNoThrow(try PrintLayout.validateReferences(preview))
        XCTAssertEqual(PrintLayout.printedSolids([0, 1, 2], of: preview), [0])
        XCTAssertEqual(PrintLayout.printedSolids([2], of: preview), [])
        for references in [[], [2, 1], [1, 1], [3], [-1]] {
            preview.referenceSolids = references
            XCTAssertThrowsError(try PrintLayout.validateReferences(preview), "\(references)")
        }
        preview.referenceSolids = nil
        XCTAssertNoThrow(try PrintLayout.validateReferences(preview))
        XCTAssertEqual(PrintLayout.printedSolids([0, 1, 2], of: preview), [0, 1, 2])

        // Orientation requests for reference parts fail before any worker runs.
        preview.referenceSolids = [2]
        do {
            _ = try await EngineService().printOrientations(preview, solids: [1, 2], python: "/nonexistent/python3", threads: 1)
            XCTFail("Reference parts have no printing orientation")
        } catch { XCTAssertTrue(error.localizedDescription.contains("Reference parts are not printed")) }
    }

    func testPrintingVersionMustKeepTheDesignsReferenceParts() throws {
        let parts = [PartNode(id: "case", name: "Case", solidIndices: [0]), PartNode(id: "pcb", name: "PCB", solidIndices: [1])]
        var design = mesh(parts: parts, solids: 2)
        design.referenceSolids = [1]
        var printing = mesh(parts: parts, solids: 2)
        printing.referenceSolids = [1]
        XCTAssertNoThrow(try PrintLayout.validateParts(of: printing, matching: design))
        for references in [nil, [0], [0, 1]] as [[Int]?] {
            printing.referenceSolids = references
            XCTAssertThrowsError(try PrintLayout.validateParts(of: printing, matching: design)) { error in
                guard let failure = error as? EngineBuildFailure else { return XCTFail("Expected a design failure") }
                XCTAssertTrue(failure.isRepairable)
                XCTAssertTrue(failure.report?.contains("Design reference parts: pcb.") == true, failure.report ?? "")
                XCTAssertFalse(failure.message.contains("rad_"), "The user-facing message must not expose program terms")
            }
        }
    }

    func testPrintViewPreferenceTravelsWithEachDesignAndItsCopies() async throws {
        let root = FileManager.default.temporaryDirectory.appendingPathComponent("RAD-print-view-\(UUID())", isDirectory: true)
        try FileManager.default.createDirectory(at: root, withIntermediateDirectories: false)
        defer { try? FileManager.default.removeItem(at: root) }
        let store = ProjectAutosaveStore(rootDirectory: root.appendingPathComponent("RAD", isDirectory: true))
        let revision = ProjectRevision(model: .bracket)
        var printing = RADProject(messages: [ChatMessage(role: "user", content: "Create a design")], revisions: [revision],
                                  selectedRevisionID: revision.id, projectID: UUID(), printVersionView: true)
        printing.projectName = "Printing design"
        var ordinary = printing
        ordinary.projectID = UUID(); ordinary.projectName = "Ordinary design"; ordinary.printVersionView = nil
        ordinary.printLayoutView = true
        let printingURL = try await store.write(project: printing, projectID: try XCTUnwrap(printing.projectID), title: "P", sequence: 1)
        let ordinaryURL = try await store.write(project: ordinary, projectID: try XCTUnwrap(ordinary.projectID), title: "O", sequence: 1)
        XCTAssertEqual(try ProjectStore.read(from: printingURL).printVersionView, true)
        XCTAssertNil(try ProjectStore.read(from: printingURL).printLayoutView)
        XCTAssertNil(try ProjectStore.read(from: ordinaryURL).printVersionView)
        XCTAssertEqual(try ProjectStore.read(from: ordinaryURL).printLayoutView, true)
        let copy = root.appendingPathComponent("Copy.rad")
        try await store.writeCopy(printing, to: copy)
        XCTAssertEqual(try ProjectStore.read(from: copy).printVersionView, true)
        XCTAssertEqual(try SaveNewProject.snapshot(from: printing).printVersionView, true)
        XCTAssertEqual(try SaveNewProject.snapshot(from: ordinary).printLayoutView, true)
        XCTAssertEqual(try ProjectStore.decode(ProjectStore.encode(printing)).printVersionView, true)
        XCTAssertEqual(try ProjectStore.decode(ProjectStore.encode(ordinary)).printLayoutView, true)
    }

    @MainActor
    func testNewDesignsStartInTheOrdinaryViewAndTogglesStayLocal() {
        let first = AppState(), second = AppState()
        XCTAssertFalse(first.printVersionView)
        XCTAssertFalse(first.printLayoutView)
        first.printVersionView = true
        first.printLayoutView = true
        XCTAssertTrue(first.printVersionView && first.printLayoutView)
        XCTAssertFalse(second.printVersionView || second.printLayoutView)
        XCTAssertNil(first.displayedMesh, "Nothing is shown until a build's printing view is ready")
        XCTAssertFalse(first.isPreparingPrintView, "Without a build there is nothing to prepare")
        XCTAssertNil(first.projectURL)
    }

    private func mesh(parts: [PartNode]?, solids: Int) -> PreviewMesh {
        PreviewMesh(vertices: [], triangles: [], edges: [], bounds: [0, 0, 0, 1, 1, 1], volume: 1, area: 6,
                    solids: solids, buildSeconds: 0, threads: 1, buildID: "/unregistered/build",
                    parts: parts)
    }
}
