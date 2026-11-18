import Foundation
import XCTest
@testable import RAD

/// The 3D printing views and printing exports share one geometry pipeline.
final class AppStatePrintViewTests: XCTestCase {
    private static let program = """
    plate = box((40, 20, 6)) - cylinder(3, 3, center=(0, 0, 1.5)) - cylinder(1.6, 10)
    if rad.print:
        plate = plate | cylinder(1.6, 0.4, center=(0, 0, -0.2))
    peg = cylinder(3, 15, center=(30, 0, 7.5))
    result = Assembly("Kit", id="kit")
    result.add(plate, id="plate", name="Plate", print_down=(0, 0, 1))
    result.add(peg, id="peg", name="Peg")
    """

    @MainActor
    func testViewsShowPrintingVersionAndPackedLayoutAndExportsUseBoth() async throws {
        guard ProcessInfo.processInfo.environment["RAD_RUN_ENGINE_TESTS"] == "1" else {
            throw XCTSkip("Set RAD_RUN_ENGINE_TESTS=1 for the 3D printing views and export pipeline test.")
        }
        let directory = FileManager.default.temporaryDirectory.appendingPathComponent("RAD-print-view-\(UUID().uuidString)")
        try FileManager.default.createDirectory(at: directory, withIntermediateDirectories: true)
        defer { try? FileManager.default.removeItem(at: directory) }
        let model = ModelSpec(title: "Counterbored kit", summary: "Printing version fixture.", units: "mm", script: Self.program)
        let engine = EngineService()
        let preview = try await engine.buildPreview(model, python: EngineService.defaultPython, threads: 2)
        XCTAssertEqual(preview.printVersion, true)
        let parts = try XCTUnwrap(preview.parts)
        let state = AppState(autosaveStore: ProjectAutosaveStore(rootDirectory: directory.appendingPathComponent("autosaves")))
        defer { state.cancel() }
        let revision = ProjectRevision(model: model, preview: preview, partLayout: PartLayout(nodes: parts), parts: parts)
        state.revisions = [revision]
        state.selectedRevisionID = revision.id
        XCTAssertTrue(state.loadStoredRevision())
        state.selectParts([1])

        // The design itself, oriented and packed.
        let cameraKey = state.viewportCameraKey
        var command = state.viewportCommand
        state.printLayoutView = true
        try await waitForPrintView(state)
        XCTAssertNotEqual(state.viewportCommand, command, "Placing the parts frames them")
        XCTAssertEqual(state.displayedMesh?.buildID, preview.buildID)
        XCTAssertEqual(Set(state.displayedSolidTransforms.keys), [0, 1])
        state.printLayoutView = false
        XCTAssertFalse(state.isPreparingPrintView)
        XCTAssertTrue(state.displayedSolidTransforms.isEmpty)

        // The printing version where the design was built, under the same camera.
        command = state.viewportCommand
        state.printVersionView = true
        try await waitForPrintView(state)
        XCTAssertEqual(state.viewportCommand, command, "Showing the printing version keeps the camera")
        XCTAssertEqual(state.viewportCameraKey, cameraKey)
        let printing = try XCTUnwrap(state.displayedMesh)
        XCTAssertNotEqual(printing.buildID, preview.buildID)
        XCTAssertNotEqual(printing.artifactPath, preview.artifactPath)
        XCTAssertGreaterThan(printing.volume, preview.volume, "The printing version bridges the bore.")
        XCTAssertTrue(state.displayedSolidTransforms.isEmpty)
        XCTAssertEqual(state.selectedSolids, [1])
        XCTAssertEqual(state.measurementReport?.solidIndices, [1])

        // Both: the printing version oriented and packed, as exported.
        state.printLayoutView = true
        try await waitForPrintView(state)
        XCTAssertEqual(state.displayedMesh?.buildID, printing.buildID)
        let layout = state.displayedSolidTransforms
        XCTAssertEqual(Set(layout.keys), [0, 1])
        let viewBounds = try (0..<2).map { solid -> [Double] in
            let part = try XCTUnwrap(printing.partStats?.first { $0.solidIndex == solid })
            return ViewportRigidTransform.bounds(part.bounds, matrix: try XCTUnwrap(ViewportRigidTransform.matrix(layout[solid]!)))
        }
        for box in viewBounds { XCTAssertEqual(box[2], 0, accuracy: 0.05) }
        XCTAssertGreaterThan(max(viewBounds[1][0] - viewBounds[0][3], viewBounds[0][0] - viewBounds[1][3],
                                 viewBounds[1][1] - viewBounds[0][4], viewBounds[0][1] - viewBounds[1][4]), 0)
        XCTAssertEqual(layout[0]?[10] ?? 0, -1, accuracy: 1e-9, "The pinned top face of the plate faces the bed")
        state.togglePartVisibility([1])
        XCTAssertEqual(state.displayedSolidTransforms, layout, "Hiding a part never repacks the plate")
        state.togglePartVisibility([1])

        let full = directory.appendingPathComponent("printing.stl")
        state.exportSTL(to: full, for3DPrinting: true)
        try await finish(state, expecting: full)
        let exported = try EngineSTLExportTests.summary(Data(contentsOf: full))
        let union = (0..<3).map { axis in viewBounds.map { $0[axis] }.min()! } + (0..<3).map { axis in viewBounds.map { $0[axis + 3] }.max()! }
        XCTAssertEqual(exported.bounds, union, accuracy: 0.1)
        let subset = directory.appendingPathComponent("peg.stl")
        state.exportSTL(to: subset, solidIndices: [1], for3DPrinting: true)
        try await finish(state, expecting: subset)
        let peg = try EngineSTLExportTests.summary(Data(contentsOf: subset)).bounds
        XCTAssertEqual(peg[2], 0, accuracy: 0.05)
        XCTAssertEqual(peg[5] - peg[2], 15, accuracy: 0.1)

        // Prepared views return at once, without an empty interim display.
        command = state.viewportCommand
        state.printVersionView = false
        XCTAssertEqual(state.viewportCommand, command, "Leaving the printing version keeps the camera")
        XCTAssertFalse(state.isPreparingPrintView)
        XCTAssertEqual(state.displayedMesh?.buildID, preview.buildID)
        XCTAssertEqual(Set(state.displayedSolidTransforms.keys), [0, 1])
        state.printVersionView = true
        XCTAssertFalse(state.isPreparingPrintView)
        XCTAssertEqual(state.displayedSolidTransforms, layout)
        XCTAssertEqual(state.viewportCommand, command)
        state.printLayoutView = false
        XCTAssertNotEqual(state.viewportCommand, command, "Returning the parts frames them")
    }

    /// A case and lid around a circuit board modeled as a reference part.
    private static let enclosure = """
    shell = box((40, 30, 12), center=(0, 0, 6)) - box((36, 26, 12), center=(0, 0, 8))
    lid = box((40, 30, 2), center=(0, 0, 13))
    result = Assembly("Enclosure", id="enclosure")
    result.add(shell, id="case", name="Case", print_down=(0, 0, -1))
    result.add(lid, id="lid", name="Lid")
    pcb = result.folder("PCB", id="pcb", reference=True)
    pcb.add(box((30, 20, 1.6), center=(0, 0, 4)), id="board", name="Board")
    """

    @MainActor
    func testReferencePartsStayInTheDesignButLeaveThePrintingLayoutAndExports() async throws {
        guard ProcessInfo.processInfo.environment["RAD_RUN_ENGINE_TESTS"] == "1" else {
            throw XCTSkip("Set RAD_RUN_ENGINE_TESTS=1 for the 3D printing views and export pipeline test.")
        }
        let directory = FileManager.default.temporaryDirectory.appendingPathComponent("RAD-print-reference-\(UUID().uuidString)")
        try FileManager.default.createDirectory(at: directory, withIntermediateDirectories: true)
        defer { try? FileManager.default.removeItem(at: directory) }
        let model = ModelSpec(title: "Enclosure", summary: "Reference part fixture.", units: "mm", script: Self.enclosure)
        let preview = try await EngineService().buildPreview(model, python: EngineService.defaultPython, threads: 2)
        XCTAssertEqual(preview.referenceSolids, [2])
        XCTAssertNil(preview.printVersion)
        let stats = try XCTUnwrap(preview.partStats)
        let parts = try XCTUnwrap(preview.parts)
        let state = AppState(autosaveStore: ProjectAutosaveStore(rootDirectory: directory.appendingPathComponent("autosaves")))
        defer { state.cancel() }
        let revision = ProjectRevision(model: model, preview: preview, partLayout: PartLayout(nodes: parts), parts: parts)
        state.revisions = [revision]
        state.selectedRevisionID = revision.id
        XCTAssertTrue(state.loadStoredRevision())
        XCTAssertTrue(state.displayedOmittedSolids.isEmpty, "The design shows and measures its reference parts")
        state.selectParts([1, 2])
        XCTAssertEqual(state.measurementReport?.solidIndices, [1, 2])
        state.clearSelections()

        // The plate holds the printed parts only; the printing version is the same geometry.
        state.printLayoutView = true
        try await waitForPrintView(state)
        XCTAssertEqual(Set(state.displayedSolidTransforms.keys), [0, 1])
        XCTAssertEqual(state.displayedOmittedSolids, [2])
        state.printVersionView = true
        try await waitForPrintView(state)
        XCTAssertEqual(Set(state.displayedSolidTransforms.keys), [0, 1])
        XCTAssertEqual(state.displayedOmittedSolids, [2])
        state.printLayoutView = false
        XCTAssertTrue(state.displayedOmittedSolids.isEmpty, "The printing version keeps its reference parts")
        state.printVersionView = false

        let printing = directory.appendingPathComponent("printing.stl")
        state.exportSTL(to: printing, for3DPrinting: true)
        try await finish(state, expecting: printing)
        let printed = try EngineSTLExportTests.summary(Data(contentsOf: printing))
        XCTAssertEqual(printed.volume, stats[0].volume + stats[1].volume, accuracy: (stats[0].volume + stats[1].volume) * 0.01)
        XCTAssertEqual(printed.bounds[2], 0, accuracy: 0.05)
        let whole = directory.appendingPathComponent("design.stl")
        state.exportSTL(to: whole)
        try await finish(state, expecting: whole)
        XCTAssertEqual(try EngineSTLExportTests.summary(Data(contentsOf: whole)).volume, preview.volume,
                       accuracy: preview.volume * 0.01, "Ordinary exports keep reference parts")
        let folder = directory.appendingPathComponent("parts")
        state.exportSTL(to: folder, for3DPrinting: true, exportIndividualParts: true, replaceExisting: true)
        try await finish(state, expecting: folder)
        XCTAssertEqual(try FileManager.default.contentsOfDirectory(atPath: folder.path).sorted(), ["Case.stl", "Lid.stl"])

        let board = directory.appendingPathComponent("board.stl")
        state.exportSTL(to: board, solidIndices: [2], for3DPrinting: true)
        let deadline = ContinuousClock.now + .seconds(60)
        while state.isWorking, ContinuousClock.now < deadline { try await Task.sleep(for: .milliseconds(10)) }
        XCTAssertTrue(state.error?.contains("reference parts, which are not printed") == true, state.error ?? "no error")
        XCTAssertFalse(FileManager.default.fileExists(atPath: board.path))
    }

    @MainActor
    private func waitForPrintView(_ state: AppState) async throws {
        let deadline = ContinuousClock.now + .seconds(60)
        while state.isPreparingPrintView, ContinuousClock.now < deadline { try await Task.sleep(for: .milliseconds(20)) }
        XCTAssertNil(state.error)
        XCTAssertFalse(state.isPreparingPrintView)
    }

    @MainActor
    private func finish(_ state: AppState, expecting url: URL) async throws {
        let deadline = ContinuousClock.now + .seconds(60)
        while state.isWorking, ContinuousClock.now < deadline { try await Task.sleep(for: .milliseconds(10)) }
        XCTAssertFalse(state.isWorking)
        XCTAssertNil(state.error)
        guard FileManager.default.fileExists(atPath: url.path) else {
            throw RADError.message("No STL was exported: \(state.error ?? state.activity)")
        }
    }
}

private func XCTAssertEqual(_ actual: [Double], _ expected: [Double], accuracy: Double,
                            file: StaticString = #filePath, line: UInt = #line) {
    XCTAssertEqual(actual.count, expected.count, file: file, line: line)
    for (a, b) in zip(actual, expected) { XCTAssertEqual(a, b, accuracy: accuracy, file: file, line: line) }
}
