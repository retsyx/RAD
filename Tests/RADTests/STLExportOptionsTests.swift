import AppKit
import UniformTypeIdentifiers
import XCTest
@testable import RAD

final class STLExportOptionsTests: XCTestCase {
    @MainActor
    func testOptionsPersistAcrossDialogsAndUseEachProjectFolderName() throws {
        let suite = "RADSTLExportOptionsTests-\(UUID().uuidString)"
        let defaults = try XCTUnwrap(UserDefaults(suiteName: suite))
        defer { defaults.removePersistentDomain(forName: suite) }
        let firstPanel = panel(named: "Tentacle PCB Case.stl")
        let first = STLExportOptionsView(panel: firstPanel, projectName: "Tentacle PCB Case", defaults: defaults)
        XCTAssertFalse(first.for3DPrinting)
        XCTAssertFalse(first.exportIndividualParts)
        try checkbox("Export individual parts", in: first).performClick(nil)
        try checkbox("For 3D printing", in: first).performClick(nil)

        // Reopening a dialog for another design uses the same app preferences,
        // while the destination remains specific to the current design.
        let secondPanel = panel(named: "Étagère selected parts.stl")
        let second = STLExportOptionsView(panel: secondPanel, projectName: "Étagère", defaults: defaults)
        XCTAssertTrue(second.for3DPrinting)
        XCTAssertTrue(second.exportIndividualParts)
        XCTAssertEqual(secondPanel.nameFieldStringValue, "Étagère")
        XCTAssertTrue(secondPanel.allowedContentTypes.isEmpty)
        XCTAssertTrue(secondPanel.allowsOtherFileTypes)
        XCTAssertEqual(secondPanel.nameFieldLabel, "Folder:")
    }

    @MainActor
    func testReturningToSTLOutputRestoresFilenameAndKeepsPrintingPreference() throws {
        let suite = "RADSTLExportOptionsTests-\(UUID().uuidString)"
        let defaults = try XCTUnwrap(UserDefaults(suiteName: suite))
        defer { defaults.removePersistentDomain(forName: suite) }
        let savePanel = panel(named: "My selected parts.stl")
        let types = savePanel.allowedContentTypes
        let label = savePanel.nameFieldLabel
        let view = STLExportOptionsView(panel: savePanel, projectName: "Tentacle PCB Case", defaults: defaults)
        try checkbox("For 3D printing", in: view).performClick(nil)
        let individual = try checkbox("Export individual parts", in: view)
        individual.performClick(nil)
        XCTAssertEqual(savePanel.nameFieldStringValue, "Tentacle PCB Case")
        individual.performClick(nil)
        XCTAssertEqual(savePanel.nameFieldStringValue, "My selected parts.stl")
        XCTAssertEqual(savePanel.allowedContentTypes, types)
        XCTAssertFalse(savePanel.allowsOtherFileTypes)
        XCTAssertEqual(savePanel.nameFieldLabel, label)

        let nextPanel = panel(named: "Another.stl")
        let next = STLExportOptionsView(panel: nextPanel, projectName: "Another", defaults: defaults)
        XCTAssertTrue(next.for3DPrinting)
        XCTAssertFalse(next.exportIndividualParts)
        XCTAssertEqual(nextPanel.nameFieldStringValue, "Another.stl")
        XCTAssertEqual(nextPanel.allowedContentTypes, types)
    }

    @MainActor
    private func panel(named filename: String) -> NSSavePanel {
        let panel = NSSavePanel()
        panel.allowedContentTypes = [UTType(filenameExtension: "stl") ?? .data]
        panel.allowsOtherFileTypes = false
        panel.nameFieldStringValue = filename
        return panel
    }

    @MainActor
    private func checkbox(_ title: String, in view: NSView) throws -> NSButton {
        func find(in view: NSView) -> NSButton? {
            if let button = view as? NSButton, button.title == title { return button }
            return view.subviews.lazy.compactMap { find(in: $0) }.first
        }
        return try XCTUnwrap(find(in: view))
    }
}
