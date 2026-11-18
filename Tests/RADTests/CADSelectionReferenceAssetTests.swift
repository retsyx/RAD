import Foundation
import XCTest
@testable import RAD

final class CADSelectionReferenceAssetTests: XCTestCase {
    @MainActor
    func testOldSelectionInputsAreAvailableWithoutReplacingCurrentInputsAndConflictsAreAtomic() throws {
        let root = FileManager.default.temporaryDirectory.appendingPathComponent("RAD-selection-inputs-\(UUID())")
        try FileManager.default.createDirectory(at: root, withIntermediateDirectories: true)
        defer { try? FileManager.default.removeItem(at: root) }
        let current = CADInputAsset(id: "current", name: "Current reference", format: "stl", data: Data("current reference bytes".utf8))
        let previous = CADInputAsset(id: "previous", name: "Previous reference", format: "stl", data: Data("previous reference bytes".utf8))
        let spec = ModelSpec(title: "Current", summary: "Current", units: "mm", script: "result=imported_assets['current']", inputAssets: [current])
        let tools = try CADAgentTools(directory: root, python: "/missing/python", threads: 1, current: spec,
            currentPreview: nil, engine: EngineService(), referenceAssets: [previous])
        XCTAssertEqual(try tools.resolveAssets(ids: ["current", "previous"]), [current, previous])
        try tools.registerReferenceAssets([previous])
        XCTAssertEqual(try tools.resolveAssets(ids: ["previous"]), [previous])
        let unused = CADInputAsset(id: "unused", name: "Uncommitted", format: "stl", data: Data("uncommitted bytes".utf8))
        var conflict = current
        conflict.data = Data("different bytes".utf8)
        XCTAssertThrowsError(try tools.registerReferenceAssets([unused, conflict]))
        XCTAssertThrowsError(try tools.resolveAssets(ids: ["unused"]), "A rejected snapshot must not partly register its references.")
        XCTAssertEqual(try tools.resolveAssets(ids: ["current", "previous"]), [current, previous])
        XCTAssertFalse(FileManager.default.fileExists(atPath: root.appendingPathComponent("renders").path))
    }
}
