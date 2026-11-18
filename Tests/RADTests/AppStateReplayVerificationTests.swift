import Foundation
import XCTest
@testable import RAD

/// A saved design whose validation receipt no longer matches earns a new
/// receipt from its replay; one that fails validation is reported, not shown.
final class AppStateReplayVerificationTests: XCTestCase {
    @MainActor
    func testStaleReceiptReplayRecordsItsNewReceipt() async throws {
        let (state, directory) = try fixture()
        defer { state.cancel(); try? FileManager.default.removeItem(at: directory) }
        let stale = String(repeating: "0", count: 64)
        let revision = ProjectRevision(model: ModelSpec(title: "Box", summary: "A box.", units: "mm",
                                                        script: "result = box((2, 3, 4))"),
                                       validatedHash: stale)
        state.revisions = [revision]
        state.selectedRevisionID = revision.id
        XCTAssertTrue(state.loadStoredRevision())
        try await wait { state.mesh != nil }
        let receipt = try XCTUnwrap(state.mesh?.validatedHash, "A replay that displays has passed validation.")
        XCTAssertNotEqual(receipt, stale)
        XCTAssertEqual(state.revisions[0].validatedHash, receipt)
        XCTAssertEqual(state.revisions[0].preview?.validatedHash, receipt)
        XCTAssertNil(state.error)
    }

    @MainActor
    func testFailedReplayValidationIsReportedWithoutDisplayingTheDesign() async throws {
        let (state, directory) = try fixture()
        defer { state.cancel(); try? FileManager.default.removeItem(at: directory) }
        // The inner sphere removes all material: the part encloses no volume.
        let revision = ProjectRevision(model: ModelSpec(title: "Empty part", summary: "A part without material.", units: "mm",
                                                        script: "result = sphere(2) - sphere(3)"))
        state.revisions = [revision]
        state.selectedRevisionID = revision.id
        XCTAssertTrue(state.loadStoredRevision())
        try await wait { state.error != nil }
        XCTAssertTrue(state.error?.contains("The CAD engine could not build this design.") == true, state.error ?? "")
        XCTAssertNil(state.mesh, "A design that fails validation is never displayed.")
        XCTAssertNil(state.revisions[0].validatedHash)
    }

    @MainActor
    private func fixture() throws -> (AppState, URL) {
        guard ProcessInfo.processInfo.environment["RAD_RUN_ENGINE_TESTS"] == "1" else {
            throw XCTSkip("Set RAD_RUN_ENGINE_TESTS=1 to verify saved design replays.")
        }
        let directory = FileManager.default.temporaryDirectory.appendingPathComponent("RAD-replay-verification-\(UUID().uuidString)")
        try FileManager.default.createDirectory(at: directory, withIntermediateDirectories: true)
        return (AppState(autosaveStore: ProjectAutosaveStore(rootDirectory: directory.appendingPathComponent("autosaves"))), directory)
    }

    @MainActor
    private func wait(_ condition: () -> Bool) async throws {
        let deadline = ContinuousClock.now + .seconds(90)
        while !condition(), ContinuousClock.now < deadline { try await Task.sleep(for: .milliseconds(20)) }
        XCTAssertTrue(condition())
    }
}
