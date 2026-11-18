import XCTest
@testable import RAD

final class EngineFailureDiagnosticTests: XCTestCase {
    private func withFolder(_ body: (URL) throws -> Void) throws {
        let root = FileManager.default.temporaryDirectory.appendingPathComponent("RAD-failure-test-\(UUID())")
        try FileManager.default.createDirectory(at: root, withIntermediateDirectories: true)
        defer { try? FileManager.default.removeItem(at: root) }
        try body(EngineArtifact.canonicalDirectory(root))
    }

    private func report(_ message: String, version: Int = 1) throws -> Data {
        try JSONSerialization.data(withJSONObject: [
            "formatVersion": version, "message": message, "stage": "preview_meshing", "cellSize": 0.5,
            "failureKind": "design"
        ])
    }

    func testStructuredFailurePreservesMultilineDetails() throws {
        try withFolder { folder in
            let message = """
            Part "Lid" (id lid) has an open surface at the preview cell size of 0.5 mm (12 boundary edges).
            Features thinner than about 1 mm do not close.
            \(String(repeating: "Mesher detail. ", count: 200))
            Make thin walls, pins and gaps thicker.
            Location: [12.5, -3, 7] mm
            """
            try report(message).write(to: folder.appendingPathComponent("error.json"))
            let diagnostic = EngineFailureDiagnostic.message(in: folder, stderr: "Unrelated native stderr noise")
            XCTAssertEqual(diagnostic, "RAD CAD engine: " + message)
            let failure = RADError.message("The CAD engine could not build this design.\n\n" + diagnostic)
            XCTAssertEqual(EngineFailureDiagnostic.repairText(for: failure), message,
                           "Repair must retain every line of the report beyond the former one-line limit.")
        }
    }

    func testMissingMalformedUnsupportedAndOversizeReportsUseBoundedStderr() throws {
        try withFolder { folder in
            let file = folder.appendingPathComponent("error.json")
            let stderr = String(repeating: "previous log noise\n", count: 400) + "RAD CAD engine: Original failure"
            let fallback = String(stderr.suffix(3_000))
            XCTAssertEqual(EngineFailureDiagnostic.message(in: folder, stderr: stderr), fallback)
            for data in [Data("invalid JSON".utf8), try report("unsupported", version: 2),
                         try report(" \n\t"), try report(String(repeating: "é", count: 24_001)),
                         Data(repeating: 32, count: 65_537)] {
                try data.write(to: file)
                XCTAssertEqual(EngineFailureDiagnostic.message(in: folder, stderr: stderr), fallback)
            }
        }
    }

    func testFailureReportsNeverFollowSymbolicOrHardLinks() throws {
        try withFolder { folder in
            let outside = folder.appendingPathComponent("outside.json")
            try report("Unrelated external diagnostic").write(to: outside)
            let file = folder.appendingPathComponent("error.json")
            try FileManager.default.createSymbolicLink(at: file, withDestinationURL: outside)
            XCTAssertEqual(EngineFailureDiagnostic.message(in: folder, stderr: "fallback"), "fallback")
            try FileManager.default.removeItem(at: file)
            try FileManager.default.linkItem(at: outside, to: file)
            XCTAssertEqual(EngineFailureDiagnostic.message(in: folder, stderr: "fallback"), "fallback")
        }
    }

    func testRepairTextRemainsBoundedForVeryLongFailures() {
        let message = "Preview meshing failed\n" + String(repeating: "part diagnostic\n", count: 2_000)
        let failure = RADError.message("The CAD engine could not build this design.\n\nRAD CAD engine: " + message)
        XCTAssertEqual(EngineFailureDiagnostic.repairText(for: failure), String(message.prefix(16_000)))
    }
}
