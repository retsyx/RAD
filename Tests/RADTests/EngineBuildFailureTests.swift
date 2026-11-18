import XCTest
@testable import RAD

final class EngineBuildFailureTests: XCTestCase {
    func testRepairFollowsTheFailureKindNotItsWording() {
        let reworded = EngineBuildFailure(kind: .design, message: "Your design did not survive validation.", report: "Fillet failed")
        XCTAssertTrue(reworded.isRepairable, "A reported design failure stays repairable whatever its message says.")
        XCTAssertEqual(EngineFailureDiagnostic.repairText(for: reworded), "Fillet failed")
        let formerlyRepairable = "The CAD engine could not build this design.\n\nRAD CAD engine: Fillet failed"
        for kind in [EngineBuildFailure.Kind.runtime, .sandbox, .timeout, .engine] {
            XCTAssertFalse(EngineBuildFailure(kind: kind, message: formerlyRepairable, report: "Fillet failed").isRepairable,
                           "\(kind) failures are never repaired, even with a design failure's wording.")
        }
    }

    func testWorkerDiagnosticsClassifyFailuresAndKeepTheirMessages() {
        func failure(_ text: String, reported: Bool = true, kind: String? = "design", sandboxed: Bool = true) -> EngineBuildFailure {
            let report = reported ? text.components(separatedBy: "RAD CAD engine:").last?.trimmingCharacters(in: .whitespacesAndNewlines) : nil
            return EngineBuildFailure(action: "build this design", sandboxed: sandboxed,
                                      diagnostic: .init(text: text, report: report, kind: reported ? kind : nil))
        }
        let design = failure("RAD CAD engine: Part 1 encloses no volume at the preview cell size of 0.1 mm.")
        XCTAssertEqual(design.kind, .design)
        XCTAssertEqual(design.localizedDescription,
                       "The CAD engine could not build this design.\n\nRAD CAD engine: Part 1 encloses no volume at the preview cell size of 0.1 mm.")
        XCTAssertEqual(design.report, "Part 1 encloses no volume at the preview cell size of 0.1 mm.")

        let missing = failure("RAD CAD engine: No module named 'numpy'")
        XCTAssertEqual(missing.kind, .runtime)
        XCTAssertEqual(missing.localizedDescription,
                       "RAD’s CAD runtime is incomplete. Reinstall RAD and try again.\n\nRAD CAD engine: No module named 'numpy'")
        let sandbox = failure("sandbox-exec: sandbox_apply: Operation not permitted", reported: false)
        XCTAssertEqual(sandbox.kind, .sandbox)
        XCTAssertTrue(sandbox.localizedDescription.hasPrefix("The macOS sandbox could not start this design."))
        XCTAssertEqual(failure("sandbox-exec: sandbox_apply: Operation not permitted", reported: false, sandboxed: false).kind, .engine)
        // The worker's classification decides, not the wording.
        for environment in ["RAD CAD engine: [Errno 1] Operation not permitted: '/Users'",
                            "RAD CAD engine: Fillet failed"] {
            XCTAssertEqual(failure(environment, kind: "runtime").kind, .runtime, environment)
            XCTAssertTrue(failure(environment, kind: "runtime").localizedDescription.hasPrefix("The CAD engine could not build this design."))
        }
        XCTAssertEqual(failure("RAD CAD engine: Python design at line 3: PermissionError: denied").kind, .design)
        XCTAssertEqual(failure("RAD CAD engine: Fillet failed", kind: nil).kind, .engine,
                       "A report without the worker's classification is not assumed to be the design's.")
        let crashed = failure("Segmentation fault: 11", reported: false)
        XCTAssertEqual(crashed.kind, .engine)
        XCTAssertEqual(crashed.localizedDescription, "The CAD engine could not build this design.\n\nSegmentation fault: 11")
        XCTAssertNil(crashed.report)
    }

    func testReportsComeFromTheReportFileOrTheWorkersLastStderrLine() throws {
        let root = FileManager.default.temporaryDirectory.appendingPathComponent("RAD-build-failure-\(UUID().uuidString)")
        try FileManager.default.createDirectory(at: root, withIntermediateDirectories: false)
        defer { try? FileManager.default.removeItem(at: root) }
        let folder = try EngineArtifact.canonicalDirectory(root)
        let noise = "warning: unrelated\nRAD CAD engine: earlier attempt\n"
        let fromStderr = EngineFailureDiagnostic.read(in: folder, stderr: noise + "RAD CAD engine:  Box is open \n")
        XCTAssertEqual(fromStderr.report, "Box is open")
        XCTAssertNil(EngineFailureDiagnostic.read(in: folder, stderr: "Killed: 9").report)
        XCTAssertNil(fromStderr.kind)
        try JSONSerialization.data(withJSONObject: ["formatVersion": 1, "message": "Fillet failed\n", "stage": "worker",
                                                    "failureKind": "design"])
            .write(to: folder.appendingPathComponent("error.json"))
        let fromFile = EngineFailureDiagnostic.read(in: folder, stderr: noise)
        XCTAssertEqual(fromFile.text, "RAD CAD engine: Fillet failed\n")
        XCTAssertEqual(fromFile.report, "Fillet failed")
        XCTAssertEqual(fromFile.kind, "design")
        XCTAssertEqual(fromFile.stage, "worker")
        XCTAssertNil(fromStderr.stage)
        // The failure keeps the stage, so callers act on where the build failed.
        let failure = EngineBuildFailure(action: "build this design", sandboxed: false, diagnostic: fromFile)
        XCTAssertEqual(failure.stage, "worker")
        XCTAssertNil(EngineBuildFailure(kind: .design, message: "Box is open").stage)
    }

    @MainActor
    func testAProgramErrorIsAReportedRepairableDesignFailure() async throws {
        guard ProcessInfo.processInfo.environment["RAD_RUN_ENGINE_TESTS"] == "1" else {
            throw XCTSkip("Set RAD_RUN_ENGINE_TESTS=1 to run native CAD engine tests.")
        }
        let spec = ModelSpec(title: "Broken", summary: "A program that raises.", units: "mm",
                             script: "raise ValueError('deliberate design failure')\n")
        do {
            _ = try await EngineService().buildPreview(spec, python: EngineService.defaultPython, threads: 1)
            XCTFail("The failing program must not build.")
        } catch let failure as EngineBuildFailure {
            XCTAssertEqual(failure.kind, .design)
            XCTAssertTrue(failure.localizedDescription.hasPrefix("The CAD engine could not build this design.\n\nRAD CAD engine:"))
            XCTAssertTrue(EngineFailureDiagnostic.repairText(for: failure).contains("deliberate design failure"))
        }
    }
}
