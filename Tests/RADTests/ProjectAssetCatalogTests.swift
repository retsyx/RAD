import XCTest
@testable import RAD

/// The asset catalog lives in its own section, so ordinary saves do not
/// rewrite it, and its size is bounded by the history limits, not a count.
final class ProjectAssetCatalogTests: XCTestCase {
    private var folder: URL!

    override func setUpWithError() throws {
        ProjectPackage.clearSessionCache()
        folder = FileManager.default.temporaryDirectory.appendingPathComponent("RAD-catalog-\(UUID())", isDirectory: true)
        try FileManager.default.createDirectory(at: folder, withIntermediateDirectories: true)
    }

    override func tearDownWithError() throws {
        try? FileManager.default.removeItem(at: folder)
        ProjectPackage.clearSessionCache()
    }

    private func project(events: Int) -> RADProject {
        let message = ChatMessage(role: "user", content: "Create a bracket")
        let revision = ProjectRevision(model: .bracket, parts: [PartNode(id: "body", name: "Body", solidIndices: [0])],
                                       userMessageID: nil)
        var task = ProjectTaskAudit(userMessageID: message.id, model: "test-model", reasoningEffort: "medium", status: "completed")
        task.events = (0..<events).map { index in
            ProjectActivityEvent(id: "event-\(index)", category: "command", title: "Step \(index)", status: "completed",
                                 input: "input \(index)", output: "output \(index)")
        }
        return RADProject(messages: [message], revisions: [revision], selectedRevisionID: revision.id, projectID: UUID(),
                          taskAudits: [task])
    }

    private func save(_ project: RADProject, to url: URL) throws -> ProjectPackage.WriteMetrics {
        var metrics = ProjectPackage.WriteMetrics()
        _ = try ProjectPackage.write(project, to: url, didFinish: { metrics = $0 })
        return metrics
    }

    func testStateSavesLeaveTheCatalogAndAddedAssetsRewriteOnlyTheirBucket() throws {
        let url = folder.appendingPathComponent("Design.rad", isDirectory: true)
        var design = project(events: 200)
        _ = try save(design, to: url)
        let manifest = try XCTUnwrap(JSONSerialization.jsonObject(with: ProjectStore.readArchive(from: url)) as? [String: Any])
        XCTAssertNil(manifest["assets"])
        let reference = try XCTUnwrap(manifest["catalog"] as? [String: Any])
        let catalog = try assetCatalog(url)
        XCTAssertGreaterThan(catalog.count, 300)

        design.draftPrompt = "A state-only edit"
        let state = try save(design, to: url)
        XCTAssertEqual(state.encodedCatalogBuckets, 0)
        let after = try XCTUnwrap(JSONSerialization.jsonObject(with: ProjectStore.readArchive(from: url)) as? [String: Any])
        XCTAssertEqual(after["catalog"] as? [String: String], reference as? [String: String],
                       "A state-only save keeps the same catalog section")

        design.taskAudits![0].events.append(ProjectActivityEvent(id: "event-new", category: "command", title: "New",
                                                                 status: "completed", output: "a new output payload"))
        let appended = try save(design, to: url)
        XCTAssertEqual(appended.encodedCatalogBuckets, 1, "One new asset rewrites only its hash-prefix bucket")
        XCTAssertEqual(try assetCatalog(url).count, catalog.count + 1)
        ProjectPackage.clearSessionCache()
        XCTAssertEqual(try ProjectStore.read(from: url).taskAudits, design.taskAudits)
    }

    func testAssetCountIsBoundedByHistoryLimitsNotASeparateCap() throws {
        // More distinct payloads than the former 32,768-asset limit.
        let design = project(events: 17_000)
        let snapshot = try ProjectArchive.split(design)
        XCTAssertGreaterThan(snapshot.assets.count, 32_768)
        var document = try JSONDecoder().decode(ProjectArchive.Document.self, from: snapshot.metadata)
        document.assets = snapshot.assets
        XCTAssertEqual(try ProjectArchive.restore(document).taskAudits, design.taskAudits)
    }
}
