import XCTest
@testable import RAD

/// Every package section is a typed record encoded once by one sorted-key
/// encoder. Earlier format versions are not opened.
final class ProjectPackageFormatTests: XCTestCase {
    private var folder: URL!

    override func setUpWithError() throws {
        ProjectPackage.clearSessionCache()
        folder = FileManager.default.temporaryDirectory.appendingPathComponent("RAD-format-\(UUID())", isDirectory: true)
        try FileManager.default.createDirectory(at: folder, withIntermediateDirectories: true)
    }

    override func tearDownWithError() throws {
        try? FileManager.default.removeItem(at: folder)
        ProjectPackage.clearSessionCache()
    }

    private struct StoredFile: Equatable {
        var bytes: Data
        var inode: UInt64
        var modified: Date
    }

    private func files(_ url: URL) throws -> [String: StoredFile] {
        let resolved = url.resolvingSymlinksInPath()
        let enumerator = try XCTUnwrap(FileManager.default.enumerator(at: resolved, includingPropertiesForKeys: [.isRegularFileKey]))
        var result = [String: StoredFile]()
        for case let file as URL in enumerator where try file.resourceValues(forKeys: [.isRegularFileKey]).isRegularFile == true {
            let attributes = try FileManager.default.attributesOfItem(atPath: file.path)
            result[String(file.resolvingSymlinksInPath().path.dropFirst(resolved.path.count + 1))] = StoredFile(
                bytes: try Data(contentsOf: file),
                inode: try XCTUnwrap(attributes[.systemFileNumber] as? NSNumber).uint64Value,
                modified: try XCTUnwrap(attributes[.modificationDate] as? Date))
        }
        return result
    }

    private func manifest(_ url: URL) throws -> [String: Any] {
        try XCTUnwrap(JSONSerialization.jsonObject(with: ProjectStore.readArchive(from: url)) as? [String: Any])
    }

    private func assertSameHistory(_ actual: RADProject, _ expected: RADProject, file: StaticString = #filePath, line: UInt = #line) throws {
        let actualArchive = try ProjectArchive.split(actual), expectedArchive = try ProjectArchive.split(expected)
        XCTAssertEqual(String(decoding: actualArchive.metadata, as: UTF8.self), String(decoding: expectedArchive.metadata, as: UTF8.self),
                       file: file, line: line)
        XCTAssertEqual(actualArchive.assets, expectedArchive.assets, file: file, line: line)
    }

    func testEarlierFormatVersionsAreNotOpened() throws {
        let url = folder.appendingPathComponent("Earlier.rad", isDirectory: true)
        try ProjectPackage.write(ProjectPackageFixture.project(), to: url)
        let current = "\"formatVersion\":\(ProjectPackage.formatVersion)"
        var text = String(decoding: try ProjectStore.readArchive(from: url), as: UTF8.self)
        XCTAssertTrue(text.contains(current))
        text = text.replacingOccurrences(of: current, with: "\"formatVersion\":\(ProjectPackage.formatVersion - 1)")
        try Data(text.utf8).write(to: url.appendingPathComponent("project.json"))
        ProjectPackage.clearSessionCache()
        XCTAssertThrowsError(try ProjectStore.projectID(at: url))
        XCTAssertThrowsError(try ProjectStore.read(from: url))
    }

    func testEverySectionRoundTripsThroughTypedRecordsWithStableBytes() throws {
        let project = ProjectPackageFixture.project()
        let first = folder.appendingPathComponent("First.rad", isDirectory: true)
        try ProjectPackage.write(project, to: first, thumbnail: ProjectPackageFixture.png)
        let root = try manifest(first)
        XCTAssertNotNil(root["queuedRequests"])
        XCTAssertNotNil(root["draftAttachments"])
        XCTAssertNotNil(root["thumbnail"])
        let stored = try files(first)
        XCTAssertEqual(stored.keys.filter { $0.contains("/recipe-") }.count, 2)
        XCTAssertEqual(stored.keys.filter { $0.contains("/parts-") }.count, 2)
        XCTAssertEqual(stored.keys.filter { $0.hasPrefix("conversation/messages-") }.count, 1)
        XCTAssertEqual(stored.keys.filter { $0.contains("/task-") }.count, 2)
        XCTAssertEqual(stored.keys.filter { $0.contains("/events-") }.count, 2)
        XCTAssertFalse(stored.keys.filter { $0.hasPrefix("catalog/assets-") }.isEmpty)

        ProjectPackage.clearSessionCache()
        let restored = try ProjectStore.read(from: first)
        try assertSameHistory(restored, project)
        XCTAssertEqual(restored.thumbnailData, ProjectPackageFixture.png)

        // Decoding and encoding again reproduces every file byte for byte.
        let second = folder.appendingPathComponent("Second.rad", isDirectory: true)
        try ProjectPackage.write(restored, to: second, thumbnail: restored.thumbnailData)
        let copy = try files(second)
        XCTAssertEqual(Set(copy.keys), Set(stored.keys))
        for (path, file) in stored where path != "project.json" { XCTAssertEqual(copy[path]?.bytes, file.bytes, path) }
        let commit = try XCTUnwrap(root["commitID"] as? String)
        let copiedCommit = try XCTUnwrap(try manifest(second)["commitID"] as? String)
        XCTAssertEqual(String(decoding: try XCTUnwrap(stored["project.json"]).bytes, as: UTF8.self)
                        .replacingOccurrences(of: commit, with: copiedCommit),
                       String(decoding: try XCTUnwrap(copy["project.json"]).bytes, as: UTF8.self))
    }

    func testSignedZeroAndExtremeNumbersRoundTripExactly() throws {
        var project = ProjectPackageFixture.project()
        let tiny = Double.leastNonzeroMagnitude, huge = Double.greatestFiniteMagnitude
        project.partsPaneFraction = -0.0
        project.messages[3].selection?.parts[0].anchor = ProjectSelectionAnchor(
            center: [-0.0, tiny, -1e-300], area: huge, volume: tiny)
        project.messages[3].selection?.points[0].point = [0.1, -0.0, 1_000_000.1]
        project.revisions[1].jointPositions = ["hinge": -0.0, "slider": -1e9, "rail": -tiny]
        project.revisions[1].partColors = ["base": PartColor(red: -0.0, green: tiny, blue: 0.1)]
        project.taskAudits![0].events[0].durationSeconds = tiny
        project.taskAudits![0].events[1].durationSeconds = huge
        project.queuedRequests![0].selection?.jointPositions = ["hinge": -0.0, "rail": 1e-308]
        let url = folder.appendingPathComponent("Numbers.rad", isDirectory: true)
        try ProjectPackage.write(project, to: url)
        XCTAssertTrue(String(decoding: try ProjectStore.readArchive(from: url), as: UTF8.self).contains("\"partsPaneFraction\":-0"))
        // A warm state-only save, then a cold read.
        project.showGrid = true
        try ProjectPackage.write(project, to: url)
        ProjectPackage.clearSessionCache()
        let restored = try ProjectStore.read(from: url)
        try assertSameHistory(restored, project)
        XCTAssertEqual(restored.partsPaneFraction.sign, .minus)
        let selection = try XCTUnwrap(restored.messages[3].selection)
        let anchor = try XCTUnwrap(selection.parts[0].anchor)
        XCTAssertEqual(anchor.center?.map(\.bitPattern), [(-0.0).bitPattern, tiny.bitPattern, (-1e-300).bitPattern])
        XCTAssertEqual(selection.points[0].point.map(\.bitPattern), [(0.1).bitPattern, (-0.0).bitPattern, (1_000_000.1).bitPattern])
        XCTAssertEqual(selection.points[0].normal.map(\.bitPattern), [(-0.0).bitPattern, (-0.0).bitPattern, (-1.0).bitPattern])
        XCTAssertEqual(anchor.area, huge)
        XCTAssertEqual(anchor.volume, tiny)
        XCTAssertEqual(restored.revisions[1].jointPositions?.mapValues(\.bitPattern),
                       ["hinge": (-0.0).bitPattern, "slider": (-1e9).bitPattern, "rail": (-tiny).bitPattern])
        XCTAssertEqual(restored.revisions[1].partColors?["base"]?.red.sign, .minus)
        XCTAssertEqual(restored.taskAudits?[0].events[0].durationSeconds, tiny)
        XCTAssertEqual(restored.taskAudits?[0].events[1].durationSeconds, huge)
        XCTAssertEqual(restored.queuedRequests?[0].selection?.jointPositions?["hinge"]?.sign, .minus)
        XCTAssertEqual(restored.queuedRequests?[0].selection?.jointPositions?["rail"], 1e-308)
    }
}
