import Foundation
import XCTest
@testable import RAD

final class RecentProjectsTests: XCTestCase {
    @MainActor
    func testRecentOrderDeduplicationLimitAndPersistence() throws {
        let suite = "RAD-recents-\(UUID())"
        let defaults = try XCTUnwrap(UserDefaults(suiteName: suite))
        defer { defaults.removePersistentDomain(forName: suite) }
        let recent = RecentProjects(defaults: defaults, maximumCount: 3)
        let first = file("/tmp/RAD recent tests/Enclosure – מסך.RAD")
        let second = file("/tmp/RAD recent tests/Second.rad")
        let third = file("/tmp/RAD recent tests/Third.rad")
        let fourth = file("/tmp/RAD recent tests/Fourth.rad")
        recent.record(first)
        recent.record(second)
        recent.record(third)
        recent.record(URL(fileURLWithPath: "/tmp/RAD recent tests/sub/../Enclosure – מסך.RAD"))
        XCTAssertEqual(recent.urls, [first, third, second])
        recent.record(fourth)
        XCTAssertEqual(recent.urls, [fourth, first, third])
        let reopened = RecentProjects(defaults: defaults, initialURL: second)
        XCTAssertEqual(reopened.urls, recent.urls)
        XCTAssertEqual(defaults.stringArray(forKey: RecentProjects.defaultsKey), [fourth.path, first.path, third.path])
    }

    @MainActor
    func testClearMenuPersistsWithoutReseedingFromLastSession() throws {
        let suite = "RAD-recents-clear-\(UUID())"
        let defaults = try XCTUnwrap(UserDefaults(suiteName: suite))
        defer { defaults.removePersistentDomain(forName: suite) }
        let lastSession = file("/tmp/Last design.rad")
        let first = RecentProjects(defaults: defaults, initialURL: lastSession)
        XCTAssertEqual(first.urls, [lastSession])
        first.clear()
        XCTAssertTrue(first.urls.isEmpty)
        XCTAssertNotNil(defaults.object(forKey: RecentProjects.defaultsKey))
        let reopened = RecentProjects(defaults: defaults, initialURL: lastSession)
        XCTAssertTrue(reopened.urls.isEmpty)
        reopened.record(lastSession)
        XCTAssertEqual(reopened.urls, [lastSession], "An explicit successful open can repopulate a cleared menu")
    }

    @MainActor
    func testRenamingReplacesTheOldPathWithoutRemovingDifferentProjects() throws {
        let suite = "RAD-recents-rename-\(UUID())"
        let defaults = try XCTUnwrap(UserDefaults(suiteName: suite))
        defer { defaults.removePersistentDomain(forName: suite) }
        let recent = RecentProjects(defaults: defaults)
        let previous = file("/tmp/First folder/Case.rad")
        let other = file("/tmp/Second folder/Case.rad")
        let renamed = file("/tmp/First folder/HDMI enclosure.rad")
        recent.record(previous)
        recent.record(other)
        recent.record(renamed)
        recent.record(renamed, replacing: previous)
        XCTAssertEqual(recent.urls, [renamed, other])
        XCTAssertEqual(RecentProjects(defaults: defaults).urls, [renamed, other])
    }

    @MainActor
    func testOnlyLocalRADReferencesAreRecordedAndFilesNeedNotExist() throws {
        let suite = "RAD-recents-local-\(UUID())"
        let defaults = try XCTUnwrap(UserDefaults(suiteName: suite))
        defer { defaults.removePersistentDomain(forName: suite) }
        let recent = RecentProjects(defaults: defaults)
        let reference = file("/tmp/RAD-does-not-exist-\(UUID())/Design.RAD")
        recent.record(reference)
        let invalid = [try XCTUnwrap(URL(string: "https://example.com/Design.rad")),
                       try XCTUnwrap(URL(string: "file://remote-server/tmp/Design.rad")),
                       file("/tmp/PCB.kicad_pcb"), file("/tmp/Design.json")]
        for url in invalid { recent.record(url, replacing: reference) }
        XCTAssertEqual(recent.urls, [reference], "An invalid new URL must not erase the previous successful design")
    }

    @MainActor
    func testSymlinkAliasesShareOneEntryWithoutReadingProjectContents() throws {
        let suite = "RAD-recents-alias-\(UUID())"
        let defaults = try XCTUnwrap(UserDefaults(suiteName: suite))
        defer { defaults.removePersistentDomain(forName: suite) }
        let root = FileManager.default.temporaryDirectory.appendingPathComponent("RAD-recent-alias-\(UUID())", isDirectory: true)
        try FileManager.default.createDirectory(at: root, withIntermediateDirectories: false)
        defer { try? FileManager.default.removeItem(at: root) }
        let original = root.appendingPathComponent("Original.rad")
        let alias = root.appendingPathComponent("Alias.rad")
        // Deliberately not a valid RAD archive: this model stores references only.
        try Data("not a design archive".utf8).write(to: original)
        try FileManager.default.createSymbolicLink(at: alias, withDestinationURL: original)
        let recent = RecentProjects(defaults: defaults)
        recent.record(original)
        recent.record(alias)
        XCTAssertEqual(recent.urls, [original.standardizedFileURL.resolvingSymlinksInPath()])
    }

    @MainActor
    func testStoredInvalidEntriesAreSanitizedAndAnExplicitEmptyListIsRespected() throws {
        let suite = "RAD-recents-stored-\(UUID())"
        let defaults = try XCTUnwrap(UserDefaults(suiteName: suite))
        defer { defaults.removePersistentDomain(forName: suite) }
        let valid = file("/tmp/Retained.rad")
        defaults.set([valid.path, valid.path, "/tmp/Invalid.txt", "https://example.com/Invalid.rad"],
                     forKey: RecentProjects.defaultsKey)
        let recent = RecentProjects(defaults: defaults, initialURL: file("/tmp/Do not seed.rad"))
        XCTAssertEqual(recent.urls, [valid])
        XCTAssertEqual(defaults.stringArray(forKey: RecentProjects.defaultsKey), [valid.path])
        recent.clear()
        XCTAssertTrue(RecentProjects(defaults: defaults, initialURL: valid).urls.isEmpty)
    }

    private func file(_ path: String) -> URL {
        URL(fileURLWithPath: path).standardizedFileURL.resolvingSymlinksInPath()
    }
}
