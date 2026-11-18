import Darwin
import Foundation
import XCTest
@testable import RAD

final class ThreeMFIndividualExportTests: XCTestCase {
    func testPhysicalOwnershipAndSelectionUseCurrentNamesWithoutDuplicatingFolders() {
        let tree = [PartNode(id: "case", name: "Case", solidIndices: [], children: [
            PartNode(id: "base", name: "Renamed base", solidIndices: [0]),
            PartNode(id: "lid", name: "Lid", solidIndices: [2]),
            PartNode(id: "group-with-body", name: "Carrier", solidIndices: [1, 4], children: [
                PartNode(id: "duplicate", name: "Overlapping owner", solidIndices: [4]),
                PartNode(id: "invalid", name: "Unavailable", solidIndices: [99])
            ])
        ])]
        let all = ThreeMFExport.parts(in: tree, solidCount: 6)
        XCTAssertEqual(all.map(\.name), ["Renamed base", "Lid", "Carrier", "Part 4", "Part 6"])
        XCTAssertEqual(all.map(\.solidIndices), [[0], [2], [1, 4], [3], [5]])
        XCTAssertEqual(all.map(\.fileName), ["Renamed base.3mf", "Lid.3mf", "Carrier.3mf", "Part 4.3mf", "Part 6.3mf"])

        let subset = ThreeMFExport.parts(in: tree, selected: [2, 3, 4, 90], solidCount: 6)
        XCTAssertEqual(subset.map(\.name), ["Lid", "Carrier", "Part 4"])
        XCTAssertEqual(subset.map(\.solidIndices), [[2], [4], [3]])
        XCTAssertTrue(ThreeMFExport.parts(in: tree, selected: [], solidCount: 6).isEmpty)
    }

    func testNamesPreserveUnicodeAndPunctuationWhileResolvingFilesystemCollisions() {
        let names = ["Base", "base", "Base (2)", "Báse", "Ba\u{301}se", "Bräckét Ø20 + grip"]
        let nodes = names.enumerated().map { PartNode(id: "\($0.offset)", name: $0.element, solidIndices: [$0.offset]) }
        let parts = ThreeMFExport.parts(in: nodes, solidCount: names.count)
        XCTAssertEqual(parts.map(\.name), names)
        XCTAssertEqual(parts.map(\.fileName), ["Base.3mf", "base (3).3mf", "Base (2).3mf", "Báse.3mf", "Báse (2).3mf", "Bräckét Ø20 + grip.3mf"])
        XCTAssertEqual(ThreeMFExport.folderName(for: " Tentacle PCB Case.RAD "), "Tentacle PCB Case")
        XCTAssertEqual(ThreeMFExport.folderName(for: "Design.v2"), "Design.v2")
        XCTAssertEqual(ThreeMFExport.folderName(for: ".rad"), "Design")
    }

    func testUnsafeAndLongNamesStayInsideTheExportFolder() throws {
        let root = try temporaryDirectory()
        defer { try? FileManager.default.removeItem(at: root) }
        let names = ["../../Base/left:right\n", ".", "..", String(repeating: "🧩", count: 150)]
        let parts = ThreeMFExport.parts(in: names.enumerated().map {
            PartNode(id: "\($0.offset)", name: $0.element, solidIndices: [$0.offset])
        }, solidCount: names.count)
        let folder = root.appendingPathComponent(ThreeMFExport.folderName(for: "Case.rad"))
        let bytes = parts.indices.map { Data("part-\($0)".utf8) }
        try ThreeMFExport.write(bytes, parts: parts, to: folder)
        XCTAssertEqual(Set(try FileManager.default.contentsOfDirectory(atPath: folder.path)), Set(parts.map(\.fileName)))
        for (part, expected) in zip(parts, bytes) {
            XCTAssertLessThanOrEqual(part.fileName.utf8.count, 255)
            XCTAssertFalse(part.fileName.contains("/"))
            XCTAssertFalse(part.fileName.contains(":"))
            XCTAssertFalse(part.fileName.hasPrefix("."))
            XCTAssertEqual(try Data(contentsOf: folder.appendingPathComponent(part.fileName)), expected)
        }
        XCTAssertEqual(try FileManager.default.contentsOfDirectory(atPath: root.path), ["Case"])
    }

    /// File providers such as iCloud Desktop hide dot-prefixed items and their
    /// contents, so staging must never use a hidden sibling of the destination.
    func testPublishedExportIsAnOrdinaryVisibleFolder() throws {
        let root = try temporaryDirectory()
        defer { try? FileManager.default.removeItem(at: root) }
        let manager = FileManager.default
        let reference = root.appendingPathComponent("Reference")
        try manager.createDirectory(at: reference, withIntermediateDirectories: false)
        let destination = root.appendingPathComponent("Case")
        let parts = [ThreeMFExport.Part(name: "Base", solidIndices: [0]),
                     ThreeMFExport.Part(name: "Lid", solidIndices: [1])]
        try ThreeMFExport.write([Data("base".utf8), Data("lid".utf8)], parts: parts, to: destination)
        XCTAssertEqual(Set(try manager.contentsOfDirectory(atPath: root.path)), ["Case", "Reference"])
        XCTAssertEqual(try mode(of: destination), try mode(of: reference))
        for url in [destination] + parts.map({ destination.appendingPathComponent($0.fileName) }) {
            var information = stat()
            XCTAssertEqual(lstat(url.path, &information), 0)
            XCTAssertEqual(information.st_flags & UInt32(UF_HIDDEN), 0, url.lastPathComponent)
            XCTAssertFalse(try url.resourceValues(forKeys: [.isHiddenKey]).isHidden ?? true, url.lastPathComponent)
        }
    }

    func testExistingFolderFileAndSymlinkRemainUntouched() throws {
        let root = try temporaryDirectory()
        defer { try? FileManager.default.removeItem(at: root) }
        let parts = [ThreeMFExport.Part(name: "Base", solidIndices: [0])]
        let manager = FileManager.default
        let existingFolder = root.appendingPathComponent("Case")
        try manager.createDirectory(at: existingFolder, withIntermediateDirectories: false)
        let sentinel = existingFolder.appendingPathComponent("keep.txt")
        let original = Data("existing output".utf8)
        try original.write(to: sentinel)
        XCTAssertThrowsError(try ThreeMFExport.write([Data("new output".utf8)], parts: parts, to: existingFolder))
        XCTAssertEqual(try manager.contentsOfDirectory(atPath: existingFolder.path), ["keep.txt"])
        XCTAssertEqual(try Data(contentsOf: sentinel), original)

        let existingFile = root.appendingPathComponent("File")
        try original.write(to: existingFile)
        XCTAssertThrowsError(try ThreeMFExport.write([Data()], parts: parts, to: existingFile))
        XCTAssertEqual(try Data(contentsOf: existingFile), original)

        let link = root.appendingPathComponent("Link")
        let target = root.appendingPathComponent("missing-target")
        try manager.createSymbolicLink(at: link, withDestinationURL: target)
        XCTAssertThrowsError(try ThreeMFExport.write([Data()], parts: parts, to: link))
        XCTAssertEqual(try manager.destinationOfSymbolicLink(atPath: link.path), target.path)
        XCTAssertFalse(manager.fileExists(atPath: target.path))
        XCTAssertEqual(Set(try manager.contentsOfDirectory(atPath: root.path)), ["Case", "File", "Link"])
    }

    func testApprovedReplacementPublishesCompleteFolderAndRemovesOldContents() throws {
        let root = try temporaryDirectory()
        defer { try? FileManager.default.removeItem(at: root) }
        let manager = FileManager.default
        let destination = root.appendingPathComponent("MagSafe Charger Stand")
        try manager.createDirectory(at: destination.appendingPathComponent("Old folder"), withIntermediateDirectories: true)
        try Data("old base".utf8).write(to: destination.appendingPathComponent("Base.3mf"))
        try Data("obsolete part".utf8).write(to: destination.appendingPathComponent("Obsolete.3mf"))
        try Data("old nested output".utf8).write(to: destination.appendingPathComponent("Old folder/old.3mf"))
        let parts = [ThreeMFExport.Part(name: "Base", solidIndices: [0]),
                     ThreeMFExport.Part(name: "Lid", solidIndices: [1])]
        let bytes = [Data("new base".utf8), Data("new lid".utf8)]
        try ThreeMFExport.write(bytes, parts: parts, to: destination, replaceExisting: true)
        XCTAssertEqual(Set(try manager.contentsOfDirectory(atPath: destination.path)), ["Base.3mf", "Lid.3mf"])
        for (part, expected) in zip(parts, bytes) {
            XCTAssertEqual(try Data(contentsOf: destination.appendingPathComponent(part.fileName)), expected)
        }
        XCTAssertEqual(try manager.contentsOfDirectory(atPath: root.path), ["MagSafe Charger Stand"])
    }

    func testApprovedReplacementHandlesFilesAndSymlinksWithoutChangingLinkTargets() throws {
        let root = try temporaryDirectory()
        defer { try? FileManager.default.removeItem(at: root) }
        let manager = FileManager.default
        let part = ThreeMFExport.Part(name: "Base", solidIndices: [0])
        let bytes = Data("new base".utf8)
        let file = root.appendingPathComponent("Existing file")
        try Data("old file".utf8).write(to: file)
        try ThreeMFExport.write([bytes], parts: [part], to: file, replaceExisting: true)
        XCTAssertEqual(try Data(contentsOf: file.appendingPathComponent("Base.3mf")), bytes)

        let target = root.appendingPathComponent("Original target")
        try manager.createDirectory(at: target, withIntermediateDirectories: false)
        let original = Data("original target remains".utf8)
        try original.write(to: target.appendingPathComponent("keep.txt"))
        for (name, linkTarget) in [("Existing link", target), ("Dangling link", root.appendingPathComponent("Missing target"))] {
            let link = root.appendingPathComponent(name)
            try manager.createSymbolicLink(at: link, withDestinationURL: linkTarget)
            try ThreeMFExport.write([bytes], parts: [part], to: link, replaceExisting: true)
            XCTAssertFalse(try link.resourceValues(forKeys: [.isSymbolicLinkKey]).isSymbolicLink ?? true)
            XCTAssertEqual(try Data(contentsOf: link.appendingPathComponent("Base.3mf")), bytes)
        }
        XCTAssertEqual(try manager.contentsOfDirectory(atPath: target.path), ["keep.txt"])
        XCTAssertEqual(try Data(contentsOf: target.appendingPathComponent("keep.txt")), original)
        XCTAssertFalse(manager.fileExists(atPath: root.appendingPathComponent("Missing target").path))
        XCTAssertEqual(Set(try manager.contentsOfDirectory(atPath: root.path)),
                       ["Existing file", "Original target", "Existing link", "Dangling link"])
    }

    func testApprovedReplacementPreservesOldOutputWhenPublishingFails() throws {
        guard getuid() != 0 else { throw XCTSkip("Filesystem permission failures need an unprivileged test process.") }
        let root = try temporaryDirectory()
        let manager = FileManager.default
        defer {
            try? manager.setAttributes([.posixPermissions: 0o700], ofItemAtPath: root.path)
            try? manager.removeItem(at: root)
        }
        let destination = root.appendingPathComponent("Case")
        try manager.createDirectory(at: destination, withIntermediateDirectories: false)
        let original = Data("existing output".utf8)
        try original.write(to: destination.appendingPathComponent("Old.3mf"))
        try manager.setAttributes([.posixPermissions: 0o500], ofItemAtPath: root.path)
        XCTAssertThrowsError(try ThreeMFExport.write([Data("replacement".utf8)],
            parts: [.init(name: "Base", solidIndices: [0])], to: destination, replaceExisting: true))
        XCTAssertEqual(try manager.contentsOfDirectory(atPath: destination.path), ["Old.3mf"])
        XCTAssertEqual(try Data(contentsOf: destination.appendingPathComponent("Old.3mf")), original)
        XCTAssertEqual(try manager.contentsOfDirectory(atPath: root.path), ["Case"])
    }

    func testCancelledApprovedReplacementPreservesExistingOutput() async throws {
        let root = try temporaryDirectory()
        defer { try? FileManager.default.removeItem(at: root) }
        let destination = root.appendingPathComponent("Case")
        try FileManager.default.createDirectory(at: destination, withIntermediateDirectories: false)
        let original = Data("existing output".utf8)
        try original.write(to: destination.appendingPathComponent("Old.3mf"))
        let task = Task.detached {
            withUnsafeCurrentTask { $0?.cancel() }
            try ThreeMFExport.write([Data("new base".utf8)],
                parts: [.init(name: "Base", solidIndices: [0])], to: destination, replaceExisting: true)
        }
        do {
            try await task.value
            XCTFail("A cancelled replacement must retain the old export.")
        } catch is CancellationError {}
        XCTAssertEqual(try FileManager.default.contentsOfDirectory(atPath: destination.path), ["Old.3mf"])
        XCTAssertEqual(try Data(contentsOf: destination.appendingPathComponent("Old.3mf")), original)
        XCTAssertEqual(try FileManager.default.contentsOfDirectory(atPath: root.path), ["Case"])
    }

    func testInvalidFileCountsNeverPublishPartialOutput() throws {
        let root = try temporaryDirectory()
        defer { try? FileManager.default.removeItem(at: root) }
        let destination = root.appendingPathComponent("Case")
        let parts = [ThreeMFExport.Part(name: "Base", solidIndices: [0]),
                     ThreeMFExport.Part(name: "Lid", solidIndices: [1])]
        XCTAssertThrowsError(try ThreeMFExport.write([Data("base".utf8)], parts: parts, to: destination))
        XCTAssertThrowsError(try ThreeMFExport.write([], parts: [], to: destination))
        XCTAssertEqual(try FileManager.default.contentsOfDirectory(atPath: root.path), [])
    }

    func testCancellationDoesNotCreateAnExportFolder() async throws {
        let root = try temporaryDirectory()
        defer { try? FileManager.default.removeItem(at: root) }
        let destination = root.appendingPathComponent("Case")
        let task = Task.detached {
            withUnsafeCurrentTask { $0?.cancel() }
            try ThreeMFExport.write([Data("base".utf8)],
                                           parts: [.init(name: "Base", solidIndices: [0])], to: destination)
        }
        do {
            try await task.value
            XCTFail("A cancelled export must not publish a folder.")
        } catch is CancellationError {}
        XCTAssertEqual(try FileManager.default.contentsOfDirectory(atPath: root.path), [])
    }

    private func mode(of url: URL) throws -> Int {
        try XCTUnwrap(FileManager.default.attributesOfItem(atPath: url.path)[.posixPermissions] as? Int)
    }

    private func temporaryDirectory() throws -> URL {
        let directory = FileManager.default.temporaryDirectory.appendingPathComponent("RAD-individual-export-\(UUID().uuidString)")
        try FileManager.default.createDirectory(at: directory, withIntermediateDirectories: false)
        return directory
    }
}
