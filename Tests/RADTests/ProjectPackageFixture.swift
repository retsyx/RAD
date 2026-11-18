import CryptoKit
import Foundation
@testable import RAD

/// A design that fills every package section, built from fixed values so
/// written packages can be compared exactly.
enum ProjectPackageFixture {
    /// A 4 × 3 sRGB PNG, used for an image attachment, a saved render and the
    /// Finder thumbnail.
    static let png = Data(base64Encoded: "iVBORw0KGgoAAAANSUhEUgAAAAQAAAADCAYAAAC09K7GAAAAAXNSR0IArs4c6QAAAERlWElmTU0AKgAAAAgAAYdpAAQAAAABAAAAGgAAAAAAA6ABAAMAAAABAAEAAKACAAQAAAABAAAABKADAAQAAAABAAAAAwAAAAB3TCISAAAAG0lEQVQIHWNkYDjx34aBgQGGWRgCgDwkgCEAAIG7AyUaBIWiAAAAAElFTkSuQmCC")!

    /// A fixed bracket program. Storage tests never build it.
    static let bracket = ModelSpec(
        title: ModelSpec.bracket.title, summary: ModelSpec.bracket.summary, units: "mm",
        script: """
        base = box((80, 50, 6), center=(0, 0, 3))
        wall = box((80, 6, 40), center=(0, 22, 20))
        holes = [cylinder(3, 10, center=(x, y, 3)) for x, y in [(-28, -14), (28, -14), (-28, 8), (28, 8)]]
        opening = cylinder(8, 12, center=(0, 22, 25), axis=(0, 1, 0))
        result = difference(base | wall, *holes, opening)
        """)

    /// A small binary STL reference input: 40 facets.
    static let stl: Data = {
        var data = Data("RAD fixture STL".utf8) + Data(count: 65)
        withUnsafeBytes(of: UInt32(40).littleEndian) { data.append(contentsOf: $0) }
        for index in 0..<40 {
            let z = Float(index) * 0.1
            for value in [0, 0, 1, 0, 0, z, 1, 0, z, 0, 1, z] as [Float] {
                withUnsafeBytes(of: value.bitPattern.littleEndian) { data.append(contentsOf: $0) }
            }
            data.append(contentsOf: [0, 0])
        }
        return data
    }()

    static func id(_ value: Int) -> UUID {
        UUID(uuidString: String(format: "00000000-0000-4000-8000-%012d", value))!
    }

    /// Seconds whose shortest and 17-digit decimal forms differ.
    static func date(_ offset: Double) -> Date {
        Date(timeIntervalSinceReferenceDate: 800_000_000.123456789 + offset)
    }

    static func digest(_ data: Data) -> String {
        SHA256.hexDigest(data)
    }

    static func project() -> RADProject {
        let sourceText = "# Bracket dimensions\nwidth = 80.1\nthickness = 0.30000000000000004\n"
        let source = SourceAttachment(id: id(1), name: "dimensions.py", path: "/missing/dimensions.py",
                                      content: sourceText, sha256: digest(Data(sourceText.utf8)))
        let image = SourceAttachment(id: id(2), name: "sketch.png", path: "/missing/sketch.png", content: "",
                                     sha256: digest(png), imageData: png)
        let first = id(10), second = id(11)
        let selection = ProjectSelectionContext(revisionID: first, capturedAt: date(100.1), parts: [
            ProjectSelectedPart(partID: "base", name: "Base", path: ["Bracket", "Base"], solidIndex: 0,
                                anchor: ProjectSelectionAnchor(center: [0.1, -0.0, 2.5e-7], volume: 1234.5678,
                                                               bounds: [-40, -25, 0, 40, 25, 6.000000000000001]))
        ], points: [
            ProjectSelectedPoint(partID: "base", name: "Base", path: ["Bracket", "Base"], solidIndex: 0,
                                 point: [12.5, -0.0, 0.30000000000000004], normal: [-0.0, -0.0, -1]),
            ProjectSelectedPoint(partID: "upright", name: "Upright", path: ["Bracket", "Upright"], solidIndex: 1,
                                 point: [1e-17, 22.000000000000004, 40], normal: [0, 1, -0.0])
        ], jointPositions: ["hinge": -0.0])
        let messages = [
            ChatMessage(id: id(20), role: "user", content: "Create a bracket with \"quoted\" text, -0 and 1e-0 / slashes",
                        date: date(0), attachments: [source]),
            ChatMessage(id: id(21), role: "assistant", content: "Checking the runtime.", date: date(5.5), phase: "commentary"),
            ChatMessage(id: id(22), role: "assistant", content: "Bracket ready.", date: date(60.25), phase: "final_answer"),
            ChatMessage(id: id(23), role: "user", content: "Make the base thicker ✓", date: date(120.7),
                        attachments: [image], selection: selection),
            ChatMessage(id: id(24), role: "assistant", content: "Thicker base ready.", date: date(180.9), phase: "final_answer")
        ]
        let parts = [PartNode(id: "bracket", name: "Bracket", solidIndices: [], children: [
            PartNode(id: "base", name: "Base", solidIndices: [0], color: PartColor(red: 0.1, green: 0.2, blue: 0.3)),
            PartNode(id: "upright", name: "Upright", solidIndices: [1])
        ])]
        let model = bracket
        var revisionA = ProjectRevision(id: first, model: model, date: date(61),
                                        partLayout: PartLayout(nodes: parts, movedNodeIDs: ["upright"], userFolderIDs: [],
                                                               renamedNodes: ["upright": "Wall"], collapsedNodeIDs: ["bracket"]),
                                        hiddenSolids: [1], parts: parts)
        revisionA.runtime = CADRuntime(python: "3.12.7", numpy: "2.5.2", kernel: "7.8.1")
        revisionA.userMessageID = messages[0].id
        revisionA.responseMessageID = messages[2].id
        revisionA.name = "First bracket"
        revisionA.validatedHash = String(repeating: "ab", count: 32)
        var thicker = bracket
        thicker.script += "\n# Thicker base: 8.000000000000002 mm\n"
        thicker.inputAssets = [CADInputAsset(id: "reference", name: "reference.stl", format: "stl", data: stl)]
        thicker.reorganizePartIDs = ["upright"]
        var revisionB = ProjectRevision(id: second, model: thicker, date: date(181), parts: parts)
        revisionB.userMessageID = messages[3].id
        revisionB.responseMessageID = messages[4].id
        revisionB.partColors = ["upright": PartColor(red: 0.75, green: 0.1, blue: 0.125), "base": PartColor(red: 1, green: 0, blue: -0.0)]
        revisionB.jointPositions = ["hinge": -0.0, "slider": 12.345678901234567]
        let codex = ProjectTaskAudit(id: id(30), userMessageID: messages[0].id, startedAt: date(1), finishedAt: date(60),
                                     agent: "codex", model: "gpt-5", reasoningEffort: "high", status: "completed", events: [
            ProjectActivityEvent(id: "build-1", date: date(2), category: "tool", title: "rad_build", status: "completed",
                                 input: "{\"script\":\"result = 1\"}", output: "Built 2 solids in 0.1 s", durationSeconds: 1.25),
            ProjectActivityEvent(id: "change-1", date: date(3), category: "file_change", title: "candidate.py", status: "completed",
                                 files: [ProjectActivityFile(name: "candidate.py", path: "/tmp/task/candidate.py", data: Data(model.script.utf8))]),
            ProjectActivityEvent(id: "render-1", date: date(4), category: "tool", title: "rad_render", status: "completed",
                                 output: "{\"zoom\":0.1}", durationSeconds: 0.30000000000000004, imageData: png),
            ProjectActivityEvent(id: "command-1", date: date(5), category: "command", title: "ls", status: "failed",
                                 input: "ls", output: "", durationSeconds: 0)
        ], additionalUserMessageIDs: [messages[3].id])
        let external = ProjectTaskAudit(id: id(31), startedAt: date(190), finishedAt: date(195), model: "helper-model",
                                        status: "completed", events: [
            ProjectActivityEvent(id: "inspect-1", date: date(191), category: "diagnostic", title: "Inspect", status: "completed",
                                 output: "Bounds -0 to 40")
        ], externalClient: ProjectExternalAgentIdentity(clientID: "client-1", name: "Helper", version: "1.0"),
           request: "Inspect the bracket", committedRevisionID: second, commitKey: "commit-1", commitArtifactID: "artifact-1")
        var settings = CodexProjectSettings(vendor: .claude, model: "opus", reasoningEffort: "xhigh", reasoningByModel: ["opus": "xhigh"])
        settings.otherAgents = ["codex": AgentModelChoice(model: "gpt-5", reasoningEffort: "high", reasoningByModel: ["gpt-5": "high"])]
        return RADProject(messages: messages, revisions: [revisionA, revisionB], selectedRevisionID: second, projectID: id(99),
                          codexSettings: settings, projectName: "Legacy bracket", showGrid: false, projectionMode: .isometric,
                          displayMode: .shaded, recentPartColors: [PartColor(red: 0.1, green: 0.2, blue: 0.3)],
                          taskAudits: [codex, external],
                          queuedRequests: [
                            ProjectQueuedRequest(id: id(40), content: "Add fillets", date: date(200.01), attachments: [source],
                                                 codexSettings: settings, selection: selection),
                            ProjectQueuedRequest(id: id(41), content: "Then color it", date: date(200.02), attachments: [])
                          ],
                          draftPrompt: "Draft with -0", draftAttachments: [image], queuePaused: true,
                          partsPaneFraction: 0.6180339887498949)
    }
}
