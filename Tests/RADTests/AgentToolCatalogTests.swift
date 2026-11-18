import Foundation
import XCTest
@testable import RAD

final class AgentToolCatalogTests: XCTestCase {
    func testHostPropertiesDescribeAuthorizationImagesAndPermissionPrompts() throws {
        let catalog = try AgentToolCatalog.native()
        var capabilities: [ExternalAgentCapability: Set<String>] = [:]
        for tool in catalog.tools { capabilities[tool.host.capability, default: []].insert(tool.name) }
        XCTAssertEqual(capabilities[.capture], ["rad_screenshot"])
        XCTAssertEqual(capabilities[.export], ["rad_export_3mf"])
        XCTAssertEqual(capabilities[.build], ["rad_build", "rad_import", "rad_request_permission", "rad_set_part_colors",
                                              "rad_verify"], "rad_verify runs programs, as builds do.")
        XCTAssertNil(capabilities[.commit], "Only the external transport commits candidates.")
        XCTAssertNil(capabilities[.manageProjects])
        XCTAssertEqual(catalog.imageToolNames, ["rad_render", "rad_screenshot"])
        XCTAssertEqual(ProjectActivityEvent.savedImageTools, catalog.imageToolNames)
        XCTAssertEqual(catalog.tools.filter(\.host.requestsPermission).map(\.name), ["rad_request_permission"])
        XCTAssertEqual(catalog.tools.filter { !$0.host.programs.isEmpty }.map(\.name), ["rad_build", "rad_verify"])
        XCTAssertEqual(catalog.tools.filter { !$0.host.references.isEmpty }.map(\.name), ["rad_import"])
        XCTAssertEqual(catalog.tools.filter { !$0.host.sources.isEmpty }.map(\.name), ["rad_import"])
        XCTAssertNil(catalog["rad_export_step"], "Exports are 3MF.")
        for tool in catalog.tools {
            XCTAssertEqual(tool.annotations["readOnlyHint"] as? Bool, [.read, .capture].contains(tool.host.capability), tool.name)
        }
    }

    func testExternalSchemasNameOnlyTheirOwnParameters() throws {
        // External clients pass programID; no description may name the native programPath.
        for tool in try AgentToolCatalog.external().tools {
            let properties = try XCTUnwrap(tool.inputSchema["properties"] as? [String: Any], tool.name)
            XCTAssertNil(properties["programPath"], tool.name)
            XCTAssertFalse(tool.description.contains("programPath"), tool.name)
            for (name, value) in properties {
                let description = (value as? [String: Any])?["description"] as? String ?? ""
                XCTAssertFalse(description.contains("programPath"), "\(tool.name).\(name): \(description)")
            }
        }
    }

    @MainActor
    func testModelingGuideNamesNoNativeParameter() throws {
        // External clients also download the modeling guide (rad_runtime's apiGuidePath).
        XCTAssertFalse(try CADAgentTools.modelingGuide().contains("programPath"))
    }

    func testCatalogRejectsDuplicateToolsAndHostArgumentsOutsideTheSchema() throws {
        func catalog(_ tools: [[String: Any]]) throws -> Data { try JSONSerialization.data(withJSONObject: tools) }
        func tool(_ name: String, host: [String: Any]) -> [String: Any] {
            ["name": name, "description": "Fixture", "annotations": ["readOnlyHint": true],
             "inputSchema": ["type": "object", "properties": ["path": ["type": "string"]]], "host": host]
        }
        let valid = try AgentToolCatalog.load(catalog([tool("one", host: ["capability": "export", "exportPath": "path"])]))
        XCTAssertEqual(valid["one"]?.host.exportPath, "path")
        XCTAssertEqual(valid["one"]?.host.returnsImage, false)
        XCTAssertThrowsError(try AgentToolCatalog.load(catalog([tool("one", host: ["capability": "read"]),
                                                                tool("one", host: ["capability": "read"])])))
        XCTAssertThrowsError(try AgentToolCatalog.load(catalog([tool("one", host: ["capability": "read", "programs": ["queryPath"]])])))
        XCTAssertThrowsError(try AgentToolCatalog.load(catalog([tool("one", host: ["capability": "build", "references": ["source"]])])))
        XCTAssertThrowsError(try AgentToolCatalog.load(catalog([tool("one", host: ["capability": "build", "sources": ["producers"]])])))
        let references = try AgentToolCatalog.load(catalog([tool("one", host: ["capability": "build", "references": ["path"]])]))
        XCTAssertEqual(references["one"]?.host.references, ["path"])
        XCTAssertThrowsError(try AgentToolCatalog.load(catalog([tool("one", host: ["capability": "everything"])])))
        XCTAssertThrowsError(try AgentToolCatalog.load(catalog([tool("one", host: [:])])))
    }

    func testResultSerializesMetadataOnceAndWrapsItForTheBridge() throws {
        let metadata: [String: Any] = ["path": "/workspace/renders/a.png", "zoom": 1.5, "items": [1, 2], "name": "Bracket"]
        let image = CADAgentToolResult.Image(path: "/workspace/renders/a.png", data: Data([0x89, 0x50]))
        let result = try CADAgentToolResult(metadata: metadata, image: image)
        XCTAssertEqual(result.text, #"{"items":[1,2],"name":"Bracket","path":"/workspace/renders/a.png","zoom":1.5}"#)
        XCTAssertEqual(result.envelope as NSDictionary, ["content": [
            ["type": "text", "text": result.text],
            ["type": "image_path", "path": image.path, "mimeType": "image/png"]
        ]] as NSDictionary)
        let failed = try CADAgentToolResult(metadata: ["status": "error"], isError: true)
        XCTAssertEqual(failed.envelope["isError"] as? Bool, true)
        XCTAssertEqual((failed.envelope["content"] as? [[String: Any]])?.count, 1)
    }

    @MainActor
    func testCallsRecordTheSourceFilesTheirHostPropertiesName() async throws {
        let root = FileManager.default.temporaryDirectory.appendingPathComponent("RAD-tool-sources-\(UUID().uuidString)")
        try FileManager.default.createDirectory(at: root, withIntermediateDirectories: false)
        defer { try? FileManager.default.removeItem(at: root) }
        let tools = try CADAgentTools(directory: root, python: "/missing/python", threads: 1, current: nil,
                                      currentPreview: nil, engine: EngineService())
        let workspace = try CADAgentWorkspace(directory: root)
        try workspace.write(Data("result = box((1, 1, 1), center=(1 / 2, 1 / 2, 1 / 2))\n".utf8), path: "program")
        var events: [ProjectActivityEvent] = []
        let call = CADAgentToolCall(tools: tools, workspace: workspace, onActivity: { events.append($0) })
        let cases: [(String, [String: Any], [String])] = [
            ("rad_build", ["programPath": "program"], ["program.py"])
        ]
        for (index, (name, arguments, expected)) in cases.enumerated() {
            do {
                _ = try await call.perform(id: "tool:\(index)", name: name, arguments: arguments)
                XCTFail("\(name) needs a CAD runtime.")
            } catch {}
            let running = try XCTUnwrap(events.first { $0.id == "tool:\(index)" && $0.status == "running" })
            XCTAssertEqual(running.files?.map(\.name), expected, name)
            XCTAssertEqual(events.last?.status, "failed")
            XCTAssertEqual(events.last?.files?.map(\.name), expected)
        }
    }
}
