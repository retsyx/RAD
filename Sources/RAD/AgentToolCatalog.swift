import Foundation

/// One CAD tool. Agents see its name, description, input schema and
/// annotations; `host` tells RAD how to authorize, route and audit a call.
struct AgentTool: @unchecked Sendable {
    struct Host: Decodable, Sendable {
        /// What an external client must be allowed to do to call the tool.
        var capability: ExternalAgentCapability
        /// The tool returns the exact PNG it rendered, which is saved with its activity event.
        var returnsImage = false
        /// The tool asks the user in the task's owning design window.
        var requestsPermission = false
        /// Arguments naming CAD programs recorded with the call's activity.
        var programs: [String] = []
        /// The argument naming the file the tool exports. External
        /// clients receive a file RAD assigns instead.
        var exportPath: String?
        /// Arguments naming producer files (conversion code, dependency
        /// descriptions) recorded with the call's activity.
        var sources: [String] = []
        /// Arguments naming original reference inputs. External clients may
        /// name only files they uploaded that RAD did not generate.
        var references: [String] = []

        init(capability: ExternalAgentCapability) { self.capability = capability }

        private enum CodingKeys: String, CodingKey {
            case capability, returnsImage, requestsPermission, programs, exportPath, sources, references
        }

        init(from decoder: Decoder) throws {
            let values = try decoder.container(keyedBy: CodingKeys.self)
            capability = try values.decode(ExternalAgentCapability.self, forKey: .capability)
            returnsImage = try values.decodeIfPresent(Bool.self, forKey: .returnsImage) ?? false
            requestsPermission = try values.decodeIfPresent(Bool.self, forKey: .requestsPermission) ?? false
            programs = try values.decodeIfPresent([String].self, forKey: .programs) ?? []
            exportPath = try values.decodeIfPresent(String.self, forKey: .exportPath)
            sources = try values.decodeIfPresent([String].self, forKey: .sources) ?? []
            references = try values.decodeIfPresent([String].self, forKey: .references) ?? []
        }

        /// Every argument a property names.
        var arguments: [String] { programs + (exportPath.map { [$0] } ?? []) + sources + references }
    }

    let name: String
    let description: String
    /// A JSON Schema object.
    let inputSchema: [String: Any]
    let annotations: [String: Any]
    let host: Host

    /// The tool as `tools/list` presents it.
    var descriptor: [String: Any] {
        ["name": name, "description": description, "inputSchema": inputSchema, "annotations": annotations]
    }
}

/// An immutable set of tools with unique names.
struct AgentToolRegistry: @unchecked Sendable {
    let tools: [AgentTool]
    private let byName: [String: AgentTool]

    init(_ tools: [AgentTool]) throws {
        guard !tools.isEmpty, tools.count <= 128 else { throw RADError.message("RAD's CAD tool catalog is invalid.") }
        var byName: [String: AgentTool] = [:]
        for tool in tools {
            guard byName.updateValue(tool, forKey: tool.name) == nil else {
                throw RADError.message("RAD's CAD tool catalog is invalid.")
            }
        }
        self.tools = tools
        self.byName = byName
    }

    subscript(name: String) -> AgentTool? { byName[name] }

    /// Tools whose exact PNG results are saved with their activity events.
    var imageToolNames: Set<String> { Set(tools.filter(\.host.returnsImage).map(\.name)) }
}

/// The CAD tool catalog has one source, Engine/agent-tools.json, which
/// Engine/agent_server.py also serves. It is read once. The external transport
/// adds explicit task routing and opaque file handles.
enum AgentToolCatalog {
    static func native() throws -> AgentToolRegistry { try nativeCatalog.get() }
    static func external() throws -> AgentToolRegistry { try externalCatalog.get() }

    private static let nativeCatalog = Result {
        try load(Data(contentsOf: ResourcePaths.root.appendingPathComponent("Engine/agent-tools.json")))
    }
    private static let externalCatalog = Result { try AgentToolRegistry(native().tools.map(externalTool)) }

    static func load(_ data: Data) throws -> AgentToolRegistry {
        struct Entry: Decodable { var name: String; var host: AgentTool.Host }
        let invalid = RADError.message("RAD's CAD tool catalog is invalid.")
        guard data.count <= 1_000_000, let objects = try JSONSerialization.jsonObject(with: data) as? [[String: Any]] else {
            throw invalid
        }
        let entries = try JSONDecoder().decode([Entry].self, from: data)
        let tools = try zip(objects, entries).map { object, entry in
            guard object["name"] as? String == entry.name, let description = object["description"] as? String,
                  let schema = object["inputSchema"] as? [String: Any],
                  let properties = schema["properties"] as? [String: Any],
                  let annotations = object["annotations"] as? [String: Any],
                  Set(entry.host.arguments).isSubset(of: properties.keys) else { throw invalid }
            return AgentTool(name: entry.name, description: description, inputSchema: schema,
                             annotations: annotations, host: entry.host)
        }
        return try AgentToolRegistry(tools)
    }

    private static func externalTool(_ native: AgentTool) throws -> AgentTool {
        var schema = native.inputSchema
        guard var properties = schema["properties"] as? [String: Any] else {
            throw RADError.message("RAD's CAD tool catalog is invalid.")
        }
        var required = schema["required"] as? [String] ?? []
        if let program = properties.removeValue(forKey: "programPath") {
            properties["programID"] = program
            required = required.map { $0 == "programPath" ? "programID" : $0 }
        }
        if let output = native.host.exportPath {
            properties.removeValue(forKey: output)
            required.removeAll { $0 == output }
        }
        for key in properties.keys where key == "path" || key.hasSuffix("Path") || key.hasSuffix("Paths") || key == "programID" {
            guard var property = properties[key] as? [String: Any] else { continue }
            property["description"] = "Opaque file ID returned by a task upload or a previous tool result; filesystem paths are not accepted."
            properties[key] = property
        }
        properties["taskID"] = ["type": "string", "description": "Task ID returned by rad_task_begin for the explicitly chosen design."]
        required.append("taskID")
        schema["properties"] = properties
        schema["required"] = required
        let description = native.description.replacingOccurrences(of: "programPath", with: "programID")
            .replacingOccurrences(of: "local paths", with: "opaque file IDs")
            .replacingOccurrences(of: "workspace path", with: "task file ID")
            .replacingOccurrences(of: "successful final response", with: "successful rad_commit")
            .replacingOccurrences(of: "final response", with: "rad_commit")
            + " External task files use opaque upload/download IDs. Builds and color edits are speculative until rad_commit."
        return AgentTool(name: native.name, description: description, inputSchema: schema,
                         annotations: native.annotations, host: native.host)
    }
}
