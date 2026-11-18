import Foundation
import XCTest
@testable import RAD

/// Engine/worker.py accepts each operation's exact field set; these are the
/// fields the app sent before requests were typed.
final class EngineWorkerRequestTests: XCTestCase {
    private let source = EngineWorkerRequest.NativeSource(identity: "/builds/a/model.rsdf", partRoots: [4, 9, 12])
    private let pose = AssemblySolidTransform(solid: 1, matrix: [1, 0, 0, 5, 0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1])

    func testBuildRequestsCarryOnlyTheProgramAndStagedInputs() throws {
        let plain = EngineWorkerRequest.build(title: "Box", summary: "A box.", units: "mm", script: "result = 1\n",
                                              inputAssets: [])
        XCTAssertEqual(try object(plain), ["title": "Box", "summary": "A box.", "units": "mm", "script": "result = 1\n"])
        let imported = EngineWorkerRequest.build(title: "Box", summary: "A box.", units: "mm", script: "result = 1\n",
            inputAssets: [.init(id: "asset-1", name: "part.stl", format: "stl", path: "input-000.stl")])
        XCTAssertEqual(try object(imported), [
            "title": "Box", "summary": "A box.", "units": "mm", "script": "result = 1\n",
            "inputAssets": [["id": "asset-1", "name": "part.stl", "format": "stl", "path": "input-000.stl"]]
        ])
        let printing = EngineWorkerRequest.build(title: "Box", summary: "A box.", units: "mm", script: "result = 1\n",
                                                 inputAssets: [], print: true)
        XCTAssertEqual(try object(printing), ["title": "Box", "summary": "A box.", "units": "mm", "script": "result = 1\n",
                                              "print": true])
        let fields = EngineWorkerRequest.build(title: "Box", summary: "A box.", units: "mm", script: "result = 1\n",
                                               inputAssets: [], verifyFields: true)
        XCTAssertEqual(try object(fields), ["title": "Box", "summary": "A box.", "units": "mm", "script": "result = 1\n",
                                            "verifyFields": true])
        XCTAssertEqual(try object(.runtime), ["operation": "runtime"])
    }

    func testNativeRequestsNameTheStagedGraphAndOmitDefaultOptions() throws {
        let native: NSDictionary = ["artifact": "source.rsdf", "artifactIdentity": "/builds/a/model.rsdf", "partRoots": [4, 9, 12]]
        func with(_ fields: [String: Any]) -> NSDictionary {
            let merged = NSMutableDictionary(dictionary: native)
            merged.addEntries(from: fields)
            return merged
        }
        XCTAssertEqual(try object(.export(source, .init(solidIndices: nil))), with(["operation": "export"]))
        let export = EngineWorkerRequest.Export(solidIndices: [0, 2], solidTransforms: [pose], individualParts: [[0], [2]],
                                                cellSize: 0.25, tolerance: 0.02)
        XCTAssertEqual(try object(.export(source, export)), with([
            "operation": "export", "solidIndices": [0, 2], "solidTransforms": [["solid": 1, "matrix": pose.matrix]],
            "individualParts": [[0], [2]], "cellSize": 0.25, "tolerance": 0.02
        ]))
        XCTAssertEqual(try object(.measure(source, entities: [.part(solid: 0), .point(solid: 2, point: [1, 2, 3])],
                                           solidTransforms: nil)), with([
            "operation": "measure", "entities": [["kind": "part", "solid": 0], ["kind": "point", "solid": 2, "point": [1, 2, 3]]]
        ]))
        XCTAssertEqual(try object(.measure(source, entities: [.part(solid: 1)], solidTransforms: [pose])), with([
            "operation": "measure", "entities": [["kind": "part", "solid": 1]],
            "solidTransforms": [["solid": 1, "matrix": pose.matrix]]
        ]))
        XCTAssertEqual(try object(.clearance(source, pairs: [[[0], [1, 2]]], poses: [[pose], []])), with([
            "operation": "clearance", "pairs": [[[0], [1, 2]]], "poses": [[["solid": 1, "matrix": pose.matrix]], []]
        ]))
        XCTAssertEqual(try object(.printOrientations(source, solidIndices: nil, printDown: [])), with(["operation": "print_orient"]))
        XCTAssertEqual(try object(.printOrientations(source, solidIndices: [2],
                                                     printDown: [.init(solid: 2, direction: [0, 0, 1])])), with([
            "operation": "print_orient", "solidIndices": [2], "printDown": [["solid": 2, "direction": [0, 0, 1]]]
        ]))
        XCTAssertEqual(try object(.validate(source, cellSize: nil)), with(["operation": "validate"]))
        XCTAssertEqual(try object(.validate(source, cellSize: 0.1)), with(["operation": "validate", "cellSize": 0.1]))
        XCTAssertEqual(try object(.probe(source, solids: nil, points: [[0, 0, 0]], directions: nil, maxDistance: nil)),
                       with(["operation": "probe", "points": [[0, 0, 0]]]))
        XCTAssertEqual(try object(.probe(source, solids: [1], points: [[0, 0, 0]], directions: [[0, 0, 1]], maxDistance: 50)),
                       with(["operation": "probe", "solids": [1], "points": [[0, 0, 0]], "directions": [[0, 0, 1]],
                             "maxDistance": 50]))
    }

    func testRequestIsWrittenWhereTheWorkerReadsIt() throws {
        let folder = FileManager.default.temporaryDirectory.appendingPathComponent("RAD-worker-request-\(UUID().uuidString)")
        try FileManager.default.createDirectory(at: folder, withIntermediateDirectories: false)
        defer { try? FileManager.default.removeItem(at: folder) }
        try EngineWorkerRequest.runtime.write(to: folder)
        let written = try JSONSerialization.jsonObject(with: Data(contentsOf: folder.appendingPathComponent("request.json")))
        XCTAssertEqual(written as? NSDictionary, ["operation": "runtime"])
    }

    private func object(_ request: EngineWorkerRequest) throws -> NSDictionary {
        try XCTUnwrap(JSONSerialization.jsonObject(with: JSONEncoder().encode(request)) as? NSDictionary)
    }
}
