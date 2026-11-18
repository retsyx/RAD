import Foundation

/// The `request.json` a CAD worker reads. Each case is one operation
/// Engine/worker.py accepts and encodes exactly the fields it allows.
enum EngineWorkerRequest: Encodable, Sendable {
    /// An imported input staged beside the request as `path`.
    struct InputAsset: Encodable, Sendable {
        var id: String
        var name: String
        var format: String
        var path: String
    }

    /// A build's RSDF graph staged beside the request as `source.rsdf`. The
    /// identity, the build's own artifact path, lets the long-lived worker
    /// reuse a graph it already loaded; the roots name each part's node.
    struct NativeSource: Sendable {
        var identity: String
        var partRoots: [Int]
    }

    struct Export: Sendable, Hashable {
        /// Nil exports every solid.
        var solidIndices: [Int]?
        var solidTransforms: [AssemblySolidTransform] = []
        /// One STL file per group of solids, in order.
        var individualParts: [[Int]]? = nil
        /// Nil meshes at the worker's export resolution.
        var cellSize: Double? = nil
        /// Simplification tolerance in mm (0 keeps the full mesh; the worker
        /// caps it at a quarter of the export cell); nil uses its default.
        var tolerance: Double? = nil
    }

    /// A measured item: a whole part, or a point on a part's surface in that
    /// part's built coordinates.
    enum MeasureEntity: Encodable, Sendable, Hashable {
        case part(solid: Int)
        case point(solid: Int, point: [Double])

        private enum Key: String, CodingKey { case kind, solid, point }

        func encode(to encoder: Encoder) throws {
            var values = encoder.container(keyedBy: Key.self)
            switch self {
            case .part(let solid):
                try values.encode("part", forKey: .kind)
                try values.encode(solid, forKey: .solid)
            case let .point(solid, point):
                try values.encode("point", forKey: .kind)
                try values.encode(solid, forKey: .solid)
                try values.encode(point, forKey: .point)
            }
        }

        var solid: Int {
            switch self {
            case .part(let solid), .point(let solid, _): solid
            }
        }
    }

    /// Replay a CAD program. Imported geometry is staged beside the request.
    /// `print` builds the program's 3D printing version; `verifyFields` runs
    /// the program and analyses its custom fields instead of meshing.
    case build(title: String, summary: String, units: String, script: String, inputAssets: [InputAsset],
               print: Bool = false, verifyFields: Bool = false)
    case runtime
    case export(NativeSource, Export)
    /// Minimum distance and closest points between one or two items.
    case measure(NativeSource, entities: [MeasureEntity], solidTransforms: [AssemblySolidTransform]?)
    /// Clearance of each pair of solid groups at each pose.
    case clearance(NativeSource, pairs: [[[Int]]], poses: [[AssemblySolidTransform]])
    /// Printing orientations of the selected solids; nil selects every solid.
    case printOrientations(NativeSource, solidIndices: [Int]?, printDown: [PrintDownPin])
    /// Re-mesh every part finer than its preview and report surface defects.
    case validate(NativeSource, cellSize: Double?)
    /// Signed distances and normals at points of the built design, with
    /// optional ray hits; nil probes every solid.
    case probe(NativeSource, solids: [Int]?, points: [[Double]], directions: [[Double]]?, maxDistance: Double?)

    private enum Key: String, CodingKey {
        case operation, title, summary, units, script, inputAssets, print, verifyFields, artifact, artifactIdentity, partRoots,
             solidIndices, solidTransforms, individualParts, cellSize, entities, pairs, poses, printDown,
             solids, points, directions, maxDistance, tolerance
    }

    func encode(to encoder: Encoder) throws {
        var values = encoder.container(keyedBy: Key.self)
        func encodeSource(_ source: NativeSource, operation: String) throws {
            try values.encode(operation, forKey: .operation)
            try values.encode("source.rsdf", forKey: .artifact)
            try values.encode(source.identity, forKey: .artifactIdentity)
            try values.encode(source.partRoots, forKey: .partRoots)
        }
        switch self {
        case let .build(title, summary, units, script, inputAssets, print, verifyFields):
            try values.encode(title, forKey: .title)
            try values.encode(summary, forKey: .summary)
            try values.encode(units, forKey: .units)
            try values.encode(script, forKey: .script)
            if !inputAssets.isEmpty { try values.encode(inputAssets, forKey: .inputAssets) }
            if print { try values.encode(true, forKey: .print) }
            if verifyFields { try values.encode(true, forKey: .verifyFields) }
        case .runtime:
            try values.encode("runtime", forKey: .operation)
        case let .export(source, export):
            try encodeSource(source, operation: "export")
            try values.encodeIfPresent(export.solidIndices, forKey: .solidIndices)
            if !export.solidTransforms.isEmpty { try values.encode(export.solidTransforms, forKey: .solidTransforms) }
            try values.encodeIfPresent(export.individualParts, forKey: .individualParts)
            try values.encodeIfPresent(export.cellSize, forKey: .cellSize)
            try values.encodeIfPresent(export.tolerance, forKey: .tolerance)
        case let .measure(source, entities, solidTransforms):
            try encodeSource(source, operation: "measure")
            try values.encode(entities, forKey: .entities)
            if let solidTransforms, !solidTransforms.isEmpty { try values.encode(solidTransforms, forKey: .solidTransforms) }
        case let .clearance(source, pairs, poses):
            try encodeSource(source, operation: "clearance")
            try values.encode(pairs, forKey: .pairs)
            try values.encode(poses, forKey: .poses)
        case let .printOrientations(source, solidIndices, printDown):
            try encodeSource(source, operation: "print_orient")
            try values.encodeIfPresent(solidIndices, forKey: .solidIndices)
            if !printDown.isEmpty { try values.encode(printDown, forKey: .printDown) }
        case let .validate(source, cellSize):
            try encodeSource(source, operation: "validate")
            try values.encodeIfPresent(cellSize, forKey: .cellSize)
        case let .probe(source, solids, points, directions, maxDistance):
            try encodeSource(source, operation: "probe")
            try values.encodeIfPresent(solids, forKey: .solids)
            try values.encode(points, forKey: .points)
            try values.encodeIfPresent(directions, forKey: .directions)
            try values.encodeIfPresent(maxDistance, forKey: .maxDistance)
        }
    }

    func write(to folder: URL) throws {
        try JSONEncoder().encode(self).write(to: folder.appendingPathComponent("request.json"), options: .atomic)
    }
}
