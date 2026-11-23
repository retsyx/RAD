import Foundation

/// Compact facts about a selected part at submission time, from the build's
/// statistics. This is request intent, not a serialized mesh or measurement
/// cache. The referenced revision's program remains the source of geometry.
struct ProjectSelectionAnchor: Codable, Sendable, Equatable {
    /// Volume centroid.
    var center: [Double]? = nil
    var area: Double? = nil
    var volume: Double? = nil
    var bounds: [Double]? = nil
}

struct ProjectSelectedPart: Codable, Sendable, Equatable {
    /// Stable user-visible hierarchy identity; a solid-index fallback is used
    /// only when no named part metadata exists in the captured revision.
    var partID: String
    var name: String
    var path: [String]
    var solidIndex: Int
    var anchor: ProjectSelectionAnchor? = nil
}

/// A picked surface point, in its part's built coordinates (before any
/// mechanism pose), with the outward unit normal there.
struct ProjectSelectedPoint: Codable, Sendable, Equatable {
    var partID: String
    var name: String
    var path: [String]
    var solidIndex: Int
    var point: [Double]
    var normal: [Double]
}

/// Compact landmarks identify a picked crease in its origin recipe. The
/// preview's mesh segments and transient chain index stay in the viewport.
struct ProjectSelectedEdge: Codable, Sendable, Equatable {
    var partID: String
    var name: String
    var path: [String]
    var solidIndex: Int
    var closed: Bool
    var length: Double
    var landmarks: [[Double]]
}

/// Request landmarks in built coordinates. The normal is local to the
/// representative point, including on curved patches.
struct ProjectSelectedPatch: Codable, Sendable, Equatable {
    var partID: String
    var name: String
    var path: [String]
    var solidIndex: Int
    var area: Double
    var centroid: [Double]
    var bounds: [Double]
    var point: [Double]
    var normal: [Double]
}

extension CADCreaseEdge {
    /// Open edges include their endpoints and midpoint. Closed rims use
    /// three positions spaced around the loop, all in built coordinates.
    var selectionLandmarks: [[Double]] {
        guard let start = points.first, let end = points.last, points.count >= 2 else { return [] }
        let closed = start == end
        let fractions = closed ? [1.0 / 3, 2.0 / 3] : [0.5]
        let targets = fractions.map { $0 * length }
        var landmarks = [start], traveled = 0.0, target = 0
        for (a, b) in zip(points, points.dropFirst()) {
            let delta = SIMD3(b[0] - a[0], b[1] - a[1], b[2] - a[2])
            let span = (delta * delta).sum().squareRoot()
            while target < targets.count, span > 0, targets[target] <= traveled + span {
                let t = (targets[target] - traveled) / span
                landmarks.append([a[0] + t * delta.x, a[1] + t * delta.y, a[2] + t * delta.z])
                target += 1
            }
            if target == targets.count { break }
            traveled += span
        }
        if !closed { landmarks.append(end) }
        return landmarks
    }
}

struct ProjectSelectionContext: Codable, Sendable, Equatable {
    var revisionID: UUID
    var capturedAt = Date()
    /// Whole-part selections, including explicitly selected folders' contents.
    /// Owning parts of selected points do not enter this collection.
    var parts: [ProjectSelectedPart]
    /// Surface points in the user's selection order.
    var points: [ProjectSelectedPoint]
    var edges: [ProjectSelectedEdge]? = nil
    var patches: [ProjectSelectedPatch]? = nil
    /// Reproduce the selected mechanism pose when reconciling an older request.
    var jointPositions: [String: Double]? = nil
    /// Captured in the 3D printing version view: points and anchors belong to
    /// the program's printing version in its built coordinates.
    var printVersion: Bool? = nil

    static let maximumParts = 8_192
    static let maximumPoints = 8_192
    static let maximumEdges = 8_192
    static let maximumPatches = 8_192

    var isEmpty: Bool { parts.isEmpty && points.isEmpty && (edges ?? []).isEmpty && (patches ?? []).isEmpty }

    static func capture(revisionID: UUID, preview: PreviewMesh?, nodes: [PartNode],
                        selectedSolids: Set<Int>, selectedNodeIDs: Set<String>, selectedPoints: [CADSurfacePoint] = [],
                        selectedEdges: [CADCreaseEdge] = [], selectedPatches: [CADSurfacePatch] = [],
                        assemblyPose: AssemblyPose? = nil, printVersion: Bool = false) -> ProjectSelectionContext? {
        struct Owner {
            var id: String
            var name: String
            var path: [String]
        }
        var owners: [Int: Owner] = [:]
        var wholeParts = selectedSolids
        func visit(_ nodes: [PartNode], ancestors: [String]) {
            for node in nodes {
                let path = ancestors + [node.name]
                for index in node.solidIndices { owners[index] = Owner(id: node.id, name: node.name, path: path) }
                if selectedNodeIDs.contains(node.id) { wholeParts.formUnion(node.allSolidIndices) }
                visit(node.children ?? [], ancestors: path)
            }
        }
        visit(nodes, ancestors: [])
        // Saved hierarchy identities remain selectable while native geometry
        // is replaying. Preserve that part intent without inventing anchors or
        // starting a worker solely to capture the request.
        let solidCount = preview?.solids ?? ((owners.keys.filter { $0 >= 0 && $0 < 1_000_000 }.max() ?? -1) + 1)

        let partFacts = Dictionary((preview?.partStats ?? []).map { ($0.solidIndex, $0) }) { first, _ in first }

        func owner(_ index: Int) -> Owner {
            owners[index] ?? Owner(id: "solid:\(index)", name: "Part \(index + 1)", path: ["Part \(index + 1)"])
        }
        func posed(_ anchor: ProjectSelectionAnchor?, solid: Int) -> ProjectSelectionAnchor? {
            guard var anchor, let matrix = assemblyPose?.solidTransforms[solid] else { return anchor }
            anchor.center = anchor.center.map { AssemblySolidTransform.point($0, by: matrix) }
            // Rotating built bounds does not give the posed part's bounds.
            anchor.bounds = nil
            return anchor
        }
        let parts = wholeParts.filter { $0 >= 0 && $0 < solidCount }.sorted().map { index in
            let identity = owner(index)
            let anchor = partFacts[index].map { fact in
                ProjectSelectionAnchor(center: fact.centroid, area: fact.area, volume: fact.volume, bounds: fact.bounds)
            }
            return ProjectSelectedPart(partID: identity.id, name: identity.name, path: identity.path,
                                       solidIndex: index, anchor: posed(anchor, solid: index))
        }
        var seen = Set<CADSurfacePoint>()
        let points = selectedPoints.filter { $0.isValid && $0.solidIndex < solidCount && seen.insert($0).inserted }
            .prefix(maximumPoints).map { point in
                let identity = owner(point.solidIndex)
                return ProjectSelectedPoint(partID: identity.id, name: identity.name, path: identity.path,
                                            solidIndex: point.solidIndex, point: point.point, normal: point.normal)
            }
        var seenEdges = Set<CADCreaseEdge>()
        let edges = selectedEdges.filter { $0.isValid && $0.solidIndex < solidCount && seenEdges.insert($0).inserted }
            .prefix(maximumEdges).map { edge in
                let identity = owner(edge.solidIndex)
                return ProjectSelectedEdge(partID: identity.id, name: identity.name, path: identity.path,
                    solidIndex: edge.solidIndex, closed: edge.points.first == edge.points.last,
                    length: edge.length, landmarks: edge.selectionLandmarks)
            }
        var seenPatches = Set<CADSurfacePatch>()
        let patches = selectedPatches.filter { $0.isValid && $0.solidIndex < solidCount && seenPatches.insert($0).inserted }
            .prefix(maximumPatches).map { patch in
                let identity = owner(patch.solidIndex)
                return ProjectSelectedPatch(partID: identity.id, name: identity.name, path: identity.path,
                    solidIndex: patch.solidIndex, area: patch.area, centroid: patch.centroid, bounds: patch.bounds,
                    point: patch.point, normal: patch.normal)
            }
        guard !parts.isEmpty || !points.isEmpty || !edges.isEmpty || !patches.isEmpty else { return nil }
        return ProjectSelectionContext(revisionID: revisionID, parts: parts, points: Array(points),
                                       edges: edges.isEmpty ? nil : edges, patches: patches.isEmpty ? nil : patches,
                                       jointPositions: assemblyPose?.positions.isEmpty == false ? assemblyPose?.positions : nil,
                                       printVersion: printVersion ? true : nil)
    }

    func validate() throws {
        if let positions = jointPositions {
            guard positions.count <= 8_192, positions.allSatisfy({
                !$0.key.isEmpty && $0.key.utf8.count <= 1_024 && $0.value.isFinite && abs($0.value) <= 1e9
            }) else { throw RADError.message("The saved selection has invalid assembly positions.") }
        }
        func invalid() -> RADError { .message("The saved selection references are invalid.") }
        func text(_ value: String, maximum: Int = 1_024) -> Bool {
            !value.isEmpty && value.utf8.count <= maximum &&
                value.unicodeScalars.allSatisfy { !CharacterSet.controlCharacters.contains($0) }
        }
        func path(_ values: [String]) -> Bool {
            !values.isEmpty && values.count <= 64 && values.allSatisfy { text($0) }
        }
        func vector(_ values: [Double]?, count: Int = 3) -> Bool {
            values.map { $0.count == count && $0.allSatisfy { $0.isFinite && abs($0) <= 1_000_000.1 } } ?? true
        }
        func nonnegative(_ value: Double?) -> Bool { value.map { $0.isFinite && $0 >= 0 } ?? true }
        func anchor(_ value: ProjectSelectionAnchor?) -> Bool {
            guard let value else { return true }
            return vector(value.center) && nonnegative(value.area) && nonnegative(value.volume) &&
                vector(value.bounds, count: 6) && (value.bounds.map { bounds in (0..<3).allSatisfy { bounds[$0] <= bounds[$0 + 3] } } ?? true)
        }
        func surfacePoint(_ point: ProjectSelectedPoint) -> CADSurfacePoint {
            CADSurfacePoint(solidIndex: point.solidIndex, point: point.point, normal: point.normal)
        }
        func validPatch(_ patch: ProjectSelectedPatch) -> Bool {
            text(patch.partID) && text(patch.name) && path(patch.path) &&
                CADSurfacePatch(solidIndex: patch.solidIndex, index: 0, area: patch.area, centroid: patch.centroid,
                                bounds: patch.bounds, point: patch.point, normal: patch.normal).isValid
        }
        guard capturedAt.timeIntervalSinceReferenceDate.isFinite,
              !isEmpty,
              parts.count <= Self.maximumParts, points.count <= Self.maximumPoints,
              Set(parts.map(\.solidIndex)).count == parts.count,
              Set(points.map(surfacePoint)).count == points.count,
              (patches ?? []).count <= Self.maximumPatches, (patches ?? []).allSatisfy(validPatch),
              (edges ?? []).count <= Self.maximumEdges,
              (edges ?? []).allSatisfy({ $0.solidIndex >= 0 && $0.solidIndex < 1_000_000 && text($0.partID) &&
                  text($0.name) && path($0.path) && $0.length.isFinite && $0.length > 0 &&
                  $0.landmarks.count == 3 && $0.landmarks.allSatisfy { vector($0) } }),
              parts.allSatisfy({ $0.solidIndex >= 0 && $0.solidIndex < 1_000_000 && text($0.partID) && text($0.name) && path($0.path) && anchor($0.anchor) }),
              points.allSatisfy({ surfacePoint($0).isValid && text($0.partID) && text($0.name) && path($0.path) }) else { throw invalid() }
    }
}
