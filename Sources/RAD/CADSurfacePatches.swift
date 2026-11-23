import Foundation
import simd

/// Measurements of a connected region of one part's display surface, in its
/// built coordinates. The representative point lies on a triangle even when
/// the area-weighted centroid falls inside a curved part.
struct CADSurfacePatch: Hashable, Sendable {
    let solidIndex: Int
    let index: Int
    let area: Double
    let centroid: [Double]
    let bounds: [Double]
    let point: [Double]
    let normal: [Double]

    var isValid: Bool {
        guard (0..<1_000_000).contains(solidIndex), index >= 0, area > 0, area.isFinite, centroid.count == 3,
              bounds.count == 6, point.count == 3, normal.count == 3 else { return false }
        let withinLimits: (Double) -> Bool = { abs($0) <= 1_000_000.1 }
        return centroid.allSatisfy(withinLimits) && bounds.allSatisfy(withinLimits) && point.allSatisfy(withinLimits)
            && (0..<3).allSatisfy { bounds[$0] <= bounds[$0 + 3] }
            && abs(simd_length_squared(SIMD3<Double>(normal)) - 1) < 1e-5
    }

    static func == (lhs: Self, rhs: Self) -> Bool {
        lhs.solidIndex == rhs.solidIndex && lhs.index == rhs.index
    }

    func hash(into hasher: inout Hasher) {
        hasher.combine(solidIndex)
        hasher.combine(index)
    }
}

/// Patch lookup follows the triangle order of each solid's actual SceneKit
/// geometry element. Mask indices are grouped once so picking or highlighting
/// a patch never needs to scan the other triangles.
struct CADSurfacePatches: Sendable {
    private struct Connected: Sendable {
        var measures: [Measures]
        var facePatch: [Int32]
        var offsets: [Int]
        var indices: Data
    }
    private enum Solid: Sendable {
        case connected(Connected)
        case separateFaces(positions: [Float], triangles: [Int32])
    }

    private var solids = [Int: Solid]()

    func patch(solid: Int, faceIndex: Int) -> CADSurfacePatch? {
        switch solids[solid] {
        case .connected(let part):
            guard part.facePatch.indices.contains(faceIndex) else { return nil }
            let index = Int(part.facePatch[faceIndex])
            guard part.measures.indices.contains(index) else { return nil }
            return part.measures[index].patch(solid: solid, index: index)
        case .separateFaces(let positions, let triangles):
            guard (0..<triangles.count / 3).contains(faceIndex) else { return nil }
            let i = 3 * faceIndex
            let a = Self.point(triangles[i], positions), b = Self.point(triangles[i + 1], positions)
            let c = Self.point(triangles[i + 2], positions), area = simd_length(simd_cross(b - a, c - a)) / 2
            guard area > 0 else { return nil }
            var facts = Measures()
            facts.add(a, b, c, area: area)
            return facts.patch(solid: solid, index: faceIndex)
        case nil: return nil
        }
    }

    func triangleIndices(for patch: CADSurfacePatch) -> Data? {
        switch solids[patch.solidIndex] {
        case .connected(let part):
            guard part.measures.indices.contains(patch.index) else { return nil }
            return part.indices[part.offsets[patch.index]..<part.offsets[patch.index + 1]]
        case .separateFaces(_, let triangles):
            guard (0..<triangles.count / 3).contains(patch.index) else { return nil }
            return triangles.withUnsafeBufferPointer { buffer in
                Data(buffer: UnsafeBufferPointer(rebasing: buffer[3 * patch.index..<3 * patch.index + 3]))
            }
        case nil: return nil
        }
    }

    private static func side(_ a: UInt32, _ b: UInt32) -> UInt64 {
        UInt64(min(a, b)) << 32 | UInt64(max(a, b))
    }

    private static func point(_ index: Int32, _ positions: [Float]) -> SIMD3<Double> {
        let i = Int(index) * 3
        return SIMD3(Double(positions[i]), Double(positions[i + 1]), Double(positions[i + 2]))
    }

    init() {}

    init(positions: [Float], groups: [Int: [Int32]], edgePoints: [Float], edgeGroups: [Int: [Int]]) throws {
        // Stroke coordinates and shading-split vertices are copies of the same
        // Float32 positions. Exact matches avoid bridging close parallel walls.
        let count = positions.count / 3
        var welded = [UInt32]()
        welded.reserveCapacity(count)
        var barriers = [Int: Set<UInt64>]()
        do {
            var ids = [SIMD3<Float>: UInt32]()
            ids.reserveCapacity(count)
            for index in 0..<count {
                if index.isMultiple(of: 1024) { try Task.checkCancellation() }
                let offset = index * 3
                let point = SIMD3(positions[offset], positions[offset + 1], positions[offset + 2])
                if let id = ids[point] { welded.append(id) }
                else { ids[point] = UInt32(index); welded.append(UInt32(index)) }
            }
            for (solid, segments) in edgeGroups {
                var sides = Set<UInt64>()
                sides.reserveCapacity(segments.count)
                for (index, segment) in segments.enumerated() {
                    if index.isMultiple(of: 1024) { try Task.checkCancellation() }
                    let i = segment * 6
                    let a = SIMD3(edgePoints[i], edgePoints[i + 1], edgePoints[i + 2])
                    let b = SIMD3(edgePoints[i + 3], edgePoints[i + 4], edgePoints[i + 5])
                    if let first = ids[a], let second = ids[b], first != second {
                        sides.insert(Self.side(first, second))
                    }
                }
                barriers[solid] = sides
            }
        }
        var incidence = VertexIncidence(vertexCount: count)
        for (solid, triangles) in groups {
            try Task.checkCancellation()
            solids[solid] = try Self.makeSolid(triangles: triangles, positions: positions, welded: welded,
                                               barriers: barriers[solid] ?? barriers[-1] ?? [], incidence: &incidence)
        }
    }

    /// List the sides by their lower welded vertex, then count the faces at
    /// each upper vertex. Both passes are linear even for a high-valence fan.
    /// Reuse the vertex tables across all solids without clearing the entire
    /// design's vertex count for each part.
    private struct VertexIncidence {
        var heads: [Int32]
        var firstFace: [Int32]
        var secondFace: [Int32]

        init(vertexCount: Int) {
            heads = [Int32](repeating: -1, count: vertexCount)
            firstFace = heads
            secondFace = heads
        }

        mutating func joinManifoldSides(triangles: [Int32], welded: [UInt32], barriers: Set<UInt64>,
                                        areas: [Double], components: inout Components) throws {
            var links = [SIMD2<Int32>](repeating: .zero, count: triangles.count)
            var lowerVertices = [UInt32]()
            for face in areas.indices {
                if face.isMultiple(of: 1024) { try Task.checkCancellation() }
                guard areas[face] > 0 else { continue }
                let i = face * 3
                var a = welded[Int(triangles[i + 2])]
                for corner in i..<(i + 3) {
                    let b = welded[Int(triangles[corner])]
                    let low = min(a, b), high = max(a, b)
                    if heads[Int(low)] == -1 { lowerVertices.append(low) }
                    links[corner] = SIMD2(heads[Int(low)], Int32(high))
                    heads[Int(low)] = Int32(corner)
                    a = b
                }
            }
            for (index, low) in lowerVertices.enumerated() {
                if index.isMultiple(of: 512) { try Task.checkCancellation() }
                var corner = heads[Int(low)]
                while corner >= 0 {
                    let link = links[Int(corner)], high = Int(link.y)
                    if firstFace[high] == -1 {
                        firstFace[high] = corner / 3
                        secondFace[high] = -1
                    } else if secondFace[high] == -1 {
                        secondFace[high] = corner / 3
                    } else {
                        secondFace[high] = -2 // Nonmanifold, regardless of further incidents.
                    }
                    corner = link.x
                }
                // Only now can we know that exactly two faces shared a side.
                corner = heads[Int(low)]
                while corner >= 0 {
                    let link = links[Int(corner)], high = Int(link.y), first = firstFace[high]
                    if first >= 0 {
                        if secondFace[high] >= 0, !barriers.contains(CADSurfacePatches.side(low, UInt32(high))) {
                            components.join(first, secondFace[high])
                        }
                        firstFace[high] = -1
                    }
                    corner = link.x
                }
                heads[Int(low)] = -1
            }
        }
    }

    private struct Components {
        // A root stores its negative component size, other entries the parent.
        var parents: [Int32]
        var joined = false

        mutating func root(of face: Int32) -> Int32 {
            var root = face
            while parents[Int(root)] >= 0 { root = parents[Int(root)] }
            var current = face
            while current != root {
                let next = parents[Int(current)]
                parents[Int(current)] = root
                current = next
            }
            return root
        }

        mutating func join(_ first: Int32, _ second: Int32) {
            var a = root(of: first), b = root(of: second)
            guard a != b else { return }
            if parents[Int(a)] > parents[Int(b)] { swap(&a, &b) }
            parents[Int(a)] += parents[Int(b)]
            parents[Int(b)] = a
            joined = true
        }
    }

    /// Packed metrics avoid separate coordinate-array allocations for every
    /// patch in a heavily faceted part. Only a picked record needs those arrays.
    private struct Measures: Sendable {
        var area = 0.0
        var weightedCenter = SIMD3<Double>.zero
        var low = SIMD3<Double>(repeating: .infinity)
        var high = SIMD3<Double>(repeating: -.infinity)
        var largest = 0.0
        var point = SIMD3<Double>.zero
        var normal = SIMD3<Double>.zero
        var triangleCount = 0

        mutating func add(_ a: SIMD3<Double>, _ b: SIMD3<Double>, _ c: SIMD3<Double>,
                          area weight: Double) {
            let center = (a + b + c) / 3
            weightedCenter += weight * center
            area += weight
            low = simd_min(low, simd_min(a, simd_min(b, c)))
            high = simd_max(high, simd_max(a, simd_max(b, c)))
            triangleCount += 1
            if weight > largest {
                largest = weight
                point = center
                normal = simd_cross(b - a, c - a) / (2 * weight)
            }
        }

        func patch(solid: Int, index: Int) -> CADSurfacePatch {
            let center = weightedCenter / area
            return CADSurfacePatch(solidIndex: solid, index: index, area: area, centroid: [center.x, center.y, center.z],
                                   bounds: [low.x, low.y, low.z, high.x, high.y, high.z],
                                   point: [point.x, point.y, point.z], normal: [normal.x, normal.y, normal.z])
        }
    }

    private static func makeSolid(triangles: [Int32], positions: [Float], welded: [UInt32],
                                  barriers: Set<UInt64>, incidence: inout VertexIncidence) throws -> Solid {
        let count = triangles.count / 3
        var components = Components(parents: [Int32](repeating: -1, count: count))
        func point(_ corner: Int) -> SIMD3<Double> {
            Self.point(triangles[corner], positions)
        }
        var areas = [Double](repeating: 0, count: count)
        for face in 0..<count {
            if face.isMultiple(of: 1024) { try Task.checkCancellation() }
            let i = face * 3, a = point(i)
            areas[face] = simd_length(simd_cross(point(i + 1) - a, point(i + 2) - a)) / 2
        }
        try incidence.joinManifoldSides(triangles: triangles, welded: welded, barriers: barriers,
                                        areas: areas, components: &components)
        // Each triangle of an entirely faceted or disconnected part already
        // is its own patch. Keep the original shared buffers; one triangle's
        // measurements are constant work when the user actually picks it.
        guard components.joined else { return .separateFaces(positions: positions, triangles: triangles) }
        var rootPatch = [Int32](repeating: -1, count: count), facePatch = rootPatch
        var measures = [Measures]()
        for face in 0..<count {
            if face.isMultiple(of: 1024) { try Task.checkCancellation() }
            let i = face * 3
            let area = areas[face]
            guard area > 0 else { continue }
            let root = Int(components.root(of: Int32(face)))
            if rootPatch[root] < 0 {
                rootPatch[root] = Int32(measures.count)
                measures.append(Measures())
            }
            let patch = rootPatch[root]
            facePatch[face] = patch
            measures[Int(patch)].add(point(i), point(i + 1), point(i + 2), area: area)
        }

        let bytesPerTriangle = 3 * MemoryLayout<Int32>.stride
        var offsets = [0]
        offsets.reserveCapacity(measures.count + 1)
        for (index, facts) in measures.enumerated() {
            if index.isMultiple(of: 1024) { try Task.checkCancellation() }
            offsets.append(offsets[index] + facts.triangleCount * bytesPerTriangle)
        }
        var data = Data(count: offsets.last ?? 0)
        try data.withUnsafeMutableBytes { raw in
            let indices = raw.bindMemory(to: Int32.self)
            var next = offsets.map { $0 / MemoryLayout<Int32>.stride }
            for (face, patch) in facePatch.enumerated() where patch >= 0 {
                if face.isMultiple(of: 1024) { try Task.checkCancellation() }
                let p = Int(patch), i = next[p], start = face * 3
                indices[i] = triangles[start]
                indices[i + 1] = triangles[start + 1]
                indices[i + 2] = triangles[start + 2]
                next[p] += 3
            }
        }
        return .connected(Connected(measures: measures, facePatch: facePatch, offsets: offsets, indices: data))
    }
}
