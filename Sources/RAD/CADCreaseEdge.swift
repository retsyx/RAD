import Foundation
import simd

/// One connected preview crease in a part's built coordinates. A closed curve
/// repeats its first point at the end. Indices are local to the current preview.
struct CADCreaseEdge: Hashable, Sendable {
    let solidIndex: Int
    let index: Int
    let points: [[Double]]
    let length: Double
    let isValid: Bool

    init(solidIndex: Int, index: Int, points: [[Double]]) {
        self.solidIndex = solidIndex
        self.index = index
        self.points = points
        var length = 0.0
        var previous: SIMD3<Double>?
        for coordinates in points {
            guard coordinates.count == 3, coordinates.allSatisfy({ abs($0) <= 1_000_000.1 }) else {
                length = .nan
                break
            }
            let point = SIMD3<Double>(coordinates)
            if let previous { length += simd_distance(previous, point) }
            previous = point
        }
        self.length = length
        isValid = solidIndex >= 0 && index >= 0 && points.count >= 2 && length.isFinite && length > 0
    }

    static func == (lhs: Self, rhs: Self) -> Bool {
        lhs.solidIndex == rhs.solidIndex && lhs.index == rhs.index
    }

    func hash(into hasher: inout Hasher) {
        hasher.combine(solidIndex)
        hasher.combine(index)
    }

    /// Follow undirected segments through degree-two vertices while the line
    /// turns by at most 35 degrees, the kernel's corner rule. A junction ends
    /// every incident curve. A cycle with no corners is a single closed edge.
    static func group(points: [Float], segments: [Int], solid: Int) throws -> [Self] {
        struct Vertex {
            let point: SIMD3<Float>
            var degree = 0
            var first = -1
            var second = -1

            mutating func attach(_ segment: Int) {
                if degree == 0 { first = segment }
                if degree == 1 { second = segment }
                degree += 1
            }
        }
        var lookup = [SIMD3<Float>: Int]()
        var vertices = [Vertex]()
        func vertex(_ point: SIMD3<Float>) -> Int {
            if let existing = lookup[point] { return existing }
            let index = vertices.count
            vertices.append(Vertex(point: point))
            lookup[point] = index
            return index
        }

        var ends = [SIMD2<Int>]()
        var unique = Set<SIMD2<Int>>()
        for (offset, segment) in segments.enumerated() {
            if offset.isMultiple(of: 512) { try Task.checkCancellation() }
            let base = segment * 6
            let first = SIMD3(points[base], points[base + 1], points[base + 2])
            let second = SIMD3(points[base + 3], points[base + 4], points[base + 5])
            // Match the minimum length accepted by the ordinary stroke geometry.
            guard first.x.isFinite, first.y.isFinite, first.z.isFinite, second.x.isFinite, second.y.isFinite,
                  second.z.isFinite, simd_distance_squared(SIMD3<Double>(first), SIMD3<Double>(second)) > 1e-24 else { continue }
            let a = vertex(first), b = vertex(second)
            guard unique.insert(SIMD2(min(a, b), max(a, b))).inserted else { continue }
            vertices[a].attach(ends.count)
            vertices[b].attach(ends.count)
            ends.append(SIMD2(a, b))
        }

        func otherEnd(of segment: Int, from vertex: Int) -> Int {
            ends[segment].x == vertex ? ends[segment].y : ends[segment].x
        }
        let cosine = cos(35 * Double.pi / 180)
        var continues = [Bool](repeating: false, count: vertices.count)
        for (index, v) in vertices.enumerated() where v.degree == 2 {
            if index.isMultiple(of: 512) { try Task.checkCancellation() }
            let center = SIMD3<Double>(v.point)
            let incoming = center - SIMD3<Double>(vertices[otherEnd(of: v.first, from: index)].point)
            let outgoing = SIMD3<Double>(vertices[otherEnd(of: v.second, from: index)].point) - center
            continues[index] = simd_dot(incoming, outgoing) >= cosine * simd_length(incoming) * simd_length(outgoing)
        }

        var visited = [Bool](repeating: false, count: ends.count)
        var curves = [Self]()
        func walk(_ seed: Int, from start: Int) throws {
            var chain = [[Double]]()
            func append(_ vertex: Int) {
                let p = vertices[vertex].point
                chain.append([Double(p.x), Double(p.y), Double(p.z)])
            }
            var segment = seed, vertex = start
            append(start)
            while !visited[segment] {
                if chain.count.isMultiple(of: 512) { try Task.checkCancellation() }
                visited[segment] = true
                vertex = otherEnd(of: segment, from: vertex)
                append(vertex)
                guard continues[vertex] else { break }
                let v = vertices[vertex]
                segment = v.first == segment ? v.second : v.first
            }
            curves.append(Self(solidIndex: solid, index: curves.count, points: chain))
        }
        for (index, pair) in ends.enumerated() where !visited[index] {
            if index.isMultiple(of: 512) { try Task.checkCancellation() }
            if !continues[pair.x] { try walk(index, from: pair.x) }
            else if !continues[pair.y] { try walk(index, from: pair.y) }
        }
        for index in ends.indices where !visited[index] { try walk(index, from: ends[index].x) }
        return curves
    }
}
