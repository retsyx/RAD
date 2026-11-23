import Foundation
import simd

/// Screen-space index for one camera, viewport, set of visible parts and set of
/// placements. Each visible segment occupies one bucket at a scale large
/// enough to contain its extent, including long segments across the viewport.
struct CADCreaseProjection {
    private struct Segment {
        let edge: Int
        let a, b: SIMD2<Double>
        let depthA, depthB: Double
    }

    private let edges: [CADCreaseEdge]
    private let orthographic: Bool
    private var segments = [Segment]()
    private var levels = [[SIMD2<Int>: [Int]]]()
    private static let cell = 32.0

    init(edges: [CADCreaseEdge], projection: CameraRig.Projection, placements: [Int: simd_double4x4], hidden: Set<Int>) {
        self.edges = edges
        orthographic = projection.orthographic
        let halfHeight = orthographic ? projection.orthographicScale : projection.tangent
        guard projection.width > 0, projection.height > 0, halfHeight > 0 else { return }
        let center = SIMD2(projection.width, projection.height) / 2
        let scale = center.y / halfHeight
        // Include strokes straddling the viewport border.
        let limit = center + SIMD2(repeating: 2.0)
        for (index, edge) in edges.enumerated() where edge.isValid && !hidden.contains(edge.solidIndex) {
            let placement = placements[edge.solidIndex]
            var previous: SIMD3<Double>?
            for xyz in edge.points {
                let local = SIMD3<Double>(xyz)
                var world = local
                if let placement {
                    let placed = placement * SIMD4(local, 1)
                    world = SIMD3(placed.x, placed.y, placed.z)
                }
                let delta = world - projection.position
                let camera = SIMD3(simd_dot(delta, projection.right) * scale,
                                   simd_dot(delta, projection.up) * scale,
                                   -simd_dot(delta, projection.backward))
                if let previous, let segment = clipped(edge: index, a: previous, b: camera, center: center, limit: limit) {
                    insert(segment)
                }
                previous = camera
            }
        }
    }

    /// Clip in camera space before dividing by depth, including segments that
    /// cross the eye plane. X and Y have already been scaled to view points.
    private func clipped(edge: Int, a: SIMD3<Double>, b: SIMD3<Double>,
                         center: SIMD2<Double>, limit: SIMD2<Double>) -> Segment? {
        let wa = orthographic ? 1.0 : a.z, wb = orthographic ? 1.0 : b.z
        var start = 0.0, end = 1.0
        func clip(_ first: Double, _ second: Double) -> Bool {
            if first < 0 {
                guard second >= 0 else { return false }
                start = max(start, first / (first - second))
            } else if second < 0 {
                end = min(end, first / (first - second))
            }
            return start <= end
        }
        guard clip(a.z - 1e-10, b.z - 1e-10),
              clip(wa * limit.x + a.x, wb * limit.x + b.x),
              clip(wa * limit.x - a.x, wb * limit.x - b.x),
              clip(wa * limit.y + a.y, wb * limit.y + b.y),
              clip(wa * limit.y - a.y, wb * limit.y - b.y) else { return nil }
        let first = start == 0 ? a : a + (b - a) * start
        let last = end == 1 ? b : b + (a - b) * (1 - end)
        guard first.z > 0, last.z > 0 else { return nil }
        let p = center + SIMD2(first.x, first.y) / (orthographic ? 1 : first.z)
        let q = center + SIMD2(last.x, last.y) / (orthographic ? 1 : last.z)
        guard p.x.isFinite, p.y.isFinite, q.x.isFinite, q.y.isFinite else { return nil }
        return Segment(edge: edge, a: p, b: q, depthA: first.z, depthB: last.z)
    }

    private mutating func insert(_ segment: Segment) {
        let span = segment.b - segment.a
        let size = max(abs(span.x), abs(span.y))
        var cell = Self.cell, level = 0
        while cell < size { cell *= 2; level += 1 }
        while levels.count <= level { levels.append([:]) }
        let midpoint = (segment.a + segment.b) / (2 * cell)
        let bucket = SIMD2(Int(floor(midpoint.x)), Int(floor(midpoint.y)))
        levels[level][bucket, default: []].append(segments.count)
        segments.append(segment)
    }

    /// Visibility is tested at the candidate's projected point on the stroke,
    /// with camera-axis depth interpolated there. Several segments on a single
    /// curve may have different visibility.
    func pick(at cursor: SIMD2<Double>, radius: Double,
              isVisible: (CADCreaseEdge, SIMD2<Double>, Double) -> Bool) -> CADCreaseEdge? {
        guard radius.isFinite, radius >= 0, cursor.x.isFinite, cursor.y.isFinite else { return nil }
        struct Hit {
            let segment: Int
            let screen: SIMD2<Double>
            let depth, distanceSquared: Double
        }
        var hits = [Hit]()
        var cell = Self.cell
        for buckets in levels {
            // Segment centers can be up to half their bucket size from the stroke.
            let reach = SIMD2(repeating: radius + cell / 2)
            let low = (cursor - reach) / cell, high = (cursor + reach) / cell
            for y in Int(floor(low.y))...Int(floor(high.y)) {
                for x in Int(floor(low.x))...Int(floor(high.x)) {
                    guard let candidates = buckets[SIMD2(x, y)] else { continue }
                    for index in candidates {
                        let segment = segments[index]
                        let span = segment.b - segment.a
                        let lengthSquared = simd_length_squared(span)
                        let fraction = lengthSquared > 1e-16
                            ? min(1, max(0, simd_dot(cursor - segment.a, span) / lengthSquared))
                            : (segment.depthA <= segment.depthB ? 0.0 : 1.0)
                        let screen = segment.a + fraction * span
                        let distanceSquared = simd_distance_squared(screen, cursor)
                        guard distanceSquared <= radius * radius else { continue }
                        let depth = orthographic
                            ? segment.depthA * (1 - fraction) + segment.depthB * fraction
                            : 1 / ((1 - fraction) / segment.depthA + fraction / segment.depthB)
                        hits.append(Hit(segment: index, screen: screen, depth: depth, distanceSquared: distanceSquared))
                    }
                }
            }
            cell *= 2
        }
        hits.sort {
            if $0.distanceSquared != $1.distanceSquared { return $0.distanceSquared < $1.distanceSquared }
            if $0.depth != $1.depth { return $0.depth < $1.depth }
            return $0.segment < $1.segment
        }
        for hit in hits {
            let edge = edges[segments[hit.segment].edge]
            if isVisible(edge, hit.screen, hit.depth) { return edge }
        }
        return nil
    }
}
