import SwiftUI
import simd

/// RAD's mark in flat form: the outline of the smooth union of two spheres.
/// Scripts/make-icon.swift ray-marches the same spheres and blend radius for
/// the application, document and folder icons; keep the constants identical.
struct RADMark: Shape {
    private static let centerA = SIMD2<Double>(-0.38, -0.16), radiusA = 0.43
    private static let centerB = SIMD2<Double>(0.44, 0.29), radiusB = 0.31
    private static let blend = 0.45

    /// The field's zero contour, traced once in field units (y up).
    private static let outline: [SIMD2<Double>] = {
        let start = centerA - simd_normalize(centerB - centerA) * radiusA
        let step = 0.01
        var point = start
        var points = [start]
        for index in 0..<4_000 {
            let normal = simd_normalize(gradient(point))
            point += SIMD2(-normal.y, normal.x) * step
            for _ in 0..<3 {
                let g = gradient(point)
                point -= g * (field(point) / simd_length_squared(g))
            }
            if index > 10, simd_distance(point, start) < step { break }
            points.append(point)
        }
        return points
    }()

    private static let bounds: (min: SIMD2<Double>, max: SIMD2<Double>) = outline.reduce(
        (SIMD2(repeating: .infinity), SIMD2(repeating: -.infinity))
    ) { (simd_min($0.0, $1), simd_max($0.1, $1)) }

    private static func field(_ p: SIMD2<Double>) -> Double {
        let a = simd_length(p - centerA) - radiusA
        let b = simd_length(p - centerB) - radiusB
        let h = max(blend - abs(a - b), 0) / blend
        return min(a, b) - h * h * blend * 0.25
    }

    private static func gradient(_ p: SIMD2<Double>) -> SIMD2<Double> {
        let e = 1e-6
        return SIMD2(field(p + SIMD2(e, 0)) - field(p - SIMD2(e, 0)),
                     field(p + SIMD2(0, e)) - field(p - SIMD2(0, e))) / (2 * e)
    }

    func path(in rect: CGRect) -> Path {
        let size = Self.bounds.max - Self.bounds.min
        let scale = min(rect.width / size.x, rect.height / size.y)
        let origin = CGPoint(x: rect.midX - size.x * scale / 2, y: rect.midY - size.y * scale / 2)
        var path = Path()
        path.addLines(Self.outline.map { p in
            CGPoint(x: origin.x + (p.x - Self.bounds.min.x) * scale,
                    y: origin.y + (Self.bounds.max.y - p.y) * scale)
        })
        path.closeSubpath()
        return path
    }
}
