import Foundation

/// A picked point on a part's surface: the ray hit on the display mesh, in
/// the part's built (unposed) coordinates, with its outward unit normal.
/// Points carry no topology identity; they are at the preview's resolution.
struct CADSurfacePoint: Codable, Hashable, Sendable {
    var solidIndex: Int
    var point: [Double]
    var normal: [Double]

    private enum CodingKeys: String, CodingKey {
        case solidIndex = "solid", point, normal
    }

    /// Finite coordinates within the design limits and a unit normal.
    var isValid: Bool {
        solidIndex >= 0 && solidIndex < 1_000_000 && point.count == 3 && normal.count == 3 &&
            point.allSatisfy { $0.isFinite && abs($0) <= 1_000_000.1 } &&
            normal.allSatisfy(\.isFinite) && abs(normal.reduce(0) { $0 + $1 * $1 } - 1) <= 1e-3
    }
}

/// One part's preview mesh statistics (`partStats`), in built coordinates
/// unless posed. Values are at the preview mesher's resolution.
struct CADPart: Codable, Sendable, Equatable {
    var solidIndex: Int
    var area: Double
    var volume: Double
    var bounds: [Double]
    var centroid: [Double]
    var triangles: Int
    /// Disconnected pieces of this one rigid part (outward surfaces).
    var components: Int
    /// Sealed internal voids (surfaces enclosing negative volume).
    var cavities: Int? = nil
    /// Bound of the distance from the preview triangles to the exact surface, in millimeters.
    var maxDeviation: Double? = nil

    private enum CodingKeys: String, CodingKey {
        case solidIndex = "solid", area, volume, bounds, centroid, triangles, components, cavities, maxDeviation
    }
}

enum CADMeasurementUnit: String, Codable, Sendable {
    case length = "mm"
    case area = "mm²"
    case volume = "mm³"
    case angle = "°"
    case scalar = ""
}

struct CADMeasurementMetric: Codable, Sendable, Identifiable {
    var id: String
    var label: String
    var values: [Double]
    var unit: CADMeasurementUnit

    var formattedValue: String {
        let number = values.map { value in
            (abs(value) < 0.0000005 ? 0 : value).formatted(.number.precision(.fractionLength(0...6)))
        }.joined(separator: ", ")
        return number + (unit == .scalar ? "" : unit == .angle ? unit.rawValue : " " + unit.rawValue)
    }
}

struct CADMeasurementReport: Codable, Sendable {
    var title: String
    var metrics: [CADMeasurementMetric]
    var points: [CADSurfacePoint]
    var solidIndices: [Int]
    var closestPoints: [[Double]]? = nil
}

enum CADMeasurements {
    /// Metric identifiers of the selected point at `index` share this prefix.
    static func pointPrefix(_ index: Int) -> String { "point\(index + 1):" }

    /// Part statistics at the current rigid placements. Volume and area do
    /// not change; the centroid moves with its part, and the posed bounds
    /// enclose the transformed corners of the built bounds.
    static func posedParts(_ parts: [CADPart], transforms: [Int: [Double]]) -> [CADPart] {
        parts.map { part in
            guard let matrix = transforms[part.solidIndex], part.bounds.count == 6, part.centroid.count == 3 else { return part }
            var posed = part
            posed.centroid = AssemblySolidTransform.point(part.centroid, by: matrix)
            var minimum = [Double](repeating: .infinity, count: 3), maximum = [Double](repeating: -.infinity, count: 3)
            for corner in 0..<8 {
                let point = AssemblySolidTransform.point((0..<3).map { part.bounds[corner >> $0 & 1 == 0 ? $0 : $0 + 3] }, by: matrix)
                for axis in 0..<3 {
                    minimum[axis] = min(minimum[axis], point[axis])
                    maximum[axis] = max(maximum[axis], point[axis])
                }
            }
            posed.bounds = minimum + maximum
            return posed
        }
    }

    /// A surface point at its part's current rigid placement.
    static func posedPoint(_ point: CADSurfacePoint, transforms: [Int: [Double]]) -> (position: [Double], normal: [Double]) {
        guard let matrix = transforms[point.solidIndex] else { return (point.point, point.normal) }
        return (AssemblySolidTransform.point(point.point, by: matrix),
                AssemblySolidTransform.point(point.normal, by: matrix, direction: true))
    }

    /// Point and part measurements computed locally from the selection and
    /// the build's statistics, posed with `solidTransforms`. Distances to a
    /// part come from the worker (`EngineService.measure`).
    static func report(_ preview: PreviewMesh, points: [CADSurfacePoint] = [], solids: Set<Int>,
                       solidTransforms: [Int: [Double]] = [:]) throws -> CADMeasurementReport {
        guard !solids.isEmpty || !points.isEmpty else {
            throw RADError.message("Select a part or a surface point to measure.")
        }
        guard solids.allSatisfy({ $0 >= 0 && $0 < preview.solids }),
              points.allSatisfy({ $0.isValid && $0.solidIndex < preview.solids }),
              Set(points).count == points.count else {
            throw RADError.message("The selection is no longer available. Select it again.")
        }
        var metrics: [CADMeasurementMetric] = []
        func add(_ id: String, _ label: String, _ values: [Double], _ unit: CADMeasurementUnit) {
            metrics.append(CADMeasurementMetric(id: id, label: label, values: values, unit: unit))
        }
        let posed = points.map { posedPoint($0, transforms: solidTransforms) }
        for (index, point) in posed.enumerated() {
            let prefix = points.count > 1 || !solids.isEmpty ? "Point \(index + 1) · " : ""
            add(pointPrefix(index) + "position", prefix + "Coordinates X, Y, Z", point.position, .length)
            add(pointPrefix(index) + "normal", prefix + "Normal", point.normal, .scalar)
        }
        var closestPoints: [[Double]]?
        if posed.count == 2 {
            let first = posed[0].position, second = posed[1].position
            add("pair:distance", "Distance", [distance(first, second)], .length)
            add("pair:delta", "Delta X, Y, Z", subtract(second, first), .length)
            closestPoints = [first, second]
        }
        if posed.count == 3 {
            let corners = posed.map(\.position)
            let a = subtract(corners[0], corners[1]), b = subtract(corners[2], corners[1])
            let lengths = [distance(corners[0], corners[1]), distance(corners[1], corners[2]), distance(corners[2], corners[0])]
            if lengths[0] > 1e-12, lengths[1] > 1e-12 {
                add("triangle:angle", "Angle at point 2", [acos(max(-1, min(1, dot(a, b) / (lengths[0] * lengths[1])))) * 180 / .pi], .angle)
            }
            let cross = [a[1] * b[2] - a[2] * b[1], a[2] * b[0] - a[0] * b[2], a[0] * b[1] - a[1] * b[0]]
            add("triangle:area", "Triangle area", [sqrt(dot(cross, cross)) / 2], .area)
            add("triangle:perimeter", "Triangle perimeter", [lengths.reduce(0, +)], .length)
            add("triangle:sides", "Triangle sides", lengths, .length)
        }
        if !solids.isEmpty {
            guard let stats = preview.partStats else {
                throw RADError.message("Rebuild the design to enable measurements.")
            }
            let parts = posedParts(stats.filter { solids.contains($0.solidIndex) }, transforms: solidTransforms)
            guard parts.count == solids.count, parts.allSatisfy({ $0.volume.isFinite && $0.volume > 0 && $0.area.isFinite && $0.area > 0 && $0.bounds.count == 6 && $0.centroid.count == 3 }) else {
                throw RADError.message("The selected parts are no longer available. Select them again.")
            }
            let volume = parts.reduce(0) { $0 + $1.volume }
            let area = parts.reduce(0) { $0 + $1.area }
            guard volume.isFinite, volume > 0, area.isFinite else { throw RADError.message("The selected parts have invalid measurement metadata.") }
            let minimum = (0..<3).map { axis in parts.map { $0.bounds[axis] }.min()! }
            let maximum = (0..<3).map { axis in parts.map { $0.bounds[axis + 3] }.max()! }
            let centroid = (0..<3).map { axis in parts.reduce(0) { $0 + $1.centroid[axis] * $1.volume } / volume }
            add("parts:count", "Parts", [Double(parts.count)], .scalar)
            add("parts:volume", "Volume", [volume], .volume)
            add("parts:area", "Surface area", [area], .area)
            add("parts:size", "Size X, Y, Z", zip(maximum, minimum).map(-), .length)
            add("parts:minimum", "Minimum X, Y, Z", minimum, .length)
            add("parts:maximum", "Maximum X, Y, Z", maximum, .length)
            add("parts:centroid", "Volume centroid", centroid, .length)
        }
        let title = points.count == 1 && solids.isEmpty ? "Surface point" : "Measurements"
        return CADMeasurementReport(title: title, metrics: metrics, points: points, solidIndices: solids.sorted(),
                                    closestPoints: closestPoints)
    }

    /// The worker measures the minimum distance only between two parts, or
    /// between a point and a part: the items of a two-item selection with a part.
    static func workerEntities(points: [CADSurfacePoint], solids: Set<Int>) -> [EngineWorkerRequest.MeasureEntity]? {
        guard points.count + solids.count == 2, !solids.isEmpty else { return nil }
        return points.map { .point(solid: $0.solidIndex, point: $0.point) } + solids.sorted().map { .part(solid: $0) }
    }

    private static func subtract(_ a: [Double], _ b: [Double]) -> [Double] { zip(a, b).map(-) }
    private static func dot(_ a: [Double], _ b: [Double]) -> Double { zip(a, b).reduce(0) { $0 + $1.0 * $1.1 } }
    private static func distance(_ a: [Double], _ b: [Double]) -> Double { let delta = subtract(a, b); return sqrt(dot(delta, delta)) }
}
