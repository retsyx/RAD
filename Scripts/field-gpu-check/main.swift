// GPU field program check: compares the Metal evaluator with the kernel's
// float64 references written by Scripts/field_gpu_validate.py and measures
// throughput. Build (from the repository root):
//   swiftc -O Scripts/field-gpu-check/main.swift Sources/RAD/FieldProgram.swift \
//     Sources/RAD/FieldProgramShaders.swift Sources/RAD/FieldProgramEvaluator.swift -o field-gpu-check
// Run: field-gpu-check <build output folder> <samples.bin>  (prints JSON)
import Foundation
import Metal
import simd

struct Reader {
    let data: Data
    var offset = 0
    mutating func u32() -> Int {
        let value = data.subdata(in: offset..<offset + 4).withUnsafeBytes { $0.loadUnaligned(as: UInt32.self) }
        offset += 4
        return Int(UInt32(littleEndian: value))
    }
    mutating func doubles(_ count: Int) -> [Double] {
        let values = data.subdata(in: offset..<offset + 8 * count).withUnsafeBytes { raw in
            (0..<count).map { raw.loadUnaligned(fromByteOffset: 8 * $0, as: Double.self) }
        }
        offset += 8 * count
        return values
    }
}

func percentile(_ values: [Double], _ p: Double) -> Double {
    guard !values.isEmpty else { return 0 }
    let sorted = values.sorted()
    return sorted[min(sorted.count - 1, Int(p * Double(sorted.count - 1) + 0.5))]
}

func summary(_ values: [Double]) -> [String: Double] {
    ["p50": percentile(values, 0.5), "p90": percentile(values, 0.9), "p99": percentile(values, 0.99),
     "max": values.max() ?? 0, "mean": values.isEmpty ? 0 : values.reduce(0, +) / Double(values.count)]
}

let arguments = CommandLine.arguments
guard arguments.count >= 3 else {
    FileHandle.standardError.write("usage: field-gpu-check <build output> <samples.bin> [benchmark points]\n".data(using: .utf8)!)
    exit(2)
}
let folder = URL(fileURLWithPath: arguments[1])
let resultJSON = try Data(contentsOf: folder.appendingPathComponent("result.json"))
var reader = Reader(data: try Data(contentsOf: URL(fileURLWithPath: arguments[2])))
precondition(reader.u32() == 0x5347_4652, "samples magic")
_ = reader.u32()
let partCount = reader.u32()
let loadStart = Date()
guard let program = try FieldProgram.read(resultJSON: resultJSON, in: folder, solids: partCount) else {
    print("{\"error\": \"no field program\"}")
    exit(1)
}
let loadSeconds = Date().timeIntervalSince(loadStart)
// Experiments: RAD_FIELD_PATCH names a JSON list of [old, new] source replacements.
var source = FieldProgramShaders.library
if let patch = ProcessInfo.processInfo.environment["RAD_FIELD_PATCH"],
   let pairs = try JSONSerialization.jsonObject(with: Data(contentsOf: URL(fileURLWithPath: patch))) as? [[String]] {
    for pair in pairs {
        precondition(source.contains(pair[0]), "patch target missing: \(pair[0])")
        source = source.replacingOccurrences(of: pair[0], with: pair[1])
    }
}
let evaluator = try FieldProgramEvaluator(program: program, source: source)
var report: [String: Any] = ["words": program.words.count, "loadValidateSeconds": loadSeconds, "device": evaluator.device.name]
var parts: [[String: Any]] = []
var allInstructions: [Double] = []

for _ in 0..<partCount {
    let solid = reader.u32(), pointCount = reader.u32(), surfaceCount = reader.u32(), rayCount = reader.u32()
    let cell = reader.doubles(1)[0]
    let points = reader.doubles(3 * pointCount), values = reader.doubles(pointCount)
    let surface = reader.doubles(3 * surfaceCount), kernelNormals = reader.doubles(3 * surfaceCount)
    let sameStepNormals = reader.doubles(3 * surfaceCount)
    let rays = reader.doubles(7 * rayCount), kernelHits = reader.doubles(2 * rayCount)
    var entry: [String: Any] = ["solid": solid]
    let part = program.parts[solid]
    entry["hasField"] = part.hasField
    guard part.hasField else { parts.append(entry); continue }
    let p32 = (0..<pointCount).map { SIMD3<Float>(Float(points[3 * $0]), Float(points[3 * $0 + 1]), Float(points[3 * $0 + 2])) }
    let gpu = try evaluator.distances(part: solid, points: p32)
    var bandErrors: [Double] = [], relative: [Double] = [], nearErrors: [Double] = []
    var signMismatch = 0, signChecked = 0, farSignMismatch = 0, farCount = 0, bandCount = 0, worstBand: [Double] = []
    var scale = 1.0
    for i in 0..<pointCount {
        let magnitude = max(abs(points[3 * i]), abs(points[3 * i + 1]), abs(points[3 * i + 2]))
        scale = max(scale, magnitude)
        let k = values[i], g = Double(gpu[i])
        if program.tape(part: solid, at: p32[i]) != nil {
            bandCount += 1
            let error = abs(g - k)
            bandErrors.append(error)
            relative.append(error / max(magnitude, 1))
            if abs(k) < cell { nearErrors.append(error) }
            if worstBand.isEmpty || error > worstBand[3] { worstBand = [points[3 * i], points[3 * i + 1], points[3 * i + 2], error, k, g] }
            // Signs are decided within float32 rounding of the coordinates.
            if abs(k) > 1e-5 * max(magnitude, 1) {
                signChecked += 1
                if (k < 0) != (g < 0) { signMismatch += 1 }
            }
            let tape = program.tape(part: solid, at: p32[i])!
            if abs(k) < cell { allInstructions.append(Double(program.tapeInstructions[tape] ?? 0)) }
        } else {
            farCount += 1
            if (k < 0) != (g < 0) && abs(k) > 1e-6 { farSignMismatch += 1 }
        }
    }
    entry["points"] = pointCount
    entry["bandPoints"] = bandCount
    entry["farPoints"] = farCount
    entry["bandError"] = summary(bandErrors)
    entry["nearSurfaceError"] = summary(nearErrors)
    entry["relativeError"] = summary(relative)
    entry["worstBand"] = worstBand
    entry["signMismatches"] = signMismatch
    entry["signChecked"] = signChecked
    entry["farSignMismatches"] = farSignMismatch
    entry["coordinateScale"] = scale
    // Normals at surface points against the kernel's (float64, eps 1e-5).
    let s32 = (0..<surfaceCount).map { SIMD3<Float>(Float(surface[3 * $0]), Float(surface[3 * $0 + 1]), Float(surface[3 * $0 + 2])) }
    var normalReport: [String: Any] = [:]
    for factor in [1e-4, 1e-3, 1e-2] {
        let eps = Float(factor * cell)
        let normals = try evaluator.normals(part: solid, points: s32, eps: eps)
        var angles: [Double] = []
        for i in 0..<surfaceCount {
            let k = SIMD3<Double>(kernelNormals[3 * i], kernelNormals[3 * i + 1], kernelNormals[3 * i + 2])
            let g = SIMD3<Double>(normals[i])
            guard simd_length(k) > 0.5, simd_length(g) > 0.5 else { continue }
            angles.append(acos(min(1, max(-1, simd_dot(k, g)))) * 180 / .pi)
        }
        normalReport["eps=\(factor)cell"] = summary(angles)
    }
    entry["normalAngleDegrees"] = normalReport
    do {  // float32 error alone: the kernel's normals at the GPU's step (1e-3 cell)
        let normals = try evaluator.normals(part: solid, points: s32, eps: Float(1e-3 * cell))
        var angles: [Double] = []
        for i in 0..<surfaceCount {
            let k = SIMD3<Double>(sameStepNormals[3 * i], sameStepNormals[3 * i + 1], sameStepNormals[3 * i + 2])
            let g = SIMD3<Double>(normals[i])
            guard simd_length(k) > 0.5, simd_length(g) > 0.5 else { continue }
            angles.append(acos(min(1, max(-1, simd_dot(k, g)))) * 180 / .pi)
        }
        entry["normalSameStep"] = summary(angles)
    }
    // Rays from outside toward the surface.
    var traced: [FieldProgramEvaluator.Ray] = []
    for i in 0..<rayCount {
        let r = Array(rays[(7 * i)..<(7 * i + 7)])
        traced.append(.init(origin: SIMD3(Float(r[0]), Float(r[1]), Float(r[2])), direction: SIMD3(Float(r[3]), Float(r[4]), Float(r[5])),
                            t0: 0, t1: Float(r[6])))
    }
    let tolerance = Float(1e-4 * cell)
    let hits = try evaluator.trace(part: solid, rays: traced, tolerance: tolerance, maxSteps: 256, eps: Float(1e-3 * cell))
    var hitAgree = 0, tErrors: [Double] = [], steps: [Double] = []
    for i in 0..<rayCount {
        let kernelHit = kernelHits[2 * i] > 0.5
        if hits[i].hit == kernelHit { hitAgree += 1 }
        if kernelHit && hits[i].hit { tErrors.append(abs(Double(hits[i].t) - kernelHits[2 * i + 1])) }
        steps.append(Double(hits[i].steps))
    }
    entry["rays"] = rayCount
    entry["rayHitAgreement"] = rayCount > 0 ? Double(hitAgree) / Double(rayCount) : 1
    entry["rayDistanceError"] = summary(tErrors)
    entry["raySteps"] = summary(steps)
    entry["tapes"] = part.tapes
    entry["maxTapeInstructions"] = part.maximumInstructions
    parts.append(entry)
}
report["parts"] = parts
report["instructionsPerSurfaceEvaluation"] = summary(allInstructions)

// Frame benchmark: per part, the frame's rays (8 x 4 pixel tiles) from a
// proxy-like start just before the surface, as a renderer traces them.
let width = reader.u32(), height = reader.u32()
var frameSeconds = 0.0, frameSamples = 0, framePixels = 0, frameHits = 0
var frameInstructions: [Double] = []
for solid in 0..<partCount {
    let count = reader.u32()
    let records = reader.doubles(7 * count)
    guard count > 0, program.parts[solid].hasField else { continue }
    let rays = (0..<count).map { i -> FieldProgramEvaluator.Ray in
        let r = Array(records[(7 * i)..<(7 * i + 7)])
        return .init(origin: SIMD3(Float(r[0]), Float(r[1]), Float(r[2])), direction: SIMD3(Float(r[3]), Float(r[4]), Float(r[5])),
                     t0: 0, t1: Float(r[6]))
    }
    let cell = program.parts[solid].brickSize / 8
    var time = 0.0
    let relaxation = Float(ProcessInfo.processInfo.environment["RAD_FIELD_RELAXATION"] ?? "1") ?? 1
    let frameTolerance = Float(ProcessInfo.processInfo.environment["RAD_FIELD_TOLERANCE"] ?? "0.01") ?? 0.01
    _ = try evaluator.trace(part: solid, rays: rays, tolerance: frameTolerance * cell, maxSteps: 64, eps: 1e-3 * cell,
                            relaxation: relaxation, seconds: &time)
    var best = Double.infinity
    var hits: [FieldProgramEvaluator.Hit] = []
    for _ in 0..<3 {
        hits = try evaluator.trace(part: solid, rays: rays, tolerance: frameTolerance * cell, maxSteps: 64, eps: 1e-3 * cell,
                                   relaxation: relaxation, seconds: &time)
        best = min(best, time)
    }
    frameSeconds += best
    framePixels += count
    for (i, hit) in hits.enumerated() {
        frameSamples += hit.steps + (hit.hit ? 4 : 0)
        if hit.hit { frameHits += 1 }
        if i % 7 == 0, let tape = program.tape(part: solid, at: hit.position) {
            frameInstructions.append(Double(program.tapeInstructions[tape] ?? 0))
        } else if i % 7 == 0 {
            frameInstructions.append(0)
        }
    }
}
let evaluationsPerSecond = Double(frameSamples) / max(frameSeconds, 1e-12)
report["frame"] = ["width": width, "height": height, "pixels": framePixels, "hits": frameHits,
                   "gpuMilliseconds": frameSeconds * 1000, "samplesPerPixel": Double(frameSamples) / Double(max(framePixels, 1)),
                   "evaluationsPerSecond": evaluationsPerSecond, "instructions": summary(frameInstructions),
                   "estimate2MpxMilliseconds": 2_070_000 * 12 / evaluationsPerSecond * 1000]
let json = try JSONSerialization.data(withJSONObject: report, options: [.prettyPrinted, .sortedKeys])
print(String(data: json, encoding: .utf8)!)
