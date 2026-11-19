import Foundation
import Metal
import simd
import XCTest
@testable import RAD

/// The GPU field program loader and Metal evaluator (SDF-DESIGN.md, "GPU
/// field programs"). The fixture is the kernel's program of
/// sphere(5, center=(1, 2, 3)) with 4 mm bricks and a 0.1 mm margin; the
/// engine-backed test compares a built design's program with the worker's
/// probe operation.
final class FieldProgramGPUTests: XCTestCase {
    static let sphereProgram: [UInt32] = [
        0x47504652, 0x00000002, 0x00000001, 0x0000007C, 0x00000010, 0x00000028, 0x00000000, 0x00000028, 0x00000000, 0x00000010,
        0x00000008, 0x00001000, 0x00000000, 0x00000000, 0x00000000, 0x00000000, 0x00000000, 0x00000028, 0x00000004, 0x00000004,
        0x00000004, 0x00000000, 0xC0E00000, 0xC0C00000, 0xC0A00000, 0x40800000, 0x3E800000, 0x3DCCCCCC, 0xC0800000, 0xC0400000,
        0xC0000000, 0x40C00000, 0x40E00000, 0x41000000, 0x3F800000, 0x00000001, 0x00000001, 0x00000001, 0x00000001, 0x00000040,
        0x9FF051E3, 0x9ED26C48, 0x9ED26C48, 0x9FF051E3, 0x9ED26C48, 0x0000001A, 0x0000001A, 0x9ED26C48, 0x9ED26C48, 0x0000001A,
        0x0000001A, 0x9ED26C48, 0x9FF051E3, 0x9ED26C48, 0x9ED26C48, 0x9FF051E3, 0x9ED26C48, 0x0000001A, 0x0000001A, 0x9ED26C48,
        0x0000001A, 0x0000001A, 0x0000001A, 0x0000001A, 0x0000001A, 0x0000001A, 0x0000001A, 0x0000001A, 0x9ED26C48, 0x0000001A,
        0x0000001A, 0x9ED26C48, 0x9ED26C48, 0x0000001A, 0x0000001A, 0x9ED26C48, 0x0000001A, 0x0000001A, 0x0000001A, 0x0000001A,
        0x0000001A, 0x0000001A, 0x0000001A, 0x0000001A, 0x9ED26C48, 0x0000001A, 0x0000001A, 0x9ED26C48, 0x9FF051E3, 0x9ED26C48,
        0x9ED26C48, 0x9FF051E3, 0x9ED26C48, 0x0000001A, 0x0000001A, 0x9ED26C48, 0x9ED26C48, 0x0000001A, 0x0000001A, 0x9ED26C48,
        0x9FF051E3, 0x9ED26C48, 0x9ED26C48, 0x9FF051E3, 0x3F800000, 0x00000000, 0x00000000, 0x00000000, 0x00000094, 0x00000000,
        0x00000074, 0xC0000078, 0x00000000, 0x00000000, 0x00000000, 0x00000000, 0x40A00000, 0x00000000, 0x00000000, 0x00000000,
        0x3F800000, 0x40000000, 0x40400000, 0x00000000,
    ]

    private func device() throws -> any MTLDevice {
        guard let device = MTLCreateSystemDefaultDevice() else { throw XCTSkip("No Metal device.") }
        return device
    }

    func testFixtureValidatesAndDescribesItsPart() throws {
        let program = try FieldProgram(words: Self.sphereProgram)
        XCTAssertEqual(program.parts.count, 1)
        let part = try XCTUnwrap(program.parts.first)
        XCTAssertTrue(part.hasField)
        XCTAssertEqual(part.dimensions, SIMD3<UInt32>(4, 4, 4))
        XCTAssertEqual(part.brickSize, 4)
        XCTAssertEqual(part.tapes, 1)
        XCTAssertEqual(program.tapeInstructions.values.max(), 1, "translate and sphere fuse into one instruction")
        XCTAssertEqual(program.shaderMacros["RAD_FIELD_VALUES"], 2)
        XCTAssertEqual(program.shaderMacros["RAD_FIELD_MESHES"], 0)
        XCTAssertEqual(program.shaderMacros["RAD_FIELD_OPS_A"], NSNumber(value: UInt32(1 << 20 | 1 << 12)),
                       "the interpreter compiles the sphere and its fused translation only")
        XCTAssertEqual(program.shaderMacros["RAD_FIELD_OPS_B"], 0)
        XCTAssertEqual(program.shaderMacros["RAD_FIELD_GRADIENT_PASS"], 1, "small register files take the four-wide gradient")
        XCTAssertNotNil(program.tape(part: 0, at: SIMD3(6, 2, 3)), "the surface lies in a band brick")
        XCTAssertNil(program.tape(part: 0, at: SIMD3(8.9, 9.9, 3)), "a corner brick is far from the surface")
        XCTAssertNil(program.tape(part: 0, at: SIMD3(100, 0, 0)), "outside the grid")
    }

    func testCorruptProgramsAreRejected() throws {
        let words = Self.sphereProgram
        let grid = Int(words[16 + 1]), cells = Int(words[16 + 23])
        let band = try XCTUnwrap((grid..<(grid + cells)).first { words[$0] & 0xC000_0000 == 0 })
        let far = try XCTUnwrap((grid..<(grid + cells)).first { words[$0] & 0x8000_0000 != 0 })
        let header = Int(words[band]) << 2, tape = header + 4
        var cases: [(String, [UInt32])] = []
        func corrupt(_ name: String, _ change: (inout [UInt32]) -> Void) {
            var copy = words
            change(&copy)
            cases.append((name, copy))
        }
        corrupt("magic") { $0[0] = 0 }
        corrupt("length") { $0[3] += 1 }
        corrupt("truncated") { $0.removeLast() }
        corrupt("tape header") { $0[header + 1] = 1 }
        corrupt("tape past the end") { $0[band] = UInt32($0.count / 4) }
        corrupt("register") { $0[tape] = ($0[tape] & ~0xF_0000) | 9 << 16 }
        corrupt("opcode") { $0[tape] = ($0[tape] & ~0x7F) | 99 }
        corrupt("record bits") { $0[tape] |= 1 << 30 }
        corrupt("post operation") { $0[tape] |= 9 << 12 }
        corrupt("parameters") { $0[tape + 2] = UInt32($0.count) }
        corrupt("unaligned parameters") { $0[tape + 2] += 1 }
        corrupt("fused map") { $0[tape + 3] = 11 << 28 | ($0[tape + 3] & 0x0FFF_FFFF) }
        corrupt("octant pointer backwards") { $0[far] = 0x4000_0000 | UInt32(grid >> 3) }
        corrupt("far bound") { $0[far] = 0xBFFF_FFFF }
        corrupt("lipschitz") { $0[header] = Float(0.5).bitPattern }
        corrupt("status") { $0[16] = 7 }
        for (name, copy) in cases {
            XCTAssertThrowsError(try FieldProgram(words: copy), name)
        }
    }

    func testSphereDistancesNormalsAndHits() throws {
        // Both gradient paths: the four-wide pass and four samples in the trace loop.
        for pass in [1, 0] {
            try checkSphere(macros: ["RAD_FIELD_GRADIENT_PASS": NSNumber(value: pass)])
        }
    }

    private func checkSphere(macros: [String: NSNumber]) throws {
        let program = try FieldProgram(words: Self.sphereProgram)
        let evaluator = try FieldProgramEvaluator(program: program, device: try device(), macros: macros)
        let centre = SIMD3<Float>(1, 2, 3)
        var generator = SystemRandomNumberGenerator()
        var points: [SIMD3<Float>] = []
        for _ in 0..<4000 {
            let direction = simd_normalize(SIMD3<Float>(Float.random(in: -1...1, using: &generator),
                                                        Float.random(in: -1...1, using: &generator),
                                                        Float.random(in: -1...1, using: &generator)) + 1e-3)
            points.append(centre + direction * Float.random(in: 0...12, using: &generator))
        }
        let values = try evaluator.distances(part: 0, points: points)
        XCTAssertTrue(points.contains { program.tape(part: 0, at: $0) == nil }, "some points lie outside the band")
        for (point, value) in zip(points, values) {
            let exact = simd_length(point - centre) - 5
            if program.tape(part: 0, at: point) != nil {
                XCTAssertEqual(value, exact, accuracy: 1e-5, "band values are the field")
            } else {
                XCTAssertEqual(value < 0, exact < 0, "far values keep the sign")
                XCTAssertLessThanOrEqual(abs(value), abs(exact) + 1e-5, "far values bound the distance from below")
            }
        }
        let surface = points.filter { program.tape(part: 0, at: $0) != nil }.map { centre + simd_normalize($0 - centre) * 5 }
        let normals = try evaluator.normals(part: 0, points: surface, eps: 1e-3)
        for (point, normal) in zip(surface, normals) {
            XCTAssertGreaterThan(simd_dot(normal, simd_normalize(point - centre)), 0.99999)
        }
        // Rays from far outside (crossing far bricks) and from inside.
        var rays: [FieldProgramEvaluator.Ray] = []
        for point in surface.prefix(500) {
            let direction = simd_normalize(centre - point)
            rays.append(.init(origin: point - 20 * direction, direction: direction, t0: 0, t1: 40))
        }
        rays.append(.init(origin: centre, direction: SIMD3(1, 0, 0), t0: 0, t1: 40))
        rays.append(.init(origin: centre + SIMD3(0, 0, 20), direction: SIMD3(1, 0, 0), t0: 0, t1: 40))
        let hits = try evaluator.trace(part: 0, rays: rays, tolerance: 1e-5, maxSteps: 256, eps: 1e-3)
        for (ray, hit) in zip(rays.prefix(500), hits.prefix(500)) {
            XCTAssertTrue(hit.hit)
            XCTAssertFalse(hit.startedInside)
            XCTAssertEqual(hit.t, 20, accuracy: 1e-3)
            XCTAssertEqual(simd_dot(hit.normal, -ray.direction), 1, accuracy: 1e-4)
        }
        XCTAssertTrue(hits[500].hit && hits[500].startedInside, "a ray from inside reports where it leaves")
        XCTAssertEqual(hits[500].t, 5, accuracy: 1e-3)
        XCTAssertFalse(hits[501].hit, "a ray passing the sphere misses")
        XCTAssertEqual(hits[501].t, 40)
    }

    func testShaderFunctionsCompileWithFastMath() throws {
        // Renderers prepend the functions to their own source, possibly with fast math.
        let options = MTLCompileOptions()
        options.preprocessorMacros = try FieldProgram(words: Self.sphereProgram).shaderMacros
        let source = FieldProgramShaders.functions + """

        fragment float4 field_fragment(float4 position [[position]], device const uint *program [[buffer(0)]]) {
            const float d = rad_field_distance(program, 0, position.xyz);
            const RADFieldHit hit = rad_field_trace(program, 0, position.xyz, float3(0, 0, 1), 0, 10, 1e-4, 64, 1e-3, 1);
            return float4(rad_field_gradient(program, 0, position.xyz, 1e-3) * d, hit.t);
        }
        """
        _ = try device().makeLibrary(source: source, options: options)
    }

    @MainActor
    func testBuiltProgramsMatchTheKernel() async throws {
        guard ProcessInfo.processInfo.environment["RAD_RUN_ENGINE_TESTS"] == "1" else {
            throw XCTSkip("Set RAD_RUN_ENGINE_TESTS=1 for the GPU field program comparison with the kernel.")
        }
        let script = """
        star = [((9 if i % 2 else 5) * math.cos(i * math.pi / 40), (9 if i % 2 else 5) * math.sin(i * math.pi / 40)) for i in range(80)]
        body = union(box((20, 14, 8), radius=1), sphere(6, center=(10, 0, 4)), smooth=2) - cylinder(2, 30, axis=(1, 0, 0))
        plate = polygon(star).extrude(3).translate(0, 30, 0)
        lattice = implicit(sin(x) * cos(y) + sin(y) * cos(z) + sin(z) * cos(x) + 0.2, bounds=((-5, -5, -5), (5, 5, 5)))
        result = [body, plate, intersection(sphere(5), lattice).translate(30, 0, 0)]
        """
        let service = EngineService()
        let preview = try await service.buildPreview(ModelSpec(title: "Field program", summary: "GPU field comparison.",
                                                               units: "mm", script: script),
                                                     python: EngineService.defaultPython, threads: 2)
        let program = try XCTUnwrap(preview.fieldProgram, "reading a build loads its field program")
        XCTAssertTrue(program.parts.allSatisfy(\.hasField))
        XCTAssertTrue(program.readsPolygons)
        XCTAssertEqual(program.shaderMacros["RAD_FIELD_GRADIENT_PASS"], 0, "polygon programs take four samples")
        let evaluator = try FieldProgramEvaluator(program: program, device: try device())
        let fourWideEvaluator = try FieldProgramEvaluator(program: program, device: try device(),
                                                          macros: ["RAD_FIELD_GRADIENT_PASS": 1])
        for solid in 0..<preview.solids {
            // Mesh vertices of the part, nudged off the surface.
            var points: [SIMD3<Float>] = []
            for (triangle, owner) in zip(stride(from: 0, to: preview.triangleIndices.count, by: 3), preview.triangleSolids ?? [])
            where owner == solid && points.count < 3000 {
                let v = Int(preview.triangleIndices[triangle])
                let p = SIMD3<Float>(preview.positions[3 * v], preview.positions[3 * v + 1], preview.positions[3 * v + 2])
                points.append(p + SIMD3<Float>(repeating: Float(points.count % 7) * 0.02 - 0.06))
            }
            let probe = try await service.probe(preview, solids: [solid], points: points.map { [Double($0.x), Double($0.y), Double($0.z)] },
                                                python: EngineService.defaultPython, threads: 2)
            let gpu = try evaluator.distances(part: solid, points: points)
            var compared = 0
            for (index, point) in points.enumerated() where program.tape(part: solid, at: point) != nil {
                let kernel = probe.samples[index][0].distance
                XCTAssertEqual(Double(gpu[index]), kernel, accuracy: 2e-5, "part \(solid) at \(point)")
                compared += 1
            }
            XCTAssertGreaterThan(compared, points.count / 2, "most surface points lie in band bricks")
            // Normals by four samples (this program's default: it reads
            // polygons) and by the four-wide pass, which runs the tape once
            // for the stencil and matches the samples up to float rounding
            // (operations on four values may round differently).
            let eps: Float = 1e-3
            let stencil: [SIMD3<Float>] = [SIMD3(1, -1, -1), SIMD3(-1, -1, 1), SIMD3(-1, 1, -1), SIMD3(1, 1, 1)]
            // (Where every stencil point lies in a band cell: a far cell's bound is not the field.)
            let band = points.filter { p in stencil.allSatisfy { program.tape(part: solid, at: p + eps * $0) != nil } }
            let samples = try evaluator.distances(part: solid, points: band.flatMap { p in stencil.map { p + eps * $0 } })
            let normals = try evaluator.normals(part: solid, points: band, eps: eps)
            let fourWide = try fourWideEvaluator.normals(part: solid, points: band, eps: eps)
            var identical = 0
            for index in band.indices {
                var sum = SIMD3<Float>.zero
                for (k, direction) in stencil.enumerated() { sum += direction * samples[4 * index + k] }
                guard simd_length(sum) > 0 else { continue }
                XCTAssertLessThan(simd_length(normals[index] - simd_normalize(sum)), 1e-5, "part \(solid) at \(band[index])")
                let difference = simd_length(fourWide[index] - simd_normalize(sum))
                XCTAssertLessThan(difference, 1e-3, "part \(solid) at \(band[index])")
                if difference < 1e-5 { identical += 1 }
            }
            XCTAssertGreaterThan(identical, band.count / 2, "part \(solid)")
            // Traces toward the surface hit the same points with either
            // gradient path (the march is the same code, though compiled
            // separately) and agree on normals.
            let rays = points.prefix(500).map { point -> FieldProgramEvaluator.Ray in
                let direction = simd_normalize(SIMD3<Float>(0.3, -0.5, 0.8))
                return .init(origin: point - 2 * direction, direction: direction, t0: 0, t1: 4)
            }
            let hits = try evaluator.trace(part: solid, rays: rays, tolerance: 1e-4, eps: eps)
            let fourWideHits = try fourWideEvaluator.trace(part: solid, rays: rays, tolerance: 1e-4, eps: eps)
            XCTAssertGreaterThan(hits.filter(\.hit).count, rays.count / 2, "part \(solid)")
            for (hit, other) in zip(hits, fourWideHits) {
                XCTAssertEqual(hit.hit, other.hit)
                XCTAssertEqual(hit.t, other.t, accuracy: 1e-3)
                if hit.hit { XCTAssertLessThan(simd_length(hit.normal - other.normal), 1e-3, "part \(solid)") }
            }
        }
    }
}
