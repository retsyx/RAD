import Foundation
import Metal

/// Batch GPU evaluation of a field program (distances, normals, ray hits)
/// with the kernels of `FieldProgramShaders.library`. Renderers call the
/// same device functions from their own shaders; this class serves
/// validation, measurement and tools. Self-contained like FieldProgram.
final class FieldProgramEvaluator {
    struct Ray {
        var origin: SIMD3<Float>
        var direction: SIMD3<Float>  // unit length
        var t0: Float
        var t1: Float
    }

    struct Hit: Equatable {
        var hit: Bool
        var startedInside: Bool
        var t: Float
        var position: SIMD3<Float>
        var normal: SIMD3<Float>
        var steps: Int
    }

    enum Failure: Error, LocalizedError {
        case unavailable(String)
        var errorDescription: String? {
            switch self {
            case .unavailable(let detail): "The GPU field evaluator is unavailable (\(detail))."
            }
        }
    }

    let device: any MTLDevice
    let program: FieldProgram
    private let queue: any MTLCommandQueue
    private let words: any MTLBuffer
    private let distancePipeline: any MTLComputePipelineState
    private let gradientPipeline: any MTLComputePipelineState
    private let tracePipeline: any MTLComputePipelineState

    /// Compiles the library (safe math) and uploads the program. `macros`
    /// override entries of `program.shaderMacros` (tests compare the
    /// evaluator's optional paths, such as RAD_FIELD_GRADIENT_PASS).
    init(program: FieldProgram, device: (any MTLDevice)? = MTLCreateSystemDefaultDevice(),
         source: String = FieldProgramShaders.library, macros: [String: NSNumber] = [:]) throws {
        guard let device else { throw Failure.unavailable("no Metal device") }
        self.device = device
        self.program = program
        let options = MTLCompileOptions()
        options.preprocessorMacros = program.shaderMacros.merging(macros) { $1 }
        if #available(macOS 15.0, *) {
            options.mathMode = .safe
        } else {
            options.fastMathEnabled = false
        }
        let library = try device.makeLibrary(source: source, options: options)
        func pipeline(_ name: String) throws -> any MTLComputePipelineState {
            guard let function = library.makeFunction(name: name) else { throw Failure.unavailable("missing \(name)") }
            return try device.makeComputePipelineState(function: function)
        }
        distancePipeline = try pipeline("rad_field_distance_kernel")
        gradientPipeline = try pipeline("rad_field_gradient_kernel")
        tracePipeline = try pipeline("rad_field_trace_kernel")
        guard let queue = device.makeCommandQueue(), let words = program.makeBuffer(device: device) else {
            throw Failure.unavailable("no command queue or buffer")
        }
        self.queue = queue
        self.words = words
    }

    private func packed(_ points: [SIMD3<Float>]) -> [Float] {
        var values = [Float]()
        values.reserveCapacity(3 * points.count)
        for p in points { values += [p.x, p.y, p.z] }
        return values
    }

    private func buffer<T>(_ values: [T]) throws -> any MTLBuffer {
        let length = max(MemoryLayout<T>.stride * values.count, 16)
        let made: (any MTLBuffer)? = values.isEmpty
            ? device.makeBuffer(length: length, options: .storageModeShared)
            : values.withUnsafeBytes { device.makeBuffer(bytes: $0.baseAddress!, length: $0.count, options: .storageModeShared) }
        guard let made else { throw Failure.unavailable("no buffer") }
        return made
    }

    /// Runs `pipeline` over `count` threads; returns the GPU seconds.
    @discardableResult
    private func dispatch(_ pipeline: any MTLComputePipelineState, count: Int,
                          encode: (any MTLComputeCommandEncoder) -> Void) throws -> Double {
        guard count > 0 else { return 0 }
        guard let commands = queue.makeCommandBuffer(), let encoder = commands.makeComputeCommandEncoder() else {
            throw Failure.unavailable("no command buffer")
        }
        encoder.setComputePipelineState(pipeline)
        encoder.setBuffer(words, offset: 0, index: 0)
        encode(encoder)
        let width = min(pipeline.maxTotalThreadsPerThreadgroup, 64)
        encoder.dispatchThreads(MTLSize(width: count, height: 1, depth: 1),
                                threadsPerThreadgroup: MTLSize(width: width, height: 1, depth: 1))
        encoder.endEncoding()
        commands.commit()
        commands.waitUntilCompleted()
        if let error = commands.error { throw error }
        return commands.gpuEndTime - commands.gpuStartTime
    }

    /// Field values (exact within the band, signed distance bounds elsewhere)
    /// of part `part` at points in the build's coordinates.
    func distances(part: Int, points: [SIMD3<Float>], seconds: UnsafeMutablePointer<Double>? = nil) throws -> [Float] {
        let input = try buffer(packed(points))
        let output = try buffer([Float](repeating: 0, count: points.count))
        var arguments = SIMD2<UInt32>(UInt32(part), UInt32(points.count))
        let time = try dispatch(distancePipeline, count: points.count) { encoder in
            encoder.setBuffer(input, offset: 0, index: 1)
            encoder.setBuffer(output, offset: 0, index: 2)
            encoder.setBytes(&arguments, length: MemoryLayout<SIMD2<UInt32>>.size, index: 3)
        }
        seconds?.pointee = time
        return Array(UnsafeBufferPointer(start: output.contents().assumingMemoryBound(to: Float.self), count: points.count))
    }

    /// Unit outward normals by tetrahedral differences with step `eps`.
    func normals(part: Int, points: [SIMD3<Float>], eps: Float, seconds: UnsafeMutablePointer<Double>? = nil) throws
        -> [SIMD3<Float>] {
        let input = try buffer(packed(points))
        let output = try buffer([Float](repeating: 0, count: 3 * points.count))
        var arguments = SIMD2<UInt32>(UInt32(part), UInt32(points.count))
        var step = eps
        let time = try dispatch(gradientPipeline, count: points.count) { encoder in
            encoder.setBuffer(input, offset: 0, index: 1)
            encoder.setBuffer(output, offset: 0, index: 2)
            encoder.setBytes(&arguments, length: MemoryLayout<SIMD2<UInt32>>.size, index: 3)
            encoder.setBytes(&step, length: 4, index: 4)
        }
        seconds?.pointee = time
        let values = UnsafeBufferPointer(start: output.contents().assumingMemoryBound(to: Float.self), count: 3 * points.count)
        return (0..<points.count).map { SIMD3(values[3 * $0], values[3 * $0 + 1], values[3 * $0 + 2]) }
    }

    /// Sphere-traced hits of unit rays over [t0, t1].
    func trace(part: Int, rays: [Ray], tolerance: Float, maxSteps: Int = 256, eps: Float, relaxation: Float = 1,
               seconds: UnsafeMutablePointer<Double>? = nil) throws -> [Hit] {
        var packedRays = [Float]()
        packedRays.reserveCapacity(8 * rays.count)
        for ray in rays {
            packedRays += [ray.origin.x, ray.origin.y, ray.origin.z, ray.t0, ray.direction.x, ray.direction.y, ray.direction.z, ray.t1]
        }
        let input = try buffer(packedRays)
        let output = try buffer([UInt32](repeating: 0, count: 8 * rays.count))
        struct Parameters {
            var part: UInt32
            var maxSteps: UInt32
            var tolerance: Float
            var eps: Float
            var relaxation: Float
        }
        var parameters = Parameters(part: UInt32(part), maxSteps: UInt32(maxSteps), tolerance: tolerance, eps: eps,
                                    relaxation: relaxation)
        var count = UInt32(rays.count)
        let time = try dispatch(tracePipeline, count: rays.count) { encoder in
            encoder.setBuffer(input, offset: 0, index: 1)
            encoder.setBuffer(output, offset: 0, index: 2)
            encoder.setBytes(&parameters, length: MemoryLayout<Parameters>.size, index: 3)
            encoder.setBytes(&count, length: 4, index: 4)
        }
        seconds?.pointee = time
        let raw = UnsafeBufferPointer(start: output.contents().assumingMemoryBound(to: UInt32.self), count: 8 * rays.count)
        func float(_ i: Int) -> Float { Float(bitPattern: raw[i]) }
        return (0..<rays.count).map { i in
            let b = 8 * i, flags = raw[b + 7]
            return Hit(hit: flags & 1 != 0, startedInside: flags & 2 != 0, t: float(b + 3),
                       position: SIMD3(float(b), float(b + 1), float(b + 2)),
                       normal: SIMD3(float(b + 4), float(b + 5), float(b + 6)), steps: Int(flags >> 8))
        }
    }
}
