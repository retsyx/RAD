import AppKit
import Metal
import QuartzCore
import SceneKit
import simd
import XCTest
@testable import RAD

/// Measurement tool (RAD_SLX_PROBE): per-block GPU cost maps of a design's
/// exact geometry, and how long light frames on another queue wait beside
/// it, unsliced and sliced. Skipped in ordinary test runs.
@MainActor
final class ExactSliceProbeTests: XCTestCase {
    struct Config: Decodable {
        var design: String
        var output: String
        var width: Int?
        var height: Int?
        var view: String?
        var display: String?
        var experiments: [String]
        /// Shader diagnostics for the timed passes (4: no multisample tracing).
        var debug: Int?
    }

    func testSliceProbe() async throws {
        guard let path = ProcessInfo.processInfo.environment["RAD_SLX_PROBE"] else {
            throw XCTSkip("Set RAD_SLX_PROBE to a probe JSON file.")
        }
        let config = try JSONDecoder().decode(Config.self, from: Data(contentsOf: URL(fileURLWithPath: path)))
        let output = URL(fileURLWithPath: config.output)
        try FileManager.default.createDirectory(at: output, withIntermediateDirectories: true)
        let spec = try JSONDecoder().decode(ModelSpec.self, from: Data(contentsOf: URL(fileURLWithPath: config.design)))
        let started = Date()
        let preview = try await EngineService().buildPreview(spec, python: EngineService.defaultPython,
                                                             threads: ProcessInfo.processInfo.activeProcessorCount)
        var log = ["build \(String(format: "%.1f", Date().timeIntervalSince(started))) s, proxy triangles \(preview.proxySolids?.count ?? 0)"]
        let width = config.width ?? 2000, height = config.height ?? 1640
        let surface = CADSceneView(frame: NSRect(x: 0, y: 0, width: Double(width) / 2, height: Double(height) / 2))
        surface.preparesRendererResources = false
        surface.update(mesh: preview, mode: ViewportDisplayMode(rawValue: config.display ?? "shaded") ?? .shaded,
                       grid: false, command: nil, selectedSolids: [], hiddenSolids: [], projection: .perspective,
                       solidTransforms: preview.assemblyPose?.solidTransforms ?? [:])
        _ = try await surface.waitForCaptureResources(until: .now + .seconds(120))
        try surface.configureAgentView(named: config.view ?? "isometric", camera: CADAgentCamera())
        surface.layout()
        _ = try await surface.waitForCaptureResources(until: .now + .seconds(120))
        surface.fieldDebugMode = config.debug ?? 0
        let capture = try CADViewportCapture(view: surface, width: width, height: height)
        surface.fieldDebugMode = 0
        log.append("exact solids \(surface.exactSolids.count), field \(capture.frame.field != nil), \(width)x\(height)")
        let probe = try Probe(capture: capture)
        for experiment in config.experiments where experiment == "steps" {
            // March samples per pixel (debug 5: samples / 1024 in the stored normal, shown as gray).
            surface.fieldDebugMode = 5
            let stepsCapture = try CADViewportCapture(view: surface, width: width, height: height)
            surface.fieldDebugMode = 0
            let image = try stepsCapture.image()
            try CADAgentRenderer.encodePNG(image).write(to: output.appendingPathComponent("steps.png"))
            print("SLX steps: written")
        }
        for experiment in config.experiments where experiment != "steps" {
            let line = try probe.run(experiment, output: output)
            print("SLX \(experiment): \(line)")
            log.append("\(experiment): \(line)")
            try log.joined(separator: "\n").write(to: output.appendingPathComponent("log.txt"), atomically: true, encoding: .utf8)
        }
    }
}

/// Offscreen exact passes of one capture with GPU stage timestamps.
final class Probe: @unchecked Sendable {
    let capture: CADViewportCapture
    let device: any MTLDevice
    let queue: any MTLCommandQueue
    let lightQueue: any MTLCommandQueue
    let renderer: SCNRenderer
    let passes: CADViewportPasses
    let target: any MTLTexture
    let counters: any MTLCounterSampleBuffer
    let counterCapacity = 4_096
    let lightPipeline: any MTLRenderPipelineState
    let lightDepth: any MTLDepthStencilState
    let lightColor: any MTLTexture
    let lightDepthTexture: any MTLTexture
    let lightResolve: any MTLTexture
    /// Ticks per second of GPU timestamps.
    let tickRate: Double

    init(capture: CADViewportCapture) throws {
        self.capture = capture
        device = try XCTUnwrap(MTLCreateSystemDefaultDevice())
        queue = try XCTUnwrap(device.makeCommandQueue())
        lightQueue = try XCTUnwrap(device.makeCommandQueue())
        renderer = SCNRenderer(device: device, options: nil)
        renderer.scene = capture.source.scene
        renderer.pointOfView = capture.source.camera
        renderer.usesReverseZ = true
        renderer.autoenablesDefaultLighting = false
        XCTAssertTrue(renderer.prepare(capture.source.scene, shouldAbortBlock: nil))
        passes = try CADViewportPasses(device: device)
        let descriptor = MTLTextureDescriptor.texture2DDescriptor(pixelFormat: CADViewportPasses.colorFormat,
                                                                  width: capture.width, height: capture.height, mipmapped: false)
        descriptor.usage = [.renderTarget, .shaderRead]
        descriptor.storageMode = .private
        target = try XCTUnwrap(device.makeTexture(descriptor: descriptor))
        let set = try XCTUnwrap(device.counterSets?.first { $0.name == MTLCommonCounterSet.timestamp.rawValue })
        let counterDescriptor = MTLCounterSampleBufferDescriptor()
        counterDescriptor.counterSet = set
        counterDescriptor.sampleCount = counterCapacity
        counterDescriptor.storageMode = .shared
        counters = try device.makeCounterSampleBuffer(descriptor: counterDescriptor)
        let (cpu0, gpu0) = device.sampleTimestamps()
        usleep(200_000)
        let (cpu1, gpu1) = device.sampleTimestamps()
        // Both timestamps count nanoseconds on Apple GPUs; calibrate anyway.
        tickRate = Double(gpu1 - gpu0) / (Double(cpu1 - cpu0) / 1e9)
        // Light frames: full-screen 4x MSAA memoryless pass resolved, a trivial fragment.
        let source = """
        #include <metal_stdlib>
        using namespace metal;
        struct O { float4 p [[position]]; };
        vertex O light_v(uint vid [[vertex_id]]) { O o; o.p = float4(vid == 1 ? 3.0f : -1.0f, vid == 2 ? 3.0f : -1.0f, 0.5f, 1.0f); return o; }
        fragment float4 light_f(O in [[stage_in]]) { return float4(fract(in.p.x * 0.01f), 0.5f, 0.5f, 1.0f); }
        """
        let library = try device.makeLibrary(source: source, options: nil)
        let pipeline = MTLRenderPipelineDescriptor()
        pipeline.vertexFunction = library.makeFunction(name: "light_v")
        pipeline.fragmentFunction = library.makeFunction(name: "light_f")
        pipeline.rasterSampleCount = 4
        pipeline.colorAttachments[0].pixelFormat = .bgra8Unorm_srgb
        pipeline.depthAttachmentPixelFormat = .depth32Float
        lightPipeline = try device.makeRenderPipelineState(descriptor: pipeline)
        let depth = MTLDepthStencilDescriptor()
        depth.depthCompareFunction = .greater
        depth.isDepthWriteEnabled = true
        lightDepth = try XCTUnwrap(device.makeDepthStencilState(descriptor: depth))
        let gpu = device, w = capture.width, h = capture.height
        func make(_ format: MTLPixelFormat, multisample: Bool, memoryless: Bool) throws -> any MTLTexture {
            let d = MTLTextureDescriptor.texture2DDescriptor(pixelFormat: format, width: w, height: h,
                                                             mipmapped: false)
            if multisample { d.textureType = .type2DMultisample; d.sampleCount = 4 }
            d.usage = .renderTarget
            d.storageMode = memoryless ? .memoryless : .private
            return try XCTUnwrap(gpu.makeTexture(descriptor: d))
        }
        lightColor = try make(.bgra8Unorm_srgb, multisample: true, memoryless: true)
        lightDepthTexture = try make(.depth32Float, multisample: true, memoryless: true)
        lightResolve = try make(.bgra8Unorm_srgb, multisample: false, memoryless: false)
    }

    var width: Int { capture.width }
    var height: Int { capture.height }

    // MARK: Stage timestamps

    private var nextSample = 0
    /// Pass indices (into `counters`, 4 samples each) per experiment.
    private func installCounters() {
        nextSample = 0
        let buffer = counters, capacity = counterCapacity
        let tile = Int(ProcessInfo.processInfo.environment["RAD_SLX_TILE"] ?? "0") ?? 0
        CADViewportPasses.experimentPassHook = { [unowned self] pass in
            if tile > 0 { pass.tileWidth = tile; pass.tileHeight = tile }
            guard nextSample + 4 <= capacity else { return }
            let attachment = pass.sampleBufferAttachments[0]!
            attachment.sampleBuffer = buffer
            attachment.startOfVertexSampleIndex = nextSample
            attachment.endOfVertexSampleIndex = nextSample + 1
            attachment.startOfFragmentSampleIndex = nextSample + 2
            attachment.endOfFragmentSampleIndex = nextSample + 3
            nextSample += 4
        }
    }

    private func removeCounters() {
        let tile = Int(ProcessInfo.processInfo.environment["RAD_SLX_TILE"] ?? "0") ?? 0
        CADViewportPasses.experimentPassHook = tile > 0 ? { pass in pass.tileWidth = tile; pass.tileHeight = tile } : nil
    }

    /// (vertex start, vertex end, fragment start, fragment end) in seconds for each recorded pass.
    private func stages() -> [(Double, Double, Double, Double)] {
        guard nextSample > 0, let data = try? counters.resolveCounterRange(0..<nextSample) else { return [] }
        let values = data.withUnsafeBytes { Array($0.bindMemory(to: MTLCounterResultTimestamp.self)) }
        return stride(from: 0, to: values.count - 3, by: 4).map { i in
            (Double(values[i].timestamp) / tickRate, Double(values[i + 1].timestamp) / tickRate,
             Double(values[i + 2].timestamp) / tickRate, Double(values[i + 3].timestamp) / tickRate)
        }
    }

    // MARK: Experiments

    func run(_ experiment: String, output: URL) throws -> String {
        let parts = experiment.split(separator: ":").map(String.init)
        switch parts[0] {
        case "once": return try once(reps: parts.count > 1 ? Int(parts[1])! : 5)
        case "tiles": return try tiles(block: parts.count > 1 ? Int(parts[1])! : 32, output: output)
        case "heavy-parts": return try heavyParts(count: parts.count > 1 ? Int(parts[1])! : 6)
        case "scheme": return try scheme(Array(parts.dropFirst()))
        case "wait-once": return try waitOnce()
        case "drain-once":
            // Sparse light frames: each waits one drain of the whole pass's in-flight tiles.
            lightInterval = 0.045
            defer { lightInterval = 0.00833 }
            return try waitOnce(runs: 6)
        case "wait-sliced": return try waitSliced(cold: parts.count > 1 && parts[1] == "cold")
        case "sliced": return try sliced(cold: parts.count > 1 && parts[1] == "cold")
        default: return "unknown"
        }
    }

    private func newJob(_ commandBuffer: any MTLCommandBuffer) throws -> CADExactJob {
        try XCTUnwrap(passes.makeJob(capture.frame, source: capture.source, renderer: renderer, commandBuffer: commandBuffer))
    }

    /// The whole geometry in one pass (one command buffer with the probe pass): GPU and stage times.
    func once(reps: Int) throws -> String {
        try warmUp()
        var times: [Double] = [], fragment: [Double] = [], vertex: [Double] = []
        for index in 0..<(reps + 2) {
            installCounters()
            let buffer = try XCTUnwrap(queue.makeCommandBuffer())
            let job = try newJob(buffer)
            passes.encodeAll(job, frame: capture.frame, commandBuffer: buffer)
            buffer.commit()
            buffer.waitUntilCompleted()
            let s = stages()
            if index >= 2 {
                times.append((buffer.gpuEndTime - buffer.gpuStartTime) * 1000)
                if let first = s.first {
                    vertex.append((first.1 - first.0) * 1000)
                    fragment.append((first.3 - first.2) * 1000)
                }
            }
        }
        removeCounters()
        times.sort(); fragment.sort(); vertex.sort()
        return String(format: "GPU min %.2f median %.2f ms; geometry pass vertex %.2f fragment %.2f ms (min)",
                      times.first!, times[times.count / 2], vertex.first ?? -1, fragment.first ?? -1)
    }

    /// Keeps the GPU busy with whole frames for about a second (clocks ramp up).
    func warmUp() throws {
        let start = CACurrentMediaTime()
        while CACurrentMediaTime() - start < 1.0 {
            let buffer = try XCTUnwrap(queue.makeCommandBuffer())
            let job = try newJob(buffer)
            passes.encodeAll(job, frame: capture.frame, commandBuffer: buffer)
            buffer.commit()
            buffer.waitUntilCompleted()
        }
    }

    /// GPU milliseconds of each rectangle drawn alone by `job`: one pass per command buffer, committed in
    /// batches without waiting, so the GPU stays busy at steady clocks; each one's time runs from the previous
    /// one's end (they execute in order) to its own.
    private func timeAlone(_ job: CADExactJob, _ rects: [SIMD4<Int32>], block: Int, clearing: Bool = false) throws -> [Double] {
        var results: [Double] = []
        var index = 0
        var previousEnd = 0.0
        while index < rects.count {
            let batch = Array(rects[index..<min(rects.count, index + 400)])
            var buffers: [any MTLCommandBuffer] = []
            var clears: [any MTLCommandBuffer] = []
            for rect in batch {
                if clearing {
                    // Untimed: cleared layers, so nothing earlier rejects this pass's fragments.
                    let clear = try XCTUnwrap(queue.makeCommandBuffer())
                    job.cleared = false
                    passes.encode(job, slice: CADSlicePlan.Slice(rects: [], covered: 0), commandBuffer: clear)
                    clear.commit()
                    clears.append(clear)
                }
                let buffer = try XCTUnwrap(queue.makeCommandBuffer())
                // Scissored to the rectangle (a one-block plan would count as the whole frame).
                passes.encode(job, slice: CADSlicePlan.Slice(rects: [rect], covered: 1, whole: false, blocks: 1),
                              commandBuffer: buffer)
                buffer.commit()
                buffers.append(buffer)
            }
            buffers.last?.waitUntilCompleted()
            for (i, buffer) in buffers.enumerated() {
                buffer.waitUntilCompleted()
                if clearing { previousEnd = max(previousEnd, clears[i].gpuEndTime) }
                results.append((buffer.gpuEndTime - max(buffer.gpuStartTime, previousEnd)) * 1000)
                previousEnd = buffer.gpuEndTime
            }
            index += batch.count
        }
        return results
    }

    private func clearedJob() throws -> CADExactJob {
        let setup = try XCTUnwrap(queue.makeCommandBuffer())
        let job = try newJob(setup)
        job.plan = CADSlicePlan(width: width, height: height, rects: [], block: 32)
        passes.encodeSlice(job, budget: 1e-12, commandBuffer: setup)
        setup.commit()
        setup.waitUntilCompleted()
        return job
    }

    /// Reached blocks of a block size, row order.
    private func reachedBlocks(_ job: CADExactJob, block: Int) -> [SIMD4<Int32>] {
        let columns = (width + block - 1) / block, rows = (height + block - 1) / block
        let rects = job.draws.map { $0.rect ?? SIMD4(0, 0, Int32(width), Int32(height)) }
        var result: [SIMD4<Int32>] = []
        for by in 0..<rows {
            for bx in 0..<columns {
                let r = SIMD4(Int32(bx * block), Int32(by * block), Int32(min(width, (bx + 1) * block)), Int32(min(height, (by + 1) * block)))
                if rects.contains(where: { $0.x < r.z && $0.z > r.x && $0.y < r.w && $0.w > r.y }) { result.append(r) }
            }
        }
        return result
    }

    /// Fragment-stage time of every reached block alone (one pass each).
    func tiles(block: Int, output: URL) throws -> String {
        try warmUp()
        let job = try clearedJob()
        let blocks = reachedBlocks(job, block: block)
        let times = try timeAlone(job, blocks, block: block)
        let fragments = times.sorted()
        let total = fragments.reduce(0, +)
        func q(_ f: Double) -> Double { fragments.isEmpty ? 0 : fragments[min(fragments.count - 1, Int(Double(fragments.count - 1) * f))] }
        var csv = "x,y,fragment_ms\n"
        for (r, t) in zip(blocks, times) { csv += String(format: "%d,%d,%.4f\n", Int(r.x) / block, Int(r.y) / block, t) }
        try csv.write(to: output.appendingPathComponent("tiles\(block).csv"), atomically: true, encoding: .utf8)
        return String(format: "%d blocks of %d px: fragment sum %.1f ms (/20 cores: %.1f); per block p50 %.3f p90 %.3f p99 %.3f max %.3f ms",
                      blocks.count, block, total, total / 20, q(0.5), q(0.9), q(0.99), q(1))
    }

    /// The heaviest 32-px blocks alone, with all draws and with each draw alone.
    func heavyParts(count: Int) throws -> String {
        let block = 32
        try warmUp()
        let job = try clearedJob()
        let blocks = reachedBlocks(job, block: block)
        let times = try timeAlone(job, blocks, block: block)
        let heavy = zip(blocks, times).sorted { $0.1 > $1.1 }.prefix(count)
        var lines: [String] = []
        for (rect, total) in heavy {
            let reaching = job.draws.enumerated().filter { $0.element.rect.map({ $0.x < rect.z && $0.z > rect.x && $0.y < rect.w && $0.w > rect.y }) ?? true }
            var perDraw: [(Int, Double)] = []
            for (index, draw) in reaching {
                let single = CADExactJob(key: job.key, field: job.field, view: job.view, projection: job.projection,
                                         width: width, height: height, targets: job.targets, draws: [draw])
                let t = try timeAlone(single, [rect, rect, rect], block: block, clearing: true).min() ?? -1
                perDraw.append((index, t))
            }
            let again = try timeAlone(job, [rect, rect, rect], block: block, clearing: true).min() ?? -1
            perDraw.sort { $0.1 > $1.1 }
            let top = perDraw.prefix(4).map { String(format: "solid %d (order %d) %.1f", job.draws[$0.0].solid, $0.0, $0.1) }.joined(separator: ", ")
            lines.append(String(format: "block (%d,%d) all %.1f ms (again %.1f); %d draws reach it, sum alone %.1f; top: %@", rect.x, rect.y, total, again,
                                perDraw.count, perDraw.reduce(0) { $0 + $1.1 }, top))
        }
        return "\n  " + lines.joined(separator: "\n  ")
    }

    /// Slices of a scheme: "whole"; "blocks:B" (every reached B-block its own slice, row order);
    /// "scatter:K:B" (reached B-blocks dealt to K slices by an interleaving hash, so every slice spans the screen).
    private func slices(_ spec: [String], job: CADExactJob) -> [CADSlicePlan.Slice] {
        switch spec.first ?? "whole" {
        case "blocks":
            let block = Int(spec[1])!
            return reachedBlocks(job, block: block).map { CADSlicePlan.Slice(rects: [$0], covered: 1, blocks: 1) }
        case "scatter":
            let k = Int(spec[1])!, block = Int(spec[2])!
            var groups = [[SIMD4<Int32>]](repeating: [], count: k)
            for r in reachedBlocks(job, block: block) {
                let bx = Int(r.x) / block, by = Int(r.y) / block
                // A 2-D interleave: consecutive blocks of a row and of a column fall in different slices.
                let h = (bx &* 7 &+ by &* 3 &+ ((bx / k) ^ (by / k)) &* 5) % k
                groups[h].append(r)
            }
            return groups.filter { !$0.isEmpty }.map { CADSlicePlan.Slice(rects: $0, covered: 1, blocks: $0.count) }
        case "bands":
            // Full-width bands of B rows, band j in slice j mod K (only the reached x-range).
            let k = Int(spec[1])!, band = Int(spec[2])!
            let reached = reachedBlocks(job, block: band)
            var groups = [[SIMD4<Int32>]](repeating: [], count: k)
            var rows: [Int: (Int32, Int32)] = [:]
            for r in reached { let row = Int(r.y) / band; rows[row] = (min(rows[row]?.0 ?? r.x, r.x), max(rows[row]?.1 ?? r.z, r.z)) }
            for (row, span) in rows.sorted(by: { $0.key < $1.key }) {
                groups[row % k].append(SIMD4(span.0, Int32(row * band), span.1, Int32(min(height, (row + 1) * band))))
            }
            return groups.filter { !$0.isEmpty }.map { CADSlicePlan.Slice(rects: $0, covered: 1, blocks: $0.count) }
        default:
            return [CADSlicePlan.Slice(rects: [], covered: 1, whole: true, blocks: 1)]
        }
    }

    /// One run of a scheme: slices committed with at most two in flight; optional light frames beside them.
    private var secondTargets: CADExactTargets?

    private func runScheme(_ spec: [String], light: String?) throws -> (busy: Double, slices: [Double], waits: [Double]) {
        let setup = try XCTUnwrap(queue.makeCommandBuffer())
        let job = try newJob(setup)
        setup.commit()
        // Options after the scheme: "alt" alternates two layer sets (a second full-size set), "cb:N" puts N
        // slices in each command buffer.
        let alternate = spec.contains("alt")
        let perBuffer = spec.first(where: { $0.hasPrefix("cb") }).map { Int($0.dropFirst(2))! } ?? 1
        let base = spec.filter { $0 != "alt" && !$0.hasPrefix("cb") }
        if alternate, secondTargets == nil {
            secondTargets = CADExactTargets(device: device, width: width, height: height, translucent: false, revealage: false)
        }
        let jobs = alternate ? [job, CADExactJob(key: job.key, field: job.field, view: job.view, projection: job.projection,
                                                 width: width, height: height, targets: secondTargets!, draws: job.draws)] : [job]
        let list = slices(base, job: job)
        var committed: [any MTLCommandBuffer] = []
        var done = false
        let lock = NSLock()
        var waits: [Double] = []
        let group = DispatchGroup()
        if let light {
            group.enter()
            Thread.detachNewThread { [self] in
                defer { group.leave() }
                var frames: [(any MTLCommandBuffer, Double)] = []
                let interval = light == "sparse" ? 0.045 : 0.00833
                let start = CACurrentMediaTime()
                while !lock.withLock({ done }) {
                    // The app keeps at most two frames in flight.
                    if light == "app", frames.count >= 2 { frames[frames.count - 2].0.waitUntilCompleted() }
                    guard let buffer = try? lightFrame() else { break }
                    let t = CACurrentMediaTime()
                    buffer.commit()
                    frames.append((buffer, t))
                    let next = start + Double(frames.count) * interval - CACurrentMediaTime()
                    if next > 0 { usleep(UInt32(next * 1e6)) }
                }
                for (buffer, t) in frames {
                    buffer.waitUntilCompleted()
                    waits.append((buffer.gpuEndTime - t) * 1000)
                }
            }
        }
        var index = 0
        while index < list.count {
            if committed.count >= 2 { committed[committed.count - 2].waitUntilCompleted() }
            let buffer = try XCTUnwrap(queue.makeCommandBuffer())
            for slice in list[index..<min(list.count, index + perBuffer)] {
                passes.encode(jobs[index % jobs.count], slice: slice, commandBuffer: buffer)
                index += 1
            }
            buffer.commit()
            committed.append(buffer)
        }
        committed.last?.waitUntilCompleted()
        lock.withLock { done = true }
        group.wait()
        var busy = 0.0, cursor = 0.0, durations: [Double] = []
        for b in committed.sorted(by: { $0.gpuStartTime < $1.gpuStartTime }) {
            if b.gpuEndTime > cursor { busy += b.gpuEndTime - max(b.gpuStartTime, cursor) }
            durations.append((b.gpuEndTime - max(b.gpuStartTime, cursor)) * 1000)
            cursor = max(cursor, b.gpuEndTime)
        }
        return (busy * 1000, durations, waits)
    }

    func scheme(_ spec: [String]) throws -> String {
        try warmUp()
        var best: (busy: Double, slices: [Double], waits: [Double])?
        for _ in 0..<2 {
            let r = try runScheme(spec, light: nil)
            if best == nil || r.busy < best!.busy { best = r }
        }
        let s = best!.slices.sorted()
        var line = String(format: "busy %.1f ms, %d slices (p50 %.1f max %.1f)", best!.busy, s.count, s[s.count / 2], s.last!)
        var sparse: [Double] = [], app: [Double] = []
        for _ in 0..<3 { sparse += try runScheme(spec, light: "sparse").waits }
        for _ in 0..<2 { app += try runScheme(spec, light: "app").waits }
        line += "; sparse " + Self.summary(sparse) + "; app " + Self.summary(app)
        return line
    }

    // MARK: Light frames

    private func lightFrame() throws -> any MTLCommandBuffer {
        let buffer = try XCTUnwrap(lightQueue.makeCommandBuffer())
        let pass = MTLRenderPassDescriptor()
        pass.colorAttachments[0].texture = lightColor
        pass.colorAttachments[0].resolveTexture = lightResolve
        pass.colorAttachments[0].loadAction = .clear
        pass.colorAttachments[0].storeAction = .multisampleResolve
        pass.depthAttachment.texture = lightDepthTexture
        pass.depthAttachment.loadAction = .clear
        pass.depthAttachment.clearDepth = 0
        pass.depthAttachment.storeAction = .dontCare
        let encoder = try XCTUnwrap(buffer.makeRenderCommandEncoder(descriptor: pass))
        encoder.setRenderPipelineState(lightPipeline)
        encoder.setDepthStencilState(lightDepth)
        encoder.drawPrimitives(type: .triangle, vertexStart: 0, vertexCount: 3)
        encoder.endEncoding()
        return buffer
    }

    /// Commits a light frame every 8.33 ms while `running` holds; returns commit -> GPU end per frame
    /// that ended while the exact work ran.
    /// Seconds between light frames (sparse frames measure single drains).
    var lightInterval = 0.00833

    private func lightFrames(until done: () -> Bool, end: () -> Double) throws -> [Double] {
        var frames: [(any MTLCommandBuffer, Double)] = []
        let start = CACurrentMediaTime()
        while !done(), frames.count < 2_000 {
            let buffer = try lightFrame()
            let t = CACurrentMediaTime()
            buffer.commit()
            frames.append((buffer, t))
            let wait = start + Double(frames.count) * lightInterval - CACurrentMediaTime()
            if wait > 0 { usleep(UInt32(wait * 1e6)) }
        }
        let finish = end()
        return frames.compactMap { buffer, t in
            buffer.waitUntilCompleted()
            return buffer.gpuEndTime <= finish ? (buffer.gpuEndTime - t) * 1000 : nil
        }
    }

    private static func summary(_ waits: [Double]) -> String {
        let w = waits.sorted()
        func q(_ f: Double) -> Double { w.isEmpty ? .nan : w[min(w.count - 1, Int(Double(w.count - 1) * f))] }
        return String(format: "light frames n=%d commit->done p50 %.1f p90 %.1f p99 %.1f max %.1f ms", w.count, q(0.5), q(0.9), q(0.99), q(1))
    }

    func waitOnce(runs: Int = 3) throws -> String {
        var waits: [Double] = []
        var busy: [Double] = []
        for _ in 0..<runs {
            let buffer = try XCTUnwrap(queue.makeCommandBuffer())
            let job = try newJob(buffer)
            passes.encodeAll(job, frame: capture.frame, commandBuffer: buffer)
            buffer.commit()
            usleep(1000)
            waits += try lightFrames(until: { buffer.status == .completed || buffer.status == .error },
                                     end: { buffer.waitUntilCompleted(); return buffer.gpuEndTime })
            busy.append((buffer.gpuEndTime - buffer.gpuStartTime) * 1000)
        }
        return String(format: "exact pass %.1f ms beside frames; ", busy.min()!) + Self.summary(waits)
    }

    /// The live loop's slicing (e3d5f4f rules: budget, block halving, slices in flight, chained commits
    /// emulated by committing the next slice as soon as fewer than the limit run), measuring each slice.
    private func runSliced(cold: Bool, beside: Bool) throws -> (busy: Double, slices: [Double], waits: [Double], count: Int) {
        let setup = try XCTUnwrap(queue.makeCommandBuffer())
        let job = try newJob(setup)
        if cold {
            CADExactCost.shared.forget(job.costKey.resources)
            job.plan = CADSlicePlan(width: width, height: height, rects: job.draws.map(\.rect),
                                    block: CADExactCost.shared.blockSide(job.costKey))
        }
        setup.commit()
        var committed: [(buffer: any MTLCommandBuffer, covered: Int, blocks: Int, side: Int)] = []
        var done = false
        let lock = NSLock()
        var waits: [Double] = []
        let group = DispatchGroup()
        if beside {
            group.enter()
            Thread.detachNewThread { [self] in
                waits = (try? lightFrames(until: { lock.withLock { done } }, end: { CACurrentMediaTime() + 10 })) ?? []
                group.leave()
            }
        }
        var gpuEnd = 0.0
        var measured: [Double] = []
        var settled = 0
        func settle(keeping ahead: Int) {
            while committed.count - settled > ahead {
                let item = committed[settled]
                item.buffer.waitUntilCompleted()
                let seconds = item.buffer.gpuEndTime - max(item.buffer.gpuStartTime, gpuEnd)
                gpuEnd = max(gpuEnd, item.buffer.gpuEndTime)
                measured.append(seconds * 1000)
                CADExactCost.shared.recordSlice(job.costKey, seconds: seconds, covered: item.covered)
                CADExactCost.shared.recordBlocks(job.costKey, side: item.side, seconds: seconds, count: item.blocks)
                settled += 1
            }
        }
        while job.hasSlicesToEncode {
            let limit = CADExactCost.shared.slicesInFlight(job.costKey)
            settle(keeping: limit - 1)
            let buffer = try XCTUnwrap(queue.makeCommandBuffer())
            guard let slice = passes.encodeSlice(job, budget: CADViewportPasses.sliceBudget, commandBuffer: buffer) else { break }
            buffer.commit()
            committed.append((buffer, slice.covered, slice.blocks, job.plan.block))
        }
        settle(keeping: 0)
        lock.withLock { done = true }
        group.wait()
        var busy = 0.0, cursor = 0.0
        for item in committed.sorted(by: { $0.buffer.gpuStartTime < $1.buffer.gpuStartTime }) {
            let b = item.buffer
            if b.gpuEndTime > cursor { busy += b.gpuEndTime - max(b.gpuStartTime, cursor); cursor = b.gpuEndTime }
        }
        return (busy * 1000, measured, waits, committed.count)
    }

    func sliced(cold: Bool) throws -> String {
        var best: (busy: Double, slices: [Double], waits: [Double], count: Int)?
        for _ in 0..<2 {
            let result = try runSliced(cold: cold, beside: false)
            if best == nil || result.busy < best!.busy { best = result }
        }
        let s = best!.slices.sorted()
        return String(format: "sliced busy %.1f ms, %d slices, slice p50 %.1f p95 %.1f max %.1f ms",
                      best!.busy, best!.count, s[s.count / 2], s[Int(Double(s.count - 1) * 0.95)], s.last!)
    }

    func waitSliced(cold: Bool) throws -> String {
        var waits: [Double] = [], busy: [Double] = []
        for _ in 0..<2 {
            let result = try runSliced(cold: cold, beside: true)
            waits += result.waits
            busy.append(result.busy)
        }
        return String(format: "sliced busy %.1f ms beside frames; ", busy.min()!) + Self.summary(waits)
    }
}
