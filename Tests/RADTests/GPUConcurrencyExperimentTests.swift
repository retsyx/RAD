import AppKit
import Metal
import QuartzCore
import SceneKit
import XCTest
@testable import RAD

/// Experiment (RAD_GPU_EXPERIMENT=<design json>): do mesh frames on one queue
/// wait behind an exact frame running on another queue?
@MainActor
final class GPUConcurrencyExperimentTests: XCTestCase {
    final class Recorder: @unchecked Sendable {
        let lock = NSLock()
        var rows: [(String, Double, Double, Double)] = []
        func add(_ kind: String, _ submit: Double, _ start: Double, _ end: Double) { lock.withLock { rows.append((kind, submit, start, end)) } }
    }

    func testMeshFramesBesideAnExactFrame() async throws {
        guard let path = ProcessInfo.processInfo.environment["RAD_GPU_EXPERIMENT"] else { throw XCTSkip("experiment") }
        let spec = try JSONDecoder().decode(ModelSpec.self, from: Data(contentsOf: URL(fileURLWithPath: path)))
        let preview = try await EngineService().buildPreview(spec, python: EngineService.defaultPython,
                                                             threads: ProcessInfo.processInfo.activeProcessorCount)
        var meshOnly = preview
        meshOnly.proxy = nil
        func capture(_ mesh: PreviewMesh) async throws -> CADViewportCapture {
            let surface = CADSceneView(frame: NSRect(x: 0, y: 0, width: 1000, height: 820))
            surface.preparesRendererResources = false
            surface.update(mesh: mesh, mode: .shaded, grid: true, command: nil, selectedSolids: [], hiddenSolids: [])
            _ = try await surface.waitForCaptureResources(until: .now + .seconds(120))
            try surface.configureAgentView(named: "isometric", camera: CADAgentCamera())
            surface.layout()
            _ = try await surface.waitForCaptureResources(until: .now + .seconds(120))
            return try CADViewportCapture(view: surface, width: 2000, height: 1640)
        }
        let exact = try await capture(preview)
        let mesh = try await capture(meshOnly)
        let device = try XCTUnwrap(MTLCreateSystemDefaultDevice())
        func renderer(_ capture: CADViewportCapture) -> SCNRenderer {
            let r = SCNRenderer(device: device, options: nil)
            r.scene = capture.source.scene
            r.pointOfView = capture.source.camera
            r.usesReverseZ = true
            r.autoenablesDefaultLighting = false
            _ = r.prepare(capture.source.scene, shouldAbortBlock: nil)
            return r
        }
        func target() throws -> any MTLTexture {
            let d = MTLTextureDescriptor.texture2DDescriptor(pixelFormat: CADViewportPasses.colorFormat, width: 2000, height: 1640, mipmapped: false)
            d.usage = [.renderTarget, .shaderRead]
            d.storageMode = .private
            return try XCTUnwrap(device.makeTexture(descriptor: d))
        }
        let exactRenderer = renderer(exact), meshRenderer = renderer(mesh)
        let exactPasses = try CADViewportPasses(device: device), meshPasses = try CADViewportPasses(device: device)
        let exactTarget = try target(), meshTarget = try target()
        let recorder = Recorder()
        let separateQueue = ProcessInfo.processInfo.environment["RAD_GPU_SAME_QUEUE"] == nil
        let meshQueue = try XCTUnwrap(device.makeCommandQueue())
        let exactQueue = separateQueue ? try XCTUnwrap(device.makeCommandQueue()) : meshQueue
        let exactFrame = exact.frame, exactSource = exact.source, meshFrame = mesh.frame, meshSource = mesh.source
        // Warm both paths.
        for (q, p, f, s, r, t) in [(meshQueue, meshPasses, meshFrame, meshSource, meshRenderer, meshTarget),
                                   (exactQueue, exactPasses, exactFrame, exactSource, exactRenderer, exactTarget)] {
            let b = try XCTUnwrap(q.makeCommandBuffer())
            p.encode(f, source: s, renderer: r, into: t, commandBuffer: b)
            b.commit(); b.waitUntilCompleted()
        }
        let stop = Date().addingTimeInterval(4)
        let exactThread = Thread {
            // Starts after 1 s of mesh frames alone.
            Thread.sleep(forTimeInterval: 1)
            while Date() < stop.addingTimeInterval(-0.5) {
                let b = exactQueue.makeCommandBuffer()!
                let submit = CACurrentMediaTime()
                exactPasses.encode(exactFrame, source: exactSource, renderer: exactRenderer, into: exactTarget, commandBuffer: b)
                b.commit(); b.waitUntilCompleted()
                recorder.add("exact", submit, b.gpuStartTime, b.gpuEndTime)
            }
        }
        exactThread.start()
        let meshDone = DispatchSemaphore(value: 0)
        let meshThread = Thread {
            var next = CACurrentMediaTime()
            while Date() < stop {
                next += 1.0 / 120
                let b = meshQueue.makeCommandBuffer()!
                let submit = CACurrentMediaTime()
                meshPasses.encode(meshFrame, source: meshSource, renderer: meshRenderer, into: meshTarget, commandBuffer: b)
                b.addCompletedHandler { b in recorder.add("mesh", submit, b.gpuStartTime, b.gpuEndTime) }
                b.commit()
                let wait = next - CACurrentMediaTime()
                if wait > 0 { Thread.sleep(forTimeInterval: wait) }
            }
            meshDone.signal()
        }
        meshThread.start()
        while meshDone.wait(timeout: .now()) == .timedOut { try await Task.sleep(for: .milliseconds(20)) }
        try await Task.sleep(for: .milliseconds(800))
        let rows = recorder.lock.withLock { recorder.rows }.sorted { $0.1 < $1.1 }
        let exactSpans = rows.filter { $0.0 == "exact" }.map { ($0.2, $0.3) }
        func overlapsExact(_ start: Double, _ end: Double) -> Bool { exactSpans.contains { $0.0 < end && start < $0.1 } }
        var alone: [Double] = [], beside: [Double] = []
        for row in rows where row.0 == "mesh" {
            let latency = (row.3 - row.1) * 1000
            if overlapsExact(row.1, row.3) { beside.append(latency) } else { alone.append(latency) }
        }
        func stats(_ v: [Double]) -> String {
            guard !v.isEmpty else { return "-" }
            let s = v.sorted(); return String(format: "n %d med %.1f p95 %.1f max %.1f ms", s.count, s[s.count / 2], s[Int(Double(s.count) * 0.95)], s.last!)
        }
        let exactTimes = exactSpans.map { ($0.1 - $0.0) * 1000 }
        print("GPUEXP separateQueue=\(separateQueue) exact frames \(stats(exactTimes))")
        print("GPUEXP mesh submit->done alone: \(stats(alone))")
        print("GPUEXP mesh submit->done beside exact: \(stats(beside))")
    }
}

/// Experiment (RAD_GPU_EXPERIMENT_LOAD=1): does SceneKit keep loaded color and
/// depth attachments when a pass descriptor loads them?
@MainActor
final class SceneKitLoadExperimentTests: XCTestCase {
    func testSceneKitHonorsLoadedAttachments() throws {
        guard ProcessInfo.processInfo.environment["RAD_GPU_EXPERIMENT_LOAD"] != nil else { throw XCTSkip("experiment") }
        let device = try XCTUnwrap(MTLCreateSystemDefaultDevice())
        let queue = try XCTUnwrap(device.makeCommandQueue())
        let scene = SCNScene()
        scene.background.contents = NSColor.red
        let camera = SCNNode(); camera.camera = SCNCamera(); camera.position = SCNVector3(0, 0, 10)
        scene.rootNode.addChildNode(camera)
        // A small box at the center, drawn with depth test.
        let box = SCNNode(geometry: SCNBox(width: 1, height: 1, length: 1, chamferRadius: 0))
        box.geometry?.firstMaterial?.diffuse.contents = NSColor.blue
        box.geometry?.firstMaterial?.lightingModel = .constant
        scene.rootNode.addChildNode(box)
        let renderer = SCNRenderer(device: device, options: nil)
        renderer.scene = scene
        renderer.pointOfView = camera
        renderer.usesReverseZ = true
        func texture(_ format: MTLPixelFormat, _ samples: Int, _ usage: MTLTextureUsage) -> any MTLTexture {
            let d = MTLTextureDescriptor()
            d.textureType = samples > 1 ? .type2DMultisample : .type2D
            d.pixelFormat = format; d.width = 64; d.height = 64; d.sampleCount = samples
            d.usage = usage; d.storageMode = .private
            return device.makeTexture(descriptor: d)!
        }
        let color = texture(CADViewportPasses.colorFormat, 4, [.renderTarget, .shaderRead])
        let depth = texture(.depth32Float, 4, [.renderTarget, .shaderRead])
        let resolved = texture(CADViewportPasses.colorFormat, 1, [.renderTarget, .shaderRead])
        let buffer = try XCTUnwrap(queue.makeCommandBuffer())
        // Pass 1 (ours): clear color to green, depth to "nearest" (reverse-Z 1) over the left half via clear + scissored quad is
        // unnecessary: clear depth to 1 (nearest) everywhere so SceneKit's box must fail the depth test if depth is loaded.
        let first = MTLRenderPassDescriptor()
        first.colorAttachments[0].texture = color
        first.colorAttachments[0].loadAction = .clear
        first.colorAttachments[0].clearColor = MTLClearColor(red: 0, green: 1, blue: 0, alpha: 1)
        first.colorAttachments[0].storeAction = .store
        first.depthAttachment.texture = depth
        first.depthAttachment.loadAction = .clear
        first.depthAttachment.clearDepth = 1
        first.depthAttachment.storeAction = .store
        buffer.makeRenderCommandEncoder(descriptor: first)!.endEncoding()
        let second = MTLRenderPassDescriptor()
        second.colorAttachments[0].texture = color
        second.colorAttachments[0].resolveTexture = resolved
        second.colorAttachments[0].loadAction = .load
        second.colorAttachments[0].storeAction = .storeAndMultisampleResolve
        second.depthAttachment.texture = depth
        second.depthAttachment.loadAction = .load
        second.depthAttachment.storeAction = .store
        renderer.render(withViewport: CGRect(x: 0, y: 0, width: 64, height: 64), commandBuffer: buffer, passDescriptor: second)
        let readback = try XCTUnwrap(device.makeBuffer(length: 64 * 64 * 4, options: .storageModeShared))
        let blit = buffer.makeBlitCommandEncoder()!
        blit.copy(from: resolved, sourceSlice: 0, sourceLevel: 0, sourceOrigin: MTLOrigin(), sourceSize: MTLSize(width: 64, height: 64, depth: 1),
                  to: readback, destinationOffset: 0, destinationBytesPerRow: 256, destinationBytesPerImage: 256 * 64)
        blit.endEncoding()
        buffer.commit(); buffer.waitUntilCompleted()
        let p = readback.contents().assumingMemoryBound(to: UInt8.self)
        func px(_ x: Int, _ y: Int) -> String { let i = (y * 64 + x) * 4; return "b\(p[i]) g\(p[i+1]) r\(p[i+2]) a\(p[i+3])" }
        print("LOADEXP corner \(px(2, 2)) center \(px(32, 32))")
    }
}
