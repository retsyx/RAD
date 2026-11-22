import AppKit
import Metal
import SceneKit

/// An independent copy of a viewport's scene and highlight state, rendered
/// offscreen through the viewport's own passes. Native geometry buffers are
/// immutable and shared; mutable materials, cameras and lights are copied,
/// so a user's orbit or color change never affects the image and capturing
/// never changes the live view.
final class CADViewportCapture: @unchecked Sendable {
    let source: CADViewportSceneSource
    let frame: CADViewportFrame

    @MainActor
    init(view: CADSceneView, width: Int, height: Int) throws {
        guard let camera = view.pointOfView else {
            throw RADError.message("The design camera is unavailable for capture.")
        }
        source = try Self.isolatedCopy(of: view.scene, camera: camera)
        source.synchronizeCameras()
        frame = view.makeFrame(width: width, height: height)
    }

    var width: Int { frame.width }
    var height: Int { frame.height }

    /// GPU busy time (union of the command buffers' execution) and command
    /// buffers of a capture, for measurements.
    struct Statistics {
        var gpuSeconds = 0.0
        var commandBuffers = 0
    }

    /// Renders on the calling thread and waits for the GPU. Cancellation of
    /// the calling task stops preparation before any frame is encoded.
    func image() throws -> CGImage {
        try Self.image(of: frame, source: source)
    }

    /// Renders a frame of any scene source offscreen through the viewport's
    /// passes, preparing a renderer of its own unless one is supplied. Exact
    /// layers are drawn in the viewport's slices, each its own command buffer
    /// so other GPU work interleaves, then composited; the call waits for all.
    static func image(of frame: CADViewportFrame, source: CADViewportSceneSource,
                      renderer existing: SCNRenderer? = nil, passes existingPasses: CADViewportPasses? = nil,
                      sliceBudget: Double = CADViewportPasses.sliceBudget,
                      statistics: UnsafeMutablePointer<Statistics>? = nil) throws -> CGImage {
        try Task.checkCancellation()
        guard let device = existing?.device ?? MTLCreateSystemDefaultDevice(), let queue = device.makeCommandQueue() else {
            throw RADError.message("Metal is unavailable, so RAD cannot render a design image.")
        }
        let renderer = existing ?? SCNRenderer(device: device, options: nil)
        renderer.scene = source.scene
        renderer.pointOfView = source.camera
        renderer.usesReverseZ = true
        renderer.autoenablesDefaultLighting = false
        guard renderer.prepare(source.scene, shouldAbortBlock: { Task.isCancelled }) else {
            try Task.checkCancellation()
            throw RADError.message("RAD could not prepare the design image.")
        }
        try Task.checkCancellation()
        let width = frame.width, height = frame.height
        let passes = try existingPasses ?? CADViewportPasses(device: device)
        let descriptor = MTLTextureDescriptor.texture2DDescriptor(pixelFormat: CADViewportPasses.colorFormat,
                                                                  width: width, height: height, mipmapped: false)
        descriptor.usage = [.renderTarget, .shaderRead]
        descriptor.storageMode = .shared
        guard let target = device.makeTexture(descriptor: descriptor), let commandBuffer = queue.makeCommandBuffer() else {
            throw RADError.message("RAD could not allocate the design image.")
        }
        var committed: [any MTLCommandBuffer] = []
        let failed = frame.key.map { CADExactCost.shared.hasFailed(resources: $0.resources, translucent: $0.translucent) } ?? false
        var costKey: CADExactCost.Key?
        if frame.field != nil, !failed,
           let job = passes.makeJob(frame, source: source, renderer: renderer, commandBuffer: commandBuffer) {
            costKey = job.costKey
            commandBuffer.commit()
            committed.append(commandBuffer)
            // One queue orders the slices before the composite. A few slices
            // run ahead; one that fails or takes far too long ends exact
            // faces of the design in this mode (display meshes instead).
            var waited = 0, gpuEnd = 0.0, slow = false
            func settle(keeping ahead: Int) {
                while committed.count - waited > ahead {
                    let buffer = committed[waited]
                    buffer.waitUntilCompleted()
                    let seconds = buffer.gpuEndTime - max(buffer.gpuStartTime, gpuEnd)
                    gpuEnd = max(gpuEnd, buffer.gpuEndTime)
                    if buffer.status != .completed || seconds > CADExactCost.failingSliceSeconds {
                        if CADExactCost.shared.recordStrike(job.costKey) || buffer.status != .completed { slow = true }
                    }
                    waited += 1
                }
            }
            func slices(_ encode: (any MTLCommandBuffer) -> Bool) throws {
                while !slow {
                    try Task.checkCancellation()
                    guard let slice = queue.makeCommandBuffer() else {
                        throw RADError.message("RAD could not render the design image.")
                    }
                    guard encode(slice) else { return }
                    slice.commit()
                    committed.append(slice)
                    settle(keeping: 4)
                }
            }
            try slices { passes.encodeSlice(job, budget: sliceBudget, commandBuffer: $0) != nil }
            job.completed = job.encoded
            passes.prepareMasks(job, frame: frame)
            try slices { passes.encodeMaskSlice(job, budget: sliceBudget, commandBuffer: $0) != nil }
            for coverage in job.masks.values { coverage.completed = coverage.encoded }
            settle(keeping: 0)
            if slow {
                var meshes = frame
                meshes.field = nil
                meshes.key = nil
                return try image(of: meshes, source: source, renderer: renderer, passes: passes, sliceBudget: sliceBudget,
                                 statistics: statistics)
            }
            guard let composite = queue.makeCommandBuffer(),
                  passes.encodeComposite(frame, job: job, source: source, renderer: renderer, into: target,
                                         commandBuffer: composite) else {
                throw RADError.message("RAD could not render the design image.")
            }
            composite.commit()
            committed.append(composite)
        } else {
            passes.encode(frame, source: source, renderer: renderer, into: target, commandBuffer: commandBuffer)
            commandBuffer.commit()
            committed.append(commandBuffer)
        }
        committed.last?.waitUntilCompleted()
        if let costKey, committed.contains(where: { $0.status != .completed }) {
            // Exact layers the GPU could not draw: display meshes instead.
            CADExactCost.shared.recordStrike(costKey)
            var meshes = frame
            meshes.field = nil
            meshes.key = nil
            return try image(of: meshes, source: source, renderer: renderer, passes: passes, sliceBudget: sliceBudget,
                             statistics: statistics)
        }
        var busy = 0.0, cursor = 0.0
        for buffer in committed.sorted(by: { $0.gpuStartTime < $1.gpuStartTime }) {
            buffer.waitUntilCompleted()
            if buffer.status != .completed { throw RADError.message("RAD could not render the design image.") }
            if buffer.gpuEndTime > cursor {
                busy += buffer.gpuEndTime - max(buffer.gpuStartTime, cursor)
                cursor = buffer.gpuEndTime
            }
        }
        statistics?.pointee.gpuSeconds = busy
        statistics?.pointee.commandBuffers = committed.count
        let bytesPerRow = width * 4
        var pixels = Data(count: bytesPerRow * height)
        pixels.withUnsafeMutableBytes { raw in
            target.getBytes(raw.baseAddress!, bytesPerRow: bytesPerRow,
                            from: MTLRegionMake2D(0, 0, width, height), mipmapLevel: 0)
        }
        // The target stores BGRA bytes in sRGB, with opaque alpha.
        guard let provider = CGDataProvider(data: pixels as CFData),
              let space = CGColorSpace(name: CGColorSpace.sRGB),
              let image = CGImage(width: width, height: height, bitsPerComponent: 8, bitsPerPixel: 32,
                                  bytesPerRow: bytesPerRow, space: space,
                                  bitmapInfo: CGBitmapInfo(rawValue: CGBitmapInfo.byteOrder32Little.rawValue
                                                           | CGImageAlphaInfo.premultipliedFirst.rawValue),
                                  provider: provider, decode: nil, shouldInterpolate: false,
                                  intent: .defaultIntent) else {
            throw RADError.message("RAD could not encode the design image.")
        }
        return image
    }

    func pngData() throws -> Data {
        let png = try CADAgentRenderer.encodePNG(try image())
        try Task.checkCancellation()
        return png
    }

    @MainActor
    private static func isolatedCopy(of source: SCNScene, camera sourceCamera: SCNNode) throws -> CADViewportSceneSource {
        let scene = SCNScene()
        scene.background.contents = source.background.contents
        scene.lightingEnvironment.contents = source.lightingEnvironment.contents
        scene.lightingEnvironment.intensity = source.lightingEnvironment.intensity
        let root = source.rootNode.clone()
        var copiedCamera: SCNNode?
        func isolate(_ original: SCNNode, _ copy: SCNNode) {
            if original === sourceCamera { copiedCamera = copy }
            if let geometry = original.geometry?.copy() as? SCNGeometry {
                if let originalGeometry = original.geometry {
                    CADStrokeGeometry.copyParameters(from: originalGeometry, to: geometry)
                }
                geometry.materials = original.geometry?.materials.map { ($0.copy() as? SCNMaterial) ?? $0 } ?? []
                copy.geometry = geometry
            }
            copy.camera = original.camera?.copy() as? SCNCamera
            copy.light = original.light?.copy() as? SCNLight
            for (originalChild, copiedChild) in zip(original.childNodes, copy.childNodes) {
                isolate(originalChild, copiedChild)
            }
        }
        isolate(source.rootNode, root)
        scene.rootNode.addChildNode(root)
        guard let camera = copiedCamera, let copied = CADViewportSceneSource.adopt(scene: scene, camera: camera) else {
            throw RADError.message("The design camera is unavailable for capture.")
        }
        return copied
    }
}
