import Foundation
import Metal
import QuartzCore
import SceneKit

/// Draws viewport frames on a dedicated thread, never on the main thread.
/// The main thread publishes the latest frame after each state change; the
/// render thread draws one frame per publication, coalescing publications
/// that arrive while it is busy. Nothing renders while nothing changes.
final class ViewportRenderLoop: @unchecked Sendable {
    struct Request {
        var frame: CADViewportFrame
        var source: CADViewportSceneSource
        var generation: UInt64
    }

    let layer: CAMetalLayer
    let renderer: SCNRenderer
    private let queue: any MTLCommandQueue
    private let passes: CADViewportPasses
    private let condition = NSCondition()
    private var pending: Request?
    private var stopped = false
    /// Frames encoded but not yet completed by the GPU.
    private let inFlight = DispatchSemaphore(value: 2)
    private let completion: @Sendable (UInt64, Double) -> Void

    /// `renderer` is drawn only by this loop's thread; other threads may use
    /// it for asynchronous resource preparation. `completion` runs on the
    /// main queue with each frame's generation and GPU seconds once the GPU
    /// has finished it.
    init?(layer: CAMetalLayer, renderer: SCNRenderer, completion: @escaping @Sendable (UInt64, Double) -> Void) {
        guard let device = renderer.device ?? layer.device ?? MTLCreateSystemDefaultDevice(),
              let queue = device.makeCommandQueue(),
              let passes = try? CADViewportPasses(device: device) else { return nil }
        layer.device = device
        self.layer = layer
        self.renderer = renderer
        self.queue = queue
        self.passes = passes
        self.completion = completion
        let thread = Thread { [self] in run() }
        thread.name = "RAD viewport renderer"
        thread.qualityOfService = .userInteractive
        thread.start()
    }

    /// Replaces any frame not yet started; the newest state always wins.
    func submit(_ request: Request) {
        condition.lock()
        pending = request
        condition.signal()
        condition.unlock()
    }

    /// Drops a frame that has not started, such as when the view detaches.
    func cancelPending() {
        condition.lock()
        pending = nil
        condition.unlock()
    }

    /// Ends the render thread after any frame in progress.
    func stop() {
        condition.lock()
        stopped = true
        pending = nil
        condition.signal()
        condition.unlock()
    }

    private func run() {
        while true {
            condition.lock()
            while pending == nil && !stopped { condition.wait() }
            if stopped { condition.unlock(); return }
            let request = pending!
            pending = nil
            condition.unlock()
            autoreleasepool { draw(request) }
        }
    }

    private func draw(_ request: Request) {
        let frame = request.frame
        let trace = ViewportFrameTrace.shared, traced = request.generation
        trace.update(traced) { $0.drawStart = ViewportFrameTrace.now; $0.exact = frame.field != nil; $0.pixels = frame.width * frame.height }
        let size = CGSize(width: frame.width, height: frame.height)
        if layer.drawableSize != size { layer.drawableSize = size }
        inFlight.wait()
        trace.update(traced) { $0.afterInFlight = ViewportFrameTrace.now }
        // A drawable is unavailable while the display sleeps or the window is
        // hidden, and one acquired across a resize has the old size. Keep the
        // request unless a newer one has replaced it, and retry shortly; no
        // state can strand later requests.
        guard let drawable = layer.nextDrawable(),
              drawable.texture.width == frame.width, drawable.texture.height == frame.height,
              let commandBuffer = queue.makeCommandBuffer() else {
            inFlight.signal()
            condition.lock()
            if pending == nil, !stopped {
                pending = request
                _ = condition.wait(until: Date().addingTimeInterval(0.1))
            }
            condition.unlock()
            return
        }
        trace.update(traced) { $0.afterDrawable = ViewportFrameTrace.now }
        commandBuffer.label = "RAD viewport frame \(request.generation)"
        passes.encode(frame, source: request.source, renderer: renderer, into: drawable.texture, commandBuffer: commandBuffer)
        trace.update(traced) { $0.afterEncode = ViewportFrameTrace.now }
        if ViewportFrameTrace.enabled {
            drawable.addPresentedHandler { presented in
                let time = presented.presentedTime
                trace.update(traced) { $0.presented = time }
            }
        }
        commandBuffer.present(drawable)
        let generation = request.generation, completion = completion, inFlight = inFlight
        commandBuffer.addCompletedHandler { buffer in
            inFlight.signal()
            let seconds = buffer.gpuEndTime - buffer.gpuStartTime
            let start = buffer.gpuStartTime, end = buffer.gpuEndTime
            trace.update(generation) { $0.gpuStart = start; $0.gpuEnd = end }
            DispatchQueue.main.async {
                trace.update(generation) { $0.completedOnMain = ViewportFrameTrace.now }
                completion(generation, seconds)
            }
        }
        commandBuffer.commit()
        trace.update(traced) { $0.committed = ViewportFrameTrace.now }
    }
}
