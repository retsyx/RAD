import Foundation
import Metal
import QuartzCore
import SceneKit

/// Draws viewport frames on a dedicated thread, never on the main thread.
/// The main thread publishes the latest frame after each state change; the
/// render thread draws one frame per publication, coalescing publications
/// that arrive while it is busy. Nothing renders while nothing changes.
///
/// Exact faces are drawn into stored layers by a job keyed by the frame's
/// geometry identity (`CADExactKey`). A design whose exact geometry fits one
/// slice budget draws it in the frame that shows it. Heavier geometry is
/// drawn in screen slices of about `CADViewportPasses.sliceBudget` GPU time,
/// each its own command buffer on a separate queue, at most two in flight,
/// so newer frames wait for at most about one slice. A newer frame with the
/// same key keeps the job and changes only what is composited; another key
/// abandons its remaining slices. Until the slices complete, frames whose
/// content differs from the screen's are drawn with display meshes; then the
/// newest frame composites the stored layers.
final class ViewportRenderLoop: @unchecked Sendable {
    struct Request {
        var frame: CADViewportFrame
        var source: CADViewportSceneSource
        var generation: UInt64
    }

    /// How a presented frame drew its published state.
    struct Outcome: Sendable, Equatable {
        /// The frame composited exact faces.
        var exact: Bool
        /// The frame shows its published state completely: exact faces with
        /// their exact outlines, or the display meshes the state asked for.
        var final: Bool
        /// GPU seconds of the frame's exact geometry (all of its slices) and
        /// composite, reported once per geometry.
        var geometrySeconds: Double?
    }

    /// Exact slices in flight at once: the GPU stays busy between slices
    /// while a newer frame waits for at most about one of them.
    static let slicesInFlight = 2

    let layer: CAMetalLayer
    let renderer: SCNRenderer
    private let frameQueue: any MTLCommandQueue
    private let sliceQueue: any MTLCommandQueue
    private let passes: CADViewportPasses
    /// Orders stored-layer access between the queues: slices wait for the
    /// frames that read the previous layers, frames for slices that write them.
    private let sliceEvent: any MTLEvent
    private let frameEvent: any MTLEvent
    private var sliceSerial: UInt64 = 0
    private var frameSerial: UInt64 = 0
    private let condition = NSCondition()
    private var pending: Request?
    private var stopped = false
    private var releaseRequested = false
    /// GPU completions of slices, applied by the render thread.
    private enum Finished {
        case slice(CADExactJob, seconds: Double, covered: Int)
        case mask(CADExactJob, CADMaskCoverage)
    }
    private var finished: [Finished] = []
    private var slicesRunning = 0
    /// Frames encoded but not yet completed by the GPU.
    private let inFlight = DispatchSemaphore(value: 2)
    private let completion: @Sendable (UInt64, Double, Outcome) -> Void

    // Render-thread state.
    private var job: CADExactJob?
    /// The newest request showing the job's geometry, composited once its
    /// slices complete.
    private var latest: Request?
    private var needsComposite = false
    /// The presentation revision of the newest presented frame.
    private var presentedRevision: UInt64?

    /// `renderer` is drawn only by this loop's thread; other threads may use
    /// it for asynchronous resource preparation. `completion` runs on the
    /// main queue with each presented frame's generation, GPU seconds and
    /// outcome once the GPU has finished it.
    init?(layer: CAMetalLayer, renderer: SCNRenderer,
          completion: @escaping @Sendable (UInt64, Double, Outcome) -> Void) {
        guard let device = renderer.device ?? layer.device ?? MTLCreateSystemDefaultDevice(),
              let frameQueue = device.makeCommandQueue(), let sliceQueue = device.makeCommandQueue(),
              let sliceEvent = device.makeEvent(), let frameEvent = device.makeEvent(),
              let passes = try? CADViewportPasses(device: device) else { return nil }
        layer.device = device
        frameQueue.label = "RAD viewport frames"
        sliceQueue.label = "RAD viewport exact slices"
        self.layer = layer
        self.renderer = renderer
        self.frameQueue = frameQueue
        self.sliceQueue = sliceQueue
        self.sliceEvent = sliceEvent
        self.frameEvent = frameEvent
        self.passes = passes
        self.completion = completion
        let thread = Thread { [self] in run() }
        thread.name = "RAD viewport renderer"
        thread.qualityOfService = .userInteractive
        thread.start()
    }

    /// Exact work encoded so far (diagnostics and tests).
    var statistics: CADViewportPasses.Statistics { passes.statistics }

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

    /// Abandons exact work and releases the stored layers, such as when the
    /// view detaches; GPU work already committed finishes.
    func releaseResources() {
        condition.lock()
        pending = nil
        releaseRequested = true
        condition.signal()
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
            while !stopped && pending == nil && !releaseRequested && finished.isEmpty && !hasJobWork() {
                condition.wait()
            }
            if stopped { condition.unlock(); return }
            let request = pending
            pending = nil
            let release = releaseRequested
            releaseRequested = false
            let completed = finished
            finished.removeAll()
            condition.unlock()
            autoreleasepool {
                if release { abandonJob(); passes.releaseStoredLayers() }
                apply(completed)
                if let request { handle(request) } else { advance() }
            }
        }
    }

    // MARK: Decisions

    /// Whether the job has work the render thread can do now. Called with
    /// the condition locked.
    private func hasJobWork() -> Bool {
        guard let job else { return false }
        if job.hasSlicesToEncode { return slicesRunning < Self.slicesInFlight }
        guard job.isComplete else { return false }
        if needsComposite, latest != nil { return true }
        return slicesRunning < Self.slicesInFlight && job.masks.values.contains { !$0.cleared || !$0.plan.isDone }
    }

    private func apply(_ completed: [Finished]) {
        for item in completed {
            switch item {
            case .slice(let owner, let seconds, let covered):
                owner.completed += 1
                owner.gpuSeconds += seconds
                CADExactCost.shared.recordSlice(owner.costKey, seconds: seconds, covered: covered)
                if owner === job, owner.isComplete {
                    CADExactCost.shared.recordFrame(owner.costKey, seconds: owner.gpuSeconds, covered: owner.plan.total)
                    needsComposite = true
                }
            case .mask(let owner, let coverage):
                coverage.completed += 1
                if owner === job, coverage.isComplete { needsComposite = true }
            }
        }
    }

    private func handle(_ request: Request) {
        let frame = request.frame
        guard let key = frame.key, frame.field != nil else {
            // No exact faces: wireframe, or resources still preparing.
            abandonJob()
            _ = presentMesh(request, final: true)
            return
        }
        if let job, job.key == key {
            latest = request
            needsComposite = true
            if job.isComplete {
                composite()
            } else if frame.presentation != presentedRevision {
                // The screen does not show this state yet: display meshes
                // stand in until the exact layers are complete.
                _ = presentMesh(request, final: false)
            }
            return
        }
        abandonJob()
        if frame.navigating, !frame.exactFitsBudget {
            _ = presentMesh(request, final: true)
            return
        }
        start(request)
    }

    private func abandonJob() {
        job = nil
        latest = nil
        needsComposite = false
    }

    /// Starts a job for a frame with a new geometry identity: in the frame
    /// itself when its estimated cost fits one slice, in slices otherwise.
    private func start(_ request: Request) {
        let frame = request.frame
        guard let commandBuffer = frameQueue.makeCommandBuffer() else { return }
        commandBuffer.label = "RAD viewport frame \(request.generation)"
        let trace = ViewportFrameTrace.shared
        trace.update(request.generation) { $0.drawStart = ViewportFrameTrace.now; $0.pixels = frame.width * frame.height }
        // The probe pass records the view and projection of this frame.
        guard let job = passes.makeJob(frame, source: request.source, renderer: renderer, commandBuffer: commandBuffer) else {
            commandBuffer.commit()
            _ = presentMesh(request, final: true)
            return
        }
        let estimate = CADExactCost.shared.frameRate(job.costKey) * Double(job.plan.total)
        if frame.navigating, estimate > CADViewportPasses.sliceBudget {
            // Navigation never waits for slices.
            commandBuffer.commit()
            _ = presentMesh(request, final: true)
            return
        }
        if estimate <= CADViewportPasses.sliceBudget {
            // One frame draws and shows the geometry.
            self.job = job
            latest = request
            guard let drawable = acquireDrawable(for: request) else {
                commandBuffer.commit()
                abandonJob()
                return
            }
            commandBuffer.encodeWaitForEvent(sliceEvent, value: sliceSerial)
            passes.encodeAll(job, frame: frame, commandBuffer: commandBuffer)
            let matched = passes.encodeComposite(frame, job: job, source: request.source, renderer: renderer,
                                                 into: drawable.texture, commandBuffer: commandBuffer)
            frameSerial += 1
            commandBuffer.encodeSignalEvent(frameEvent, value: frameSerial)
            guard matched else {
                // The camera changed while encoding: show nothing, redraw newer state.
                commandBuffer.commit()
                inFlight.signal()
                abandonJob()
                requeue(request)
                return
            }
            job.reported = true
            needsComposite = false
            let covered = job.plan.total, costKey = job.costKey
            commit(commandBuffer, drawable: drawable, request: request, kind: "single",
                   outcome: Outcome(exact: true, final: true, geometrySeconds: 0)) { seconds in
                CADExactCost.shared.recordFrame(costKey, seconds: seconds, covered: covered)
                CADExactCost.shared.recordSlice(costKey, seconds: seconds, covered: covered)
            }
            return
        }
        commandBuffer.commit()
        self.job = job
        latest = request
        needsComposite = true
        if frame.presentation != presentedRevision { _ = presentMesh(request, final: false) }
        advance()
    }

    /// Encodes the job's next work: geometry slices, then the newest frame's
    /// composite, then outline coverage slices.
    private func advance() {
        guard let job else { return }
        if job.hasSlicesToEncode {
            while job.hasSlicesToEncode, running() < Self.slicesInFlight {
                let encoded = encodeSlice(job) { commandBuffer in
                    passes.encodeSlice(job, budget: CADViewportPasses.sliceBudget, commandBuffer: commandBuffer)
                        .map { .slice(job, seconds: 0, covered: $0.covered) }
                }
                if !encoded { break }
            }
            return
        }
        guard job.isComplete else { return }
        if needsComposite, latest != nil {
            composite()
            return
        }
        while running() < Self.slicesInFlight, job.masks.values.contains(where: { !$0.cleared || !$0.plan.isDone }) {
            let encoded = encodeSlice(job) { commandBuffer in
                passes.encodeMaskSlice(job, budget: CADViewportPasses.sliceBudget, commandBuffer: commandBuffer)
                    .map { .mask(job, $0) }
            }
            if !encoded { break }
        }
    }

    private func running() -> Int { condition.withLock { slicesRunning } }

    /// Encodes one slice command buffer on the slice queue; false when there
    /// was nothing to encode.
    private func encodeSlice(_ job: CADExactJob, _ body: (any MTLCommandBuffer) -> Finished?) -> Bool {
        guard let commandBuffer = sliceQueue.makeCommandBuffer() else { return false }
        commandBuffer.label = "RAD exact slice"
        commandBuffer.encodeWaitForEvent(frameEvent, value: frameSerial)
        guard let record = body(commandBuffer) else { return false }
        sliceSerial += 1
        commandBuffer.encodeSignalEvent(sliceEvent, value: sliceSerial)
        condition.withLock { slicesRunning += 1 }
        let trace = ViewportFrameTrace.shared, committed = ViewportFrameTrace.now
        commandBuffer.addCompletedHandler { [weak self] buffer in
            let seconds = max(0, buffer.gpuEndTime - buffer.gpuStartTime)
            if ViewportFrameTrace.enabled {
                var covered = 0
                if case .slice(_, _, let count) = record { covered = count }
                trace.addSlice(.init(job: ObjectIdentifier(job), mask: { if case .mask = record { return true }; return false }(),
                                     covered: covered, committed: committed, gpuStart: buffer.gpuStartTime,
                                     gpuEnd: buffer.gpuEndTime))
            }
            guard let self else { return }
            let result: Finished
            switch record {
            case .slice(let owner, _, let covered): result = .slice(owner, seconds: seconds, covered: covered)
            case .mask: result = record
            }
            self.condition.lock()
            self.slicesRunning -= 1
            self.finished.append(result)
            self.condition.signal()
            self.condition.unlock()
        }
        commandBuffer.commit()
        return true
    }

    /// Composites the newest request from the complete job, with its exact
    /// outline coverage when that is small enough to draw in this frame.
    private func composite() {
        guard let job, let request = latest else { return }
        let frame = request.frame
        let trace = ViewportFrameTrace.shared
        trace.update(request.generation) { $0.drawStart = ViewportFrameTrace.now; $0.pixels = frame.width * frame.height }
        guard let drawable = acquireDrawable(for: request), let commandBuffer = frameQueue.makeCommandBuffer() else { return }
        commandBuffer.label = "RAD viewport frame \(request.generation)"
        commandBuffer.encodeWaitForEvent(sliceEvent, value: sliceSerial)
        passes.prepareMasks(job, frame: frame)
        let pendingMasks = job.masks.values.filter { !$0.cleared || !$0.plan.isDone }
        let maskCost = pendingMasks.reduce(0) { $0 + $1.plan.remaining }
        if !pendingMasks.isEmpty,
           CADExactCost.shared.sliceRate(job.costKey) * Double(maskCost) <= CADViewportPasses.sliceBudget {
            while let coverage = passes.encodeMaskSlice(job, budget: .infinity, commandBuffer: commandBuffer) {
                coverage.completed = coverage.encoded
            }
        }
        let matched = passes.encodeComposite(frame, job: job, source: request.source, renderer: renderer,
                                             into: drawable.texture, commandBuffer: commandBuffer)
        frameSerial += 1
        commandBuffer.encodeSignalEvent(frameEvent, value: frameSerial)
        guard matched else {
            commandBuffer.commit()
            inFlight.signal()
            abandonJob()
            requeue(request)
            return
        }
        needsComposite = false
        let geometry = job.reported ? nil : job.gpuSeconds
        job.reported = true
        commit(commandBuffer, drawable: drawable, request: request, kind: "composite",
               outcome: Outcome(exact: true, final: passes.masksComplete(job), geometrySeconds: geometry))
    }

    // MARK: Presentation

    /// Waits for a frame slot and the layer's next drawable of the frame's
    /// size. A drawable is unavailable while the display sleeps or the window
    /// is hidden, and one acquired across a resize has the old size: the
    /// request is kept unless a newer one replaced it and retried shortly, so
    /// no state can strand later requests.
    private func acquireDrawable(for request: Request) -> (any CAMetalDrawable)? {
        let frame = request.frame
        let size = CGSize(width: frame.width, height: frame.height)
        if layer.drawableSize != size { layer.drawableSize = size }
        inFlight.wait()
        let trace = ViewportFrameTrace.shared
        trace.update(request.generation) { $0.afterInFlight = ViewportFrameTrace.now }
        guard let drawable = layer.nextDrawable(), drawable.texture.width == frame.width,
              drawable.texture.height == frame.height else {
            inFlight.signal()
            condition.lock()
            if pending == nil, !stopped {
                pending = request
                _ = condition.wait(until: Date().addingTimeInterval(0.1))
            }
            condition.unlock()
            return nil
        }
        trace.update(request.generation) { $0.afterDrawable = ViewportFrameTrace.now }
        return drawable
    }

    private func requeue(_ request: Request) {
        condition.lock()
        if pending == nil, !stopped { pending = request }
        condition.unlock()
    }

    /// Draws a frame with every solid's display mesh. Returns false when no
    /// drawable was available (the request is retried).
    private func presentMesh(_ request: Request, final: Bool) -> Bool {
        let frame = request.frame
        let trace = ViewportFrameTrace.shared
        trace.update(request.generation) { $0.drawStart = ViewportFrameTrace.now; $0.pixels = frame.width * frame.height }
        guard let drawable = acquireDrawable(for: request), let commandBuffer = frameQueue.makeCommandBuffer() else { return false }
        commandBuffer.label = "RAD viewport frame \(request.generation)"
        passes.encodeMesh(frame, source: request.source, renderer: renderer, into: drawable.texture, commandBuffer: commandBuffer)
        commit(commandBuffer, drawable: drawable, request: request, kind: final ? "mesh" : "stand-in",
               outcome: Outcome(exact: false, final: final, geometrySeconds: nil))
        return true
    }

    /// Presents and commits a frame; `measured` receives its GPU seconds.
    private func commit(_ commandBuffer: any MTLCommandBuffer, drawable: any CAMetalDrawable, request: Request, kind: String,
                        outcome: Outcome, measured: (@Sendable (Double) -> Void)? = nil) {
        let trace = ViewportFrameTrace.shared, generation = request.generation
        trace.update(generation) { $0.afterEncode = ViewportFrameTrace.now; $0.kind = kind; $0.exact = outcome.exact }
        if ViewportFrameTrace.enabled {
            drawable.addPresentedHandler { presented in
                let time = presented.presentedTime
                trace.update(generation) { $0.presented = time }
            }
        }
        presentedRevision = request.frame.presentation
        commandBuffer.present(drawable)
        let completion = completion, inFlight = inFlight
        commandBuffer.addCompletedHandler { buffer in
            inFlight.signal()
            let seconds = max(0, buffer.gpuEndTime - buffer.gpuStartTime)
            let start = buffer.gpuStartTime, end = buffer.gpuEndTime
            trace.update(generation) { $0.gpuStart = start; $0.gpuEnd = end }
            measured?(seconds)
            var outcome = outcome
            if let geometry = outcome.geometrySeconds { outcome.geometrySeconds = geometry + seconds }
            DispatchQueue.main.async {
                trace.update(generation) { $0.completedOnMain = ViewportFrameTrace.now }
                completion(generation, seconds, outcome)
            }
        }
        commandBuffer.commit()
        trace.update(generation) { $0.committed = ViewportFrameTrace.now }
    }
}
