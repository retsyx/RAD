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
/// geometry identity (`CADExactKey`). A design whose exact geometry fits the
/// frame budget draws it in the frame that shows it. Heavier geometry is
/// drawn in a few screen slices of about `CADViewportPasses.sliceBudget` GPU
/// time, each its own command buffer on a separate queue: one runs while the
/// next is encoded behind it and committed by its completion, so the GPU
/// does not wait for this thread and at most one slice is committed when a
/// newer view abandons the job. The GPU switches to a newer frame's passes
/// once the tiles in flight finish, whatever the slice's size. A newer frame
/// with the same key keeps the job and changes only what is composited;
/// another key abandons its remaining slices. Until the slices complete,
/// frames whose content differs from the screen's are drawn with display
/// meshes; then the newest frame composites the stored layers.
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
        /// The GPU failed a slice or took far too long for it.
        case struck(CADExactJob)
    }
    private var finished: [Finished] = []
    private var slicesRunning = 0
    /// The next slice, encoded while one runs and committed by the running
    /// slice's completion, so the GPU does not wait for the render thread
    /// between slices. Dropped with its job.
    private var chained: (job: CADExactJob, buffer: any MTLCommandBuffer, serial: UInt64)?
    /// The serial of the newest committed slice, which frames wait for.
    private var committedSliceSerial: UInt64 = 0
    /// GPU end of the latest completed slice: slices run one after another
    /// on their queue, so each one's own time starts no earlier.
    private var sliceGPUEnd = 0.0
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
        if job.hasSlicesToEncode { return slicesRunning == 0 || chained == nil }
        guard job.isComplete else { return false }
        if needsComposite, latest != nil { return true }
        return slicesRunning == 0 && job.masks.values.contains { !$0.cleared || !$0.plan.isDone }
    }

    private func apply(_ completed: [Finished]) {
        for item in completed {
            switch item {
            case .slice(let owner, let seconds, let covered):
                owner.completed += 1
                owner.gpuSeconds += seconds
                CADExactCost.shared.recordSlice(owner.costKey, seconds: seconds, covered: covered)
                if owner === job, owner.isComplete {
                    // Only whole frames measure what one frame can draw at once.
                    if owner.encoded == 1 {
                        CADExactCost.shared.recordFrame(owner.costKey, seconds: owner.gpuSeconds, covered: owner.plan.total)
                    }
                    needsComposite = true
                }
            case .mask(let owner, let coverage):
                coverage.completed += 1
                if owner === job, coverage.isComplete { needsComposite = true }
            case .struck(let owner):
                if CADExactCost.shared.recordStrike(owner.costKey), owner === job, let latest {
                    // Display meshes from now on; show the newest state with them.
                    abandonJob()
                    requeue(latest)
                }
            }
        }
    }

    /// What the render thread does with a newly published frame.
    enum Action: Equatable {
        /// Abandon any job and draw display meshes, as the frame asks.
        case meshes
        /// Composite the job's complete layers.
        case composite
        /// Keep the job in progress; display meshes stand in for the frame
        /// when the screen does not show its content yet.
        case keep(standIn: Bool)
        /// Abandon any job and start one for the frame's geometry.
        case start
    }

    /// The decision for `frame` given the current job's key and progress and
    /// the presentation revision of the newest presented frame.
    static func action(for frame: CADViewportFrame, job: (key: CADExactKey, complete: Bool)?,
                       presented: UInt64?, failed: Bool = false) -> Action {
        // No exact faces (`key` comes with `field`): wireframe, resources
        // still preparing, or exact layers the GPU could not draw.
        guard let key = frame.key, !failed else { return .meshes }
        if let job, job.key == key {
            return job.complete ? .composite : .keep(standIn: frame.presentation != presented)
        }
        return frame.navigating && !frame.exactFitsBudget ? .meshes : .start
    }

    private func handle(_ request: Request) {
        let failed = request.frame.key.map {
            CADExactCost.shared.hasFailed(resources: $0.resources, translucent: $0.translucent)
        } ?? false
        switch Self.action(for: request.frame, job: job.map { ($0.key, $0.isComplete) }, presented: presentedRevision,
                           failed: failed) {
        case .meshes:
            abandonJob()
            _ = presentMesh(request, final: true)
        case .composite:
            latest = request
            needsComposite = true
            composite()
        case .keep(let standIn):
            latest = request
            needsComposite = true
            // Display meshes stand in until the exact layers are complete.
            if standIn { _ = presentMesh(request, final: false) }
        case .start:
            abandonJob()
            start(request)
        }
    }

    private func abandonJob() {
        job = nil
        latest = nil
        needsComposite = false
        // Released after unlocking: releasing an uncommitted command buffer
        // runs its completed handlers.
        let dropped = condition.withLock {
            defer { chained = nil }
            return chained
        }
        withExtendedLifetime(dropped) {}
    }

    private func committedSerial() -> UInt64 { condition.withLock { committedSliceSerial } }

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
        if frame.navigating, estimate > CADViewportPasses.frameBudget {
            // Navigation never waits for slices.
            commandBuffer.commit()
            _ = presentMesh(request, final: true)
            return
        }
        if estimate <= CADViewportPasses.frameBudget {
            // One frame draws and shows the geometry.
            self.job = job
            latest = request
            guard let drawable = acquireDrawable(for: request) else {
                commandBuffer.commit()
                abandonJob()
                return
            }
            commandBuffer.encodeWaitForEvent(sliceEvent, value: committedSerial())
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
                   outcome: Outcome(exact: true, final: true, geometrySeconds: 0), geometry: job) { seconds in
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
            while job.hasSlicesToEncode {
                let (running, waiting) = condition.withLock { (slicesRunning, chained != nil) }
                let chain = running > 0
                guard !chain || !waiting else { break }
                let encoded = encodeSlice(job, chain: chain) { commandBuffer in
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
        while running() == 0, condition.withLock({ chained == nil }),
              job.masks.values.contains(where: { !$0.cleared || !$0.plan.isDone }) {
            let encoded = encodeSlice(job) { commandBuffer in
                passes.encodeMaskSlice(job, budget: CADViewportPasses.sliceBudget, commandBuffer: commandBuffer)
                    .map { .mask(job, $0) }
            }
            if !encoded { break }
        }
    }

    private func running() -> Int { condition.withLock { slicesRunning } }

    /// Encodes one slice command buffer on the slice queue, committed now or,
    /// with `chain`, by the running slice's completion; false when there was
    /// nothing to encode.
    private func encodeSlice(_ job: CADExactJob, chain: Bool = false, _ body: (any MTLCommandBuffer) -> Finished?) -> Bool {
        guard let commandBuffer = sliceQueue.makeCommandBuffer() else { return false }
        commandBuffer.label = "RAD exact slice"
        commandBuffer.encodeWaitForEvent(frameEvent, value: frameSerial)
        guard let record = body(commandBuffer) else { return false }
        sliceSerial += 1
        let serial = sliceSerial
        commandBuffer.encodeSignalEvent(sliceEvent, value: serial)
        let trace = ViewportFrameTrace.shared, committed = ViewportFrameTrace.now
        commandBuffer.addCompletedHandler { [weak self] buffer in
            // Metal also calls completed handlers when it releases a command
            // buffer that was never committed, such as a dropped chained slice.
            guard buffer.status == .completed || buffer.status == .error else { return }
            if ViewportFrameTrace.enabled {
                var covered = 0
                if case .slice(_, _, let count) = record { covered = count }
                trace.addSlice(.init(job: ObjectIdentifier(job), mask: { if case .mask = record { return true }; return false }(),
                                     covered: covered, committed: committed, gpuStart: buffer.gpuStartTime,
                                     gpuEnd: buffer.gpuEndTime))
            }
            guard let self else { return }
            self.condition.lock()
            let seconds = max(0, buffer.gpuEndTime - max(buffer.gpuStartTime, self.sliceGPUEnd))
            self.sliceGPUEnd = max(self.sliceGPUEnd, buffer.gpuEndTime)
            var result: Finished
            switch record {
            case .slice(let owner, _, let covered): result = .slice(owner, seconds: seconds, covered: covered)
            case .mask, .struck: result = record
            }
            if buffer.status == .error || seconds > CADExactCost.failingSliceSeconds {
                // The slice still counts; a failed one leaves its rows cleared.
                self.finished.append(result)
                result = .struck(job)
            }
            self.slicesRunning -= 1
            self.finished.append(result)
            let next = self.chained
            if let next {
                self.chained = nil
                self.slicesRunning += 1
                self.committedSliceSerial = max(self.committedSliceSerial, next.serial)
            }
            self.condition.signal()
            self.condition.unlock()
            next?.buffer.commit()
        }
        let commitNow: Bool = condition.withLock {
            if chain, slicesRunning > 0 {
                chained = (job, commandBuffer, serial)
                return false
            }
            slicesRunning += 1
            committedSliceSerial = max(committedSliceSerial, serial)
            return true
        }
        if commitNow { commandBuffer.commit() }
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
        commandBuffer.encodeWaitForEvent(sliceEvent, value: committedSerial())
        passes.prepareMasks(job, frame: frame)
        let pendingMasks = job.masks.values.filter { !$0.cleared || !$0.plan.isDone }
        let maskCost = pendingMasks.reduce(0) { $0 + $1.plan.remaining }
        if !pendingMasks.isEmpty,
           CADExactCost.shared.sliceRate(job.costKey) * Double(maskCost) <= CADViewportPasses.frameBudget {
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
    /// A frame that also drew `geometry` fails it when the GPU fails the frame.
    private func commit(_ commandBuffer: any MTLCommandBuffer, drawable: any CAMetalDrawable, request: Request, kind: String,
                        outcome: Outcome, geometry: CADExactJob? = nil, measured: (@Sendable (Double) -> Void)? = nil) {
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
        commandBuffer.addCompletedHandler { [weak self] buffer in
            inFlight.signal()
            if buffer.status == .error, let geometry, let self {
                self.condition.withLock {
                    self.finished.append(.struck(geometry))
                    self.condition.signal()
                }
            }
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
