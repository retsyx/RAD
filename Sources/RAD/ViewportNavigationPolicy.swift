import Foundation

/// Chooses exact or display-mesh faces for each published viewport frame.
/// Frames at rest show exact faces once their layers are drawn. While the
/// user navigates (orbit, pan, zoom and orientation transitions), a design
/// whose exact frames would miss the display's frame budget draws its display
/// meshes instead; the end of navigation publishes the state again. The
/// estimate is the largest measured cost since the design's exact resources
/// or the frame size changed, so a cheap view never flips a gesture to exact
/// frames that a costlier view then stalls. Captures are always exact.
struct ViewportNavigationPolicy: Equatable, Sendable {
    enum Source: Hashable, Sendable {
        /// Right-drag orbit or pan.
        case drag
        /// Middle-drag pan.
        case pan
        /// Trackpad scrolling with phases, including momentum.
        case scroll
        /// Wheel events without phases; a single deferred publish ends them.
        case wheel
        case magnify
        /// Orientation cube transitions.
        case transition
    }

    /// What the measured costs belong to.
    struct Context: Equatable, Sendable {
        var resources: ObjectIdentifier?
        var pixels: Int
    }

    /// The share of the display's frame interval an exact frame may use.
    static let budgetFraction = 0.8
    /// Wheel scrolling has no end event: it ends this long after its last event.
    static let wheelSettle: Duration = .milliseconds(200)

    private(set) var active: Set<Source> = []
    private(set) var context: Context?
    /// The largest GPU seconds per output pixel of an exact frame's geometry
    /// and composite measured in this context.
    private(set) var exactSecondsPerPixel: Double?
    /// Completed frames by kind (diagnostics and measurements).
    private(set) var exactFrames = 0
    private(set) var meshFrames = 0

    var isNavigating: Bool { !active.isEmpty }

    mutating func begin(_ source: Source) { active.insert(source) }

    /// Returns true when this ends all navigation.
    @discardableResult
    mutating func end(_ source: Source) -> Bool {
        active.remove(source) != nil && active.isEmpty
    }

    /// Ends every source, such as when the window resigns key. Returns true
    /// when navigation was active.
    @discardableResult
    mutating func endAll() -> Bool {
        let wasNavigating = isNavigating
        active.removeAll()
        return wasNavigating
    }

    /// Forgets the estimate when the exact resources or the frame size change.
    mutating func setContext(resources: ObjectIdentifier?, pixels: Int) {
        let next = Context(resources: resources, pixels: pixels)
        guard next != context else { return }
        context = next
        exactSecondsPerPixel = nil
    }

    /// Whether an exact frame of `pixels` fits the budget of a frame interval
    /// of `interval` seconds by the largest measured cost; unmeasured designs
    /// do not.
    func exactFits(pixels: Int, interval: Double) -> Bool {
        guard let rate = exactSecondsPerPixel, pixels > 0, interval > 0 else { return false }
        return rate * Double(pixels) <= interval * Self.budgetFraction
    }

    func drawsExactFaces(pixels: Int, interval: Double) -> Bool {
        !isNavigating || exactFits(pixels: pixels, interval: interval)
    }

    /// Records a completed frame. `geometry` exact frames drew their exact
    /// layers (`gpuSeconds` covers them and the composite); others only
    /// composited stored layers and do not change the estimate.
    mutating func record(exact: Bool, gpuSeconds: Double, pixels: Int, geometry: Bool = true) {
        if exact {
            exactFrames += 1
            if geometry, gpuSeconds.isFinite, gpuSeconds > 0, pixels > 0 {
                exactSecondsPerPixel = max(exactSecondsPerPixel ?? 0, gpuSeconds / Double(pixels))
            }
        } else {
            meshFrames += 1
        }
    }
}
