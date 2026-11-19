import Foundation

/// Chooses exact or display-mesh faces for each published viewport frame.
/// Frames at rest are exact. While the user navigates (orbit, pan, zoom and
/// orientation transitions), a design whose measured exact frames would miss
/// the display's frame budget draws its display meshes instead; the end of
/// navigation publishes an exact frame. Captures are always exact.
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

    /// The share of the display's frame interval an exact frame may use.
    static let budgetFraction = 0.8
    /// Wheel scrolling has no end event: it ends this long after its last event.
    static let wheelSettle: Duration = .milliseconds(200)

    private(set) var active: Set<Source> = []
    /// GPU seconds per output pixel of the newest completed exact frame.
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

    /// Whether an exact frame of `pixels` would exceed the budget of a frame
    /// interval of `interval` seconds, judged by the newest exact frame.
    func wouldMiss(pixels: Int, interval: Double) -> Bool {
        guard let rate = exactSecondsPerPixel, pixels > 0, interval > 0 else { return false }
        return rate * Double(pixels) > interval * Self.budgetFraction
    }

    func drawsExactFaces(pixels: Int, interval: Double) -> Bool {
        !isNavigating || !wouldMiss(pixels: pixels, interval: interval)
    }

    /// Records a completed frame's GPU time.
    mutating func record(exact: Bool, gpuSeconds: Double, pixels: Int) {
        if exact {
            exactFrames += 1
            if gpuSeconds.isFinite, gpuSeconds > 0, pixels > 0 { exactSecondsPerPixel = gpuSeconds / Double(pixels) }
        } else {
            meshFrames += 1
        }
    }
}
