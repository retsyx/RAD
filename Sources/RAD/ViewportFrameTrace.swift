import Foundation
import QuartzCore

/// Temporary diagnostic: per-frame timestamps of the live viewport, recorded
/// only while `enabled` is set by a measurement test.
final class ViewportFrameTrace: @unchecked Sendable {
    struct Frame: Sendable {
        var generation: UInt64
        /// mesh, stand-in, single (exact geometry and composite) or composite.
        var kind = ""
        var exact = false
        var pixels = 0
        var published: Double = 0
        var drawStart: Double = 0
        var afterInFlight: Double = 0
        var afterDrawable: Double = 0
        var afterEncode: Double = 0
        var committed: Double = 0
        var gpuStart: Double = 0
        var gpuEnd: Double = 0
        var completedOnMain: Double = 0
        var presented: Double = 0
    }

    /// One exact slice command buffer (geometry or outline coverage).
    struct Slice: Sendable {
        var job: ObjectIdentifier
        var mask: Bool
        var covered: Int
        var committed: Double
        var gpuStart: Double
        var gpuEnd: Double
    }

    static let shared = ViewportFrameTrace()
    private let lock = NSLock()
    private var frames: [UInt64: Frame] = [:]
    private var slices: [Slice] = []
    nonisolated(unsafe) static var enabled = false

    func update(_ generation: UInt64, _ change: (inout Frame) -> Void) {
        guard Self.enabled else { return }
        lock.withLock {
            var frame = frames[generation] ?? Frame(generation: generation)
            change(&frame)
            frames[generation] = frame
        }
    }

    func addSlice(_ slice: Slice) {
        guard Self.enabled else { return }
        lock.withLock { slices.append(slice) }
    }

    func take() -> [Frame] {
        lock.withLock {
            let result = frames.values.sorted { $0.generation < $1.generation }
            frames.removeAll()
            return result
        }
    }

    func takeSlices() -> [Slice] {
        lock.withLock {
            let result = slices
            slices.removeAll()
            return result
        }
    }

    static var now: Double { CACurrentMediaTime() }
}
