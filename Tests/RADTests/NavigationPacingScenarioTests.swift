import AppKit
import QuartzCore
import XCTest
@testable import RAD

/// Measurement tool (RAD_NAV_SCENARIO): replays orbit, hover and zoom input in
/// a live viewport window and records each frame's and exact slice's timeline.
/// The scenario JSON names `designs` (build specs), `output`, window `sizes`
/// in points, display `modes` and `phases`.
@MainActor
final class NavigationPacingScenarioTests: XCTestCase {
    struct Scenario: Decodable {
        var designs: [String]
        var output: String
        var sizes: [[Double]]
        var modes: [String]?
        var phases: [String]?
    }

    static let allPhases = ["hover-rest", "hover-orbit", "drag-release", "wheel", "trackpad"]

    func testNavigationPacing() async throws {
        guard let path = ProcessInfo.processInfo.environment["RAD_NAV_SCENARIO"] else {
            throw XCTSkip("Set RAD_NAV_SCENARIO to measure live navigation pacing.")
        }
        guard !NSScreen.screens.isEmpty else { throw XCTSkip("A window server is required.") }
        let scenario = try JSONDecoder().decode(Scenario.self, from: Data(contentsOf: URL(fileURLWithPath: path)))
        let output = URL(fileURLWithPath: scenario.output)
        try FileManager.default.createDirectory(at: output, withIntermediateDirectories: true)
        let application = NSApplication.shared
        _ = application.setActivationPolicy(.accessory)
        application.finishLaunching()
        var report: [[String: Any]] = []
        for designPath in scenario.designs {
            let spec = try JSONDecoder().decode(ModelSpec.self, from: Data(contentsOf: URL(fileURLWithPath: designPath)))
            let started = Date()
            let preview = try await EngineService().buildPreview(spec, python: EngineService.defaultPython,
                                                                 threads: ProcessInfo.processInfo.activeProcessorCount)
            let buildSeconds = Date().timeIntervalSince(started)
            let name = URL(fileURLWithPath: designPath).deletingPathExtension().lastPathComponent
            print("NAV \(name): build \(String(format: "%.1f", buildSeconds)) s, \(preview.triangleCount) triangles, proxy \(preview.proxySolids?.count ?? 0)")
            for size in scenario.sizes {
                for mode in scenario.modes ?? ["shaded"] {
                    var entry = try await measure(preview: preview, name: name, width: size[0], height: size[1],
                                                  mode: ViewportDisplayMode(rawValue: mode) ?? .shaded,
                                                  phases: scenario.phases ?? Self.allPhases)
                    entry["design"] = name
                    entry["mode"] = mode
                    entry["build"] = buildSeconds
                    entry["triangles"] = preview.triangleCount
                    report.append(entry)
                    let data = try JSONSerialization.data(withJSONObject: report, options: [.prettyPrinted, .sortedKeys])
                    try data.write(to: output.appendingPathComponent("report.json"))
                }
            }
        }
    }

    private func measure(preview: PreviewMesh, name: String, width: Double, height: Double,
                         mode: ViewportDisplayMode, phases: [String]) async throws -> [String: Any] {
        let view = CADSceneView(frame: NSRect(x: 0, y: 0, width: width, height: height))
        let frames = ViewportRenderCounter(view)
        let window = NSWindow(contentRect: NSRect(x: 40, y: 40, width: width, height: height), styleMask: [.titled],
                              backing: .buffered, defer: false)
        window.isReleasedWhenClosed = false
        window.animationBehavior = .none
        window.contentView = view
        defer { window.orderOut(nil); window.contentView = nil; window.close() }
        view.update(mesh: preview, mode: mode, grid: true, command: nil, selectedSolids: [], hiddenSolids: [],
                    solidTransforms: preview.assemblyPose?.solidTransforms ?? [:])
        window.orderFrontRegardless()
        view.layout()
        _ = try await view.waitForCaptureResources(until: .now + .seconds(120))
        func settle(_ limit: Duration = .seconds(30)) async throws {
            let deadline = ContinuousClock.now + limit
            while view.hasPendingFrames, ContinuousClock.now < deadline { try await Task.sleep(for: .milliseconds(5)) }
        }
        // The first exact image of the design.
        ViewportFrameTrace.enabled = true
        let firstStart = CACurrentMediaTime()
        try await settle(.seconds(120))
        let firstExact = CACurrentMediaTime() - firstStart
        ViewportFrameTrace.enabled = false
        _ = ViewportFrameTrace.shared.take()
        _ = ViewportFrameTrace.shared.takeSlices()
        try await Task.sleep(for: .milliseconds(300))
        try await settle()
        _ = frames
        let center = NSPoint(x: width / 2, y: height / 2)
        func mouse(_ type: NSEvent.EventType, _ point: NSPoint) -> NSEvent {
            NSEvent.mouseEvent(with: type, location: view.convert(point, to: nil), modifierFlags: [],
                               timestamp: ProcessInfo.processInfo.systemUptime, windowNumber: window.windowNumber,
                               context: nil, eventNumber: 1, clickCount: 1, pressure: 0)!
        }
        // Main-thread responsiveness: gaps between 1 ms heartbeats.
        var heartbeats: [Double] = []
        let heartbeat = Timer(timeInterval: 0.001, repeats: true) { _ in heartbeats.append(CACurrentMediaTime()) }
        RunLoop.main.add(heartbeat, forMode: .common)
        defer { heartbeat.invalidate() }
        let backing = view.convertToBacking(view.bounds).size
        var results: [String: Any] = ["size": [width, height], "pixels": Int(backing.width * backing.height),
                                      "firstExactSeconds": firstExact]
        var handlerTimes: [Double] = []
        func timed(_ block: () -> Void) {
            let t = CACurrentMediaTime()
            block()
            handlerTimes.append(CACurrentMediaTime() - t)
        }
        func pace(_ index: Int, _ start: Double, _ rate: Double) async throws {
            let next = start + Double(index + 1) / rate
            let wait = next - CACurrentMediaTime()
            if wait > 0 { try await Task.sleep(for: .microseconds(Int(wait * 1_000_000))) } else { await Task.yield() }
        }
        func scroll(_ delta: Double, phase: Int64, momentum: Int64) {
            let event = CGEvent(scrollWheelEvent2Source: nil, units: .pixel, wheelCount: 1, wheel1: 0, wheel2: 0, wheel3: 0)!
            event.setIntegerValueField(.scrollWheelEventIsContinuous, value: 1)
            event.setDoubleValueField(.scrollWheelEventFixedPtDeltaAxis1, value: delta)
            event.setDoubleValueField(.scrollWheelEventPointDeltaAxis1, value: delta)
            event.setIntegerValueField(.scrollWheelEventScrollPhase, value: phase)
            event.setIntegerValueField(.scrollWheelEventMomentumPhase, value: momentum)
            event.location = window.convertPoint(toScreen: view.convert(center, to: nil))
            let ns = NSEvent(cgEvent: event)!
            timed { view.scrollWheel(with: ns) }
        }
        for phase in phases {
            _ = ViewportFrameTrace.shared.take()
            _ = ViewportFrameTrace.shared.takeSlices()
            heartbeats.removeAll()
            handlerTimes.removeAll()
            ViewportFrameTrace.enabled = true
            let policyBefore = (view.navigationPolicy.exactFrames, view.navigationPolicy.meshFrames)
            let t0 = CACurrentMediaTime()
            var navigationStart = t0, navigationEnd = t0
            switch phase {
            case "hover-rest":
                // Hover across the design at rest: composites of stored layers.
                for i in 0..<60 {
                    let p = NSPoint(x: center.x - 120 + Double(i) * 4, y: center.y + 40 * sin(Double(i) / 6))
                    timed { view.mouseMoved(with: mouse(.mouseMoved, p)) }
                    try await pace(i, t0, 60)
                }
                navigationEnd = CACurrentMediaTime()
            case "hover-orbit":
                // Hover across the model at 60 Hz, then orbit at 120 Hz.
                for i in 0..<45 {
                    let p = NSPoint(x: center.x - 150 + Double(i) * 6, y: center.y + 30 * sin(Double(i) / 5))
                    timed { view.mouseMoved(with: mouse(.mouseMoved, p)) }
                    try await pace(i, t0, 60)
                }
                navigationStart = CACurrentMediaTime()
                timed { view.rightMouseDown(with: mouse(.rightMouseDown, center)) }
                let start = CACurrentMediaTime()
                for i in 0..<300 {
                    let t = Double(i) / 120
                    let p = NSPoint(x: center.x + 220 * sin(2 * .pi * t / 2.5), y: center.y + 70 * sin(2 * .pi * t / 1.7))
                    timed { view.rightMouseDragged(with: mouse(.rightMouseDragged, p)) }
                    try await pace(i, start, 120)
                }
                timed { view.rightMouseUp(with: mouse(.rightMouseUp, center)) }
                navigationEnd = CACurrentMediaTime()
            case "drag-release":
                // Short drags with pauses shorter than heavy designs' exact
                // layers (abandoned in progress), then a final release.
                navigationStart = CACurrentMediaTime()
                for drag in 0..<5 {
                    timed { view.rightMouseDown(with: mouse(.rightMouseDown, center)) }
                    let start = CACurrentMediaTime()
                    for i in 0..<36 {
                        let p = NSPoint(x: center.x + Double(i) * 5 * (drag % 2 == 0 ? 1 : -1), y: center.y + Double(i))
                        timed { view.rightMouseDragged(with: mouse(.rightMouseDragged, p)) }
                        try await pace(i, start, 120)
                    }
                    timed { view.rightMouseUp(with: mouse(.rightMouseUp, center)) }
                    navigationEnd = CACurrentMediaTime()
                    if drag < 4 { try await Task.sleep(for: .milliseconds(150)) }
                }
            case "wheel":
                // A mouse wheel: line ticks without phases, in bursts with
                // pauses longer than the settle delay.
                navigationStart = CACurrentMediaTime()
                for burst in 0..<4 {
                    for _ in 0..<5 {
                        let event = CGEvent(scrollWheelEvent2Source: nil, units: .line, wheelCount: 1,
                                            wheel1: burst < 2 ? 1 : -1, wheel2: 0, wheel3: 0)!
                        event.location = window.convertPoint(toScreen: view.convert(center, to: nil))
                        let ns = NSEvent(cgEvent: event)!
                        timed { view.scrollWheel(with: ns) }
                        try await Task.sleep(for: .milliseconds(45))
                    }
                    navigationEnd = CACurrentMediaTime() + ViewportNavigationPolicy.wheelSettle.seconds
                    if burst < 3 { try await Task.sleep(for: .milliseconds(350)) }
                }
            case "trackpad":
                // Two-finger scrolling at 120 Hz with momentum: about 2x in,
                // then back out.
                navigationStart = CACurrentMediaTime()
                for direction in [1.0, -1.0] {
                    var start = CACurrentMediaTime()
                    for i in 0..<60 {
                        scroll(direction * 1.2, phase: i == 0 ? 1 : 2, momentum: 0)
                        try await pace(i, start, 120)
                    }
                    scroll(0, phase: 4, momentum: 0)
                    start = CACurrentMediaTime()
                    for i in 0..<40 {
                        scroll(direction * 1.2 * Double(40 - i) / 40, phase: 0, momentum: i == 0 ? 1 : 2)
                        try await pace(i, start, 120)
                    }
                    scroll(0, phase: 0, momentum: 3)
                    navigationEnd = CACurrentMediaTime()
                    if direction > 0 { try await Task.sleep(for: .milliseconds(200)) }
                }
            default: continue
            }
            try await Task.sleep(for: .milliseconds(400))
            try await settle()
            try await Task.sleep(for: .milliseconds(300))
            ViewportFrameTrace.enabled = false
            let trace = ViewportFrameTrace.shared.take()
            let slices = ViewportFrameTrace.shared.takeSlices()
            let gaps = zip(heartbeats.dropFirst(), heartbeats).map { $0 - $1 }.sorted()
            let policyAfter = (view.navigationPolicy.exactFrames, view.navigationPolicy.meshFrames)
            var entry: [String: Any] = [
                "t0": t0, "navigationStart": navigationStart, "navigationEnd": navigationEnd,
                "exactFrames": policyAfter.0 - policyBefore.0, "meshFrames": policyAfter.1 - policyBefore.1,
                "mainMaxGapMs": (gaps.last ?? 0) * 1000,
                "mainGapsOver20ms": gaps.filter { $0 > 0.020 }.count,
                "handlerMaxMs": (handlerTimes.max() ?? 0) * 1000,
                "handlerMeanMs": handlerTimes.isEmpty ? 0 : handlerTimes.reduce(0, +) / Double(handlerTimes.count) * 1000,
                "events": handlerTimes.count,
            ]
            entry["frames"] = trace.map { f -> [String: Any] in
                ["g": f.generation, "kind": f.kind, "exact": f.exact, "px": f.pixels, "pub": f.published, "start": f.drawStart,
                 "inflight": f.afterInFlight, "drawable": f.afterDrawable, "encoded": f.afterEncode,
                 "committed": f.committed, "gpuStart": f.gpuStart, "gpuEnd": f.gpuEnd, "main": f.completedOnMain,
                 "presented": f.presented]
            }
            entry["slices"] = slices.map { s -> [String: Any] in
                ["job": s.job.hashValue, "mask": s.mask, "covered": s.covered, "committed": s.committed,
                 "gpuStart": s.gpuStart, "gpuEnd": s.gpuEnd]
            }
            results[phase] = entry
            print("NAV \(name) \(Int(width))x\(Int(height)) \(mode.rawValue) \(phase): frames \(trace.count) slices \(slices.count) exact \(entry["exactFrames"]!) mesh \(entry["meshFrames"]!) main max gap \(String(format: "%.1f", (gaps.last ?? 0) * 1000)) ms")
        }
        if let statistics = view.exactStatistics {
            results["statistics"] = ["geometrySlices": statistics.geometrySlices, "maskSlices": statistics.maskSlices,
                                     "composites": statistics.composites, "storedBytes": statistics.storedBytes]
        }
        return results
    }
}

private extension Duration {
    var seconds: Double { Double(components.seconds) + Double(components.attoseconds) * 1e-18 }
}
