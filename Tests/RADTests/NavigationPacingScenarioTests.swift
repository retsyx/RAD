import AppKit
import QuartzCore
import XCTest
@testable import RAD

/// Measurement tool (RAD_NAV_SCENARIO): replays orbit, hover and zoom input in
/// a live viewport window and records each frame's timeline.
@MainActor
final class NavigationPacingScenarioTests: XCTestCase {
    struct Scenario: Decodable {
        var designs: [String]
        var output: String
        var sizes: [[Double]]
        var mode: String?
        var phases: [String]?
    }

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
                let result = try await measure(preview: preview, name: name, width: size[0], height: size[1],
                                               mode: ViewportDisplayMode(rawValue: scenario.mode ?? "shaded") ?? .shaded,
                                               phases: scenario.phases ?? ["hover-orbit", "drag-release", "wheel", "trackpad"])
                var entry = result
                entry["design"] = name
                entry["build"] = buildSeconds
                entry["triangles"] = preview.triangleCount
                report.append(entry)
            }
        }
        let data = try JSONSerialization.data(withJSONObject: report, options: [.prettyPrinted, .sortedKeys])
        try data.write(to: output.appendingPathComponent("report.json"))
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
        func settle() async throws {
            let deadline = ContinuousClock.now + .seconds(10)
            while view.hasPendingFrames, ContinuousClock.now < deadline { try await Task.sleep(for: .milliseconds(5)) }
        }
        try await settle()
        try await Task.sleep(for: .milliseconds(500))
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
        var results: [String: Any] = ["size": [width, height], "pixels": Int(view.convertToBacking(view.bounds).size.width * view.convertToBacking(view.bounds).size.height)]
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
        for phase in phases {
            _ = ViewportFrameTrace.shared.take()
            heartbeats.removeAll()
            handlerTimes.removeAll()
            ViewportFrameTrace.enabled = true
            let policyBefore = (view.navigationPolicy.exactFrames, view.navigationPolicy.meshFrames)
            let t0 = CACurrentMediaTime()
            var navigationStart = t0, navigationEnd = t0
            switch phase {
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
                    try await Task.sleep(for: .milliseconds(150))
                }
                navigationEnd = CACurrentMediaTime()
            case "wheel":
                // A mouse wheel: line ticks without phases, in bursts with pauses.
                navigationStart = CACurrentMediaTime()
                for burst in 0..<4 {
                    for tick in 0..<5 {
                        let event = CGEvent(scrollWheelEvent2Source: nil, units: .line, wheelCount: 1,
                                            wheel1: burst < 2 ? 1 : -1, wheel2: 0, wheel3: 0)!
                        event.location = window.convertPoint(toScreen: view.convert(center, to: nil))
                        let ns = NSEvent(cgEvent: event)!
                        timed { view.scrollWheel(with: ns) }
                        _ = tick
                        try await Task.sleep(for: .milliseconds(45))
                    }
                    try await Task.sleep(for: .milliseconds(350))
                }
                navigationEnd = CACurrentMediaTime()
            case "trackpad":
                // Two-finger scrolling with phases at 120 Hz, in and out.
                navigationStart = CACurrentMediaTime()
                for direction in [1, -1] {
                    let start = CACurrentMediaTime()
                    for i in 0..<120 {
                        let event = CGEvent(scrollWheelEvent2Source: nil, units: .pixel, wheelCount: 1,
                                            wheel1: Int32(direction * 6), wheel2: 0, wheel3: 0)!
                        event.setIntegerValueField(.scrollWheelEventIsContinuous, value: 1)
                        event.setIntegerValueField(.scrollWheelEventScrollPhase, value: i == 0 ? 1 : (i == 119 ? 4 : 2))
                        event.location = window.convertPoint(toScreen: view.convert(center, to: nil))
                        let ns = NSEvent(cgEvent: event)!
                        timed { view.scrollWheel(with: ns) }
                        try await pace(i, start, 120)
                    }
                    try await Task.sleep(for: .milliseconds(200))
                }
                navigationEnd = CACurrentMediaTime()
            default: continue
            }
            try await Task.sleep(for: .milliseconds(400))
            try await settle()
            try await Task.sleep(for: .milliseconds(300))
            ViewportFrameTrace.enabled = false
            let trace = ViewportFrameTrace.shared.take()
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
                ["g": f.generation, "exact": f.exact, "px": f.pixels, "pub": f.published, "start": f.drawStart,
                 "inflight": f.afterInFlight, "drawable": f.afterDrawable, "encoded": f.afterEncode,
                 "committed": f.committed, "gpuStart": f.gpuStart, "gpuEnd": f.gpuEnd, "main": f.completedOnMain,
                 "presented": f.presented]
            }
            results[phase] = entry
            print("NAV \(name) \(Int(width))x\(Int(height)) \(phase): frames \(trace.count) exact \(entry["exactFrames"]!) mesh \(entry["meshFrames"]!) main max gap \(String(format: "%.1f", (gaps.last ?? 0) * 1000)) ms")
        }
        return results
    }
}
