import AppKit
import Metal
import QuartzCore
import SceneKit
import simd
import XCTest
@testable import RAD

/// Renders and times built designs through the viewport's capture path, as
/// described by a JSON scenario (RAD_EXACT_SCENARIO). A measurement tool for
/// exact faces, skipped in ordinary test runs.
@MainActor
final class ExactFacesScenarioTests: XCTestCase {
    struct Scenario: Decodable {
        struct Render: Decodable {
            var name: String
            var view: String?
            var position: [Double]?
            var target: [Double]?
            var zoom: Double?
            /// Visible half-height at the target in millimetres (sets zoom).
            var halfHeight: Double?
            var projection: String?
            var display: String?
            var width: Int?
            var height: Int?
            var scale: Double?
            /// false renders display meshes (the build's proxies removed).
            var exact: Bool?
            var debug: Int?
            var selected: [Int]?
            var hidden: [Int]?
            /// GPU frames to time (median reported); 0 renders once.
            var timing: Int?
            /// Sliced captures to time (GPU busy, the union of their command
            /// buffers' execution; the minimum is reported).
            var slicedTiming: Int?
        }
        var design: String
        var output: String
        var renders: [Render]
    }

    func testScenario() async throws {
        guard let path = ProcessInfo.processInfo.environment["RAD_EXACT_SCENARIO"] else {
            throw XCTSkip("Set RAD_EXACT_SCENARIO to a scenario JSON file to render and time exact faces.")
        }
        let scenario = try JSONDecoder().decode(Scenario.self, from: Data(contentsOf: URL(fileURLWithPath: path)))
        let output = URL(fileURLWithPath: scenario.output)
        try FileManager.default.createDirectory(at: output, withIntermediateDirectories: true)
        let spec = try JSONDecoder().decode(ModelSpec.self, from: Data(contentsOf: URL(fileURLWithPath: scenario.design)))
        let started = Date()
        let preview = try await EngineService().buildPreview(spec, python: EngineService.defaultPython, threads: ProcessInfo.processInfo.activeProcessorCount)
        var log = ["build \(String(format: "%.2f", Date().timeIntervalSince(started))) s, \(preview.triangleCount) triangles, "
                   + "\(preview.proxySolids?.count ?? 0) proxy triangles, program \(preview.fieldProgram != nil)"]
        for part in preview.proxy?.parts ?? [] {
            log.append("  part \(part.solid): \(part.status) \(part.triangles ?? 0) shell \(part.shell ?? 0) "
                       + "dev \(part.maxDeviation ?? 0) \(part.reason ?? "")")
        }
        log.append("  offset \(preview.proxy?.offset ?? 0) cell \(preview.proxy?.cellSize ?? 0)")
        var meshOnly = preview
        meshOnly.proxy = nil
        for render in scenario.renders {
            let width = render.width ?? 1024, height = render.height ?? 768
            let scale = render.scale ?? 1
            let surface = CADSceneView(frame: NSRect(x: 0, y: 0, width: Double(width) / scale, height: Double(height) / scale))
            surface.preparesRendererResources = false
            surface.fieldDebugMode = render.debug ?? 0
            let exact = render.exact ?? true
            surface.update(mesh: exact ? preview : meshOnly, mode: ViewportDisplayMode(rawValue: render.display ?? "edges") ?? .edges,
                           grid: false, command: nil, selectedSolids: Set(render.selected ?? []),
                           hiddenSolids: Set(render.hidden ?? []),
                           projection: ViewportProjectionMode(rawValue: render.projection ?? "perspective") ?? .perspective,
                           solidTransforms: preview.assemblyPose?.solidTransforms ?? [:])
            _ = try await surface.waitForCaptureResources(until: .now + .seconds(120))
            var camera = CADAgentCamera()
            camera.position = render.position
            camera.target = render.target
            camera.zoom = render.zoom ?? 1
            try surface.configureAgentView(named: render.view ?? "front", camera: camera)
            if let half = render.halfHeight, let position = render.position, let target = render.target {
                // Perspective zoom dollies toward the target; orthographic zoom scales.
                let distance = simd_distance(SIMD3(position[0], position[1], position[2]), SIMD3(target[0], target[1], target[2]))
                let base = (surface.agentCaptureMetadata["usesOrthographicProjection"] as? Bool) == true
                    ? (surface.agentCaptureMetadata["orthographicScale"] as? Double ?? half) : distance * tan(19 * Double.pi / 180)
                camera.zoom = base / half
                try surface.configureAgentView(named: render.view ?? "front", camera: camera)
            }
            surface.layout()
            _ = try await surface.waitForCaptureResources(until: .now + .seconds(120))
            let capture = try CADViewportCapture(view: surface, width: width, height: height)
            let image = try capture.image()
            let png = try CADAgentRenderer.encodePNG(image)
            try png.write(to: output.appendingPathComponent(render.name + ".png"))
            var line = "\(render.name): exact solids \(surface.exactSolids.count), field \(capture.frame.field != nil)"
            if let frames = render.timing, frames > 0 {
                let times = try Self.time(capture, frames: frames)
                line += String(format: ", GPU median %.2f ms (min %.2f, max %.2f)", times[times.count / 2], times.first!, times.last!)
            }
            if let count = render.slicedTiming, count > 0 {
                var busy: [Double] = [], buffers = 0
                for _ in 0..<count {
                    var statistics = CADViewportCapture.Statistics()
                    _ = try CADViewportCapture.image(of: capture.frame, source: capture.source, statistics: &statistics)
                    busy.append(statistics.gpuSeconds * 1000)
                    buffers = statistics.commandBuffers
                }
                line += String(format: ", sliced GPU busy min %.2f ms (%d command buffers)", busy.min()!, buffers)
            }
            log.append(line)
        }
        try log.joined(separator: "\n").write(to: output.appendingPathComponent("log.txt"), atomically: true, encoding: .utf8)
        print(log.joined(separator: "\n"))
    }

    /// Sorted GPU milliseconds of `frames` frames after three warm-up frames.
    static func time(_ capture: CADViewportCapture, frames: Int) throws -> [Double] {
        let device = try XCTUnwrap(MTLCreateSystemDefaultDevice())
        let queue = try XCTUnwrap(device.makeCommandQueue())
        let renderer = SCNRenderer(device: device, options: nil)
        renderer.scene = capture.source.scene
        renderer.pointOfView = capture.source.camera
        renderer.usesReverseZ = true
        renderer.autoenablesDefaultLighting = false
        XCTAssertTrue(renderer.prepare(capture.source.scene, shouldAbortBlock: nil))
        let passes = try CADViewportPasses(device: device)
        let descriptor = MTLTextureDescriptor.texture2DDescriptor(pixelFormat: CADViewportPasses.colorFormat,
                                                                  width: capture.width, height: capture.height, mipmapped: false)
        descriptor.usage = [.renderTarget, .shaderRead]
        descriptor.storageMode = .private
        let target = try XCTUnwrap(device.makeTexture(descriptor: descriptor))
        var times: [Double] = []
        for index in 0..<(frames + 3) {
            let buffer = try XCTUnwrap(queue.makeCommandBuffer())
            passes.encode(capture.frame, source: capture.source, renderer: renderer, into: target, commandBuffer: buffer)
            buffer.commit()
            buffer.waitUntilCompleted()
            if index >= 3 { times.append((buffer.gpuEndTime - buffer.gpuStartTime) * 1000) }
        }
        return times.sorted()
    }
}
