import AppKit
import Metal
import SceneKit
import simd
import XCTest
@testable import RAD

/// Exact faces: field parts drawn from their proxies by per-pixel marches to
/// the field's crossing (SDF-DESIGN.md, D6). Engine-backed tests build small
/// designs; the navigation policy tests need no engine.
@MainActor
final class ExactFacesTests: XCTestCase {
    // MARK: Fixtures

    private static var previews: [String: PreviewMesh] = [:]

    private func preview(_ script: String) async throws -> PreviewMesh {
        guard ProcessInfo.processInfo.environment["RAD_RUN_ENGINE_TESTS"] == "1" else {
            throw XCTSkip("Set RAD_RUN_ENGINE_TESTS=1 to build designs with exact faces.")
        }
        guard MTLCreateSystemDefaultDevice() != nil else { throw XCTSkip("No Metal device.") }
        if let preview = Self.previews[script] { return preview }
        let preview = try await EngineService().buildPreview(
            ModelSpec(title: "Exact faces", summary: "Exact face test.", units: "mm", script: script),
            python: EngineService.defaultPython, threads: 4)
        XCTAssertNotNil(preview.fieldProgram)
        XCTAssertTrue(preview.proxy?.parts?.allSatisfy { $0.status == "proxy" } == true, "every part has a proxy")
        Self.previews[script] = preview
        return preview
    }

    struct Pixels {
        let width: Int
        let height: Int
        private let bytes: [UInt8]

        init(_ image: CGImage) {
            width = image.width
            height = image.height
            var data = [UInt8](repeating: 0, count: width * height * 4)
            data.withUnsafeMutableBytes { raw in
                let context = CGContext(data: raw.baseAddress, width: image.width, height: image.height, bitsPerComponent: 8,
                                        bytesPerRow: image.width * 4, space: CGColorSpace(name: CGColorSpace.sRGB)!,
                                        bitmapInfo: CGImageAlphaInfo.premultipliedLast.rawValue)!
                context.draw(image, in: CGRect(x: 0, y: 0, width: image.width, height: image.height))
            }
            bytes = data
        }

        /// sRGB components 0...1 of the pixel at column x, row y (top-left origin).
        func rgb(_ x: Int, _ y: Int) -> SIMD3<Double> {
            let base = (y * width + x) * 4
            return SIMD3(Double(bytes[base]), Double(bytes[base + 1]), Double(bytes[base + 2])) / 255
        }

        var isBlank: Bool {
            let first = rgb(0, 0)
            return (0..<height).allSatisfy { y in (0..<width).allSatisfy { simd_reduce_max(abs(rgb($0, y) - first)) < 0.01 } }
        }
    }

    struct Shot {
        let pixels: Pixels
        let metadata: [String: Any]
        let exactSolids: Int
    }

    private func render(_ preview: PreviewMesh, exact: Bool = true, display: ViewportDisplayMode = .shaded,
                        projection: ViewportProjectionMode = .isometric, view: String = "front",
                        camera: CADAgentCamera = .init(), width: Int = 480, height: Int = 360, scale: Double = 1,
                        selected: Set<Int> = [], colors: [Int: PartColor] = [:]) async throws -> Shot {
        var mesh = preview
        if !exact { mesh.proxy = nil }
        let surface = CADSceneView(frame: NSRect(x: 0, y: 0, width: Double(width) / scale, height: Double(height) / scale))
        surface.preparesRendererResources = false
        surface.update(mesh: mesh, mode: display, grid: false, command: nil, selectedSolids: selected, hiddenSolids: [],
                       projection: projection, solidColors: colors)
        var prepared = try await surface.waitForCaptureResources(until: .now + .seconds(60))
        XCTAssertTrue(prepared)
        try surface.configureAgentView(named: view, camera: camera)
        surface.layout()
        prepared = try await surface.waitForCaptureResources(until: .now + .seconds(60))
        XCTAssertTrue(prepared)
        let metadata = surface.agentCaptureMetadata
        let capture = try CADViewportCapture(view: surface, width: width, height: height)
        XCTAssertEqual(capture.frame.field != nil, exact && display != .wireframe)
        return Shot(pixels: Pixels(try capture.image()), metadata: metadata, exactSolids: surface.exactSolids.count)
    }

    /// The design camera's ray through a pixel center of an orthographic capture.
    private func orthographicRay(_ metadata: [String: Any], x: Int, y: Int, width: Int, height: Int)
        -> (origin: SIMD3<Double>, direction: SIMD3<Double>) {
        let m = metadata["cameraTransform"] as! [Double]
        let right = SIMD3(m[0], m[1], m[2]), up = SIMD3(m[4], m[5], m[6]), backward = SIMD3(m[8], m[9], m[10])
        let position = SIMD3(m[12], m[13], m[14])
        let scale = metadata["orthographicScale"] as! Double
        let nx = (Double(x) + 0.5) / Double(width) * 2 - 1, ny = 1 - (Double(y) + 0.5) / Double(height) * 2
        let aspect = Double(width) / Double(height)
        return (position + simd_normalize(right) * (nx * scale * aspect) + simd_normalize(up) * (ny * scale),
                -simd_normalize(backward))
    }

    // MARK: Depth

    /// A sphere resting through a plate (two parts): the visible contact curve
    /// is decided per pixel by the exact depths of both solids. At this zoom a
    /// pixel is 2 µm, far below the proxies' offset (about 30 pixels), so
    /// proxy depth or a failed march would move the curve by many pixels.
    func testExactDepthPlacesAContactCurveWhereTheAnalyticFieldsDo() async throws {
        let preview = try await preview("result = [sphere(5), box((30, 30, 1))]")
        var camera = CADAgentCamera()
        let target = SIMD3<Double>(4.975, 0, 0.5)
        camera.target = [target.x, target.y, target.z]
        camera.position = [target.x - 4, target.y - 8, target.z + 6]
        camera.zoom = 8
        let colors = [0: PartColor(red: 0.9, green: 0.1, blue: 0.1), 1: PartColor(red: 0.1, green: 0.8, blue: 0.1)]
        let width = 400, height = 300
        let exact = try await render(preview, projection: .isometric, camera: camera, width: width, height: height, colors: colors)
        let mesh = try await render(preview, exact: false, projection: .isometric, camera: camera, width: width, height: height,
                                    colors: colors)
        XCTAssertEqual(exact.exactSolids, 2)
        // Analytic first surface per pixel: 1 sphere, 2 plate, 0 nothing.
        func expected(_ x: Int, _ y: Int) -> Int {
            let ray = orthographicRay(exact.metadata, x: x, y: y, width: width, height: height)
            var best = (Double.infinity, 0)
            let b = simd_dot(ray.origin, ray.direction), c = simd_length_squared(ray.origin) - 25
            if b * b - c >= 0 { best = (-b - sqrt(b * b - c), 1) }
            var near = -Double.infinity, far = Double.infinity
            for axis in 0..<3 {
                let extent = axis == 2 ? 0.5 : 15
                let t0 = (-extent - ray.origin[axis]) / ray.direction[axis], t1 = (extent - ray.origin[axis]) / ray.direction[axis]
                near = max(near, min(t0, t1)); far = min(far, max(t0, t1))
            }
            if near <= far, near < best.0 { best = (near, 2) }
            return best.1
        }
        func observed(_ shot: Shot, _ x: Int, _ y: Int) -> Int {
            let c = shot.pixels.rgb(x, y)
            return c.x > c.y + 0.05 ? 1 : c.y > c.x + 0.05 ? 2 : 0
        }
        var analytic = [[Int]](repeating: [Int](repeating: 0, count: width), count: height)
        for y in 0..<height { for x in 0..<width { analytic[y][x] = expected(x, y) } }
        var compared = 0, exactWrong = 0, meshWrong = 0, sphere = 0, plate = 0
        for y in 1..<(height - 1) {
            for x in 1..<(width - 1) {
                let value = analytic[y][x]
                guard (-1...1).allSatisfy({ dy in (-1...1).allSatisfy { analytic[y + dy][x + $0] == value } }) else { continue }
                compared += 1
                if value == 1 { sphere += 1 } else if value == 2 { plate += 1 }
                if observed(exact, x, y) != value { exactWrong += 1 }
                if observed(mesh, x, y) != value { meshWrong += 1 }
            }
        }
        XCTAssertGreaterThan(sphere, compared / 10, "the view shows the sphere")
        XCTAssertGreaterThan(plate, compared / 10, "the view shows the plate")
        XCTAssertLessThan(Double(exactWrong) / Double(compared), 0.002,
                          "exact depth agrees with the analytic fields away from their curves (mesh: \(meshWrong) of \(compared))")
    }

    // MARK: Grazing rays

    /// Rays passing a fold of a gyroid sheet on their way to the sheet behind
    /// it outlast the march's steps by far. They must still reach a surface:
    /// no background may show where the display mesh is solid (the old march
    /// left 64 such pixels in these views).
    func testRaysGrazingAFoldReachTheSurfaceBehindIt() async throws {
        let preview = try await preview("""
            pitch = 33.0
            k = 2 * math.pi / pitch
            g = sin(k * x) * cos(k * y) + sin(k * y) * cos(k * z) + sin(k * z) * cos(k * x)
            result = implicit(abs(g) - 0.44, ((-25, -25, -25), (25, 25, 25)))
            """)
        let color = try XCTUnwrap(CADSceneView.sceneBackgroundColor.usingColorSpace(.sRGB))
        let background = SIMD3(Double(color.redComponent), Double(color.greenComponent), Double(color.blueComponent))
        var camera = CADAgentCamera()
        camera.zoom = 2
        var holes = 0
        for view in ["isometric", "left", "top"] {
            let exact = try await render(preview, projection: .perspective, view: view, camera: camera, width: 800, height: 600)
            let mesh = try await render(preview, exact: false, projection: .perspective, view: view, camera: camera,
                                        width: 800, height: 600)
            let width = mesh.pixels.width, height = mesh.pixels.height
            var covered = [Bool](repeating: false, count: width * height)
            for y in 0..<height {
                for x in 0..<width { covered[y * width + x] = simd_reduce_max(abs(mesh.pixels.rgb(x, y) - background)) > 30.0 / 255 }
            }
            // Background in the exact image with the mesh solid around it
            // (silhouettes differ by the mesh's deviation).
            for y in 3..<(height - 3) {
                for x in 3..<(width - 3) where simd_reduce_max(abs(exact.pixels.rgb(x, y) - background)) <= 6.0 / 255 {
                    let solid = (-3...3).allSatisfy { dy in (-3...3).allSatisfy { dx in covered[(y + dy) * width + x + dx] } }
                    if solid { holes += 1 }
                }
            }
        }
        XCTAssertEqual(holes, 0, "background shows through exact faces where the display mesh is solid")
    }

    // MARK: Translucent layers

    /// Every proxy fragment owns one crossing: front faces the entering one,
    /// back faces the exiting one. A hollow sphere has four layers through
    /// its center and two through its rim; wrong ownership adds or drops
    /// layers and changes transmittance by about a third.
    func testTranslucentLayersMatchTheDisplayMeshLayers() async throws {
        let preview = try await preview("result = difference(sphere(5), sphere(3.5))")
        let exact = try await render(preview, display: .translucent, projection: .perspective, view: "isometric")
        let mesh = try await render(preview, exact: false, display: .translucent, projection: .perspective, view: "isometric")
        var differences: [Double] = []
        for y in 0..<exact.pixels.height {
            for x in 0..<exact.pixels.width {
                let a = exact.pixels.rgb(x, y), b = mesh.pixels.rgb(x, y)
                guard simd_reduce_max(abs(a - mesh.pixels.rgb(0, 0))) > 0.02 || simd_reduce_max(abs(b - mesh.pixels.rgb(0, 0))) > 0.02
                else { continue }
                differences.append(simd_reduce_max(abs(a - b)) * 255)
            }
        }
        differences.sort()
        let mean = differences.reduce(0, +) / Double(differences.count)
        XCTAssertGreaterThan(differences.count, 20_000)
        XCTAssertLessThan(mean, 3, "translucent exact layers match the mesh's on average")
        XCTAssertLessThan(differences[Int(Double(differences.count) * 0.98)], 12, "no area loses or gains a layer")
    }

    /// Two overlapping parts whose centers are at the same depth swap their
    /// draw order between two views a degree apart. Accumulation and
    /// transmittance are commutative, so the exact image changes like the
    /// display meshes' image does.
    func testTranslucentPartsAreIndependentOfDrawOrder() async throws {
        let preview = try await preview("result = [box((4, 4, 4), center=(-1, 0, 0)), box((4, 4, 4), center=(1, 0, 0))]")
        func shot(_ yaw: Double, exact: Bool) async throws -> Shot {
            var camera = CADAgentCamera()
            camera.yawDegrees = -90 + yaw
            camera.elevationDegrees = 0
            return try await render(preview, exact: exact, display: .translucent, projection: .isometric, camera: camera)
        }
        let left = try await shot(-1, exact: true), right = try await shot(1, exact: true)
        let meshLeft = try await shot(-1, exact: false), meshRight = try await shot(1, exact: false)
        // The overlap of the two boxes lies at the image center.
        func patch(_ shot: Shot) -> SIMD3<Double> {
            var sum = SIMD3<Double>.zero
            for y in 170..<190 { for x in 230..<250 { sum += shot.pixels.rgb(x, y) } }
            return sum / 400
        }
        let exactChange = simd_reduce_max(abs(patch(left) - patch(right))) * 255
        let meshChange = simd_reduce_max(abs(patch(meshLeft) - patch(meshRight))) * 255
        XCTAssertLessThan(abs(exactChange - meshChange), 2, "exact \(exactChange), mesh \(meshChange)")
        XCTAssertLessThan(simd_reduce_max(abs(patch(left) - patch(meshLeft))) * 255, 4)
    }

    // MARK: Strokes

    /// Crease strokes come from the display mesh, slightly inside the exact
    /// surface: their depth allowance keeps them visible over exact faces,
    /// while nearer faces still hide the box's back edges.
    func testCreaseStrokesStayVisibleAndOccludedOverExactFaces() async throws {
        let preview = try await preview("result = box((10, 10, 10))")
        let exact = try await render(preview, display: .edges, projection: .perspective, view: "isometric")
        let mesh = try await render(preview, exact: false, display: .edges, projection: .perspective, view: "isometric")
        func ink(_ shot: Shot) -> Set<Int> {
            var pixels = Set<Int>()
            for y in 0..<shot.pixels.height {
                for x in 0..<shot.pixels.width where simd_reduce_max(shot.pixels.rgb(x, y)) < 0.06 {
                    pixels.insert(y * shot.pixels.width + x)
                }
            }
            return pixels
        }
        let exactInk = ink(exact), meshInk = ink(mesh)
        XCTAssertGreaterThan(meshInk.count, 500)
        XCTAssertGreaterThan(Double(exactInk.count), 0.85 * Double(meshInk.count), "visible creases keep their strokes")
        XCTAssertLessThan(Double(exactInk.count), 1.15 * Double(meshInk.count), "hidden back edges stay hidden")
        XCTAssertGreaterThan(Double(exactInk.intersection(meshInk).count), 0.8 * Double(meshInk.count))
    }

    // MARK: Highlights

    /// A selected part's outline is the contour of its exact projection: a
    /// circle around a sphere, at the same distance from its center in
    /// every direction.
    func testSelectionOutlineFollowsTheExactSilhouette() async throws {
        let preview = try await preview("result = sphere(5)")
        let width = 480, height = 360
        let shot = try await render(preview, projection: .isometric, width: width, height: height, selected: [0])
        let scale = shot.metadata["orthographicScale"] as! Double
        let radius = 5 / scale * Double(height) / 2
        var outer: [Double] = []
        for step in 0..<360 {
            let angle = Double(step) * .pi / 180
            var last = 0.0
            var r = radius - 4
            while r < radius + 20 {
                let x = Int(Double(width) / 2 + r * cos(angle)), y = Int(Double(height) / 2 - r * sin(angle))
                let c = shot.pixels.rgb(x, y)
                if c.x > 0.6, c.y > 0.3, c.z < 0.45 { last = r }
                r += 0.25
            }
            outer.append(last)
        }
        let mean = outer.reduce(0, +) / Double(outer.count)
        let deviation = sqrt(outer.map { ($0 - mean) * ($0 - mean) }.reduce(0, +) / Double(outer.count))
        XCTAssertGreaterThan(mean, radius + 0.5, "the outline lies outside the silhouette")
        XCTAssertLessThan(mean, radius + 8)
        XCTAssertLessThan(deviation, 0.75, "the outline is a circle around the exact silhouette")
    }

    // MARK: Captures

    /// Independent captures of the same state render identical pixels, and
    /// exact faces shade like SceneKit's physically based display meshes.
    func testCapturesAreReproducibleAndShadeLikeDisplayMeshes() async throws {
        let preview = try await preview("result = sphere(5)")
        let first = try await render(preview, projection: .perspective, view: "isometric")
        let second = try await render(preview, projection: .perspective, view: "isometric")
        let mesh = try await render(preview, exact: false, projection: .perspective, view: "isometric")
        XCTAssertFalse(first.pixels.isBlank)
        var identical = true, differences: [Double] = []
        for y in 0..<first.pixels.height {
            for x in 0..<first.pixels.width {
                identical = identical && first.pixels.rgb(x, y) == second.pixels.rgb(x, y)
                // Interior pixels only: silhouettes differ by the mesh's deviation.
                let center = SIMD2<Double>(Double(x) - 240, Double(y) - 180)
                if simd_length(center) < 80 {
                    differences.append(simd_reduce_max(abs(first.pixels.rgb(x, y) - mesh.pixels.rgb(x, y))) * 255)
                }
            }
        }
        XCTAssertTrue(identical)
        differences.sort()
        XCTAssertLessThan(differences.reduce(0, +) / Double(differences.count), 1.5)
        XCTAssertLessThan(differences[Int(Double(differences.count) * 0.99)], 4)
    }

    // MARK: Navigation

    func testNavigationPolicyDrawsMeshesOnlyWhileNavigatingDesignsThatMissTheBudget() {
        var policy = ViewportNavigationPolicy()
        let interval = 1.0 / 60, pixels = 1_000_000
        XCTAssertTrue(policy.drawsExactFaces(pixels: pixels, interval: interval), "at rest")
        policy.begin(.drag)
        XCTAssertFalse(policy.drawsExactFaces(pixels: pixels, interval: interval), "unmeasured designs move with meshes")
        policy.record(exact: true, gpuSeconds: 0.004, pixels: pixels)
        XCTAssertTrue(policy.drawsExactFaces(pixels: pixels, interval: interval), "a light design stays exact while moving")
        policy.record(exact: true, gpuSeconds: 0.030, pixels: pixels)
        XCTAssertFalse(policy.drawsExactFaces(pixels: pixels, interval: interval), "a heavy design draws meshes while moving")
        XCTAssertTrue(policy.drawsExactFaces(pixels: pixels / 4, interval: interval), "the estimate scales with the frame's pixels")
        policy.record(exact: false, gpuSeconds: 0.001, pixels: pixels)
        XCTAssertFalse(policy.drawsExactFaces(pixels: pixels, interval: interval), "mesh frames do not change the estimate")
        XCTAssertEqual(policy.meshFrames, 1)
        XCTAssertEqual(policy.exactFrames, 2)
        policy.begin(.scroll)
        XCTAssertFalse(policy.end(.drag), "another source still navigates")
        XCTAssertFalse(policy.end(.drag), "ending twice changes nothing")
        XCTAssertTrue(policy.end(.scroll), "the last source ends navigation")
        XCTAssertTrue(policy.drawsExactFaces(pixels: pixels, interval: interval), "exact at rest")
        policy.begin(.transition)
        policy.begin(.wheel)
        XCTAssertTrue(policy.endAll())
        XCTAssertFalse(policy.endAll())
        XCTAssertFalse(policy.isNavigating)
    }

    /// In a live window, a heavy design orbits with display meshes and shows
    /// exact faces again in the frame published when the drag ends; wheel
    /// zoom, which reports no end, returns to exact faces after it settles.
    func testLiveViewportDrawsMeshesWhileNavigatingAHeavyDesign() async throws {
        let preview = try await preview("result = sphere(5)")
        guard !NSScreen.screens.isEmpty else { throw XCTSkip("A window server is required for live viewport drawing.") }
        let application = NSApplication.shared
        let previousPolicy = application.activationPolicy()
        XCTAssertTrue(application.setActivationPolicy(.regular))
        defer { _ = application.setActivationPolicy(previousPolicy) }
        application.finishLaunching()
        let view = CADSceneView(frame: NSRect(x: 0, y: 0, width: 480, height: 360))
        let frames = ViewportRenderCounter(view)
        let window = NSWindow(contentRect: NSRect(x: 60, y: 60, width: 480, height: 360), styleMask: [.titled],
                              backing: .buffered, defer: false)
        window.isReleasedWhenClosed = false
        window.contentView = view
        defer { window.orderOut(nil); window.contentView = nil; window.close() }
        view.update(mesh: preview, mode: .edges, grid: false, command: nil, selectedSolids: [], hiddenSolids: [])
        application.activate(ignoringOtherApps: true)
        window.makeKeyAndOrderFront(nil)
        view.layout()
        let prepared = try await view.waitForCaptureResources(until: .now + .seconds(30))
        XCTAssertTrue(prepared)
        func settle(after previous: Int) async throws {
            let deadline = ContinuousClock.now + .seconds(5)
            while (frames.count <= previous || view.hasPendingFrames), ContinuousClock.now < deadline {
                try await Task.sleep(for: .milliseconds(10))
            }
            XCTAssertGreaterThan(frames.count, previous, "a frame completes")
        }
        try await settle(after: 0)
        XCTAssertTrue(view.renderedExactFaces, "frames at rest are exact")
        XCTAssertEqual(view.exactSolids, [0])
        // The live scene's own field nodes draw the part, as its copies do.
        let live = Pixels(try view.imageOfLiveScene(width: 480, height: 360))
        let copy = Pixels(try CADViewportCapture(view: view, width: 480, height: 360).image())
        XCTAssertGreaterThan(simd_reduce_max(abs(live.rgb(240, 180) - live.rgb(2, 2))), 0.2, "the sphere is drawn")
        var largest = 0.0
        for y in stride(from: 0, to: 360, by: 3) {
            for x in stride(from: 0, to: 480, by: 3) { largest = max(largest, simd_reduce_max(abs(live.rgb(x, y) - copy.rgb(x, y)))) }
        }
        XCTAssertLessThan(largest, 0.02, "the live scene and a capture copy render alike")

        // A measured exact frame far over the budget.
        view.navigationPolicy.record(exact: true, gpuSeconds: 1, pixels: 1)
        func event(_ type: NSEvent.EventType, _ x: Double, _ y: Double) throws -> NSEvent {
            try XCTUnwrap(NSEvent.mouseEvent(with: type, location: view.convert(NSPoint(x: x, y: y), to: nil),
                                             modifierFlags: [], timestamp: 0, windowNumber: window.windowNumber,
                                             context: nil, eventNumber: 1, clickCount: 1, pressure: 0))
        }
        view.rightMouseDown(with: try event(.rightMouseDown, 200, 200))
        var before = frames.count
        view.rightMouseDragged(with: try event(.rightMouseDragged, 240, 220))
        try await settle(after: before)
        XCTAssertFalse(view.renderedExactFaces, "orbiting a heavy design draws display meshes")
        XCTAssertTrue(view.navigationPolicy.isNavigating)
        before = frames.count
        view.rightMouseUp(with: try event(.rightMouseUp, 240, 220))
        try await settle(after: before)
        XCTAssertTrue(view.renderedExactFaces, "ending the drag publishes an exact frame")
        XCTAssertFalse(view.navigationPolicy.isNavigating)

        // Wheel events carry no phases: one deferred publish ends them. (The
        // exact frame above measured this light design; measure heavy again.)
        view.navigationPolicy.record(exact: true, gpuSeconds: 1, pixels: 1)
        let scroll = try XCTUnwrap(CGEvent(scrollWheelEvent2Source: nil, units: .pixel, wheelCount: 1, wheel1: 3,
                                           wheel2: 0, wheel3: 0))
        before = frames.count
        view.scrollWheel(with: try XCTUnwrap(NSEvent(cgEvent: scroll)))
        try await settle(after: before)
        XCTAssertFalse(view.renderedExactFaces, "wheel zoom of a heavy design draws display meshes")
        before = frames.count
        try await Task.sleep(for: ViewportNavigationPolicy.wheelSettle + .milliseconds(100))
        try await settle(after: before)
        XCTAssertTrue(view.renderedExactFaces, "the settled wheel zoom publishes an exact frame")
        XCTAssertFalse(view.hasPendingFrames)

        // After idle, wake and reveal each draw one fresh exact frame.
        try await Task.sleep(for: .milliseconds(300))
        before = frames.count
        NSWorkspace.shared.notificationCenter.post(name: NSWorkspace.didWakeNotification, object: nil)
        try await settle(after: before)
        XCTAssertTrue(view.renderedExactFaces, "waking draws a fresh exact frame")
        // Without ordering animations, which leave AppKit animation threads
        // blocked in a test process.
        window.animationBehavior = .none
        window.orderOut(nil)
        try await Task.sleep(for: .milliseconds(100))
        before = frames.count
        window.makeKeyAndOrderFront(nil)
        try await settle(after: before)
        XCTAssertTrue(view.renderedExactFaces, "revealing the window draws a fresh exact frame")
        XCTAssertTrue(view.renderedShowsModel)
    }
}
