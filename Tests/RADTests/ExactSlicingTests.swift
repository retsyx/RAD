import AppKit
import Metal
import SceneKit
import simd
import XCTest
@testable import RAD

/// Exact layers drawn in screen slices and composited: slice planning, the
/// geometry identity that keeps or abandons exact work, the navigation
/// estimate, and equivalence of sliced, single-slice and reused layers.
/// Engine-backed tests build small designs.
@MainActor
final class ExactSlicingTests: XCTestCase {
    // MARK: Slice planning

    private struct Generator: RandomNumberGenerator {
        var state: UInt64
        mutating func next() -> UInt64 {
            state = state &* 6_364_136_223_846_793_005 &+ 1_442_695_040_888_963_407
            return state >> 11
        }
    }

    /// Every block a proxy reaches belongs to exactly one slice, in row
    /// order; a slice is one rectangle: a run of one block row's blocks, or
    /// a strip of whole rows spanning their reached columns; its estimate
    /// stays within the budget unless it is a single block; blocks no proxy
    /// reaches get no slice.
    func testSlicePlanCoversEveryReachedBlockOnceWithinItsBudget() {
        var random = Generator(state: 7)
        let size = CADSlicePlan.block
        for trial in 0..<300 {
            let width = Int.random(in: 1...2_400, using: &random), height = Int.random(in: 1...1_600, using: &random)
            let rects: [SIMD4<Int32>?] = (0..<Int.random(in: 0...6, using: &random)).map { _ in
                if Int.random(in: 0..<12, using: &random) == 0 { return nil }
                let x0 = Int32.random(in: -60...Int32(width + 20), using: &random)
                let y0 = Int32.random(in: -60...Int32(height + 20), using: &random)
                return SIMD4(x0, y0, x0 + Int32.random(in: 0...1_200, using: &random), y0 + Int32.random(in: 0...900, using: &random))
            }
            var plan = CADSlicePlan(width: width, height: height, rects: rects)
            let columns = (width + size - 1) / size, rows = (height + size - 1) / size
            // The reference coverage of each block.
            var expected = [Int](repeating: 0, count: columns * rows)
            for rect in rects {
                let r = rect ?? SIMD4(0, 0, Int32(width), Int32(height))
                for by in 0..<rows {
                    for bx in 0..<columns {
                        let w = min(Int(r.z), min(width, (bx + 1) * size)) - max(Int(r.x), bx * size)
                        let h = min(Int(r.w), min(height, (by + 1) * size)) - max(Int(r.y), by * size)
                        if w > 0, h > 0 { expected[by * columns + bx] += w * h }
                    }
                }
            }
            XCTAssertEqual(plan.total, expected.reduce(0, +))
            // Small budgets give runs of blocks, large ones strips of rows.
            let budget = trial % 3 == 0 ? Int.random(in: 100_000...4_000_000, using: &random) : Int.random(in: 1...300_000, using: &random)
            var owners = [Int](repeating: -1, count: expected.count)
            var index = 0, sum = 0, lastBlock = -1
            while let slice = plan.next(budget: budget) {
                XCTAssertEqual(slice.rects.count, 1, "one rectangle per slice")
                let r = slice.rects[0]
                XCTAssertEqual(Int(r.x) % size, 0)
                XCTAssertEqual(Int(r.y) % size, 0)
                XCTAssertTrue(Int(r.z) % size == 0 || Int(r.z) == width)
                XCTAssertTrue(Int(r.w) % size == 0 || Int(r.w) == height)
                let firstRow = Int(r.y) / size, lastRow = (Int(r.w) - 1) / size
                var covered = 0, blocks = 0
                for by in firstRow...lastRow {
                    let reached = (0..<columns).filter { expected[by * columns + $0] > 0 }
                    if lastRow > firstRow {
                        XCTAssertFalse(reached.isEmpty, "a strip's rows are reached")
                        XCTAssertTrue(reached.allSatisfy { owners[by * columns + $0] == -1 }, "a strip holds whole rows")
                    }
                    for bx in (Int(r.x) / size)..<((Int(r.z) + size - 1) / size) where expected[by * columns + bx] > 0 {
                        let at = by * columns + bx
                        XCTAssertEqual(owners[at], -1, "a block belongs to one slice")
                        XCTAssertGreaterThan(at, lastBlock, "slices follow row order")
                        owners[at] = index
                        lastBlock = at
                        covered += expected[at]
                        blocks += 1
                    }
                }
                XCTAssertGreaterThan(blocks, 0)
                XCTAssertEqual(slice.covered, covered)
                XCTAssertTrue(slice.covered <= budget || blocks == 1, "\(slice.covered) over \(budget)")
                XCTAssertEqual(slice.whole, index == 0 && plan.isDone)
                sum += slice.covered
                index += 1
            }
            XCTAssertTrue(plan.isDone)
            XCTAssertEqual(sum, plan.total)
            XCTAssertEqual(plan.remaining, 0)
            for (block, value) in expected.enumerated() {
                XCTAssertEqual(owners[block] != -1, value > 0, "exactly the reached blocks have slices")
            }
        }
    }

    /// A job's slices start at the first budget and two blocks; budgets
    /// double up to the cap and so do block counts, so a cost estimate that
    /// is too low cannot make an early slice large.
    func testSliceBudgetsStartSmallAndGrow() {
        let first = CADViewportPasses.firstSliceBudget, cap = CADViewportPasses.sliceBudget
        XCTAssertEqual(CADViewportPasses.sliceBudget(after: 0), first)
        XCTAssertEqual(CADViewportPasses.sliceBudget(after: 1), 2 * first)
        XCTAssertEqual(CADViewportPasses.sliceBudget(after: 2), 4 * first)
        XCTAssertEqual(CADViewportPasses.sliceBudget(after: 100), cap)
        XCTAssertTrue((0..<30).map { CADViewportPasses.sliceBudget(after: $0) }.allSatisfy { $0 <= cap })
        XCTAssertEqual((0..<4).map { CADViewportPasses.sliceBlocks(after: $0) }, [2, 4, 8, 16])
        XCTAssertGreaterThan(CADViewportPasses.sliceBlocks(after: 100), 1_000_000)
        // Two blocks at most, however cheap the estimate.
        var plan = CADSlicePlan(width: 1_000, height: 1_000, rects: [nil])
        let slice = try? XCTUnwrap(plan.next(budget: .max, blocks: 2))
        XCTAssertEqual(slice?.rects, [SIMD4(0, 0, 256, 128)])
        XCTAssertEqual(plan.next(budget: .max, blocks: 8)?.rects, [SIMD4(256, 0, 1_000, 128)], "the rest of the row")
        XCTAssertEqual(plan.next(budget: .max, blocks: 16)?.rects, [SIMD4(0, 128, 1_000, 384)], "two whole rows")
    }

    /// Slices plan with an average of recent measured costs weighed by their
    /// pixels (a dense strip does not make every later slice small); whether a frame draws its
    /// geometry itself uses the largest cost of frames drawn at once. A last
    /// slice takes up to 1.25 budgets, so no small remainder pays a pass.
    func testSliceBudgetsAverageMeasuredCostsAndSplitEvenly() {
        let cost = CADExactCost()
        let opaque = CADExactCost.Key(resources: ObjectIdentifier(Self.self), width: 100, height: 100, translucent: false)
        let translucent = CADExactCost.Key(resources: ObjectIdentifier(Self.self), width: 100, height: 100, translucent: true)
        XCTAssertFalse(cost.hasSliceRate(opaque))
        XCTAssertEqual(cost.sliceRate(opaque), CADExactCost.defaultOpaqueRate)
        XCTAssertEqual(cost.sliceRate(translucent), CADExactCost.defaultTranslucentRate)
        cost.recordSlice(opaque, seconds: 0.010, covered: 10_000)
        XCTAssertTrue(cost.hasSliceRate(opaque))
        XCTAssertEqual(cost.sliceRate(opaque), 1e-6, accuracy: 1e-12)
        cost.recordSlice(opaque, seconds: 0.090, covered: 30_000)
        let decay = CADExactCost.decay
        let average = (0.010 * decay + 0.090) / (10_000 * decay + 30_000)
        XCTAssertEqual(cost.sliceRate(opaque), average, accuracy: 1e-12, "recent slices, weighed by their pixels")
        cost.recordSlice(opaque, seconds: 1, covered: CADExactCost.minimumCovered - 1)
        XCTAssertEqual(cost.sliceRate(opaque), average, accuracy: 1e-12, "tiny slices do not measure")
        XCTAssertEqual(cost.frameRate(opaque), average, accuracy: 1e-12, "slices stand in for unmeasured frames")
        cost.recordFrame(opaque, seconds: 0.004, covered: 1_000)
        cost.recordFrame(opaque, seconds: 0.001, covered: 1_000)
        XCTAssertEqual(cost.frameRate(opaque), 4e-6, accuracy: 1e-12, "the largest frame")
        cost.forget(ObjectIdentifier(Self.self))
        XCTAssertFalse(cost.hasSliceRate(opaque), "released resources forget their measurements")

        // A frame of 8 block rows of 128,000 covered pixels (the last 104 rows): strips of two rows
        // within 300,000, then the last 232,000 together instead of a small remainder.
        var plan = CADSlicePlan(width: 1_000, height: 1_000, rects: [nil])
        var slices: [Int] = []
        while let slice = plan.next(budget: CADViewportPasses.evenBudget(plan, 300_000)) { slices.append(slice.covered) }
        XCTAssertEqual(slices, [256_000, 256_000, 256_000, 232_000])
        var small = CADSlicePlan(width: 1_000, height: 1_000, rects: [nil])
        XCTAssertEqual(CADViewportPasses.evenBudget(small, 2_000_000), 2_000_000, "everything fits one slice")
        XCTAssertEqual(small.next(budget: 2_000_000)?.whole, true)
    }

    // MARK: Geometry identity

    private func key(camera: simd_float4x4 = matrix_identity_float4x4, width: Int = 100,
                     instances: [CADExactKey.Instance] = [.init(solid: 0, model: matrix_identity_float4x4)],
                     translucent: Bool = false) -> CADExactKey {
        CADExactKey(camera: camera, projection: SIMD2(38, 0), width: width, height: 80,
                    resources: ObjectIdentifier(Self.self), instances: instances, translucent: translucent,
                    complete: false, debug: 0)
    }

    private func frame(_ key: CADExactKey?, presentation: UInt64 = 1, navigating: Bool = false,
                       fits: Bool = true) -> CADViewportFrame {
        var frame = CADViewportFrame(width: 100, height: 80, referenceHeight: 80, translucent: false,
                                     palette: CADSceneView.highlightPalette)
        frame.key = key
        frame.presentation = presentation
        frame.navigating = navigating
        frame.exactFitsBudget = fits
        return frame
    }

    /// A newer frame with the same geometry identity keeps the job and only
    /// changes what is composited; any other identity abandons it.
    func testFramesKeepOrAbandonExactWorkByGeometryIdentity() {
        let resting = key()
        typealias Loop = ViewportRenderLoop
        XCTAssertEqual(Loop.action(for: frame(nil), job: (resting, false), presented: 1), .meshes, "no exact faces")
        XCTAssertEqual(Loop.action(for: frame(resting), job: nil, presented: nil), .start)
        XCTAssertEqual(Loop.action(for: frame(resting, navigating: true, fits: false), job: nil, presented: nil), .meshes,
                       "heavy designs move with display meshes")
        XCTAssertEqual(Loop.action(for: frame(resting, navigating: true, fits: true), job: nil, presented: nil), .start)
        XCTAssertEqual(Loop.action(for: frame(resting), job: (resting, true), presented: 1), .composite)
        XCTAssertEqual(Loop.action(for: frame(resting, navigating: true, fits: false), job: (resting, true), presented: 1),
                       .composite, "complete layers of an unchanged view composite even while navigating")
        XCTAssertEqual(Loop.action(for: frame(resting, presentation: 1), job: (resting, false), presented: 1),
                       .keep(standIn: false), "the screen already shows this state")
        XCTAssertEqual(Loop.action(for: frame(resting, presentation: 2), job: (resting, false), presented: 1),
                       .keep(standIn: true), "a hover change shows display meshes until the layers are complete")
        var moved = matrix_identity_float4x4
        moved.columns.3 = SIMD4(0, 0, 5, 1)
        XCTAssertEqual(Loop.action(for: frame(key(camera: moved)), job: (resting, false), presented: 1), .start)
        XCTAssertEqual(Loop.action(for: frame(key(width: 101)), job: (resting, true), presented: 1), .start)
        XCTAssertEqual(Loop.action(for: frame(key(instances: [])), job: (resting, true), presented: 1), .start,
                       "hiding a part draws the layers again")
        var placed = CADExactKey.Instance(solid: 0, model: matrix_identity_float4x4)
        placed.model.columns.3 = SIMD4(1, 0, 0, 1)
        XCTAssertEqual(Loop.action(for: frame(key(instances: [placed])), job: (resting, true), presented: 1), .start)

        // Opaque layers store parts, not colors: selection and color changes
        // composite again. Translucent layers bake their colors.
        let plain = CADFieldInstance(solid: 3, model: matrix_identity_float4x4, albedo: SIMD3(0.5, 0.5, 0.5), emission: .zero)
        var selected = plain
        selected.albedo = SIMD3(0.9, 0.6, 0.2)
        selected.emission = SIMD3(0.1, 0.05, 0)
        XCTAssertEqual(CADExactKey.instances([plain], translucent: false), CADExactKey.instances([selected], translucent: false))
        XCTAssertNotEqual(CADExactKey.instances([plain], translucent: true), CADExactKey.instances([selected], translucent: true))
    }

    // MARK: Navigation estimate

    /// The estimate is the largest exact cost measured since the resources
    /// or the frame size changed: a cheap view never makes a gesture draw
    /// exact frames that a costlier view would stall.
    func testNavigationEstimateKeepsTheLargestCostUntilResourcesOrSizeChange() {
        let first = NSObject(), second = NSObject()
        let pixels = 1_000_000, interval = 1.0 / 120
        var policy = ViewportNavigationPolicy()
        policy.setContext(resources: ObjectIdentifier(first), pixels: pixels)
        policy.begin(.drag)
        XCTAssertFalse(policy.drawsExactFaces(pixels: pixels, interval: interval), "unmeasured")
        policy.record(exact: true, gpuSeconds: 0.002, pixels: pixels)
        XCTAssertTrue(policy.drawsExactFaces(pixels: pixels, interval: interval), "a cheap design stays exact")
        policy.record(exact: true, gpuSeconds: 0.030, pixels: pixels)
        XCTAssertFalse(policy.drawsExactFaces(pixels: pixels, interval: interval))
        policy.record(exact: true, gpuSeconds: 0.001, pixels: pixels)
        XCTAssertFalse(policy.drawsExactFaces(pixels: pixels, interval: interval), "a cheaper view keeps the largest cost")
        policy.record(exact: true, gpuSeconds: 0.5, pixels: pixels, geometry: false)
        policy.record(exact: false, gpuSeconds: 0.5, pixels: pixels)
        XCTAssertEqual(policy.exactSecondsPerPixel, 0.030 / Double(pixels), "composites and meshes do not measure geometry")
        policy.setContext(resources: ObjectIdentifier(first), pixels: pixels)
        XCTAssertEqual(policy.exactSecondsPerPixel, 0.030 / Double(pixels), "an unchanged context keeps it")
        policy.setContext(resources: ObjectIdentifier(second), pixels: pixels)
        XCTAssertNil(policy.exactSecondsPerPixel, "new resources forget it")
        policy.record(exact: true, gpuSeconds: 0.002, pixels: pixels)
        policy.setContext(resources: ObjectIdentifier(second), pixels: pixels / 2)
        XCTAssertNil(policy.exactSecondsPerPixel, "a new frame size forgets it")
        XCTAssertTrue(policy.end(.drag))
        XCTAssertTrue(policy.drawsExactFaces(pixels: pixels, interval: interval), "exact at rest")
    }

    // MARK: Antialiasing variants

    /// Traced multisample rays are the default; builds with
    /// `-DRAD_ANALYTIC_EXACT_AA` derive coverage analytically. Both shader
    /// variants compile, with and without the gradient pass.
    func testBothExactAntialiasingVariantsCompile() throws {
        let device = try XCTUnwrap(MTLCreateSystemDefaultDevice())
        for analytic in [0, 1] {
            for gradient in [0, 1] {
                let options = MTLCompileOptions()
                options.preprocessorMacros = ["RAD_EXACT_ANALYTIC_AA": NSNumber(value: analytic),
                                              "RAD_FIELD_GRADIENT_PASS": NSNumber(value: gradient)]
                if #available(macOS 15.0, *) { options.mathMode = .safe } else { options.fastMathEnabled = false }
                let library = try device.makeLibrary(source: FieldProgramShaders.functions + CADFieldShading.source
                                                     + CADFieldPipelines.shaderSource, options: options)
                for name in ["cad_field_vertex", "cad_field_geometry", "cad_field_visibility", "cad_field_accumulate",
                             "cad_field_revealage", "cad_field_mask"] {
                    XCTAssertNotNil(library.makeFunction(name: name), "\(name) analytic \(analytic) gradient \(gradient)")
                }
            }
        }
    }

    // MARK: Engine-backed fixtures

    private static var previews: [String: PreviewMesh] = [:]

    private func preview(_ script: String) async throws -> PreviewMesh {
        guard ProcessInfo.processInfo.environment["RAD_RUN_ENGINE_TESTS"] == "1" else {
            throw XCTSkip("Set RAD_RUN_ENGINE_TESTS=1 to build designs with exact faces.")
        }
        guard MTLCreateSystemDefaultDevice() != nil else { throw XCTSkip("No Metal device.") }
        if let preview = Self.previews[script] { return preview }
        let preview = try await EngineService().buildPreview(
            ModelSpec(title: "Exact slicing", summary: "Exact slicing test.", units: "mm", script: script),
            python: EngineService.defaultPython, threads: 4)
        Self.previews[script] = preview
        return preview
    }

    private func capture(_ preview: PreviewMesh, display: ViewportDisplayMode, selected: Set<Int> = [],
                         width: Int = 480, height: Int = 360) async throws -> CADViewportCapture {
        let surface = CADSceneView(frame: NSRect(x: 0, y: 0, width: width, height: height))
        surface.preparesRendererResources = false
        surface.update(mesh: preview, mode: display, grid: true, command: nil, selectedSolids: selected, hiddenSolids: [],
                       projection: .perspective)
        var prepared = try await surface.waitForCaptureResources(until: .now + .seconds(60))
        XCTAssertTrue(prepared)
        try surface.configureAgentView(named: "isometric")
        surface.layout()
        prepared = try await surface.waitForCaptureResources(until: .now + .seconds(60))
        XCTAssertTrue(prepared)
        return try CADViewportCapture(view: surface, width: width, height: height)
    }

    private func pixels(_ image: CGImage) -> [UInt8] {
        var data = [UInt8](repeating: 0, count: image.width * image.height * 4)
        data.withUnsafeMutableBytes { raw in
            let context = CGContext(data: raw.baseAddress, width: image.width, height: image.height, bitsPerComponent: 8,
                                    bytesPerRow: image.width * 4, space: CGColorSpace(name: CGColorSpace.sRGB)!,
                                    bitmapInfo: CGImageAlphaInfo.premultipliedLast.rawValue)!
            context.draw(image, in: CGRect(x: 0, y: 0, width: image.width, height: image.height))
        }
        return data
    }

    // MARK: Equivalence

    /// Slices never split a pixel quad and each slice draws exactly the
    /// fragments of its own pixels, so band-by-band layers composite to the
    /// same image as layers drawn at once, opaque and translucent, with
    /// grid, crease strokes and exact selection outlines over them.
    /// Translucent accumulation maps each slice onto its own pass, which
    /// rounds clip coordinates differently: a few pixels may differ by 2/255.
    func testSlicedLayersCompositeLikeLayersDrawnAtOnce() async throws {
        // Two parts in contact (contact curves) and a hollow part (translucent layers).
        let preview = try await preview("result = [sphere(5), box((30, 30, 1)), difference(sphere(3), sphere(2.5)).translate((9, 0, 4))]")
        for display in [ViewportDisplayMode.edges, .translucent] {
            let capture = try await capture(preview, display: display, selected: [0])
            XCTAssertNotNil(capture.frame.field)
            let device = try XCTUnwrap(MTLCreateSystemDefaultDevice())
            let singlePasses = try CADViewportPasses(device: device), slicedPasses = try CADViewportPasses(device: device)
            let single = pixels(try CADViewportCapture.image(of: capture.frame, source: capture.source, passes: singlePasses,
                                                             sliceBudget: .infinity))
            let sliced = pixels(try CADViewportCapture.image(of: capture.frame, source: capture.source, passes: slicedPasses,
                                                             sliceBudget: 1e-12))
            XCTAssertEqual(singlePasses.statistics.geometrySlices, 1)
            XCTAssertGreaterThan(slicedPasses.statistics.geometrySlices, 1, "one band per slice")
            XCTAssertGreaterThanOrEqual(slicedPasses.statistics.maskSlices, 1, "the outline coverage is stored")
            var largest = 0, differing = 0
            for index in stride(from: 0, to: single.count, by: 4) {
                let difference = (0..<3).map { abs(Int(single[index + $0]) - Int(sliced[index + $0])) }.max()!
                largest = max(largest, difference)
                if difference > 0 { differing += 1 }
            }
            if display == .translucent {
                XCTAssertLessThanOrEqual(largest, 3, "\(display): \(differing) pixels differ")
                XCTAssertLessThan(differing, single.count / 4 / 1_000)
            } else {
                XCTAssertLessThanOrEqual(largest, 1, "\(display): \(differing) pixels differ")
            }
        }
    }

    /// In a live window, hover and selection changes at rest composite the
    /// stored layers without marching; outline coverage is marched once per
    /// selection; moving the camera draws the layers again.
    func testHoverAndSelectionAtRestCompositeStoredLayers() async throws {
        let preview = try await preview("result = [sphere(5), box((30, 30, 1))]")
        guard !NSScreen.screens.isEmpty else { throw XCTSkip("A window server is required for live viewport drawing.") }
        let application = NSApplication.shared
        _ = application.setActivationPolicy(.accessory)
        application.finishLaunching()
        let view = CADSceneView(frame: NSRect(x: 0, y: 0, width: 480, height: 360))
        let frames = ViewportRenderCounter(view)
        let window = NSWindow(contentRect: NSRect(x: 80, y: 80, width: 480, height: 360), styleMask: [.titled],
                              backing: .buffered, defer: false)
        window.isReleasedWhenClosed = false
        window.animationBehavior = .none
        window.contentView = view
        defer { window.orderOut(nil); window.contentView = nil; window.close() }
        view.update(mesh: preview, mode: .edges, grid: true, command: nil, selectedSolids: [], hiddenSolids: [])
        window.orderFrontRegardless()
        view.layout()
        let prepared = try await view.waitForCaptureResources(until: .now + .seconds(30))
        XCTAssertTrue(prepared)
        func settle(after previous: Int) async throws {
            let deadline = ContinuousClock.now + .seconds(10)
            while (frames.count <= previous || view.hasPendingFrames), ContinuousClock.now < deadline {
                try await Task.sleep(for: .milliseconds(10))
            }
            XCTAssertGreaterThan(frames.count, previous, "a frame completes")
            XCTAssertFalse(view.hasPendingFrames)
        }
        try await settle(after: 0)
        XCTAssertTrue(view.renderedExactFaces)
        let rested = try XCTUnwrap(view.exactStatistics)
        XCTAssertGreaterThan(rested.geometrySlices, 0)

        func mouse(_ type: NSEvent.EventType, _ x: Double, _ y: Double) throws -> NSEvent {
            try XCTUnwrap(NSEvent.mouseEvent(with: type, location: view.convert(NSPoint(x: x, y: y), to: nil),
                                             modifierFlags: [], timestamp: 0, windowNumber: window.windowNumber,
                                             context: nil, eventNumber: 1, clickCount: 1, pressure: 0))
        }
        // Hover over the design.
        var before = frames.count
        view.mouseMoved(with: try mouse(.mouseMoved, 240, 180))
        try await settle(after: before)
        let hovered = try XCTUnwrap(view.exactStatistics)
        XCTAssertEqual(hovered.geometrySlices, rested.geometrySlices, "hovering composites the stored layers")
        XCTAssertGreaterThan(hovered.composites, rested.composites)
        XCTAssertTrue(view.renderedExactFaces)

        // Select the sphere: exact again at once, its outline marched once.
        before = frames.count
        view.update(mesh: preview, mode: .edges, grid: true, command: nil, selectedSolids: [0], hiddenSolids: [])
        try await settle(after: before)
        let selected = try XCTUnwrap(view.exactStatistics)
        XCTAssertEqual(selected.geometrySlices, rested.geometrySlices, "selection composites the stored layers")
        XCTAssertGreaterThan(selected.maskSlices, hovered.maskSlices, "the exact outline is marched")
        XCTAssertTrue(view.renderedExactFaces)

        before = frames.count
        view.mouseMoved(with: try mouse(.mouseMoved, 200, 150))
        try await settle(after: before)
        let rehovered = try XCTUnwrap(view.exactStatistics)
        XCTAssertEqual(rehovered.geometrySlices, rested.geometrySlices)
        XCTAssertEqual(rehovered.maskSlices, selected.maskSlices, "hover reuses the stored outline coverage")

        // Moving the camera draws the layers again.
        before = frames.count
        view.rightMouseDown(with: try mouse(.rightMouseDown, 200, 200))
        view.rightMouseDragged(with: try mouse(.rightMouseDragged, 240, 220))
        view.rightMouseUp(with: try mouse(.rightMouseUp, 240, 220))
        try await settle(after: before)
        let orbited = try XCTUnwrap(view.exactStatistics)
        XCTAssertGreaterThan(orbited.geometrySlices, rehovered.geometrySlices)
        XCTAssertTrue(view.renderedExactFaces)
    }

    private final class Completions: @unchecked Sendable {
        private let lock = NSLock()
        private var outcomes: [(UInt64, ViewportRenderLoop.Outcome)] = []
        func append(_ generation: UInt64, _ outcome: ViewportRenderLoop.Outcome) {
            lock.withLock { outcomes.append((generation, outcome)) }
        }
        func drewExact(_ generation: UInt64) -> Bool {
            lock.withLock { outcomes.contains { $0.0 == generation && $0.1.exact && $0.1.final } }
        }
    }

    /// One slice runs while the next one is encoded behind it. Frames with new geometry
    /// abandon the job and release that uncommitted slice, and Metal runs a
    /// released command buffer's completed handlers: the render loop must
    /// ignore them and keep drawing. (Releasing it under the loop's lock once
    /// deadlocked the render thread and every caller of `submit`.)
    func testAbandoningAQueuedSliceKeepsTheRenderLoopDrawing() async throws {
        let preview = try await preview("result = [sphere(5), box((30, 30, 1))]")
        guard !NSScreen.screens.isEmpty else { throw XCTSkip("A window server is required for live viewport drawing.") }
        let capture = try await capture(preview, display: .edges, width: 1600, height: 1200)
        let frame = capture.frame, source = capture.source
        let key = try XCTUnwrap(frame.key)
        // Measured as heavy: every frame slices, a band per slice.
        let costKey = CADExactCost.Key(resources: key.resources, width: frame.width, height: frame.height, translucent: false)
        CADExactCost.shared.recordFrame(costKey, seconds: 1, covered: 1)
        CADExactCost.shared.recordSlice(costKey, seconds: 1, covered: CADExactCost.minimumCovered)
        defer { CADExactCost.shared.forget(key.resources) }

        let application = NSApplication.shared
        _ = application.setActivationPolicy(.accessory)
        application.finishLaunching()
        let host = NSView(frame: NSRect(x: 0, y: 0, width: 400, height: 300))
        let layer = CAMetalLayer()
        layer.pixelFormat = CADViewportPasses.colorFormat
        layer.framebufferOnly = true
        host.layer = layer
        host.wantsLayer = true
        let window = NSWindow(contentRect: NSRect(x: 80, y: 80, width: 400, height: 300), styleMask: [.titled],
                              backing: .buffered, defer: false)
        window.isReleasedWhenClosed = false
        window.animationBehavior = .none
        window.contentView = host
        defer { window.orderOut(nil); window.contentView = nil; window.close() }
        window.orderFrontRegardless()

        let device = try XCTUnwrap(MTLCreateSystemDefaultDevice())
        let renderer = SCNRenderer(device: device, options: nil)
        renderer.scene = source.scene
        renderer.pointOfView = source.camera
        renderer.usesReverseZ = true
        renderer.autoenablesDefaultLighting = false
        XCTAssertTrue(renderer.prepare(source.scene, shouldAbortBlock: nil))
        let completions = Completions()
        let loop = try XCTUnwrap(ViewportRenderLoop(layer: layer, renderer: renderer) { generation, _, outcome in
            completions.append(generation, outcome)
        })
        // Submitted off the main thread, so a stuck loop fails the test
        // instead of hanging it.
        let count: UInt64 = 60
        let submitted = expectation(description: "every frame is submitted")
        DispatchQueue.global(qos: .userInitiated).async {
            for generation in 1...count {
                var request = frame
                request.key?.debug = Int(generation)
                loop.submit(.init(frame: request, source: source, generation: generation))
                usleep(1_000)
            }
            submitted.fulfill()
        }
        guard await XCTWaiter().fulfillment(of: [submitted], timeout: 30) == .completed else {
            return XCTFail("the render loop stopped taking frames")
        }
        let deadline = ContinuousClock.now + .seconds(30)
        while !completions.drewExact(count), ContinuousClock.now < deadline {
            try await Task.sleep(for: .milliseconds(10))
        }
        XCTAssertTrue(completions.drewExact(count), "the newest frame is composited from its slices")
        XCTAssertGreaterThan(loop.statistics.geometrySlices, 1, "frames were sliced")
        loop.stop()
    }
}
