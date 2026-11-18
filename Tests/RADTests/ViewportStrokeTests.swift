import AppKit
import Metal
import SceneKit
import XCTest
@testable import RAD

final class ViewportStrokeTests: XCTestCase {
    @MainActor
    func testDetachedStrokePreparationPreservesSurfaceChannelsAfterSkippedSegments() async throws {
        let segments: [[Double]] = [[0, 0, 0, 0, 0, 0], [1, 2, 3, 4, 6, 3]]
        let surfaces = [CADStrokeSurface(first: SIMD3(1, 0, 0)),
                        CADStrokeSurface(first: SIMD3(0, 0, 4), second: SIMD3(0, -2, 0))]
        let prepared = try await Task.detached {
            try CADStrokeGeometry.prepare(segments, surfaces: surfaces)
        }.value
        let geometry = CADStrokeGeometry.make(prepared, color: .white, viewportHeight: 192)
        let positions = try XCTUnwrap(geometry.sources(for: .vertex).first)
        let directions = try XCTUnwrap(geometry.sources(for: .normal).first)
        let channels = geometry.sources(for: .texcoord)
        XCTAssertEqual(positions.vectorCount, 4)
        XCTAssertEqual(channels.count, 4)
        XCTAssertEqual(try value(positions, vector: 0, component: 0), 1)
        XCTAssertEqual(try value(positions, vector: 3, component: 1), 6)
        XCTAssertEqual(try value(directions, vector: 0, component: 0), 0.6, accuracy: 1e-6)
        XCTAssertEqual(try value(directions, vector: 0, component: 1), 0.8, accuracy: 1e-6)
        XCTAssertEqual(try value(channels[0], vector: 0, component: 1), -1)
        XCTAssertEqual(try value(channels[0], vector: 3, component: 1), 1)
        XCTAssertEqual(try value(channels[1], vector: 0, component: 0), 0)
        XCTAssertEqual(try value(channels[2], vector: 0, component: 0), 1)
        XCTAssertEqual(try value(channels[3], vector: 0, component: 0), -1)
        let element = try XCTUnwrap(geometry.elements.first)
        XCTAssertEqual(try (0..<6).map { try index(element, at: $0) }, [0, 1, 2, 2, 1, 3])
    }

    func testCanceledStrokePreparationDoesNotReturnBuffers() async throws {
        let preparation = Task.detached {
            withUnsafeCurrentTask { $0?.cancel() }
            return try CADStrokeGeometry.prepare([[0, 0, 0, 1, 0, 0]])
        }
        do {
            _ = try await preparation.value
            XCTFail("Canceled preparation must not publish drawing resources.")
        } catch is CancellationError { }
    }

    @MainActor
    func testStrokeBuffersRejectInvalidSegmentsAndContainFiniteIndexedTriangles() throws {
        let geometry = CADStrokeGeometry.make([
            [-1, 0, 0, 1, 0, 0],
            [0, 0, 0, 0, 0, 0],
            [0, 0, 0, .nan, 1, 0],
            [0, 0, 0, 1, 0],
            [0, 0, 0, .infinity, 0, 0]
        ], color: .white, viewportHeight: 192)
        let positions = try XCTUnwrap(geometry.sources(for: .vertex).first)
        XCTAssertGreaterThan(positions.vectorCount, 0)
        for source in geometry.sources {
            XCTAssertEqual(source.vectorCount, positions.vectorCount)
            XCTAssertTrue(source.usesFloatComponents)
            for vector in 0..<source.vectorCount {
                for component in 0..<source.componentsPerVector {
                    XCTAssertTrue(try value(source, vector: vector, component: component).isFinite,
                                  "Malformed or degenerate input must never upload a NaN to Metal.")
                }
            }
        }
        for element in geometry.elements {
            XCTAssertEqual(element.primitiveType, .triangles)
            for offset in 0..<(element.primitiveCount * 3) {
                let index = try index(element, at: offset)
                XCTAssertGreaterThanOrEqual(index, 0)
                XCTAssertLessThan(index, positions.vectorCount)
            }
        }
        let empty = CADStrokeGeometry.make([[0, 0, 0, 0, 0, 0]], color: .white, viewportHeight: 192)
        XCTAssertEqual(empty.elements.reduce(0) { $0 + $1.primitiveCount }, 0)
    }

    @MainActor
    func testOrdinaryEdgeRetainsItsScreenWidthThroughBothProjectionZooms() throws {
        for orthographic in [false, true] {
            let geometry = CADStrokeGeometry.make([[-1, 0, 0, 1, 0, 0]], color: .white, viewportHeight: 192)
            let nearWidth = try strokeWidth(render(geometry, orthographic: orthographic, zoomedOut: false))
            let farWidth = try strokeWidth(render(geometry, orthographic: orthographic, zoomedOut: true))
            XCTAssertGreaterThan(nearWidth, 0, "Ordinary CAD ink must render on Metal.")
            XCTAssertEqual(Double(nearWidth), Double(CADStrokeGeometry.normalWidth), accuracy: 1.5)
            XCTAssertEqual(Double(nearWidth), Double(farWidth), accuracy: 1,
                "Ordinary edges must retain their screen width while zooming either projection.")
        }
    }

    @MainActor
    func testCopiedStrokeParametersRespectLogicalViewportAndRenderTargetResolution() throws {
        let original = CADStrokeGeometry.make([[-1, 0, 0, 1, 0, 0]], color: .white, viewportHeight: 192)
        CADStrokeGeometry.setViewportHeight(192, on: original)
        let copy = try XCTUnwrap(original.copy() as? SCNGeometry)
        CADStrokeGeometry.copyParameters(from: original, to: copy)
        XCTAssertEqual(try number(copy, key: CADStrokeGeometry.viewportHeightKey), 192)
        let normal = try strokeWidth(render(copy, orthographic: true, zoomedOut: false))
        let retina = try strokeWidth(render(copy, orthographic: true, zoomedOut: false, size: CGSize(width: 512, height: 384)))
        XCTAssertEqual(Double(retina), Double(normal * 2), accuracy: 2,
                       "Cloned agent screenshots need the same logical stroke width at their actual render resolution.")
        CADStrokeGeometry.setViewportHeight(384, on: copy)
        let resized = try strokeWidth(render(copy, orthographic: true, zoomedOut: false, size: CGSize(width: 512, height: 384)))
        XCTAssertEqual(Double(resized), Double(normal), accuracy: 1)
    }

    @MainActor
    func testOrdinaryEdgesRemainOccludedByFrontFaces() throws {
        for orthographic in [false, true] {
            for z in [-0.5, -0.01] {
                let behind = CADStrokeGeometry.make([[-0.7, 0, z, 0.7, 0, z]], color: .white, viewportHeight: 192)
                XCTAssertTrue(try XCTUnwrap(behind.firstMaterial).readsFromDepthBuffer)
                XCTAssertEqual(try strokeWidth(render(behind, orthographic: orthographic, zoomedOut: false, occluder: true)), 0,
                               "Ordinary CAD ink must not reveal an edge just behind a solid face (z=\(z)).")
            }
            for z in [0.0, 0.1] {
                let visible = CADStrokeGeometry.make([[-0.7, 0, z, 0.7, 0, z]], color: .white, viewportHeight: 192)
                XCTAssertGreaterThan(try strokeWidth(render(visible, orthographic: orthographic, zoomedOut: false, occluder: true)), 0,
                                     "Front-facing and exactly coplanar edges must remain visible (z=\(z)).")
            }
        }
    }

    @MainActor
    func testCircularCreaseEdgesRenderInNativeCaptures() async throws {
        _ = try XCTUnwrap(MTLCreateSystemDefaultDevice())
        let view = CADSceneView(frame: NSRect(x: 0, y: 0, width: 512, height: 512))
        let preview = circularPreview()
        let colors = [0: PartColor(red: 0.38, green: 0.67, blue: 0.82)]
        view.update(mesh: preview, mode: .edges, grid: false, command: ViewportCommand(.top),
                    selectedSolids: [], hiddenSolids: [], projection: .isometric, solidColors: colors)
        view.layout()
        try await ready(view)
        let baseline = try await CADAgentRenderer.capturePNG(view: view)
        try baseline.write(to: URL(fileURLWithPath: "/tmp/rad-edge-baseline.png"))
        try assertCircularCoverage(baseline, radius: 14, view: view, gold: false, label: "Inner crease edge")
        try assertCircularCoverage(baseline, radius: 18, view: view, gold: false, label: "Outer crease edge")
        XCTAssertTrue(seeds(in: view).isEmpty, "Ordinary crease edges need no highlight overlay.")
    }

    @MainActor
    private func seeds(in view: CADSceneView) -> [CADViewportFrame.Seeds] {
        view.makeFrame(width: Int(view.bounds.width), height: Int(view.bounds.height)).seeds
    }

    @MainActor
    private func assertCircularCoverage(_ data: Data, radius: Double, view: CADSceneView, gold: Bool, label: String,
                                        file: StaticString = #filePath, line: UInt = #line) throws {
        let image = try XCTUnwrap(NSBitmapImageRep(data: data), file: file, line: line)
        let scaleX = CGFloat(image.pixelsWide) / view.bounds.width
        let scaleY = CGFloat(image.pixelsHigh) / view.bounds.height
        let samples = 360
        var covered = 0
        for index in 0..<samples {
            // Sample between tessellation vertices too: depth round-off can
            // leave their endpoints intact while making long arcs disappear.
            let angle = (Double(index) + 0.37) * 2 * Double.pi / Double(samples)
            let projected = view.projectPoint(SCNVector3(radius * cos(angle), radius * sin(angle), 0))
            let x = Int((CGFloat(projected.x) * scaleX).rounded())
            let y = image.pixelsHigh - 1 - Int((CGFloat(projected.y) * scaleY).rounded())
            var found = false
            for dy in -1...1 {
                for dx in -1...1 {
                    guard x + dx >= 0, x + dx < image.pixelsWide, y + dy >= 0, y + dy < image.pixelsHigh,
                          let color = image.colorAt(x: x + dx, y: y + dy)?.usingColorSpace(.sRGB) else { continue }
                    if gold {
                        found = found || (color.redComponent > 0.5 && color.redComponent > color.blueComponent + 0.12 &&
                                          color.greenComponent > color.blueComponent + 0.08)
                    } else {
                        found = found || (color.redComponent < 0.30 && color.greenComponent < 0.35 && color.blueComponent < 0.40)
                    }
                }
            }
            if found { covered += 1 }
        }
        XCTAssertGreaterThanOrEqual(Double(covered) / Double(samples), 0.99,
                                    "\(label) must be continuous rather than dashed by depth fighting (\(covered)/\(samples) samples visible).",
                                    file: file, line: line)
    }

    @MainActor
    private func render(_ geometry: SCNGeometry, orthographic: Bool, zoomedOut: Bool,
                        size: CGSize = CGSize(width: 256, height: 192), occluder: Bool = false) throws -> NSBitmapImageRep {
        let device = try XCTUnwrap(MTLCreateSystemDefaultDevice())
        let scene = SCNScene()
        scene.background.contents = NSColor.black
        let camera = SCNNode()
        camera.camera = SCNCamera()
        camera.camera?.usesOrthographicProjection = orthographic
        camera.camera?.orthographicScale = zoomedOut ? 4 : 2
        camera.camera?.zNear = 0.1
        camera.camera?.zFar = 100
        camera.camera?.wantsHDR = false
        camera.position = SCNVector3(0, 0, zoomedOut ? 8 : 4)
        scene.rootNode.addChildNode(camera)
        if occluder {
            let plane = SCNPlane(width: 2, height: 2)
            let material = SCNMaterial()
            material.lightingModel = .constant
            material.diffuse.contents = NSColor(srgbRed: 0.05, green: 0.15, blue: 0.3, alpha: 1)
            material.isDoubleSided = true
            plane.materials = [material]
            scene.rootNode.addChildNode(SCNNode(geometry: plane))
        }
        let stroke = SCNNode(geometry: geometry)
        stroke.renderingOrder = 2
        scene.rootNode.addChildNode(stroke)
        let renderer = SCNRenderer(device: device, options: nil)
        renderer.scene = scene
        renderer.pointOfView = camera
        renderer.autoenablesDefaultLighting = false
        XCTAssertTrue(renderer.prepare(scene, shouldAbortBlock: nil))
        let image = renderer.snapshot(atTime: 0, with: size, antialiasingMode: .multisampling4X)
        return try XCTUnwrap(NSBitmapImageRep(data: XCTUnwrap(image.tiffRepresentation)))
    }

    private func strokeWidth(_ image: NSBitmapImageRep) throws -> Int {
        let x = image.pixelsWide / 2
        return try (0..<image.pixelsHigh).reduce(0) { count, y in
            let color = try XCTUnwrap(image.colorAt(x: x, y: y)?.usingColorSpace(.sRGB))
            return count + (color.redComponent > 0.2 && color.greenComponent > 0.2 && color.blueComponent > 0.2 ? 1 : 0)
        }
    }

    private func value(_ source: SCNGeometrySource, vector: Int, component: Int) throws -> Double {
        let offset = source.dataOffset + vector * source.dataStride + component * source.bytesPerComponent
        XCTAssertLessThanOrEqual(offset + source.bytesPerComponent, source.data.count)
        return try source.data.withUnsafeBytes { bytes in
            if source.bytesPerComponent == 4 { return Double(bytes.loadUnaligned(fromByteOffset: offset, as: Float.self)) }
            if source.bytesPerComponent == 8 { return bytes.loadUnaligned(fromByteOffset: offset, as: Double.self) }
            throw XCTSkip("Unexpected floating-point vertex component size.")
        }
    }

    private func index(_ element: SCNGeometryElement, at index: Int) throws -> Int {
        let offset = index * element.bytesPerIndex
        let data = try XCTUnwrap(element.data)
        XCTAssertLessThanOrEqual(offset + element.bytesPerIndex, data.count)
        return try data.withUnsafeBytes { bytes in
            switch element.bytesPerIndex {
            case 1: return Int(bytes.loadUnaligned(fromByteOffset: offset, as: UInt8.self))
            case 2: return Int(bytes.loadUnaligned(fromByteOffset: offset, as: UInt16.self))
            case 4: return Int(bytes.loadUnaligned(fromByteOffset: offset, as: UInt32.self))
            default: throw XCTSkip("Unexpected geometry index size.")
            }
        }
    }

    @MainActor
    private func number(_ geometry: SCNGeometry, key: String) throws -> Double {
        try XCTUnwrap(geometry.value(forKey: key) as? NSNumber).doubleValue
    }

    @MainActor
    private func ready(_ view: CADSceneView) async throws {
        let deadline = ContinuousClock.now + .seconds(10)
        while view.isPreparingResources, ContinuousClock.now < deadline { try await Task.sleep(for: .milliseconds(5)) }
        XCTAssertFalse(view.isPreparingResources)
    }

    @MainActor
    private func mouse(_ type: NSEvent.EventType, at point: NSPoint) throws -> NSEvent {
        if type == .mouseExited {
            return try XCTUnwrap(NSEvent.enterExitEvent(with: type, location: point, modifierFlags: [], timestamp: 0,
                windowNumber: 0, context: nil, eventNumber: 1, trackingNumber: 0, userData: nil))
        }
        return try XCTUnwrap(NSEvent.mouseEvent(with: type, location: point, modifierFlags: [], timestamp: 0,
            windowNumber: 0, context: nil, eventNumber: 1, clickCount: 0, pressure: 0))
    }

    private func circularPreview() -> PreviewMesh {
        let count = 240
        func circle(_ radius: Double) -> [[Double]] {
            (0...count).map { index in
                let angle = Double(index) * 2 * Double.pi / Double(count)
                return [radius * cos(angle), radius * sin(angle), 0]
            }
        }
        let outer = circle(18), inner = circle(14)
        let segments = [outer, inner].flatMap { points in
            zip(points, points.dropFirst()).map { $0.0 + $0.1 }
        }
        let triangles = (0..<count).map { [0, $0 + 1, ($0 + 1) % count + 1] }
        let preview = PreviewMesh(vertices: [[0, 0, 0]] + Array(outer.dropLast()), triangles: triangles,
            edges: segments, bounds: [-18, -18, 0, 18, 18, 0], volume: 0, area: .pi * 18 * 18,
            solids: 1, buildSeconds: 0, threads: 1, buildID: "/tmp/viewport-stroke-circle",
            triangleSolids: Array(repeating: 0, count: count), edgeSolids: Array(repeating: 0, count: count * 2))
        return preview
    }
}
