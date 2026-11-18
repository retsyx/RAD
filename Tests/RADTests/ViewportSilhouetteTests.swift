import AppKit
import Metal
import SceneKit
import simd
import XCTest
@testable import RAD

final class ViewportSilhouetteTests: XCTestCase {
    @MainActor
    func testNestedSelectedItemsRetainTheirIndividualProjectionOutlines() throws {
        let scene = SCNScene()
        scene.background.contents = NSColor.black
        let camera = SCNNode()
        camera.camera = SCNCamera()
        camera.camera?.usesOrthographicProjection = true
        camera.camera?.orthographicScale = 2.5
        camera.position = SCNVector3(0, 0, 6)
        scene.rootNode.addChildNode(camera)
        // Overlapping items keep separate projection passes.
        let batches = try [Float(3), Float(1.5)].map { width in
            CADViewportFrame.Batch(rect: .full, masks: [try ViewportPassTestRenderer.mask(
                triangles: square(width), selected: true)])
        }
        let shot = try ViewportPassTestRenderer.render(scene: scene, camera: camera, width: 320, height: 240,
            referenceHeight: 240, palette: palette(), batches: batches)
        let bitmap = shot.image, renderer = shot.renderer
        for distance in [0.75, 1.5] {
            for position in [SCNVector3(distance, 0, 0), SCNVector3(0, distance, 0),
                             SCNVector3(-distance, 0, 0), SCNVector3(0, -distance, 0)] {
                let point = renderer.projectPoint(position)
                XCTAssertTrue((-2...2).contains { dy in (-2...2).contains { dx in
                    isGold(bitmap, x: Int(point.x.rounded()) + dx,
                        y: bitmap.pixelsHigh - Int(point.y.rounded()) + dy)
                } }, "An overlapping selection must retain each item's outline instead of outlining only their union.")
            }
        }
        XCTAssertFalse(isGold(bitmap, x: bitmap.pixelsWide / 2, y: bitmap.pixelsHigh / 2),
            "The silhouette overlay must not fill either selected projection.")
    }

    @MainActor
    func testSelectedCubeUsesOnlyItsProjectedOutlineThroughANearerSolid() throws {
        for orthographic in [false, true] {
            let scene = SCNScene()
            scene.background.contents = NSColor.black
            let camera = SCNNode()
            camera.camera = SCNCamera()
            camera.camera?.usesOrthographicProjection = orthographic
            camera.camera?.orthographicScale = 2.6
            camera.camera?.fieldOfView = 38
            camera.camera?.zNear = 0.1
            camera.camera?.zFar = 30
            camera.position = SCNVector3(4, 3, 6)
            camera.look(at: SCNVector3Zero)
            scene.rootNode.addChildNode(camera)

            let occluder = SCNNode(geometry: SCNPlane(width: 20, height: 20))
            occluder.geometry?.materials = [constantMaterial(NSColor(srgbRed: 0.12, green: 0.14, blue: 0.18, alpha: 1))]
            occluder.simdPosition = simd_normalize(camera.simdPosition) * 3
            occluder.look(at: camera.position)
            scene.rootNode.addChildNode(occluder)

            let cube = try ViewportPassTestRenderer.mask(triangles: box(), selected: true)
            let shot = try ViewportPassTestRenderer.render(scene: scene, camera: camera, width: 384, height: 288,
                referenceHeight: 288, palette: palette(), batches: [.init(rect: .full, masks: [cube])])
            let bitmap = shot.image, renderer = shot.renderer
            let corners = [-1.0, 1.0].flatMap { x in [-1.0, 1.0].flatMap { y in [-1.0, 1.0].map { z in
                let point = renderer.projectPoint(SCNVector3(x, y, z))
                return SIMD2<Double>(Double(point.x), Double(bitmap.pixelsHigh) - Double(point.y))
            } } }
            let boundary = hull(corners)
            XCTAssertEqual(boundary.count, 6, "The oblique cube fixture must have a hexagonal projection.")
            var goldCount = 0, interiorCount = 0, covered = 0, sampleCount = 0
            for y in 0..<bitmap.pixelsHigh {
                for x in 0..<bitmap.pixelsWide where isGold(bitmap, x: x, y: y) {
                    goldCount += 1
                    let distance = distanceToBoundary(SIMD2(Double(x) + 0.5, Double(y) + 0.5), polygon: boundary)
                    if distance > 7 { interiorCount += 1 }
                }
            }
            for (start, end) in zip(boundary, Array(boundary.dropFirst()) + [boundary[0]]) {
                let steps = Int(simd_length(end - start))
                for step in 0...steps {
                    let point = start + (end - start) * Double(step) / Double(steps)
                    sampleCount += 1
                    if (-3...3).contains(where: { dy in (-3...3).contains { dx in
                        isGold(bitmap, x: Int(point.x.rounded()) + dx, y: Int(point.y.rounded()) + dy)
                    } }) { covered += 1 }
                }
            }
            XCTAssertGreaterThan(goldCount, 400, "The outline must render behind the independent nearer plane.")
            XCTAssertEqual(interiorCount, 0, "A selected cube must not draw its internal projected edges or triangle seams.")
            XCTAssertGreaterThan(Double(covered) / Double(sampleCount), 0.97,
                "The outline must cover the complete projection boundary in both camera projections.")
        }
    }

    @MainActor
    func testProductionPartHighlightsKeepOccludedOutlinesAndDepthTestTheirFill() async throws {
        _ = try XCTUnwrap(MTLCreateSystemDefaultDevice())
        let view = CADSceneView(frame: NSRect(x: 0, y: 0, width: 320, height: 240))
        let preview = squares()
        do {
            for occluded in [false, true] {
                view.update(mesh: preview, mode: .shaded, grid: false, command: nil,
                    selectedSolids: [0], hiddenSolids: occluded ? [] : [1])
                let deadline = ContinuousClock.now + .seconds(10)
                while view.isPreparingResources, ContinuousClock.now < deadline {
                    try await Task.sleep(for: .milliseconds(5))
                }
                XCTAssertFalse(view.isPreparingResources)
                let camera = try XCTUnwrap(view.pointOfView)
                camera.position = SCNVector3(0, 0, 6)
                camera.look(at: SCNVector3Zero)
                camera.camera?.usesOrthographicProjection = true
                camera.camera?.orthographicScale = 2.5
                let png = try await CADAgentRenderer.capturePNG(view: view)
                let image = try XCTUnwrap(NSBitmapImageRep(data: png))
                let middle = try XCTUnwrap(image.colorAt(x: image.pixelsWide / 2, y: image.pixelsHigh / 2)?.usingColorSpace(.sRGB))
                if occluded {
                    XCTAssertLessThan(middle.redComponent - middle.blueComponent, 0.1,
                        "The highlight fill must remain behind nearer geometry in the normal surface pass.")
                    var outlinePixels = 0
                    for y in 0..<image.pixelsHigh {
                        for x in 0..<image.pixelsWide where isGold(image, x: x, y: y) { outlinePixels += 1 }
                    }
                    XCTAssertGreaterThan(outlinePixels, 300,
                        "Selected parts need their complete no-depth outline in captures.")
                } else {
                    XCTAssertGreaterThan(middle.redComponent - middle.blueComponent, 0.15,
                        "Visible selected parts must receive the highlight color in the normal pass.")
                }
            }
        }
    }

    @MainActor
    private func bitmap(_ view: CADSceneView) async throws -> NSBitmapImageRep {
        let png = try await CADAgentRenderer.capturePNG(view: view)
        return try XCTUnwrap(NSBitmapImageRep(data: png))
    }

    @MainActor
    private func ready(_ view: CADSceneView) async throws {
        let deadline = ContinuousClock.now + .seconds(10)
        while view.isPreparingResources, ContinuousClock.now < deadline {
            try await Task.sleep(for: .milliseconds(5))
        }
        XCTAssertFalse(view.isPreparingResources)
    }

    @MainActor
    private func event(_ type: NSEvent.EventType, at point: NSPoint) throws -> NSEvent {
        if type == .mouseExited {
            return try XCTUnwrap(NSEvent.enterExitEvent(with: type, location: point, modifierFlags: [], timestamp: 0,
                windowNumber: 0, context: nil, eventNumber: 1, trackingNumber: 0, userData: nil))
        }
        return try XCTUnwrap(NSEvent.mouseEvent(with: type, location: point, modifierFlags: [], timestamp: 0,
            windowNumber: 0, context: nil, eventNumber: 1, clickCount: 0, pressure: 0))
    }

    @MainActor
    private func hasGold(_ image: NSBitmapImageRep, around position: SCNVector3, in view: CADSceneView,
                         hover: Bool = false) -> Bool {
        let projected = view.projectPoint(position)
        let x = Int((CGFloat(projected.x) * CGFloat(image.pixelsWide) / view.bounds.width).rounded())
        let y = image.pixelsHigh - Int((CGFloat(projected.y) * CGFloat(image.pixelsHigh) / view.bounds.height).rounded())
        let linear = hover ? CADSceneView.highlightPalette.hover : CADSceneView.highlightPalette.selected
        func gamma(_ component: Float) -> CGFloat {
            let value = CGFloat(component)
            return value <= 0.0031308 ? 12.92 * value : 1.055 * pow(value, 1 / 2.4) - 0.055
        }
        let reference = [gamma(linear.x), gamma(linear.y), gamma(linear.z)]
        return (-3...3).contains { dy in (-3...3).contains { dx in
            let column = x + dx, row = y + dy
            guard column >= 0, row >= 0, column < image.pixelsWide, row < image.pixelsHigh,
                  let color = image.colorAt(x: column, y: row) else { return false }
            // colorAt labels the PNG's encoded RGB samples as calibrated RGB.
            // Compare the stored channels directly with the shader's sRGB
            // output; converting that label would apply another gamma curve.
            return max(abs(color.redComponent - reference[0]), abs(color.greenComponent - reference[1]),
                abs(color.blueComponent - reference[2])) < 0.04
        } }
    }

    private func palette() -> CADHighlightPalette {
        CADHighlightPalette(selected: NSColor(srgbRed: 0.96, green: 0.63, blue: 0.23, alpha: 1),
                            hover: .green, hoverHalo: .darkGray, ink: .black, background: .black)
    }

    private func square(_ width: Float) -> [[SIMD3<Float>]] {
        let h = width / 2
        return [[SIMD3(-h, -h, 0), SIMD3(h, -h, 0), SIMD3(h, h, 0)], [SIMD3(-h, -h, 0), SIMD3(h, h, 0), SIMD3(-h, h, 0)]]
    }

    /// Triangles of an axis-aligned 2×2×2 cube centered at the origin.
    private func box() -> [[SIMD3<Float>]] {
        var triangles = [[SIMD3<Float>]]()
        for axis in 0..<3 {
            for sign in [Float(-1), 1] {
                func corner(_ u: Float, _ v: Float) -> SIMD3<Float> {
                    var point = SIMD3<Float>(repeating: 0)
                    point[axis] = sign; point[(axis + 1) % 3] = u; point[(axis + 2) % 3] = v
                    return point
                }
                triangles += [[corner(-1, -1), corner(1, -1), corner(1, 1)], [corner(-1, -1), corner(1, 1), corner(-1, 1)]]
            }
        }
        return triangles
    }

    private func constantMaterial(_ color: NSColor) -> SCNMaterial {
        let material = SCNMaterial()
        material.lightingModel = .constant
        material.diffuse.contents = color
        material.isDoubleSided = true
        return material
    }

    private func isGold(_ image: NSBitmapImageRep, x: Int, y: Int) -> Bool {
        guard x >= 0, y >= 0, x < image.pixelsWide, y < image.pixelsHigh,
              let color = image.colorAt(x: x, y: y)?.usingColorSpace(.sRGB) else { return false }
        return color.redComponent > 0.7 && color.greenComponent > 0.4 && color.blueComponent < 0.5
    }

    private func hull(_ points: [SIMD2<Double>]) -> [SIMD2<Double>] {
        let sorted = points.sorted { $0.x == $1.x ? $0.y < $1.y : $0.x < $1.x }
        func cross(_ a: SIMD2<Double>, _ b: SIMD2<Double>, _ c: SIMD2<Double>) -> Double {
            let first = b - a, second = c - a
            return first.x * second.y - first.y * second.x
        }
        func chain(_ points: [SIMD2<Double>]) -> [SIMD2<Double>] {
            var result = [SIMD2<Double>]()
            for point in points {
                while result.count >= 2, cross(result[result.count - 2], result[result.count - 1], point) <= 0 {
                    result.removeLast()
                }
                result.append(point)
            }
            return Array(result.dropLast())
        }
        return chain(sorted) + chain(Array(sorted.reversed()))
    }

    private func distanceToBoundary(_ point: SIMD2<Double>, polygon: [SIMD2<Double>]) -> Double {
        zip(polygon, Array(polygon.dropFirst()) + [polygon[0]]).map { start, end in
            let edge = end - start
            let phase = max(0, min(1, simd_dot(point - start, edge) / simd_length_squared(edge)))
            return simd_length(point - start - edge * phase)
        }.min()!
    }

    private func squares() -> PreviewMesh {
        let preview = PreviewMesh(vertices: [[-1, -1, 0], [1, -1, 0], [1, 1, 0], [-1, 1, 0],
                                            [-2, -2, 0.3], [2, -2, 0.3], [2, 2, 0.3], [-2, 2, 0.3]],
            triangles: [[0, 1, 2], [0, 2, 3], [4, 5, 6], [4, 6, 7]], edges: [],
            bounds: [-2, -2, 0, 2, 2, 0.3], volume: 1, area: 20, solids: 2,
            buildSeconds: 0, threads: 1, buildID: "/tmp/viewport-silhouette-squares",
            triangleSolids: [0, 0, 1, 1], edgeSolids: [])
        return preview
    }
}
