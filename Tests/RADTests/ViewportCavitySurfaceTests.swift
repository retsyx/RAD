import AppKit
import Metal
import SceneKit
import simd
import XCTest
@testable import RAD

final class ViewportCavitySurfaceTests: XCTestCase {
    @MainActor
    func testHollowSphereKeepsEveryTriangleFacingAlongItsShadingNormals() async throws {
        guard ProcessInfo.processInfo.environment["RAD_RUN_ENGINE_TESTS"] == "1" else {
            throw XCTSkip("Set RAD_RUN_ENGINE_TESTS=1 to verify the native hollow-sphere preview and Metal rendering.")
        }
        _ = NSApplication.shared
        _ = try XCTUnwrap(MTLCreateSystemDefaultDevice())
        let spec = ModelSpec(title: "Hollow sphere with cable opening",
                             summary: "Regression for inverted display triangles on a spherical cavity.",
                             units: "mm", script: "result = sphere(50) - sphere(48) - cylinder(6, 60, center=(0, 0, 50))")
        let preview = try await EngineService().buildPreview(spec, python: EngineService.defaultPython, threads: 2)
        // The nested-array views unpack the whole mesh on each access.
        let normals = try XCTUnwrap(preview.normals), vertices = preview.vertices
        var cavity = 0, inverted = [Int]()
        func vector(_ values: [Double]) -> SIMD3<Double> { SIMD3(values[0], values[1], values[2]) }
        for (index, triangle) in preview.triangles.enumerated() {
            let a = vector(vertices[triangle[0]])
            let b = vector(vertices[triangle[1]])
            let c = vector(vertices[triangle[2]])
            let plane = simd_cross(b - a, c - a)
            guard simd_length_squared(plane) > 1e-20 else { continue }
            if simd_length((a + b + c) / 3) < 49 {
                cavity += 1
                // The cavity's outward side faces its center.
                XCTAssertLessThan(simd_dot(plane, (a + b + c) / 3), 0, "Cavity triangle \(index) faces away from the cavity.")
            }
            let shading = triangle.reduce(SIMD3<Double>.zero) { $0 + vector(normals[$1]) }
            if simd_dot(plane, shading) <= 0 { inverted.append(index) }
        }
        XCTAssertGreaterThan(cavity, 100)
        XCTAssertTrue(inverted.isEmpty, "Display triangles must agree with their shading normals: \(inverted.prefix(12))")

        for mode in [ViewportDisplayMode.shaded, .edges] {
            let view = CADSceneView(frame: NSRect(x: 0, y: 0, width: 600, height: 600))
            view.update(mesh: preview, mode: mode, grid: false, command: nil,
                        selectedSolids: [], hiddenSolids: [], projection: .perspective)
            let deadline = ContinuousClock.now + .seconds(30)
            while view.isPreparingResources, ContinuousClock.now < deadline {
                try await Task.sleep(for: .milliseconds(10))
            }
            XCTAssertFalse(view.isPreparingResources)
            try view.configureAgentView(named: "top", camera: .init(zoom: 1.5))
            view.layout()
            let png = try await CADAgentRenderer.capturePNG(view: view)
            let image = try XCTUnwrap(NSBitmapImageRep(data: png))
            // Looking into the opening shows the lit cavity floor, not a dark hole.
            let center = try XCTUnwrap(image.colorAt(x: image.pixelsWide / 2, y: image.pixelsHigh / 2)?.usingColorSpace(.sRGB))
            XCTAssertGreaterThan(center.greenComponent, 0.3, "The cavity behind the opening must render its interior surface.")
        }
    }
}
