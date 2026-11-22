import AppKit
import Metal
import SceneKit
import simd

/// SceneKit node categories drawn by the viewport's scene passes. Hit testing
/// queries `pickable`.
enum CADViewportCategory {
    /// Faces of solids drawn from their display meshes.
    static let faces = 1
    static let decorations = 2
    /// Translucent mode's depth-only copies of each mesh solid's faces.
    static let visibility = 8
    /// Translucent mode's transmittance copies of each mesh solid's faces.
    static let revealage = 16
    /// The geometry-free node that records each frame's camera matrices.
    static let probe = 32
    /// Display meshes of solids with exact faces, and their translucent
    /// copies: drawn only by mesh frames, always hit tested and framed.
    static let meshFaces = 64
    static let meshVisibility = 128
    static let meshRevealage = 256
    /// The geometry-free nodes that draw exact faces in each scene pass.
    static let fieldOpaque = 512
    static let fieldVisibility = 1024
    static let fieldAccumulation = 2048
    static let fieldRevealage = 4096

    static let pickable = faces | meshFaces
    static let opaqueScene = faces | decorations | probe | fieldOpaque
    static let translucentDecorations = visibility | decorations | probe | fieldVisibility
    static let translucentAccumulation = faces | fieldAccumulation
    static let translucentRevealage = revealage | fieldRevealage
    static let opaqueMeshScene = faces | meshFaces | decorations | probe
    static let translucentMeshDecorations = visibility | meshVisibility | decorations | probe
    static let translucentMeshAccumulation = faces | meshFaces
    static let translucentMeshRevealage = revealage | meshRevealage
}

/// Translucent faces keep this fraction of their complete shaded color.
enum CADTranslucency {
    static let alpha: Float = 0.28
}

/// Linear highlight colors and the scene background used by the compositor.
struct CADHighlightPalette: Sendable {
    var selected: SIMD4<Float>
    var hover: SIMD4<Float>
    var hoverHalo: SIMD4<Float>
    var ink: SIMD4<Float>
    var background: SIMD4<Float>

    init(selected: NSColor, hover: NSColor, hoverHalo: NSColor, ink: NSColor, background: NSColor) {
        self.selected = Self.linear(selected)
        self.hover = Self.linear(hover)
        self.hoverHalo = Self.linear(hoverHalo)
        self.ink = Self.linear(ink)
        self.background = Self.linear(background)
    }

    static func linear(_ color: NSColor) -> SIMD4<Float> {
        let color = color.usingColorSpace(.extendedSRGB) ?? color
        func linear(_ component: CGFloat) -> Float {
            let value = Double(component)
            return Float(value <= 0.04045 ? value / 12.92 : pow((value + 0.055) / 1.055, 2.4))
        }
        return SIMD4(linear(color.redComponent), linear(color.greenComponent), linear(color.blueComponent),
                     Float(color.alphaComponent))
    }
}

/// Immutable Metal geometry for unoccluded highlights. The live viewport and
/// independent captures share these buffers; nothing writes them after creation.
enum CADHighlightBuffers {
    /// Segment centerlines (line list) or a point marker's center (one point).
    final class Seeds: @unchecked Sendable {
        let positions: any MTLBuffer
        let vertexCount: Int
        let primitive: MTLPrimitiveType
        /// Local bounds, for the highlighted screen area.
        let bounds: [Double]

        init?(device: any MTLDevice, points: [SIMD3<Float>], primitive: MTLPrimitiveType) {
            guard !points.isEmpty else { return nil }
            var packed = [Float]()
            packed.reserveCapacity(points.count * 3)
            var box = CADSilhouetteBatching.Bounds()
            for point in points {
                packed += [point.x, point.y, point.z]
                box.include(point)
            }
            guard let buffer = device.makeBuffer(bytes: packed, length: packed.count * MemoryLayout<Float>.stride,
                                                 options: .storageModeShared) else { return nil }
            positions = buffer
            vertexCount = points.count
            self.primitive = primitive
            bounds = box.values
        }
    }

    /// A part projected as filled coverage for its outline.
    final class Mask: @unchecked Sendable {
        let positions: any MTLBuffer
        let triangles: (any MTLBuffer)?
        let triangleIndexCount: Int

        init?(device: any MTLDevice, positions: any MTLBuffer, triangleIndices: Data) {
            let count = triangleIndices.count / MemoryLayout<Int32>.stride
            guard count >= 3, let triangles = triangleIndices.withUnsafeBytes({ raw in
                raw.baseAddress.flatMap { device.makeBuffer(bytes: $0, length: count * MemoryLayout<Int32>.stride,
                                                            options: .storageModeShared) }
            }) else { return nil }
            let triangleCount = count - count % 3
            self.positions = positions
            self.triangles = triangles
            triangleIndexCount = triangleCount
        }
    }
}

/// Everything a frame needs besides SceneKit's committed scene state.
struct CADViewportFrame: @unchecked Sendable {
    enum SeedChannel: Int, Sendable { case selectedEdge, hoveredEdge, selectedVertex, hoveredVertex }
    struct Seeds {
        var buffer: CADHighlightBuffers.Seeds
        var channel: SeedChannel
        var model: simd_float4x4
    }
    struct Mask {
        var buffer: CADHighlightBuffers.Mask
        var selected: Bool
        var model: simd_float4x4
        /// Exact solids cover their field projection instead of their mesh.
        var field: (resources: CADFieldResources, solid: Int)? = nil
    }
    /// Disjoint masks share one projection and contour pass.
    struct Batch {
        var rect: CADSilhouetteBatching.Rect
        var masks: [Mask]
    }

    /// Output size in pixels.
    var width: Int
    var height: Int
    /// Logical viewport height in points; screen-point widths scale by
    /// output pixels per point.
    var referenceHeight: Double
    var translucent: Bool
    var palette: CADHighlightPalette
    var seeds: [Seeds] = []
    var batches: [Batch] = []
    /// Projected bounds of every seed and mask, before the highlight kernel.
    var highlightBounds: CADSilhouetteBatching.Rect?
    var verticesEnabled = false
    /// Exact faces; nil draws every solid from its display mesh.
    var field: CADFieldFrame?
    /// The identity of the exact layers this frame shows (with `field`).
    var key: CADExactKey?
    /// Published while the user navigates, and whether an exact frame of this
    /// design then fits the display's frame budget.
    var navigating = false
    var exactFitsBudget = true
    /// Revision of everything the frame shows besides its choice of exact or
    /// display-mesh faces.
    var presentation: UInt64 = 0

    var hasHighlights: Bool { !seeds.isEmpty || !batches.isEmpty }

    /// Screen-point scale of output pixels, matching the compositing shaders.
    var pixelScale: Double { min(8, max(0.125, Double(height) / max(referenceHeight, 1))) }

    /// Kernel reach in pixels for the active edge (and vertex) styles.
    var kernelRadius: Int {
        let radius = Double(verticesEnabled ? CADHighlightStyle.maximumRadius : CADHighlightStyle.maximumEdgeRadius)
        let largest = Int(ceil(Double(CADHighlightStyle.maximumRadius) * 8 + 0.5))
        return min(largest, Int(ceil(radius * pixelScale + 0.5)))
    }
}

/// The SceneKit objects one frame renders. The live viewport and every
/// capture own separate scene graphs; renderers only read committed state.
final class CADViewportSceneSource: @unchecked Sendable {
    static let decorationsCameraName = "radTranslucentDecorationsCamera"
    static let accumulationCameraName = "radTranslucentAccumulationCamera"
    static let revealageCameraName = "radTranslucentRevealageCamera"
    static let opaqueExactCameraName = "radExactOpaqueCamera"
    static let decorationsMeshCameraName = "radMeshTranslucentDecorationsCamera"
    static let accumulationMeshCameraName = "radMeshTranslucentAccumulationCamera"
    static let revealageMeshCameraName = "radMeshTranslucentRevealageCamera"
    static let probeCameraName = "radProbeCamera"
    static let probeName = "radFrameProbe"
    static let fieldNodeNames: [(CADFieldPassDelegate.Pass, String, Int)] = [
        (.opaque, "radFieldOpaque", CADViewportCategory.fieldOpaque),
        (.visibility, "radFieldVisibility", CADViewportCategory.fieldVisibility),
        (.accumulation, "radFieldAccumulation", CADViewportCategory.fieldAccumulation),
        (.revealage, "radFieldRevealage", CADViewportCategory.fieldRevealage),
    ]

    /// Pass cameras of one kind of frame: exact faces or display meshes.
    struct Cameras {
        var opaque: SCNNode
        var decorations: SCNNode
        var accumulation: SCNNode
        var revealage: SCNNode
    }

    let scene: SCNScene
    /// The design camera, also used for the opaque scene pass of mesh frames.
    let camera: SCNNode
    let exact: Cameras
    let mesh: Cameras
    /// Draws only the probe: the view and projection of the next scene passes.
    let probeCamera: SCNNode
    let probe: CADViewportFrameProbe
    /// The frame's exact faces, read by the field nodes' delegates.
    let field: CADFieldPassState
    /// Renderer delegates are not retained by their nodes.
    private let fieldDelegates: [CADFieldPassDelegate]
    /// The probe and field nodes, whose delegates are cleared while this
    /// source still holds the scene: SceneKit can crash deallocating a scene
    /// whose nodes still have renderer delegates.
    private let delegateNodes: [SCNNode]

    var decorationsCamera: SCNNode { exact.decorations }
    var accumulationCamera: SCNNode { exact.accumulation }
    var revealageCamera: SCNNode { exact.revealage }

    private init(scene: SCNScene, camera: SCNNode, exact: Cameras, mesh: Cameras, probeCamera: SCNNode,
                 probe: CADViewportFrameProbe, field: CADFieldPassState, fieldDelegates: [CADFieldPassDelegate],
                 delegateNodes: [SCNNode]) {
        self.scene = scene
        self.camera = camera
        self.exact = exact
        self.mesh = mesh
        self.probeCamera = probeCamera
        self.probe = probe
        self.field = field
        self.fieldDelegates = fieldDelegates
        self.delegateNodes = delegateNodes
    }

    deinit {
        for node in delegateNodes { node.rendererDelegate = nil }
    }

    /// Fresh delegates for the field nodes, sharing one frame state.
    private static func fieldDelegates() -> (CADFieldPassState, [CADFieldPassDelegate]) {
        let state = CADFieldPassState()
        return (state, fieldNodeNames.map { CADFieldPassDelegate(pass: $0.0, state: state) })
    }

    /// Adds the pass cameras under `camera` and the matrix probe and exact
    /// face nodes to `scene`. Pass cameras mirror the design camera's projection.
    static func install(in scene: SCNScene, camera: SCNNode) -> CADViewportSceneSource {
        camera.camera?.categoryBitMask = CADViewportCategory.opaqueMeshScene
        func passCamera(_ name: String, mask: Int) -> SCNNode {
            let node = SCNNode()
            node.name = name
            node.camera = SCNCamera()
            node.camera?.categoryBitMask = mask
            camera.addChildNode(node)
            return node
        }
        let exact = Cameras(
            opaque: passCamera(opaqueExactCameraName, mask: CADViewportCategory.opaqueScene),
            decorations: passCamera(decorationsCameraName, mask: CADViewportCategory.translucentDecorations),
            accumulation: passCamera(accumulationCameraName, mask: CADViewportCategory.translucentAccumulation),
            revealage: passCamera(revealageCameraName, mask: CADViewportCategory.translucentRevealage))
        let mesh = Cameras(
            opaque: camera,
            decorations: passCamera(decorationsMeshCameraName, mask: CADViewportCategory.translucentMeshDecorations),
            accumulation: passCamera(accumulationMeshCameraName, mask: CADViewportCategory.translucentMeshAccumulation),
            revealage: passCamera(revealageMeshCameraName, mask: CADViewportCategory.translucentMeshRevealage))
        let probeCamera = passCamera(probeCameraName, mask: CADViewportCategory.probe)
        // No geometry or bounds: these nodes must never change automatic
        // clipping. SceneKit calls a renderer delegate only if the node has
        // it when it joins the scene, so delegates are attached first.
        let probe = CADViewportFrameProbe()
        let probeNode = SCNNode()
        probeNode.name = probeName
        probeNode.categoryBitMask = CADViewportCategory.probe
        probeNode.renderingOrder = 1_000_000
        probeNode.rendererDelegate = probe
        scene.rootNode.addChildNode(probeNode)
        let (state, delegates) = fieldDelegates()
        var nodes = [probeNode]
        for ((_, name, category), delegate) in zip(fieldNodeNames, delegates) {
            let node = SCNNode()
            node.name = name
            node.categoryBitMask = category
            // Exact faces fill visibility depth before mesh solids and decorations.
            node.renderingOrder = -1
            node.rendererDelegate = delegate
            scene.rootNode.addChildNode(node)
            nodes.append(node)
        }
        let source = CADViewportSceneSource(scene: scene, camera: camera, exact: exact, mesh: mesh, probeCamera: probeCamera,
                                            probe: probe, field: state, fieldDelegates: delegates, delegateNodes: nodes)
        source.synchronizeCameras()
        return source
    }

    /// Finds the pass cameras, probe and field nodes in a cloned scene graph,
    /// attaching fresh delegates so the copy never reports into the original.
    static func adopt(scene: SCNScene, camera: SCNNode) -> CADViewportSceneSource? {
        func child(_ name: String) -> SCNNode? { camera.childNode(withName: name, recursively: false) }
        guard let decorations = child(decorationsCameraName), let accumulation = child(accumulationCameraName),
              let revealage = child(revealageCameraName), let opaqueExact = child(opaqueExactCameraName),
              let decorationsMesh = child(decorationsMeshCameraName), let accumulationMesh = child(accumulationMeshCameraName),
              let revealageMesh = child(revealageMeshCameraName), let probeCamera = child(probeCameraName),
              let probeNode = scene.rootNode.childNode(withName: probeName, recursively: true) else { return nil }
        var fieldNodes: [SCNNode] = []
        for (_, name, _) in fieldNodeNames {
            guard let node = scene.rootNode.childNode(withName: name, recursively: true) else { return nil }
            fieldNodes.append(node)
        }
        // Each node rejoins the scene with its new delegate, which SceneKit
        // registers only on insertion.
        func attach(_ delegate: any SCNNodeRendererDelegate, to node: SCNNode) {
            let parent = node.parent
            node.removeFromParentNode()
            node.rendererDelegate = delegate
            (parent ?? scene.rootNode).addChildNode(node)
        }
        let probe = CADViewportFrameProbe()
        attach(probe, to: probeNode)
        let (state, delegates) = fieldDelegates()
        for (node, delegate) in zip(fieldNodes, delegates) { attach(delegate, to: node) }
        return CADViewportSceneSource(
            scene: scene, camera: camera,
            exact: Cameras(opaque: opaqueExact, decorations: decorations, accumulation: accumulation, revealage: revealage),
            mesh: Cameras(opaque: camera, decorations: decorationsMesh, accumulation: accumulationMesh,
                          revealage: revealageMesh),
            probeCamera: probeCamera, probe: probe, field: state, fieldDelegates: delegates, delegateNodes: [probeNode] + fieldNodes)
    }

    /// Copies the design camera's projection into each pass camera.
    func synchronizeCameras() {
        guard let source = camera.camera else { return }
        for node in [exact.opaque, exact.decorations, exact.accumulation, exact.revealage, mesh.decorations,
                     mesh.accumulation, mesh.revealage, probeCamera] {
            guard let target = node.camera else { continue }
            if target.fieldOfView != source.fieldOfView { target.fieldOfView = source.fieldOfView }
            if target.projectionDirection != source.projectionDirection { target.projectionDirection = source.projectionDirection }
            if target.usesOrthographicProjection != source.usesOrthographicProjection {
                target.usesOrthographicProjection = source.usesOrthographicProjection
            }
            if target.orthographicScale != source.orthographicScale { target.orthographicScale = source.orthographicScale }
            if target.automaticallyAdjustsZRange != source.automaticallyAdjustsZRange {
                target.automaticallyAdjustsZRange = source.automaticallyAdjustsZRange
            }
            if target.zNear != source.zNear { target.zNear = source.zNear }
            if target.zFar != source.zFar { target.zFar = source.zFar }
            if target.wantsHDR != source.wantsHDR { target.wantsHDR = source.wantsHDR }
            if target.exposureOffset != source.exposureOffset { target.exposureOffset = source.exposureOffset }
        }
    }
}

/// Records the exact view and projection SceneKit used for a scene pass, so
/// highlight passes align with the rendered faces to the pixel.
final class CADViewportFrameProbe: NSObject, SCNNodeRendererDelegate, @unchecked Sendable {
    private let lock = NSLock()
    private var matrices: (view: simd_float4x4, projection: simd_float4x4)?

    func reset() { lock.withLock { matrices = nil } }

    func take() -> simd_float4x4? { lock.withLock { matrices.map { $0.projection * $0.view } } }

    /// The pass's view and projection, for passes that trace view rays.
    func takeMatrices() -> (view: simd_float4x4, projection: simd_float4x4)? { lock.withLock { matrices } }

    func renderNode(_ node: SCNNode, renderer: SCNRenderer, arguments: [String: Any]) {
        guard let view = (arguments[SCNViewTransform] as? NSValue)?.scnMatrix4Value,
              let projection = (arguments[SCNProjectionTransform] as? NSValue)?.scnMatrix4Value else { return }
        let value = (simd_float4x4(view), simd_float4x4(projection))
        lock.withLock { matrices = value }
    }
}

/// Encodes viewport frames: SceneKit scene passes into tile-memory
/// multisample targets, then RAD's own highlight passes. Multisample targets
/// never leave tile memory; highlight targets cover only the highlighted area
/// and exist only while something is highlighted. Exact faces are drawn into
/// stored layers, in screen slices when one frame cannot draw them at once,
/// and composited by the scene passes of every frame that shows them.
final class CADViewportPasses: @unchecked Sendable {
    static let colorFormat = MTLPixelFormat.bgra8Unorm_srgb
    static let sampleCount = 4
    /// GPU seconds of one exact slice: new input waits at most about this long.
    static let sliceBudget = 0.004

    /// Exact work encoded by these passes (diagnostics and tests).
    struct Statistics: Equatable, Sendable {
        var geometrySlices = 0
        var maskSlices = 0
        var composites = 0
    }

    let device: any MTLDevice
    private let pipelines: Pipelines
    private let exactPipelines: CADExactPipelines
    private var targets: Targets?
    private var stored: CADExactTargets?
    private let statisticsLock = NSLock()
    private var counts = Statistics()

    init(device: any MTLDevice) throws {
        self.device = device
        pipelines = try Pipelines.shared(for: device)
        exactPipelines = try CADExactPipelines.shared(for: device)
    }

    var statistics: Statistics { statisticsLock.withLock { counts } }
    private func count(_ change: (inout Statistics) -> Void) { statisticsLock.withLock { change(&counts) } }

    /// Bytes of the stored exact layers.
    var storedBytes: Int { stored?.bytes ?? 0 }

    /// Releases the stored exact layers; GPU work already encoded keeps its own.
    func releaseStoredLayers() { stored = nil }

    // MARK: Complete frames

    /// Encodes one complete frame into `target`, a single-sample `colorFormat`
    /// texture of the frame's size: display meshes when the frame has no exact
    /// faces, otherwise all of its exact layers at once and their composite.
    /// Returns false when the design camera changed while the frame was
    /// encoded, so it shows no exact faces. The renderer must not be used
    /// concurrently.
    @discardableResult
    func encode(_ frame: CADViewportFrame, source: CADViewportSceneSource, renderer: SCNRenderer,
                into target: any MTLTexture, commandBuffer: any MTLCommandBuffer) -> Bool {
        guard frame.field != nil, let job = makeJob(frame, source: source, renderer: renderer, commandBuffer: commandBuffer) else {
            encodeMesh(frame, source: source, renderer: renderer, into: target, commandBuffer: commandBuffer)
            return true
        }
        encodeAll(job, frame: frame, commandBuffer: commandBuffer)
        return encodeComposite(frame, job: job, source: source, renderer: renderer, into: target, commandBuffer: commandBuffer)
    }

    /// Encodes every remaining slice of a job and of its frame's outline
    /// coverage into one command buffer, which orders them before its composite.
    func encodeAll(_ job: CADExactJob, frame: CADViewportFrame, commandBuffer: any MTLCommandBuffer) {
        while encodeSlice(job, budget: .infinity, commandBuffer: commandBuffer) != nil {}
        job.completed = job.encoded
        prepareMasks(job, frame: frame)
        while let coverage = encodeMaskSlice(job, budget: .infinity, commandBuffer: commandBuffer) {
            coverage.completed = coverage.encoded
        }
        for coverage in job.masks.values { coverage.completed = coverage.encoded }
    }

    /// Encodes a frame that draws every solid from its display mesh.
    func encodeMesh(_ frame: CADViewportFrame, source: CADViewportSceneSource, renderer: SCNRenderer,
                    into target: any MTLTexture, commandBuffer: any MTLCommandBuffer) {
        _ = encodeScene(frame, composite: nil, job: nil, source: source, renderer: renderer, into: target,
                        commandBuffer: commandBuffer)
    }

    // MARK: Exact layers

    /// Starts an exact frame's layers: records the view and projection of
    /// the frame's scene passes with a probe-only scene pass (encoded into
    /// `commandBuffer`, which must be committed) and prepares the stored
    /// layers and the slice plan. Nil when the frame has no exact faces.
    func makeJob(_ frame: CADViewportFrame, source: CADViewportSceneSource, renderer: SCNRenderer,
                 commandBuffer: any MTLCommandBuffer) -> CADExactJob? {
        guard let field = frame.field, let key = frame.key, frame.width > 0, frame.height > 0,
              let targets = targets(width: frame.width, height: frame.height) else { return nil }
        prepare(renderer, source: source)
        source.probe.reset()
        renderScene(renderer, camera: source.probeCamera, viewport: CGRect(x: 0, y: 0, width: frame.width, height: frame.height),
                    commandBuffer: commandBuffer, color: targets.multisampleColor, resolve: nil, clear: MTLClearColor(),
                    depth: targets.multisampleDepth)
        guard let matrices = source.probe.takeMatrices() else { return nil }
        let revealage = key.translucent && !field.complete
        if stored?.fits(width: frame.width, height: frame.height, translucent: key.translucent, revealage: revealage) != true {
            stored = nil
            stored = CADExactTargets(device: device, width: frame.width, height: frame.height, translucent: key.translucent,
                                     revealage: revealage)
        }
        guard let layers = stored else { return nil }
        return CADExactJob(key: key, field: field, view: matrices.view, projection: matrices.projection,
                           width: frame.width, height: frame.height, targets: layers)
    }

    /// Encodes a job's next slice: its geometry within a screen rectangle
    /// whose estimated GPU time stays within `budget` seconds (the first
    /// slice also clears the stored layers). Returns it, or nil when every
    /// slice has been encoded.
    @discardableResult
    func encodeSlice(_ job: CADExactJob, budget: Double, commandBuffer: any MTLCommandBuffer) -> CADSlicePlan.Slice? {
        let covered = budget.isFinite ? max(1, Int(budget / CADExactCost.shared.sliceRate(job.costKey))) : Int.max
        var next = job.plan.next(budget: covered)
        if next == nil {
            guard !job.cleared else { return nil }
            next = CADSlicePlan.Slice(rect: .zero, covered: 0)
        }
        guard let slice = next else { return nil }
        let clear = !job.cleared
        job.cleared = true
        job.encoded += 1
        count { $0.geometrySlices += 1 }
        if job.key.translucent {
            encodeTranslucentSlice(job, slice: slice, clear: clear, commandBuffer: commandBuffer)
        } else {
            encodeOpaqueSlice(job, slice: slice, clear: clear, commandBuffer: commandBuffer)
        }
        return slice
    }

    private func fieldUniforms(_ job: CADExactJob) -> CADFieldResources.Uniforms {
        CADFieldResources.Uniforms(view: job.view, projection: job.projection, height: job.height, lights: job.field.lights,
                                   alpha: CADTranslucency.alpha, debug: job.field.debug,
                                   sampleOffsets: job.field.resources.sampleOffsets)
    }

    private func scissor(slice rect: SIMD4<Int32>) -> MTLScissorRect {
        MTLScissorRect(x: Int(rect.x), y: Int(rect.y), width: Int(rect.z - rect.x), height: Int(rect.w - rect.y))
    }

    /// Opaque geometry: each sample's normal, part and depth.
    private func encodeOpaqueSlice(_ job: CADExactJob, slice: CADSlicePlan.Slice, clear: Bool,
                                   commandBuffer: any MTLCommandBuffer) {
        let layers = job.targets
        guard let normal = layers.normal, let part = layers.part else { return }
        let pass = MTLRenderPassDescriptor()
        for (index, texture) in [normal, part].enumerated() {
            pass.colorAttachments[index].texture = texture
            pass.colorAttachments[index].loadAction = clear ? .clear : .load
            pass.colorAttachments[index].clearColor = MTLClearColor()
            pass.colorAttachments[index].storeAction = .store
        }
        pass.depthAttachment.texture = layers.depth
        pass.depthAttachment.loadAction = clear ? .clear : .load
        pass.depthAttachment.clearDepth = 0
        pass.depthAttachment.storeAction = .store
        guard let encoder = commandBuffer.makeRenderCommandEncoder(descriptor: pass) else { return }
        encoder.label = "RAD exact geometry"
        let resources = job.field.resources
        if slice.covered > 0,
           let pipeline = try? resources.pipeline(.geometry, colors: [CADFieldResources.normalFormat, CADFieldResources.partFormat],
                                                  depth: CADFieldResources.depthFormat, samples: CADFieldResources.sampleCount) {
            var uniforms = fieldUniforms(job)
            encoder.setRenderPipelineState(pipeline)
            encoder.setDepthStencilState(resources.depthWrite)
            encoder.setCullMode(.back)
            encoder.setFrontFacing(.counterClockwise)
            encoder.setScissorRect(scissor(slice: slice.rect))
            encoder.setVertexBytes(&uniforms, length: MemoryLayout<CADFieldResources.Uniforms>.stride, index: 1)
            encoder.setFragmentBytes(&uniforms, length: MemoryLayout<CADFieldResources.Uniforms>.stride, index: 1)
            resources.encode(job.draws, within: slice.rect, encoder: encoder)
        }
        encoder.endEncoding()
    }

    /// Translucent layers: visibility depth and transmittance per sample in
    /// the stored targets, accumulated color resolved within the slice and
    /// copied into the stored accumulation.
    private func encodeTranslucentSlice(_ job: CADExactJob, slice: CADSlicePlan.Slice, clear: Bool,
                                        commandBuffer: any MTLCommandBuffer) {
        let layers = job.targets, resources = job.field.resources
        guard let accumulated = layers.accumulated else { return }
        var uniforms = fieldUniforms(job)
        let samples = CADFieldResources.sampleCount
        let pass = MTLRenderPassDescriptor()
        pass.depthAttachment.texture = layers.depth
        pass.depthAttachment.loadAction = clear ? .clear : .load
        pass.depthAttachment.clearDepth = 0
        pass.depthAttachment.storeAction = .store
        if let revealage = layers.revealage {
            pass.colorAttachments[0].texture = revealage
            pass.colorAttachments[0].loadAction = clear ? .clear : .load
            pass.colorAttachments[0].clearColor = MTLClearColor(red: 1, green: 1, blue: 1, alpha: 1)
            pass.colorAttachments[0].storeAction = .store
        }
        guard let encoder = commandBuffer.makeRenderCommandEncoder(descriptor: pass) else { return }
        encoder.label = "RAD exact translucent layers"
        let colors: [MTLPixelFormat] = layers.revealage == nil ? [] : [.r16Float]
        if slice.covered > 0, let visibility = try? resources.pipeline(.visibility, colors: colors,
                                                                      depth: CADFieldResources.depthFormat, samples: samples) {
            encoder.setScissorRect(scissor(slice: slice.rect))
            encoder.setVertexBytes(&uniforms, length: MemoryLayout<CADFieldResources.Uniforms>.stride, index: 1)
            encoder.setFragmentBytes(&uniforms, length: MemoryLayout<CADFieldResources.Uniforms>.stride, index: 1)
            encoder.setFrontFacing(.counterClockwise)
            encoder.setRenderPipelineState(visibility)
            encoder.setDepthStencilState(resources.depthWrite)
            encoder.setCullMode(.back)
            resources.encode(job.draws, within: slice.rect, encoder: encoder)
            if layers.revealage != nil, let revealage = try? resources.pipeline(.revealage, colors: [.r16Float],
                                                                               depth: CADFieldResources.depthFormat,
                                                                               samples: samples) {
                encoder.setRenderPipelineState(revealage)
                encoder.setDepthStencilState(resources.depthIgnored)
                encoder.setCullMode(.none)
                resources.encode(job.draws, within: slice.rect, encoder: encoder)
            }
        }
        encoder.endEncoding()
        if clear {
            // Pixels no slice reaches accumulate nothing.
            let reset = MTLRenderPassDescriptor()
            reset.colorAttachments[0].texture = accumulated
            reset.colorAttachments[0].loadAction = .clear
            reset.colorAttachments[0].clearColor = MTLClearColor()
            reset.colorAttachments[0].storeAction = .store
            commandBuffer.makeRenderCommandEncoder(descriptor: reset)?.endEncoding()
        }
        let width = Int(slice.rect.z - slice.rect.x), height = Int(slice.rect.w - slice.rect.y)
        guard slice.covered > 0, width > 0, height > 0, let targets = targets(width: job.width, height: job.height),
              let multisample = targets.multisampleHalf4(),
              let resolved = targets.areaTexture("exact.accumulated", .rgba16Float, width: width, height: height),
              let pipeline = try? resources.pipeline(.accumulation, colors: [.rgba16Float], depth: .invalid, samples: samples)
        else { return }
        // The slice's rectangle maps onto the pass, pixel for pixel.
        let accumulation = restrictedPass(resolved, width: width, height: height, multisample: multisample)
        guard let layer = commandBuffer.makeRenderCommandEncoder(descriptor: accumulation) else { return }
        layer.label = "RAD exact accumulation"
        layer.setViewport(MTLViewport(originX: 0, originY: 0, width: Double(width), height: Double(height), znear: 0, zfar: 1))
        uniforms.remap = remap(x: Int(slice.rect.x), y: Int(slice.rect.y), width: width, height: height,
                               frameWidth: job.width, frameHeight: job.height)
        layer.setRenderPipelineState(pipeline)
        layer.setCullMode(.none)
        layer.setFrontFacing(.counterClockwise)
        layer.setVertexBytes(&uniforms, length: MemoryLayout<CADFieldResources.Uniforms>.stride, index: 1)
        layer.setFragmentBytes(&uniforms, length: MemoryLayout<CADFieldResources.Uniforms>.stride, index: 1)
        resources.encode(job.draws, within: slice.rect, encoder: layer)
        layer.endEncoding()
        guard let blit = commandBuffer.makeBlitCommandEncoder() else { return }
        blit.label = "RAD exact accumulation copy"
        blit.copy(from: resolved, sourceSlice: 0, sourceLevel: 0, sourceOrigin: MTLOrigin(x: 0, y: 0, z: 0),
                  sourceSize: MTLSize(width: width, height: height, depth: 1), to: accumulated, destinationSlice: 0,
                  destinationLevel: 0, destinationOrigin: MTLOrigin(x: Int(slice.rect.x), y: Int(slice.rect.y), z: 0))
        blit.endEncoding()
    }

    /// Creates the stored outline coverage of `frame`'s batches that have
    /// exact parts, and forgets coverage no longer highlighted.
    func prepareMasks(_ job: CADExactJob, frame: CADViewportFrame) {
        var used = Set<CADMaskCoverage.Key>()
        for batch in frame.batches {
            guard let key = maskKey(batch, frame: frame) else { continue }
            used.insert(key)
            guard job.masks[key] == nil else { continue }
            let masks = batch.masks.filter { $0.field != nil }
            let rects = masks.flatMap { mask in mask.field.map { job.rects(of: [$0.solid]) } ?? [] }
            job.masks[key] = CADMaskCoverage(key: key, masks: masks, device: device, rects: rects)
        }
        job.masks = job.masks.filter { used.contains($0.key) }
    }

    /// The stored coverage key of a batch with exact parts.
    private func maskKey(_ batch: CADViewportFrame.Batch, frame: CADViewportFrame) -> CADMaskCoverage.Key? {
        let items = batch.masks.compactMap { mask -> CADMaskCoverage.Key.Item? in
            guard let field = mask.field else { return nil }
            let m = mask.model
            return .init(solid: field.solid, model: [m.columns.0, m.columns.1, m.columns.2, m.columns.3].flatMap { [$0.x, $0.y, $0.z, $0.w] },
                         selected: mask.selected)
        }
        guard !items.isEmpty else { return nil }
        let pixels = pixelRect(batch.rect, width: frame.width, height: frame.height)
        guard pixels.width > 0, pixels.height > 0 else { return nil }
        return .init(rect: SIMD4(Int32(pixels.x), Int32(pixels.y), Int32(pixels.x + pixels.width), Int32(pixels.y + pixels.height)),
                     items: items)
    }

    /// Encodes the next slice of a job's unfinished outline coverage, and
    /// returns that coverage, or nil when none remains.
    @discardableResult
    func encodeMaskSlice(_ job: CADExactJob, budget: Double, commandBuffer: any MTLCommandBuffer) -> CADMaskCoverage? {
        let pending = job.masks.values.filter { !$0.cleared || !$0.plan.isDone }
            .sorted { ($0.key.rect.y, $0.key.rect.x) < ($1.key.rect.y, $1.key.rect.x) }
        guard let coverage = pending.first else { return nil }
        let resources = job.field.resources
        let covered = budget.isFinite ? max(1, Int(budget / CADExactCost.shared.sliceRate(job.costKey))) : Int.max
        guard let slice = coverage.plan.next(budget: covered) ?? (coverage.cleared ? nil : .init(rect: .zero, covered: 0)) else {
            return nil
        }
        let clear = !coverage.cleared
        coverage.cleared = true
        coverage.encoded += 1
        count { $0.maskSlices += 1 }
        let pass = MTLRenderPassDescriptor()
        pass.colorAttachments[0].texture = coverage.texture
        pass.colorAttachments[0].loadAction = clear ? .clear : .load
        pass.colorAttachments[0].clearColor = MTLClearColor()
        pass.colorAttachments[0].storeAction = .store
        guard let encoder = commandBuffer.makeRenderCommandEncoder(descriptor: pass) else { return coverage }
        encoder.label = "RAD exact outline coverage"
        let origin = SIMD4(coverage.key.rect.x, coverage.key.rect.y, coverage.key.rect.x, coverage.key.rect.y)
        if slice.covered > 0,
           let pipeline = try? resources.pipeline(.mask, colors: [.rgba8Unorm], depth: .invalid, samples: 1) {
            let width = Int(coverage.key.rect.z - coverage.key.rect.x), height = Int(coverage.key.rect.w - coverage.key.rect.y)
            var uniforms = fieldUniforms(job)
            uniforms.remap = remap(x: Int(coverage.key.rect.x), y: Int(coverage.key.rect.y), width: width, height: height,
                                   frameWidth: job.width, frameHeight: job.height)
            encoder.setRenderPipelineState(pipeline)
            encoder.setCullMode(.back)
            encoder.setFrontFacing(.counterClockwise)
            encoder.setScissorRect(scissor(slice: slice.rect))
            encoder.setVertexBytes(&uniforms, length: MemoryLayout<CADFieldResources.Uniforms>.stride, index: 1)
            encoder.setFragmentBytes(&uniforms, length: MemoryLayout<CADFieldResources.Uniforms>.stride, index: 1)
            for mask in coverage.masks {
                guard let field = mask.field else { continue }
                var channel: SIMD4<Float> = mask.selected ? SIMD4(1, 0, 0, 0) : SIMD4(0, 1, 0, 0)
                encoder.setFragmentBytes(&channel, length: MemoryLayout<SIMD4<Float>>.stride, index: 4)
                let draws = resources.draws([CADFieldInstance(solid: field.solid, model: mask.model, albedo: .zero, emission: .zero)],
                                            view: job.view, projection: job.projection, width: job.width, height: job.height,
                                            contacts: false)
                resources.encode(draws, within: slice.rect &+ origin, encoder: encoder)
            }
        }
        encoder.endEncoding()
        return coverage
    }

    /// Whether every outline coverage of the job's frame is stored.
    func masksComplete(_ job: CADExactJob) -> Bool { job.masks.values.allSatisfy(\.isComplete) }

    /// Encodes a frame that composites a job's stored layers (which must be
    /// complete when the GPU runs it). Returns false when a scene pass drew
    /// another view than the layers', so the frame must not be shown.
    func encodeComposite(_ frame: CADViewportFrame, job: CADExactJob, source: CADViewportSceneSource, renderer: SCNRenderer,
                         into target: any MTLTexture, commandBuffer: any MTLCommandBuffer) -> Bool {
        let instances = frame.field?.instances ?? job.field.instances
        let composite = CADExactComposite(
            layers: job.targets, view: job.view, projection: job.projection,
            materials: job.key.translucent ? nil : CADExactComposite.materials(instances, device: device),
            lights: job.field.lights, debug: job.field.debug, pipelines: exactPipelines)
        count { $0.composites += 1 }
        return encodeScene(frame, composite: composite, job: job, source: source, renderer: renderer, into: target,
                           commandBuffer: commandBuffer)
    }

    // MARK: Scene passes

    private func prepare(_ renderer: SCNRenderer, source: CADViewportSceneSource) {
        if renderer.scene !== source.scene { renderer.scene = source.scene }
        renderer.usesReverseZ = true
        renderer.autoenablesDefaultLighting = false
    }

    /// The scene passes (exact cameras with a composite, mesh cameras without),
    /// the highlight passes and the final composite. Returns whether every
    /// scene pass drew the composite's own view.
    private func encodeScene(_ frame: CADViewportFrame, composite: CADExactComposite?, job: CADExactJob?,
                             source: CADViewportSceneSource, renderer: SCNRenderer, into target: any MTLTexture,
                             commandBuffer: any MTLCommandBuffer) -> Bool {
        guard frame.width > 0, frame.height > 0, target.width == frame.width, target.height == frame.height,
              let targets = targets(width: frame.width, height: frame.height) else { return true }
        prepare(renderer, source: source)
        let viewport = CGRect(x: 0, y: 0, width: frame.width, height: frame.height)
        source.probe.reset()
        let cameras = composite == nil ? source.mesh : source.exact
        // Without a display-mesh layer, transmittance derives from accumulated alpha.
        let exactComplete = composite != nil && frame.field?.complete == true
        source.field.begin(composite)
        var translucentTargets: (decorations: any MTLTexture, accumulated: any MTLTexture, revealage: any MTLTexture)?
        if frame.translucent {
            guard let decorations = targets.decorations(), let accumulated = targets.accumulated(),
                  let revealage = targets.revealage(), let half4 = targets.multisampleHalf4(),
                  let half1 = targets.multisampleHalf1() else { source.field.end(); return true }
            renderScene(renderer, camera: cameras.decorations, viewport: viewport, commandBuffer: commandBuffer,
                        color: targets.multisampleColor, resolve: decorations, clear: MTLClearColor(),
                        depth: targets.multisampleDepth)
            renderScene(renderer, camera: cameras.accumulation, viewport: viewport, commandBuffer: commandBuffer,
                        color: half4, resolve: accumulated, clear: MTLClearColor(), depth: targets.multisampleDepth)
            if !exactComplete {
                renderScene(renderer, camera: cameras.revealage, viewport: viewport, commandBuffer: commandBuffer,
                            color: half1, resolve: revealage, clear: MTLClearColor(red: 1, green: 1, blue: 1, alpha: 1),
                            depth: targets.multisampleDepth)
            }
            translucentTargets = (decorations, accumulated, revealage)
        } else {
            let background = frame.palette.background
            renderScene(renderer, camera: cameras.opaque, viewport: viewport, commandBuffer: commandBuffer,
                        color: targets.multisampleColor, resolve: target,
                        clear: MTLClearColor(red: Double(background.x), green: Double(background.y),
                                             blue: Double(background.z), alpha: Double(background.w)),
                        depth: targets.multisampleDepth)
        }
        let matched = source.field.end()
        let matrices = source.probe.takeMatrices()
        var uniforms = HighlightUniforms(frame: frame)
        if frame.translucent, exactComplete { uniforms.revealage.x = 1 }
        var horizontal: (any MTLTexture)?
        if frame.hasHighlights, let matrices, let area = highlightArea(frame) {
            uniforms.highlightArea = area
            uniforms.hasHighlights = 1
            horizontal = encodeHighlights(frame, matrices: matrices, area: area, uniforms: uniforms,
                                          targets: targets, job: composite == nil ? nil : job, commandBuffer: commandBuffer)
            if horizontal == nil { uniforms.hasHighlights = 0 }
        }
        if let translucentTargets {
            let exactAccumulated = composite?.layers.accumulated
            if exactAccumulated != nil { uniforms.revealage.z = 1 }
            let pass = MTLRenderPassDescriptor()
            pass.colorAttachments[0].texture = target
            pass.colorAttachments[0].loadAction = .dontCare
            pass.colorAttachments[0].storeAction = .store
            guard let encoder = commandBuffer.makeRenderCommandEncoder(descriptor: pass) else { return matched }
            encoder.label = "RAD translucent composite"
            encoder.setRenderPipelineState(pipelines.translucentComposite)
            var rect = SIMD4<Float>(-1, -1, 1, 1)
            encoder.setVertexBytes(&rect, length: MemoryLayout<SIMD4<Float>>.stride, index: 0)
            encoder.setFragmentTexture(translucentTargets.accumulated, index: 0)
            encoder.setFragmentTexture(translucentTargets.revealage, index: 1)
            encoder.setFragmentTexture(translucentTargets.decorations, index: 2)
            encoder.setFragmentTexture(horizontal ?? pipelines.empty, index: 3)
            encoder.setFragmentTexture(exactAccumulated ?? pipelines.empty, index: 4)
            encoder.setFragmentBytes(&uniforms, length: MemoryLayout<HighlightUniforms>.stride, index: 0)
            encoder.drawPrimitives(type: .triangleStrip, vertexStart: 0, vertexCount: 4)
            encoder.endEncoding()
        } else if let horizontal {
            let pass = MTLRenderPassDescriptor()
            pass.colorAttachments[0].texture = target
            pass.colorAttachments[0].loadAction = .load
            pass.colorAttachments[0].storeAction = .store
            guard let encoder = commandBuffer.makeRenderCommandEncoder(descriptor: pass) else { return matched }
            encoder.label = "RAD highlight overlay"
            encoder.setRenderPipelineState(pipelines.overlay)
            encoder.setScissorRect(scissor(uniforms.highlightArea))
            var rect = SIMD4<Float>(-1, -1, 1, 1)
            encoder.setVertexBytes(&rect, length: MemoryLayout<SIMD4<Float>>.stride, index: 0)
            encoder.setFragmentTexture(horizontal, index: 0)
            encoder.setFragmentBytes(&uniforms, length: MemoryLayout<HighlightUniforms>.stride, index: 0)
            encoder.drawPrimitives(type: .triangleStrip, vertexStart: 0, vertexCount: 4)
            encoder.endEncoding()
        }
        return matched
    }

    private func renderScene(_ renderer: SCNRenderer, camera: SCNNode, viewport: CGRect, commandBuffer: any MTLCommandBuffer,
                             color: any MTLTexture, resolve: (any MTLTexture)?, clear: MTLClearColor, depth: any MTLTexture) {
        let pass = MTLRenderPassDescriptor()
        pass.colorAttachments[0].texture = color
        pass.colorAttachments[0].resolveTexture = resolve
        pass.colorAttachments[0].loadAction = .clear
        pass.colorAttachments[0].clearColor = clear
        pass.colorAttachments[0].storeAction = resolve == nil ? .dontCare : .multisampleResolve
        pass.depthAttachment.texture = depth
        pass.depthAttachment.loadAction = .clear
        pass.depthAttachment.clearDepth = renderer.usesReverseZ ? 0 : 1
        pass.depthAttachment.storeAction = .dontCare
        renderer.pointOfView = camera
        renderer.render(withViewport: viewport, commandBuffer: commandBuffer, passDescriptor: pass)
    }

    // MARK: Highlight passes

    /// Pixel rectangle (x0, y0, x1, y1 inclusive, top-left origin) that can
    /// receive highlight coverage: every seed and mask plus the kernel reach.
    private func highlightArea(_ frame: CADViewportFrame) -> SIMD4<Int32>? {
        guard let bounds = frame.highlightBounds else { return nil }
        let pixels = pixelRect(bounds, width: frame.width, height: frame.height)
        let radius = frame.kernelRadius
        let x0 = max(0, pixels.x - radius), y0 = max(0, pixels.y - radius)
        let x1 = min(frame.width - 1, pixels.x + pixels.width - 1 + radius)
        let y1 = min(frame.height - 1, pixels.y + pixels.height - 1 + radius)
        guard x0 <= x1, y0 <= y1 else { return nil }
        return SIMD4(Int32(x0), Int32(y0), Int32(x1), Int32(y1))
    }

    /// Top-left-origin pixel rectangle covering an NDC rectangle.
    private func pixelRect(_ rect: CADSilhouetteBatching.Rect, width: Int, height: Int) -> (x: Int, y: Int, width: Int, height: Int) {
        let x0 = Int(floor((rect.low.x + 1) / 2 * Double(width)))
        let x1 = Int(ceil((rect.high.x + 1) / 2 * Double(width)))
        let y0 = Int(floor((1 - rect.high.y) / 2 * Double(height)))
        let y1 = Int(ceil((1 - rect.low.y) / 2 * Double(height)))
        let left = min(max(0, x0), width), top = min(max(0, y0), height)
        let right = min(max(left, x1), width), bottom = min(max(top, y1), height)
        return (left, top, right - left, bottom - top)
    }

    /// The NDC rectangle of a top-left-origin pixel rectangle.
    private func remap(x: Int, y: Int, width: Int, height: Int, frame: CADViewportFrame) -> SIMD4<Float> {
        remap(x: x, y: y, width: width, height: height, frameWidth: frame.width, frameHeight: frame.height)
    }

    private func remap(x: Int, y: Int, width: Int, height: Int, frameWidth: Int, frameHeight: Int) -> SIMD4<Float> {
        let fw = Float(frameWidth), fh = Float(frameHeight)
        return SIMD4(Float(x) / fw * 2 - 1, 1 - Float(y + height) / fh * 2,
                     Float(x + width) / fw * 2 - 1, 1 - Float(y) / fh * 2)
    }

    private func scissor(_ area: SIMD4<Int32>) -> MTLScissorRect {
        MTLScissorRect(x: Int(area.x), y: Int(area.y), width: Int(area.z - area.x + 1), height: Int(area.w - area.y + 1))
    }

    /// A pass restricted to the top-left `width`×`height` pixels of `target`.
    private func restrictedPass(_ target: any MTLTexture, width: Int, height: Int,
                                multisample: (any MTLTexture)? = nil) -> MTLRenderPassDescriptor {
        let pass = MTLRenderPassDescriptor()
        if let multisample {
            pass.colorAttachments[0].texture = multisample
            pass.colorAttachments[0].resolveTexture = target
            pass.colorAttachments[0].loadAction = .clear
            pass.colorAttachments[0].clearColor = MTLClearColor(red: 0, green: 0, blue: 0, alpha: 0)
            pass.colorAttachments[0].storeAction = .multisampleResolve
        } else {
            pass.colorAttachments[0].texture = target
            pass.colorAttachments[0].loadAction = .load
            pass.colorAttachments[0].storeAction = .store
        }
        pass.renderTargetWidth = width
        pass.renderTargetHeight = height
        return pass
    }

    /// Seeds and contours, then the horizontal distance pass. Every texture
    /// here covers only the highlight area, whose origin maps to texel zero.
    /// Returns the horizontal distances read by the final compositor.
    private func encodeHighlights(_ frame: CADViewportFrame, matrices: (view: simd_float4x4, projection: simd_float4x4),
                                  area: SIMD4<Int32>, uniforms: HighlightUniforms, targets: Targets, job: CADExactJob?,
                                  commandBuffer: any MTLCommandBuffer) -> (any MTLTexture)? {
        let viewProjection = matrices.projection * matrices.view
        let areaX = Int(area.x), areaY = Int(area.y)
        let areaWidth = Int(area.z - area.x) + 1, areaHeight = Int(area.w - area.y) + 1
        guard let multisampleMask = targets.multisampleMask(),
              let seedMask = targets.areaTexture("seeds", .rgba8Unorm, width: areaWidth, height: areaHeight),
              let horizontal = targets.areaTexture("horizontal", .rgba16Float, width: areaWidth, height: areaHeight)
        else { return nil }
        let areaViewport = MTLViewport(originX: 0, originY: 0, width: Double(areaWidth), height: Double(areaHeight),
                                       znear: 0, zfar: 1)
        let areaRemap = remap(x: areaX, y: areaY, width: areaWidth, height: areaHeight, frame: frame)
        guard let seedEncoder = commandBuffer.makeRenderCommandEncoder(descriptor:
                restrictedPass(seedMask, width: areaWidth, height: areaHeight, multisample: multisampleMask)) else { return nil }
        seedEncoder.label = "RAD highlight seeds"
        seedEncoder.setViewport(areaViewport)
        for item in frame.seeds {
            let points = item.buffer.primitive == .point
            seedEncoder.setRenderPipelineState(points ? pipelines.seedPoints : pipelines.seedLines)
            var mask = MaskUniforms(modelViewProjection: viewProjection * item.model, remap: areaRemap,
                                    channel: Self.channel(item.channel))
            seedEncoder.setVertexBuffer(item.buffer.positions, offset: 0, index: 0)
            seedEncoder.setVertexBytes(&mask, length: MemoryLayout<MaskUniforms>.stride, index: 1)
            seedEncoder.setFragmentBytes(&mask, length: MemoryLayout<MaskUniforms>.stride, index: 1)
            seedEncoder.drawPrimitives(type: item.buffer.primitive, vertexStart: 0, vertexCount: item.buffer.vertexCount)
        }
        seedEncoder.endEncoding()

        for batch in frame.batches {
            let pixels = pixelRect(batch.rect, width: frame.width, height: frame.height)
            guard pixels.width > 0, pixels.height > 0,
                  let scratch = targets.areaTexture("mask", .rgba8Unorm, width: pixels.width, height: pixels.height),
                  let encoder = commandBuffer.makeRenderCommandEncoder(descriptor:
                    restrictedPass(scratch, width: pixels.width, height: pixels.height, multisample: multisampleMask))
            else { continue }
            encoder.label = "RAD outline coverage"
            encoder.setViewport(MTLViewport(originX: 0, originY: 0, width: Double(pixels.width),
                                            height: Double(pixels.height), znear: 0, zfar: 1))
            // Map exactly this batch's pixels onto the restricted target.
            let batchRemap = remap(x: pixels.x, y: pixels.y, width: pixels.width, height: pixels.height, frame: frame)
            // Exact parts' coverage, marched once per view into stored
            // coverage; until it is complete their display meshes stand in.
            let stored = job.flatMap { job in maskKey(batch, frame: frame).flatMap { job.masks[$0] } }
            let exact = stored?.isComplete == true
            if let stored, exact {
                encoder.setRenderPipelineState(pipelines.coverage)
                var rect = SIMD4<Float>(-1, -1, 1, 1)
                encoder.setVertexBytes(&rect, length: MemoryLayout<SIMD4<Float>>.stride, index: 0)
                encoder.setFragmentTexture(stored.texture, index: 0)
                encoder.drawPrimitives(type: .triangleStrip, vertexStart: 0, vertexCount: 4)
            }
            encoder.setRenderPipelineState(pipelines.mask)
            for item in batch.masks {
                if exact, item.field != nil { continue }
                let channel: SIMD4<Float> = item.selected ? SIMD4(1, 0, 0, 0) : SIMD4(0, 1, 0, 0)
                var mask = MaskUniforms(modelViewProjection: viewProjection * item.model, remap: batchRemap,
                                        channel: channel)
                encoder.setVertexBuffer(item.buffer.positions, offset: 0, index: 0)
                encoder.setVertexBytes(&mask, length: MemoryLayout<MaskUniforms>.stride, index: 1)
                encoder.setFragmentBytes(&mask, length: MemoryLayout<MaskUniforms>.stride, index: 1)
                if let triangles = item.buffer.triangles, item.buffer.triangleIndexCount > 0 {
                    encoder.drawIndexedPrimitives(type: .triangle, indexCount: item.buffer.triangleIndexCount,
                                                  indexType: .uint32, indexBuffer: triangles, indexBufferOffset: 0)
                }
            }
            encoder.endEncoding()

            // The batch lies inside the highlight area, whose seeds it joins.
            guard let contourEncoder = commandBuffer.makeRenderCommandEncoder(descriptor:
                    restrictedPass(seedMask, width: areaWidth, height: areaHeight)) else { continue }
            contourEncoder.label = "RAD outline contour"
            contourEncoder.setViewport(areaViewport)
            contourEncoder.setRenderPipelineState(pipelines.contour)
            let x = max(0, pixels.x - areaX), y = max(0, pixels.y - areaY)
            contourEncoder.setScissorRect(MTLScissorRect(x: x, y: y, width: max(0, min(pixels.width, areaWidth - x)),
                                                         height: max(0, min(pixels.height, areaHeight - y))))
            var rect = SIMD4<Float>(-1, -1, 1, 1)
            contourEncoder.setVertexBytes(&rect, length: MemoryLayout<SIMD4<Float>>.stride, index: 0)
            var region = SIMD4<Int32>(Int32(pixels.x - areaX), Int32(pixels.y - areaY), Int32(pixels.width), Int32(pixels.height))
            contourEncoder.setFragmentTexture(scratch, index: 0)
            contourEncoder.setFragmentBytes(&region, length: MemoryLayout<SIMD4<Int32>>.stride, index: 0)
            contourEncoder.drawPrimitives(type: .triangleStrip, vertexStart: 0, vertexCount: 4)
            contourEncoder.endEncoding()
        }

        let distances = MTLRenderPassDescriptor()
        distances.colorAttachments[0].texture = horizontal
        distances.colorAttachments[0].loadAction = .dontCare
        distances.colorAttachments[0].storeAction = .store
        distances.renderTargetWidth = areaWidth
        distances.renderTargetHeight = areaHeight
        guard let encoder = commandBuffer.makeRenderCommandEncoder(descriptor: distances) else { return nil }
        encoder.label = "RAD highlight distances"
        encoder.setViewport(areaViewport)
        encoder.setRenderPipelineState(pipelines.horizontal)
        var rect = SIMD4<Float>(-1, -1, 1, 1)
        var uniforms = uniforms
        encoder.setVertexBytes(&rect, length: MemoryLayout<SIMD4<Float>>.stride, index: 0)
        encoder.setFragmentTexture(seedMask, index: 0)
        encoder.setFragmentBytes(&uniforms, length: MemoryLayout<HighlightUniforms>.stride, index: 0)
        encoder.drawPrimitives(type: .triangleStrip, vertexStart: 0, vertexCount: 4)
        encoder.endEncoding()
        return horizontal
    }

    private static func channel(_ channel: CADViewportFrame.SeedChannel) -> SIMD4<Float> {
        switch channel {
        case .selectedEdge: SIMD4(1, 0, 0, 0)
        case .hoveredEdge: SIMD4(0, 1, 0, 0)
        case .selectedVertex: SIMD4(0, 0, 1, 0)
        case .hoveredVertex: SIMD4(0, 0, 0, 1)
        }
    }

    // MARK: Targets

    private func targets(width: Int, height: Int) -> Targets? {
        if let targets, targets.width == width, targets.height == height { return targets }
        targets = Targets(device: device, width: width, height: height)
        return targets
    }

    /// Per-size render targets. Multisample attachments are memoryless; the
    /// stored single-sample targets are created only once a frame needs them.
    private final class Targets {
        let width: Int
        let height: Int
        let multisampleColor: any MTLTexture
        let multisampleDepth: any MTLTexture
        private let device: any MTLDevice
        private var cache: [String: any MTLTexture] = [:]

        init?(device: any MTLDevice, width: Int, height: Int) {
            guard let color = Self.make(device, CADViewportPasses.colorFormat, width, height, multisample: true),
                  let depth = Self.make(device, .depth32Float, width, height, multisample: true) else { return nil }
            self.device = device
            self.width = width
            self.height = height
            multisampleColor = color
            multisampleDepth = depth
        }

        func multisampleHalf4() -> (any MTLTexture)? { texture("ms.rgba16f", .rgba16Float, multisample: true) }
        func multisampleHalf1() -> (any MTLTexture)? { texture("ms.r16f", .r16Float, multisample: true) }
        func multisampleMask() -> (any MTLTexture)? { texture("ms.mask", .rgba8Unorm, multisample: true) }
        func decorations() -> (any MTLTexture)? { texture("decorations", CADViewportPasses.colorFormat) }
        func accumulated() -> (any MTLTexture)? { texture("accumulated", .rgba16Float) }
        func revealage() -> (any MTLTexture)? { texture("revealage", .r16Float) }

        /// A texture at least `width`×`height`, for passes restricted to that
        /// region. Sizes round up so ordinary hover movement reuses it, and a
        /// much larger texture is released once highlights become small again.
        func areaTexture(_ key: String, _ format: MTLPixelFormat, width: Int, height: Int) -> (any MTLTexture)? {
            let granule = 128
            func rounded(_ value: Int, limit: Int) -> Int { min(limit, (value + granule - 1) / granule * granule) }
            if let texture = cache[key], texture.width >= width, texture.height >= height,
               texture.width * texture.height <= max(4 * width * height, granule * granule * 4) {
                return texture
            }
            cache[key] = nil
            guard let texture = Self.make(device, format, rounded(width, limit: self.width),
                                          rounded(height, limit: self.height), multisample: false) else { return nil }
            cache[key] = texture
            return texture
        }

        private func texture(_ key: String, _ format: MTLPixelFormat, multisample: Bool = false) -> (any MTLTexture)? {
            if let texture = cache[key] { return texture }
            let texture = Self.make(device, format, width, height, multisample: multisample)
            cache[key] = texture
            return texture
        }

        static func make(_ device: any MTLDevice, _ format: MTLPixelFormat, _ width: Int, _ height: Int,
                         multisample: Bool) -> (any MTLTexture)? {
            let descriptor = MTLTextureDescriptor.texture2DDescriptor(pixelFormat: format, width: width, height: height,
                                                                      mipmapped: false)
            if multisample {
                descriptor.textureType = .type2DMultisample
                descriptor.sampleCount = CADViewportPasses.sampleCount
                descriptor.usage = .renderTarget
                descriptor.storageMode = .memoryless
            } else {
                descriptor.usage = [.renderTarget, .shaderRead]
                descriptor.storageMode = .private
            }
            return device.makeTexture(descriptor: descriptor)
        }
    }

    // MARK: Pipelines

    private struct MaskUniforms {
        var modelViewProjection: simd_float4x4
        var remap: SIMD4<Float>
        var channel: SIMD4<Float>
    }

    /// Mirrors `CADHighlightUniforms` in the shader source.
    private struct HighlightUniforms {
        var selected: SIMD4<Float>
        var hover: SIMD4<Float>
        var hoverHalo: SIMD4<Float>
        var ink: SIMD4<Float>
        var background: SIMD4<Float>
        var referenceHeight: Float
        var verticesEnabled: Float
        var hasHighlights: Float = 0
        /// Output pixels; screen-point widths scale by this over the reference.
        var frameHeight: Float
        /// Frame pixels (x0, y0, x1, y1 inclusive) covered by the area targets.
        var highlightArea = SIMD4<Int32>(0, 0, -1, -1)
        /// x: transmittance derives from the accumulated alpha (y: each layer's
        /// alpha); z: exact layers add their stored accumulation.
        var revealage = SIMD4<Float>(0, CADTranslucency.alpha, 0, 0)

        init(frame: CADViewportFrame) {
            selected = frame.palette.selected
            hover = frame.palette.hover
            hoverHalo = frame.palette.hoverHalo
            ink = frame.palette.ink
            background = frame.palette.background
            referenceHeight = Float(max(1, frame.referenceHeight))
            verticesEnabled = frame.verticesEnabled ? 1 : 0
            frameHeight = Float(frame.height)
        }
    }

    private final class Pipelines: @unchecked Sendable {
        let seedLines: any MTLRenderPipelineState
        let seedPoints: any MTLRenderPipelineState
        let mask: any MTLRenderPipelineState
        let contour: any MTLRenderPipelineState
        let horizontal: any MTLRenderPipelineState
        let overlay: any MTLRenderPipelineState
        let translucentComposite: any MTLRenderPipelineState
        let coverage: any MTLRenderPipelineState
        let empty: any MTLTexture

        private static let lock = NSLock()
        private static var cache: [UInt64: Pipelines] = [:]

        static func shared(for device: any MTLDevice) throws -> Pipelines {
            try lock.withLock {
                if let cached = cache[device.registryID] { return cached }
                let created = try Pipelines(device: device)
                cache[device.registryID] = created
                return created
            }
        }

        private init(device: any MTLDevice) throws {
            let library = try device.makeLibrary(source: CADViewportPasses.shaderSource, options: nil)
            func function(_ name: String) throws -> any MTLFunction {
                guard let function = library.makeFunction(name: name) else {
                    throw RADError.message("RAD's viewport shaders are incomplete.")
                }
                return function
            }
            func pipeline(_ vertex: String, _ fragment: String, format: MTLPixelFormat, samples: Int = 1,
                          blend: (MTLRenderPipelineColorAttachmentDescriptor) -> Void = { _ in }) throws
                -> any MTLRenderPipelineState {
                let descriptor = MTLRenderPipelineDescriptor()
                descriptor.vertexFunction = try function(vertex)
                descriptor.fragmentFunction = try function(fragment)
                descriptor.rasterSampleCount = samples
                let attachment: MTLRenderPipelineColorAttachmentDescriptor = descriptor.colorAttachments[0]
                attachment.pixelFormat = format
                blend(attachment)
                return try device.makeRenderPipelineState(descriptor: descriptor)
            }
            // Coverage channels combine without depending on draw order.
            let maximum: (MTLRenderPipelineColorAttachmentDescriptor) -> Void = { attachment in
                attachment.isBlendingEnabled = true
                attachment.rgbBlendOperation = .max
                attachment.alphaBlendOperation = .max
                attachment.sourceRGBBlendFactor = .one
                attachment.destinationRGBBlendFactor = .one
                attachment.sourceAlphaBlendFactor = .one
                attachment.destinationAlphaBlendFactor = .one
            }
            let samples = CADViewportPasses.sampleCount
            seedLines = try pipeline("cad_mask_vertex", "cad_mask_fragment", format: .rgba8Unorm, samples: samples, blend: maximum)
            seedPoints = try pipeline("cad_point_vertex", "cad_mask_fragment", format: .rgba8Unorm, samples: samples, blend: maximum)
            mask = try pipeline("cad_mask_vertex", "cad_mask_fragment", format: .rgba8Unorm, samples: samples, blend: maximum)
            coverage = try pipeline("cad_quad", "cad_stored_coverage", format: .rgba8Unorm, samples: samples, blend: maximum)
            contour = try pipeline("cad_quad", "cad_contour", format: .rgba8Unorm, blend: maximum)
            horizontal = try pipeline("cad_quad", "cad_highlight_horizontal", format: .rgba16Float)
            // Premultiplied highlight layers over the resolved scene.
            overlay = try pipeline("cad_quad", "cad_highlight_overlay", format: CADViewportPasses.colorFormat) { attachment in
                attachment.isBlendingEnabled = true
                attachment.rgbBlendOperation = .add
                attachment.alphaBlendOperation = .add
                attachment.sourceRGBBlendFactor = .one
                attachment.destinationRGBBlendFactor = .oneMinusSourceAlpha
                attachment.sourceAlphaBlendFactor = .one
                attachment.destinationAlphaBlendFactor = .oneMinusSourceAlpha
            }
            translucentComposite = try pipeline("cad_quad", "cad_translucent_composite", format: CADViewportPasses.colorFormat)
            let descriptor = MTLTextureDescriptor.texture2DDescriptor(pixelFormat: .rgba16Float, width: 1, height: 1, mipmapped: false)
            descriptor.usage = .shaderRead
            descriptor.storageMode = .private
            guard let empty = device.makeTexture(descriptor: descriptor) else {
                throw RADError.message("RAD could not allocate viewport resources.")
            }
            self.empty = empty
        }
    }

    private static let shaderSource = """
    #include <metal_stdlib>
    using namespace metal;

    struct CADQuadOut { float4 position [[position]]; };
    struct CADPointOut { float4 position [[position]]; float size [[point_size]]; };
    struct CADMaskUniforms { float4x4 modelViewProjection; float4 remap; float4 channel; };
    struct CADHighlightUniforms {
        float4 selectedColor;
        float4 hoverColor;
        float4 hoverHaloColor;
        float4 inkColor;
        float4 backgroundColor;
        float referenceHeight;
        float verticesEnabled;
        float hasHighlights;
        float frameHeight;
        int4 highlightArea;
        float4 revealage;
    };

    constant float cadEdgeHoverRadius = \(Float(CADHighlightStyle.hoverEdgeWidth / 2))f;
    constant float cadEdgeSelectedRadius = \(Float(CADHighlightStyle.selectedEdgeWidth / 2))f;
    constant float cadEdgeHoverHaloRadius = \(Float(CADHighlightStyle.hoverEdgeHaloWidth / 2))f;
    constant float cadEdgeSelectedHaloRadius = \(Float(CADHighlightStyle.selectedEdgeHaloWidth / 2))f;
    constant float cadVertexHoverRadius = \(Float(CADHighlightStyle.hoverVertexDiameter / 2))f;
    constant float cadVertexSelectedRadius = \(Float(CADHighlightStyle.selectedVertexDiameter / 2))f;
    constant float cadVertexHoverHaloRadius = \(Float(CADHighlightStyle.hoverVertexHaloDiameter / 2))f;
    constant float cadVertexSelectedHaloRadius = \(Float(CADHighlightStyle.selectedVertexHaloDiameter / 2))f;
    constant float cadMaximumEdgeRadius = \(Float(CADHighlightStyle.maximumEdgeRadius))f;
    constant float cadMaximumRadius = \(Float(CADHighlightStyle.maximumRadius))f;
    // Bound the kernel while retaining complete circles at the largest
    // supported independent-capture scale (8×), including antialias coverage.
    constant int cadMaximumKernelRadius = \(Int(ceil(Double(CADHighlightStyle.maximumRadius) * 8 + 0.5)));

    vertex CADQuadOut cad_quad(uint vid [[vertex_id]], constant float4 &rect [[buffer(0)]]) {
        float2 corner = float2(float(vid & 1u), float(vid >> 1u));
        CADQuadOut out;
        out.position = float4(mix(rect.xy, rect.zw, corner), 0.0f, 1.0f);
        return out;
    }

    // Remaps a frame's NDC sub-rectangle onto the pass viewport. Depth and
    // w are untouched, so SceneKit's near and far clipping still applies.
    static float4 cad_mask_clip(float3 position, constant CADMaskUniforms &u) {
        float4 clip = u.modelViewProjection * float4(position, 1.0f);
        float2 center = (u.remap.xy + u.remap.zw) * 0.5f;
        float2 extent = (u.remap.zw - u.remap.xy) * 0.5f;
        clip.xy = (clip.xy - center * clip.w) / extent;
        return clip;
    }

    vertex CADQuadOut cad_mask_vertex(const device packed_float3 *positions [[buffer(0)]],
                                      constant CADMaskUniforms &u [[buffer(1)]], uint vid [[vertex_id]]) {
        CADQuadOut out;
        out.position = cad_mask_clip(float3(positions[vid]), u);
        return out;
    }

    vertex CADPointOut cad_point_vertex(const device packed_float3 *positions [[buffer(0)]],
                                        constant CADMaskUniforms &u [[buffer(1)]], uint vid [[vertex_id]]) {
        CADPointOut out;
        out.position = cad_mask_clip(float3(positions[vid]), u);
        out.size = 1.0f;
        return out;
    }

    fragment float4 cad_mask_fragment(constant CADMaskUniforms &u [[buffer(1)]]) { return u.channel; }

    // Stored exact coverage of a batch, texel for pixel.
    fragment float4 cad_stored_coverage(CADQuadOut in [[stage_in]], texture2d<float, access::read> coverage [[texture(0)]]) {
        return coverage.read(uint2(in.position.xy));
    }

    // Coverage minus its four-neighbor interior is the projected outline.
    // Reads clamp to this batch's region, which is padded inside the frame
    // and coincides with the frame edge where a large surface is clipped.
    fragment float4 cad_contour(CADQuadOut in [[stage_in]], texture2d<float, access::read> mask [[texture(0)]],
                                constant int4 &region [[buffer(0)]]) {
        int2 last = region.zw - 1;
        int2 p = clamp(int2(in.position.xy) - region.xy, int2(0), last);
        float2 covered = step(0.5f, mask.read(uint2(p)).rg);
        float2 interior = covered;
        interior = min(interior, step(0.5f, mask.read(uint2(max(p - int2(1, 0), int2(0)))).rg));
        interior = min(interior, step(0.5f, mask.read(uint2(min(p + int2(1, 0), last))).rg));
        interior = min(interior, step(0.5f, mask.read(uint2(max(p - int2(0, 1), int2(0)))).rg));
        interior = min(interior, step(0.5f, mask.read(uint2(min(p + int2(0, 1), last))).rg));
        return float4(covered - interior, 0.0f, 0.0f);
    }

    static float cad_scale(constant CADHighlightUniforms &u) {
        return clamp(u.frameHeight / max(u.referenceHeight, 1.0f), 0.125f, 8.0f);
    }

    static int cad_radius(float scale, constant CADHighlightUniforms &u) {
        float maximumRadius = u.verticesEnabled > 0.5f ? cadMaximumRadius : cadMaximumEdgeRadius;
        return min(cadMaximumKernelRadius, int(ceil(maximumRadius * scale + 0.5f)));
    }

    // Highlight-area textures map the area's first pixel to texel zero; no
    // seed lies outside the area, so reads stop at its edges.
    fragment float4 cad_highlight_horizontal(CADQuadOut in [[stage_in]], texture2d<float, access::read> seeds [[texture(0)]],
                                             constant CADHighlightUniforms &u [[buffer(0)]]) {
        int2 center = int2(in.position.xy);
        int lastColumn = u.highlightArea.z - u.highlightArea.x;
        float scale = cad_scale(u);
        bool verticesEnabled = u.verticesEnabled > 0.5f;
        int radius = cad_radius(scale, u);
        float4 best = float4(65504.0f);
        int first = max(-radius, -center.x);
        int last = min(radius, lastColumn - center.x);
        for (int dx = first; dx <= last; ++dx) {
            // RG are selected/hover edges; BA are selected/hover vertices.
            float4 seed = seeds.read(uint2(center + int2(dx, 0)));
            float distance = float(dx) * float(dx);
            if (seed.r > 0.01f) { best.r = min(best.r, distance); }
            if (seed.g > 0.01f) { best.g = min(best.g, distance); }
            if (verticesEnabled) {
                if (seed.b > 0.01f) { best.b = min(best.b, distance); }
                if (seed.a > 0.01f) { best.a = min(best.a, distance); }
            }
        }
        return best;
    }

    // `local` is a pixel's position within the highlight area.
    static float4 cad_distances(int2 local, texture2d<float, access::read> horizontal, float scale,
                                constant CADHighlightUniforms &u) {
        int radius = cad_radius(scale, u);
        int lastRow = u.highlightArea.w - u.highlightArea.y;
        int first = max(-radius, -local.y);
        int last = min(radius, lastRow - local.y);
        float4 best = float4(65504.0f);
        for (int dy = first; dy <= last; ++dy) {
            best = min(best, horizontal.read(uint2(local + int2(0, dy))) + float(dy) * float(dy));
        }
        return sqrt(best);
    }

    // Applies each style layer in order. `color` is premultiplied by the
    // combined layer coverage, `transmittance` is what remains of the base.
    static void cad_layer(thread float3 &color, thread float &transmittance, float4 layer, float coverage) {
        float alpha = coverage * layer.a;
        color = mix(color, layer.rgb, alpha);
        transmittance *= 1.0f - alpha;
    }

    static float4 cad_highlight(float4 distance, float scale, constant CADHighlightUniforms &u) {
        float3 color = float3(0.0f);
        float transmittance = 1.0f;
        // A one-pixel transition preserves the same logical widths on Retina
        // and independent agent captures, without changing the seed geometry.
        cad_layer(color, transmittance, u.hoverHaloColor, saturate(cadEdgeHoverHaloRadius * scale + 0.5f - distance.g));
        cad_layer(color, transmittance, u.hoverColor, saturate(cadEdgeHoverRadius * scale + 0.5f - distance.g));
        bool verticesEnabled = u.verticesEnabled > 0.5f;
        if (verticesEnabled) {
            cad_layer(color, transmittance, u.hoverHaloColor, saturate(cadVertexHoverHaloRadius * scale + 0.5f - distance.a));
            cad_layer(color, transmittance, u.hoverColor, saturate(cadVertexHoverRadius * scale + 0.5f - distance.a));
        }
        // Every selected feature wins over every hovered feature; vertices
        // stay legible where they coincide with a selected edge endpoint.
        cad_layer(color, transmittance, u.inkColor, saturate(cadEdgeSelectedHaloRadius * scale + 0.5f - distance.r));
        cad_layer(color, transmittance, u.selectedColor, saturate(cadEdgeSelectedRadius * scale + 0.5f - distance.r));
        if (verticesEnabled) {
            cad_layer(color, transmittance, u.inkColor, saturate(cadVertexSelectedHaloRadius * scale + 0.5f - distance.b));
            cad_layer(color, transmittance, u.selectedColor, saturate(cadVertexSelectedRadius * scale + 0.5f - distance.b));
        }
        return float4(color, 1.0f - transmittance);
    }

    fragment float4 cad_highlight_overlay(CADQuadOut in [[stage_in]], texture2d<float, access::read> horizontal [[texture(0)]],
                                          constant CADHighlightUniforms &u [[buffer(0)]]) {
        int2 local = int2(in.position.xy) - u.highlightArea.xy;
        float scale = cad_scale(u);
        return cad_highlight(cad_distances(local, horizontal, scale, u), scale, u);
    }

    fragment float4 cad_translucent_composite(CADQuadOut in [[stage_in]],
                                              texture2d<float, access::read> accumulated [[texture(0)]],
                                              texture2d<float, access::read> revealage [[texture(1)]],
                                              texture2d<float, access::read> decorations [[texture(2)]],
                                              texture2d<float, access::read> horizontal [[texture(3)]],
                                              texture2d<float, access::read> exact [[texture(4)]],
                                              constant CADHighlightUniforms &u [[buffer(0)]]) {
        int2 center = int2(in.position.xy);
        float4 sum = accumulated.read(uint2(center));
        // Exact layers accumulate in their own stored target.
        if (u.revealage.z > 0.5f) sum += exact.read(uint2(center));
        float reveal = saturate(revealage.read(uint2(center)).r);
        // Every layer has the same alpha, so without a transmittance pass the
        // accumulated alpha counts the layers: (1 - alpha) ^ count.
        if (u.revealage.x > 0.5f) reveal = pow(1.0f - u.revealage.y, max(sum.a, 0.0f) / u.revealage.y);
        float coverage = 1.0f - reveal;
        float3 averageColor = sum.rgb / max(sum.a, 1.0e-5f);
        float4 result = float4(averageColor * coverage + u.backgroundColor.rgb * reveal,
                               coverage + u.backgroundColor.a * reveal);
        float4 decoration = decorations.read(uint2(center));
        result = decoration + result * (1.0f - saturate(decoration.a));
        if (u.hasHighlights > 0.5f && all(center >= u.highlightArea.xy) && all(center <= u.highlightArea.zw)) {
            float scale = cad_scale(u);
            float4 highlight = cad_highlight(cad_distances(center - u.highlightArea.xy, horizontal, scale, u), scale, u);
            result = float4(highlight.rgb + result.rgb * (1.0f - highlight.a), result.a);
        }
        return result;
    }
    """
}
