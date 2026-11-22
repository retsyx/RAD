import Foundation
import Metal
import simd

/// What a frame's stored exact layers depend on: the design camera, the
/// frame size, the visible exact solids and their placements, their GPU
/// resources, the display mode and diagnostics. A newer frame with the same
/// key composites the stored layers; any other key draws them again.
/// Translucent layers bake their colors, so colors belong to their key.
struct CADExactKey: Equatable, @unchecked Sendable {
    struct Instance: Equatable, Sendable {
        var solid: Int
        var model: simd_float4x4
        var albedo = SIMD3<Float>.zero
        var emission = SIMD3<Float>.zero
    }

    var camera: simd_float4x4
    /// Vertical field of view or orthographic scale, and the projection kind.
    var projection: SIMD2<Float>
    var width: Int
    var height: Int
    var resources: ObjectIdentifier
    var instances: [Instance]
    var translucent: Bool
    var complete: Bool
    var debug: Int

    /// Whether a scene pass with `view` and `projection` draws the view of
    /// layers drawn with `view0` and `projection0`. Depth ranges may differ:
    /// the composite maps stored depth into the pass's range.
    static func sameView(_ view: simd_float4x4, _ projection: simd_float4x4,
                         _ view0: simd_float4x4, _ projection0: simd_float4x4) -> Bool {
        view == view0 && projection.columns.0 == projection0.columns.0 && projection.columns.1 == projection0.columns.1
            && projection.columns.2.x == projection0.columns.2.x && projection.columns.2.y == projection0.columns.2.y
            && projection.columns.2.w == projection0.columns.2.w && projection.columns.3.w == projection0.columns.3.w
    }
}

extension CADExactKey {
    init(field: CADFieldFrame, camera: simd_float4x4, projection: SIMD2<Float>, width: Int, height: Int, translucent: Bool) {
        self.init(camera: camera, projection: projection, width: width, height: height,
                  resources: ObjectIdentifier(field.resources),
                  instances: Self.instances(field.instances, translucent: translucent), translucent: translucent,
                  complete: translucent && field.complete, debug: field.debug)
    }

    /// The key's solids: opaque layers store parts, not colors, so selection
    /// and color changes composite them again; translucent layers bake colors.
    static func instances(_ instances: [CADFieldInstance], translucent: Bool) -> [Instance] {
        instances.map {
            translucent ? Instance(solid: $0.solid, model: $0.model, albedo: $0.albedo, emission: $0.emission)
                : Instance(solid: $0.solid, model: $0.model)
        }
    }
}

/// Stored exact layers of one frame size, allocated when an exact frame
/// first needs them. Opaque frames store each sample's normal, part and
/// depth; translucent frames store visibility depth, transmittance (only
/// while some visible solid draws its display mesh) and resolved
/// accumulated color.
final class CADExactTargets: @unchecked Sendable {
    let width: Int
    let height: Int
    let translucent: Bool
    /// Opaque: geometry depth; translucent: visibility depth.
    let depth: any MTLTexture
    let normal: (any MTLTexture)?
    let part: (any MTLTexture)?
    let revealage: (any MTLTexture)?
    let accumulated: (any MTLTexture)?

    init?(device: any MTLDevice, width: Int, height: Int, translucent: Bool, revealage: Bool) {
        func make(_ format: MTLPixelFormat, multisample: Bool = true) -> (any MTLTexture)? {
            let descriptor = MTLTextureDescriptor.texture2DDescriptor(pixelFormat: format, width: width, height: height,
                                                                      mipmapped: false)
            if multisample {
                descriptor.textureType = .type2DMultisample
                descriptor.sampleCount = CADFieldResources.sampleCount
            }
            descriptor.usage = [.renderTarget, .shaderRead]
            descriptor.storageMode = .private
            return device.makeTexture(descriptor: descriptor)
        }
        guard width > 0, height > 0, let depth = make(CADFieldResources.depthFormat) else { return nil }
        self.width = width
        self.height = height
        self.translucent = translucent
        self.depth = depth
        if translucent {
            normal = nil
            part = nil
            if revealage {
                guard let texture = make(.r16Float) else { return nil }
                self.revealage = texture
            } else {
                self.revealage = nil
            }
            guard let accumulated = make(.rgba16Float, multisample: false) else { return nil }
            self.accumulated = accumulated
        } else {
            guard let normal = make(CADFieldResources.normalFormat), let part = make(CADFieldResources.partFormat) else {
                return nil
            }
            self.normal = normal
            self.part = part
            self.revealage = nil
            self.accumulated = nil
        }
    }

    /// Bytes of GPU memory these layers hold.
    var bytes: Int {
        [depth, normal, part, revealage, accumulated].compactMap { $0 }.reduce(0) { $0 + $1.allocatedSize }
    }

    func fits(width: Int, height: Int, translucent: Bool, revealage: Bool) -> Bool {
        self.width == width && self.height == height && self.translucent == translucent
            && (!revealage || self.revealage != nil)
    }
}

/// Screen slices of one exact frame's geometry: square blocks of 256 pixels
/// that some proxy reaches, in row order, grouped into slices whose
/// estimated covered pixels (each proxy's projected bounds, counted per
/// proxy) stay within a budget, at least one block each. Small scissored
/// regions cost the GPU far more per pixel than one pass over the frame, so
/// a block is the smallest slice. Blocks no proxy reaches get no slice;
/// their stored layers keep the clear.
struct CADSlicePlan: Sendable {
    static let block = 256

    struct Slice: Equatable, Sendable {
        /// Pixel rectangles (x0, y0, x1, y1 exclusive, top-left origin) of
        /// the slice's blocks, adjacent blocks of a row joined; empty when
        /// the slice only clears.
        var rects: [SIMD4<Int32>]
        var covered: Int
        /// The slice holds every block of the plan, so it needs no scissor.
        var whole = false
    }

    let width: Int
    let height: Int
    /// Block side in pixels.
    let block: Int
    let columns: Int
    let rows: Int
    /// Reached blocks (row, column, estimated covered pixels) in row order.
    private let blocks: [(row: Int, column: Int, covered: Int)]
    private var next = 0
    /// Estimated covered pixels of the whole frame.
    let total: Int
    private(set) var remaining: Int

    /// `rects` are proxies' pixel rectangles; nil covers the whole frame.
    init(width: Int, height: Int, rects: [SIMD4<Int32>?], block size: Int = Self.block) {
        self.width = max(0, width)
        self.height = max(0, height)
        block = size
        columns = (self.width + size - 1) / size
        rows = (self.height + size - 1) / size
        var covered = [Int](repeating: 0, count: columns * rows)
        let full = SIMD4<Int32>(0, 0, Int32(self.width), Int32(self.height))
        for rect in rects {
            let r = rect ?? full
            let x0 = Int(max(0, r.x)), y0 = Int(max(0, r.y))
            let x1 = Int(min(Int32(self.width), r.z)), y1 = Int(min(Int32(self.height), r.w))
            guard x0 < x1, y0 < y1 else { continue }
            for by in (y0 / size)...((y1 - 1) / size) {
                let h = min(y1, (by + 1) * size) - max(y0, by * size)
                for bx in (x0 / size)...((x1 - 1) / size) {
                    covered[by * columns + bx] += (min(x1, (bx + 1) * size) - max(x0, bx * size)) * h
                }
            }
        }
        let columns = self.columns
        blocks = covered.enumerated().filter { $0.element > 0 }
            .map { (row: $0.offset / columns, column: $0.offset % columns, covered: $0.element) }
        total = blocks.reduce(0) { $0 + $1.covered }
        remaining = total
    }

    var isDone: Bool { next >= blocks.count }

    /// The pixel rectangle of the blocks of `row` from `first` through `last`.
    private func rect(row: Int, first: Int, last: Int) -> SIMD4<Int32> {
        let size = block
        return SIMD4(Int32(first * size), Int32(row * size), Int32(min(width, (last + 1) * size)),
                     Int32(min(height, (row + 1) * size)))
    }

    /// The next slice whose estimated covered pixels stay within `budget`,
    /// at least one block, or nil when every reached block has its slice.
    mutating func next(budget: Int) -> Slice? {
        guard next < blocks.count else { return nil }
        let first = next
        var sum = blocks[next].covered
        next += 1
        while next < blocks.count, sum + blocks[next].covered <= budget {
            sum += blocks[next].covered
            next += 1
        }
        remaining -= sum
        var rects: [SIMD4<Int32>] = []
        var start = blocks[first]
        var previous = start
        for block in blocks[(first + 1)..<next] {
            if block.row == previous.row, block.column == previous.column + 1 {
                previous = block
                continue
            }
            rects.append(rect(row: start.row, first: start.column, last: previous.column))
            start = block
            previous = block
        }
        rects.append(rect(row: start.row, first: start.column, last: previous.column))
        return Slice(rects: rects, covered: sum, whole: first == 0 && next == blocks.count)
    }
}

/// Measured exact costs, kept per field resources, frame size and display
/// mode: the largest GPU seconds per estimated covered pixel of a slice
/// (slices plan with it, so they stay within their budget) and of a complete
/// frame's geometry (whether one frame can draw it at once).
final class CADExactCost: @unchecked Sendable {
    struct Key: Hashable, Sendable {
        var resources: ObjectIdentifier
        var width: Int
        var height: Int
        var translucent: Bool
    }

    static let shared = CADExactCost()
    /// Before any measurement: conservative rates per covered pixel.
    static let defaultOpaqueRate = 600e-9
    static let defaultTranslucentRate = 3e-6
    /// A slice's fixed cost, excluded from its rate.
    static let sliceOverhead = 40e-6
    /// Slices covering fewer pixels do not change the rate.
    static let minimumCovered = 4_096

    /// A slice slower than this, or one the GPU fails, ends exact faces of
    /// its design in that display mode for the session: such geometry would
    /// stall or reset the GPU on every view.
    static let failingSliceSeconds = 0.25

    struct Failure: Hashable, Sendable {
        var resources: ObjectIdentifier
        var translucent: Bool
    }

    private let lock = NSLock()
    private var slices: [Key: Double] = [:]
    private var frames: [Key: Double] = [:]
    private var failures = Set<Failure>()

    func recordFailure(_ key: Key) {
        lock.withLock { _ = failures.insert(Failure(resources: key.resources, translucent: key.translucent)) }
    }

    func hasFailed(resources: ObjectIdentifier, translucent: Bool) -> Bool {
        lock.withLock { failures.contains(Failure(resources: resources, translucent: translucent)) }
    }

    /// Rates never fall below this: a frame's covered pixels are always worth some time.
    static let smallestRate = 1e-10

    func sliceRate(_ key: Key) -> Double {
        max(Self.smallestRate, lock.withLock { slices[key] } ?? (key.translucent ? Self.defaultTranslucentRate : Self.defaultOpaqueRate))
    }

    func frameRate(_ key: Key) -> Double {
        max(Self.smallestRate, lock.withLock { frames[key] ?? slices[key] }
            ?? (key.translucent ? Self.defaultTranslucentRate : Self.defaultOpaqueRate))
    }

    func recordSlice(_ key: Key, seconds: Double, covered: Int) {
        guard covered >= Self.minimumCovered, seconds.isFinite, seconds > 0 else { return }
        let rate = max(0, seconds - Self.sliceOverhead) / Double(covered)
        lock.withLock { slices[key] = max(slices[key] ?? 0, rate) }
    }

    func recordFrame(_ key: Key, seconds: Double, covered: Int) {
        guard covered > 0, seconds.isFinite, seconds > 0 else { return }
        let rate = seconds / Double(covered)
        lock.withLock { frames[key] = max(frames[key] ?? 0, rate) }
    }

    /// Forgets the measurements of resources that no longer exist.
    func forget(_ resources: ObjectIdentifier) {
        lock.withLock {
            slices = slices.filter { $0.key.resources != resources }
            frames = frames.filter { $0.key.resources != resources }
            failures = failures.filter { $0.resources != resources }
        }
    }
}

/// The outline coverage of one highlight batch's exact parts, marched once
/// per view into a stored texture covering the batch, then reused by every
/// frame of that view until the selection or placements change.
final class CADMaskCoverage: @unchecked Sendable {
    struct Key: Hashable, Sendable {
        var rect: SIMD4<Int32>
        var items: [Item]
        struct Item: Hashable, Sendable {
            var solid: Int
            var model: [Float]
            var selected: Bool
        }
    }

    let key: Key
    let texture: any MTLTexture
    /// The batch's exact parts: their proxy draws and outline channels.
    let items: [(draw: CADFieldDraw, selected: Bool)]
    var plan: CADSlicePlan
    var cleared = false
    var encoded = 0
    var completed = 0
    var isComplete: Bool { plan.isDone && completed == encoded && cleared }

    init?(key: Key, items: [(draw: CADFieldDraw, selected: Bool)], device: any MTLDevice) {
        let width = Int(key.rect.z - key.rect.x), height = Int(key.rect.w - key.rect.y)
        guard width > 0, height > 0 else { return nil }
        // Multisample like the batch's own coverage pass, which resolves it.
        let descriptor = MTLTextureDescriptor.texture2DDescriptor(pixelFormat: .rgba8Unorm, width: width, height: height,
                                                                  mipmapped: false)
        descriptor.textureType = .type2DMultisample
        descriptor.sampleCount = CADFieldResources.sampleCount
        descriptor.usage = [.renderTarget, .shaderRead]
        descriptor.storageMode = .private
        guard let texture = device.makeTexture(descriptor: descriptor) else { return nil }
        self.key = key
        self.texture = texture
        self.items = items
        // Plan in the batch's own pixels.
        let origin = SIMD4(key.rect.x, key.rect.y, key.rect.x, key.rect.y)
        plan = CADSlicePlan(width: width, height: height, rects: items.map { $0.draw.rect.map { $0 &- origin } })
    }
}

/// One exact frame's geometry in progress or stored: drawn in slices into
/// stored layers, then composited by every frame with the same key.
final class CADExactJob: @unchecked Sendable {
    let key: CADExactKey
    let field: CADFieldFrame
    let view: simd_float4x4
    let projection: simd_float4x4
    let width: Int
    let height: Int
    let targets: CADExactTargets
    let draws: [CADFieldDraw]
    let costKey: CADExactCost.Key
    var plan: CADSlicePlan
    /// The first slice clears the stored layers.
    var cleared = false
    var encoded = 0
    var completed = 0
    /// GPU seconds of the completed slices.
    var gpuSeconds = 0.0
    var masks: [CADMaskCoverage.Key: CADMaskCoverage] = [:]
    /// Whether a composite of this geometry has reported its total cost.
    var reported = false

    init(key: CADExactKey, field: CADFieldFrame, view: simd_float4x4, projection: simd_float4x4,
         width: Int, height: Int, targets: CADExactTargets) {
        self.key = key
        self.field = field
        self.view = view
        self.projection = projection
        self.width = width
        self.height = height
        self.targets = targets
        draws = field.resources.draws(field.instances, view: view, projection: projection, width: width, height: height,
                                      contacts: !key.translucent)
        costKey = .init(resources: key.resources, width: width, height: height, translucent: key.translucent)
        plan = CADSlicePlan(width: width, height: height, rects: draws.map(\.rect))
    }

    var isComplete: Bool { cleared && plan.isDone && completed == encoded }
    var hasSlicesToEncode: Bool { !cleared || !plan.isDone }

    /// Pixel rectangles of the proxies of `solids`.
    func rects(of solids: Set<Int>) -> [SIMD4<Int32>?] {
        draws.filter { solids.contains($0.solid) }.map(\.rect)
    }
}

/// The stored layers SceneKit's scene passes composite, with the view they
/// were drawn from (CADFieldPassDelegate draws them).
struct CADExactComposite: @unchecked Sendable {
    let layers: CADExactTargets
    let view: simd_float4x4
    let projection: simd_float4x4
    /// Albedo and emission per stored part value (solid + 1).
    let materials: (any MTLBuffer)?
    let lights: CADFieldLights
    let debug: Int
    let pipelines: CADExactPipelines

    /// Mirrors `CADExactUniforms` in the shader source.
    struct Uniforms {
        var ambient: SIMD4<Float>
        var lightDirection0: SIMD4<Float>
        var lightDirection1: SIMD4<Float>
        var lightColor0: SIMD4<Float>
        var lightColor1: SIMD4<Float>
        /// Metalness, roughness.
        var material: SIMD4<Float>
        /// P00, P11, P20, P21 of the stored layers' projection.
        var projection: SIMD4<Float>
        /// Width, height, orthographic, debug.
        var frame: SIMD4<Float>
        /// Depth terms (a, b, c, d: ndc = (a z + b) / (c z + d)) of the stored
        /// layers' projection and of the pass's, and whether they differ.
        var sourceDepth: SIMD4<Float>
        var targetDepth: SIMD4<Float>
        var remap: SIMD4<Float>
    }

    static func depthTerms(_ p: simd_float4x4) -> SIMD4<Float> {
        SIMD4(p.columns.2.z, p.columns.3.z, p.columns.2.w, p.columns.3.w)
    }

    func uniforms(passProjection: simd_float4x4) -> Uniforms {
        let directions = lights.directions + Array(repeating: SIMD3<Float>(0, 0, 1), count: 2)
        let colors = lights.colors + Array(repeating: SIMD3<Float>.zero, count: 2)
        let source = Self.depthTerms(projection), target = Self.depthTerms(passProjection)
        return Uniforms(ambient: SIMD4(lights.ambient, 0), lightDirection0: SIMD4(directions[0], 0),
                        lightDirection1: SIMD4(directions[1], 0), lightColor0: SIMD4(colors[0], 0),
                        lightColor1: SIMD4(colors[1], 0),
                        material: SIMD4(CADFieldResources.metalness, CADFieldResources.roughness, 0, 0),
                        projection: SIMD4(projection.columns.0.x, projection.columns.1.y, projection.columns.2.x,
                                          projection.columns.2.y),
                        frame: SIMD4(Float(layers.width), Float(layers.height), projection.columns.3.w == 1 ? 1 : 0,
                                     Float(debug)),
                        sourceDepth: source, targetDepth: target, remap: SIMD4(source == target ? 0 : 1, 0, 0, 0))
    }

    func encode(_ pass: CADFieldPassDelegate.Pass, encoder: any MTLRenderCommandEncoder, colorFormat: MTLPixelFormat,
                depthFormat: MTLPixelFormat, samples: Int, projection passProjection: simd_float4x4) {
        let kind: CADExactPipelines.Kind
        switch pass {
        case .opaque:
            guard layers.normal != nil, layers.part != nil, materials != nil else { return }
            kind = .opaque
        case .visibility: kind = .visibility
        case .revealage:
            guard layers.revealage != nil else { return }
            kind = .revealage
        case .accumulation: return
        }
        guard let pipeline = try? pipelines.pipeline(kind, color: colorFormat, depth: depthFormat, samples: samples) else { return }
        var values = uniforms(passProjection: passProjection)
        encoder.setRenderPipelineState(pipeline)
        encoder.setDepthStencilState(kind == .revealage ? pipelines.depthIgnored : pipelines.depthAlways)
        encoder.setCullMode(.none)
        encoder.setTriangleFillMode(.fill)
        encoder.setFragmentBytes(&values, length: MemoryLayout<Uniforms>.stride, index: 0)
        if let materials { encoder.setFragmentBuffer(materials, offset: 0, index: 1) }
        encoder.setFragmentTexture(layers.normal, index: 0)
        encoder.setFragmentTexture(layers.part, index: 1)
        encoder.setFragmentTexture(layers.depth, index: 2)
        encoder.setFragmentTexture(layers.revealage, index: 3)
        encoder.drawPrimitives(type: .triangle, vertexStart: 0, vertexCount: 3)
    }

    /// Per-part materials of a frame's instances, indexed by solid + 1.
    static func materials(_ instances: [CADFieldInstance], device: any MTLDevice) -> (any MTLBuffer)? {
        let count = (instances.map(\.solid).max() ?? -1) + 2
        var values = [SIMD4<Float>](repeating: .zero, count: 2 * count)
        for instance in instances where instance.solid >= 0 {
            values[2 * (instance.solid + 1)] = SIMD4(instance.albedo, 1)
            values[2 * (instance.solid + 1) + 1] = SIMD4(instance.emission, 0)
        }
        return device.makeBuffer(bytes: values, length: values.count * MemoryLayout<SIMD4<Float>>.stride,
                                 options: .storageModeShared)
    }
}

/// Pipelines that composite stored exact layers inside SceneKit's scene
/// passes, per device and pass formats. They run per sample.
final class CADExactPipelines: @unchecked Sendable {
    enum Kind: String { case opaque, visibility, revealage }

    let device: any MTLDevice
    let depthAlways: any MTLDepthStencilState
    let depthIgnored: any MTLDepthStencilState
    private let library: any MTLLibrary
    private let lock = NSLock()
    private var pipelines: [String: any MTLRenderPipelineState] = [:]

    private static let lock = NSLock()
    private static var cache: [UInt64: CADExactPipelines] = [:]

    static func shared(for device: any MTLDevice) throws -> CADExactPipelines {
        try lock.withLock {
            if let cached = cache[device.registryID] { return cached }
            let created = try CADExactPipelines(device: device)
            cache[device.registryID] = created
            return created
        }
    }

    private init(device: any MTLDevice) throws {
        self.device = device
        let options = MTLCompileOptions()
        if #available(macOS 15.0, *) { options.mathMode = .safe } else { options.fastMathEnabled = false }
        library = try device.makeLibrary(source: Self.shaderSource, options: options)
        func depth(_ compare: MTLCompareFunction, write: Bool) throws -> any MTLDepthStencilState {
            let descriptor = MTLDepthStencilDescriptor()
            descriptor.depthCompareFunction = compare
            descriptor.isDepthWriteEnabled = write
            guard let state = device.makeDepthStencilState(descriptor: descriptor) else {
                throw RADError.message("RAD could not prepare the design's exact surfaces.")
            }
            return state
        }
        depthAlways = try depth(.always, write: true)
        depthIgnored = try depth(.always, write: false)
    }

    func pipeline(_ kind: Kind, color: MTLPixelFormat, depth: MTLPixelFormat, samples: Int) throws -> any MTLRenderPipelineState {
        let key = "\(kind.rawValue)/\(color.rawValue)/\(depth.rawValue)/\(samples)"
        if let pipeline = lock.withLock({ pipelines[key] }) { return pipeline }
        let descriptor = MTLRenderPipelineDescriptor()
        descriptor.label = "RAD exact composite \(kind.rawValue)"
        descriptor.vertexFunction = library.makeFunction(name: "cad_exact_fullscreen")
        descriptor.fragmentFunction = library.makeFunction(name: "cad_exact_\(kind.rawValue)")
        descriptor.rasterSampleCount = samples
        descriptor.depthAttachmentPixelFormat = depth
        let attachment: MTLRenderPipelineColorAttachmentDescriptor = descriptor.colorAttachments[0]
        attachment.pixelFormat = color
        switch kind {
        case .opaque: break
        case .visibility: attachment.writeMask = []
        case .revealage:
            attachment.isBlendingEnabled = true
            attachment.rgbBlendOperation = .add
            attachment.alphaBlendOperation = .add
            attachment.sourceRGBBlendFactor = .zero
            attachment.destinationRGBBlendFactor = .sourceColor
            attachment.sourceAlphaBlendFactor = .zero
            attachment.destinationAlphaBlendFactor = .sourceAlpha
        }
        let pipeline = try device.makeRenderPipelineState(descriptor: descriptor)
        lock.withLock { pipelines[key] = pipeline }
        return pipeline
    }

    private static let shaderSource = """
    #include <metal_stdlib>
    using namespace metal;
    \(CADFieldShading.source)

    struct CADExactQuadOut { float4 position [[position]]; };

    vertex CADExactQuadOut cad_exact_fullscreen(uint vid [[vertex_id]]) {
        CADExactQuadOut out;
        out.position = float4(vid == 1 ? 3.0f : -1.0f, vid == 2 ? 3.0f : -1.0f, 0.0f, 1.0f);
        return out;
    }

    struct CADExactUniforms {
        float4 ambient;
        float4 lightDirection0;
        float4 lightDirection1;
        float4 lightColor0;
        float4 lightColor1;
        float4 material;     // metalness, roughness
        float4 projection;   // P00, P11, P20, P21
        float4 frame;        // width, height, orthographic, debug
        float4 sourceDepth;  // stored layers' depth terms: ndc = (a z + b) / (c z + d)
        float4 targetDepth;  // the pass's
        float4 remap;        // x: the depth terms differ
    };

    struct CADExactMaterial { float4 albedo; float4 emission; };

    // Stored depth in the pass's depth range (SceneKit fits it to the scene).
    static float cad_exact_depth(float d, constant CADExactUniforms &u) {
        if (u.remap.x < 0.5f) return d;
        const float4 s = u.sourceDepth, t = u.targetDepth;
        const float z = (s.y - d * s.w) / (d * s.z - s.x);
        return (t.x * z + t.y) / (t.z * z + t.w);
    }

    struct CADExactOpaqueOut {
        float4 color [[color(0)]];
        float depth [[depth(any)]];
    };

    // Shades each stored sample as the exact pass did where it was marched:
    // the view ray through the pixel center, roughness widened by the
    // normal's change across the quad (fully at part boundaries, where the
    // marched quad's neighbors missed or belonged to another part).
    fragment CADExactOpaqueOut cad_exact_opaque(CADExactQuadOut in [[stage_in]], uint sample [[sample_id]],
                                                texture2d_ms<float, access::read> normals [[texture(0)]],
                                                texture2d_ms<ushort, access::read> parts [[texture(1)]],
                                                depth2d_ms<float, access::read> depths [[texture(2)]],
                                                constant CADExactUniforms &u [[buffer(0)]],
                                                constant CADExactMaterial *materials [[buffer(1)]]) {
        const uint2 p = uint2(in.position.xy);
        const uint part = parts.read(p, sample).r;
        const float2 e = normals.read(p, sample).rg;
        const float3 n = cad_octahedral_decode(e);
        const float3 dx = dfdx(n), dy = dfdy(n);
        const bool boundary = fwidth(float(part)) > 0.0f;
        if (part == 0) discard_fragment();
        const int debug = int(u.frame.w);
        float3 color;
        if (debug == 1) {
            color = float3(e.x);
        } else if (debug == 2) {
            color = n * 0.5f + 0.5f;
        } else if (debug == 3) {
            color = e.x > 0.5f ? float3(1, 0, 0) : float3(0.2f);
        } else {
            const float2 center = float2(p) + 0.5f;
            const float2 ndc = float2(center.x / u.frame.x * 2.0f - 1.0f, 1.0f - center.y / u.frame.y * 2.0f);
            const float3 v = u.frame.z > 0.5f ? float3(0, 0, 1)
                : normalize(float3(-(ndc.x + u.projection.z) / u.projection.x, -(ndc.y + u.projection.w) / u.projection.y, 1.0f));
            const float variance = boundary ? 0.18f : 0.25f * (dot(dx, dx) + dot(dy, dy));
            color = cad_shade_lit(n, v, materials[part].albedo.rgb, materials[part].emission.rgb, u.material.x, u.material.y,
                                  variance, u.ambient.rgb, u.lightDirection0.xyz, u.lightColor0.rgb,
                                  u.lightDirection1.xyz, u.lightColor1.rgb);
        }
        CADExactOpaqueOut out;
        out.color = float4(color, 1.0f);
        out.depth = cad_exact_depth(depths.read(p, sample), u);
        return out;
    }

    struct CADExactDepthOut { float depth [[depth(any)]]; };

    // Translucent mode's solid visibility depth, for ordinary decorations.
    fragment CADExactDepthOut cad_exact_visibility(CADExactQuadOut in [[stage_in]], uint sample [[sample_id]],
                                                   depth2d_ms<float, access::read> depths [[texture(2)]],
                                                   constant CADExactUniforms &u [[buffer(0)]]) {
        const float d = depths.read(uint2(in.position.xy), sample);
        if (d <= 0.0f) discard_fragment();
        CADExactDepthOut out;
        out.depth = cad_exact_depth(d, u);
        return out;
    }

    // Transmittance of the exact layers, multiplied into the pass's.
    fragment float4 cad_exact_revealage(CADExactQuadOut in [[stage_in]], uint sample [[sample_id]],
                                        texture2d_ms<float, access::read> revealage [[texture(3)]]) {
        return float4(revealage.read(uint2(in.position.xy), sample).r);
    }
    """
}
