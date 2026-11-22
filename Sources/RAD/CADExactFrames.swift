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

    init(field: CADFieldFrame, camera: simd_float4x4, projection: SIMD2<Float>, width: Int, height: Int, translucent: Bool) {
        self.camera = camera
        self.projection = projection
        self.width = width
        self.height = height
        resources = ObjectIdentifier(field.resources)
        instances = field.instances.map {
            translucent ? Instance(solid: $0.solid, model: $0.model, albedo: $0.albedo, emission: $0.emission)
                : Instance(solid: $0.solid, model: $0.model)
        }
        self.translucent = translucent
        complete = translucent && field.complete
        debug = field.debug
    }

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

/// Screen slices of one exact frame's geometry: 32-pixel tiles grouped,
/// in row order, into rectangles whose estimated covered pixels (each
/// proxy's projected bounds, counted per proxy) stay within a budget.
/// Tiles no proxy reaches are skipped; their stored layers keep the clear.
struct CADSlicePlan: Sendable {
    static let tile = 32

    struct Slice: Equatable, Sendable {
        /// Pixel rectangle (x0, y0, x1, y1 exclusive, top-left origin).
        var rect: SIMD4<Int32>
        var covered: Int
    }

    let width: Int
    let height: Int
    let columns: Int
    let rows: Int
    private var covered: [Int]
    private var rowTotals: [Int]
    private var firstColumn: [Int]
    private var lastColumn: [Int]
    private var row = 0
    private var column = 0
    /// Estimated covered pixels of the whole frame.
    let total: Int
    private(set) var remaining: Int

    /// `rects` are proxies' pixel rectangles; nil covers the whole frame.
    init(width: Int, height: Int, rects: [SIMD4<Int32>?]) {
        self.width = max(0, width)
        self.height = max(0, height)
        columns = (self.width + Self.tile - 1) / Self.tile
        rows = (self.height + Self.tile - 1) / Self.tile
        var covered = [Int](repeating: 0, count: columns * rows)
        let full = SIMD4<Int32>(0, 0, Int32(self.width), Int32(self.height))
        for rect in rects {
            let r = rect ?? full
            let x0 = Int(max(0, r.x)), y0 = Int(max(0, r.y))
            let x1 = Int(min(Int32(self.width), r.z)), y1 = Int(min(Int32(self.height), r.w))
            guard x0 < x1, y0 < y1 else { continue }
            for ty in (y0 / Self.tile)...((y1 - 1) / Self.tile) {
                let tileY0 = ty * Self.tile, tileY1 = min(self.height, tileY0 + Self.tile)
                let h = min(y1, tileY1) - max(y0, tileY0)
                for tx in (x0 / Self.tile)...((x1 - 1) / Self.tile) {
                    let tileX0 = tx * Self.tile, tileX1 = min(self.width, tileX0 + Self.tile)
                    covered[ty * columns + tx] += (min(x1, tileX1) - max(x0, tileX0)) * h
                }
            }
        }
        self.covered = covered
        let columns = self.columns, rows = self.rows
        let totals = (0..<rows).map { r in covered[(r * columns)..<((r + 1) * columns)].reduce(0, +) }
        rowTotals = totals
        firstColumn = (0..<rows).map { r in (0..<columns).first { covered[r * columns + $0] > 0 } ?? columns }
        lastColumn = (0..<rows).map { r in (0..<columns).last { covered[r * columns + $0] > 0 } ?? -1 }
        total = totals.reduce(0, +)
        remaining = total
    }

    var isDone: Bool { remaining == 0 }

    /// The next slice whose estimated covered pixels stay within `budget`
    /// (at least one tile), or nil when every covered tile has its slice.
    mutating func next(budget: Int) -> Slice? {
        while row < rows {
            if column < firstColumn[row] { column = firstColumn[row] }
            while column <= lastColumn[row], covered[row * columns + column] == 0 { column += 1 }
            if column > lastColumn[row] { row += 1; column = 0; continue }
            break
        }
        guard row < rows else { return nil }
        let tile = Int32(Self.tile)
        if column == firstColumn[row] {
            // Whole rows, as one band, while they fit.
            var sum = 0, end = row, low = columns, high = -1
            while end < rows, sum + rowTotals[end] <= budget {
                sum += rowTotals[end]
                if rowTotals[end] > 0 { low = min(low, firstColumn[end]); high = max(high, lastColumn[end]) }
                end += 1
            }
            if end > row, sum > 0 {
                let rect = SIMD4<Int32>(Int32(low) * tile, Int32(row) * tile, min(Int32(width), Int32(high + 1) * tile),
                                        min(Int32(height), Int32(end) * tile))
                row = end
                column = 0
                remaining -= sum
                return Slice(rect: rect, covered: sum)
            }
        }
        // Part of a row: consecutive tiles while they fit.
        var sum = covered[row * columns + column], end = column + 1
        while end <= lastColumn[row], sum + covered[row * columns + end] <= budget {
            sum += covered[row * columns + end]
            end += 1
        }
        let rect = SIMD4<Int32>(Int32(column) * tile, Int32(row) * tile, min(Int32(width), Int32(end) * tile),
                                min(Int32(height), Int32(row + 1) * tile))
        column = end
        if column > lastColumn[row] { row += 1; column = 0 }
        remaining -= sum
        return Slice(rect: rect, covered: sum)
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

    private let lock = NSLock()
    private var slices: [Key: Double] = [:]
    private var frames: [Key: Double] = [:]

    func sliceRate(_ key: Key) -> Double {
        lock.withLock { slices[key] } ?? (key.translucent ? Self.defaultTranslucentRate : Self.defaultOpaqueRate)
    }

    func frameRate(_ key: Key) -> Double {
        lock.withLock { frames[key] ?? slices[key] } ?? (key.translucent ? Self.defaultTranslucentRate : Self.defaultOpaqueRate)
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
    /// Exact masks of the batch, in its pass order.
    let masks: [CADViewportFrame.Mask]
    var plan: CADSlicePlan
    var cleared = false
    var encoded = 0
    var completed = 0
    var isComplete: Bool { plan.isDone && completed == encoded && cleared }

    init?(key: Key, masks: [CADViewportFrame.Mask], device: any MTLDevice, rects: [SIMD4<Int32>?]) {
        let width = Int(key.rect.z - key.rect.x), height = Int(key.rect.w - key.rect.y)
        guard width > 0, height > 0 else { return nil }
        let descriptor = MTLTextureDescriptor.texture2DDescriptor(pixelFormat: .rgba8Unorm, width: width, height: height,
                                                                  mipmapped: false)
        descriptor.usage = [.renderTarget, .shaderRead]
        descriptor.storageMode = .private
        guard let texture = device.makeTexture(descriptor: descriptor) else { return nil }
        self.key = key
        self.texture = texture
        self.masks = masks
        // Plan in the batch's own pixels.
        let local = rects.map { rect in
            rect.map { SIMD4($0.x - key.rect.x, $0.y - key.rect.y, $0.z - key.rect.x, $0.w - key.rect.y) }
        }
        plan = CADSlicePlan(width: width, height: height, rects: local)
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
