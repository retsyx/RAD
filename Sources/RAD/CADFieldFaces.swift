import AppKit
import Metal
import SceneKit
import simd

/// The build's exact-face proxies (result.json `proxy`): per part, a closed
/// outward mesh of the offset surface whose triangles lie in the shell
/// 0 < f <= `shell` of the part's field (SDF-DESIGN.md, D6).
struct CADFieldProxyManifest: Decodable, Sendable, Equatable {
    struct Part: Decodable, Sendable, Equatable {
        var solid: Int
        var status: String
        var triangles: Int?
        var shell: Double?
        var minimum: Double?
        var maxDeviation: Double?
        var reason: String?
    }
    var offset: Double?
    var cellSize: Double?
    var parts: [Part]?
    var error: String?

    private struct Envelope: Decodable { let proxy: CADFieldProxyManifest? }

    static func manifest(in resultJSON: Data) throws -> CADFieldProxyManifest? {
        try JSONDecoder().decode(Envelope.self, from: resultJSON).proxy
    }
}

/// Validated CPU inputs of the exact face pass: the field program and every
/// field part's proxy triangles. Immutable; copies of one preview share it
/// and its GPU resources.
struct CADFieldGeometry: Sendable {
    struct Part: Sendable, Equatable {
        let solid: Int
        /// First index and index count of the part's proxy triangles.
        let indexStart: Int
        let indexCount: Int
        /// Field values on the proxy never exceed this bound.
        let shell: Float
        /// Center and corners of the part's bounds in built coordinates, for
        /// draw order and contact parts.
        let center: SIMD3<Float>
        let lower: SIMD3<Float>
        let upper: SIMD3<Float>
        /// Display mesh deviation bound: crease strokes come from the display
        /// mesh, so exact faces may hide them by up to this much.
        let deviation: Float
    }

    let program: FieldProgram
    let positions: Data
    let indices: Data
    let vertexCount: Int
    let parts: [Int: Part]
    let gpu: CADFieldGPU

    /// The proxies of a validated preview, or nil when the build has none.
    /// Parts whose proxy or program record is unusable keep mesh rendering.
    static func prepare(_ mesh: PreviewMesh, solidBounds: [Int: [Double]]) throws -> CADFieldGeometry? {
        guard let program = mesh.fieldProgram, let manifest = mesh.proxy, manifest.error == nil,
              let records = manifest.parts, let positions = mesh.proxyPositions,
              let triangles = mesh.proxyTriangleIndices, let owners = mesh.proxySolids,
              positions.count.isMultiple(of: 3), triangles.count.isMultiple(of: 3),
              owners.count == triangles.count / 3, !owners.isEmpty, positions.count / 3 <= 4_000_000,
              owners.count <= 4_000_000, program.parts.count == mesh.solids else { return nil }
        for (index, value) in positions.enumerated() {
            if index.isMultiple(of: 4_096) { try Task.checkCancellation() }
            guard value.isFinite, abs(value) < 1e9 else { return nil }
        }
        let vertexCount = UInt32(positions.count / 3)
        for (index, vertex) in triangles.enumerated() {
            if index.isMultiple(of: 4_096) { try Task.checkCancellation() }
            guard vertex < vertexCount else { return nil }
        }
        var ranges: [Int: (start: Int, count: Int)] = [:]
        var previous = -1
        for (triangle, solid) in owners.enumerated() {
            if triangle.isMultiple(of: 4_096) { try Task.checkCancellation() }
            guard solid >= 0, solid < mesh.solids, solid >= previous else { return nil }
            if solid != previous { ranges[solid] = (triangle * 3, 0) }
            ranges[solid]!.count += 3
            previous = solid
        }
        var deviations: [Int: Double] = [:]
        for part in mesh.partStats ?? [] { deviations[part.solidIndex] = part.maxDeviation }
        var parts: [Int: Part] = [:]
        for record in records where record.status == "proxy" {
            guard let range = ranges[record.solid], let shell = record.shell, shell.isFinite, shell > 0,
                  record.solid < program.parts.count, program.parts[record.solid].hasField,
                  record.triangles == range.count / 3 else { continue }
            let bounds = solidBounds[record.solid] ?? mesh.bounds
            let lower = bounds.count == 6 ? SIMD3<Float>(Float(bounds[0]), Float(bounds[1]), Float(bounds[2])) : .zero
            let upper = bounds.count == 6 ? SIMD3<Float>(Float(bounds[3]), Float(bounds[4]), Float(bounds[5])) : .zero
            let center = (lower + upper) / 2
            let deviation = deviations[record.solid].flatMap { $0.isFinite && $0 >= 0 ? Float($0) : nil } ?? 0
            parts[record.solid] = Part(solid: record.solid, indexStart: range.start, indexCount: range.count,
                                       shell: Float(shell), center: center, lower: lower, upper: upper,
                                       deviation: deviation)
        }
        guard !parts.isEmpty else { return nil }
        try Task.checkCancellation()
        return CADFieldGeometry(program: program, positions: positions.withUnsafeBufferPointer { Data(buffer: $0) },
                                indices: triangles.withUnsafeBufferPointer { Data(buffer: $0) },
                                vertexCount: Int(vertexCount), parts: parts, gpu: CADFieldGPU())
    }
}

/// The GPU resources of one preview's field geometry, prepared once per
/// device off the main thread and shared by the live view and captures.
final class CADFieldGPU: @unchecked Sendable {
    private let lock = NSLock()
    private var ready: [UInt64: CADFieldResources] = [:]
    private var pending: [UInt64: Task<CADFieldResources?, Never>] = [:]

    func cached(device: any MTLDevice) -> CADFieldResources? { lock.withLock { ready[device.registryID] } }

    /// Joins or starts the single preparation for `device`; nil when the
    /// device cannot compile or allocate them (parts then draw their meshes).
    func resources(for geometry: CADFieldGeometry, device: any MTLDevice) async -> CADFieldResources? {
        let task: Task<CADFieldResources?, Never> = lock.withLock {
            if let resources = ready[device.registryID] { return Task { resources } }
            if let task = pending[device.registryID] { return task }
            let task = Task.detached(priority: .userInitiated) { [weak self] () -> CADFieldResources? in
                let resources = try? CADFieldResources(geometry: geometry, device: device)
                self?.finish(device.registryID, resources)
                return resources
            }
            pending[device.registryID] = task
            return task
        }
        return await task.value
    }

    private func finish(_ device: UInt64, _ resources: CADFieldResources?) {
        lock.withLock {
            pending[device] = nil
            if let resources { ready[device] = resources }
        }
    }
}

/// Lights of RAD's studio rig in the shading units of SceneKit's physically
/// based model: linear color times intensity / 1000, directions toward the
/// light in view space (the directional lights are children of the camera).
struct CADFieldLights: Sendable, Equatable {
    var ambient = SIMD3<Float>.zero
    var directions: [SIMD3<Float>] = []
    var colors: [SIMD3<Float>] = []

    /// Reads the rig: ambient lights anywhere, directional lights under `camera`.
    @MainActor
    static func rig(scene: SCNScene, camera: SCNNode) -> CADFieldLights {
        var lights = CADFieldLights()
        func linear(_ light: SCNLight) -> SIMD3<Float> {
            let color = (light.color as? NSColor) ?? .white
            let value = CADHighlightPalette.linear(color)
            return SIMD3(value.x, value.y, value.z) * Float(light.intensity) / 1000
        }
        scene.rootNode.enumerateHierarchy { node, _ in
            guard let light = node.light else { return }
            if light.type == .ambient { lights.ambient += linear(light); return }
            guard light.type == .directional, node.parent === camera, lights.directions.count < 2 else { return }
            let z = node.simdTransform.columns.2
            lights.directions.append(simd_normalize(SIMD3(z.x, z.y, z.z)))
            lights.colors.append(linear(light))
        }
        return lights
    }
}

/// One exact solid of a frame.
struct CADFieldInstance: Sendable, Equatable {
    var solid: Int
    var model: simd_float4x4
    var albedo: SIMD3<Float>
    var emission: SIMD3<Float>
}

/// The exact faces a frame shows. Without it, field parts draw their meshes.
struct CADFieldFrame: @unchecked Sendable {
    var resources: CADFieldResources
    var instances: [CADFieldInstance]
    var lights: CADFieldLights
    /// Every visible solid is exact (none draws its display mesh), so
    /// translucent transmittance derives from the accumulated alpha.
    var complete = false
    /// Shader diagnostics (0 off; 1 march steps, 2 normals, 3 sample-traced
    /// pixels, 4 no multisample tracing).
    var debug: Int = 0
}

/// One proxy draw of an exact solid, prepared once per exact frame geometry.
struct CADFieldDraw {
    var values: CADFieldResources.Instance
    var solid: Int
    var indexStart: Int
    var indexCount: Int
    /// Pixel rectangle (x0, y0, x1, y1 exclusive, top-left origin) holding the
    /// proxy's projection; nil when the part reaches behind the eye.
    var rect: SIMD4<Int32>?
}

/// Immutable GPU buffers and pipelines for one preview's field geometry.
final class CADFieldResources: @unchecked Sendable {
    /// Exact passes: stored geometry (opaque), translucent layers, and
    /// highlight outline coverage.
    enum Mode: Int, CaseIterable, Sendable {
        case geometry, visibility, accumulation, revealage, mask
    }

    static let metalness: Float = 0.28
    static let roughness: Float = 0.42
    static let sampleCount = 4
    /// Stored geometry formats: octahedral view-space normal, solid + 1, depth.
    static let normalFormat = MTLPixelFormat.rg16Snorm
    static let partFormat = MTLPixelFormat.r16Uint
    static let depthFormat = MTLPixelFormat.depth32Float

    let device: any MTLDevice
    let program: any MTLBuffer
    let positions: any MTLBuffer
    let indices: any MTLBuffer
    let parts: [Int: CADFieldGeometry.Part]
    let depthWrite: any MTLDepthStencilState
    let depthRead: any MTLDepthStencilState
    let depthIgnored: any MTLDepthStencilState
    /// Offsets of the multisample positions from the pixel center, in pixels.
    let sampleOffsets: [SIMD2<Float>]
    private let library: any MTLLibrary
    private let lock = NSLock()
    private var pipelines: [String: any MTLRenderPipelineState] = [:]

    init(geometry: CADFieldGeometry, device: any MTLDevice) throws {
        self.device = device
        func buffer(_ data: Data) throws -> any MTLBuffer {
            guard let buffer = data.withUnsafeBytes({ raw in
                raw.baseAddress.flatMap { device.makeBuffer(bytes: $0, length: max(raw.count, 4), options: .storageModeShared) }
            }) else { throw RADError.message("RAD could not allocate the design's exact surfaces.") }
            return buffer
        }
        guard let program = geometry.program.makeBuffer(device: device) else {
            throw RADError.message("RAD could not allocate the design's exact surfaces.")
        }
        self.program = program
        positions = try buffer(geometry.positions)
        indices = try buffer(geometry.indices)
        parts = geometry.parts
        library = try CADFieldPipelines.library(device: device, macros: geometry.program.shaderMacros)
        func depth(_ compare: MTLCompareFunction, write: Bool) throws -> any MTLDepthStencilState {
            let descriptor = MTLDepthStencilDescriptor()
            descriptor.depthCompareFunction = compare
            descriptor.isDepthWriteEnabled = write
            guard let state = device.makeDepthStencilState(descriptor: descriptor) else {
                throw RADError.message("RAD could not prepare the design's exact surfaces.")
            }
            return state
        }
        // Reverse-Z: nearer is greater.
        depthWrite = try depth(.greater, write: true)
        depthRead = try depth(.greater, write: false)
        depthIgnored = try depth(.always, write: false)
        sampleOffsets = device.getDefaultSamplePositions(sampleCount: Self.sampleCount).map { SIMD2(Float($0.x) - 0.5, Float($0.y) - 0.5) }
        // The exact passes' formats, compiled before the model is revealed.
        let samples = Self.sampleCount
        _ = try pipeline(.geometry, colors: [Self.normalFormat, Self.partFormat], depth: Self.depthFormat, samples: samples)
        _ = try pipeline(.visibility, colors: [], depth: Self.depthFormat, samples: samples)
        _ = try pipeline(.visibility, colors: [.r16Float], depth: Self.depthFormat, samples: samples)
        _ = try pipeline(.revealage, colors: [.r16Float], depth: Self.depthFormat, samples: samples)
        _ = try pipeline(.accumulation, colors: [.rgba16Float], depth: .invalid, samples: samples)
        _ = try pipeline(.mask, colors: [.rgba8Unorm], depth: .invalid, samples: 1)
    }

    /// The pipeline for a mode and pass formats, compiled on first use.
    func pipeline(_ mode: Mode, colors: [MTLPixelFormat], depth: MTLPixelFormat, samples: Int) throws -> any MTLRenderPipelineState {
        let key = "\(mode.rawValue)/\(colors.map { String($0.rawValue) }.joined(separator: ","))/\(depth.rawValue)/\(samples)"
        if let pipeline = lock.withLock({ pipelines[key] }) { return pipeline }
        let descriptor = MTLRenderPipelineDescriptor()
        descriptor.label = "RAD exact faces \(mode)"
        descriptor.vertexFunction = library.makeFunction(name: "cad_field_vertex")
        let fragment: String
        switch mode {
        case .geometry: fragment = "cad_field_geometry"
        case .visibility: fragment = "cad_field_visibility"
        case .accumulation: fragment = "cad_field_accumulate"
        case .revealage: fragment = "cad_field_revealage"
        case .mask: fragment = "cad_field_mask"
        }
        descriptor.fragmentFunction = library.makeFunction(name: fragment)
        guard descriptor.vertexFunction != nil, descriptor.fragmentFunction != nil else {
            throw RADError.message("RAD's exact surface shaders are incomplete.")
        }
        descriptor.rasterSampleCount = samples
        descriptor.depthAttachmentPixelFormat = depth
        for (index, format) in colors.enumerated() {
            let attachment: MTLRenderPipelineColorAttachmentDescriptor = descriptor.colorAttachments[index]
            attachment.pixelFormat = format
            switch mode {
            case .geometry: break
            case .visibility:
                // Shares a pass with the transmittance layers' target.
                attachment.writeMask = []
            case .accumulation:
                // Premultiplied color and alpha add, independent of order.
                attachment.isBlendingEnabled = true
                attachment.rgbBlendOperation = .add
                attachment.alphaBlendOperation = .add
                attachment.sourceRGBBlendFactor = .one
                attachment.destinationRGBBlendFactor = .one
                attachment.sourceAlphaBlendFactor = .one
                attachment.destinationAlphaBlendFactor = .one
            case .revealage:
                // Each crossing multiplies the remaining transmittance.
                attachment.isBlendingEnabled = true
                attachment.rgbBlendOperation = .add
                attachment.alphaBlendOperation = .add
                attachment.sourceRGBBlendFactor = .zero
                attachment.destinationRGBBlendFactor = .sourceColor
                attachment.sourceAlphaBlendFactor = .zero
                attachment.destinationAlphaBlendFactor = .sourceAlpha
            case .mask:
                attachment.isBlendingEnabled = true
                attachment.rgbBlendOperation = .max
                attachment.alphaBlendOperation = .max
                attachment.sourceRGBBlendFactor = .one
                attachment.destinationRGBBlendFactor = .one
                attachment.sourceAlphaBlendFactor = .one
                attachment.destinationAlphaBlendFactor = .one
            }
        }
        let pipeline = try device.makeRenderPipelineState(descriptor: descriptor)
        lock.withLock { pipelines[key] = pipeline }
        return pipeline
    }

    /// Per-pass constants (CADFieldUniforms in the shader source).
    struct Uniforms {
        var view: simd_float4x4
        var viewProjection: simd_float4x4
        var inverseView: simd_float4x4
        var ambient: SIMD4<Float>
        var lightDirection0: SIMD4<Float>
        var lightDirection1: SIMD4<Float>
        var lightColor0: SIMD4<Float>
        var lightColor1: SIMD4<Float>
        /// Footprint per unit view depth (perspective) or per pixel, orthographic flag, alpha, debug mode.
        var projection: SIMD4<Float>
        var samples01: SIMD4<Float>
        var samples23: SIMD4<Float>
        /// Maximum march steps per ray, hit tolerance as a fraction of the pixel footprint.
        var march = SIMD4<Float>(48, 0.05, 0, 0)

        init(view: simd_float4x4, projection: simd_float4x4, height: Int, lights: CADFieldLights,
             alpha: Float, debug: Int, sampleOffsets: [SIMD2<Float>]) {
            self.view = view
            viewProjection = projection * view
            inverseView = view.inverse
            ambient = SIMD4(lights.ambient, 0)
            let directions = lights.directions + Array(repeating: SIMD3<Float>(0, 0, 1), count: 2)
            let colors = lights.colors + Array(repeating: SIMD3<Float>.zero, count: 2)
            lightDirection0 = SIMD4(directions[0], 0)
            lightDirection1 = SIMD4(directions[1], 0)
            lightColor0 = SIMD4(colors[0], 0)
            lightColor1 = SIMD4(colors[1], 0)
            let orthographic = projection.columns.3.w == 1
            // Clip y spans 2 over the height; perspective divides by depth.
            let scale = 2 / (abs(projection.columns.1.y) * Float(max(height, 1)))
            self.projection = SIMD4(scale, orthographic ? 1 : 0, alpha, Float(debug))
            let offsets = sampleOffsets + Array(repeating: SIMD2<Float>.zero, count: 4)
            samples01 = SIMD4(lowHalf: offsets[0], highHalf: offsets[1])
            samples23 = SIMD4(lowHalf: offsets[2], highHalf: offsets[3])
        }
    }

    /// Per-draw constants (CADFieldInstance in the shader source).
    struct Instance {
        var worldFromPart: simd_float4x4
        var partFromWorld: simd_float4x4
        var albedo: SIMD4<Float>
        var emission: SIMD4<Float>
        var material: SIMD4<Float>
        var part: SIMD4<UInt32>
        var contacts = (SIMD4<UInt32>.zero, SIMD4<UInt32>.zero)
        var contactFromPart = (matrix_identity_float4x4, matrix_identity_float4x4, matrix_identity_float4x4,
                               matrix_identity_float4x4, matrix_identity_float4x4, matrix_identity_float4x4,
                               matrix_identity_float4x4, matrix_identity_float4x4)
        var contactBounds = (simd_float4x4(), simd_float4x4(), simd_float4x4(), simd_float4x4())

        init(_ instance: CADFieldInstance, shell: Float, contacts: [Contact] = []) {
            worldFromPart = instance.model
            partFromWorld = instance.model.inverse
            albedo = SIMD4(instance.albedo, 1)
            emission = SIMD4(instance.emission, 0)
            material = SIMD4(CADFieldResources.metalness, CADFieldResources.roughness, shell, 0)
            let used = contacts.prefix(CADFieldResources.maximumContacts)
            part = SIMD4(UInt32(instance.solid), UInt32(used.count), 0, 0)
            var indices = [UInt32](repeating: 0, count: 8)
            var matrices = [simd_float4x4](repeating: matrix_identity_float4x4, count: 8)
            var bounds = [SIMD4<Float>](repeating: .zero, count: 16)
            for (index, contact) in used.enumerated() {
                indices[index] = UInt32(contact.solid)
                matrices[index] = contact.model.inverse * instance.model
                bounds[2 * index] = SIMD4(contact.lower, 0)
                bounds[2 * index + 1] = SIMD4(contact.upper, 0)
            }
            contactBounds = (simd_float4x4(bounds[0], bounds[1], bounds[2], bounds[3]),
                             simd_float4x4(bounds[4], bounds[5], bounds[6], bounds[7]),
                             simd_float4x4(bounds[8], bounds[9], bounds[10], bounds[11]),
                             simd_float4x4(bounds[12], bounds[13], bounds[14], bounds[15]))
            self.contacts = (SIMD4(indices[0], indices[1], indices[2], indices[3]),
                             SIMD4(indices[4], indices[5], indices[6], indices[7]))
            contactFromPart = (matrices[0], matrices[1], matrices[2], matrices[3], matrices[4], matrices[5],
                               matrices[6], matrices[7])
        }
    }

    /// A larger part whose posed bounds overlap an instance's.
    struct Contact {
        var solid: Int
        var model: simd_float4x4
        var lower: SIMD3<Float>
        var upper: SIMD3<Float>
    }

    /// Parts whose posed bounds overlap, at most this many per part, have
    /// antialiased contact curves; more than `maximumContactParts` parts skip it.
    static let maximumContacts = 8
    static let maximumContactParts = 512

    /// For each instance, the larger instances (by bounds volume, then solid)
    /// whose posed bounds overlap its own, largest overlap first. One side of
    /// each contact suffices: the smaller part leaves the samples inside the
    /// larger to it, and depth decides the rest.
    func contacts(_ instances: [CADFieldInstance]) -> [[Contact]] {
        guard instances.count > 1, instances.count <= Self.maximumContactParts else {
            return Array(repeating: [], count: instances.count)
        }
        let boxes: [(SIMD3<Float>, SIMD3<Float>)?] = instances.map { instance in
            guard let part = parts[instance.solid] else { return nil }
            return Self.posedBounds(part.lower, part.upper, instance.model)
        }
        func volume(_ box: (SIMD3<Float>, SIMD3<Float>)) -> Float {
            let size = simd_max(box.1 - box.0, .zero)
            return size.x * size.y * size.z
        }
        return instances.indices.map { index in
            guard let box = boxes[index] else { return [] }
            var overlaps: [(volume: Float, contact: Contact)] = []
            for other in instances.indices where other != index {
                guard let candidate = boxes[other], let part = parts[instances[other].solid] else { continue }
                let larger = volume(candidate) > volume(box)
                    || (volume(candidate) == volume(box) && instances[other].solid > instances[index].solid)
                let size = simd_min(box.1, candidate.1) - simd_max(box.0, candidate.0)
                guard larger, size.x >= 0, size.y >= 0, size.z >= 0 else { continue }
                overlaps.append((size.x * size.y * size.z,
                                 Contact(solid: instances[other].solid, model: instances[other].model,
                                         lower: part.lower, upper: part.upper)))
            }
            overlaps.sort { $0.volume > $1.volume }
            return overlaps.prefix(Self.maximumContacts).map(\.contact)
        }
    }

    static func posedBounds(_ lower: SIMD3<Float>, _ upper: SIMD3<Float>, _ model: simd_float4x4)
        -> (SIMD3<Float>, SIMD3<Float>) {
        var low = SIMD3<Float>(repeating: .infinity), high = SIMD3<Float>(repeating: -.infinity)
        for corner in 0..<8 {
            let local = SIMD3(corner & 1 == 0 ? lower.x : upper.x, corner & 2 == 0 ? lower.y : upper.y,
                              corner & 4 == 0 ? lower.z : upper.z)
            let world = model * SIMD4(local, 1)
            low = simd_min(low, SIMD3(world.x, world.y, world.z))
            high = simd_max(high, SIMD3(world.x, world.y, world.z))
        }
        return (low, high)
    }

    /// The proxy draws of `instances`, nearest posed bounds center first,
    /// with their contact parts when `contacts` is set and the pixel
    /// rectangles their proxies can reach in a `width`×`height` frame.
    func draws(_ instances: [CADFieldInstance], view: simd_float4x4, projection: simd_float4x4,
               width: Int, height: Int, contacts: Bool) -> [CADFieldDraw] {
        let neighbors = contacts ? self.contacts(instances) : Array(repeating: [], count: instances.count)
        let viewProjection = projection * view
        var ordered: [(depth: Float, draw: CADFieldDraw)] = []
        ordered.reserveCapacity(instances.count)
        for (instance, neighbors) in zip(instances, neighbors) {
            guard let part = parts[instance.solid], part.indexCount > 0 else { continue }
            let center = view * instance.model * SIMD4(part.center, 1)
            // The proxy lies within the shell of the surface, and the surface
            // within the display mesh's deviation of the bounds.
            let margin = SIMD3<Float>(repeating: part.shell + part.deviation)
            let rect = Self.pixelRect(lower: part.lower - margin, upper: part.upper + margin, model: instance.model,
                                      viewProjection: viewProjection, width: width, height: height)
            ordered.append((center.z, CADFieldDraw(values: Instance(instance, shell: part.shell, contacts: neighbors),
                                                   solid: instance.solid, indexStart: part.indexStart,
                                                   indexCount: part.indexCount, rect: rect)))
        }
        // View space looks down -z: the nearest center has the largest z.
        ordered.sort { $0.depth > $1.depth }
        return ordered.map(\.draw)
    }

    /// The pixel rectangle of posed bounds, padded by two pixels for
    /// multisample rays; nil when a corner lies at or behind the eye plane.
    static func pixelRect(lower: SIMD3<Float>, upper: SIMD3<Float>, model: simd_float4x4,
                          viewProjection: simd_float4x4, width: Int, height: Int) -> SIMD4<Int32>? {
        var low = SIMD2<Float>(repeating: .infinity), high = SIMD2<Float>(repeating: -.infinity)
        let transform = viewProjection * model
        for corner in 0..<8 {
            let local = SIMD3(corner & 1 == 0 ? lower.x : upper.x, corner & 2 == 0 ? lower.y : upper.y,
                              corner & 4 == 0 ? lower.z : upper.z)
            let clip = transform * SIMD4(local, 1)
            guard clip.w > 1e-6, clip.x.isFinite, clip.y.isFinite else { return nil }
            let ndc = SIMD2(clip.x, clip.y) / clip.w
            low = simd_min(low, ndc)
            high = simd_max(high, ndc)
        }
        let x0 = (low.x + 1) / 2 * Float(width) - 2, x1 = (high.x + 1) / 2 * Float(width) + 2
        let y0 = (1 - high.y) / 2 * Float(height) - 2, y1 = (1 - low.y) / 2 * Float(height) + 2
        func clamped(_ value: Float, _ limit: Int) -> Int32 { Int32(max(0, min(Float(limit), value.rounded(.down)))) }
        let rect = SIMD4(clamped(x0, width), clamped(y0, height), clamped(x1.rounded(.up), width), clamped(y1.rounded(.up), height))
        return rect
    }

    /// Viewports and scissor rectangles per draw: each instance of a draw
    /// uses the pair of its instance index.
    static let maximumViewports = 16

    /// Encodes the draws with the bound pipeline and per-pass uniforms. With
    /// `rects`, each draw meeting some of them is instanced over groups of up
    /// to 16 rectangles, each instance clipped to its rectangle's scissor and
    /// placed by `viewport` (the frame's viewport for stored layers, an
    /// offset one for targets that hold the rectangle elsewhere); `scissor`
    /// maps a rectangle to the target. Without, every draw covers the pass once.
    func encode(_ draws: [CADFieldDraw], rects tiles: [SIMD4<Int32>]?, viewport: (SIMD4<Int32>, Int) -> MTLViewport,
                scissor: (SIMD4<Int32>, Int) -> MTLScissorRect, encoder: any MTLRenderCommandEncoder) {
        encoder.setVertexBuffer(positions, offset: 0, index: 0)
        encoder.setFragmentBuffer(program, offset: 0, index: 3)
        func meets(_ draw: CADFieldDraw, _ rect: SIMD4<Int32>) -> Bool {
            guard let bounds = draw.rect else { return true }
            return bounds.x < rect.z && bounds.z > rect.x && bounds.y < rect.w && bounds.w > rect.y
        }
        func encode(_ draw: CADFieldDraw, instances: Int) {
            var values = draw.values
            encoder.setVertexBytes(&values, length: MemoryLayout<Instance>.stride, index: 2)
            encoder.setFragmentBytes(&values, length: MemoryLayout<Instance>.stride, index: 2)
            encoder.drawIndexedPrimitives(type: .triangle, indexCount: draw.indexCount, indexType: .uint32,
                                          indexBuffer: indices, indexBufferOffset: draw.indexStart * 4,
                                          instanceCount: instances)
        }
        guard let tiles else {
            for draw in draws { encode(draw, instances: 1) }
            return
        }
        var start = 0
        while start < tiles.count {
            let group = Array(tiles[start..<min(tiles.count, start + Self.maximumViewports)])
            encoder.setViewports(group.enumerated().map { viewport($0.element, start + $0.offset) })
            encoder.setScissorRects(group.enumerated().map { scissor($0.element, start + $0.offset) })
            for draw in draws where group.contains(where: { meets(draw, $0) }) {
                encode(draw, instances: group.count)
            }
            start += group.count
        }
    }
}

/// Field shader libraries per device and program macros. Programs with the
/// same register files and opcodes share one compiled library.
enum CADFieldPipelines {
    private static let lock = NSLock()
    private static var libraries: [String: any MTLLibrary] = [:]

    /// Exact silhouettes and contact curves are antialiased by tracing the
    /// multisample rays; builds with `-Xswiftc -DRAD_ANALYTIC_EXACT_AA` derive
    /// each sample's coverage analytically from ray 0 instead.
    #if RAD_ANALYTIC_EXACT_AA
    static let analyticAntialiasing = true
    #else
    static let analyticAntialiasing = false
    #endif

    static func library(device: any MTLDevice, macros programMacros: [String: NSNumber]) throws -> any MTLLibrary {
        var macros = programMacros
        macros["RAD_EXACT_ANALYTIC_AA"] = NSNumber(value: analyticAntialiasing ? 1 : 0)
        let key = "\(device.registryID)|" + macros.keys.sorted().map { "\($0)=\(macros[$0]!)" }.joined(separator: ",")
        if let library = lock.withLock({ libraries[key] }) { return library }
        let options = MTLCompileOptions()
        options.preprocessorMacros = macros
        if #available(macOS 15.0, *) { options.mathMode = .safe } else { options.fastMathEnabled = false }
        let library = try device.makeLibrary(source: FieldProgramShaders.functions + CADFieldShading.source + shaderSource,
                                             options: options)
        lock.withLock { libraries[key] = library }
        return library
    }

    static let shaderSource = #"""

// ---------------------------------------------------------------- RAD exact faces
// Each part's proxy (a closed mesh outside the solid within the shell
// 0 < f <= shell) is rasterized; each fragment marches its view ray in part
// coordinates to the field's crossing: front faces forward to the entering
// crossing, back faces backward to the exiting one. Rays that leave the
// shell miss. Pixels near silhouettes trace their multisample rays too and
// cover only the samples whose rays hit. Opaque crossings are stored per
// sample (normal, part, depth) and shaded by the viewport's composite.

struct CADFieldUniforms {
    float4x4 view;
    float4x4 viewProjection;
    float4x4 inverseView;
    float4 ambient;
    float4 lightDirection0;
    float4 lightDirection1;
    float4 lightColor0;
    float4 lightColor1;
    float4 projection;   // footprint scale, orthographic, alpha, debug
    float4 samples01;    // sample offsets from the pixel center, in pixels
    float4 samples23;
    float4 march;        // maximum steps per ray, hit tolerance (fraction of the footprint)
};

struct CADFieldInstance {
    float4x4 worldFromPart;
    float4x4 partFromWorld;
    float4 albedo;
    float4 emission;
    float4 material;     // metalness, roughness, shell
    uint4 part;          // field part, contact parts
    uint4 contacts[2];   // larger parts whose bounds overlap this one's
    float4x4 contactFromPart[8];
    float4 contactBounds[16]; // their bounds (lower, upper) in their coordinates
};

struct CADFieldVarying {
    float4 position [[position]];
    float3 partPosition;
    uint viewport [[viewport_array_index]];
};

// Slices draw each instance into its own viewport and scissor rectangle
// (one per tile); viewports place a tile in its target without changing
// clip coordinates.
vertex CADFieldVarying cad_field_vertex(const device packed_float3 *positions [[buffer(0)]],
                                        constant CADFieldUniforms &u [[buffer(1)]],
                                        constant CADFieldInstance &instance [[buffer(2)]],
                                        uint vid [[vertex_id]], uint instanceIndex [[instance_id]]) {
    const float3 p = float3(positions[vid]);
    CADFieldVarying out;
    out.position = u.viewProjection * (instance.worldFromPart * float4(p, 1.0f));
    out.partPosition = p;
    out.viewport = instanceIndex;
    return out;
}

#define CAD_FIELD_NONE 8u

// The rays of one fragment: 0 through the proxy point, 1 to 4 through the
// multisample positions at the same view depth.
struct CADFieldRays {
    float3 viewPoint;
    float4x4 partFromView;
    float footprintScale;
    bool orthographic;
    float sense;          // +1 front faces (entering crossing), -1 back faces (exiting)
};

struct CADFieldResult {
    uint mask;            // rays that hit
    float closest;        // ray 0's closest approach while outside, in footprints
    float3 position;      // the shading ray's crossing (part coordinates)
    float3 normal;        // its unit outward normal (zero until computed)
    float footprint;      // its footprint at the crossing
    uint shaded;          // the shading ray (the first that hit), or CAD_FIELD_NONE
    uint steps;           // field samples taken
    float safe;           // ray 0's samples prove no surface within half a pixel of it up to here
    float value;          // the field at the shading ray's crossing
    float3 direction;     // the shading ray's marched direction
    float contact;        // ray 0's crossing: the nearest contact part's field, in footprints
};

inline void cad_field_ray(thread const CADFieldRays &r, constant CADFieldUniforms &u, uint i,
                          thread float3 &origin, thread float3 &direction, thread float &footprint, thread float &slope) {
    float3 v = r.viewPoint;
    const float scale = r.orthographic ? r.footprintScale : r.footprintScale * max(-v.z, 1e-6f);
    if (i > 0) {
        const float2 offset = i == 1 ? u.samples01.xy : i == 2 ? u.samples01.zw : i == 3 ? u.samples23.xy : u.samples23.zw;
        v.xy += float2(offset.x, -offset.y) * scale;  // pixel y runs down, view y up
    }
    const float3 forward = r.orthographic ? float3(0.0f, 0.0f, -1.0f) : normalize(v);
    origin = (r.partFromView * float4(v, 1.0f)).xyz;
    direction = normalize((r.partFromView * float4(forward * r.sense, 0.0f)).xyz);
    footprint = scale;
    // Footprint change per unit of t along the marched direction.
    slope = r.orthographic ? 0.0f : r.footprintScale * (-forward.z) * r.sense;
}

// Marches rays first..last with one call of the interpreter for every
// sample (march, bisection, and the shading ray's normal unless the
// program's gradient pass computes it afterwards). Over-relaxed sphere
// tracing by |f| / L, stepping back whenever the unbounding spheres of two
// samples do not overlap (Keinert et al. 2014), so no surface is skipped; a
// sign change is bisected; |f| below the tolerance hits. A ray rising above
// the shell has left the proxy's interior and misses. A ray that runs out of
// steps within half a pixel of the surface (a grazing one) hits there.
#define CAD_FIELD_RELAXATION 1.5f
// With `contacts`, every crossing is then sampled in the fields of the parts
// whose bounds overlap this one (at the same call): a multisample crossing
// inside another part belongs to that part, which is nearer there.
void cad_field_march(device const uint *w, thread const RADFieldGrid &own, thread RADFieldCell &cell,
                     thread const CADFieldRays &rays, constant CADFieldUniforms &u,
                     constant CADFieldInstance &instance, bool contacts,
                     uint first, uint last, thread CADFieldResult &r) {
    const float shell = instance.material.z;
    const uint contactCount = contacts ? min(instance.part.y, 8u) : 0u;
    RADFieldGrid g = own;
    RADFieldCell own_cell = cell;
    uint candidate = 0;
    bool occluded = false;
    float hit_footprint = 1;
    uint ray = first;
    float3 o, d;
    float footprint0, slope;
    cad_field_ray(rays, u, ray, o, d, footprint0, slope);
    // Multisample rays refine coverage only: half the budget.
    uint budget = first == 0 ? uint(u.march.x) : uint(u.march.x) / 2;
    float t = 0, previous_t = 0, previous = 0, previous_radius = 0, step = 0, omega = CAD_FIELD_RELAXATION;
    float a = 0, b = 0, fa = 0, m = 0, eps = 0;
    uint phase = 0, k = 0, bisections = 0, ray_steps = 0;
    bool safe = ray == 0;
    if (ray > 0) { t = r.safe; previous_t = t; }
    float3 x = o + t * d, hit_position = o, gradient = 0;
    for (uint guard = 0; guard < 1024; ++guard) {
        RADFieldSample s = rad_field_sample_cached(w, g, cell, x);
        ++r.steps;
        bool hit = false, done = false, crossed = false;
        if (phase == 3) {
            // A contact part's field at this ray's crossing.
            if (s.value < 0) occluded = true;
            if (ray == 0) r.contact = min(r.contact, s.value / hit_footprint);
            ++candidate;
            crossed = true;
        } else if (phase == 2) {
            gradient += rad_field_stencil[k] * s.value;
            if (++k < 4) { x = hit_position + eps * rad_field_stencil[k]; continue; }
            const float len = length(gradient);
            r.normal = len > 0 ? gradient / len : float3(0);
            crossed = true;
        } else if (phase == 0) {
            const float footprint = max(footprint0 + slope * t, 1e-7f);
            const float tolerance = max(u.march.y * footprint, 4e-7f * (length(x) + 1.0f) * s.lipschitz);
            if (ray == 0 && s.value >= 0) r.closest = min(r.closest, s.value / footprint);
            ++ray_steps;
            if (fabs(s.value) < tolerance) {
                hit = true;
            } else if (s.value < 0) {
                if (ray_steps == 1) {
                    hit = true;  // starts inside: a proxy point within float rounding of the surface
                } else {
                    a = previous_t; b = t; fa = previous; phase = 1;
                    m = 0.5f * (a + b);
                    x = o + m * d;
                    continue;
                }
            } else {
                float radius = s.value / s.lipschitz;
                if (omega > 1 && ray_steps > 1 && radius + previous_radius < step) {
                    // The relaxed step may have passed a surface: step back.
                    omega = 1;
                    t = previous_t;
                    radius = previous_radius;
                } else if (ray_steps >= budget) {
                    hit = s.value < 0.5f * footprint;
                    done = !hit;
                } else if (s.value > shell && ray_steps > 1 && s.value >= previous) {
                    done = true;  // rising above the shell: left the proxy's interior
                } else {
                    // The empty spheres of ray 0's samples contain the tube
                    // of half a pixel around it (where the multisample rays
                    // run) while each sphere is wider than the tube and
                    // consecutive spheres' tube cross-sections meet.
                    if (safe) {
                        const float tube = 0.5f * footprint;
                        safe = radius > tube && (ray_steps == 1 ||
                            sqrt(max(previous_radius * previous_radius - tube * tube, 0.0f)) +
                            sqrt(radius * radius - tube * tube) >= t - previous_t);
                        if (safe) r.safe = t;
                    }
                    previous_t = t;
                    previous = s.value;
                    previous_radius = radius;
                }
                if (!hit && !done) {
                    step = omega * radius;
                    t += step;
                    x = o + t * d;
                    continue;
                }
            }
        } else {  // bisection sample at m
            if (fabs(s.value) < max(u.march.y * max(footprint0 + slope * m, 1e-7f), 1e-6f) || ++bisections >= 24) {
                t = m;
                hit = true;
            } else {
                if ((s.value < 0) == (fa < 0)) { a = m; fa = s.value; } else { b = m; }
                m = 0.5f * (a + b);
                x = o + m * d;
                continue;
            }
        }
        if (hit) {
            hit_position = o + t * d;
            hit_footprint = max(footprint0 + slope * t, 1e-7f);
            candidate = 0;
            occluded = false;
            if (r.shaded == CAD_FIELD_NONE) {
                r.shaded = ray;
                r.position = hit_position;
                r.value = s.value;
                r.direction = d;
                r.footprint = max(footprint0 + slope * t, 1e-7f);
#if !RAD_FIELD_GRADIENT_PASS
                eps = max(0.25f * r.footprint, 2e-5f * (length(hit_position) + 1.0f));
                phase = 2; k = 0; gradient = 0;
                x = hit_position + eps * rad_field_stencil[0];
                continue;
#endif
            }
            crossed = true;
        }
        if (crossed) {
            // The next contact part whose box holds the crossing.
            g = own;
            while (candidate < contactCount) {
                const float3 p = (instance.contactFromPart[candidate] * float4(hit_position, 1.0f)).xyz;
                const float reach = 1.5f * hit_footprint;
                if (all(p >= instance.contactBounds[2 * candidate].xyz - reach) &&
                    all(p <= instance.contactBounds[2 * candidate + 1].xyz + reach)) {
                    if (phase != 3) own_cell = cell;
                    g = rad_field_grid(w, instance.contacts[candidate / 4][candidate % 4]);
                    cell = rad_field_cell(w, g, p);
                    x = p;
                    break;
                }
                ++candidate;
            }
            if (candidate < contactCount) { phase = 3; continue; }
            if (phase == 3) cell = own_cell;
            phase = 0;
            if (!occluded || ray == 0) r.mask |= 1u << ray;
            done = true;
        }
        if (done) {
            if (++ray > last) break;
            cad_field_ray(rays, u, ray, o, d, footprint0, slope);
            // Multisample rays start where ray 0 proved their way clear.
            t = r.safe; previous_t = t; previous = 0; previous_radius = 0; step = 0; omega = CAD_FIELD_RELAXATION;
            phase = 0; bisections = 0; ray_steps = 0; safe = false;
            x = o + t * d;
        }
    }
}

// The field crossing of a proxy fragment, with multisample coverage near
// silhouettes. `traced` reports whether the sample rays were traced.
CADFieldResult cad_field_surface(CADFieldVarying in, constant CADFieldUniforms &u, constant CADFieldInstance &instance,
                                 device const uint *w, float sense, bool antialias, thread bool &traced,
                                 bool contacts = false) {
    CADFieldRays rays;
    const float4 world = instance.worldFromPart * float4(in.partPosition, 1.0f);
    rays.viewPoint = (u.view * world).xyz;
    rays.partFromView = instance.partFromWorld * u.inverseView;
    rays.footprintScale = u.projection.x;
    rays.orthographic = u.projection.y > 0.5f;
    rays.sense = sense;
    CADFieldResult r;
    r.mask = 0; r.closest = 3.0e38f; r.position = in.partPosition; r.normal = 0; r.footprint = 1;
    r.shaded = CAD_FIELD_NONE; r.steps = 0; r.safe = 0; r.value = 0; r.direction = 0; r.contact = 3.0e38f;
    const RADFieldGrid g = rad_field_grid(w, instance.part.x);
    RADFieldCell cell = rad_field_cell(w, g, in.partPosition);
    traced = false;
    for (uint stage = 0; stage < 2; ++stage) {
        if (stage == 1) {
            // Every lane reaches this point, so quad derivatives are defined.
            // Distance to the silhouette in pixels: n.v over its change per pixel.
            float facing = 0;
            if (r.mask & 1u) {
                const float3 n = normalize((u.view * (instance.worldFromPart * float4(r.normal, 0.0f))).xyz);
                const float3 v = rays.orthographic ? float3(0, 0, 1) : normalize(-rays.viewPoint);
                facing = fabs(dot(n, v));
            }
            const float change = fwidth(facing);
            const bool hit = (r.mask & 1u) != 0;
            const int debugMode = int(u.projection.w);
            // Samples lie within 0.4 pixel of the center: trace them only
            // where the silhouette may pass within about a pixel.
            // Contact curves with other parts likewise.
            traced = antialias && debugMode != 4 && (hit ? facing < change || fabs(r.contact) < 1.0f : r.closest < 0.75f);
            if (!traced) break;
        }
        cad_field_march(w, g, cell, rays, u, instance, contacts && antialias, stage == 0 ? 0u : 1u,
                        stage == 0 ? 0u : 4u, r);
#if RAD_FIELD_GRADIENT_PASS
        if (r.shaded != CAD_FIELD_NONE && all(r.normal == 0)) {
            const float eps = max(0.25f * r.footprint, 2e-5f * (length(r.position) + 1.0f));
            r.normal = rad_field_gradient_cached(w, g, cell, r.position, eps);
        }
#endif
    }
    if (!traced) r.mask = (r.mask & 1u) ? 0xFu : 0u;
    else r.mask >>= 1;
    // The march stops within its tolerance on the near side: one Newton step
    // along the ray puts depth on the surface, where crease strokes lie.
    if (r.shaded != CAD_FIELD_NONE) {
        const float approach = -dot(r.normal, r.direction);
        if (approach > 0.05f) r.position += r.direction * clamp(r.value / approach, -r.footprint, r.footprint);
    }
    return r;
}

// Translucent layers shade where they are marched; roughness widens by the
// normal's variation across the fragment quad.
static float3 cad_shade(float3 n, float3 v, constant CADFieldInstance &instance, constant CADFieldUniforms &u) {
    const float3 dx = dfdx(n), dy = dfdy(n);
    const float variance = 0.25f * (dot(dx, dx) + dot(dy, dy));
    return cad_shade_lit(n, v, instance.albedo.rgb, instance.emission.rgb, instance.material.x, instance.material.y,
                         variance, u.ambient.rgb, u.lightDirection0.xyz, u.lightColor0.rgb,
                         u.lightDirection1.xyz, u.lightColor1.rgb);
}

static float3 cad_field_color(thread const CADFieldResult &r, bool front, constant CADFieldInstance &instance,
                              constant CADFieldUniforms &u, bool traced) {
    const float4 world = instance.worldFromPart * float4(r.position, 1.0f);
    const float3 viewPosition = (u.view * world).xyz;
    float3 n = normalize((u.view * (instance.worldFromPart * float4(r.normal, 0.0f))).xyz);
    // Double-sided translucent layers face the eye.
    if (!front) n = -n;
    const float3 v = u.projection.y > 0.5f ? float3(0, 0, 1) : normalize(-viewPosition);
    const int debug = int(u.projection.w);
    if (debug == 1) return float3(saturate(float(r.steps) / 32.0f));
    if (debug == 2) return n * 0.5f + 0.5f;
    if (debug == 3) return traced ? float3(1, 0, 0) : float3(0.2f);
    return cad_shade(n, v, instance, u);
}

// One depth serves every sample of the pixel: the farthest of them, by the
// proxy's depth change across half a pixel, so decorations lying on the
// surface (crease strokes) pass at every sample as over a display mesh.
static float cad_field_depth(thread const CADFieldResult &r, constant CADFieldInstance &instance,
                             constant CADFieldUniforms &u, float rasterized, float halfPixel) {
    const float4 clip = u.viewProjection * (instance.worldFromPart * float4(r.position, 1.0f));
    // Reverse-Z: the crossing lies behind the proxy, never nearer.
    return min(clip.z / clip.w, rasterized) - halfPixel;
}

struct CADFieldGeometryOut {
    float2 normal [[color(0)]];   // octahedral unit view-space normal (debug 1, 3: a scalar in x)
    ushort part [[color(1)]];     // solid index + 1
    float depth [[depth(less)]];
    uint mask [[sample_mask]];
};

// Opaque crossings, stored per sample for the composite to shade.
fragment CADFieldGeometryOut cad_field_geometry(CADFieldVarying in [[stage_in]],
                                               constant CADFieldUniforms &u [[buffer(1)]],
                                               constant CADFieldInstance &instance [[buffer(2)]],
                                               device const uint *w [[buffer(3)]]) {
    bool traced;
    const CADFieldResult r = cad_field_surface(in, u, instance, w, 1.0f, true, traced, true);
    const float halfPixel = 0.5f * (fabs(dfdx(in.position.z)) + fabs(dfdy(in.position.z)));
    if (r.mask == 0 || r.shaded == CAD_FIELD_NONE) discard_fragment();
    CADFieldGeometryOut out;
    const int debug = int(u.projection.w);
    if (debug == 1) {
        out.normal = float2(saturate(float(r.steps) / 32.0f), 0.0f);
    } else if (debug == 3) {
        out.normal = float2(traced ? 1.0f : 0.0f, 0.0f);
    } else {
        const float3 n = (u.view * (instance.worldFromPart * float4(r.normal, 0.0f))).xyz;
        const float len = length(n);
        out.normal = cad_octahedral_encode(len > 0.0f ? n / len : float3(0.0f, 0.0f, 1.0f));
    }
    out.part = ushort(instance.part.x + 1u);
    out.depth = cad_field_depth(r, instance, u, in.position.z, halfPixel);
    out.mask = r.mask;
    return out;
}

struct CADFieldDepthOut {
    float depth [[depth(less)]];
    uint mask [[sample_mask]];
};

// Translucent mode's solid visibility depth: nearest entering crossings.
fragment CADFieldDepthOut cad_field_visibility(CADFieldVarying in [[stage_in]],
                                               constant CADFieldUniforms &u [[buffer(1)]],
                                               constant CADFieldInstance &instance [[buffer(2)]],
                                               device const uint *w [[buffer(3)]]) {
    bool traced;
    const CADFieldResult r = cad_field_surface(in, u, instance, w, 1.0f, false, traced);
    const float halfPixel = 0.5f * (fabs(dfdx(in.position.z)) + fabs(dfdy(in.position.z)));
    if (r.mask == 0 || r.shaded == CAD_FIELD_NONE) discard_fragment();
    CADFieldDepthOut out;
    out.depth = cad_field_depth(r, instance, u, in.position.z, halfPixel);
    out.mask = r.mask;
    return out;
}

struct CADFieldLayerOut {
    float4 color [[color(0)]];
    uint mask [[sample_mask]];
};

// Every proxy fragment owns one crossing: front faces the entering one,
// back faces the exiting one. Premultiplied shaded color adds.
fragment CADFieldLayerOut cad_field_accumulate(CADFieldVarying in [[stage_in]], bool front [[front_facing]],
                                               constant CADFieldUniforms &u [[buffer(1)]],
                                               constant CADFieldInstance &instance [[buffer(2)]],
                                               device const uint *w [[buffer(3)]]) {
    bool traced;
    const CADFieldResult r = cad_field_surface(in, u, instance, w, front ? 1.0f : -1.0f, true, traced);
    if (r.mask == 0 || r.shaded == CAD_FIELD_NONE) discard_fragment();
    CADFieldLayerOut out;
    out.color = float4(cad_field_color(r, front, instance, u, traced), 1.0f) * u.projection.z;
    out.mask = r.mask;
    return out;
}

// Each crossing multiplies the transmittance by 1 - alpha.
fragment CADFieldLayerOut cad_field_revealage(CADFieldVarying in [[stage_in]], bool front [[front_facing]],
                                              constant CADFieldUniforms &u [[buffer(1)]],
                                              constant CADFieldInstance &instance [[buffer(2)]],
                                              device const uint *w [[buffer(3)]]) {
    bool traced;
    const CADFieldResult r = cad_field_surface(in, u, instance, w, front ? 1.0f : -1.0f, true, traced);
    if (r.mask == 0 || r.shaded == CAD_FIELD_NONE) discard_fragment();
    CADFieldLayerOut out;
    out.color = float4(1.0f - u.projection.z);
    out.mask = r.mask;
    return out;
}

// Highlight outline coverage of a part's exact projection.
fragment float4 cad_field_mask(CADFieldVarying in [[stage_in]],
                               constant CADFieldUniforms &u [[buffer(1)]],
                               constant CADFieldInstance &instance [[buffer(2)]],
                               device const uint *w [[buffer(3)]],
                               constant float4 &channel [[buffer(4)]]) {
    bool traced;
    const CADFieldResult r = cad_field_surface(in, u, instance, w, 1.0f, false, traced);
    if (r.mask == 0) discard_fragment();
    return channel;
}
"""#
}

/// SceneKit's physically based shading for RAD's material and light rig,
/// shared by the translucent layers and the opaque composite: Lambert
/// diffuse, GGX distribution, height-correlated Smith visibility, Schlick
/// Fresnel, ambient times albedo, roughness widened by `variance`, the
/// normal's screen-space variation.
enum CADFieldShading {
    static let source = #"""

// ---------------------------------------------------------------- RAD exact shading
static float cad_ggx(float alpha, float NoH) {
    const float a2 = alpha * alpha;
    const float d = (NoH * a2 - NoH) * NoH + 1.0f;
    return a2 / (d * d * M_PI_F);
}
static float3 cad_schlick(float3 F0, float LoH) {
    const float x = 1.0f - LoH, x2 = x * x;
    return F0 + (1.0f - F0) * (x2 * x2 * x);
}
static float cad_smith(float alpha, float NoL, float NoV) {
    const float a2 = alpha * alpha;
    const float l = NoV * sqrt(NoL * NoL + a2 * (1.0f - NoL * NoL));
    const float v = NoL * sqrt(NoV * NoV + a2 * (1.0f - NoV * NoV));
    return 0.5f / (l + v + 1e-6f);
}
static float3 cad_shade_lit(float3 n, float3 v, float3 albedo, float3 emission, float metalness, float roughness0,
                            float variance, float3 ambient, float3 l0, float3 c0, float3 l1, float3 c1) {
    const float roughness = clamp(roughness0, 0.01f, 1.0f);
    const float alpha0 = roughness * roughness;
    const float alpha = sqrt(saturate(alpha0 * alpha0 + min(variance, 0.18f)));
    const float NoV = fabs(dot(n, v));
    const float3 F0 = mix(float3(0.04f), albedo, metalness);
    float3 diffuse = 0, specular = 0;
    for (int i = 0; i < 2; ++i) {
        const float3 l = i == 0 ? l0 : l1;
        const float3 c = i == 0 ? c0 : c1;
        const float3 h = normalize(l + v);
        const float NoL = saturate(dot(n, l));
        const float NoH = saturate(dot(n, h));
        const float LoH = saturate(dot(l, h));
        diffuse += c * NoL;
        specular += c * cad_schlick(F0, LoH) * (NoL * cad_ggx(alpha, NoH) * cad_smith(alpha, NoL, NoV) * M_PI_F);
    }
    return ambient * albedo + diffuse * albedo * (1.0f - metalness) + specular + emission;
}

// Unit normals stored as two signed values (octahedral mapping).
static float2 cad_octahedral_encode(float3 n) {
    n /= max(fabs(n.x) + fabs(n.y) + fabs(n.z), 1e-20f);
    if (n.z < 0.0f) {
        const float2 s = select(float2(-1.0f), float2(1.0f), n.xy >= 0.0f);
        return (1.0f - fabs(n.yx)) * s;
    }
    return n.xy;
}
static float3 cad_octahedral_decode(float2 e) {
    float3 n = float3(e, 1.0f - fabs(e.x) - fabs(e.y));
    const float t = saturate(-n.z);
    n.xy += select(float2(t), float2(-t), n.xy >= 0.0f);
    return normalize(n);
}
"""#
}

/// Shared between a scene source's field nodes and the passes encoding a
/// frame: the stored exact layers SceneKit's scene passes composite, set
/// only while SceneKit renders the frame.
final class CADFieldPassState: @unchecked Sendable {
    private let lock = NSLock()
    private var composite: CADExactComposite?
    private var mismatched = false

    func begin(_ composite: CADExactComposite?) { lock.withLock { self.composite = composite; mismatched = false } }
    /// Returns whether every pass drew the stored layers' own view.
    @discardableResult
    func end() -> Bool { lock.withLock { composite = nil; return !mismatched } }
    var current: CADExactComposite? { lock.withLock { composite } }
    func reportMismatch() { lock.withLock { mismatched = true } }
}

/// A geometry-free node's renderer delegate that composites the frame's
/// stored exact layers into SceneKit's current scene pass, drawn before the
/// pass's own geometry: shaded opaque samples with their depth, translucent
/// visibility depth, or transmittance. One node per pass; it keeps no
/// mutable state of its own, so a capture's copy attaches a fresh delegate
/// with its own state.
final class CADFieldPassDelegate: NSObject, SCNNodeRendererDelegate, @unchecked Sendable {
    enum Pass: Sendable { case opaque, visibility, accumulation, revealage }

    let pass: Pass
    let state: CADFieldPassState

    init(pass: Pass, state: CADFieldPassState) {
        self.pass = pass
        self.state = state
    }

    func renderNode(_ node: SCNNode, renderer: SCNRenderer, arguments: [String: Any]) {
        guard let composite = state.current, pass != .accumulation,
              let encoder = renderer.currentRenderCommandEncoder,
              let view = (arguments[SCNViewTransform] as? NSValue)?.scnMatrix4Value,
              let projection = (arguments[SCNProjectionTransform] as? NSValue)?.scnMatrix4Value else { return }
        let descriptor = renderer.currentRenderPassDescriptor
        guard let color = descriptor.colorAttachments[0].texture else { return }
        let viewMatrix = simd_float4x4(view), projectionMatrix = simd_float4x4(projection)
        // The stored layers belong to one view; a pass of another (a camera
        // committed after the geometry was drawn) must not show them.
        guard CADExactKey.sameView(viewMatrix, projectionMatrix, composite.view, composite.projection) else {
            state.reportMismatch()
            return
        }
        encoder.pushDebugGroup("RAD exact composite")
        composite.encode(pass, encoder: encoder, colorFormat: color.pixelFormat,
                         depthFormat: descriptor.depthAttachment.texture?.pixelFormat ?? .invalid,
                         samples: color.sampleCount, projection: projectionMatrix)
        encoder.popDebugGroup()
    }
}
