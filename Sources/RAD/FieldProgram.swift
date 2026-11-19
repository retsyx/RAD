import Darwin
import Foundation
import Metal

/// A build's GPU field program: the worker's `field.bin` (SDF-DESIGN.md, "GPU
/// field programs"), validated so that the Metal evaluator only follows
/// offsets inside the buffer and every loop ends. The words are uploaded as
/// one buffer; the shader functions in `FieldProgramShaders` read it.
/// Self-contained (Foundation and Metal only) so tools can compile it alone.
struct FieldProgram: Sendable {
    static let fileName = "field.bin"
    static let magic: UInt32 = 0x4750_4652  // "RFPG"
    static let version: UInt32 = 2
    static let maximumWords = (1 << 26) + (1 << 22)  // the kernel's 2^26 limit plus tables
    static let valueRegisters = 16
    static let pointRegisters = 8
    static let partWords = 24, meshWords = 24, polygonWords = 8, headerWords = 16

    enum Failure: Error, LocalizedError {
        case invalid(String)
        var errorDescription: String? {
            switch self {
            case .invalid(let detail): "The design's GPU field data is invalid (\(detail))."
            }
        }
    }

    /// One part (solid) of the program, in the build's coordinates.
    struct Part: Sendable, Equatable {
        let solid: Int
        /// False when the build left the part to mesh rendering (limits exceeded).
        let hasField: Bool
        let gridOffset: Int
        let dimensions: SIMD3<UInt32>
        let origin: SIMD3<Float>
        let brickSize: Float
        let margin: Float
        let boundsLower: SIMD3<Float>
        let boundsUpper: SIMD3<Float>
        /// Lipschitz bound of the part's field (sphere-tracing steps divide by the sample's own bound).
        let lipschitz: Float
        let levels: Int
        let tapes: Int
        let maximumInstructions: Int
        let cells: Int
    }

    /// The `fieldProgram` record of result.json.
    struct Manifest: Decodable, Sendable {
        let path: String?
        let version: Int?
        let words: Int?
        let brickSize: Double?
        let margin: Double?
        let error: String?
    }

    let words: [UInt32]
    let parts: [Part]
    let meshes: Int
    let polygons: Int
    /// Instructions of every validated tape, by word offset.
    let tapeInstructions: [Int: Int]
    /// Registers the tapes use (highest index + 1) and whether they read meshes or polygons.
    let valueRegistersUsed: Int
    let pointRegistersUsed: Int
    let readsMeshes: Bool
    let readsPolygons: Bool
    /// The opcodes the tapes use (point maps fused into leaves included): bit n
    /// of word n / 32.
    let opcodesUsed: [UInt32]

    /// Preprocessor macros for the evaluator's source (MTLCompileOptions.preprocessorMacros):
    /// register files of the next power of two at or above the tapes' use,
    /// mesh and polygon code only when tapes read them, the interpreter's
    /// cases for the opcodes they use (RAD_FIELD_OPS_A, _B, _C), and the
    /// four-sample gradient pass where register files are small (at most 4
    /// values and 2 points, no meshes or polygons: measured faster there, and
    /// slower in fragment functions with larger files).
    var shaderMacros: [String: NSNumber] {
        func rounded(_ count: Int, _ limit: Int) -> Int {
            var size = 2
            while size < count && size < limit { size *= 2 }
            return size
        }
        let values = rounded(valueRegistersUsed, Self.valueRegisters), points = rounded(pointRegistersUsed, Self.pointRegisters)
        let gradientPass = values <= 4 && points <= 2 && !readsMeshes && !readsPolygons
        return ["RAD_FIELD_VALUES": NSNumber(value: values),
                "RAD_FIELD_POINTS": NSNumber(value: points),
                "RAD_FIELD_GRADIENT_PASS": NSNumber(value: gradientPass ? 1 : 0),
                "RAD_FIELD_MESHES": NSNumber(value: readsMeshes ? 1 : 0),
                "RAD_FIELD_POLYGONS": NSNumber(value: readsPolygons ? 1 : 0),
                "RAD_FIELD_OPS_A": NSNumber(value: opcodesUsed[0]),
                "RAD_FIELD_OPS_B": NSNumber(value: opcodesUsed[1]),
                "RAD_FIELD_OPS_C": NSNumber(value: opcodesUsed[2])]
    }

    private struct Envelope: Decodable {
        let fieldProgram: Manifest?
    }

    static func manifest(in resultJSON: Data) throws -> Manifest? {
        try JSONDecoder().decode(Envelope.self, from: resultJSON).fieldProgram
    }

    /// The program a build published beside `resultJSON`, or nil when the
    /// build has none (an older build or a compile failure).
    static func read(resultJSON: Data, in folder: URL, solids: Int) throws -> FieldProgram? {
        guard let manifest = try manifest(in: resultJSON), manifest.error == nil else { return nil }
        guard manifest.path == fileName, manifest.version == Int(version), let count = manifest.words,
              count > headerWords, count <= maximumWords else {
            throw Failure.invalid("unexpected manifest")
        }
        let url = folder.appendingPathComponent(fileName)
        let descriptor = open(url.path, O_RDONLY | O_NOFOLLOW)
        guard descriptor >= 0 else { throw Failure.invalid("missing file") }
        defer { close(descriptor) }
        var status = stat()
        guard fstat(descriptor, &status) == 0, (status.st_mode & S_IFMT) == S_IFREG, status.st_size == off_t(4 * count) else {
            throw Failure.invalid("size")
        }
        let words = try [UInt32](unsafeUninitializedCapacity: count) { buffer, initialized in
            let bytes = UnsafeMutableRawBufferPointer(buffer)
            var done = 0
            while done < bytes.count {
                let received = pread(descriptor, bytes.baseAddress! + done, bytes.count - done, off_t(done))
                guard received > 0 else { throw Failure.invalid("read") }
                done += received
            }
            initialized = count
        }
        let program = try FieldProgram(words: words.map { UInt32(littleEndian: $0) })
        guard program.parts.count == solids else { throw Failure.invalid("part count") }
        return program
    }

    init(words: [UInt32]) throws {
        func fail(_ detail: String) -> Failure { .invalid(detail) }
        let n = words.count
        guard n >= Self.headerWords, n <= Self.maximumWords, words[0] == Self.magic, words[1] == Self.version,
              Int(words[3]) == n else { throw fail("header") }
        let partCount = Int(words[2]), partTable = Int(words[4])
        let meshTable = Int(words[5]), meshCount = Int(words[6])
        let polygonTable = Int(words[7]), polygonCount = Int(words[8])
        let maximumInstructions = Int(words[11])
        guard words[9] == UInt32(Self.valueRegisters), words[10] == UInt32(Self.pointRegisters),
              partCount >= 1, partCount <= 65536, maximumInstructions >= 1, maximumInstructions <= 1 << 20,
              partTable == Self.headerWords, meshTable == partTable + Self.partWords * partCount,
              polygonTable == meshTable + Self.meshWords * meshCount,
              polygonTable + Self.polygonWords * polygonCount <= n else { throw fail("tables") }
        func float(_ at: Int) -> Float { Float(bitPattern: words[at]) }
        func vector(_ at: Int) -> SIMD3<Float> { SIMD3(float(at), float(at + 1), float(at + 2)) }
        func within(_ offset: Int, _ count: Int) -> Bool { offset >= 0 && count >= 0 && count <= n && offset <= n - count }

        // Meshes and polygons referenced by tapes must be valid; an unused
        // record is all zero.
        var meshValid = [Bool](repeating: false, count: meshCount)
        for mesh in 0..<meshCount {
            let r = meshTable + Self.meshWords * mesh
            let v = Int(words[r]), t = Int(words[r + 3]), branches = Int(words[r + 8]), bodies = Int(words[r + 9])
            if v == 0 && t == 0 { continue }
            guard v >= 3, t >= 4, branches >= 1, bodies >= 1, v <= 2_000_000 * 3, t <= 2_000_000 * 4,
                  within(Int(words[r + 1]), 3 * v), within(Int(words[r + 2]), 3 * v), within(Int(words[r + 4]), 3 * t),
                  within(Int(words[r + 5]), 3 * t), within(Int(words[r + 6]), 3 * t), within(Int(words[r + 7]), 9 * branches),
                  within(Int(words[r + 10]), 8 * bodies) else { throw fail("mesh \(mesh)") }
            for i in 0..<(3 * t) {
                guard Int(words[Int(words[r + 4]) + i]) < v, Int(words[Int(words[r + 5]) + i]) < t else { throw fail("mesh \(mesh) triangles") }
            }
            func validTree(_ at: Int, _ count: Int, items: Int) -> Bool {
                for index in 0..<count {
                    let b = at + 9 * index
                    let right = Int(words[b + 6]), first = Int(words[b + 7]), size = Int(words[b + 8])
                    if right == 0 {
                        if first + size > items { return false }
                    } else if !(right > index + 1 && right < count) {
                        return false
                    }
                }
                return true
            }
            guard validTree(Int(words[r + 7]), branches, items: t) else { throw fail("mesh \(mesh) hierarchy") }
            var rootsLength = 0
            for body in 0..<bodies {
                let b = Int(words[r + 10]) + 8 * body
                rootsLength = max(rootsLength, Int(words[b + 6]) + Int(words[b + 7]))
            }
            guard within(Int(words[r + 11]), rootsLength) else { throw fail("mesh \(mesh) bodies") }
            for i in 0..<rootsLength where Int(words[Int(words[r + 11]) + i]) >= branches { throw fail("mesh \(mesh) roots") }
            if bodies > 1 {
                let bodyBranches = Int(words[r + 13])
                guard bodyBranches >= 1, within(Int(words[r + 12]), 9 * bodyBranches), within(Int(words[r + 14]), bodies),
                      within(Int(words[r + 15]), t), validTree(Int(words[r + 12]), bodyBranches, items: bodies) else {
                    throw fail("mesh \(mesh) body hierarchy")
                }
                for i in 0..<bodies where Int(words[Int(words[r + 14]) + i]) >= bodies { throw fail("mesh \(mesh) body order") }
                for i in 0..<t where Int(words[Int(words[r + 15]) + i]) >= bodies { throw fail("mesh \(mesh) body owners") }
            }
            meshValid[mesh] = true
        }
        var polygonSizes = [Int](repeating: 0, count: polygonCount)
        for polygon in 0..<polygonCount {
            let r = polygonTable + Self.polygonWords * polygon
            let count = Int(words[r]), nodes = Int(words[r + 3])
            if count == 0 { continue }
            guard count >= 3, nodes >= 1, within(Int(words[r + 1]), 7 * count), within(Int(words[r + 2]), 7 * nodes) else {
                throw fail("polygon \(polygon)")
            }
            for index in 0..<nodes {
                let b = Int(words[r + 2]) + 7 * index
                let begin = Int(words[b + 4]), end = Int(words[b + 5]), right = Int(words[b + 6])
                guard begin <= end, end <= count, right == 0 || (right > index + 1 && right < nodes) else {
                    throw fail("polygon \(polygon) hierarchy")
                }
            }
            polygonSizes[polygon] = count
        }

        // Tapes: decode every instruction of every tape a cell names.
        var tapes: [Int: Int] = [:]
        var valuesUsed = 1, pointsUsed = 1, meshes = false, polygons = false
        var opcodes: [UInt32] = [0, 0, 0]
        let parameterCounts: [Int: Int] = [
            1: 12, 2: 3, 3: 4, 4: 1, 5: 3, 6: 1, 7: 7, 8: 2, 9: 3, 10: 6, 11: 0, 12: 3,
            20: 1, 21: 4, 22: 2, 23: 6, 24: 8, 25: 2, 26: 4, 27: 4, 28: 0, 29: 1, 30: 3, 31: 0, 32: 6, 33: 3, 34: 3, 35: 0,
            40: 1, 41: 1, 42: 1, 43: 1, 44: 1, 45: 1, 46: 2, 47: 6, 48: 5, 49: 0, 50: 0, 51: 1, 52: 1, 53: 0, 54: 1, 55: 7,
            60: 1, 61: 0, 62: 0, 63: 0, 64: 0, 65: 0, 66: 0, 67: 0, 68: 0, 69: 0, 70: 0, 71: 0, 72: 0, 73: 1, 74: 0, 75: 2,
            76: 0, 77: 0, 78: 1, 79: 1, 80: 1,
        ]
        // A tape is 16-byte records: x = op | fused map << 7 | dst << 8 | post
        // << 12 | a << 16 | post register << 20 | b << 24 | swap << 28, y the
        // third operand or the post parameter, z the parameters' offset (a
        // MESH: its index), w a fused map's op << 28 | its parameters' offset.
        // Parameter runs are 16-byte aligned and read as whole float4s.
        func run(_ offset: Int, _ count: Int) -> Bool { offset % 4 == 0 && within(offset, (count + 3) & ~3) }
        func validateTape(_ start: Int) throws -> Int {
            if let known = tapes[start] { return known }
            guard start % 4 == 0 else { throw fail("tape \(start) alignment") }
            var at = start, instructions = 0
            let V = Self.valueRegisters, P = Self.pointRegisters
            while true {
                guard at + 4 <= n else { throw fail("tape \(start) runs past the end") }
                let x = words[at], y = Int(words[at + 1]), z = Int(words[at + 2]), mapWord = words[at + 3]
                at += 4
                let op = Int(x & 0x7F), fused = x & 0x80 != 0, post = Int((x >> 12) & 0xF), other = Int((x >> 20) & 0xF)
                let d = Int((x >> 8) & 0xF), a = Int((x >> 16) & 0xF), b = Int((x >> 24) & 0xF), swap = x & (1 << 28) != 0
                guard x >> 29 == 0 else { throw fail("tape \(start) record") }
                if op == 0 {
                    guard a < V, x == UInt32(a) << 16 else { throw fail("tape \(start) result") }
                    valuesUsed = max(valuesUsed, a + 1)
                    tapes[start] = instructions
                    return instructions
                }
                instructions += 1
                guard instructions <= maximumInstructions, let parameters = parameterCounts[op] else {
                    throw fail("tape \(start) instruction \(instructions)")
                }
                opcodes[op / 32] |= 1 << UInt32(op % 32)
                if fused {
                    let map = Int(mapWord >> 28)
                    guard (20...35).contains(op), (1...12).contains(map), map != 11, let count = parameterCounts[map],
                          run(Int(mapWord & 0x0FFF_FFFF), count) else { throw fail("tape \(start) fused map") }
                    opcodes[0] |= 1 << UInt32(map)
                } else {
                    guard mapWord == 0 else { throw fail("tape \(start) map word") }
                }
                // Post operations follow instructions with a value result and a free third-operand word.
                let producer = op >= 20 && op != 48
                guard post <= 8, post == 0 || producer, post != 0 || (other == 0 && !swap),
                      post == 0 || post <= 6 || (other == 0 && !swap), post == 0 || Float(bitPattern: UInt32(y)).isFinite
                else { throw fail("tape \(start) post operation") }
                var ok: Bool
                switch op {
                case 11: ok = d < P && a < V && b < V && y < V
                case 1...12: ok = d < P && a < P
                case 28: ok = d < V && a < P && z < meshCount && meshValid[z]
                case 31:
                    ok = d < V && a < P && z % 4 == 0 && within(z, 2) && Int(words[z]) < polygonCount && polygonSizes[Int(words[z])] >= 3
                    if ok {
                        let count = words[z + 1], size = polygonSizes[Int(words[z])]
                        if count != 0xFFFF_FFFF {
                            guard count <= 1 << 20, within(z + 2, 2 * Int(count)) else { throw fail("tape \(start) polygon") }
                            for r in 0..<Int(count) {
                                let begin = Int(words[z + 2 + 2 * r]), end = Int(words[z + 3 + 2 * r])
                                ok = ok && begin <= end && end <= size
                            }
                        }
                    }
                case 35: ok = d < V && a < P && b < 3
                case 20...34: ok = d < V && a < P
                case 46, 47, 55: ok = d < V && a < V && b < P
                case 48: ok = d < V && a < V && b < V && y < P
                case 49...54, 61...64, 74, 77, 79, 80: ok = d < V && a < V && b < V
                case 60: ok = d < V
                default: ok = d < V && a < V
                }
                if op != 28 && op != 31 { ok = ok && (parameters == 0 ? z == 0 : run(z, parameters)) }
                guard ok, post == 0 || other < V else { throw fail("tape \(start) operands") }
                // Register use: every operand field the op reads or writes.
                switch op {
                case 1...10, 12: pointsUsed = max(pointsUsed, d + 1, a + 1)
                case 11: pointsUsed = max(pointsUsed, d + 1); valuesUsed = max(valuesUsed, a + 1, b + 1, y + 1)
                case 20...35: valuesUsed = max(valuesUsed, d + 1); pointsUsed = max(pointsUsed, a + 1)
                case 46, 47, 55: valuesUsed = max(valuesUsed, d + 1, a + 1); pointsUsed = max(pointsUsed, b + 1)
                case 48: valuesUsed = max(valuesUsed, d + 1, a + 1, b + 1); pointsUsed = max(pointsUsed, y + 1)
                case 60: valuesUsed = max(valuesUsed, d + 1)
                default: valuesUsed = max(valuesUsed, d + 1, a + 1, b + 1)
                }
                if post != 0 { valuesUsed = max(valuesUsed, other + 1) }
                meshes = meshes || op == 28
                polygons = polygons || op == 31
            }
        }

        var parts: [Part] = []
        for index in 0..<partCount {
            let r = partTable + Self.partWords * index
            if words[r] == 1 {
                parts.append(Part(solid: index, hasField: false, gridOffset: 0, dimensions: .zero, origin: .zero, brickSize: 0,
                                  margin: 0, boundsLower: .zero, boundsUpper: .zero, lipschitz: 1, levels: 0, tapes: 0,
                                  maximumInstructions: 0, cells: 0))
                continue
            }
            guard words[r] == 0 else { throw fail("part \(index) status") }
            let dims = SIMD3(words[r + 2], words[r + 3], words[r + 4])
            let top = Int(dims.x) * Int(dims.y) * Int(dims.z)
            let grid = Int(words[r + 1]), total = Int(words[r + 23]), levels = Int(words[r + 5])
            let brick = float(r + 9), margin = float(r + 11), lipschitz = float(r + 18)
            let floats = (6...18).map { float(r + $0) }
            guard top >= 1, dims.x <= 4096, dims.y <= 4096, dims.z <= 4096, total >= top, (total - top) % 8 == 0,
                  total <= n, levels <= 16, floats.allSatisfy(\.isFinite), brick > 0, margin >= 0,
                  lipschitz >= 1 else { throw fail("part \(index) record") }
            // One word per cell (the top grid padded to whole 8-word blocks, then
            // octant blocks): far cells hold a signed bound, refined cells a
            // forward pointer to an octant block, others a tape's header record.
            let pad = (8 - top % 8) % 8, blocks = grid + top + pad
            guard grid % 8 == 0, within(grid, top + pad + (total - top)) else { throw fail("part \(index) cells") }
            for cell in 0..<total {
                let word = words[grid + (cell < top ? cell : cell + pad)]
                if word & 0x8000_0000 != 0 {
                    guard Float(bitPattern: (word & 0x3FFF_FFFF) << 1).isFinite else { throw fail("part \(index) cell \(cell)") }
                } else if word & 0x4000_0000 != 0 {
                    let target = Int(word & 0x3FFF_FFFF) << 3, child = top + (target - blocks)
                    guard target >= blocks, child > cell, child + 8 <= total else { throw fail("part \(index) cell \(cell) octants") }
                } else {
                    let header = Int(word) << 2
                    guard within(header, 4), float(header).isFinite, float(header) >= 1, words[header + 1] == 0,
                          words[header + 2] == 0, words[header + 3] == 0 else { throw fail("part \(index) cell \(cell)") }
                    _ = try validateTape(header + 4)
                }
            }
            parts.append(Part(solid: index, hasField: true, gridOffset: grid, dimensions: dims, origin: vector(r + 6),
                              brickSize: brick, margin: margin, boundsLower: vector(r + 12), boundsUpper: vector(r + 15),
                              lipschitz: lipschitz, levels: levels, tapes: Int(words[r + 19]),
                              maximumInstructions: Int(words[r + 22]), cells: total))
        }
        self.words = words
        self.parts = parts
        self.meshes = meshCount
        self.polygons = polygonCount
        self.tapeInstructions = tapes
        self.valueRegistersUsed = valuesUsed
        self.pointRegistersUsed = pointsUsed
        self.readsMeshes = meshes
        self.readsPolygons = polygons
        self.opcodesUsed = opcodes
    }

    /// The words as one shared buffer for the evaluator's functions.
    func makeBuffer(device: any MTLDevice) -> (any MTLBuffer)? {
        words.withUnsafeBytes { device.makeBuffer(bytes: $0.baseAddress!, length: $0.count, options: .storageModeShared) }
    }

    /// The tape a point of a part runs (the word offset of its first
    /// instruction), or nil outside the band, as the shaders look it up.
    func tape(part: Int, at point: SIMD3<Float>) -> Int? {
        let record = parts[part]
        guard record.hasField else { return nil }
        let cell = ((point - record.origin) * (1 / record.brickSize)).rounded(.down)
        let dims = SIMD3<Float>(record.dimensions)
        guard all(cell .>= 0), all(cell .< dims) else { return nil }
        var corner = record.origin + cell * record.brickSize
        var size = record.brickSize
        let c = SIMD3<Int>(cell)
        var word = words[record.gridOffset + c.x + Int(record.dimensions.x) * (c.y + Int(record.dimensions.y) * c.z)]
        for _ in 0..<16 where word >> 30 == 1 {
            size *= 0.5
            var octant = 0
            for axis in 0..<3 where point[axis] >= corner[axis] + size {
                octant |= 1 << axis
                corner[axis] += size
            }
            word = words[(Int(word & 0x3FFF_FFFF) << 3) + octant]
        }
        return word & 0x8000_0000 != 0 ? nil : (Int(word) << 2) + 4
    }
}
