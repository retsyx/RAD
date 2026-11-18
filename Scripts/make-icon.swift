import AppKit
import SwiftUI
import simd

// RAD's mark is the smooth union of two spheres, ray-marched from its signed
// distance field in Palette.accent gold. The application and document icons
// place it on a dark macOS tile; the iCloud folder badge uses the mark alone.
// The window header draws the flat outline of the same field (RADMark), so keep
// the sphere and blend constants below identical to RADMark's.

let root = URL(fileURLWithPath: CommandLine.arguments[1])

typealias V = SIMD3<Double>

@inline(__always) func clamp(_ x: Double, _ a: Double, _ b: Double) -> Double { min(max(x, a), b) }
@inline(__always) func mix(_ a: Double, _ b: Double, _ t: Double) -> Double { a + (b - a) * t }
@inline(__always) func mix(_ a: V, _ b: V, _ t: Double) -> V { a + (b - a) * t }
@inline(__always) func smoothstep(_ a: Double, _ b: Double, _ x: Double) -> Double {
    let t = clamp((x - a) / (b - a), 0, 1); return t * t * (3 - 2 * t)
}
func linear(_ s: Double) -> Double { s <= 0.04045 ? s / 12.92 : pow((s + 0.055) / 1.055, 2.4) }
func encoded(_ l: Double) -> Double { let c = clamp(l, 0, 1); return c <= 0.0031308 ? c * 12.92 : 1.055 * pow(c, 1 / 2.4) - 0.055 }
func linear(_ r: Double, _ g: Double, _ b: Double) -> V { V(linear(r), linear(g), linear(b)) }

let gold = linear(0.96, 0.70, 0.36)            // Palette.accent
let background = linear(0.071, 0.086, 0.106)   // Palette.background
let raised = linear(0.145, 0.161, 0.184)       // Palette.raised

// Shared with RADMark: two spheres joined by a polynomial smooth minimum.
let centerA = V(-0.38, -0.16, 0), radiusA = 0.43
let centerB = V(0.44, 0.29, -0.08), radiusB = 0.31
let blend = 0.45

@inline(__always) func smoothMin(_ a: Double, _ b: Double, _ k: Double) -> Double {
    let h = max(k - abs(a - b), 0) / k
    return min(a, b) - h * h * k * 0.25
}
@inline(__always) func field(_ p: V) -> Double {
    smoothMin(simd_length(p - centerA) - radiusA, simd_length(p - centerB) - radiusB, blend)
}
func normal(_ p: V) -> V {
    let e = 1e-4
    let a = V(1, -1, -1), b = V(-1, -1, 1), c = V(-1, 1, -1), d = V(1, 1, 1)
    return simd_normalize(a * field(p + a * e) + b * field(p + b * e) + c * field(p + c * e) + d * field(p + d * e))
}
func occlusion(_ p: V, _ n: V) -> Double {
    var occ = 0.0, weight = 1.0
    for i in 0..<6 {
        let h = 0.01 + 0.10 * Double(i)
        occ += (h - field(p + n * h)) * weight
        weight *= 0.85
    }
    return clamp(1 - 1.6 * occ, 0, 1)
}
func softShadow(_ origin: V, _ direction: V, _ hardness: Double) -> Double {
    var result = 1.0, t = 0.02
    for _ in 0..<96 {
        let h = field(origin + direction * t)
        result = min(result, hardness * h / t)
        t += clamp(h, 0.005, 0.15)
        if result < -1 || t > 4 { break }
    }
    result = max(result, -1)
    return 0.25 * (1 + result) * (1 + result) * (2 - result)
}

// Camera, studio lights and the floor that receives a soft shadow.
let target = V(0.05, 0, 0)
let elevation = 20 * Double.pi / 180
let eye = target + V(0, sin(elevation), cos(elevation)) * 5
let forward = simd_normalize(target - eye)
let right = simd_normalize(simd_cross(forward, V(0, 1, 0)))
let up = simd_cross(right, forward)
let lens = 0.235
let floorY = -0.70
let keyLight = simd_normalize(V(-0.55, 0.80, 0.45))
let floorShadowLight = simd_normalize(V(0.25, 1, 0.5))
let tileHalf = 824.0 / 1024.0

/// A soft-edged rectangular area light seen in reflections.
func panel(_ r: V, _ direction: V, _ halfWidth: Double, _ halfHeight: Double, _ softness: Double) -> Double {
    let d = simd_normalize(direction)
    let u = simd_normalize(simd_cross(d, V(0, 1, 0)))
    let v = simd_cross(u, d)
    let c = simd_dot(r, d)
    if c <= 0 { return 0 }
    let a = simd_dot(r, u) / c, b = simd_dot(r, v) / c
    return (1 - smoothstep(halfWidth - softness, halfWidth + softness, abs(a)))
        * (1 - smoothstep(halfHeight - softness, halfHeight + softness, abs(b)))
}
func environment(_ r: V) -> V {
    let g = r.y >= 0 ? mix(0.03, 0.45, smoothstep(0.10, 1.0, r.y)) * 1.3
                     : mix(0.10, 0.015, smoothstep(0.0, -0.4, r.y))
    var c = V(g, g, g * 1.03)
    c += V(1, 1, 1) * (panel(r, keyLight, 0.55, 0.32, 0.12) * 2.6)
    c += V(0.92, 0.95, 1.0) * (panel(r, V(0.95, 0.15, -0.35), 0.10, 0.9, 0.08) * 1.8)
    c += V(1, 1, 1) * (panel(r, V(-0.9, 0.05, 0.4), 0.18, 0.12, 0.10) * 0.5)
    return c
}
func shadeMark(_ p: V, _ rd: V) -> V {
    let n = normal(p)
    let occ = occlusion(p, n)
    let cosine = clamp(simd_dot(n, -rd), 0, 1)
    let fresnel = gold + (V(1, 1, 1) - gold) * pow(1 - cosine, 5)
    let shadow = softShadow(p + n * 0.002, keyLight, 10)
    let reflection = environment(simd_reflect(rd, n)) * (0.35 + 0.65 * occ) * mix(0.35, 1.0, shadow)
    let diffuse: Double = max(simd_dot(n, keyLight), 0) * shadow * 0.06
    let ambient: Double = (0.02 + 0.02 * n.y + max(-n.y, 0) * 0.05) * occ
    return fresnel * reflection + gold * (diffuse + ambient)
}
func backdrop(_ u: Double, _ v: Double) -> V {
    let d = (u * u + (v - 0.25) * (v - 0.25)).squareRoot()
    return mix(raised * 1.15, background * 0.8, smoothstep(0.0, 1.35, d))
}
func shadeFloor(_ p: V, _ u: Double, _ v: Double) -> V {
    let shadow = softShadow(p, floorShadowLight, 1.2)
    let occ = occlusion(p, V(0, 1, 0))
    return backdrop(u, v) * ((0.25 + 0.75 * shadow) * (0.35 + 0.65 * occ))
}
/// Hue-preserving exposure and shoulder; the brightest reflections roll toward pale gold.
func toneMap(_ input: V) -> V {
    let c = input * 1.3
    let m = max(c.x, c.y, c.z, 1e-9)
    let shoulder = m / (1 + m / 1.6) * (1 + 1 / 1.6)
    let out = mix(c * (min(shoulder, 1) / m), linear(1.0, 0.95, 0.82), smoothstep(0.9, 3.0, m) * 0.9)
    return simd_clamp(out, V(0, 0, 0), V(1, 1, 1))
}

/// Ray-marches the mark into a premultiplied sRGB image. With a tile, every
/// pixel shows the lit studio scene (the caller clips it to the tile shape);
/// without one, only the mark is opaque.
func renderMark(pixels: Int, tile: Bool) -> CGImage {
    let samples = pixels <= 64 ? 6 : (pixels <= 256 ? 5 : (pixels <= 512 ? 4 : 3))
    let step = 2.0 / Double(pixels)
    var bytes = [UInt8](repeating: 0, count: pixels * pixels * 4)
    bytes.withUnsafeMutableBufferPointer { buffer in
        let base = buffer.baseAddress!
        DispatchQueue.concurrentPerform(iterations: pixels) { row in
            for column in 0..<pixels {
                var color = V(0, 0, 0), alpha = 0.0
                for sy in 0..<samples {
                    for sx in 0..<samples {
                        let x = -1 + (Double(column) + (Double(sx) + 0.5) / Double(samples)) * step
                        let y = 1 - (Double(row) + (Double(sy) + 0.5) / Double(samples)) * step
                        // The badge frames the mark alone; the tile frames the studio.
                        let u = tile ? x / tileHalf : x * 0.70 - 0.065
                        let v = tile ? y / tileHalf : y * 0.70 + 0.015
                        let rd = simd_normalize(forward + right * (u * lens) + up * (v * lens))
                        var t = 0.0, hit = false
                        for _ in 0..<160 {
                            let d = field(eye + rd * t)
                            if d < 1e-4 * max(t, 1) { hit = true; break }
                            t += d
                            if t > 20 { break }
                        }
                        let floorT = rd.y < 0 ? (floorY - eye.y) / rd.y : .infinity
                        if hit && t < floorT {
                            color += toneMap(shadeMark(eye + rd * t, rd)); alpha += 1
                        } else if tile {
                            color += toneMap(floorT.isFinite ? shadeFloor(eye + rd * floorT, u, v) : backdrop(u, v)); alpha += 1
                        }
                    }
                }
                let n = Double(samples * samples)
                let a = alpha / n
                let c = alpha > 0 ? color / alpha : V(0, 0, 0)
                let o = (row * pixels + column) * 4
                base[o] = UInt8(clamp(encoded(c.x) * a * 255 + 0.5, 0, 255))
                base[o + 1] = UInt8(clamp(encoded(c.y) * a * 255 + 0.5, 0, 255))
                base[o + 2] = UInt8(clamp(encoded(c.z) * a * 255 + 0.5, 0, 255))
                base[o + 3] = UInt8(clamp(a * 255 + 0.5, 0, 255))
            }
        }
    }
    let context = CGContext(data: &bytes, width: pixels, height: pixels, bitsPerComponent: 8,
                            bytesPerRow: pixels * 4, space: CGColorSpace(name: CGColorSpace.sRGB)!,
                            bitmapInfo: CGImageAlphaInfo.premultipliedLast.rawValue)!
    return context.makeImage()!
}

func writeIcon(named name: String, draw: (CGContext, CGFloat) -> Void) throws {
    let iconset = root.appendingPathComponent("\(name).iconset")
    try FileManager.default.createDirectory(at: iconset, withIntermediateDirectories: true)
    for size in [16, 32, 128, 256, 512] {
        for scale in [1, 2] {
            let px = size * scale
            let context = CGContext(data: nil, width: px, height: px, bitsPerComponent: 8, bytesPerRow: 0,
                                    space: CGColorSpace(name: CGColorSpace.sRGB)!,
                                    bitmapInfo: CGImageAlphaInfo.premultipliedLast.rawValue)!
            context.interpolationQuality = .high
            draw(context, CGFloat(px))
            let bitmap = NSBitmapImageRep(cgImage: context.makeImage()!)
            let suffix = scale == 2 ? "@2x" : ""
            try bitmap.representation(using: .png, properties: [:])!
                .write(to: iconset.appendingPathComponent("icon_\(size)x\(size)\(suffix).png"))
        }
    }
    let process = Process()
    process.executableURL = URL(fileURLWithPath: "/usr/bin/iconutil")
    process.arguments = ["-c", "icns", iconset.path, "-o", root.appendingPathComponent("\(name).icns").path]
    try process.run()
    process.waitUntilExit()
    guard process.terminationStatus == 0 else { exit(1) }
}

// Render each pixel size directly, rather than resampling one large image.
var tileImages: [Int: CGImage] = [:]
try writeIcon(named: "RAD") { context, s in
    let px = Int(s), unit = s / 1024
    let image = tileImages[px] ?? renderMark(pixels: px, tile: true)
    tileImages[px] = image
    // macOS icon grid: an 824-point continuous-corner tile centered on 1024.
    let tile = RoundedRectangle(cornerRadius: 185.4 * unit, style: .continuous)
        .path(in: CGRect(x: 100 * unit, y: 100 * unit, width: 824 * unit, height: 824 * unit)).cgPath
    context.saveGState()
    context.setShadow(offset: CGSize(width: 0, height: -10 * unit), blur: 24 * unit,
                      color: CGColor(gray: 0, alpha: 0.45))
    context.addPath(tile)
    context.setFillColor(CGColor(srgbRed: 0.071, green: 0.086, blue: 0.106, alpha: 1))
    context.fillPath()
    context.restoreGState()
    context.saveGState()
    context.addPath(tile)
    context.clip()
    context.draw(image, in: CGRect(x: 0, y: 0, width: s, height: s))
    // A faint top-lit inner edge separates the tile from dark surroundings.
    context.addPath(tile)
    context.setLineWidth(5 * unit)
    context.replacePathWithStrokedPath()
    context.clip()
    let rim = CGGradient(colorsSpace: CGColorSpace(name: CGColorSpace.sRGB)!,
                         colors: [CGColor(gray: 1, alpha: 0.10), CGColor(gray: 1, alpha: 0)] as CFArray,
                         locations: [0, 1])!
    context.drawLinearGradient(rim, start: CGPoint(x: 0, y: 924 * unit), end: CGPoint(x: 0, y: 400 * unit), options: [])
    context.restoreGState()
}
// Document icons use the same artwork as the application, with no alternate mark.
try Data(contentsOf: root.appendingPathComponent("RAD.icns"))
    .write(to: root.appendingPathComponent("RADDocument.icns"))

guard let folder = NSImage(named: NSImage.folderName) else { fatalError("Could not load the system folder icon") }
try writeIcon(named: "RADFolder") { context, s in
    let unit = s / 1024
    NSGraphicsContext.saveGraphicsState()
    NSGraphicsContext.current = NSGraphicsContext(cgContext: context, flipped: false)
    folder.draw(in: NSRect(x: 0, y: 0, width: s, height: s))
    NSGraphicsContext.restoreGraphicsState()
    let badge = CGRect(x: 252 * unit, y: 156 * unit, width: 520 * unit, height: 520 * unit)
    context.saveGState()
    context.setShadow(offset: CGSize(width: 0, height: -6 * unit), blur: 14 * unit, color: CGColor(gray: 0, alpha: 0.35))
    context.draw(renderMark(pixels: max(Int(badge.width.rounded(.up)), 8), tile: false), in: badge)
    context.restoreGState()
}
