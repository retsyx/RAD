// swift-tools-version: 5.9
import PackageDescription

let package = Package(
    name: "RAD",
    platforms: [.macOS(.v14)],
    products: [
        .executable(name: "RAD", targets: ["RAD"]),
        .executable(name: "RADThumbnailExtension", targets: ["RADThumbnailExtension"]),
        .library(name: "RADSDF", type: .dynamic, targets: ["RADSDFKernel"])
    ],
    targets: [
        .executableTarget(name: "RAD", resources: [.copy("Resources")]),
        .target(name: "RADThumbnailSupport", swiftSettings: [.unsafeFlags(["-application-extension"])]),
        .executableTarget(
            name: "RADThumbnailExtension",
            dependencies: ["RADThumbnailSupport"],
            swiftSettings: [.unsafeFlags(["-parse-as-library", "-application-extension"])],
            linkerSettings: [.unsafeFlags(["-Xlinker", "-e", "-Xlinker", "_NSExtensionMain"])]
        ),
        .target(
            name: "RADSDFKernel",
            path: "Sources/RADSDFKernel",
            publicHeadersPath: "include",
            cxxSettings: [.unsafeFlags(["-O3", "-fno-math-errno"])]
        ),
        .testTarget(name: "RADTests", dependencies: ["RAD"]),
        .testTarget(name: "RADThumbnailSupportTests", dependencies: ["RADThumbnailSupport"])
    ],
    cxxLanguageStandard: .cxx17
)
