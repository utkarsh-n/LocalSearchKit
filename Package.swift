// swift-tools-version: 6.2
import PackageDescription

let package = Package(
    name: "LocalSearchKit",
    platforms: [.iOS(.v17), .macOS(.v14)],
    products: [
        .library(name: "LocalSearchKit", targets: ["LocalSearchKit"]),
        .library(name: "LocalSearchLlama", targets: ["LlamaRuntime"]),
    ],
    targets: [
        .target(name: "DigUpKit", exclude: ["Platform/Portable"]),
        .binaryTarget(name: "CLlama", path: "Vendor/LlamaRuntime.xcframework"),
        .target(name: "LlamaRuntime", dependencies: ["CLlama", "DigUpKit"], resources: [
            .copy("Resources/Metal"),
        ], linkerSettings: [
            .linkedLibrary("c++"), .linkedFramework("Accelerate"), .linkedFramework("Foundation"),
            .linkedFramework("Metal"), .linkedFramework("MetalKit"),
        ]),
        .target(name: "LocalSearchKit", dependencies: ["DigUpKit", "LlamaRuntime"]),
        .testTarget(name: "LocalSearchKitTests", dependencies: ["LocalSearchKit", "LlamaRuntime"]),
        .testTarget(name: "DigUpKitTests", dependencies: ["DigUpKit"], exclude: ["Pictures"]),
        .testTarget(name: "LlamaRuntimeTests", dependencies: ["LlamaRuntime"], exclude: ["Reference"]),
    ]
)
