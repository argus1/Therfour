// swift-tools-version: 6.2
// The swift-tools-version declares the minimum version of Swift required to build this package.

import PackageDescription

let package = Package(
    name: "swift-backend",
    platforms: [
        .macOS(.v12),
    ],
    products: [
        .executable(name: "swift-backend", targets: ["swift-backend"]),
    ],
    targets: [
        .executableTarget(
            name: "swift-backend"
        ),
        .testTarget(
            name: "swift-backendTests",
            dependencies: ["swift-backend"]
        ),
    ]
)
