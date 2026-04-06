// swift-tools-version: 6.2

import PackageDescription

let package = Package(
    name: "UMAPMLXSwift",
    platforms: [
        .iOS(.v26),
        .macOS(.v26),
    ],
    products: [
        .library(name: "UMAPMLX", targets: ["UMAPMLX"]),
    ],
    dependencies: [
        .package(url: "https://github.com/ml-explore/mlx-swift", from: "0.21.0"),
    ],
    targets: [
        .target(
            name: "UMAPMLX",
            dependencies: [
                .product(name: "MLX", package: "mlx-swift"),
                .product(name: "MLXRandom", package: "mlx-swift"),
                .product(name: "MLXFast", package: "mlx-swift"),
            ],
            path: "Sources"
        ),
        .testTarget(
            name: "UMAPMLXTests",
            dependencies: ["UMAPMLX"],
            path: "Tests"
        ),
    ]
)
