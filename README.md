# UMAP-MLX-Swift

GPU-accelerated [UMAP](https://arxiv.org/abs/1802.03426) (Uniform Manifold Approximation and Projection) implementation in Swift using Apple's [MLX](https://github.com/ml-explore/mlx-swift) framework for optimal performance on Apple Silicon.

## Overview

This package implements the full UMAP dimensionality reduction pipeline on GPU via MLX:

1. **Preprocessing** -- optional normalization (z-score, min-max) and PCA reduction
2. **k-Nearest Neighbors** -- brute-force GPU pairwise distances (n <= 20K) or NNDescent approximation
3. **Fuzzy Simplicial Set** -- adaptive bandwidth search, edge symmetrization, pruning
4. **Spectral Initialization** -- power iteration for embedding starting positions
5. **SGD Optimization** -- compiled GPU kernel with attraction/repulsion forces

Reference implementation: [hanxiao/mlx-vis](https://github.com/hanxiao/mlx-vis) (Python MLX).

## Requirements

- macOS 26+ / iOS 26+
- Swift 6.2+
- Apple Silicon (M1 or later)
- Xcode with Metal Toolchain component installed

## Installation

Add to your `Package.swift`:

```swift
dependencies: [
    .package(url: "https://github.com/iliasaz/umap-mlx-swift", from: "0.1.0"),
]
```

Then add `UMAPMLX` as a dependency to your target:

```swift
.target(
    name: "MyTarget",
    dependencies: [
        .product(name: "UMAPMLX", package: "umap-mlx-swift"),
    ]
)
```

## Usage

```swift
import MLX
import UMAPMLX

// Input: (n, d) array of high-dimensional vectors
let data: MLXArray = ...  // e.g. 10000 x 768 embedding vectors

// Configure and run UMAP
var config = UMAP.Configuration()
config.nComponents = 2
config.nNeighbors = 15
config.minDist = 0.1
config.nEpochs = 200

let umap = UMAP(config)
let embedding = try umap.fitTransform(data) { epoch, total in
    print("Epoch \(epoch)/\(total)")
}

// embedding shape: (n, 2)
```

### Configuration Options

| Parameter | Default | Description |
|-----------|---------|-------------|
| `nComponents` | 2 | Output embedding dimensions |
| `nNeighbors` | 15 | Local neighborhood size for kNN graph |
| `minDist` | 0.1 | Minimum distance in embedding space |
| `spread` | 1.0 | Effective scale of embedded points |
| `nEpochs` | auto | Optimization iterations (500 if n<=10K, else 200) |
| `learningRate` | 1.0 | SGD learning rate |
| `negativeSampleRate` | 5 | Negative samples per positive edge |
| `randomSeed` | 42 | Seed for reproducibility (`nil` for random) |
| `pcaDimension` | nil | Optional PCA preprocessing target dimension |
| `normalization` | `.none` | `.none`, `.standard` (z-score), or `.minMax` |
| `knnMethod` | `.auto` | `.auto`, `.bruteForce`, or `.nnDescent` |

### Pre-computed kNN

If you already have kNN results, skip the distance computation:

```swift
let embedding = try umap.fitTransform(
    knnIndices: indices,    // (n, k) Int32
    knnDistances: distances, // (n, k) Float32
    n: totalPoints
)
```

## Testing

Tests require the Metal Toolchain (Xcode > Settings > Components):

```bash
xcodebuild test -scheme UMAPMLXSwift -destination 'platform=macOS' -skipPackagePluginValidation
```

## Origin

This project originated from [NewsComb#23](https://github.com/iliasaz/NewsComb/issues/23) to replace a CPU-based UMAP implementation (~220s for 89K vectors) with a GPU-accelerated version targeting 10-50x speedup on Apple Silicon.

## License

MIT
