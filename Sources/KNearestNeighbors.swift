import MLX
import MLXRandom

/// Compute exact k-nearest neighbors on GPU via brute-force pairwise distances.
///
/// Uses the identity `||x-y||^2 = ||x||^2 + ||y||^2 - 2*x.y` and processes
/// the data in chunks to limit GPU memory usage.
///
/// - Parameters:
///   - data: Input array of shape `(n, d)`.
///   - k: Number of nearest neighbors.
/// - Returns: Tuple of `(indices, distances)`, both of shape `(n, k)`.
///   Distances are Euclidean (not squared).
func computeBruteForceKNN(
    _ data: MLXArray,
    k: Int,
    stream: StreamOrDevice = .default
) -> (indices: MLXArray, distances: MLXArray) {
    let n = data.dim(0)

    // Squared norms: (n,)
    let sqNorms = (data * data).sum(axis: 1, stream: stream)
    eval(sqNorms)

    // Adaptive chunk size to limit GPU memory (~500 MB per chunk)
    let chunkSize = min(n, max(1000, 500_000_000 / (n * 4)))

    var allIndices = [MLXArray]()
    var allDistances = [MLXArray]()

    var prevIndices: MLXArray?
    var prevDistances: MLXArray?

    var start = 0
    while start < n {
        let end = min(start + chunkSize, n)

        // Pairwise squared distances for this chunk: (chunkLen, n)
        let chunkData = data[start ..< end]
        let chunkNorms = sqNorms[start ..< end]

        var distChunk = chunkNorms.expandedDimensions(axis: 1)
            + sqNorms.expandedDimensions(axis: 0)
            - 2.0 * matmul(chunkData, data.T, stream: stream)

        // Clamp negative values from numerical error
        distChunk = maximum(distChunk, MLXArray(Float(0)), stream: stream)

        // Mask self-distances with a large value
        let chunkLen = end - start
        if start < n {
            let rowIdx = MLXArray(Int32(start) ..< Int32(end)).expandedDimensions(axis: 1)
            let colIdx = MLXArray(Int32(0) ..< Int32(n)).expandedDimensions(axis: 0)
            let selfMask = rowIdx .== colIdx
            distChunk = which(selfMask, MLXArray(Float(1e30)), distChunk, stream: stream)
        }

        // Top-k via argsort + slice
        let sortedIdx = argSort(distChunk, axis: 1, stream: stream)[0..., ..<k]
        let sortedDist = takeAlong(distChunk, sortedIdx, axis: 1, stream: stream)

        // Pipeline: evaluate previous chunk while this one computes
        if let pi = prevIndices, let pd = prevDistances {
            eval(pi, pd)
            allIndices.append(pi)
            allDistances.append(pd)
        }

        prevIndices = sortedIdx
        prevDistances = sortedDist

        start = end
    }

    // Evaluate and append the last chunk
    if let pi = prevIndices, let pd = prevDistances {
        eval(pi, pd)
        allIndices.append(pi)
        allDistances.append(pd)
    }

    let indices = concatenated(allIndices, axis: 0, stream: stream)
    let distances = sqrt(maximum(
        concatenated(allDistances, axis: 0, stream: stream),
        MLXArray(Float(0)),
        stream: stream
    ))

    eval(indices, distances)
    return (indices: indices, distances: distances)
}

/// Dispatch kNN computation based on the configured method.
///
/// - Parameters:
///   - data: Input array of shape `(n, d)`.
///   - k: Number of nearest neighbors.
///   - method: kNN method selection.
/// - Returns: Tuple of `(indices, distances)`, both of shape `(n, k)`.
func computeKNN(
    _ data: MLXArray,
    k: Int,
    method: UMAP.KNNMethod,
    stream: StreamOrDevice = .default
) -> (indices: MLXArray, distances: MLXArray) {
    let n = data.dim(0)

    let useBruteForce: Bool
    switch method {
    case .bruteForce:
        useBruteForce = true
    case .nnDescent:
        useBruteForce = false
    case .auto:
        useBruteForce = n <= 20_000
    }

    if useBruteForce {
        return computeBruteForceKNN(data, k: k, stream: stream)
    } else {
        return computeNNDescent(data, k: k, stream: stream)
    }
}
