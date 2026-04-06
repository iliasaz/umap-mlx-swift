import MLX
import MLXRandom

/// Compute approximate k-nearest neighbors using NNDescent.
///
/// NNDescent iteratively refines a kNN graph by exploring neighbors-of-neighbors.
/// This is more memory-efficient than brute-force for large datasets (n > 20K)
/// while providing high recall (~95%+).
///
/// - Parameters:
///   - data: Input array of shape `(n, d)`.
///   - k: Number of nearest neighbors.
///   - maxIterations: Maximum refinement iterations (default 20).
///   - stream: Stream or device to evaluate on.
/// - Returns: Tuple of `(indices, distances)`, both of shape `(n, k)`.
func computeNNDescent(
    _ data: MLXArray,
    k: Int,
    maxIterations: Int = 20,
    stream: StreamOrDevice = .default
) -> (indices: MLXArray, distances: MLXArray) {
    let n = data.dim(0)
    let d = data.dim(1)

    // --- Random initialization ---
    // Initialize each point with k random neighbors
    var indices = MLXArray.zeros([n, k], dtype: .int32)
    for i in 0 ..< n {
        let randIdx = MLXRandom.randInt(low: 0, high: Int32(n), [k], type: Int32.self)
        indices[i] = randIdx
    }
    eval(indices)

    // Compute initial distances
    var distances = computeDistances(data, indices: indices, stream: stream)
    eval(distances)

    // Sort each row by distance
    let sortOrder = argSort(distances, axis: 1, stream: stream)
    indices = takeAlong(indices, sortOrder, axis: 1, stream: stream)
    distances = takeAlong(distances, sortOrder, axis: 1, stream: stream)
    eval(indices, distances)

    // --- Iterative refinement ---
    // Chunk size for processing to limit memory (~300 MB budget)
    let rowChunkSize = min(n, max(500, 300_000_000 / (k * k * d * 4)))

    for iteration in 0 ..< maxIterations {
        var totalUpdates = 0

        var rowStart = 0
        while rowStart < n {
            let rowEnd = min(rowStart + rowChunkSize, n)
            let chunkSize = rowEnd - rowStart

            // For each point in this chunk, gather candidate neighbors
            // by looking at neighbors-of-neighbors
            let chunkIndices = indices[rowStart ..< rowEnd]  // (chunkSize, k)

            // Gather neighbor lists for all neighbors in this chunk
            // candidateIndices[i] = union of neighbors[neighbors[i]]
            let nnIndices = indices[chunkIndices.reshaped([-1])]  // (chunkSize * k, k)
            let candidates = nnIndices.reshaped([chunkSize, k * k])  // (chunkSize, k*k)

            // Combine with current neighbors
            let allCandidates = concatenated(
                [chunkIndices, candidates],
                axis: 1,
                stream: stream
            )  // (chunkSize, k + k*k)

            // Compute distances to all candidates
            let nCandidates = allCandidates.dim(1)

            // Gather candidate vectors: (chunkSize, nCandidates, d)
            let flatCandidates = allCandidates.reshaped([-1])
            let candidateVectors = data[flatCandidates]
                .reshaped([chunkSize, nCandidates, d])

            // Source vectors: (chunkSize, 1, d)
            let sourceVectors = data[MLXArray(Int32(rowStart) ..< Int32(rowEnd))]
                .expandedDimensions(axis: 1)

            // Distances: (chunkSize, nCandidates)
            let diffs = sourceVectors - candidateVectors
            let candidateDistances = (diffs * diffs).sum(axis: 2, stream: stream)
                .sqrt(stream: stream)

            // Mask self-distances
            let selfIdx = MLXArray(Int32(rowStart) ..< Int32(rowEnd))
                .expandedDimensions(axis: 1)
            let candidateSelfMask = allCandidates .== selfIdx
            let maskedDistances = which(
                candidateSelfMask,
                MLXArray(Float(1e30)),
                candidateDistances,
                stream: stream
            )

            // Select top-k from candidates
            let topKOrder = argSort(maskedDistances, axis: 1, stream: stream)[0..., ..<k]
            let newIndices = takeAlong(allCandidates, topKOrder, axis: 1, stream: stream)
            let newDistances = takeAlong(maskedDistances, topKOrder, axis: 1, stream: stream)

            // Count updates
            let changed = newIndices .!= indices[rowStart ..< rowEnd]
            let nChanged = changed.asType(.int32, stream: stream).sum(stream: stream)
            eval(nChanged)
            totalUpdates += Int(nChanged.item(Int32.self))

            // Apply updates
            indices[rowStart ..< rowEnd] = newIndices
            distances[rowStart ..< rowEnd] = newDistances
            eval(indices, distances)

            rowStart = rowEnd
        }

        // Check convergence
        let updateFraction = Float(totalUpdates) / Float(n * k)
        if updateFraction < 0.001 {
            break
        }
    }

    eval(indices, distances)
    return (indices: indices, distances: distances)
}

/// Compute Euclidean distances between points and their specified neighbors.
///
/// - Parameters:
///   - data: Input data of shape `(n, d)`.
///   - indices: Neighbor indices of shape `(n, k)`.
/// - Returns: Distance array of shape `(n, k)`.
private func computeDistances(
    _ data: MLXArray,
    indices: MLXArray,
    stream: StreamOrDevice = .default
) -> MLXArray {
    let n = data.dim(0)
    let k = indices.dim(1)
    let d = data.dim(1)

    // Gather neighbor vectors
    let flatIdx = indices.reshaped([-1])
    let neighborVectors = data[flatIdx].reshaped([n, k, d])

    // Source vectors: (n, 1, d)
    let sourceVectors = data.expandedDimensions(axis: 1)

    // Euclidean distances
    let diffs = sourceVectors - neighborVectors
    let distances = (diffs * diffs).sum(axis: 2, stream: stream).sqrt(stream: stream)

    return distances
}
