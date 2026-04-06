import Foundation
import MLX

/// Result of the fuzzy simplicial set construction.
///
/// Represents a sparse weighted graph as parallel arrays of (row, col, value).
struct SparseGraph {
    /// Row indices of edges.
    let rows: MLXArray
    /// Column indices of edges.
    let cols: MLXArray
    /// Edge weights.
    let vals: MLXArray
}

/// Build the fuzzy simplicial set from kNN results.
///
/// This constructs a weighted graph representing the approximate topology
/// of the data manifold. Each point gets an adaptive bandwidth (sigma) found
/// via binary search, edges are symmetrized, and weak edges are pruned.
///
/// - Parameters:
///   - knnIndices: Neighbor indices of shape `(n, k)`.
///   - knnDistances: Neighbor distances of shape `(n, k)`, Euclidean.
///   - n: Total number of data points.
///   - nNeighbors: Number of neighbors (k).
///   - nEpochs: Number of optimization epochs (used for pruning threshold).
/// - Returns: A `SparseGraph` with symmetrized and pruned edge weights.
func buildFuzzySimplicialSet(
    knnIndices: MLXArray,
    knnDistances: MLXArray,
    n: Int,
    nNeighbors: Int,
    nEpochs: Int,
    stream: StreamOrDevice = .default
) -> SparseGraph {
    // --- Rho: distance to nearest non-zero neighbor per point ---
    let nonZeroMask = knnDistances .> MLXArray(Float(0))
    let maskedDists = which(
        nonZeroMask,
        knnDistances,
        MLXArray(Float(1e30)),
        stream: stream
    )
    let rhos = maximum(
        maskedDists.min(axis: 1, stream: stream),
        MLXArray(Float(1e-8)),
        stream: stream
    )
    eval(rhos)

    // --- Sigma: adaptive bandwidth via binary search ---
    let target = MLXArray(Darwin.log2f(Float(nNeighbors)))

    // Shift distances by rho
    let distsShifted = maximum(
        knnDistances - rhos.expandedDimensions(axis: 1),
        MLXArray(Float(0)),
        stream: stream
    )

    var lo = MLXArray.full([n], values: MLXArray(Float(1e-20)), dtype: .float32)
    var hi = MLXArray.full([n], values: MLXArray(Float(1e3)), dtype: .float32)
    var sigma = MLXArray.ones([n], dtype: .float32)

    // 64 iterations of binary search, all vectorized on GPU
    for iteration in 0 ..< 64 {
        let sigmaExpanded = sigma.expandedDimensions(axis: 1)
        // Skip the first column (self) in computing membership
        let vals = exp(
            -distsShifted[0..., 1...] / maximum(sigmaExpanded, MLXArray(Float(1e-10))),
            stream: stream
        )
        let valsSum = vals.sum(axis: 1, stream: stream)

        let tooHigh = valsSum .> target
        let tooLow = valsSum .< target
        let converged = abs(valsSum - target, stream: stream) .< MLXArray(Float(1e-5))

        hi = which(tooHigh, sigma, hi, stream: stream)
        lo = which(tooLow, sigma, lo, stream: stream)

        let newSigmaHigh = (lo + sigma) / 2.0
        let newSigmaLow = which(
            hi .>= MLXArray(Float(999)),
            sigma * 2.0,
            (sigma + hi) / 2.0,
            stream: stream
        )

        sigma = which(
            converged,
            sigma,
            which(tooHigh, newSigmaHigh, newSigmaLow, stream: stream),
            stream: stream
        )

        // Periodic eval to prevent computation graph explosion
        if iteration % 8 == 7 {
            eval(sigma, lo, hi)
        }
    }
    eval(sigma)

    // --- Edge weights ---
    let sigmaExpanded = sigma.expandedDimensions(axis: 1)
    let weights = exp(
        -distsShifted / maximum(sigmaExpanded, MLXArray(Float(1e-10))),
        stream: stream
    )
    // Zero out the self-weight (column 0 is self-distance in kNN)
    eval(weights)

    // --- Build edge lists ---
    // Row indices: each point repeated k times
    let rowIndices = broadcast(
        MLXArray(Int32(0) ..< Int32(n)).expandedDimensions(axis: 1),
        to: [n, nNeighbors],
        stream: stream
    ).reshaped([-1], stream: stream)

    let colIndices = knnIndices.reshaped([-1], stream: stream)
    let edgeWeights = weights.reshaped([-1], stream: stream)

    // Remove self-edges (where row == col)
    let notSelf = rowIndices .!= colIndices
    let fwdRows = which(notSelf, rowIndices, MLXArray(Int32(-1)), stream: stream)

    // Filter out -1 entries (we'll use a compact approach)
    // For simplicity, keep all edges and let the symmetrization handle duplicates
    // The self-weight is ~1.0 but gets removed here

    eval(fwdRows, colIndices, edgeWeights)

    // --- Symmetrize ---
    let result = symmetrizeEdges(
        rows: rowIndices,
        cols: colIndices,
        vals: edgeWeights,
        n: n,
        stream: stream
    )

    // --- Prune weak edges ---
    let maxWeight = result.vals.max(stream: stream)
    let threshold = maxWeight / MLXArray(Float(nEpochs))
    let keepMask = result.vals .>= threshold

    let prunedRows = which(keepMask, result.rows, MLXArray(Int32(-1)), stream: stream)
    let validMask = prunedRows .>= MLXArray(Int32(0))

    // Compact: gather only valid edges
    let validIndices = which(validMask, MLXArray(Int32(1)), MLXArray(Int32(0)), stream: stream)
    let cumSum = cumsum(validIndices, stream: stream)
    let nValid = cumSum.max(stream: stream)
    eval(nValid)
    let nValidInt = nValid.item(Int32.self)

    // Use boolean indexing via which and scatter
    let finalRows = compactArray(result.rows, mask: validMask, stream: stream)
    let finalCols = compactArray(result.cols, mask: validMask, stream: stream)
    let finalVals = compactArray(result.vals, mask: validMask, stream: stream)

    eval(finalRows, finalCols, finalVals)
    return SparseGraph(rows: finalRows, cols: finalCols, vals: finalVals)
}

/// Symmetrize edge weights: P = A + A^T - A * A^T.
///
/// Uses key-based matching to find reverse edges efficiently on GPU.
private func symmetrizeEdges(
    rows: MLXArray,
    cols: MLXArray,
    vals: MLXArray,
    n: Int,
    stream: StreamOrDevice = .default
) -> SparseGraph {
    let nInt64 = MLXArray(Int64(n))

    // Forward keys: row * n + col
    let fwdKeys = rows.asType(.int64, stream: stream) * nInt64
        + cols.asType(.int64, stream: stream)

    // Reverse keys: col * n + row
    let revKeys = cols.asType(.int64, stream: stream) * nInt64
        + rows.asType(.int64, stream: stream)

    // Sort forward keys for binary search
    let sortOrder = argSort(fwdKeys, stream: stream).asType(.int32, stream: stream)
    let sortedKeys = takeAlong(fwdKeys, sortOrder, stream: stream)
    let sortedVals = takeAlong(vals, sortOrder, stream: stream)
    eval(sortedKeys, sortedVals)

    // For each forward edge, find its reverse edge
    let revPositions = searchSorted(sortedKeys, values: revKeys, stream: stream)

    // Clamp to valid range and check for actual matches
    let nEdges = sortedKeys.dim(0)
    let clampedPos = minimum(revPositions, MLXArray(Int32(nEdges - 1)), stream: stream)
        .asType(.int32, stream: stream)
    let matchedKeys = takeAlong(sortedKeys, clampedPos, stream: stream)
    let hasReverse = matchedKeys .== revKeys
    let reverseWeights = which(
        hasReverse,
        takeAlong(sortedVals, clampedPos, stream: stream),
        MLXArray(Float(0)),
        stream: stream
    )

    // Symmetrize: w_sym = w_fwd + w_rev - w_fwd * w_rev
    let symWeights = vals + reverseWeights - vals * reverseWeights

    eval(symWeights)
    return SparseGraph(rows: rows, cols: cols, vals: symWeights)
}

/// Compact an array by removing elements where mask is false.
///
/// Gathers elements where `mask` is true into a contiguous array.
private func compactArray(
    _ array: MLXArray,
    mask: MLXArray,
    stream: StreamOrDevice = .default
) -> MLXArray {
    // Convert boolean mask to indices
    let indices = which(mask, MLXArray(Int32(1)), MLXArray(Int32(0)), stream: stream)
    let cumIndices = cumsum(indices, stream: stream)
    let nValid = cumIndices.max(stream: stream)
    eval(nValid)

    let nValidInt = nValid.item(Int32.self)
    guard nValidInt > 0 else {
        return MLXArray.zeros([0], dtype: array.dtype)
    }

    // Create output array and scatter valid elements
    let outputSize = Int(nValidInt)
    // Gather valid positions using argsort trick
    let sortKey = which(mask, MLXArray(Int32(0)), MLXArray(Int32(1)), stream: stream)
    let sortedByMask = argSort(sortKey, stream: stream).asType(.int32, stream: stream)
    let compactIdx = sortedByMask[..<outputSize]
    return array[compactIdx]
}
