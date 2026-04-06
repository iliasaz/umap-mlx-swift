import MLX
import MLXRandom

/// Compute a spectral embedding initialization via power iteration.
///
/// Extracts the top eigenvectors of the degree-normalized adjacency matrix
/// using power iteration with Gram-Schmidt orthogonalization. This provides
/// a better starting point than random initialization for the SGD phase.
///
/// - Parameters:
///   - graph: The sparse weighted graph from fuzzy simplicial set construction.
///   - n: Total number of data points.
///   - nComponents: Number of embedding dimensions.
///   - seed: Random seed for reproducibility.
/// - Returns: An `MLXArray` of shape `(n, nComponents)`, or `nil` if
///   spectral initialization fails (caller should fall back to random).
func spectralInit(
    graph: SparseGraph,
    n: Int,
    nComponents: Int,
    seed: UInt64?,
    stream: StreamOrDevice = .default
) -> MLXArray? {
    if let seed {
        MLXRandom.seed(seed)
    }

    // --- Degree computation via scatter-add ---
    let degrees = MLXArray.zeros([n], dtype: .float32)
        .at[graph.rows, stream: stream].add(graph.vals)
    eval(degrees)

    // Check for zero-degree nodes
    let minDegree = degrees.min(stream: stream)
    eval(minDegree)
    guard minDegree.item(Float.self) > 0 else { return nil }

    // --- Degree-normalized weights ---
    let dInvSqrt = 1.0 / sqrt(maximum(degrees, MLXArray(Float(1e-10)), stream: stream))
    let wNorm = graph.vals * takeAlong(dInvSqrt, graph.rows, stream: stream)
        * takeAlong(dInvSqrt, graph.cols, stream: stream)
    eval(wNorm)

    // --- Power iteration ---
    let k = nComponents + 1
    var V = MLXRandom.normal([n, k])
    eval(V)

    let nIterations = 100

    for iteration in 0 ..< nIterations {
        // Sparse matrix-vector multiply: result[row] += wNorm * V[col]
        let gathered = V[graph.cols]  // (nEdges, k)
        let weighted = wNorm.expandedDimensions(axis: 1) * gathered  // (nEdges, k)
        var newV = MLXArray.zeros([n, k], dtype: .float32)
            .at[graph.rows, stream: stream].add(weighted)

        // Modified Gram-Schmidt orthogonalization
        for j in 0 ..< k {
            var col = newV[0..., j]
            for i in 0 ..< j {
                let prevCol = newV[0..., i]
                let dot = (col * prevCol).sum()
                col = col - dot * prevCol
            }
            let norm = sqrt((col * col).sum() + 1e-10)
            newV[0..., j] = col / norm
        }

        V = newV

        if iteration % 10 == 9 {
            eval(V)
        }
    }
    eval(V)

    // --- Extract components ---
    // Skip column 0 (trivial eigenvector corresponding to largest eigenvalue)
    var Y = V[0..., 1 ..< (nComponents + 1)]

    // Check for NaN
    let hasNaN = isNaN(Y, stream: stream).any(stream: stream)
    eval(hasNaN)
    if hasNaN.item(Bool.self) {
        return nil
    }

    // --- Scale to [0, 10] + noise ---
    let absMax = abs(Y, stream: stream).max(stream: stream)
    eval(absMax)
    let maxVal = absMax.item(Float.self)
    guard maxVal > 0 else { return nil }

    let expansion = 10.0 / MLXArray(maxVal)
    Y = Y * expansion

    let yMin = Y.min(stream: stream)
    let yMax = Y.max(stream: stream)
    Y = 10.0 * (Y - yMin) / (yMax - yMin + 1e-10)

    // Add small noise
    Y = Y + MLXRandom.normal(Y.shape) * 0.0001

    // Final normalization to [0, 10]
    let finalMin = Y.min(stream: stream)
    let finalMax = Y.max(stream: stream)
    Y = 10.0 * (Y - finalMin) / (finalMax - finalMin + 1e-10)

    eval(Y)
    return Y
}
