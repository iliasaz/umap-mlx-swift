import MLX

/// Normalize input data according to the specified strategy.
///
/// - Parameters:
///   - data: Input array of shape `(n, d)`.
///   - method: Normalization strategy.
/// - Returns: Normalized array of the same shape.
func normalizeInput(
    _ data: MLXArray,
    method: UMAP.Normalization,
    stream: StreamOrDevice = .default
) -> MLXArray {
    switch method {
    case .none:
        return data
    case .standard:
        let mean = data.mean(axis: 0, stream: stream)
        let std = data.variance(axis: 0, stream: stream).sqrt(stream: stream)
        return (data - mean) / (std + 1e-8)
    case .minMax:
        let minVal = data.min(axis: 0, stream: stream)
        let maxVal = data.max(axis: 0, stream: stream)
        return (data - minVal) / (maxVal - minVal + 1e-8)
    }
}

/// Reduce dimensionality via PCA using eigendecomposition.
///
/// Computes PCA by building the covariance matrix and extracting the
/// top eigenvectors via `eigh`. The eigendecomposition runs on CPU for
/// numerical stability.
///
/// - Parameters:
///   - data: Input array of shape `(n, d)`.
///   - dimension: Target number of principal components.
/// - Returns: Projected array of shape `(n, dimension)`.
func pcaReduce(
    _ data: MLXArray,
    toDimension dimension: Int,
    stream: StreamOrDevice = .default
) -> MLXArray {
    let d = data.dim(1)
    guard d > dimension else { return data }

    let n = data.dim(0)
    let mean = data.mean(axis: 0, stream: stream)
    let centered = data - mean

    // Covariance matrix: (d, d)
    let cov = matmul(centered.T, centered, stream: stream) / MLXArray(Float(n - 1))

    // Eigendecomposition on CPU for numerical stability
    let (_, eigvecs) = eigh(cov, stream: .cpu)
    eval(eigvecs)

    // eigvecs columns are in ascending eigenvalue order; take the last `dimension`
    let projection = eigvecs[0..., (d - dimension)...]

    let result = matmul(centered, projection, stream: stream)
    eval(result)
    return result
}
