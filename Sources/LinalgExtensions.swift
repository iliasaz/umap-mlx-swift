import Cmlx
import MLX

/// Compute the eigenvalues and eigenvectors of a real symmetric matrix.
///
/// Wraps `mlx_linalg_eigh` from the C API, which is not yet exposed in
/// the Swift `MLXLinalg` module.
///
/// - Parameters:
///   - array: A symmetric input matrix of shape `(n, n)`.
///   - uplo: `"L"` to use the lower triangle (default), `"U"` for upper.
///   - stream: Stream or device to evaluate on.
/// - Returns: A tuple `(eigenvalues, eigenvectors)` where eigenvalues has
///   shape `(n,)` in ascending order and eigenvectors has shape `(n, n)`
///   with columns being the corresponding eigenvectors.
func eigh(
    _ array: MLXArray,
    uplo: String = "L",
    stream: StreamOrDevice = .default
) -> (MLXArray, MLXArray) {
    var eigvals = mlx_array_new()
    var eigvecs = mlx_array_new()
    mlx_linalg_eigh(&eigvals, &eigvecs, array.ctx, uplo, stream.ctx)
    return (MLXArray(eigvals), MLXArray(eigvecs))
}
