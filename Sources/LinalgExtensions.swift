import Cmlx
import MLX

/// Compute the eigenvalues and eigenvectors of a real symmetric matrix.
///
/// Wraps `mlx_linalg_eigh` from the C API directly. Named `symmetricEigh`
/// rather than `eigh` to avoid an ambiguous-overload clash with `MLX.eigh`,
/// which newer mlx-swift now exposes publicly with the same call shape.
///
/// - Parameters:
///   - array: A symmetric input matrix of shape `(n, n)`.
///   - uplo: `"L"` to use the lower triangle (default), `"U"` for upper.
///   - stream: Stream or device to evaluate on.
/// - Returns: A tuple `(eigenvalues, eigenvectors)` where eigenvalues has
///   shape `(n,)` in ascending order and eigenvectors has shape `(n, n)`
///   with columns being the corresponding eigenvectors.
func symmetricEigh(
    _ array: MLXArray,
    uplo: String = "L",
    stream: StreamOrDevice = .default
) -> (MLXArray, MLXArray) {
    var eigvals = mlx_array_new()
    var eigvecs = mlx_array_new()
    mlx_linalg_eigh(&eigvals, &eigvecs, array.ctx, uplo, stream.ctx)
    return (MLXArray(eigvals), MLXArray(eigvecs))
}
