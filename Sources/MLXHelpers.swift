import Foundation
import MLX

/// GPU-vectorized binary search equivalent to `numpy.searchsorted`.
///
/// For each value in `values`, finds the insertion index in `sortedArray`
/// such that the array remains sorted.
///
/// - Parameters:
///   - sortedArray: A 1-D sorted array of shape `(n,)`.
///   - values: A 1-D array of values to locate, shape `(m,)`.
/// - Returns: An `MLXArray` of shape `(m,)` with insertion indices.
func searchSorted(
    _ sortedArray: MLXArray,
    values: MLXArray,
    stream: StreamOrDevice = .default
) -> MLXArray {
    let n = sortedArray.dim(0)
    let m = values.dim(0)
    guard n > 0 else { return MLXArray.zeros([m], dtype: .int32) }

    var lo = MLXArray.zeros([m], dtype: .int32)
    var hi = MLXArray.full([m], values: MLXArray(Int32(n)), dtype: .int32)

    let nIterations: Int = {
        let x = Double(Swift.max(n, 2))
        return Int(Darwin.ceil(Darwin.log2(x))) + 1
    }()
    for _ in 0 ..< nIterations {
        let mid = ((lo + hi) / MLXArray(Int32(2))).asType(.int32, stream: stream)
        let midClamped = minimum(mid, MLXArray(Int32(n - 1))).asType(.int32, stream: stream)
        let midVals = sortedArray[midClamped]
        let goRight = midVals .< values
        lo = which(goRight, mid + MLXArray(Int32(1)), lo, stream: stream).asType(.int32, stream: stream)
        hi = which(goRight, hi, mid, stream: stream).asType(.int32, stream: stream)
    }
    // Clamp to valid range [0, n]
    return minimum(lo, MLXArray(Int32(n))).asType(.int32, stream: stream)
}

/// Create a diagonal mask or extract diagonal from a matrix.
///
/// Returns a boolean identity-like matrix of shape `(n, n)`.
func eyeMask(_ n: Int, stream: StreamOrDevice = .default) -> MLXArray {
    let indices = MLXArray(Int32(0) ..< Int32(n))
    let row = indices.expandedDimensions(axis: 1)
    let col = indices.expandedDimensions(axis: 0)
    return row .== col
}
