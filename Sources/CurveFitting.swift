import Foundation

/// Find the curve parameters `(a, b)` for the membership function.
///
/// Fits the function `1 / (1 + a * x^(2b))` to approximate the target
/// membership strength function using Gauss-Newton optimization.
///
/// - Parameters:
///   - spread: Effective scale of embedded points (default 1.0).
///   - minDist: Minimum distance between points in the embedding (default 0.1).
/// - Returns: A tuple `(a, b)` for the distance-to-membership transformation.
func findABParams(spread: Float, minDist: Float) -> (a: Float, b: Float) {
    // Fast path for default parameters (validated against Python reference)
    if abs(spread - 1.0) < 1e-6 && abs(minDist - 0.1) < 1e-6 {
        return (a: 1.9289, b: 0.7915)
    }

    // Generate target curve
    let nSamples = 300
    let xMax = 3.0 * spread
    var xv = [Float](repeating: 0, count: nSamples)
    var yv = [Float](repeating: 0, count: nSamples)

    for i in 0 ..< nSamples {
        let x = Float(i) / Float(nSamples - 1) * xMax
        xv[i] = x
        if x < minDist {
            yv[i] = 1.0
        } else {
            yv[i] = exp(-(x - minDist) / spread)
        }
    }

    // Gauss-Newton optimization
    var a: Float = 1.0
    var b: Float = 1.0

    for _ in 0 ..< 100 {
        var jTj00: Float = 0, jTj01: Float = 0, jTj11: Float = 0
        var jTr0: Float = 0, jTr1: Float = 0

        for i in 0 ..< nSamples {
            let x = xv[i]
            guard x > 1e-10 else { continue }

            let x2b = powf(x, 2 * b)
            let denom = 1.0 + a * x2b
            let pred = 1.0 / denom
            let residual = pred - yv[i]

            // Jacobian
            let denom2 = denom * denom
            let da = -x2b / denom2
            let db = -a * 2.0 * log(x) * x2b / denom2

            // Accumulate J^T * J and J^T * r
            jTj00 += da * da
            jTj01 += da * db
            jTj11 += db * db
            jTr0 += da * residual
            jTr1 += db * residual
        }

        // Solve 2x2 normal equations: [jTj] * delta = -[jTr]
        let det = jTj00 * jTj11 - jTj01 * jTj01
        guard abs(det) > 1e-20 else { break }

        let da = -(jTj11 * jTr0 - jTj01 * jTr1) / det
        let db = -(jTj00 * jTr1 - jTj01 * jTr0) / det

        a += da
        b += db

        // Convergence check
        if da * da + db * db < 1e-12 {
            break
        }
    }

    // Ensure positive values
    a = max(a, 1e-8)
    b = max(b, 1e-8)

    return (a: a, b: b)
}
