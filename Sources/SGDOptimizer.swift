import MLX
import MLXRandom

/// Pre-computed schedule for which edges are active at each epoch.
struct EdgeSchedule {
    /// Edge source indices, shape `(nEdges,)`.
    let edgeFrom: MLXArray
    /// Edge target indices, shape `(nEdges,)`.
    let edgeTo: MLXArray
    /// Epoch interval between activations for each edge.
    let epochsPerSample: [Float]
    /// Next epoch at which each edge becomes active.
    var epochsPerNext: [Float]
}

/// Build the edge activation schedule from graph weights.
///
/// Edges with higher weight are sampled more frequently during optimization.
///
/// - Parameters:
///   - graph: The sparse weighted graph.
///   - nEpochs: Total number of optimization epochs.
/// - Returns: An `EdgeSchedule` with pre-computed activation timing.
func buildEdgeSchedule(
    graph: SparseGraph,
    nEpochs: Int,
    stream: StreamOrDevice = .default
) -> EdgeSchedule {
    eval(graph.vals)
    let maxWeight = graph.vals.max(stream: stream)
    eval(maxWeight)
    let maxW = maxWeight.item(Float.self)

    // Extract weights to CPU for scheduling
    let nEdges = graph.vals.dim(0)
    let weightsData = graph.vals.asData(access: .copy)
    let weights: [Float] = weightsData.data.withUnsafeBytes { buffer in
        Array(buffer.bindMemory(to: Float.self))
    }

    var epochsPerSample = [Float](repeating: -1, count: nEdges)
    var epochsPerNext = [Float](repeating: -1, count: nEdges)

    let nEpochsF = Float(nEpochs)
    for i in 0 ..< nEdges {
        let nSamples = nEpochsF * (weights[i] / maxW)
        if nSamples > 0 {
            epochsPerSample[i] = nEpochsF / nSamples
            epochsPerNext[i] = epochsPerSample[i]
        }
    }

    return EdgeSchedule(
        edgeFrom: graph.rows,
        edgeTo: graph.cols,
        epochsPerSample: epochsPerSample,
        epochsPerNext: epochsPerNext
    )
}

/// Run the SGD optimization loop.
///
/// Iteratively adjusts the embedding positions using attractive forces
/// along graph edges and repulsive forces from random negative samples.
///
/// - Parameters:
///   - initialEmbedding: Starting positions, shape `(n, nComponents)`.
///   - schedule: Pre-computed edge activation schedule.
///   - a: Curve parameter a.
///   - b: Curve parameter b.
///   - nEpochs: Total number of epochs.
///   - learningRate: Initial learning rate.
///   - negativeSampleRate: Number of negative samples per positive edge.
///   - n: Total number of data points.
///   - progressCallback: Optional callback for `(currentEpoch, totalEpochs)`.
/// - Returns: The optimized embedding of shape `(n, nComponents)`.
func optimizeEmbedding(
    initialEmbedding: MLXArray,
    schedule: inout EdgeSchedule,
    a: Float,
    b: Float,
    nEpochs: Int,
    learningRate: Float,
    negativeSampleRate: Int,
    n: Int,
    progressCallback: (@Sendable (Int, Int) -> Void)?,
    stream: StreamOrDevice = .default
) -> MLXArray {
    var Y = initialEmbedding

    let aMx = MLXArray(a)
    let bMx = MLXArray(b)
    // Build the compiled SGD step
    let sgdStep = compile { (arrays: [MLXArray]) -> [MLXArray] in
        let Y = arrays[0]
        let ef = arrays[1]
        let et = arrays[2]
        let negFrom = arrays[3]
        let negTo = arrays[4]
        let alphaEpoch = arrays[5]
        let aParam = arrays[6]
        let bParam = arrays[7]

        // --- Positive forces (attraction along edges) ---
        let yFrom = Y[ef]
        let yTo = Y[et]
        let diff = yFrom - yTo
        let distSq = maximum(
            diff.square().sum(axis: 1, keepDims: true),
            MLXArray(Float(1e-6))
        )
        let powVal = distSq ** bParam
        let gradCoeff = (-2.0 * aParam * bParam) * (distSq ** (bParam - 1.0))
            / (1.0 + aParam * powVal)
        let posGrad = clip(gradCoeff * diff, min: -4.0, max: 4.0) * alphaEpoch

        // --- Negative forces (repulsion from random pairs) ---
        let yNegFrom = Y[negFrom]
        let yNegTo = Y[negTo]
        let negDiff = yNegFrom - yNegTo
        let negDistSq = maximum(
            negDiff.square().sum(axis: 1, keepDims: true),
            MLXArray(Float(1e-6))
        )
        let negPow = negDistSq ** bParam
        let negGradCoeff = (2.0 * bParam)
            / ((0.001 + negDistSq) * (1.0 + aParam * negPow))
        let negGrad = clip(negGradCoeff * negDiff, min: -4.0, max: 4.0) * alphaEpoch

        // --- Scatter updates ---
        var result = Y
        result = result.at[ef].add(posGrad)
        result = result.at[et].add(-posGrad)
        result = result.at[negFrom].add(negGrad)

        return [result]
    }

    let nEpochsF = Float(nEpochs)

    for epoch in 0 ..< nEpochs {
        // Find active edges this epoch
        let epochF = Float(epoch)
        var activeIndices = [Int32]()
        for i in 0 ..< schedule.epochsPerNext.count {
            if schedule.epochsPerNext[i] >= 0 && schedule.epochsPerNext[i] <= epochF {
                activeIndices.append(Int32(i))
                schedule.epochsPerNext[i] += schedule.epochsPerSample[i]
            }
        }

        guard !activeIndices.isEmpty else { continue }

        let activeIdx = MLXArray(activeIndices)
        let ef = takeAlong(schedule.edgeFrom, activeIdx, stream: stream)
        let et = takeAlong(schedule.edgeTo, activeIdx, stream: stream)

        let alphaEpoch = MLXArray(learningRate * (1.0 - epochF / nEpochsF))

        // Negative sampling
        let nActive = activeIndices.count
        let nNeg = negativeSampleRate * nActive
        let negFromSource = ef[MLXArray(Int32(0) ..< Int32(nNeg)) % MLXArray(Int32(nActive))]
        let negTo = MLXRandom.randInt(low: 0, high: Int32(n), [nNeg], type: Int32.self)

        let result = sgdStep([Y, ef, et, negFromSource, negTo, alphaEpoch, aMx, bMx])
        Y = result[0]

        // Periodic evaluation to prevent graph accumulation
        if (epoch + 1) % 10 == 0 || epoch == nEpochs - 1 {
            eval(Y)
        }

        progressCallback?(epoch + 1, nEpochs)
    }

    eval(Y)
    return Y
}
