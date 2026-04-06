import MLX
import MLXRandom
import OSLog

private let logger = Logger(subsystem: "com.newscomb.app", category: "umap")

extension UMAP {

    /// Fit UMAP to the input data and return the low-dimensional embedding.
    ///
    /// This runs the complete UMAP pipeline: preprocessing, kNN computation,
    /// fuzzy simplicial set construction, spectral initialization, and SGD
    /// optimization.
    ///
    /// - Parameters:
    ///   - data: Input array of shape `(n, d)` where `n` is the number of
    ///     samples and `d` is the input dimension.
    ///   - progressCallback: Optional callback reporting
    ///     `(currentEpoch, totalEpochs)` during the SGD phase.
    /// - Returns: An `MLXArray` of shape `(n, nComponents)`.
    public func fitTransform(
        _ data: MLXArray,
        progressCallback: (@Sendable (Int, Int) -> Void)? = nil
    ) throws -> MLXArray {
        let n = data.dim(0)
        let d = data.dim(1)

        guard n >= 2 else {
            throw UMAPError.emptyInput
        }
        guard n > configuration.nNeighbors else {
            throw UMAPError.insufficientPoints(
                n: n,
                minRequired: configuration.nNeighbors + 1
            )
        }

        let nEpochs = configuration.nEpochs ?? (n <= 10_000 ? 500 : 200)

        logger.info(
            "UMAP: n=\(n), d=\(d), nComponents=\(self.configuration.nComponents), nNeighbors=\(self.configuration.nNeighbors), nEpochs=\(nEpochs)"
        )

        // --- Seed ---
        if let seed = configuration.randomSeed {
            MLXRandom.seed(seed)
        }

        // --- Phase 1: Preprocessing ---
        logger.debug("UMAP: Preprocessing")
        var X = normalizeInput(data, method: configuration.normalization)

        if let pcaDim = configuration.pcaDimension, d > pcaDim {
            logger.debug("UMAP: PCA reduction from \(d) to \(pcaDim)")
            X = pcaReduce(X, toDimension: pcaDim)
        }
        eval(X)

        // --- Phase 2: kNN ---
        logger.debug("UMAP: Computing kNN")
        let knn = computeKNN(
            X,
            k: configuration.nNeighbors,
            method: configuration.knnMethod
        )
        logger.debug("UMAP: kNN complete")

        // --- Phase 3: Fuzzy simplicial set ---
        logger.debug("UMAP: Building fuzzy simplicial set")
        let graph = buildFuzzySimplicialSet(
            knnIndices: knn.indices,
            knnDistances: knn.distances,
            n: n,
            nNeighbors: configuration.nNeighbors,
            nEpochs: nEpochs
        )
        logger.debug("UMAP: Fuzzy simplicial set complete")

        // --- Phase 4: Curve fitting ---
        let (a, b) = findABParams(
            spread: configuration.spread,
            minDist: configuration.minDist
        )
        logger.debug("UMAP: Curve params a=\(a), b=\(b)")

        // --- Phase 5: Initialization ---
        logger.debug("UMAP: Spectral initialization")
        var Y: MLXArray
        if let spectral = spectralInit(
            graph: graph,
            n: n,
            nComponents: configuration.nComponents,
            seed: configuration.randomSeed
        ) {
            Y = spectral
            logger.debug("UMAP: Spectral init succeeded")
        } else {
            logger.notice("UMAP: Spectral init failed, using random initialization")
            Y = MLXRandom.normal([n, configuration.nComponents]) * 0.01
            eval(Y)
        }

        // --- Phase 6: SGD optimization ---
        logger.debug("UMAP: Starting SGD optimization (\(nEpochs) epochs)")
        var schedule = buildEdgeSchedule(graph: graph, nEpochs: nEpochs)

        Y = optimizeEmbedding(
            initialEmbedding: Y,
            schedule: &schedule,
            a: a,
            b: b,
            nEpochs: nEpochs,
            learningRate: configuration.learningRate,
            negativeSampleRate: configuration.negativeSampleRate,
            n: n,
            progressCallback: progressCallback
        )

        logger.info("UMAP: Complete")
        return Y
    }

    /// Fit UMAP from pre-computed kNN indices and distances.
    ///
    /// Use when kNN has already been computed externally.
    ///
    /// - Parameters:
    ///   - knnIndices: Neighbor indices of shape `(n, k)`.
    ///   - knnDistances: Neighbor Euclidean distances of shape `(n, k)`.
    ///   - n: Total number of data points.
    ///   - progressCallback: Optional callback reporting
    ///     `(currentEpoch, totalEpochs)`.
    /// - Returns: An `MLXArray` of shape `(n, nComponents)`.
    public func fitTransform(
        knnIndices: MLXArray,
        knnDistances: MLXArray,
        n: Int,
        progressCallback: (@Sendable (Int, Int) -> Void)? = nil
    ) throws -> MLXArray {
        guard n >= 2 else {
            throw UMAPError.emptyInput
        }

        let kI = knnIndices.dim(1)
        let kD = knnDistances.dim(1)
        guard kI == kD else {
            throw UMAPError.dimensionMismatch(expected: kI, got: kD)
        }

        let nEpochs = configuration.nEpochs ?? (n <= 10_000 ? 500 : 200)

        if let seed = configuration.randomSeed {
            MLXRandom.seed(seed)
        }

        let graph = buildFuzzySimplicialSet(
            knnIndices: knnIndices,
            knnDistances: knnDistances,
            n: n,
            nNeighbors: configuration.nNeighbors,
            nEpochs: nEpochs
        )

        let (a, b) = findABParams(
            spread: configuration.spread,
            minDist: configuration.minDist
        )

        var Y: MLXArray
        if let spectral = spectralInit(
            graph: graph,
            n: n,
            nComponents: configuration.nComponents,
            seed: configuration.randomSeed
        ) {
            Y = spectral
        } else {
            Y = MLXRandom.normal([n, configuration.nComponents]) * 0.01
            eval(Y)
        }

        var schedule = buildEdgeSchedule(graph: graph, nEpochs: nEpochs)

        Y = optimizeEmbedding(
            initialEmbedding: Y,
            schedule: &schedule,
            a: a,
            b: b,
            nEpochs: nEpochs,
            learningRate: configuration.learningRate,
            negativeSampleRate: configuration.negativeSampleRate,
            n: n,
            progressCallback: progressCallback
        )

        return Y
    }
}
