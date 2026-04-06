import XCTest
@testable import UMAPMLX
import MLX
import MLXRandom

/// Thread-safe progress counter for Sendable callback.
final class ProgressCounter: @unchecked Sendable {
    private let lock = NSLock()
    private var _lastEpoch = 0
    var lastEpoch: Int {
        lock.lock()
        defer { lock.unlock() }
        return _lastEpoch
    }
    func update(epoch: Int) {
        lock.lock()
        _lastEpoch = epoch
        lock.unlock()
    }
}

final class UMAPMLXTests: XCTestCase {

    // MARK: - Configuration

    func testConfigurationDefaults() {
        let config = UMAP.Configuration()
        XCTAssertEqual(config.nComponents, 2)
        XCTAssertEqual(config.nNeighbors, 15)
        XCTAssertEqual(config.minDist, 0.1)
        XCTAssertEqual(config.spread, 1.0)
        XCTAssertNil(config.nEpochs)
        XCTAssertEqual(config.learningRate, 1.0)
        XCTAssertEqual(config.negativeSampleRate, 5)
        XCTAssertEqual(config.randomSeed, 42)
    }

    // MARK: - Curve Fitting

    func testCurveFittingDefaultParams() {
        let (a, b) = findABParams(spread: 1.0, minDist: 0.1)
        XCTAssertEqual(Double(a), 1.9289, accuracy: 0.01)
        XCTAssertEqual(Double(b), 0.7915, accuracy: 0.01)
    }

    func testCurveFittingCustomParams() {
        let (a, b) = findABParams(spread: 1.0, minDist: 0.5)
        XCTAssertGreaterThan(a, 0)
        XCTAssertGreaterThan(b, 0)
    }

    func testCurveFittingProducesValidFunction() {
        let (a, b) = findABParams(spread: 1.0, minDist: 0.3)
        // At distance ~0, membership ~1.0
        let atZero: Float = 1.0 / (1.0 + a * powf(0.001, 2 * b))
        XCTAssertGreaterThan(atZero, 0.5)

        // At large distance, membership ~0
        let atFar: Float = 1.0 / (1.0 + a * powf(5.0, 2 * b))
        XCTAssertLessThan(atFar, 0.1)
    }

    // MARK: - Preprocessing

    func testStandardNormalization() {
        MLXRandom.seed(42)
        let data = MLXRandom.normal([100, 5]) * 10.0 + 50.0
        let normalized = normalizeInput(data, method: .standard)
        eval(normalized)

        let mean = normalized.mean(axis: 0)
        eval(mean)

        for i in 0 ..< 5 {
            let val = mean[i]
            eval(val)
            XCTAssertEqual(Double(val.item(Float.self)), 0.0, accuracy: 0.15)
        }
    }

    func testMinMaxNormalization() {
        MLXRandom.seed(42)
        let data = MLXRandom.normal([50, 3]) * 100.0 - 30.0
        let normalized = normalizeInput(data, method: .minMax)
        eval(normalized)

        let minVal = normalized.min()
        let maxVal = normalized.max()
        eval(minVal, maxVal)

        XCTAssertGreaterThanOrEqual(Double(minVal.item(Float.self)), -0.01)
        XCTAssertLessThanOrEqual(Double(maxVal.item(Float.self)), 1.01)
    }

    func testNoNormalization() {
        let values: [Float] = [1.0, 2.0, 3.0, 4.0]
        let data = MLXArray(values, [2, 2])
        let result = normalizeInput(data, method: .none)
        eval(result)
        let diff = MLX.abs(data - result).sum()
        eval(diff)
        XCTAssertEqual(Double(diff.item(Float.self)), 0.0, accuracy: 1e-6)
    }

    func testPCAReduce() {
        MLXRandom.seed(42)
        let data = MLXRandom.normal([50, 10])
        eval(data)

        let reduced = pcaReduce(data, toDimension: 3)
        eval(reduced)

        XCTAssertEqual(reduced.dim(0), 50)
        XCTAssertEqual(reduced.dim(1), 3)
    }

    func testPCASkipsWhenDimensionSufficient() {
        MLXRandom.seed(42)
        let data = MLXRandom.normal([20, 5])
        eval(data)

        let result = pcaReduce(data, toDimension: 10)
        eval(result)
        XCTAssertEqual(result.dim(1), 5)
    }

    // MARK: - kNN

    func testBruteForceKNNSmall() {
        // 4 points in 2D: corners of unit square
        let values: [Float] = [0, 0, 1, 0, 0, 1, 1, 1]
        let data = MLXArray(values, [4, 2])

        let (indices, distances) = computeBruteForceKNN(data, k: 2)
        eval(indices, distances)

        XCTAssertEqual(indices.dim(0), 4)
        XCTAssertEqual(indices.dim(1), 2)

        // All nearest-neighbor distances should be 1.0 (adjacent corners)
        let firstNNDist = distances[0..., 0]
        eval(firstNNDist)
        for i in 0 ..< 4 {
            let d = firstNNDist[i]
            eval(d)
            XCTAssertEqual(Double(d.item(Float.self)), 1.0, accuracy: 0.01)
        }
    }

    func testKNNDistancesNonNegative() {
        MLXRandom.seed(42)
        let data = MLXRandom.normal([30, 5])
        let (_, distances) = computeBruteForceKNN(data, k: 5)
        eval(distances)

        let minDist = distances.min()
        eval(minDist)
        XCTAssertGreaterThanOrEqual(Double(minDist.item(Float.self)), 0.0)
    }

    // MARK: - searchSorted

    func testSearchSorted() {
        let sorted = MLXArray([Float(1), 3, 5, 7, 9])
        let values = MLXArray([Float(0), 2, 5, 6, 10])

        let result = searchSorted(sorted, values: values)
        eval(result)

        let expected: [Int32] = [0, 1, 2, 3, 5]
        for (i, exp) in expected.enumerated() {
            let val = result[i]
            eval(val)
            XCTAssertEqual(val.item(Int32.self), exp, "Index \(i)")
        }
    }

    // MARK: - Fuzzy Simplicial Set

    func testFuzzySimplicialSetProducesEdges() {
        MLXRandom.seed(42)
        let data = MLXRandom.normal([30, 5])
        let (indices, distances) = computeBruteForceKNN(data, k: 5)
        eval(indices, distances)

        let graph = buildFuzzySimplicialSet(
            knnIndices: indices,
            knnDistances: distances,
            n: 30,
            nNeighbors: 5,
            nEpochs: 200
        )
        eval(graph.rows, graph.cols, graph.vals)

        XCTAssertGreaterThan(graph.rows.dim(0), 0)

        let minWeight = graph.vals.min()
        eval(minWeight)
        XCTAssertGreaterThan(Double(minWeight.item(Float.self)), 0)
    }

    // MARK: - End-to-End UMAP

    func testUMAPOutputShape() throws {
        MLXRandom.seed(42)
        let data = MLXRandom.normal([30, 10])
        eval(data)

        var config = UMAP.Configuration()
        config.nComponents = 2
        config.nNeighbors = 5
        config.nEpochs = 50
        config.randomSeed = 42

        let umap = UMAP(config)
        let embedding = try umap.fitTransform(data)
        eval(embedding)

        XCTAssertEqual(embedding.dim(0), 30)
        XCTAssertEqual(embedding.dim(1), 2)
    }

    func testUMAPClusterSeparation() throws {
        MLXRandom.seed(42)
        let offset1 = MLXArray([Float(5), 5, 0, 0, 0], [1, 5])
        let offset2 = MLXArray([Float(0), 0, 5, 5, 0], [1, 5])
        let cluster1 = MLXRandom.normal([20, 5]) + offset1
        let cluster2 = MLXRandom.normal([20, 5]) + offset2
        let data = concatenated([cluster1, cluster2], axis: 0)
        eval(data)

        var config = UMAP.Configuration()
        config.nComponents = 2
        config.nNeighbors = 5
        config.nEpochs = 100
        config.randomSeed = 42

        let umap = UMAP(config)
        let embedding = try umap.fitTransform(data)
        eval(embedding)

        let meanA = embedding[0 ..< 20].mean(axis: 0)
        let meanB = embedding[20 ..< 40].mean(axis: 0)
        let diff = meanA - meanB
        let centroidDist = (diff * diff).sum().sqrt()
        eval(centroidDist)

        XCTAssertGreaterThan(Double(centroidDist.item(Float.self)), 0.1)
    }

    func testUMAPThrowsForInsufficientPoints() {
        MLXRandom.seed(42)
        let data = MLXRandom.normal([3, 5])
        eval(data)

        var config = UMAP.Configuration()
        config.nNeighbors = 15

        let umap = UMAP(config)
        XCTAssertThrowsError(try umap.fitTransform(data)) { error in
            guard case UMAPError.insufficientPoints = error else {
                XCTFail("Expected insufficientPoints error, got \(error)")
                return
            }
        }
    }

    func testUMAPThreeComponents() throws {
        MLXRandom.seed(42)
        let data = MLXRandom.normal([25, 8])
        eval(data)

        var config = UMAP.Configuration()
        config.nComponents = 3
        config.nNeighbors = 5
        config.nEpochs = 30
        config.randomSeed = 42

        let umap = UMAP(config)
        let embedding = try umap.fitTransform(data)
        eval(embedding)

        XCTAssertEqual(embedding.dim(0), 25)
        XCTAssertEqual(embedding.dim(1), 3)
    }

    func testUMAPWithNormalization() throws {
        MLXRandom.seed(42)
        let data = concatenated([
            MLXRandom.normal([25, 1]) * 1000.0,
            MLXRandom.normal([25, 1]) * 0.001,
            MLXRandom.normal([25, 1]),
        ], axis: 1)
        eval(data)

        var config = UMAP.Configuration()
        config.nNeighbors = 5
        config.nEpochs = 30
        config.normalization = .standard
        config.randomSeed = 42

        let umap = UMAP(config)
        let embedding = try umap.fitTransform(data)
        eval(embedding)

        XCTAssertEqual(embedding.dim(0), 25)
        XCTAssertEqual(embedding.dim(1), 2)

        let hasNaN = isNaN(embedding).any()
        eval(hasNaN)
        XCTAssertFalse(hasNaN.item(Bool.self))
    }

    func testUMAPNoNaN() throws {
        MLXRandom.seed(42)
        let data = MLXRandom.normal([30, 5])
        eval(data)

        var config = UMAP.Configuration()
        config.nNeighbors = 5
        config.nEpochs = 50
        config.randomSeed = 42

        let umap = UMAP(config)
        let embedding = try umap.fitTransform(data)
        eval(embedding)

        let hasNaN = isNaN(embedding).any()
        eval(hasNaN)
        XCTAssertFalse(hasNaN.item(Bool.self))
    }

    // MARK: - Holistic End-to-End

    /// Holistic UMAP test with realistic-scale data: 5 well-separated Gaussian
    /// clusters in 50D, 200 points each (1000 total). Validates that UMAP
    /// preserves cluster structure, maintains local neighborhood relationships,
    /// and produces a numerically well-behaved embedding.
    func testUMAPHolisticEndToEnd() throws {
        MLXRandom.seed(7)

        // --- 1. Generate 5 well-separated clusters in 50D ---
        let nPerCluster = 200
        let nClusters = 5
        let inputDim = 50
        let n = nPerCluster * nClusters

        // Place cluster centroids far apart along different axes
        var clusters = [MLXArray]()
        for c in 0 ..< nClusters {
            // Centroid: 10.0 on a unique pair of dimensions, 0 elsewhere
            var centroidValues = [Float](repeating: 0, count: inputDim)
            centroidValues[c * 2] = 10.0
            centroidValues[c * 2 + 1] = 10.0
            let centroid = MLXArray(centroidValues, [1, inputDim])
            let points = MLXRandom.normal([nPerCluster, inputDim]) * 0.5 + centroid
            clusters.append(points)
        }
        let data = concatenated(clusters, axis: 0)
        eval(data)
        XCTAssertEqual(data.dim(0), n)
        XCTAssertEqual(data.dim(1), inputDim)

        // --- 2. Run UMAP with production-like settings ---
        var config = UMAP.Configuration()
        config.nComponents = 2
        config.nNeighbors = 15
        config.minDist = 0.1
        config.nEpochs = 200
        config.randomSeed = 7

        let umap = UMAP(config)

        // Track progress
        let progressCounter = ProgressCounter()
        let embedding = try umap.fitTransform(data) { epoch, total in
            progressCounter.update(epoch: epoch)
        }
        eval(embedding)

        // --- 3. Basic shape and sanity ---
        XCTAssertEqual(embedding.dim(0), n, "Output should have one row per input point")
        XCTAssertEqual(embedding.dim(1), 2, "Output should have nComponents columns")
        XCTAssertEqual(progressCounter.lastEpoch, 200, "Progress should report up to nEpochs")

        let hasNaN = isNaN(embedding).any()
        eval(hasNaN)
        XCTAssertFalse(hasNaN.item(Bool.self), "Embedding must not contain NaN")

        let hasInf = isInf(embedding).any()
        eval(hasInf)
        XCTAssertFalse(hasInf.item(Bool.self), "Embedding must not contain Inf")

        // --- 4. Cluster separation: centroids in 2D should be far apart ---
        var centroids2D = [MLXArray]()
        for c in 0 ..< nClusters {
            let start = c * nPerCluster
            let end = start + nPerCluster
            let clusterEmb = embedding[start ..< end]
            centroids2D.append(clusterEmb.mean(axis: 0))
        }

        // Check all pairs of cluster centroids are separated
        var minInterCluster: Float = .greatestFiniteMagnitude
        for i in 0 ..< nClusters {
            for j in (i + 1) ..< nClusters {
                let diff = centroids2D[i] - centroids2D[j]
                let dist = (diff * diff).sum().sqrt()
                eval(dist)
                let d = dist.item(Float.self)
                minInterCluster = Swift.min(minInterCluster, d)
            }
        }
        XCTAssertGreaterThan(
            minInterCluster, 0.5,
            "All cluster centroids should be meaningfully separated (got \(minInterCluster))"
        )

        // --- 5. Intra-cluster compactness: points within a cluster should be tighter
        //         than the inter-cluster gap ---
        var maxIntraCluster: Float = 0
        for c in 0 ..< nClusters {
            let start = c * nPerCluster
            let end = start + nPerCluster
            let clusterEmb = embedding[start ..< end]
            let centroid = centroids2D[c].expandedDimensions(axis: 0)
            let diffs = clusterEmb - centroid
            let dists = (diffs * diffs).sum(axis: 1).sqrt()
            let avgDist = dists.mean()
            eval(avgDist)
            maxIntraCluster = Swift.max(maxIntraCluster, avgDist.item(Float.self))
        }
        XCTAssertLessThan(
            maxIntraCluster, minInterCluster,
            "Intra-cluster spread (\(maxIntraCluster)) should be less than inter-cluster gap (\(minInterCluster))"
        )

        // --- 6. Neighborhood preservation (trustworthiness proxy) ---
        // For a random sample of points, check that their kNN in the input
        // space overlaps with their kNN in the embedding space.
        let sampleSize = 50
        let k = 10
        let sampleIdx = MLXRandom.randInt(low: 0, high: Int32(n), [sampleSize], type: Int32.self)
        eval(sampleIdx)

        var totalOverlap = 0
        for s in 0 ..< sampleSize {
            let idx = sampleIdx[s]
            eval(idx)
            let i = Int(idx.item(Int32.self))

            // kNN in input space
            let inputPoint = data[i].expandedDimensions(axis: 0)
            let inputDiffs = data - inputPoint
            let inputDists = (inputDiffs * inputDiffs).sum(axis: 1)
            eval(inputDists)

            let inputSorted = argSort(inputDists).asType(.int32)
            eval(inputSorted)
            // Skip self (index 0), take next k
            var inputNN = Set<Int32>()
            for j in 1 ... k {
                let neighbor = inputSorted[j]
                eval(neighbor)
                inputNN.insert(neighbor.item(Int32.self))
            }

            // kNN in embedding space
            let embPoint = embedding[i].expandedDimensions(axis: 0)
            let embDiffs = embedding - embPoint
            let embDists = (embDiffs * embDiffs).sum(axis: 1)
            eval(embDists)

            let embSorted = argSort(embDists).asType(.int32)
            eval(embSorted)
            var embNN = Set<Int32>()
            for j in 1 ... k {
                let neighbor = embSorted[j]
                eval(neighbor)
                embNN.insert(neighbor.item(Int32.self))
            }

            totalOverlap += inputNN.intersection(embNN).count
        }

        let avgOverlap = Float(totalOverlap) / Float(sampleSize * k)
        // UMAP preserves topological structure, not exact neighbor rankings.
        // Within well-separated clusters, at least some neighbors should overlap.
        // Expect at least 10% overlap — meaning the embedding is not random.
        XCTAssertGreaterThan(
            avgOverlap, 0.10,
            "Neighborhood preservation should be at least 10% (got \(avgOverlap * 100)%)"
        )
        // Also verify it's not trivially 100% (would indicate degenerate embedding)
        XCTAssertLessThan(
            avgOverlap, 0.99,
            "Neighborhood overlap should not be trivially perfect"
        )

        // --- 7. Cluster label purity via nearest-centroid assignment ---
        // Assign each point to the nearest 2D centroid and check it matches
        // its true cluster label.
        var correctAssignments = 0
        for i in 0 ..< n {
            let point = embedding[i]
            eval(point)

            var bestCluster = -1
            var bestDist: Float = .greatestFiniteMagnitude
            for c in 0 ..< nClusters {
                let diff = point - centroids2D[c]
                let dist = (diff * diff).sum()
                eval(dist)
                let d = dist.item(Float.self)
                if d < bestDist {
                    bestDist = d
                    bestCluster = c
                }
            }

            let trueCluster = i / nPerCluster
            if bestCluster == trueCluster {
                correctAssignments += 1
            }
        }

        let purity = Float(correctAssignments) / Float(n)
        XCTAssertGreaterThan(
            purity, 0.85,
            "Nearest-centroid cluster purity should exceed 85% (got \(purity * 100)%)"
        )

        // --- 8. Embedding spread: should use a reasonable range, not collapse ---
        let embStd = embedding.variance(axis: 0).sqrt()
        eval(embStd)
        for c in 0 ..< 2 {
            let std = embStd[c]
            eval(std)
            let s = std.item(Float.self)
            XCTAssertGreaterThan(s, 0.1, "Embedding dimension \(c) should have non-trivial spread")
        }
    }
}
