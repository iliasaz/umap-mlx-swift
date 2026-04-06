import MLX

/// GPU-accelerated UMAP dimensionality reduction using MLX on Apple Silicon.
public struct UMAP: Sendable {

    public let configuration: Configuration

    public init(_ configuration: Configuration = .init()) {
        self.configuration = configuration
    }
}

// MARK: - Configuration

extension UMAP {

    /// Parameters controlling the UMAP algorithm.
    public struct Configuration: Sendable {

        /// Dimension of the embedded space.
        public var nComponents: Int = 2

        /// Number of nearest neighbors for the kNN graph.
        public var nNeighbors: Int = 15

        /// Minimum distance between points in the embedding.
        public var minDist: Float = 0.1

        /// Effective scale of embedded points.
        public var spread: Float = 1.0

        /// Number of optimization epochs. `nil` uses an automatic value
        /// (500 if n <= 10 000, otherwise 200).
        public var nEpochs: Int? = nil

        /// SGD learning rate.
        public var learningRate: Float = 1.0

        /// Number of negative samples per positive edge during SGD.
        public var negativeSampleRate: Int = 5

        /// Random seed for reproducibility. `nil` for non-deterministic.
        public var randomSeed: UInt64? = 42

        /// Target dimension for optional PCA preprocessing. `nil` skips PCA.
        public var pcaDimension: Int? = nil

        /// Input normalization strategy.
        public var normalization: Normalization = .none

        /// kNN computation method.
        public var knnMethod: KNNMethod = .auto

        public init() {}
    }

    /// Input normalization strategies.
    public enum Normalization: Sendable {
        /// No normalization.
        case none
        /// Z-score normalization per feature: (x - mean) / std.
        case standard
        /// Min-max scaling per feature to [0, 1].
        case minMax
    }

    /// kNN computation method selection.
    public enum KNNMethod: Sendable {
        /// Brute-force for n <= 20 000, NNDescent otherwise.
        case auto
        /// Exact GPU brute-force pairwise distance computation.
        case bruteForce
        /// Approximate nearest neighbor descent.
        case nnDescent
    }
}

// MARK: - Errors

/// Errors thrown by the UMAP algorithm.
public enum UMAPError: Error, Sendable {
    /// Input array is empty.
    case emptyInput
    /// Shape mismatch between provided arrays.
    case dimensionMismatch(expected: Int, got: Int)
    /// Too few points for the requested number of neighbors.
    case insufficientPoints(n: Int, minRequired: Int)
}
