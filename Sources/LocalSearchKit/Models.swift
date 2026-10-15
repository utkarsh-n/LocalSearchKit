import DigUpKit
import Foundation

public typealias EmbeddingProvider = DigUpKit.Embedder
public typealias EmbeddingInfo = DigUpKit.WorkerInfo
public typealias EmbeddingResult = DigUpKit.Embedding
public typealias LocalSearchDocumentKind = DigUpKit.FileKind
public typealias LocalSearchSegmentKind = DigUpKit.SegmentKind
public typealias LocalSearchModality = DigUpKit.Modality
/// Return a new provider owned exclusively by this engine. Calls run serially inside its actor.
public typealias LocalSearchEmbedderFactory = @Sendable () throws -> any EmbeddingProvider

/// Complete source identity; revisions are never shortened when deciding compatibility.
public struct LocalSearchModelIdentity: Codable, Hashable, Sendable {
    public let model: String
    public let revision: String
    public let runtime: String
    public let dtype: String
    public let dimension: Int
    /// This runtime's text-only mode also changes its truncation limit (512 rather than 2048).
    public let textOnly: Bool

    public init(model: String, revision: String, runtime: String, dtype: String, dimension: Int = 768,
                textOnly: Bool = false) {
        self.model = model
        self.revision = revision
        self.runtime = runtime
        self.dtype = dtype
        self.dimension = dimension
        self.textOnly = textOnly
    }

    public init(_ info: EmbeddingInfo) {
        self.init(model: info.model, revision: info.revision, runtime: info.runtime, dtype: info.dtype,
                  dimension: info.dim, textOnly: info.textOnly)
    }
}

/// Search task framing is fixed for an index. Use a separate index for code retrieval.
public enum LocalSearchPromptPolicy: String, Codable, Sendable {
    case search, code

    var fingerprint: String {
        "embeddinggemma-prompts-v1;document=title: {title-or-none} | text: {text};query=task: "
            + (self == .code ? "code retrieval" : "search result") + " | query: {query};media=unprompted"
    }

    func query(_ text: String) -> String {
        self == .code ? Prompts.codeQuery(text) : Prompts.query(text)
    }
}

public struct LocalSearchSegment: Sendable {
    public enum Input: Sendable {
        case text(String)
        /// Stored in FTS only; no model is needed.
        case keywords(String)
        /// Precomputed embeddings must identify their complete source and use this index's prompt policy.
        case vector([Float], modality: LocalSearchModality, source: LocalSearchModelIdentity,
                    promptPolicy: LocalSearchPromptPolicy = .search)
        /// Supply a prepared JPEG/PNG supported by the native decoder; convert HEIC/PDF in the host app.
        case image(fileURL: URL, budget: Int)
        /// Caller supplies a prepared audio window; the native model keeps at most its first 30 seconds.
        case audio(fileURL: URL)
    }

    public let input: Input
    public let kind: LocalSearchSegmentKind
    public let keywords: String?
    /// App-defined coordinates, such as ["page": "3"], ["seconds": "12.5"], or ["line": "40"].
    public let locator: [String: String]

    public init(_ input: Input, kind: LocalSearchSegmentKind? = nil, keywords: String? = nil,
                locator: [String: String] = [:]) {
        self.input = input
        if let kind { self.kind = kind }
        else {
            switch input {
            case .image: self.kind = .image
            case .audio: self.kind = .audio
            default: self.kind = .chunk
            }
        }
        self.keywords = keywords
        self.locator = locator
    }

    public static func text(_ text: String, kind: LocalSearchSegmentKind = .chunk,
                            keywords: String? = nil, locator: [String: String] = [:]) -> Self {
        Self(.text(text), kind: kind, keywords: keywords, locator: locator)
    }

    public static func keywords(_ text: String, kind: LocalSearchSegmentKind = .chunk,
                                locator: [String: String] = [:]) -> Self {
        Self(.keywords(text), kind: kind, locator: locator)
    }

    public static func vector(_ values: [Float], modality: LocalSearchModality,
                              source: LocalSearchModelIdentity, kind: LocalSearchSegmentKind = .chunk,
                              promptPolicy: LocalSearchPromptPolicy = .search,
                              keywords: String? = nil, locator: [String: String] = [:]) -> Self {
        Self(.vector(values, modality: modality, source: source, promptPolicy: promptPolicy),
             kind: kind, keywords: keywords, locator: locator)
    }

    public static func image(_ fileURL: URL, budget: Int = 140, kind: LocalSearchSegmentKind = .image,
                             keywords: String? = nil, locator: [String: String] = [:]) -> Self {
        Self(.image(fileURL: fileURL, budget: budget), kind: kind, keywords: keywords, locator: locator)
    }

    public static func audio(_ fileURL: URL, keywords: String? = nil,
                             locator: [String: String] = [:]) -> Self {
        Self(.audio(fileURL: fileURL), kind: .audio, keywords: keywords, locator: locator)
    }
}

/// An app-owned object. Its opaque identifier never becomes a filesystem path or a searchable token.
public struct LocalSearchDocument: Sendable {
    public let id: String
    public let title: String
    public let kind: LocalSearchDocumentKind
    public let segments: [LocalSearchSegment]
    public let metadata: [String: String]

    public init(id: String, title: String, kind: LocalSearchDocumentKind = .doc,
                segments: [LocalSearchSegment], metadata: [String: String] = [:]) {
        self.id = id
        self.title = title
        self.kind = kind
        self.segments = segments
        self.metadata = metadata
    }
}

/// One result per document, pointing to its strongest segment. Scores rank results; they are not probabilities.
public struct LocalSearchHit: Sendable {
    public let id: String
    public let title: String
    public let kind: LocalSearchDocumentKind
    public let metadata: [String: String]
    /// Nil for a title match; otherwise the original zero-based segment position.
    public let segmentIndex: Int?
    public let segmentKind: LocalSearchSegmentKind
    public let locator: [String: String]
    public let excerpt: String?
    public let score: Double
    public let cosine: Double?
    public let semanticZ: Double?
    public let keywordRank: Int?
}

public enum LocalSearchMode: Sendable {
    case hybrid, keywords, semantic
}

public struct LocalSearchStatistics: Sendable {
    public let documents: Int
    public let segments: Int
    public let vectors: Int
    public let model: LocalSearchModelIdentity?
}

public enum LocalSearchError: Error, Sendable, CustomStringConvertible {
    case invalidInput(String)
    case invalidVector(String)
    case embedderUnavailable
    case modelMismatch
    case promptPolicyMismatch
    case corruptIndex(String)
    case modelFileInvalid(String)
    case embeddingFailed(String)

    public var description: String {
        switch self {
        case .invalidInput(let why): "Invalid search input: \(why)"
        case .invalidVector(let why): "Invalid embedding: \(why)"
        case .embedderUnavailable: "Supply an embedding provider for text, media, or semantic queries."
        case .modelMismatch: "Embedding source differs from this index; use a separate index or explicitly clear it."
        case .promptPolicyMismatch: "Prompt policy differs from this index; use a separate index or explicitly clear it."
        case .corruptIndex(let why): "Invalid local search index: \(why)"
        case .modelFileInvalid(let why): "Model file verification failed: \(why)"
        case .embeddingFailed(let why): "Embedding failed: \(why)"
        }
    }
}
