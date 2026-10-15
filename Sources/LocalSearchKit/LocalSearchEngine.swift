import DigUpKit
import Foundation
import LlamaRuntime

/// An offline, app-owned index. One actor serializes SQLite and its in-process embedding provider.
/// No crawling, downloads, transcripts, or background jobs are started. Inputs and model paths are explicit.
/// A directory belongs to one engine in one app process; do not share live indexes with app extensions.
public actor LocalSearchEngine {
    public static let databaseFile = "localsearch.sqlite"
    public static let dimension = 768
    public nonisolated let indexDirectory: URL
    public nonisolated let promptPolicy: LocalSearchPromptPolicy
    private let lease: IndexLease
    private let store: IndexStore
    private let searcher: Searcher
    private let embedderFactory: LocalSearchEmbedderFactory?
    private var embedder: OwnedProvider?
    private var dirty = false

    /// Keep one engine alive per index directory. Models load only when an embedding is requested.
    public init(indexDirectory: URL, promptPolicy: LocalSearchPromptPolicy = .search,
                embedderFactory: LocalSearchEmbedderFactory? = nil) throws {
        guard indexDirectory.isFileURL, !indexDirectory.path.contains("\0") else {
            throw LocalSearchError.invalidInput("indexDirectory must be a local file URL without null characters")
        }
        let directory = indexDirectory.standardizedFileURL.resolvingSymlinksInPath()
        let lease = try IndexLease(path: directory.appendingPathComponent(Self.databaseFile).path)
        let store = try IndexStore(directory: directory, file: Self.databaseFile)
        try Self.prepareSchema(store, policy: promptPolicy)
        try Self.validateStoredVectors(store)
        self.indexDirectory = directory
        self.promptPolicy = promptPolicy
        self.lease = lease
        self.store = store
        self.searcher = try Searcher(store: store, loadVectors: false)
        self.embedderFactory = embedderFactory
    }

    /// Verifies official pinned Q8_0 files before loading them. The returned factory never downloads weights.
    /// Without a media URL, native text-only mode truncates each text at 512 tokens; full mode uses 2048.
    /// `diagnostic` optionally receives phase milestones on the engine's executor, without enabling native logs.
    public nonisolated static func nativeFactory(textModelURL: URL, mediaModelURL: URL? = nil,
        diagnostic: (@Sendable (String) -> Void)? = nil)
        -> LocalSearchEmbedderFactory {
        {
            diagnostic?("model-verify-begin")
            do {
                try verifyModelFile(textModelURL, expected: ModelFiles.text)
                if let mediaModelURL { try verifyModelFile(mediaModelURL, expected: ModelFiles.media) }
            } catch {
                diagnostic?("model-verify-failed")
                throw error
            }
            diagnostic?("model-verify-end")
            return try LlamaEmbedder(models: LlamaModels(text: textModelURL, media: mediaModelURL),
                                     textOnly: mediaModelURL == nil, silent: diagnostic == nil, diagnostic: diagnostic)
        }
    }

    /// Inserts or replaces a complete document. Failed embedding or database work leaves its old version intact.
    public func add(_ document: LocalSearchDocument) throws { try add([document]) }

    /// Atomic batch upsert. Text segments are embedded together, then all rows commit in one transaction.
    public func add(_ documents: [LocalSearchDocument]) throws {
        guard !documents.isEmpty else { return }
        var seen = Set<String>()
        for document in documents {
            try Self.checkString(document.id, name: "document id", nonempty: true)
            try Self.checkString(document.title, name: "document title")
            guard seen.insert(document.id).inserted else {
                throw LocalSearchError.invalidInput("a batch must not repeat a document id")
            }
        }
        var identity = try storedIdentity()
        var prepared: [[PreparedSegment]] = documents.map { $0.segments.map { PreparedSegment($0) } }
        var texts: [String] = []
        var textPlaces: [(document: Int, segment: Int)] = []
        for (d, document) in documents.enumerated() {
            for (s, segment) in document.segments.enumerated() {
                guard segment.kind != .name else {
                    throw LocalSearchError.invalidInput("segment kind 'name' is reserved for the document title")
                }
                if let keywords = segment.keywords { try Self.checkString(keywords, name: "keywords") }
                switch segment.input {
                case .keywords(let text):
                    try Self.checkString(text, name: "keywords")
                    prepared[d][s].text = Self.join(text, segment.keywords)
                case .text(let text):
                    try Self.checkString(text, name: "segment text", nonempty: true)
                    prepared[d][s].text = Self.join(text, segment.keywords)
                    prepared[d][s].excerpt = String(text.prefix(400))
                    prepared[d][s].modality = .text
                    texts.append(Prompts.document(title: document.title, text: text))
                    textPlaces.append((d, s))
                case .vector(let values, let modality, let source, let policy):
                    guard policy == promptPolicy else { throw LocalSearchError.promptPolicyMismatch }
                    guard modality != .none else { throw LocalSearchError.invalidVector("a vector needs a modality") }
                    try Self.accept(source, into: &identity)
                    prepared[d][s].vector = try Self.normalized(values)
                    prepared[d][s].modality = modality
                case .image(let url, let budget):
                    try Self.checkMediaURL(url)
                    guard LlamaEmbedder.budgets.contains(budget) else {
                        throw LocalSearchError.invalidInput("image budget must be one of \(LlamaEmbedder.budgets)")
                    }
                    prepared[d][s].modality = .image
                case .audio(let url):
                    try Self.checkMediaURL(url)
                    prepared[d][s].modality = .audio
                }
            }
        }
        if !texts.isEmpty {
            let provider = try provider()
            try Self.accept(LocalSearchModelIdentity(provider.info), into: &identity)
            let vectors = try provider.embedTexts(texts)
            guard vectors.count == texts.count else {
                throw LocalSearchError.embeddingFailed("text provider returned the wrong number of vectors")
            }
            for (place, vector) in zip(textPlaces, vectors) {
                prepared[place.document][place.segment].vector = try Self.normalized(vector)
            }
        }
        // Media stays explicit: the host supplies cropped images, video frames, or prepared audio windows.
        for (d, document) in documents.enumerated() {
            for (s, segment) in document.segments.enumerated() {
                switch segment.input {
                case .image(let url, let budget):
                    let provider = try provider()
                    try Self.accept(LocalSearchModelIdentity(provider.info), into: &identity)
                    prepared[d][s].vector = try Self.mediaVector(provider.embedImages([url.path], budget: budget))
                case .audio(let url):
                    let provider = try provider()
                    try Self.accept(LocalSearchModelIdentity(provider.info), into: &identity)
                    prepared[d][s].vector = try Self.mediaVector(provider.embedAudio([url.path]))
                default: break
                }
            }
        }
        try Task.checkCancellation()
        try store.db.transaction {
            // Check again under SQLite's writer lock, so no foreign process can change the source between checks.
            var current = try storedIdentity()
            if let identity { try Self.accept(identity, into: &current) }
            for (document, segments) in zip(documents, prepared) {
                try Task.checkCancellation()
                if let old = try fileID(document.id) { try store.delete(old) }
                try store.db.run("""
                    INSERT INTO files(path, kind, size, mtime, inode, device, state)
                    VALUES(?, ?, 0, ?, 0, 0, 'done')
                    """, [.text("local-search:" + UUID().uuidString), .text(document.kind.rawValue),
                           .double(Date().timeIntervalSince1970)])
                let file = store.db.lastInsertID
                try store.db.run("INSERT INTO ls_documents(file, opaque_id, title, metadata) VALUES(?, ?, ?, ?)",
                                 [.int(file), .text(document.id), .text(document.title),
                                  .text(try Self.encode(document.metadata))])
                try store.addSegment(file: file, kind: .name, modality: .none, excerpt: document.title,
                                     text: document.title)
                let titleSegment = try lastSegment(of: file)
                try store.db.run("UPDATE fts SET name = ?, body = '' WHERE rowid = ?",
                                 [.text(SearchText.normalized(document.title)), .int(titleSegment)])
                try writeLocator(segment: titleSegment, index: nil, locator: [:])
                for (index, segment) in segments.enumerated() {
                    try Task.checkCancellation()
                    try store.addSegment(file: file, kind: segment.kind, modality: segment.modality,
                                         excerpt: segment.excerpt, text: segment.text, vector: segment.vector)
                    try writeLocator(segment: lastSegment(of: file), index: index, locator: segment.locator)
                }
            }
            if let current {
                let encoded = try Self.encode(current)
                try store.setMeta("ls_model_identity", encoded)
                try store.setMeta("vector_source", encoded)
            }
            try Task.checkCancellation()
        }
        dirty = true
    }

    /// Removes a document and all its vectors, words, and locator metadata atomically.
    @discardableResult
    public func remove(id: String) throws -> Bool { try remove(ids: [id]) > 0 }

    @discardableResult
    public func remove(ids: [String]) throws -> Int {
        for id in ids { try Self.checkString(id, name: "document id", nonempty: true) }
        let removed = try store.db.transaction {
            var count = 0
            for id in Set(ids) {
                try Task.checkCancellation()
                if let file = try fileID(id) { try store.delete(file); count += 1 }
            }
            return count
        }
        if removed > 0 { dirty = true }
        return removed
    }

    /// Explicitly discards this app's entire search index and its source identity, allowing another model.
    public func clear() throws {
        try store.db.transaction {
            try store.db.execute("DELETE FROM fts; DELETE FROM files;")
            try store.setMeta("ls_model_identity", nil)
            try store.setMeta("vector_source", nil)
        }
        dirty = true
    }

    /// The default semantic gate is 0 for fewer than 20 vectors, and DigUp's measured gate of 2 thereafter.
    /// Small-corpus scores offer a ranking, not a calibrated relevance threshold; tune zGate or minimumCosine.
    /// minimumCosine is optional and applies to semantic results or hybrid hits with no keyword match.
    public func search(_ query: String, mode: LocalSearchMode = .hybrid,
                       kinds: Set<LocalSearchDocumentKind>? = nil, limit: Int = 20,
                       zGate: Double? = nil, minimumCosine: Double? = nil,
                       wordWeight: Double? = nil, prefixLast: Bool = false) throws -> [LocalSearchHit] {
        try Self.checkString(query, name: "query")
        guard limit >= 0 else { throw LocalSearchError.invalidInput("limit must not be negative") }
        guard zGate?.isFinite ?? true else { throw LocalSearchError.invalidInput("zGate must be finite") }
        if let minimumCosine, !minimumCosine.isFinite || !(-1...1).contains(minimumCosine) {
            throw LocalSearchError.invalidInput("minimumCosine must be finite and between -1 and 1")
        }
        if let wordWeight, !wordWeight.isFinite || wordWeight < 0 {
            throw LocalSearchError.invalidInput("wordWeight must be finite and nonnegative")
        }
        guard limit > 0, !query.trimmingCharacters(in: .whitespacesAndNewlines).isEmpty else { return [] }
        if dirty { try searcher.reload(); dirty = false }
        var vector: [Float]?
        // Hybrid search over a keyword-only corpus remains usable without loading a model.
        let counts = try statistics()
        if mode == .semantic || (mode == .hybrid && counts.vectors > 0) {
            let provider = try provider()
            let vectors = try provider.embedTexts([promptPolicy.query(query)])
            guard vectors.count == 1 else {
                throw LocalSearchError.embeddingFailed("query provider returned the wrong number of vectors")
            }
            vector = try Self.normalized(vectors[0])
        }
        try Task.checkCancellation()
        let hits = try searcher.search(mode == .semantic ? "" : query, queryVector: vector,
                                       kinds: kinds, limit: minimumCosine == nil ? limit : max(limit, counts.documents),
                                       zGate: zGate ?? (counts.vectors < 20 ? 0 : 2), wordWeight: wordWeight,
                                       prefixLast: prefixLast)
        let accepted = hits.filter { hit in
            guard let minimumCosine, mode == .semantic || hit.keywordRank == nil else { return true }
            return hit.cosine.map { $0 >= minimumCosine } ?? false
        }.prefix(limit)
        return try accepted.map { hit in
            var result: LocalSearchHit?
            try store.db.query("""
                SELECT d.opaque_id, d.title, d.metadata, s.position, s.locator
                FROM ls_documents d JOIN files f ON f.id = d.file
                JOIN ls_segments s ON s.segment = ? WHERE f.path = ?
                """, [.int(hit.segmentID), .text(hit.path)]) { row in
                result = LocalSearchHit(id: row.text(0) ?? "", title: row.text(1) ?? "", kind: hit.kind,
                                        metadata: try Self.decodeMap(row.text(2)),
                                        segmentIndex: row.isNull(3) ? nil : Int(row.int(3)),
                                        segmentKind: hit.segment, locator: try Self.decodeMap(row.text(4)),
                                        excerpt: hit.excerpt, score: hit.score, cosine: hit.cosine,
                                        semanticZ: hit.z, keywordRank: hit.keywordRank)
            }
            guard let result else { throw LocalSearchError.corruptIndex("a result has no app document mapping") }
            return result
        }
    }

    public func statistics() throws -> LocalSearchStatistics {
        var documents = 0, segments = 0, vectors = 0
        try store.db.query("SELECT COUNT(*) FROM ls_documents") { documents = Int($0.int(0)) }
        try store.db.query("SELECT COUNT(*), COUNT(vec) FROM segments WHERE kind != 'name'") {
            segments = Int($0.int(0)); vectors = Int($0.int(1))
        }
        return LocalSearchStatistics(documents: documents, segments: segments, vectors: vectors,
                                     model: try storedIdentity())
    }

    /// Returns memory to the OS while preserving the index. Later requests can load the provider again.
    public func releaseModel() {
        embedder?.close()
        embedder = nil
        searcher.releaseVectors()
    }

    // MARK: Internal invariants

    private func provider() throws -> any EmbeddingProvider {
        if let embedder {
            var expected = try storedIdentity()
            try Self.accept(LocalSearchModelIdentity(embedder.value.info), into: &expected)
            return embedder.value
        }
        guard let embedderFactory else { throw LocalSearchError.embedderUnavailable }
        let provider = try embedderFactory()
        do {
            var expected = try storedIdentity()
            try Self.accept(LocalSearchModelIdentity(provider.info), into: &expected)
        } catch {
            provider.close()
            throw error
        }
        embedder = OwnedProvider(provider)
        return provider
    }

    private func storedIdentity() throws -> LocalSearchModelIdentity? {
        guard let encoded = try store.meta("ls_model_identity") else { return nil }
        guard let data = encoded.data(using: .utf8),
              let identity = try? JSONDecoder().decode(LocalSearchModelIdentity.self, from: data) else {
            throw LocalSearchError.corruptIndex("model identity is unreadable")
        }
        try Self.validate(identity)
        return identity
    }

    private func fileID(_ id: String) throws -> Int64? {
        var result: Int64?
        try store.db.query("SELECT file FROM ls_documents WHERE opaque_id = ?", [.text(id)]) { result = $0.int(0) }
        return result
    }

    private func lastSegment(of file: Int64) throws -> Int64 {
        var result: Int64 = 0
        try store.db.query("SELECT MAX(id) FROM segments WHERE file = ?", [.int(file)]) { result = $0.int(0) }
        guard result > 0 else { throw LocalSearchError.corruptIndex("segment insertion did not create a row") }
        return result
    }

    private func writeLocator(segment: Int64, index: Int?, locator: [String: String]) throws {
        try store.db.run("INSERT INTO ls_segments(segment, position, locator) VALUES(?, ?, ?)",
                         [.int(segment), index.map { .int(Int64($0)) } ?? .null, .text(try Self.encode(locator))])
    }

    private static func prepareSchema(_ store: IndexStore, policy: LocalSearchPromptPolicy) throws {
        try store.db.transaction {
            if let version = try store.meta("ls_schema"), version != "1" {
                throw LocalSearchError.corruptIndex("unsupported wrapper schema \(version)")
            }
            if try store.meta("ls_schema") == nil {
                var files: Int64 = 0
                try store.db.query("SELECT COUNT(*) FROM files") { files = $0.int(0) }
                guard files == 0 else { throw LocalSearchError.corruptIndex("database belongs to another indexer") }
            }
            if let stored = try store.meta("ls_prompt_policy"), stored != policy.fingerprint {
                throw LocalSearchError.promptPolicyMismatch
            }
            try store.db.execute("""
                CREATE TABLE IF NOT EXISTS ls_documents(
                    file INTEGER PRIMARY KEY REFERENCES files(id) ON DELETE CASCADE,
                    opaque_id TEXT NOT NULL UNIQUE, title TEXT NOT NULL, metadata TEXT NOT NULL
                );
                CREATE TABLE IF NOT EXISTS ls_segments(
                    segment INTEGER PRIMARY KEY REFERENCES segments(id) ON DELETE CASCADE,
                    position INTEGER, locator TEXT NOT NULL
                );
                """)
            try store.setMeta("ls_schema", "1")
            try store.setMeta("ls_prompt_policy", policy.fingerprint)
            var missing: Int64 = 0
            try store.db.query("SELECT COUNT(*) FROM files f LEFT JOIN ls_documents d ON d.file=f.id WHERE d.file IS NULL") {
                missing = $0.int(0)
            }
            guard missing == 0 else { throw LocalSearchError.corruptIndex("unmapped documents") }
            try store.db.query("SELECT COUNT(*) FROM segments s LEFT JOIN ls_segments l ON l.segment=s.id WHERE l.segment IS NULL") {
                missing = $0.int(0)
            }
            guard missing == 0 else { throw LocalSearchError.corruptIndex("unmapped segments") }
        }
    }

    /// Fail closed on damaged vectors rather than letting the lower-level searcher silently omit them.
    private static func validateStoredVectors(_ store: IndexStore) throws {
        var found = false
        try store.db.query("SELECT modality, vec FROM segments WHERE vec IS NOT NULL") { row in
            found = true
            guard let blob = row.blob(1), blob.count == dimension * MemoryLayout<Float16>.size,
                  let modality = Modality(rawValue: row.text(0) ?? ""), modality != .none else {
                throw LocalSearchError.corruptIndex("invalid vector size or modality")
            }
            var halves = [Float16](repeating: 0, count: dimension)
            _ = halves.withUnsafeMutableBytes { blob.copyBytes(to: $0) }
            let floats = halves.map(Float.init)
            let squared = floats.reduce(0.0) { $0 + Double($1) * Double($1) }
            guard floats.allSatisfy(\.isFinite), squared > 0.98, squared < 1.02 else {
                throw LocalSearchError.corruptIndex("non-finite, zero, or unnormalized stored vector")
            }
        }
        if found, try store.meta("ls_model_identity") == nil {
            throw LocalSearchError.corruptIndex("vectors have no source identity")
        }
    }

    private static func accept(_ source: LocalSearchModelIdentity, into expected: inout LocalSearchModelIdentity?) throws {
        try validate(source)
        if let expected, expected != source { throw LocalSearchError.modelMismatch }
        expected = source
    }

    private static func validate(_ source: LocalSearchModelIdentity) throws {
        guard source.dimension == dimension else { throw LocalSearchError.invalidVector("expected \(dimension) dimensions") }
        guard [source.model, source.revision, source.runtime, source.dtype].allSatisfy({ !$0.isEmpty }) else {
            throw LocalSearchError.invalidVector("model, full revision, runtime and dtype must be supplied")
        }
    }

    private static func normalized(_ values: [Float]) throws -> [Float] {
        guard values.count == dimension else { throw LocalSearchError.invalidVector("expected \(dimension) dimensions") }
        guard values.allSatisfy(\.isFinite) else { throw LocalSearchError.invalidVector("values must be finite") }
        let squared = values.reduce(0.0) { $0 + Double($1) * Double($1) }
        guard squared.isFinite, squared > 0 else { throw LocalSearchError.invalidVector("vector must be nonzero") }
        let norm = squared.squareRoot()
        return values.map { Float(Double($0) / norm) }
    }

    private static func mediaVector(_ embeddings: [EmbeddingResult]) throws -> [Float] {
        guard embeddings.count == 1 else {
            throw LocalSearchError.embeddingFailed("media provider returned the wrong number of vectors")
        }
        guard embeddings[0].error == nil, let vector = embeddings[0].vector else {
            throw LocalSearchError.embeddingFailed(embeddings[0].error ?? "media has no vector")
        }
        return try normalized(vector)
    }

    private static func verifyModelFile(_ url: URL, expected: ModelFiles.File) throws {
        try checkMediaURL(url)
        let resolved = url.resolvingSymlinksInPath()
        let values = try resolved.resourceValues(forKeys: [.fileSizeKey, .isRegularFileKey])
        guard values.isRegularFile == true, Int64(values.fileSize ?? -1) == expected.bytes else {
            throw LocalSearchError.modelFileInvalid("\(expected.name) has the wrong size or is not a regular file")
        }
        guard try ModelFiles.sha256(of: resolved) == expected.sha256 else {
            throw LocalSearchError.modelFileInvalid("\(expected.name) has the wrong SHA-256")
        }
    }

    private static func checkMediaURL(_ url: URL) throws {
        guard url.isFileURL, !url.path.contains("\0") else {
            throw LocalSearchError.invalidInput("media and model inputs must be local file URLs")
        }
    }

    private static func checkString(_ text: String, name: String, nonempty: Bool = false) throws {
        guard !text.contains("\0"), !nonempty || !text.trimmingCharacters(in: .whitespacesAndNewlines).isEmpty else {
            throw LocalSearchError.invalidInput("\(name) is empty or contains a null character")
        }
    }

    private static func join(_ text: String, _ keywords: String?) -> String {
        [text, keywords ?? ""].filter { !$0.isEmpty }.joined(separator: "\n")
    }

    private static func encode<T: Encodable>(_ value: T) throws -> String {
        let encoder = JSONEncoder()
        encoder.outputFormatting = [.sortedKeys]
        return String(decoding: try encoder.encode(value), as: UTF8.self)
    }

    private static func decodeMap(_ encoded: String?) throws -> [String: String] {
        guard let encoded, let data = encoded.data(using: .utf8),
              let map = try? JSONDecoder().decode([String: String].self, from: data) else {
            throw LocalSearchError.corruptIndex("document or locator metadata is unreadable")
        }
        return map
    }
}

private struct PreparedSegment {
    let kind: LocalSearchSegmentKind
    let locator: [String: String]
    var text: String?
    var excerpt: String?
    var vector: [Float]?
    var modality: LocalSearchModality = .none

    init(_ segment: LocalSearchSegment) {
        kind = segment.kind
        locator = segment.locator
        text = segment.keywords
    }
}

/// Ensures a custom provider receives its required idempotent close call when the actor is released too.
private final class OwnedProvider {
    let value: any EmbeddingProvider
    private var closed = false
    init(_ value: any EmbeddingProvider) { self.value = value }
    func close() {
        guard !closed else { return }
        closed = true
        value.close()
    }
    deinit { close() }
}

/// Two independent actors would each serialize their own calls, but race each other's vector caches.
/// This lease rejects that arrangement in this process and is released automatically with the engine.
private final class IndexLease: @unchecked Sendable {
    private final class Registry: @unchecked Sendable {
        let lock = NSLock()
        var paths = Set<String>()
    }
    private static let registry = Registry()
    private let path: String
    private var acquired = false

    init(path: String) throws {
        self.path = path
        guard Self.registry.lock.withLock({ Self.registry.paths.insert(path).inserted }) else {
            throw LocalSearchError.invalidInput("another LocalSearchEngine already owns this index directory")
        }
        acquired = true
    }

    deinit {
        if acquired { _ = Self.registry.lock.withLock { Self.registry.paths.remove(path) } }
    }
}
