import DigUpKit
import Foundation
import LocalSearchKit
import Testing

@Suite(.serialized)
struct LocalSearchEngineTests {
    @Test func keywordIndexIsLazyAndPreservesOpaqueIDsAndLocators() async throws {
        let folder = TemporaryFolder()
        let calls = ProviderCalls()
        let engine = try LocalSearchEngine(indexDirectory: folder.url, embedderFactory: fakeFactory(calls))
        try await engine.add(LocalSearchDocument(id: "object://lesson/42", title: "Electronics lesson", kind: .pdf,
                                                 segments: [.keywords("capacitance voltage", locator: ["page": "7", "anchor": "charge"])],
                                                 metadata: ["collection": "circuits"]))
        let hits = try await engine.search("capacitance")
        try #require(hits.count == 1)
        #expect(hits[0].id == "object://lesson/42")
        #expect(hits[0].segmentIndex == 0)
        #expect(hits[0].locator == ["page": "7", "anchor": "charge"])
        #expect(hits[0].metadata == ["collection": "circuits"])
        #expect(calls.loads == 0)
        #expect(try await engine.search("object lesson 42", mode: .keywords).isEmpty)
        #expect(try await engine.search("Electronics", mode: .keywords).first?.segmentIndex == nil)
    }

    @Test func hybridSearchBatchesTextAndFindsParaphrasesInATinyCorpus() async throws {
        let folder = TemporaryFolder()
        let calls = ProviderCalls()
        let engine = try LocalSearchEngine(indexDirectory: folder.url, embedderFactory: fakeFactory(calls))
        try await engine.add(threeDocuments())
        #expect(calls.loads == 1)
        try #require(calls.textBatches.count == 1)
        try #require(calls.textBatches[0].count == 3)
        #expect(calls.textBatches[0][0].hasPrefix("title: Circuit | text: "))
        let hit = try await engine.search("electrical energy storage", mode: .semantic)
        #expect(hit.first?.id == "circuit")
        #expect(hit.first?.cosine == 1)
        #expect(hit.first?.locator == ["page": "2"])
        #expect(calls.textBatches.last == ["task: search result | query: electrical energy storage"])
        // DigUp's default z=2 can reject every semantic-only result in a three-vector corpus.
        #expect(try await engine.search("electrical energy storage", mode: .semantic, zGate: 2).isEmpty)
        #expect(try await engine.search("electrical energy storage", kinds: [.audio]).isEmpty)
    }

    @Test func persistenceAfterReleaseUpdateAndRemove() async throws {
        let folder = TemporaryFolder()
        let calls = ProviderCalls()
        var first: LocalSearchEngine? = try LocalSearchEngine(indexDirectory: folder.url, embedderFactory: fakeFactory(calls))
        try await first!.add(threeDocuments())
        await first!.releaseModel()
        #expect(calls.closes == 1)
        first = nil
        let engine = try LocalSearchEngine(indexDirectory: folder.url, embedderFactory: fakeFactory(calls))
        #expect(try await engine.statistics().documents == 3)
        #expect(try await engine.search("electrical energy storage", mode: .semantic).first?.id == "circuit")
        try await engine.add(LocalSearchDocument(id: "circuit", title: "Replacement", segments: [.keywords("impedance reactance")]))
        #expect(try await engine.statistics().documents == 3)
        #expect(try await engine.statistics().vectors == 2)
        #expect(try await engine.search("impedance", mode: .keywords).first?.title == "Replacement")
        #expect(try await engine.search("capacitor", mode: .keywords).isEmpty)
        #expect(try await engine.remove(id: "circuit"))
        #expect(try await !engine.remove(id: "circuit"))
        #expect(try await engine.statistics().documents == 2)
        #expect(try await engine.search("impedance", mode: .keywords).isEmpty)
    }

    @Test func malformedVectorsAndReservedKindsAreRejectedWithoutMutation() async throws {
        let folder = TemporaryFolder()
        let engine = try LocalSearchEngine(indexDirectory: folder.url)
        let bad: [[Float]] = [[1, 2], Array(repeating: 0, count: 768),
                              [.nan] + Array(repeating: 0, count: 767),
                              [.infinity] + Array(repeating: 0, count: 767)]
        for vector in bad {
            await #expect(throws: LocalSearchError.self) {
                try await engine.add(LocalSearchDocument(id: "bad", title: "Invalid",
                    segments: [.vector(vector, modality: .text, source: fakeIdentity)]))
            }
        }
        await #expect(throws: LocalSearchError.self) {
            try await engine.add(LocalSearchDocument(id: "bad", title: "Invalid", segments: [.keywords("bad", kind: .name)]))
        }
        #expect(try await engine.statistics().documents == 0)
        #expect(try await engine.statistics().model == nil)
        try await engine.add(LocalSearchDocument(id: "valid", title: "Valid",
            segments: [.vector(basis(0, magnitude: 9), modality: .text, source: fakeIdentity, keywords: "charge")]))
        let db = try Database(path: folder.url.appendingPathComponent(LocalSearchEngine.databaseFile).path)
        var normalized: Float16 = 0
        try db.query("SELECT vec FROM segments WHERE vec IS NOT NULL") {
            let bytes = try #require($0.blob(0))
            var halves = [Float16](repeating: 0, count: 768)
            _ = halves.withUnsafeMutableBytes { bytes.copyBytes(to: $0) }
            normalized = halves[0]
        }
        #expect(normalized == 1)
    }

    @Test func fullRevisionRuntimeDtypeAndTextModeCannotMix() async throws {
        let folder = TemporaryFolder()
        let engine = try LocalSearchEngine(indexDirectory: folder.url)
        let original = LocalSearchModelIdentity(model: "fake", revision: "1234567-original", runtime: "reference",
                                                dtype: "f32")
        try await engine.add(LocalSearchDocument(id: "first", title: "First",
            segments: [.vector(basis(0), modality: .text, source: original)]))
        let alternatives = [
            LocalSearchModelIdentity(model: "fake", revision: "1234567-other", runtime: "reference", dtype: "f32"),
            LocalSearchModelIdentity(model: "other", revision: original.revision, runtime: "reference", dtype: "f32"),
            LocalSearchModelIdentity(model: "fake", revision: original.revision, runtime: "other", dtype: "f32"),
            LocalSearchModelIdentity(model: "fake", revision: original.revision, runtime: "reference", dtype: "f16"),
            LocalSearchModelIdentity(model: "fake", revision: original.revision, runtime: "reference", dtype: "f32", textOnly: true),
        ]
        for source in alternatives {
            await #expect(throws: LocalSearchError.self) {
                try await engine.add(LocalSearchDocument(id: "other", title: "Other",
                    segments: [.vector(basis(1), modality: .text, source: source)]))
            }
        }
        await #expect(throws: LocalSearchError.self) {
            try await engine.add(LocalSearchDocument(id: "other", title: "Other",
                segments: [.vector(basis(1), modality: .text, source: original, promptPolicy: .code)]))
        }
        #expect(try await engine.statistics().documents == 1)
        #expect(try await engine.statistics().model == original)
    }

    @Test func failedEmbeddingAndSQLiteInsertRollBackWholeBatch() async throws {
        let folder = TemporaryFolder()
        let engine = try LocalSearchEngine(indexDirectory: folder.url, embedderFactory: fakeFactory(ProviderCalls()))
        try await engine.add(LocalSearchDocument(id: "keep", title: "Original", segments: [.keywords("retained")]))
        await #expect(throws: TestFailure.self) {
            try await engine.add(LocalSearchDocument(id: "keep", title: "Bad", segments: [.text("FAIL")]))
        }
        let db = try Database(path: folder.url.appendingPathComponent(LocalSearchEngine.databaseFile).path)
        try db.execute("""
            CREATE TRIGGER reject_test BEFORE INSERT ON ls_documents WHEN NEW.opaque_id = 'reject'
            BEGIN SELECT RAISE(ABORT, 'forced transaction failure'); END;
            """)
        await #expect(throws: (any Error).self) {
            try await engine.add([
                LocalSearchDocument(id: "keep", title: "Changed", segments: [.text("capacitor")]),
                LocalSearchDocument(id: "reject", title: "Rejected", segments: [.keywords("discarded")]),
            ])
        }
        #expect(try await engine.search("retained", mode: .keywords).first?.title == "Original")
        #expect(try await engine.search("capacitor", mode: .keywords).isEmpty)
        #expect(try await engine.statistics().documents == 1)
        #expect(try await engine.statistics().vectors == 0)
        #expect(try await engine.statistics().model == nil)
    }

    @Test func ftsOperatorsAreQuotedAndNullCharactersCannotAliasIdentifiers() async throws {
        let folder = TemporaryFolder()
        let engine = try LocalSearchEngine(indexDirectory: folder.url)
        try await engine.add([
            LocalSearchDocument(id: "known", title: "Circuit", segments: [.keywords("capacitor")]),
            LocalSearchDocument(id: "other", title: "Garden", segments: [.keywords("photosynthesis")]),
        ])
        // OR, column syntax, quotes, and SQL text cannot act as FTS operators or statements.
        #expect(try await engine.search("\"capacitor\" OR name:*", mode: .keywords).map(\.id) == ["known"])
        #expect(try await engine.search("capacitor'); DELETE FROM files; --", mode: .keywords).map(\.id) == ["known"])
        #expect(try await engine.statistics().documents == 2)
        await #expect(throws: LocalSearchError.self) { try await engine.remove(id: "known\0suffix") }
        await #expect(throws: LocalSearchError.self) {
            try await engine.add(LocalSearchDocument(id: "known\0suffix", title: "Aliased", segments: []))
        }
        await #expect(throws: LocalSearchError.self) { try await engine.search("capacitor\0suffix") }
        #expect(try await engine.statistics().documents == 2)
    }

    @Test func mediaInputsKeepAppCoordinatesAndReportFailureAtomically() async throws {
        let folder = TemporaryFolder()
        let calls = ProviderCalls()
        let engine = try LocalSearchEngine(indexDirectory: folder.url, embedderFactory: fakeFactory(calls))
        try await engine.add([
            LocalSearchDocument(id: "frame", title: "Video", kind: .video,
                segments: [.image(folder.url.appendingPathComponent("frame.png"), budget: 70, kind: .frame,
                                  keywords: "oscilloscope", locator: ["seconds": "12.5", "endSeconds": "15"])]),
            LocalSearchDocument(id: "audio", title: "Music", kind: .audio,
                segments: [.audio(folder.url.appendingPathComponent("clip.wav"), keywords: "overtone",
                                  locator: ["seconds": "25", "endSeconds": "55"])]),
        ])
        #expect(calls.images.count == 1)
        #expect(calls.images.first?.0 == "frame.png")
        #expect(calls.images.first?.1 == 70)
        #expect(calls.audio == ["clip.wav"])
        #expect(try await engine.search("oscilloscope", mode: .keywords).first?.locator["seconds"] == "12.5")
        #expect(try await engine.search("overtone", mode: .keywords).first?.id == "audio")
        await #expect(throws: LocalSearchError.self) {
            try await engine.add(LocalSearchDocument(id: "audio", title: "Failed",
                segments: [.audio(folder.url.appendingPathComponent("FAIL.wav"))]))
        }
        #expect(try await engine.search("overtone", mode: .keywords).first?.title == "Music")
        #expect(try await engine.statistics().vectors == 2)
    }

    @Test func singletonRankingNeedsExplicitRelevanceThreshold() async throws {
        let folder = TemporaryFolder()
        let engine = try LocalSearchEngine(indexDirectory: folder.url, embedderFactory: fakeFactory(ProviderCalls()))
        try await engine.add(LocalSearchDocument(id: "circuit", title: "Circuit", segments: [.text("capacitor")]))
        // All centered z scores in a singleton are zero. The default gate offers a ranking, not no-match detection.
        #expect(try await engine.search("unrelated query", mode: .semantic).count == 1)
        #expect(try await engine.search("unrelated query", mode: .semantic, minimumCosine: 0.2).isEmpty)
        #expect(try await engine.search("electrical energy storage", mode: .semantic, minimumCosine: 0.2).first?.id == "circuit")
    }

    @Test func cachedProviderCannotCrossSourcesAfterAnExplicitClear() async throws {
        let folder = TemporaryFolder()
        let engine = try LocalSearchEngine(indexDirectory: folder.url, embedderFactory: fakeFactory(ProviderCalls()))
        try await engine.add(LocalSearchDocument(id: "first", title: "Circuit", segments: [.text("capacitor")]))
        try await engine.clear()
        let other = LocalSearchModelIdentity(model: "other", revision: "other-revision", runtime: "other", dtype: "f32")
        try await engine.add(LocalSearchDocument(id: "other", title: "Other",
            segments: [.vector(basis(1), modality: .text, source: other)]))
        await #expect(throws: LocalSearchError.self) { try await engine.search("plant", mode: .semantic) }
        #expect(try await engine.search("Other", mode: .keywords).first?.id == "other")
    }

    @Test func cancellationAfterEmbeddingPreservesPreviousRowsAndIdentity() async throws {
        let folder = TemporaryFolder()
        let engine = try LocalSearchEngine(indexDirectory: folder.url,
            embedderFactory: fakeFactory(ProviderCalls(cancelAfterTexts: true)))
        try await engine.add(LocalSearchDocument(id: "keep", title: "Original", segments: [.keywords("retained")]))
        let attempt = Task {
            try await engine.add(LocalSearchDocument(id: "keep", title: "Replacement", segments: [.text("capacitor CANCEL")]))
        }
        await #expect(throws: CancellationError.self) { try await attempt.value }
        #expect(try await engine.search("retained", mode: .keywords).first?.title == "Original")
        #expect(try await engine.statistics().vectors == 0)
        #expect(try await engine.statistics().model == nil)
    }

    @Test func droppingEngineClosesCustomProviderExactlyOnce() async throws {
        let folder = TemporaryFolder()
        let calls = ProviderCalls()
        var engine: LocalSearchEngine? = try LocalSearchEngine(indexDirectory: folder.url, embedderFactory: fakeFactory(calls))
        try await engine!.add(LocalSearchDocument(id: "text", title: "Circuit", segments: [.text("capacitor")]))
        engine = nil
        #expect(calls.closes == 1)
    }

    @Test func repeatedDuplicateOpenDoesNotReleaseFirstEngineOwnership() async throws {
        let folder = TemporaryFolder()
        let first = try LocalSearchEngine(indexDirectory: folder.url)
        for _ in 0..<3 {
            #expect(throws: LocalSearchError.self) { try LocalSearchEngine(indexDirectory: folder.url) }
        }
        try await first.add(LocalSearchDocument(id: "owned", title: "Still owned", segments: []))
        #expect(try await first.statistics().documents == 1)
    }

    @Test func reopenRejectsDamagedVectorBlobsAndChangedPromptPolicy() async throws {
        let folder = TemporaryFolder()
        var engine: LocalSearchEngine? = try LocalSearchEngine(indexDirectory: folder.url)
        try await engine!.add(LocalSearchDocument(id: "first", title: "First",
            segments: [.vector(basis(0), modality: .text, source: fakeIdentity)]))
        engine = nil
        #expect(throws: LocalSearchError.self) { try LocalSearchEngine(indexDirectory: folder.url, promptPolicy: .code) }
        let db = try Database(path: folder.url.appendingPathComponent(LocalSearchEngine.databaseFile).path)
        try db.run("UPDATE segments SET vec = ? WHERE vec IS NOT NULL", [.blob(Data([0, 0]))])
        #expect(throws: LocalSearchError.self) { try LocalSearchEngine(indexDirectory: folder.url) }
    }

    @Test func nativeFactoryVerifiesBeforeRuntimeLoad() async throws {
        let folder = TemporaryFolder()
        let bad = folder.url.appendingPathComponent("bad.gguf")
        try FileManager.default.createDirectory(at: folder.url, withIntermediateDirectories: true)
        try Data("not a model".utf8).write(to: bad)
        let engine = try LocalSearchEngine(indexDirectory: folder.url.appendingPathComponent("index"),
            embedderFactory: LocalSearchEngine.nativeFactory(textModelURL: bad))
        await #expect(throws: LocalSearchError.self) {
            try await engine.add(LocalSearchDocument(id: "text", title: "Test", segments: [.text("capacitor")]))
        }
        #expect(try await engine.statistics().documents == 0)
    }

    @Test func codePolicyUsesTheCodeQueryPrompt() async throws {
        let folder = TemporaryFolder()
        let calls = ProviderCalls()
        let engine = try LocalSearchEngine(indexDirectory: folder.url, promptPolicy: .code, embedderFactory: fakeFactory(calls))
        try await engine.add(LocalSearchDocument(id: "function", title: "Energy.swift", kind: .code,
            segments: [.text("capacitor", kind: .lines, locator: ["line": "40"])]))
        _ = try await engine.search("electrical energy storage")
        #expect(calls.textBatches.last == ["task: code retrieval | query: electrical energy storage"])
    }

    @Test(.enabled(if: ProcessInfo.processInfo.environment["LOCALSEARCH_MODEL_DIR"] != nil))
    func realNativeModelFindsThreeParaphrases() async throws {
        let directory = try #require(ProcessInfo.processInfo.environment["LOCALSEARCH_MODEL_DIR"])
        let models = URL(fileURLWithPath: directory)
        let folder = TemporaryFolder()
        let media = models.appendingPathComponent("mmproj-embeddinggemma-2-Q8_0.gguf")
        let engine = try LocalSearchEngine(indexDirectory: folder.url,
            embedderFactory: LocalSearchEngine.nativeFactory(
                textModelURL: models.appendingPathComponent("embeddinggemma-2-Q8_0.gguf"),
                mediaModelURL: FileManager.default.fileExists(atPath: media.path) ? media : nil))
        try await engine.add(threeDocuments())
        for (query, expected) in [("electrical energy storage", "circuit"),
                                  ("how leaves turn sunlight into food", "plants"),
                                  ("musical pitch and higher partials", "sound")] {
            #expect(try await engine.search(query, mode: .semantic).first?.id == expected)
        }
        await engine.releaseModel()
    }
}

private let fakeInfo = EmbeddingInfo(model: "test", revision: "full-test-revision-1", runtime: "deterministic",
                                      dtype: "f32", dim: 768, textOnly: false, loadSeconds: nil)
private let fakeIdentity = LocalSearchModelIdentity(fakeInfo)

private func basis(_ position: Int, magnitude: Float = 1) -> [Float] {
    var values = [Float](repeating: 0, count: 768)
    values[position] = magnitude
    return values
}

private func threeDocuments() -> [LocalSearchDocument] {
    [LocalSearchDocument(id: "circuit", title: "Circuit",
        segments: [.text("A capacitor stores charge across two conducting plates separated by an insulator.", locator: ["page": "2"])]),
     LocalSearchDocument(id: "plants", title: "Botany",
        segments: [.text("Photosynthesis in plants converts sunlight, water and carbon dioxide into sugars.")]),
     LocalSearchDocument(id: "sound", title: "Acoustics",
        segments: [.text("Harmonics are integer multiples of the fundamental frequency of a vibrating string.")])]
}

private func fakeFactory(_ calls: ProviderCalls) -> LocalSearchEmbedderFactory {
    { calls.recordLoad(); return DeterministicProvider(calls: calls) }
}

private enum TestFailure: Error { case embedding }

/// An orthogonal topic oracle tests search plumbing; it deliberately does not approximate model quality.
private final class DeterministicProvider: EmbeddingProvider {
    let info = fakeInfo
    let calls: ProviderCalls
    init(calls: ProviderCalls) { self.calls = calls }
    func embedTexts(_ texts: [String]) throws -> [[Float]] {
        calls.recordTexts(texts)
        if texts.contains(where: { $0.contains("FAIL") }) { throw TestFailure.embedding }
        if calls.cancelAfterTexts, texts.contains(where: { $0.contains("CANCEL") }) {
            withUnsafeCurrentTask { $0?.cancel() }
        }
        return texts.map { text in
            let text = text.lowercased()
            if ["capacitor", "electrical", "charge"].contains(where: text.contains) { return basis(0) }
            if ["photosynthesis", "plants", "leaves", "sunlight"].contains(where: text.contains) { return basis(1) }
            if ["harmonics", "musical", "partials"].contains(where: text.contains) { return basis(2) }
            return basis(3)
        }
    }
    func embedImages(_ paths: [String], budget: Int) throws -> [EmbeddingResult] {
        paths.map { path in
            calls.recordImage((URL(fileURLWithPath: path).lastPathComponent, budget))
            return EmbeddingResult(vector: basis(0), error: nil)
        }
    }
    func embedAudio(_ paths: [String]) throws -> [EmbeddingResult] {
        paths.map { path in
            calls.recordAudio(URL(fileURLWithPath: path).lastPathComponent)
            return path.contains("FAIL") ? EmbeddingResult(vector: nil, error: "invalid audio")
                : EmbeddingResult(vector: basis(2), error: nil)
        }
    }
    func close() { calls.recordClose() }
}

private final class ProviderCalls: @unchecked Sendable {
    let cancelAfterTexts: Bool
    private let lock = NSLock()
    private var storedLoads = 0, storedCloses = 0
    private var storedTexts: [[String]] = []
    private var storedImages: [(String, Int)] = []
    private var storedAudio: [String] = []
    init(cancelAfterTexts: Bool = false) { self.cancelAfterTexts = cancelAfterTexts }
    var loads: Int { lock.withLock { storedLoads } }
    var closes: Int { lock.withLock { storedCloses } }
    var textBatches: [[String]] { lock.withLock { storedTexts } }
    var images: [(String, Int)] { lock.withLock { storedImages } }
    var audio: [String] { lock.withLock { storedAudio } }
    func recordLoad() { lock.withLock { storedLoads += 1 } }
    func recordClose() { lock.withLock { storedCloses += 1 } }
    func recordTexts(_ values: [String]) { lock.withLock { storedTexts.append(values) } }
    func recordImage(_ value: (String, Int)) { lock.withLock { storedImages.append(value) } }
    func recordAudio(_ value: String) { lock.withLock { storedAudio.append(value) } }
}

private final class TemporaryFolder {
    let url = FileManager.default.temporaryDirectory.appendingPathComponent("LocalSearchKit-tests-" + UUID().uuidString)
    deinit { try? FileManager.default.removeItem(at: url) }
}
