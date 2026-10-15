import DigUpKit
import Foundation
import Testing
import LlamaRuntime

@Suite struct LifetimeTests {
    @Test(.enabled(if: LlamaModels.complete)) func closedRuntimeRejectsCalls() throws {
        let runtime = try LlamaEmbedder(models: LlamaModels.locate(), textOnly: true, silent: true)
        runtime.close()
        runtime.close()
        #expect(throws: WorkerError.self) { try runtime.embedTexts(["a query after close"]) }
        #expect(throws: WorkerError.self) { try runtime.embedImages([], budget: 140) }
        #expect(throws: WorkerError.self) { try runtime.embedAudio([]) }
    }
}
