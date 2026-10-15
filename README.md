# LocalSearchKit

LocalSearchKit adapts DigUp's local embedding and keyword search into a Swift package for app-owned content. An actor owns the index and model, and returns your document IDs with the matching page, timestamp, line, or other locator you supply.

The public integration layer starts no folder crawl, model download, updater, query log, or background service. You choose the content, storage directory, model files, and when to release model memory. Native inference runs in the app process, including on iOS.

## Add it to an app

In Xcode, add this directory as a local package and select the **LocalSearchKit** product. The generated native XCFramework is already present in `Vendor/` on this machine. Its supported slices are Apple Silicon macOS 14+, arm64 iOS 17+, and arm64 iOS Simulator. Intel Mac, Intel Simulator, Catalyst, Linux, and Windows are outside this package's build coverage.

Physical iOS builds load offline-compiled Metal kernels from the automatically bundled `LlamaRuntime` package resources. The library includes BF16 kernels for the BF16 projection retained inside the Q8_0 model. Keep the resource bundle with the app; no host-app shader compilation or manual resource copying is required. Native activation and inference require foreground access to the GPU.

```swift
import Foundation
import LocalSearchKit

// Put the index in your app's Application Support directory.
// Provide the verified model files from your app's storage or bundle.
let engine = try LocalSearchEngine(
    indexDirectory: indexURL,
    embedderFactory: LocalSearchEngine.nativeFactory(
        textModelURL: textModelURL,
        mediaModelURL: mediaModelURL
    )
)

try await engine.add(LocalSearchDocument(
    id: "lesson:pressure",
    title: "Pressure and force",
    segments: [
        .text("Pressure is force divided by area. The same force over a smaller area creates greater pressure.",
              locator: ["section": "2"])
    ],
    metadata: ["course": "Mechanics"]
))

let hits = try await engine.search("why does a sharp point press harder?")
// hits.first?.id and hits.first?.locator map directly back to your app.
try await engine.remove(id: "lesson:pressure")
await engine.releaseModel()
```

Model verification and inference happen when the actor first needs an embedding, rather than in the initializer. Native factories verify the pinned files' sizes and SHA-256 values. You can inject an exclusive `EmbeddingProvider` through `LocalSearchEmbedderFactory`; its output must contain 768 finite, nonzero values and a complete model identity. Precomputed vectors also declare their model identity and prompt policy. Changing the model or prompt policy requires an explicit new index or clearing the old one.

Use `.keywords` segments and `.keywords` search mode for model-free exact-word retrieval. Text segments use the model's document prompt; `.code` prompt policy uses its code-retrieval query prompt in a separate index. `add` inserts or replaces a full document; batch adds commit atomically. `remove` deletes its vectors, extracted words, and locators. Keep one engine alive per index directory in each process. Independent processes require coordination by the host app.

## Images and recordings

An explicit image segment accepts a prepared raster file and a vision token budget:

```swift
let frame = LocalSearchSegment.image(
    preparedJPEGURL, budget: 70, kind: .frame,
    locator: ["seconds": "18.0"]
)
let audio = LocalSearchSegment.audio(
    preparedWAVURL, locator: ["seconds": "25.0", "endSeconds": "55.0"]
)
```

The host app prepares image files, video frames, and audio windows. Supply PNG or JPEG to the native image encoder; convert HEIC or SVG first. Native audio retains at most the first 30 seconds of each supplied window. Audio embeddings provide semantic matching; transcripts for exact spoken-word search must be supplied separately as keywords. Media inference needs both model files.

The lower-level Apple extraction core also compiles for iOS: raster image conversion, OCR, PDF, plain text, HTML, RTF, source code, audio, and video paths are available. DOCX, DOC, ODT, and SVG extraction are explicitly skipped on iOS. Git subprocess filtering and desktop Spotlight metadata remain macOS features. No extraction is triggered by the public engine itself.

## Ranking and storage

DigUp's ranking combines an exhaustive scan of normalized vectors with SQLite FTS5 word coverage. Vectors are stored as 768 float16 values, or 1,536 bytes each before database overhead. App content, titles, metadata, and keyword text remain in ordinary local SQLite files; this package adds no database encryption.

Scores order results and are not probabilities. Semantic scores depend on the collection. The public API uses a z-score gate of 0 for fewer than 20 vectors and 2 thereafter; the smaller gate prevents small collections from hiding every semantic result. A small collection can still return its closest item for an unrelated query. Tune `zGate`, `minimumCosine`, and `wordWeight` against representative questions and distractors. There is no universally calibrated cosine threshold.

Text-only native mode reserves 512 tokens per input; full multimodal mode reserves 2,048. Supply text in meaningful passages, approximately 1,800 characters each, with overlap when context spans passages. A segment longer than the runtime limit is truncated for its embedding, while its full text remains searchable by words. The model card's supported 8,192-token input is broader than these runtime budgets.

## Model files

Weights are not bundled with the package. The verified files downloaded for DigUp on this Mac are in `~/Library/Application Support/DigUp/Models`. An iPhone app needs its own local copies; it cannot use the Mac's filesystem path.

| File | Bytes | Role |
| --- | ---: | --- |
| `embeddinggemma-2-Q8_0.gguf` | 309,855,456 | Text backbone and final embedding projection |
| `mmproj-embeddinggemma-2-Q8_0.gguf` | 554,821,024 | Vision and audio encoders |

Together they occupy about 865 MB decimal. The native runtime reports approximately 0.25 GB model memory in text-only mode and about 1 GB in full mode; host-app and device memory behavior requires measurement. `releaseModel()` releases the provider and in-memory vector matrix; later searches reload them.

Download provenance, immutable revisions, checksums, and licenses are recorded in [PROVENANCE.json](PROVENANCE.json). The model is Google's EmbeddingGemma 2, distributed here as ggml-org's Q8_0 GGUF conversion. See the [official model card](https://ai.google.dev/gemma/docs/embeddinggemma/model_card_2) and [pinned GGUF repository](https://huggingface.co/ggml-org/embeddinggemma-2-GGUF/tree/bfcd298762cc34d0357ece5ebdd31791a3a374d8).

## Rebuild the native library

The device build compiles Metal libraries for iOS with BF16 support. An installed Xcode Metal toolchain is required. To rebuild only the physical-device archive and its resource libraries while preserving the existing Mac and Simulator slices, pass `--device-only` to the script. The pinned patches, archive hashes, resource hashes, and build commands are recorded in `Vendor/runtime-build.json`.

Use full Xcode and CMake with Ninja. The script changes no global Xcode setting or package installation:

```sh
python3 scripts/build-native.py
```

If CMake and Ninja are absent, an isolated tool environment works:

```sh
uv run --no-project --with cmake==4.4.4 --with ninja==1.13.2 python scripts/build-native.py
```

Set `DEVELOPER_DIR` for a nondefault Xcode installation. The script checks the immutable llama.cpp commit, applies the included noncausal-embedding patch, builds three arm64 slices, and writes `Vendor/runtime-build.json`. `Vendor/` is generated and ignored; preserve or rebuild it when moving this package to another checkout.

```sh
swift test
LOCALSEARCH_MODEL_DIR=/absolute/path/to/Models swift test --filter LocalSearchKitTests
DIGUP_MODELS=/absolute/path/to/Models swift test --filter ReferenceVectorTests
```

The model-enabled checks read only explicit model files and generated/reference fixtures. They do not index your documents. Cross-platform compilation and Mac inference do not establish iPhone inference speed, memory use, or physical-device behavior.

Recorded Mac validation: **110 tests passed with no skips**, including the actual local model's retrieval test and nine text/image/audio reference vectors at cosine agreement ≥0.999. The library compiled for Mac, iOS device, and iOS Simulator SDK targets. The scanned-PDF OCR test completed successfully but took 156.6 seconds inside Apple's Vision service. Simulator execution was not performed. See [VALIDATION.json](VALIDATION.json) for that suite's commands, logs, and source/native archive hashes.

Physical iPhone validation now also passed in [Search Bench](../SearchBench/RESULTS.md): text-only and full-encoder activation, 128 real-model vectors, 50 timed text searches per mode, and all eight search fixtures ranked first. The device-only precompiled Metal build and BF16 kernel patches are recorded in `Vendor/runtime-build.json`. A targeted model-verification factory test passed after the integration change. These device checks do not establish numerical reference-vector agreement on iPhone or image/audio search latency.

## Provenance and licenses

Core algorithms and native wrapper are adapted from [DigUp v0.6.1](https://github.com/ARahim3/DigUp/tree/v0.6.1), commit `6f3a506a2e6d493463fd9b919f8c8860b62cc2c0`, under MIT. The native library is pinned to llama.cpp b11461, commit `4d756bc72bf00a4aacf410ae15a2d315f3db400d`, under MIT. EmbeddingGemma 2 weights use Apache 2.0. Retain [LICENSE](LICENSE), [NATIVE_RUNTIME_LICENSE.txt](NATIVE_RUNTIME_LICENSE.txt), and applicable [THIRD_PARTY_NOTICES.txt](THIRD_PARTY_NOTICES.txt) when distributing derived apps.

This package contains the engine and runtime; DigUp's menu bar UI, artwork, Sparkle updater, and standalone app services are not included. The original installed DigUp app and its index remain separate.


## Public snapshot

See [PUBLICATION.md](PUBLICATION.md) for publication exclusions and setup requirements.
