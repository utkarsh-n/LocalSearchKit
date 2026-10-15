# Public source snapshot

This repository contains a reviewed source snapshot. Original Git history, credentials, personal records, local signing identities, device diagnostics, build outputs, downloaded model weights, firmware, and unreviewed media are excluded. Existing upstream license and attribution files apply to their respective components. Publication does not grant a new license to third-party content.

Configure your own Apple signing team to run native apps on a physical device. Downloaded datasets, binary runtimes, fonts, demo media, and models may need to be obtained separately using the documented upstream sources or fetch/build tools. This publication review does not certify that this snapshot builds or runs.

## Project setup and omitted assets

This is an attributed adaptation of DigUp v0.6.1 and llama.cpp b11461. Retain `LICENSE`, `NATIVE_RUNTIME_LICENSE.txt`, `THIRD_PARTY_NOTICES.txt` and `PROVENANCE.json`; those notices apply to their respective upstream code.

The generated `Vendor/LlamaRuntime.xcframework` and compiled `Sources/LlamaRuntime/Resources/Metal/*.metallib` binaries are omitted. Before resolving/building the package, use `python3 scripts/build-native.py` with Xcode, CMake and Ninja available; it clones the pinned upstream source, applies the retained patches and builds the required Apple runtime slices. See `scripts/build-native.py --help` for reuse options. EmbeddingGemma GGUF model weights are separately acquired using the retained model downloader/provenance; no model payload is bundled.

Binary image/audio test references are omitted from `Tests/LlamaRuntimeTests/Reference/`; tests that require those fixtures need replacement synthetic fixtures. `/Users/me` paths and the malformed private-key marker in unit tests are intentional synthetic test inputs. They do not identify a user or contain a working key.
