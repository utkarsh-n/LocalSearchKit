#!/bin/bash
# Compile the library for Apple targets. Does not boot, query, or modify Simulator.
set -euo pipefail
cd "$(dirname "$0")/.."
swift build --target LocalSearchKit
swift build --triple arm64-apple-ios17.0 --sdk "$(xcrun --sdk iphoneos --show-sdk-path)" \
  --target LocalSearchKit --scratch-path .build/ios-device
swift build --triple arm64-apple-ios17.0-simulator --sdk "$(xcrun --sdk iphonesimulator --show-sdk-path)" \
  --target LocalSearchKit --scratch-path .build/ios-simulator
