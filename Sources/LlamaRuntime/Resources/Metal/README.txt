Physical iOS devices load precompiled Metal kernels from Metal/ios/default.metallib.
The build-native.py script creates these resources from pinned llama.cpp b11461.
The iOS runtime uses GGML_METAL_PATH_RESOURCES to locate this package resource bundle.
macOS and iOS Simulator continue using their existing embedded Metal source libraries.
