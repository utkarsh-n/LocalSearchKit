#!/usr/bin/env python3
"""Build the pinned embedding runtime for Apple Silicon Mac and iOS.

No global packages or Xcode settings are changed. Pass an existing pinned source
checkout, CMake executable or macOS archive to reuse prior compilation.
"""
import argparse
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]
COMMIT = "4d756bc72bf00a4aacf410ae15a2d315f3db400d"
TAG = "b11461"
parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("--source", type=Path)
parser.add_argument("--cmake", default=shutil.which("cmake"))
parser.add_argument("--mac-library", type=Path)
parser.add_argument("--jobs", type=int, default=min(6, os.cpu_count() or 2))
parser.add_argument("--device-only", action="store_true",
                    help="Rebuild only the iOS archive and bundled Metal libraries; preserve Mac/Simulator slices.")
args = parser.parse_args()
if not args.cmake:
    sys.exit("CMake is required. Supply --cmake /path/to/cmake; no global installation is performed.")
env = os.environ.copy()
work = ROOT / ".build/native"
work.mkdir(parents=True, exist_ok=True)
source = args.source.resolve() if args.source else work / "llama.cpp"
commands = []
def run(command, capture=False):
    command = [str(a) for a in command]
    commands.append(command)
    print("Running:", " ".join(command), flush=True)
    if capture:
        return subprocess.check_output(command, env=env, text=True).strip()
    subprocess.run(command, env=env, check=True)

if not source.exists():
    run(["git", "clone", "--depth", "1", "--branch", TAG, "https://github.com/ggml-org/llama.cpp.git", source])
actual = run(["git", "-C", source, "rev-parse", "HEAD"], capture=True)
if actual != COMMIT:
    sys.exit(f"Refusing unpinned runtime: {actual}, expected {COMMIT}")
patches = [ROOT / "scripts/llama.cpp-no-logits.patch", ROOT / "scripts/llama.cpp-metal-resource-path.patch"]
for patch in patches:
    if not patch.is_file():
        sys.exit(f"Required native patch is missing: {patch}")
    check = subprocess.run(["git", "-C", str(source), "apply", "--reverse", "--check", str(patch)], capture_output=True)
    if check.returncode:
        run(["git", "-C", source, "apply", "--check", patch])
        run(["git", "-C", source, "apply", patch])

headers = work / "headers"
headers.mkdir(exist_ok=True)
for file in [source / "include/llama.h", source / "tools/mtmd/mtmd.h", source / "tools/mtmd/mtmd-helper.h", *sorted((source / "ggml/include").glob("*.h"))]:
    shutil.copy2(file, headers / file.name)
(headers / "llama-build.h").write_text(f'#define LLAMA_BUILD_TAG "{TAG}"\n')
(headers / "module.modulemap").write_text('module CLlama {\n header "llama.h"\n header "mtmd.h"\n header "mtmd-helper.h"\n header "llama-build.h"\n export *\n}\n')

archives = []
platforms = [("macos", "macosx", "14.0"), ("ios", "iphoneos", "17.0"), ("ios-simulator", "iphonesimulator", "17.0")]
if args.device_only:
    platforms = [item for item in platforms if item[0] == "ios"]
resource_files = []
for platform, sdk, deployment in platforms:
    if platform == "macos" and args.mac_library:
        archive = args.mac_library.resolve()
    else:
        build = work / platform
        command = [args.cmake, "-S", source, "-B", build, "-G", "Ninja",
                   "-DCMAKE_BUILD_TYPE=Release", "-DCMAKE_OSX_ARCHITECTURES=arm64",
                   f"-DCMAKE_OSX_SYSROOT={sdk}", f"-DCMAKE_OSX_DEPLOYMENT_TARGET={deployment}",
                   "-DBUILD_SHARED_LIBS=OFF", "-DGGML_METAL=ON",
                   "-DGGML_METAL_EMBED_LIBRARY=" + ("OFF" if platform == "ios" else "ON"),
                   "-DGGML_NATIVE=OFF", "-DGGML_OPENMP=OFF", "-DLLAMA_BUILD_MTMD=ON",
                   "-DLLAMA_BUILD_COMMON=OFF", "-DLLAMA_BUILD_TOOLS=OFF", "-DLLAMA_BUILD_SERVER=OFF",
                   "-DLLAMA_BUILD_EXAMPLES=OFF", "-DLLAMA_BUILD_TESTS=OFF", "-DLLAMA_BUILD_APP=OFF",
                   "-DLLAMA_BUILD_UI=OFF", "-DLLAMA_OPENSSL=OFF", "-DLLAMA_CURL=OFF"]
        if platform != "macos":
            command += ["-DCMAKE_SYSTEM_NAME=iOS"]
        if platform == "ios":
            command += ["-DGGML_METAL_TARGET_OS=ios"]
        run(command)
        targets = ["llama", "mtmd"] + (["ggml-metal-lib"] if platform == "ios" else [])
        run([args.cmake, "--build", build, "--config", "Release", "-j", args.jobs, "--target", *targets])
        if platform == "ios":
            resources = ROOT / "Sources/LlamaRuntime/Resources/Metal/ios"
            resources.mkdir(parents=True, exist_ok=True)
            for name in ["default.metallib", "ggml-tensor.metallib"]:
                metal_library = build / "bin" / name
                if not metal_library.is_file():
                    if name == "default.metallib":
                        sys.exit(f"Required precompiled Metal library was not built: {metal_library}")
                    continue
                destination = resources / name
                shutil.copy2(metal_library, destination)
                resource_files.append({"path": str(destination.relative_to(ROOT)),
                                       "bytes": destination.stat().st_size,
                                       "sha256": hashlib.sha256(destination.read_bytes()).hexdigest()})
        archive = build / "libllamacpp.a"
        libs = [build / rel for rel in ["src/libllama.a", "tools/mtmd/libmtmd.a", "ggml/src/libggml.a",
                "ggml/src/libggml-base.a", "ggml/src/libggml-cpu.a", "ggml/src/ggml-metal/libggml-metal.a",
                "ggml/src/ggml-blas/libggml-blas.a", "vendor/hash/libvendor-hash.a"]]
        run(["xcrun", "libtool", "-static", "-no_warning_for_no_symbols", "-o", archive, *libs])
    archives.append(archive)

output = ROOT / "Vendor/LlamaRuntime.xcframework"
output.parent.mkdir(exist_ok=True)
manifest_path = output.parent / "runtime-build.json"
patch_records = [{"path": str(item.relative_to(ROOT)),
                  "sha256": hashlib.sha256(item.read_bytes()).hexdigest()} for item in patches]
if args.device_only:
    destination = output / "ios-arm64/libllamacpp.a"
    if not destination.is_file():
        sys.exit("--device-only requires an existing iOS XCFramework slice; run a full build first.")
    shutil.copy2(archives[0], destination)
    # Header APIs are unchanged by the loader patch; preserve the other platform slices byte for byte.
    manifest = json.loads(manifest_path.read_text()) if manifest_path.is_file() else {}
    manifest.update({"repository": "https://github.com/ggml-org/llama.cpp", "commit": COMMIT, "tag": TAG,
                     "patches": patch_records, "ios_metal_mode": "precompiled",
                     "ios_metal_resources": resource_files,
                     "ios_archive_sha256": hashlib.sha256(destination.read_bytes()).hexdigest()})
    manifest.setdefault("device_only_rebuilds", []).append({"commands": commands,
        "archive_sha256": manifest["ios_archive_sha256"], "metal_resources": resource_files})
    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n")
    print("Rebuilt iOS device slice and bundled precompiled Metal libraries:", destination)
    sys.exit(0)
staged = work / "LlamaRuntime.xcframework"
if staged.exists():
    shutil.rmtree(staged)
command = ["xcodebuild", "-create-xcframework"]
for archive in archives:
    command += ["-library", archive, "-headers", headers]
run(command + ["-output", staged])
if output.exists():
    shutil.rmtree(output)
shutil.move(staged, output)
manifest = {"repository": "https://github.com/ggml-org/llama.cpp", "commit": COMMIT, "tag": TAG,
            "patch_sha256": patch_records[0]["sha256"], "patches": patch_records,
            "architectures": ["macos-arm64", "ios-arm64", "ios-arm64-simulator"], "commands": commands,
            "ios_metal_mode": "precompiled", "ios_metal_resources": resource_files}
manifest_path.write_text(json.dumps(manifest, indent=2) + "\n")
print("Built:", output)
