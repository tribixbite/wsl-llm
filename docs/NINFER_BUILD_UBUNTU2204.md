# Building NInfer (RTX 3090 fork) on Ubuntu 22.04 / WSL2

`Don-Chad/ninfer-3090` v0.6.1 targets Ubuntu 24.04. Four things must be fixed on 22.04.
None of them touch the GPU driver.

## 1. CUDA toolkit ≥ 12.8 (we had 12.6)

**This is toolkit-only and safe in WSL2.** Do NOT install `cuda` or `cuda-drivers`
metapackages — in WSL2 the driver lives on the Windows side and `/usr/lib/wsl/lib/libcuda.so.1`
is a ~175 KB shim to the Windows KMD. Installing a Linux driver breaks GPU passthrough.

```bash
wget https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2204/x86_64/cuda-keyring_1.1-1_all.deb
sudo dpkg -i cuda-keyring_1.1-1_all.deb && sudo apt-get update
# verify FIRST that no driver packages are pulled in:
sudo apt-get install --dry-run cuda-toolkit-12-9 | grep -ciE '^Inst (cuda-drivers|nvidia-driver|libnvidia-)'   # must print 0
sudo apt-get install --yes cuda-toolkit-12-9
```
Verify after: `libcuda.so.1` size unchanged (175360 bytes) and `nvidia-smi` still lists both GPUs.

## 2. gcc-13 (22.04 ships gcc-11)

```bash
sudo add-apt-repository -y ppa:ubuntu-toolchain-r/test && sudo apt-get update
sudo apt-get install --yes gcc-13 g++-13 ninja-build pkg-config \
  libcurl4-openssl-dev libavcodec-dev libavformat-dev libavutil-dev libswscale-dev
```

## 3. Relax the pkg-config version floors (CMakeLists.txt)

22.04 has FFmpeg 4.4 (libav* 58) and libcurl 7.81; upstream asks for 60 / 7.85.

```
libavformat>=60 libavcodec>=60 libavutil>=58 libswscale>=7  ->  >=58 >=58 >=56 >=5
libcurl>=7.85                                               ->  libcurl>=7.81
```

## 4. Back-port two media APIs (the floors are real, not conservative)

Relaxing the floors alone fails to compile — the code really does use FFmpeg 6 / curl 7.85
APIs. Both are in *media* (image/video input), not the LLM core:

- `src/media/decode/decode.cpp` — `codecpar->coded_side_data` / `av_packet_side_data_get`
  don't exist before libavcodec 60. Guarded with `#if LIBAVCODEC_VERSION_MAJOR >= 60`,
  with a pre-6.0 fallback over `stream->side_data` (display-rotation metadata only).
- `src/product/media_acquire/acquire.cpp` — `CURLOPT_PROTOCOLS_STR` /
  `CURLOPT_REDIR_PROTOCOLS_STR` need curl 7.85. Guarded with
  `#if LIBCURL_VERSION_NUM >= 0x075500`, falling back to the `CURLOPT_PROTOCOLS` bitmask.

## Build

```bash
cd ~/ninfer-3090
export CC=/usr/bin/gcc-13 CXX=/usr/bin/g++-13 \
       CUDACXX=/usr/local/cuda-12.9/bin/nvcc CUDAHOSTCXX=/usr/bin/g++-13
cmake -S . -B build-sm86 -G Ninja -DCMAKE_BUILD_TYPE=Release \
  -DCMAKE_C_COMPILER="$CC" -DCMAKE_CXX_COMPILER="$CXX" \
  -DCMAKE_CUDA_COMPILER="$CUDACXX" -DCMAKE_CUDA_HOST_COMPILER="$CUDAHOSTCXX" \
  -DCMAKE_CUDA_ARCHITECTURES=86 -DNINFER_BUILD_APPS=ON \
  -DBUILD_TESTING=OFF -DNINFER_BUILD_BENCHMARKS=OFF
cmake --build build-sm86 --parallel 6     # -j6: -j24 nvcc has thermally tripped this box
```
Produces `build-sm86/apps/{ninfer,ninfer-serve,ninfer-perplexity}`.

## Model + serve

```bash
hf download neroued/Qwen3.8-27B-NInfer qwen3_8_27b.ninfer --local-dir ~/ninfer-3090/models

CUDA_VISIBLE_DEVICES=1 ~/ninfer-3090/build-sm86/apps/ninfer-serve \
  ~/ninfer-3090/models/qwen3_8_27b.ninfer \
  --host 0.0.0.0 --port 8086 \
  --max-context 65536 --kv-capacity 65536 \
  --max-concurrency 1 --max-pending-requests 16 \
  --prefill-chunk 1024 --kv-dtype int8 \
  --spec mtp --draft-tokens 3 --lm-head-draft
```

## Expectation

Author's published RTX 3090 numbers: **C1 71.0 t/s** (61.1% MTP acceptance, 149 ms TTFT,
19.6 GB), C4 100.3, **C8 165.3**. Note C1 is *below* our vLLM W4A16 stack's measured
105 t/s single-stream — NInfer's advantage is concurrency, not single-user latency.
