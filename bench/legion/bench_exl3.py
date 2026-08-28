"""ExLlamaV3 decode/prefill benchmark, methodology-matched to the llama.cpp runs.

Held constant vs the llama.cpp side:
  - same three prompts as bench/stream_bench.py (prose / code / json)
  - decode_TPS = generated_tokens / (wall - TTFT), i.e. TTFT excluded
  - a warmup generation before any measurement

Deliberately different: generation length is FORCED to a fixed count via
min_new_tokens == max_new_tokens with no stop conditions. For a throughput
comparison this is more honest than letting each engine stop where it likes,
and it makes the number independent of the chat template (which differs between
the GGUF and EXL3 repos).
"""

import argparse
import json
import time

import torch
from exllamav3 import Cache, CacheLayer_quant, Config, Generator, Job, Model, Tokenizer

PROMPTS = {
    "prose": (
        "Write a concise 800-token explanation of how an LSM tree works, including memtable, "
        "SSTables, compaction (size-tiered vs leveled), bloom filters, and tradeoffs."
    ),
    "code": (
        "Write a complete TypeScript implementation of a binary search tree with insert, delete, "
        "search, and in-order traversal methods. Include unit tests for each method. "
        "Aim for 800 tokens of code."
    ),
    "json": (
        "Output a JSON array of 40 fictional employee records, each with fields: id, name, "
        "department, salary, hire_date, manager_id, skills (array of 3 strings). "
        "Valid JSON only, no prose."
    ),
}


def build(model_dir: str, max_tokens: int, kv_bits: int | None, batch: int = 1):
    config = Config.from_directory(model_dir)
    model = Model.from_config(config)
    # max_batch_size defaults to 16. On this hybrid arch (48 gated-DeltaNet
    # layers) every batch slot carries its own recurrent state, so the default
    # silently costs gigabytes -- the same trap as llama.cpp's --parallel 4,
    # which cost 2.5 GiB and triggered a WDDM eviction. Single stream => 1.
    kwargs = {"max_num_tokens": max_tokens, "max_batch_size": batch}
    if kv_bits:
        cache = Cache(model, layer_type=CacheLayer_quant, k_bits=kv_bits, v_bits=kv_bits, **kwargs)
    else:
        cache = Cache(model, **kwargs)
    model.load()
    tokenizer = Tokenizer.from_config(config)
    generator = Generator(model=model, cache=cache, tokenizer=tokenizer, max_batch_size=batch)
    return model, tokenizer, generator


def run_one(tokenizer, generator, prompt: str, n_tokens: int) -> dict:
    ids = tokenizer.encode(prompt, add_bos=True)
    job = Job(
        input_ids=ids,
        max_new_tokens=n_tokens,
        min_new_tokens=n_tokens,   # force exact length; no early stop
        stop_conditions=[],
        seed=1337,
    )
    generator.enqueue(job)

    t0 = time.time()
    ttft = None
    produced = 0
    while generator.num_remaining_jobs():
        for res in generator.iterate():
            chunk = res.get("text", "")
            if chunk and ttft is None:
                ttft = time.time() - t0
            if res.get("stage") == "streaming":
                produced += len(res.get("token_ids", [])) if res.get("token_ids") is not None else 0
    wall = time.time() - t0
    produced = produced or n_tokens
    decode_wall = max(wall - (ttft or 0.0), 1e-6)
    return {
        "prompt_tokens": int(ids.shape[-1]),
        "gen_tokens": produced,
        "ttft_ms": (ttft or 0.0) * 1000,
        "wall_s": wall,
        "decode_tps": produced / decode_wall,
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model-dir", required=True)
    ap.add_argument("--max-tokens", type=int, default=32768, help="cache size in tokens")
    ap.add_argument("--gen", type=int, default=400, help="tokens to generate per prompt")
    ap.add_argument("--kv-bits", type=int, default=0, help="0 = fp16 cache, else quantized K/V bits")
    ap.add_argument("--batch", type=int, default=1, help="cache batch slots; >1 costs recurrent state per slot")
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    model, tokenizer, generator = build(args.model_dir, args.max_tokens, args.kv_bits or None, args.batch)
    free, total = torch.cuda.mem_get_info()
    print(f"VRAM after load: used {(total - free) / 2**20:.0f} MiB of {total / 2**20:.0f} MiB")

    run_one(tokenizer, generator, "Say hi.", 16)  # warmup

    rows = {}
    for name, prompt in PROMPTS.items():
        r = run_one(tokenizer, generator, prompt, args.gen)
        rows[name] = r
        print(f"[{name}] decode={r['decode_tps']:.2f} t/s  ttft={r['ttft_ms']:.0f}ms  "
              f"gen={r['gen_tokens']}  wall={r['wall_s']:.2f}s")

    avg = sum(r["decode_tps"] for r in rows.values()) / len(rows)
    free, total = torch.cuda.mem_get_info()
    peak = (total - free) / 2**20
    print(f"AVG decode = {avg:.2f} t/s   VRAM {peak:.0f} MiB")
    if args.out:
        json.dump({"rows": rows, "avg_decode_tps": avg, "vram_mib": peak,
                   "kv_bits": args.kv_bits, "cache_tokens": args.max_tokens},
                  open(args.out, "w"), indent=2)


if __name__ == "__main__":
    main()
