"""exl3 bench — proper token counting via generate()."""
import sys, time, os

MODEL = os.path.expanduser("~/models/Qwen3.6-27B-exl3-4.15bpw")
print(f"Loading model from {MODEL}...", flush=True)

from exllamav3 import Config, Model, Tokenizer, Cache, Generator

t0 = time.time()
config = Config.from_directory(MODEL)
model = Model.from_config(config)
tokenizer = Tokenizer.from_config(config)
cache = Cache(model, max_num_tokens=32000)
model.load()
print(f"  loaded in {time.time()-t0:.1f}s", flush=True)

generator = Generator(model=model, cache=cache, tokenizer=tokenizer)

PROSE = "Write a concise 800-token Python explanation of how an LSM tree works, including memtable, SSTables, compaction (size-tiered vs leveled), bloom filters, and tradeoffs."
CODE = "Write a complete TypeScript implementation of a binary search tree with insert, delete, search, and in-order traversal methods. Include unit tests for each method. Aim for 800 tokens of code."

print("warmup...", flush=True)
generator.generate(prompt="Hi", max_new_tokens=5)

import torch
for label, prompt in [("prose", PROSE), ("code", CODE)]:
    print(f"\n=== {label} ===", flush=True)
    prompt_ids = tokenizer.encode(prompt, encode_special_tokens=True, add_bos=True)
    print(f"  prompt tokens: {len(prompt_ids[0]) if hasattr(prompt_ids, 'shape') else len(prompt_ids)}", flush=True)
    t_start = time.time()
    out = generator.generate(prompt=prompt, max_new_tokens=800)
    wall = time.time() - t_start
    # Output includes prompt + generation
    out_ids = tokenizer.encode(out, encode_special_tokens=False, add_bos=False)
    n_total = out_ids.shape[-1] if hasattr(out_ids, 'shape') else len(out_ids)
    n_prompt = prompt_ids.shape[-1] if hasattr(prompt_ids, 'shape') else len(prompt_ids)
    completion_tokens = n_total - n_prompt
    decode_tps = completion_tokens / wall if wall > 0 else 0
    print(f"  total_toks={n_total}, prompt={n_prompt}, generated={completion_tokens}, wall={wall:.2f}s, tps={decode_tps:.2f}")
    gen_text = out[len(prompt):][:300]
    print(f"  preview: {gen_text!r}")

print("\ndone")
