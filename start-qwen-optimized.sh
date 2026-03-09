#!/bin/bash
# Optimized Qwen Server Configuration
# Based on RTX 3090 HuggingFace benchmark: 36.2 tps @ 100K context

nohup /home/matilda/llama.cpp/build/bin/llama-server \
  --model /home/matilda/unsloth/Qwen3-Coder-Next-GGUF/Qwen3-Coder-Next-UD-Q3_K_XL.gguf \
  --alias unsloth/Qwen3-Coder-Next \
  --flash-attn on \
  --seed 3407 \
  --temp 1.0 \
  --top-p 0.95 \
  --min-p 0.01 \
  --top-k 40 \
  --host 0.0.0.0 \
  --port 8080 \
  --ctx-size 102400 \
  --jinja \
  > /home/matilda/qwen-server.log 2>&1 &

echo $! > /home/matilda/qwen-server.pid
echo "Optimized Qwen server started with PID: $!"
echo "Configuration: Flash Attention ON + 100K context"
echo "Expected performance: 30-36 tokens/second"
