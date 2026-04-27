#!/usr/bin/env python3
"""Run vLLM serve with Sandermage Genesis patches active in same process.
Uses if __name__ guard for multiprocessing spawn compatibility."""
import os
import sys

def _apply_genesis():
    print("[wrapper] Applying Genesis patches...", flush=True)
    import vllm._genesis.patches.apply_all  # side-effect: runs apply_all
    print("[wrapper] Genesis applied.", flush=True)

if __name__ == '__main__':
    _apply_genesis()
    print("[wrapper] Launching vllm serve...", flush=True)
    from vllm.entrypoints.cli.main import main
    sys.exit(main())
