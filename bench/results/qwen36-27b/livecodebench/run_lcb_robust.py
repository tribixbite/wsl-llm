"""Robust LCB v6 runner with per-problem checkpointing + retries + vLLM restart.

Resumable: re-running picks up where the last run left off (checkpoint at /tmp/lcb_robust/per_problem.jsonl).
"""
import sys, json, os, time, urllib.request, subprocess, signal
sys.path.insert(0, '/home/matilda/git/SI/src')
from si.livecodebench import load_lcb, _build_user_prompt, _extract_code, _check_problem, _SYSTEM_LCB
from sandbox_fusion import set_endpoint

URL = "http://192.168.1.32:8081"
MODEL = "qwen3.6-27b"
SANDBOX = os.environ.get("SANDBOX_FUSION_ENDPOINT", "http://localhost:42137")
OUT_DIR = os.environ.get("LCB_OUT_DIR", "/tmp/lcb_robust")
CHECKPOINT = os.path.join(OUT_DIR, "per_problem.jsonl")
SUMMARY_OUT = os.path.join(OUT_DIR, "summary.json")
MAX_TOKENS = int(os.environ.get("LCB_MAX_TOKENS", "4096"))
ENABLE_THINKING = os.environ.get("LCB_THINK", "false").lower() == "true"
N_LIMIT = int(os.environ.get("LCB_N", "0")) or None  # 0 means all

os.makedirs(OUT_DIR, exist_ok=True)
set_endpoint(SANDBOX)
print(f"Sandbox: {SANDBOX}")
print(f"Output: {OUT_DIR}")
print(f"Thinking: {ENABLE_THINKING}, max_tokens: {MAX_TOKENS}")

VLLM_RESTART_SCRIPT = "/home/matilda/git/wsl-llm/scripts/serve-27b-vllm-genesis.sh"

def http_chat(prompt, max_tokens=MAX_TOKENS, temp=None, thinking=ENABLE_THINKING, retries=3):
    """Call vLLM with retries. Returns (text, completion_tokens) or (None, 0) on failure."""
    if temp is None:
        temp = 0.6 if thinking else 0.2
    body = json.dumps({
        'model': MODEL,
        'messages': [{'role':'system','content':_SYSTEM_LCB},{'role':'user','content':prompt}],
        'max_tokens': max_tokens,
        'temperature': temp, 'top_p': 0.95,
        'chat_template_kwargs':{'enable_thinking': thinking},
    }).encode()
    for attempt in range(retries):
        try:
            req = urllib.request.Request(f'{URL}/v1/chat/completions', data=body,
                                         headers={'Content-Type':'application/json'})
            with urllib.request.urlopen(req, timeout=900) as r:
                d = json.load(r)
            text = d['choices'][0]['message']['content']
            completion_tokens = d.get('usage', {}).get('completion_tokens', 0)
            return text, completion_tokens
        except Exception as e:
            err = str(e)[:100]
            if attempt < retries - 1:
                print(f"    HTTP attempt {attempt+1}/{retries} failed: {err}; sleeping then retrying...", flush=True)
                time.sleep(20 * (attempt + 1))
                # If error is "connection refused" the server is dead
                if 'refused' in err.lower() or 'reset' in err.lower():
                    if not vllm_alive():
                        restart_vllm()
            else:
                print(f"    HTTP all retries failed: {err}", flush=True)
                return None, 0
    return None, 0

def vllm_alive():
    try:
        with urllib.request.urlopen(f'{URL}/v1/models', timeout=5) as r:
            return r.status == 200
    except Exception:
        return False

def restart_vllm():
    print("    [restart] killing vLLM + zombies...", flush=True)
    subprocess.run("pkill -9 -f 'vllm.entrypoints'", shell=True)
    subprocess.run("pkill -9 -f 'VLLM::EngineCore'", shell=True)
    time.sleep(8)
    # Kill any compute-app zombies
    out = subprocess.run("nvidia-smi --query-compute-apps=pid --format=csv,noheader",
                         shell=True, capture_output=True, text=True).stdout
    for pid in out.strip().split("\n"):
        pid = pid.strip()
        if pid:
            subprocess.run(f"kill -9 {pid}", shell=True)
    time.sleep(5)
    print("    [restart] launching fresh vLLM...", flush=True)
    subprocess.Popen(
        f"nohup {VLLM_RESTART_SCRIPT} > /tmp/vllm_robust.log 2>&1 < /dev/null &",
        shell=True, preexec_fn=os.setsid
    )
    # Wait for ready
    for i in range(180):  # max 12 min
        if vllm_alive():
            print(f"    [restart] vLLM back online after {i*4}s", flush=True)
            return True
        time.sleep(4)
    print("    [restart] FAILED to come back", flush=True)
    return False

def load_done():
    """Load already-completed problems from checkpoint."""
    done = {}
    if os.path.exists(CHECKPOINT):
        with open(CHECKPOINT) as f:
            for line in f:
                line = line.strip()
                if not line: continue
                try:
                    r = json.loads(line)
                    done[r['problem_id']] = r
                except: pass
    return done

def append_result(r):
    with open(CHECKPOINT, "a") as f:
        f.write(json.dumps(r) + "\n")

def main():
    probs = load_lcb()
    if N_LIMIT:
        probs = probs[:N_LIMIT]
    done = load_done()
    print(f"Total problems: {len(probs)}; already done: {len(done)}; remaining: {len(probs) - len(done)}")
    print(flush=True)

    t_global = time.time()
    for i, p in enumerate(probs):
        if p.problem_id in done:
            continue
        # Verify vLLM alive before each problem
        if not vllm_alive():
            print(f"[{i+1}/{len(probs)}] {p.problem_id}: vLLM down, restarting...", flush=True)
            if not restart_vllm():
                print("ABORT: cannot restart vLLM", flush=True)
                break

        t0 = time.time()
        prompt = _build_user_prompt(p)
        text, completion_tokens = http_chat(prompt)
        gen_time = time.time() - t0
        if text is None:
            r = {'problem_id': p.problem_id, 'difficulty': p.difficulty, 'testtype': p.testtype,
                 'pass': False, 'status': 'GEN_FAIL', 'tokens': 0, 'gen_time': gen_time}
        else:
            code = _extract_code(text)
            if not code or len(code) < 20:
                r = {'problem_id': p.problem_id, 'difficulty': p.difficulty, 'testtype': p.testtype,
                     'pass': False, 'status': 'NO_CODE', 'tokens': completion_tokens, 'gen_time': gen_time}
            else:
                try:
                    ok = _check_problem(p, code, timeout_s=10.0)
                    r = {'problem_id': p.problem_id, 'difficulty': p.difficulty, 'testtype': p.testtype,
                         'pass': ok, 'status': 'PASS' if ok else 'FAIL', 'tokens': completion_tokens, 'gen_time': gen_time}
                except Exception as e:
                    r = {'problem_id': p.problem_id, 'difficulty': p.difficulty, 'testtype': p.testtype,
                         'pass': False, 'status': f'VERIF_ERR:{str(e)[:40]}', 'tokens': completion_tokens, 'gen_time': gen_time}
        append_result(r)

        # Progress
        done_count = len(done) + (i + 1 - sum(1 for q in probs[:i] if q.problem_id in done))
        all_done = load_done()
        passed = sum(1 for v in all_done.values() if v.get('pass'))
        elapsed_global = time.time() - t_global
        print(f"[{len(all_done)}/{len(probs)}] {p.problem_id} ({p.difficulty}): {r['status']} | tokens={r['tokens']} | gen={r['gen_time']:.1f}s | overall pass={passed}/{len(all_done)}={100*passed/max(1,len(all_done)):.1f}% | elapsed={elapsed_global/60:.1f}min", flush=True)

    # Final summary
    all_done = load_done()
    passed = sum(1 for v in all_done.values() if v.get('pass'))
    per_diff = {'easy': [0,0], 'medium': [0,0], 'hard': [0,0], 'unknown': [0,0]}
    for v in all_done.values():
        d = v.get('difficulty', 'unknown')
        if d not in per_diff: d = 'unknown'
        per_diff[d][1] += 1
        if v.get('pass'): per_diff[d][0] += 1

    summary = {
        'model': MODEL, 'endpoint': URL, 'sandbox': SANDBOX,
        'thinking': ENABLE_THINKING, 'max_tokens': MAX_TOKENS,
        'total': len(probs), 'completed': len(all_done), 'passed': passed,
        'pass_at_1': passed / max(1, len(all_done)),
        'per_difficulty': {k: {'passed': v[0], 'total': v[1], 'pass_rate': v[0]/max(1,v[1])} for k, v in per_diff.items() if v[1] > 0},
        'wall_s': time.time() - t_global,
    }
    with open(SUMMARY_OUT, 'w') as f:
        json.dump(summary, f, indent=2)
    print(f"\n=== FINAL ===")
    print(f"Pass@1: {passed}/{len(all_done)} = {100*passed/max(1,len(all_done)):.2f}%")
    for d in ['easy', 'medium', 'hard']:
        v = per_diff[d]
        if v[1] > 0:
            print(f"  {d}: {v[0]}/{v[1]} = {100*v[0]/v[1]:.1f}%")

if __name__ == "__main__":
    main()
