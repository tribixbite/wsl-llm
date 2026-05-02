"""Test N LCB problems with FULL OUTPUT CAPTURE for debugging."""
import sys, json, os, time, urllib.request
sys.path.insert(0, '/home/matilda/git/SI/src')
from si.livecodebench import load_lcb, _build_user_prompt, _extract_code, _check_problem, _SYSTEM_LCB
from sandbox_fusion import set_endpoint
set_endpoint('http://localhost:45233')

URL = "http://192.168.1.32:8081"
MODEL = "qwen3.6-27b"
N = int(sys.argv[1]) if len(sys.argv) > 1 else 20
ENABLE_THINKING = sys.argv[2].lower() == 'true' if len(sys.argv) > 2 else False
MAX_TOKENS = int(sys.argv[3]) if len(sys.argv) > 3 else 16384
OUT_DIR = sys.argv[4] if len(sys.argv) > 4 else "/tmp/lcb_debug_outputs"

os.makedirs(OUT_DIR, exist_ok=True)
print(f"Testing {N} problems, thinking={ENABLE_THINKING}, max_tokens={MAX_TOKENS}")

probs = load_lcb()[:N]
results = []
for p in probs:
    print(f"=== {p.problem_id} ({p.difficulty}) ===", flush=True)
    body = json.dumps({
        'model': MODEL,
        'messages': [
            {'role':'system','content':_SYSTEM_LCB},
            {'role':'user','content':_build_user_prompt(p)}
        ],
        'max_tokens': MAX_TOKENS,
        'temperature': 0.6 if ENABLE_THINKING else 0.2,
        'top_p': 0.95,
        'chat_template_kwargs':{'enable_thinking': ENABLE_THINKING},
    }).encode()
    req = urllib.request.Request(f'{URL}/v1/chat/completions', data=body,
                                 headers={'Content-Type':'application/json'})
    t0 = time.time()
    try:
        with urllib.request.urlopen(req, timeout=600) as r:
            d = json.load(r)
        elapsed = time.time() - t0
        text = d['choices'][0]['message']['content']
        completion_tokens = d.get('usage', {}).get('completion_tokens', 0)
        with open(f"{OUT_DIR}/{p.problem_id}.txt", 'w') as f:
            f.write(text)
        code = _extract_code(text)
        has_code = bool(code and len(code) > 20)
        if has_code:
            try:
                ok = _check_problem(p, code, timeout_s=10.0)
                status = "PASS" if ok else "FAIL"
            except Exception as e:
                status = "VERIF_ERR"; ok = False
        else:
            status = "NO_CODE"; ok = False
        print(f"  {status} tokens={completion_tokens} code_len={len(code) if code else 0} text_len={len(text)} {elapsed:.1f}s", flush=True)
        results.append({'problem_id': p.problem_id, 'difficulty': p.difficulty, 'pass': ok, 'status': status, 'tokens': completion_tokens, 'code_len': len(code) if code else 0, 'elapsed': elapsed})
    except Exception as e:
        print(f"  ERROR: {e}", flush=True)
        results.append({'problem_id': p.problem_id, 'error': str(e)})

passed = sum(1 for r in results if r.get('pass'))
print(f"\nPassed: {passed}/{N} = {100*passed/N:.1f}%")
per_diff = {}
for r in results:
    d = r.get('difficulty', 'unknown')
    if d not in per_diff: per_diff[d] = [0, 0]
    per_diff[d][1] += 1
    if r.get('pass'): per_diff[d][0] += 1
for d, (p, t) in per_diff.items():
    if t > 0: print(f"  {d}: {p}/{t} = {100*p/t:.1f}%")

with open(f"{OUT_DIR}/summary.json", 'w') as f:
    json.dump({'thinking': ENABLE_THINKING, 'max_tokens': MAX_TOKENS, 'n': N, 'passed': passed, 'per_difficulty': per_diff, 'results': results}, f, indent=2)
