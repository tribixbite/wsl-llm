#!/usr/bin/env bash
# Make the official aider polyglot benchmark runnable OUTSIDE its Docker image.
#
# Two of aider's test commands hardcode container-only absolute paths, so
# JavaScript (and C++) exercises fail on a normal host:
#
#   benchmark.py  TEST_COMMANDS[".js"]  = ["/aider/benchmark/npm-test.sh"]
#   npm-test.sh   symlinks node_modules from /npm-install/
#
# Neither /aider nor /npm-install exists outside the image. This script points
# both at real locations and provisions the shared node_modules that
# npm-test.sh expects (the exercises ship a package.json but no deps).
#
# Python needs nothing. Java needs nothing extra either: benchmark.py already
# strips @Disabled from the JUnit tests (benchmark.py ~line 1016), so the
# exercises are not silently passing on one enabled test — verify that if you
# ever see suspiciously high Java scores.
set -euo pipefail

AIDER_DIR="${AIDER_DIR:-$HOME/aider}"
NPM_INSTALL_DIR="${NPM_INSTALL_DIR:-$HOME/npm-install}"
POLYGLOT="${POLYGLOT:-$HOME/polyglot-benchmark}"

echo "=== 1. shared node_modules for the JS exercises ==="
mkdir -p "$NPM_INSTALL_DIR"
if [[ ! -d "$NPM_INSTALL_DIR/node_modules" ]]; then
  cp "$POLYGLOT/javascript/exercises/practice/affine-cipher/package.json" "$NPM_INSTALL_DIR/"
  (cd "$NPM_INSTALL_DIR" && npm install --silent)
fi
du -sh "$NPM_INSTALL_DIR/node_modules"

echo "=== 2. de-containerise npm-test.sh ==="
python3 - "$AIDER_DIR" <<'PY'
import pathlib, sys
d = pathlib.Path(sys.argv[1])
p = d / "benchmark" / "npm-test.sh"
s = p.read_text()
if "NPM_INSTALL_DIR" not in s:
    s = s.replace('[ ! -e node_modules ] && ln -s /npm-install/node_modules .',
                  'NPM_INSTALL_DIR="${NPM_INSTALL_DIR:-/npm-install}"\n'
                  '[ ! -e node_modules ] && ln -s "$NPM_INSTALL_DIR/node_modules" .')
    s = s.replace('[ ! -e package-lock.json ] && ln -s /npm-install/package-lock.json .',
                  '[ ! -e package-lock.json ] && ln -s "$NPM_INSTALL_DIR/package-lock.json" .')
    p.write_text(s)
    print("  patched npm-test.sh")
else:
    print("  npm-test.sh already patched")

p = d / "benchmark" / "benchmark.py"
s = p.read_text()
if '"/aider/benchmark/npm-test.sh"' in s:
    s = s.replace('        ".js": ["/aider/benchmark/npm-test.sh"],\n'
                  '        ".cpp": ["/aider/benchmark/cpp-test.sh"],',
                  '        ".js": [str(Path(__file__).parent / "npm-test.sh")],\n'
                  '        ".cpp": [str(Path(__file__).parent / "cpp-test.sh")],')
    p.write_text(s)
    print("  patched benchmark.py TEST_COMMANDS")
else:
    print("  benchmark.py already patched")
PY

echo "=== 3. smoke-test the JS harness on a reference solution ==="
T=$(mktemp -d)
cp -r "$POLYGLOT/javascript/exercises/practice/affine-cipher/." "$T/"
cp "$T/.meta/proof.ci.js" "$T/affine-cipher.js" 2>/dev/null || \
  cp "$T/.meta/example.js" "$T/affine-cipher.js" 2>/dev/null || true
( cd "$T" && NPM_INSTALL_DIR="$NPM_INSTALL_DIR" "$AIDER_DIR/benchmark/npm-test.sh" 2>&1 | tail -4 )
rm -rf "$T"
echo "=== ready: python, javascript, java ==="
