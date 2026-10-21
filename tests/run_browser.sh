#!/usr/bin/env bash
# Runs tests/runner.html in headless Chrome and saves the JSON it writes.
# Needs a static server on :8780 serving hw6/ (python3 -m http.server 8780).
# usage: tests/run_browser.sh <output.json> '<query string>'
out="$1"; query="$2"
CHROME="${CHROME:-/Applications/Google Chrome.app/Contents/MacOS/Google Chrome}"
start=$(date +%s)
"$CHROME" --headless=new --no-sandbox --virtual-time-budget=36000000 \
  --dump-dom "http://localhost:8780/tests/runner.html?${query}" 2>/dev/null \
  | python3 -c 'import sys,re,html; print(html.unescape(re.search(r"<pre id=\"result\">(.*?)</pre>", sys.stdin.read(), re.S).group(1)))' > "$out"
echo "{\"wallSeconds\": $(( $(date +%s) - start )), \"query\": \"${query}\"}" > "${out%.json}.time.json"
