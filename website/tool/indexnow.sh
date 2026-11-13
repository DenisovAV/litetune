#!/usr/bin/env bash
# Tell IndexNow search engines (Bing and the others that share submissions)
# that the pages in the live sitemap changed. Run it by hand after a deploy:
#
#   website/tool/indexnow.sh
#
# The key is the one file web/<32 hex>.txt, which holds its own name; the
# engine fetches https://litetune.dev/<key>.txt and accepts the submission only
# if it matches, so the deploy that carries the key has to be live first.
set -euo pipefail

HOST="litetune.dev"
WEB_DIR="$(cd "$(dirname "$0")/../web" && pwd)"

key_file=$(find "$WEB_DIR" -maxdepth 1 -regex '.*/[0-9a-f]\{32\}\.txt')
[ "$(printf '%s\n' "$key_file" | grep -c .)" -eq 1 ] \
  || { echo "expected one key file in $WEB_DIR, found: ${key_file:-none}" >&2; exit 1; }
key=$(basename "$key_file" .txt)

live=$(curl -fsS "https://$HOST/$key.txt") \
  || { echo "https://$HOST/$key.txt is not live yet; deploy first" >&2; exit 1; }
[ "$live" = "$key" ] || { echo "https://$HOST/$key.txt does not hold the key" >&2; exit 1; }

urls=$(curl -fsS "https://$HOST/sitemap.xml" | sed -n 's:.*<loc>\(.*\)</loc>.*:\1:p')
[ -n "$urls" ] || { echo "no <loc> in https://$HOST/sitemap.xml" >&2; exit 1; }

body=$(python3 - "$HOST" "$key" "$urls" <<'EOF'
import json, sys
host, key, urls = sys.argv[1], sys.argv[2], sys.argv[3].split()
print(json.dumps({"host": host, "key": key,
                  "keyLocation": f"https://{host}/{key}.txt", "urlList": urls}))
EOF
)

# 200 and 202 both mean accepted (202: the key is still being checked).
code=$(curl -sS -o /dev/stderr -w '%{http_code}' -X POST https://api.indexnow.org/indexnow \
  -H 'Content-Type: application/json; charset=utf-8' --data "$body")
echo "IndexNow answered $code for $(printf '%s\n' "$urls" | grep -c .) URLs"
case "$code" in 200|202) ;; *) exit 1 ;; esac
