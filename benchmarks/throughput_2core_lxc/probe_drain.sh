#!/bin/bash
set -u
TOKEN=$(cat /root/bench.token)
URL=http://127.0.0.1:7438

echo "=== stats before ==="
curl -s -H "Authorization: Bearer $TOKEN" $URL/v1/stats
echo
echo "=== short burst (4s, concurrency=1, pre-computed embedding) ==="
/root/bench-venv/bin/python3 /root/http_bench.py --url $URL --token $TOKEN --mode single --concurrency 1 --duration 4 --embed-dim 384 2>&1 | tail -10
echo
echo "=== stats t+5s ==="
sleep 5; curl -s -H "Authorization: Bearer $TOKEN" $URL/v1/stats; echo
echo "=== stats t+20s ==="
sleep 15; curl -s -H "Authorization: Bearer $TOKEN" $URL/v1/stats; echo
echo "=== stats t+40s ==="
sleep 20; curl -s -H "Authorization: Bearer $TOKEN" $URL/v1/stats; echo
echo "=== probe single write t+45s ==="
curl -s --max-time 5 -X POST $URL/v1/remember -H "Authorization: Bearer $TOKEN" -H "Content-Type: application/json" -d '{"text": "drain probe"}'
echo
