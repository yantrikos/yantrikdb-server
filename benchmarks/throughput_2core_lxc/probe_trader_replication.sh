#!/bin/bash
set -u

DB_DEFAULT=/opt/yantrikdb-trader/data/default/yantrik.db
DB_LEDGER=/opt/yantrikdb-trader/data/trader_ledger/yantrik.db

echo "=== default DB ==="
echo "-- sync_peers count --"
sqlite3 "$DB_DEFAULT" "SELECT COUNT(*) FROM sync_peers;"
echo "-- sync_peers rows --"
sqlite3 "$DB_DEFAULT" "SELECT * FROM sync_peers;"
echo "-- oplog by origin_actor --"
sqlite3 "$DB_DEFAULT" "SELECT origin_actor, COUNT(*) FROM oplog GROUP BY origin_actor ORDER BY 2 DESC LIMIT 10;"
echo "-- meta actor_id --"
sqlite3 "$DB_DEFAULT" "SELECT * FROM meta WHERE key='actor_id';"
echo "-- meta node_id --"
sqlite3 "$DB_DEFAULT" "SELECT * FROM meta WHERE key='node_id';"

echo
echo "=== trader_ledger DB ==="
echo "-- sync_peers count --"
sqlite3 "$DB_LEDGER" "SELECT COUNT(*) FROM sync_peers;"
echo "-- oplog by origin_actor --"
sqlite3 "$DB_LEDGER" "SELECT origin_actor, COUNT(*) FROM oplog GROUP BY origin_actor ORDER BY 2 DESC LIMIT 10;"
echo "-- meta actor_id --"
sqlite3 "$DB_LEDGER" "SELECT * FROM meta WHERE key='actor_id';"
