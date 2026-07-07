#!/usr/bin/env bash
# Refresh the local read-only snapshot (db/snapshot.db) from the EC2 collector DB.
#   1. VACUUM INTO a consistent copy on EC2 (read-only on main.db, so it is safe
#      even mid-cycle while the collectors hold a long write transaction).
#   2. Build the daily-max `edge` table INTO that private copy (no writers there,
#      so no lock contention with the live collectors).
#   3. scp the copy down to db/snapshot.db.
#
# Override host/key via env: WX_HOST=root@<ip> WX_KEY=~/.ssh/id_ed25519 ./pull_snapshot.sh
# (the EC2 IP is a Tailscale address — find the current one with `tailscale status`.)
set -e
HOST=${WX_HOST:-root@100.86.140.48}
KEY=${WX_KEY:-$HOME/.ssh/id_ed25519}
APP=/home/ubuntu/apps/polymarket-weather
DEST="$(cd "$(dirname "$0")/.." && pwd)/db/snapshot.db"

echo "creating consistent snapshot on EC2..."
ssh -i "$KEY" "$HOST" "sqlite3 $APP/db/main.db \"VACUUM INTO '/tmp/snap.db'\""

echo "building edge table on the snapshot..."
ssh -i "$KEY" "$HOST" "cd $APP && venv/bin/python -m analysis.edge --db /tmp/snap.db"

echo "copying db to $DEST ..."
scp -i "$KEY" "$HOST:/tmp/snap.db" "$DEST"
ssh -i "$KEY" "$HOST" "rm -f /tmp/snap.db"
echo "snapshot refreshed (with edge table): $DEST ($(du -h "$DEST" | cut -f1))"
