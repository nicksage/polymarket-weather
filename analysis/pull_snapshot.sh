#!/usr/bin/env bash
# Refresh the local read-only snapshot (db/snapshot.db) from the EC2 collector DB.
# First (re)builds the daily-max `edge` table on EC2 so the copy you inspect
# locally has fresh edges, then VACUUM INTOs a consistent copy (WAL-safe while
# the collectors keep writing) and scps it down.
#
# Override host/key via env: WX_HOST=root@<ip> WX_KEY=~/.ssh/id_ed25519 ./pull_snapshot.sh
# (the EC2 IP is a Tailscale address — find the current one with `tailscale status`.)
set -e
HOST=${WX_HOST:-root@100.86.140.48}
KEY=${WX_KEY:-$HOME/.ssh/id_ed25519}
APP=/home/ubuntu/apps/polymarket-weather
DEST="$(cd "$(dirname "$0")/.." && pwd)/db/snapshot.db"

echo "building edge table on EC2..."
ssh -i "$KEY" "$HOST" "cd $APP && venv/bin/python -m analysis.edge" \
    || echo "WARN: edge build failed on EC2 — pulling snapshot without fresh edge"

echo "copying db to $DEST ..."
ssh -i "$KEY" "$HOST" "sqlite3 $APP/db/main.db \"VACUUM INTO '/tmp/snap.db'\""
scp -i "$KEY" "$HOST:/tmp/snap.db" "$DEST"
ssh -i "$KEY" "$HOST" "rm -f /tmp/snap.db"
echo "snapshot refreshed (with edge table): $DEST ($(du -h "$DEST" | cut -f1))"
