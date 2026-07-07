#!/usr/bin/env bash
# Refresh the local read-only snapshot (db/snapshot.db) from the EC2 collector DB.
# Uses VACUUM INTO for a consistent copy while the collectors keep writing (WAL-safe).
# Override host/key via env: WX_HOST=root@<ip> WX_KEY=~/.ssh/id_ed25519 ./pull_snapshot.sh
# (the EC2 IP is a Tailscale address — find the current one with `tailscale status`.)
set -e
HOST=${WX_HOST:-root@100.86.140.48}
KEY=${WX_KEY:-$HOME/.ssh/id_ed25519}
REMOTE_DB=/home/ubuntu/apps/polymarket-weather/db/main.db
DEST="$(cd "$(dirname "$0")/.." && pwd)/db/snapshot.db"

ssh -i "$KEY" "$HOST" "sqlite3 $REMOTE_DB \"VACUUM INTO '/tmp/snap.db'\""
scp -i "$KEY" "$HOST:/tmp/snap.db" "$DEST"
ssh -i "$KEY" "$HOST" "rm -f /tmp/snap.db"
echo "snapshot refreshed: $DEST ($(du -h "$DEST" | cut -f1))"
