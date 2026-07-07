#!/usr/bin/env bash
# Build a read-only snapshot (db/snapshot.db) of the collector DB with a fresh
# daily-max `edge` table, for inspection. Works whether you run it:
#   * ON the EC2 host  -> builds locally, no SSH.
#   * on your machine  -> SSHes to EC2, builds there, scps the copy down.
# Either way it:
#   1. VACUUM INTOs a consistent copy of main.db (read-only, safe mid-cycle while
#      the collectors hold a long write transaction),
#   2. builds the `edge` table INTO that private copy (no writers -> no lock),
#   3. leaves it at db/snapshot.db (next to this repo).
#
# Remote overrides: WX_HOST=root@<ip> WX_KEY=~/.ssh/id_ed25519 ./pull_snapshot.sh
# (the EC2 IP is a Tailscale address — find the current one with `tailscale status`.)
set -e
HOST=${WX_HOST:-root@100.86.140.48}
KEY=${WX_KEY:-$HOME/.ssh/id_ed25519}
APP=/home/ubuntu/apps/polymarket-weather
DEST="$(cd "$(dirname "$0")/.." && pwd)/db/snapshot.db"

if [ -f "$APP/db/main.db" ]; then
    # --- Running ON the EC2 host: build locally, no SSH. ---
    echo "on EC2: building snapshot + edge locally..."
    rm -f "$DEST"
    sqlite3 "$APP/db/main.db" "VACUUM INTO '$DEST'"
    ( cd "$APP" && "$APP/venv/bin/python" -m analysis.edge --db "$DEST" )
    echo "snapshot ready (with edge table): $DEST ($(du -h "$DEST" | cut -f1))"
else
    # --- Running on a remote machine: pull from EC2. ---
    echo "creating consistent snapshot on EC2..."
    ssh -i "$KEY" "$HOST" "sqlite3 $APP/db/main.db \"VACUUM INTO '/tmp/snap.db'\""
    echo "building edge table on the snapshot..."
    ssh -i "$KEY" "$HOST" "cd $APP && venv/bin/python -m analysis.edge --db /tmp/snap.db"
    echo "copying db to $DEST ..."
    scp -i "$KEY" "$HOST:/tmp/snap.db" "$DEST"
    ssh -i "$KEY" "$HOST" "rm -f /tmp/snap.db"
    echo "snapshot refreshed (with edge table): $DEST ($(du -h "$DEST" | cut -f1))"
fi
