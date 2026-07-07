# Refresh the local read-only snapshot (db\snapshot.db) from the EC2 collector DB,
# with a fresh daily-max `edge` table. PowerShell version of pull_snapshot.sh for
# Windows users without bash on PATH. Run from anywhere:
#
#     .\analysis\pull_snapshot.ps1
#
# It SSHes to EC2, runs the snapshot+edge build there (no local bash/sqlite needed),
# and scps the copy down to db\snapshot.db. Overrides: $env:WX_HOST / $env:WX_KEY.
$ErrorActionPreference = "Stop"
$Key    = if ($env:WX_KEY)  { $env:WX_KEY }  else { "$env:USERPROFILE\.ssh\id_ed25519" }
$Remote = if ($env:WX_HOST) { $env:WX_HOST } else { "root@100.86.140.48" }
$App    = "/home/ubuntu/apps/polymarket-weather"
$Dest   = Join-Path (Split-Path $PSScriptRoot -Parent) "db\snapshot.db"

Write-Host "building snapshot + edge on EC2..."
ssh -i $Key $Remote "cd $App && bash analysis/pull_snapshot.sh"
if ($LASTEXITCODE -ne 0) { throw "remote build failed (exit $LASTEXITCODE)" }

# Download to a temp file first, then swap it in — a direct scp over an open
# db\snapshot.db (locked by a DB viewer) fails mid-transfer with a broken pipe.
$Tmp = "$Dest.tmp"
Write-Host "copying to $Dest ..."
scp -i $Key "${Remote}:$App/db/snapshot.db" $Tmp
if ($LASTEXITCODE -ne 0) { throw "scp failed (exit $LASTEXITCODE)" }

try {
    Move-Item -Force $Tmp $Dest
} catch {
    throw "Downloaded to $Tmp but could not replace $Dest — it's likely open in a " +
          "DB viewer (DB Browser / VS Code SQLite extension). Close it and run again, " +
          "or open the .tmp file."
}

$size = "{0:N0} MB" -f ((Get-Item $Dest).Length / 1MB)
Write-Host "done: $Dest ($size)"
