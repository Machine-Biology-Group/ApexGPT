#!/bin/sh
set -eu

ROOT="$(CDPATH= cd -- "$(dirname "$0")/.." && pwd)"
TARGET="$ROOT/code/apexgpt/checkpoints_base"
ARCHIVE="$TARGET/progen2-large.tar.gz"
URL="https://storage.googleapis.com/anon-progen-research/checkpoints/progen2-large.tar.gz"

mkdir -p "$TARGET"

if [ ! -f "$ARCHIVE" ]; then
  if command -v curl >/dev/null 2>&1; then
    curl -L "$URL" -o "$ARCHIVE"
  elif command -v wget >/dev/null 2>&1; then
    wget -O "$ARCHIVE" "$URL"
  else
    echo "Neither curl nor wget is available." >&2
    exit 1
  fi
fi

tar -xzf "$ARCHIVE" -C "$TARGET"
echo "Extracted ProGen2-large checkpoint to: $TARGET"
