#!/usr/bin/env sh
set -eu

REPO_DIR="$(CDPATH= cd -- "$(dirname -- "$0")/.." && pwd)"
PARENT_DIR="$(dirname "$REPO_DIR")"
REPO_NAME="$(basename "$REPO_DIR")"
STAMP="$(date +%Y%m%d)"
ARCHIVE_PATH="$PARENT_DIR/${REPO_NAME}_${STAMP}.tar.zst"
CHECKSUM_PATH="${ARCHIVE_PATH}.sha256"

tar --exclude='.git' --zstd -cf "$ARCHIVE_PATH" -C "$PARENT_DIR" "$REPO_NAME"
sha256sum "$ARCHIVE_PATH" > "$CHECKSUM_PATH"

printf 'Archive: %s\n' "$ARCHIVE_PATH"
printf 'SHA256: %s\n' "$CHECKSUM_PATH"
