#!/usr/bin/env bash

set -x
set -e

CURRENT_DIR=$(dirname "$0")
QUEUE_FILE="$CURRENT_DIR/queue.txt"
QUEUE_DONE_FILE="$CURRENT_DIR/queue_done.txt"

while true; do
    CMD=$(head -n 1 $QUEUE_FILE)
    tail -n +2 "$QUEUE_FILE" > "$QUEUE_FILE.tmp" && mv "$QUEUE_FILE.tmp" "$QUEUE_FILE"
    echo "$(date) > $CMD" >> "$QUEUE_DONE_FILE"
    eval $CMD

    sleep 5
done