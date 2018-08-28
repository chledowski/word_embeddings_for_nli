#!/usr/bin/env bash

set -e

GREEN='\033[1;32m'
NC='\033[0m' # No Color

CURRENT_DIR=$(dirname "$0")
QUEUE_NAME=$1
QUEUE_FILE="$CURRENT_DIR/queue.txt"
QUEUE_DONE_FILE="$CURRENT_DIR/queue_done.txt"
WAITING_STATE=0

if [[ -z $QUEUE_NAME ]]; then
    printf "${GREEN}Queue name is required.${NC}\n"
    exit 1
fi

printf "${GREEN}Starting queue: ${QUEUE_NAME}${NC}\n"

while true; do
    CMD=$(head -n 1 $QUEUE_FILE)

    if [[ ! -z $CMD ]]; then
        printf "${GREEN}[${QUEUE_NAME}] Running: ${CMD}${NC}\n"
        tail -n +2 "$QUEUE_FILE" > "$QUEUE_FILE.tmp" && mv "$QUEUE_FILE.tmp" "$QUEUE_FILE"
        echo -e "[RUNNING] [${QUEUE_NAME}] $(date) > $CMD" >> "$QUEUE_DONE_FILE"
        if eval $CMD; then
            echo -e "[OK] [${QUEUE_NAME}] $(date) > $CMD" >> "$QUEUE_DONE_FILE"
        else
            echo -e "[FAILED] [${QUEUE_NAME}] $(date) > $CMD" >> "$QUEUE_DONE_FILE"
            echo -e "$CMD # Failed\n$(cat $QUEUE_FILE)" > $QUEUE_FILE
            exit 1
        fi
        WAITING_STATE=0
    else
        if [[ $WAITING_STATE == 0 ]]; then
            printf "${GREEN}[${QUEUE_NAME}] Waiting for commands...${NC}\n"
        fi
        WAITING_STATE=1
    fi

    sleep 5
done