#!/usr/bin/env bash
set -euo pipefail

SLOT="${1:?Usage: ./record_v1_slot.sh SLOT [NUM_EPISODES]}"
NUM_EPISODES="${2:-10}"

if [[ "$SLOT" -lt 0 || "$SLOT" -gt 5 ]]; then
    echo "SLOT must be between 0 and 5"
    exit 1
fi

: "${FOLLOWER_PORT:?FOLLOWER_PORT is not set}"
: "${LEADER_PORT:?LEADER_PORT is not set}"

HF_USER="${HF_USER:-amiyilade}"

TASK="Pick up the tube in slot ${SLOT}."
REPO_ID="${HF_USER}/robotkraft-team3-etiq-v1-slot${SLOT}"

echo "Recording ETIQ V1"
echo "Slot:     ${SLOT}"
echo "Task:     ${TASK}"
echo "Dataset:  ${REPO_ID}"
echo "Episodes: ${NUM_EPISODES}"

lerobot-record   --robot.type=so101_follower   --robot.port="$FOLLOWER_PORT"   --robot.id=follower_arm   --teleop.type=so101_leader   --teleop.port="$LEADER_PORT"   --teleop.id=leader_arm   '--robot.cameras={ top: {type: zmq, server_address: localhost, port: 5555, camera_name: top, fps: 30, width: 1920, height: 1080} }'   --dataset.repo_id="$REPO_ID"   --dataset.single_task="$TASK"   --dataset.num_episodes="$NUM_EPISODES"   --dataset.episode_time_s=15   --dataset.reset_time_s=20   --dataset.push_to_hub=true   --dataset.streaming_encoding=true   --display_data=false   --play_sounds=false