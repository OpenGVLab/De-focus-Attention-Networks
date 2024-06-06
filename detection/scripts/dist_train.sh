#!/usr/bin/env bash

set -x

GPUS_PER_NODE=$1
NNODES=$2
CONFIG=$3
NODE_RANK=${NODE_RANK:-0}
PORT=${PORT:-29500}
MASTER_ADDR=${MASTER_ADDR:-"127.0.0.1"}
PY_ARGS=${PY_ARGS:-""}


PYTHONPATH="$(dirname $0)/..":$PYTHONPATH \
python -m torch.distributed.launch \
    --nnodes=$NNODES \
    --node_rank=$NODE_RANK \
    --master_addr=$MASTER_ADDR \
    --nproc_per_node=$GPUS_PER_NODE \
    --master_port=$PORT \
    ./scripts/train.py \
    $CONFIG \
    --launcher pytorch $PY_ARGS