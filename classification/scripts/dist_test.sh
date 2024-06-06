#!/usr/bin/env bash

set -x

MASTER_ADDR=${1}
RANK=${2}
NNODES=${3}
CONFIG=${4}
CKPT=${5}
MASTER_PORT=${PORT:-28500}
PY_ARGS=${PY_ARGS:-""}

CFGNAME=$(basename ${CONFIG} .yaml)
DIR=./work_dirs/${CFGNAME}
mkdir -p ${DIR}

SUFFIX=$(date '+%Y%m%d%H%M')


python -m torch.distributed.launch \
    --nproc_per_node=8 \
    --nnodes=${NNODES} \
    --node_rank=${RANK} \
    --master_addr=${MASTER_ADDR} \
    --master_port=${MASTER_PORT} \
    main.py \
    --cfg ${CONFIG} \
    --output ${DIR} \
    --eval \
    --pretrained ${CKPT} \
    ${PY_ARGS} 2>&1 | tee -a ${DIR}/stdout_${SUFFIX}.log