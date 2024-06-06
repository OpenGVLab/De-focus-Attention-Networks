#!/usr/bin/env bash

set -x

PARTITION=${1}
GPUS=${2}
GPUS_PER_NODE=${3}
CONFIG=${4}
CKPT=${5}
SRUN_ARGS=${SRUN_ARGS:-""}

CFGNAME=$(basename ${CONFIG} .yaml)
DIR=./work_dirs/${CFGNAME}
mkdir -p ${DIR}

SUFFIX=$(date '+%Y%m%d%H%M')


srun -p ${PARTITION} \
    ${SRUN_ARGS} \
    --cpus-per-task=8 \
    --gres=gpu:${GPUS_PER_NODE} \
    -n${GPUS} \
    --ntasks-per-node=${GPUS_PER_NODE} \
    --quotatype=spot \
    python -u main.py \
    --cfg ${CONFIG} \
    --output ${DIR} \
    --eval \
    --pretrained ${CKPT} \
    ${PY_ARGS} 2>&1 | tee -a ${DIR}/stdout_${SUFFIX}.log