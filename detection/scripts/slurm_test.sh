#!/usr/bin/env bash

set -x

PARTITION=$1
GPUS=$2
GPUS_PER_NODE=$3
CONFIG=$4
CKPT=$5
CPUS_PER_TASK=${CPUS_PER_TASK:-5}
SRUN_ARGS=${SRUN_ARGS:-""}
PY_ARGS=${PY_ARGS:-""}


PYTHONPATH="$(dirname $0)/..":$PYTHONPATH \
srun -p ${PARTITION} \
    --job-name=det_test \
    --gres=gpu:${GPUS_PER_NODE} \
    --ntasks=${GPUS} \
    --ntasks-per-node=${GPUS_PER_NODE} \
    --cpus-per-task=${CPUS_PER_TASK} \
    --kill-on-bad-exit=1 \
    ${SRUN_ARGS} \
    python -u scripts/test.py ${CONFIG} ${CKPT} --launcher="slurm" ${PY_ARGS}