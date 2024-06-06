## Dependencies
We recommend to use the following packages
* python 3.10
* pytorch==1.13.0
* torchvision==0.14.0
* numpy==1.23.0
* timm==0.6.12
* mamba-ssm==1.2.0
* mmcv==2.1.0 (build from source is required)
* mmdet==3.3


## Data preparation
Please prepare COCO dataset according to the instruction in [MMDet](https://mmdetection.readthedocs.io/en/latest/user_guides/dataset_prepare.html).


## Training
We train our base model on 8 gpus. It takes around 40 hours for 12 epochs.

Before training, please convert the pre-trained model to mmdet format with [convert_script](tools/convert_cls_to_det.py). Please set the pre-trained model path in the config (`pretrained` attribute).
### Training with torch.distributed.launch
To train on multi-nodes with torch.distributed.launch, run the commands below.

```
  sh ./scripts/dist_train.sh 8 1 ${CONFIG}
```
Note:
The first and second arguments specify the ${GPUS_PER_NODE} and ${NNODE} respectively. You need to adjust them if different node numbers are used.

### Training with slurm
If you need to run the training on a slurm cluster, use the command below to run on `${GPUS}/${GPUS_PER_NODE}` nodes with `${GPUS_PER_NODE}` gpus on each node:
```
  sh ./scripts/slurm_train.sh ${PARTITION} ${GPUS} ${GPUS_PER_NODE} ${CONFIG}
```

## Inference
For inference, the input parameters are the same as training commands, except an extra checkpoint path is needed.
### Testing with torch.distributed.launch
Run:
```
  sh ./scripts/dist_test.sh 8 1 ${CONFIG} ${CHECKPOINT}
```
### Testing with slurm
Run:
```
  sh ./scripts/slurm_test.sh ${PARTITION} ${GPUS} ${GPUS_PER_NODE} ${CONFIG} ${CHECKPOINT}
```
