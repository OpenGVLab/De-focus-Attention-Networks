## Dependencies
We recommend to use the following packages
* python 3.10
* pytorch==1.13.0
* torchvision==0.14.0
* numpy==1.23.0
* timm==0.6.12
* mamba-ssm==1.2.0


## Data preparation

Download and extract ImageNet train and val images from http://image-net.org/.
The directory structure is the standard layout for the torchvision [`datasets.ImageFolder`](https://pytorch.org/docs/stable/torchvision/datasets.html#imagefolder), and the training and validation data is expected to be in the `train/` folder and `val` folder respectively:

```
  /path/to/imagenet/
      ├── train/
      │   ├── class1/
      │   │   ├── img1.JPEG
      |   │   ├── img2.JPEG
      |   │   ├── img3.JPEG
      |   │   └── ...
      │   ├── class2/
      |   │   └── ...   
      │   ├── class3/
      |   │   └── ...
      |   └── ...
      └─── val
      │   ├── class1/
      │   │   ├── img4.JPEG
      |   │   ├── img5.JPEG
      |   │   ├── img6.JPEG
      |   │   └── ...
      │   ├── class2/
      |   │   └── ...   
      │   ├── class3/
      |   │   └── ...
```

Note that raw val images are not put into class folders, use [this script](https://github.com/pytorch/examples/blob/main/imagenet/extract_ILSVRC.sh) to get correct layout.

Finally, remember to update the dataset path in the config (`DATA:DATA_PATH` attribute).


## Training
### Training with torch.distributed.launch
To train on multi-nodes with torch.distributed.launch, run the commands below (16 gpus on 2 nodes).

On node 1:
```
  sh ./scripts/dist_train.sh ${MASTER_ADDR} 0 2 ${CONFIG}
```
On node 2:
```
  sh ./scripts/dist_train.sh ${MASTER_ADDR} 1 2 ${CONFIG}
```
Note:
The `${MASTER_ADDR}` is the ip address of rank 0 node. The second and third arguments specify the node rank and node number respectively. You need to adjust them if different node numbders are used.

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
  sh ./scripts/dist_test.sh ${MASTER_ADDR} 0 1 ${CONFIG} ${CHECKPOINT}
```
### Testing with slurm
Run:
```
  sh ./scripts/slurm_test.sh ${PARTITION} ${GPUS} ${GPUS_PER_NODE} ${CONFIG} ${CHECKPOINT}
```
