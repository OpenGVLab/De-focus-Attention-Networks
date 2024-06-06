
import os
import PIL

import torch
from torchvision import transforms
from timm.data.constants import IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD
import numpy as np

import io
from PIL import Image
import pyarrow as pa
from io import BytesIO
import tqdm
from tqdm import trange

from .CephDataset import CephDataset


def _get_images(annotations):
    images = []
    classes = []
    for line in annotations:
        if isinstance(line, bytes):
            line = line.decode()
        image_name, cls = line.strip('\n').split()
        images.append(image_name)
        classes.append(cls)
    return images, classes


class ImageNetDataset(CephDataset):
    def __init__(self, data_path, image_set, transform=None, 
                 tcs_conf_path=None,
                 on_memory=False, local_rank=None, local_size=None,
                 **kwargs):
        if tcs_conf_path is not None:
            super().__init__(tcs_conf_path)
        ann_file = os.path.join(data_path, f'meta/{image_set}.txt')
        data_path = os.path.join(data_path, image_set)
        self.image_set = image_set
        self.transform = transform
        self.data_path = data_path

        self.use_tcs = True if tcs_conf_path is not None else False
        self.images, self.classes, self.class_to_idx = self._load_database(ann_file)
        self.on_memory = on_memory
        if on_memory:
            if local_rank is None:
                local_rank = int(os.environ.get('LOCAL_RANK', 0))
            if local_size is None:
                local_size = int(os.environ.get('LOCAL_SIZE', 1))
            self.local_rank = local_rank
            self.local_size = local_size
            self.holder = {}
            self.load_onto_memory()
        print(f"length of the dataset is {len(self.images)}")

    def load_onto_memory(self):
        print("Loading images onto memory...")
        for index in trange(len(self.images)):
            if index % self.local_size != self.local_rank:
                continue
            path = self.images[index].as_py()
            full_path = os.path.join(self.data_path, path)
            if self.use_tcs:
                sample = self.loader(full_path)
            else:
                with open(full_path, 'rb') as f:
                    sample = f.read()
            self.holder[path] = sample
        # print('Loading: path {}, full_path {}, data length {}'.format(path, full_path, 
        #                                                               len(self.tcs_loader.client.get(full_path))))
        print("Loading complete!")

    def _load_database(self, annotation_file):
        if not self.use_tcs:
            annotation_file = os.path.abspath(annotation_file)
        print(f'loading annotations from {annotation_file} ...')
        if self.use_tcs:
            with BytesIO(self.loader.client.get(annotation_file)) as annotations:
                images, classes = _get_images(annotations)
        else:
            with open(annotation_file, 'rt') as annotations:
                images, classes = _get_images(annotations)

        # convert possible classes to indices
        class_names = sorted(set(classes))
        # class_to_idx = {class_name: idx for idx, class_name in enumerate(class_names)}
        class_to_idx = {class_name: int(class_name) for class_name in class_names}
        return pa.array(images), pa.array([class_to_idx[class_name] for class_name in classes]), class_to_idx

    def __len__(self):
        return len(self.images)

    def __getitem__(self, index):
        path = self.images[index].as_py()
        target = self.classes[index].as_py()
        sample = self._load_image(path)
        if self.transform is not None:
            sample = self.transform(sample)
        return sample, target

    def _load_image(self, path):
        full_path = os.path.join(self.data_path, path)
        if self.on_memory:
            try:
                return Image.open(BytesIO(self.holder[path])).convert('RGB')
            except:
                print('error acquiring data from {}'.format(path))
                return self.loader(full_path).convert('RGB')
        elif self.use_tcs:
            return self.loader(full_path).convert('RGB')
        else:
            with open(full_path, 'rb') as f:
                return Image.open(f).convert('RGB')

    def __repr__(self) -> str:
        return f"ImageNet1k Dataset phase={self.image_set} " \
            f"data_root={self.data_path}\n" \
            f"transform={self.transform}"
