import io
from PIL import Image
import cv2
import numpy as np
from torch.utils.data import Dataset

try:
    from acs.aoss.client import Client
except:
    try:
        from petrel_client.client import Client
    except:
        print('no ceph client is found')


DEFAULT_CONF_PATH = "~/.tcs/petreloss_enable_vcs.conf"

import logging
import os

LOG_TCS = os.environ.get("LOG_TCS", False)


def pil_loader(img_str):
    buff = io.BytesIO(img_str)
    return Image.open(buff)


def cv2_loader(img_bytes):
    # assert(img_bytes is not None)
    img_mem_view = memoryview(img_bytes)
    img_array = np.frombuffer(img_mem_view, np.uint8)
    imgcv2 = cv2.imdecode(img_array, cv2.IMREAD_COLOR)
    imgcv2 = cv2.cvtColor(imgcv2, cv2.COLOR_BGR2RGB)
    return Image.fromarray(imgcv2)


class TCSLoader(object):

    def __init__(self, conf_path):
        self.client = Client(conf_path)

    def __call__(self, fn):
        try:
            img_value_str = self.client.get(fn)
            img = pil_loader(img_value_str)
        except:
            print('Read image failed ({})'.format(fn))
            return None
        else:
            return img


class CephDataset(Dataset):
    def __init__(self, tcs_conf_path=None) -> None:
        super().__init__()
        if tcs_conf_path is None:
            tcs_conf_path = DEFAULT_CONF_PATH
        self.tcs_conf_path = tcs_conf_path
        self.initialized = False
        self._init_memcached()

    def _init_memcached(self):
        if not self.initialized:
            assert self.tcs_conf_path is not None
            self.loader = TCSLoader(self.tcs_conf_path)
            self.client = self.loader.client
            self.initialized = True

    def __getitem__(self, index):
        raise NotImplementedError
