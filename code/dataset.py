from __future__ import division
from __future__ import print_function

import os

from skimage import io
from torch.utils.data import Dataset
import random
import pandas as pd
import cv2
import numpy as np
from PIL import Image

class JAFFEDataset(Dataset):
    def __init__(self, root_dir: str, transform=None):
        self.root_dir = root_dir
        self.pics_list = os.listdir(self.root_dir)
        self.transform = transform

    def __len__(self):
        return len(self.pics_list)

    def __getitem__(self, idx):
        
        img_name = os.path.join(self.root_dir, self.pics_list[idx])
        if "AN" in self.pics_list[idx]:
            target = 0
        elif "DI" in self.pics_list[idx]:
            target = 1
        elif "FE" in self.pics_list[idx]:
            target = 2
        elif "HA" in self.pics_list[idx]:
            target = 3
        elif "SA" in self.pics_list[idx]:
            target = 4
        elif "SU" in self.pics_list[idx]:
            target = 5
        elif "NE" in self.pics_list[idx]:
            target = 6
        image = io.imread(img_name)
        if self.transform:
            image = self.transform(image)
        sample = {"image": image, "target": target}

        return sample

