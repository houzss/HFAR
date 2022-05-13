import os

from torch.utils.data import Dataset
from torchvision.io import read_image,ImageReadMode

import pandas as pd
import torch
import numpy as np
class SketchData(Dataset):
    def __init__(self, annotations_file, img_dir, transform=None):
        self.img_labels = pd.read_csv(annotations_file,header=None).drop(0)
        self.img_dir = img_dir
        self.transform = transform

    def __len__(self):
        return len(self.img_labels)

    def __getitem__(self, idx):
        img_path = os.path.join(self.img_dir, self.img_labels.iloc[idx,0])
        image = read_image(img_path,mode=ImageReadMode.RGB).type(torch.float32)#因为Png图片比JPG图片多一个Alpha通道，故需要指定Mode:Image Read ModeUNCHANGED = 0,
        # GRAY = 1,GRAY_ALPHA = 2,RGB = 3,RGB_ALPHA = 4
        label = torch.LongTensor(np.array(self.img_labels.iloc[idx,1:].values,dtype=np.int))
        if self.transform:
            image = self.transform(image)
        return image, label



