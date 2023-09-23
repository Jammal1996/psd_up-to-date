# -*- coding: utf-8 -*-
"""
Created on Tue Dec  6 02:11:16 2022

@author: ranad
"""
    
import torch
from torch.utils import data
from transformations_w import img_preprocess_grayscale
    
class SegmentationDataSet(data.Dataset):
    # We expect a list of input paths and a target paths 
    def __init__(self,
                 imgs_dir: str,
                 masks_dir: str,
                 transform=None
                 ):
        self.inputs = img_preprocess_grayscale(imgs_dir)
        self.targets = img_preprocess_grayscale(masks_dir)
        self.transform = transform
        self.inputs_dtype = torch.float32
        self.targets_dtype = torch.long
    
    def __len__(self):
        return len(self.inputs)
    
    # Reading inputs from the input and target lists 
    def __getitem__(self,
                    index: int):
        # Select the sample
        x = self.inputs[index]
        y = self.targets[index]

        # Preprocessing
        if self.transform is not None:
            x, y = self.transform(x, y)
        
        # Typecasting
        x, y = torch.from_numpy(x).type(self.inputs_dtype), torch.from_numpy(y).type(self.targets_dtype)

        return x, y