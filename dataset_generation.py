# -*- coding: utf-8 -*-
"""
Created on Tue Dec  6 02:24:09 2022

@author: ranad
"""

import cv2
import numpy as np
import glob
import random
from tqdm import tqdm 
from img_construction import construct 

seeds = random.sample(range(0, 30), 30)

if __name__ == "__main__":
    
    transparent_bg = []
    transparent = glob.glob(r'C:\Users\GPU\particle_size_project\Dataset\plastic particles\particles_color\*.png')
    for fname in tqdm(transparent, desc = 'transparent bg resize Progress Bar'):
        fore = cv2.imread(str(fname), cv2.IMREAD_UNCHANGED)
        fore = cv2.resize(fore, (256, 256), interpolation=cv2.INTER_AREA)
        transparent_bg.append(fore)
    
    transparent_mask = []
    transp_mask = glob.glob(r'C:\Users\GPU\particle_size_project\Dataset\plastic particles\particles_mask\*.png')
    for fname in tqdm(transp_mask, desc = 'transparent mask resize Progress Bar'):
        fore = cv2.imread(str(fname), cv2.IMREAD_UNCHANGED)
        fore = cv2.resize(fore, (256, 256), interpolation=cv2.INTER_AREA)
        transparent_mask.append(fore)
    
    for idx in tqdm(range(1), desc = "Generating composited images", unit = "two images"):
        img1 = np.zeros((1088, 1920, 3), np.uint8)
        img1[:] = (211, 211, 211)
        img2 = np.zeros((1088, 1920, 3), np.uint8)
        img1 = construct(transparent_bg, img1, seeds[idx])
        img2 = construct(transparent_mask, img2, seeds[idx])
        cv2.imwrite(r'C:\Users\GPU\particle_size_project\Dataset\examples\Composited_image_transparent_{}.png'.format(idx+256), img1)
        cv2.imwrite(r'C:\Users\GPU\particle_size_project\Dataset\examples\Composited_image_masks_{}.png'.format(idx+256), img2)
        