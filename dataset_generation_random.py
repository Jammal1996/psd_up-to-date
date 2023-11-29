# -*- coding: utf-8 -*-
"""
Created on Fri Aug  4 03:39:57 2023

@author: GPU
"""

import cv2
import numpy as np
import glob
import random
from tqdm import tqdm 
from img_construction_random import construct 

seeds = random.sample(range(0, 2500), 2500)

if __name__ == "__main__":
    h_w_lst_mask = []
    transparent_bg = []
    transparent = glob.glob(r'C:\Users\GPU\particle_size_project\Dataset\Original particles\transparent_og_skin\*.png')
    for fname in tqdm(transparent, desc = 'transparent bg resize Progress Bar'):
        # h_w_lst=[128,192,256,328,392,456,520,584,648,712]
        h_w_lst=[128,192,256,328,392,456,520]
        h=random.choice(h_w_lst)
        w=h
        h_w_lst_mask.append(h)
        fore = cv2.imread(str(fname), cv2.IMREAD_UNCHANGED)
        fore = cv2.resize(fore, (h, w), interpolation=cv2.INTER_AREA)
        transparent_bg.append(fore)
        
    transparent_mask = []
    transp_mask = glob.glob(r'C:\Users\GPU\particle_size_project\Dataset\Original particles\Particles_masks\*.png')
    idx = 0
    for fname in tqdm(transp_mask, desc = 'transparent mask resize Progress Bar'):
        fore = cv2.imread(str(fname), cv2.IMREAD_UNCHANGED)
        fore = cv2.resize(fore, (h_w_lst_mask[idx], h_w_lst_mask[idx]), interpolation=cv2.INTER_AREA)
        transparent_mask.append(fore)
        idx = idx + 1
    for idx in tqdm(range(1), desc = "Generating composited images", unit = "two images"):
        img1 = np.zeros((1088, 1920, 3), np.uint8)
        img1[:] = (211, 211, 211)
        img2 = np.zeros((1088, 1920, 3), np.uint8)
        img1 = construct(transparent_bg, img1, seeds[idx])
        img2 = construct(transparent_mask, img2, seeds[idx])
        cv2.imwrite(r'C:\Users\GPU\particle_size_project\particle_size_distribution\non-machine\samples\Composited_image_transparent{}.png'.format(idx+2), img1)
        cv2.imwrite(r'C:\Users\GPU\particle_size_project\particle_size_distribution\non-machine\samples\Composited_image_masks{}.png'.format(idx+2), img2)