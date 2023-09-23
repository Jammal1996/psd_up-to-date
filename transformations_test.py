# -*- coding: utf-8 -*-
"""
Created on Mon Aug 14 08:28:18 2023

@author: GPU
"""

import glob 
import cv2
import re
from natsort import natsorted
from typing import List, Callable, Tuple
import albumentations as A
import numpy as np
# from sklearn.externals._pilutil import bytescale

from skimage.util import crop

def img_preprocess_grayscale(images_path_string):
    """
    image preprocessing
    """
    images_path = glob.glob(images_path_string)
    images_path = natsorted(images_path)
    imgs = []
    idx = 0
    for idx, fname in enumerate(images_path):
        img = cv2.imread(str(fname), cv2.IMREAD_UNCHANGED)
        # ---------------------------------------------------------
        img = cv2.cvtColor(img, cv2.COLOR_BGRA2BGR)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        if re.search(r'mask', fname):
            _, img = cv2.threshold(img, 50, 255, cv2.THRESH_BINARY)
            # # saving the image        
            # cv2.imwrite(r"small_sample/plastic/(t=250){}.png".format(idx+1), img)
            # img = cv2.threshold(img, 25, 255, cv2.THRESH_BINARY_INV)[1]
            # idx=idx+1
            # # saving the image        
            # cv2.imwrite(r"small_sample/plastic/(t=250){}.png".format(idx+1), img)
            pass
        else:
            # img = cv2.medianBlur(img, 1)
            # clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
            # img = clahe.apply(img)
            img = cv2.medianBlur(img, 1)
            clahe = cv2.createCLAHE(clipLimit=50.0)
            img = clahe.apply(img)
            # cv2.imwrite(r"small_sample/plastic/real/small_sample(c=50){}.png".format(idx+1), img)
        # # saving the image        
        # cv2.imwrite(r"small_sample/plastic/small_sample(c=10,t=75){}.png".format(idx+1), img)
        # idx+=1
        # # ---------------------------------------------------------
        imgs.append(img)

    return imgs