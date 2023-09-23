# -*- coding: utf-8 -*-
"""
Created on Wed Jul 12 07:37:57 2023

@author: GPU
"""

# Import libraries
# --------------------------------------------------------------------
import os
import torch
import numpy as np
from dataset_w import SegmentationDataSet
from visual import save_predictions_as_imgs_bw, save_predictions_as_colored_labels, particle_analysis
from torch.utils.data import DataLoader
from transformations_w import ComposeDouble, FunctionWrapperDouble
from transformations_w import create_dense_target, normalize_01
from skimage.transform import resize
import time

# Initial setup 
# --------------------------------------------------------------------
device = "cuda" if torch.cuda.is_available() else "cpu"
save = True
# --------------------------------------------------------------------

# Params
# --------------------------------------------------------------------
img_height = 736
img_width = 1280
batch_size = 1
num_workers = 0
num_channels = 1
overlap = 0
load_model = True
pin_memory = True
# --------------------------------------------------------------------

# Directories
# --------------------------------------------------------------------
# getting the current directory
curr_dir = os.getcwd()

# getting parent directories 
parent_dir_1 = os.path.dirname(curr_dir)

# the folder path to save the model
saved_model = os.path.join(curr_dir, "best_models")

# the folder to save the history
saved_history = os.path.join(curr_dir, "history")

# Model title 
title = "(w,ln,vgg13,adam,dice)"

# The path where the model is saved
saving_model = os.path.join(saved_model, title + ".pth.tar")

# Image paths
test_dir_img = r"C:\Users\GPU\particle_size_project\particle_size_distribution\thesis\wood.png"
test_maskdir = r"C:\Users\GPU\particle_size_project\Dataset\plastic_mix_250\valid\Masks\*.png"
# --------------------------------------------------------------------

if __name__ == "__main__":
    
    # Loading Data
    validate_transform = ComposeDouble([
        FunctionWrapperDouble(resize,
                              input=True,
                              target=False,
                              output_shape=(img_height, img_width, num_channels)),
        FunctionWrapperDouble(resize,
                              input=False,
                              target=True,
                              output_shape=(img_height, img_width),
                              order=0,
                              anti_aliasing=False,
                              preserve_range=True),
        FunctionWrapperDouble(create_dense_target, input=False, target=True),
        FunctionWrapperDouble(np.moveaxis, input=True, target=False, source=-1, destination=0),
        FunctionWrapperDouble(normalize_01)
    ])
    
    # Loading Model
    print("Loading Checkpoint.........")
    model = torch.load(saving_model).to(device)
    print("Loading model is Successful !")
    
    if save:
        # Saving images
        print("saving images......")
        start_saving = time.time()
        
        # Loading dataset 
        test_dataset = SegmentationDataSet(imgs_dir=test_dir_img, masks_dir=test_maskdir, transform=validate_transform)
        test_loader = DataLoader(test_dataset, batch_size=batch_size, num_workers=num_workers, pin_memory=pin_memory, shuffle=False)
        
        # Saving predicitons as images
        total_preds = save_predictions_as_colored_labels(model, test_loader)
        end_saving = time.time()
        saving_time = (end_saving - start_saving)
        
        # Display the saving time 
        print(f"saving time: {saving_time:.2f} sec")
        
        # particle analysis with colored output image 
        particle_analysis(total_preds)
    
        