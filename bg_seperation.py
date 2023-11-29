# -*- coding: utf-8 -*-
"""
Created on Mon Feb 13 13:41:35 2023

@author: GPU
"""

import cv2
import numpy as np
import glob
from tqdm import tqdm

def bg_separation(particles):
    counter = 0
    sep_imgs = []
    sep_masks = []
    idx = 0
    for img_name in tqdm(particles, desc = "background seperation", unit = "two images"):
        counter += 1
        
        # Creating a seperated background image 
        # ------------------------------------------------------------------------------------------------------
        
        # Reading the image 
        img = cv2.imread(str(img_name), cv2.IMREAD_UNCHANGED)
        print(type(img))
        # Apply the cvtColor method to change the colorspace 
        hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    
        # Finding the proper HSV range to detect the particles succesfully 
        lower_hsv = np.array([0, 0, 0])
        higher_hsv = np.array([60, 255, 255])
    
        # Apply the in-range method to create a mask for isolating the particles from the background
        bg_mask = cv2.inRange(hsv, lower_hsv, higher_hsv)
    
        # Getting a black background image 
        black_bg_img = cv2.bitwise_and(img, img, mask=bg_mask)
    
        # Converting the image form RGB to RGBA to get "alpha" extra channel  
        black_bg_img_cp = black_bg_img.copy()
        img_alpha = cv2.cvtColor(black_bg_img_cp, cv2.COLOR_BGR2BGRA)
        
        # apply the mask on the img with black background to get the seperated background image
        sep_img = cv2.bitwise_and(img_alpha, img_alpha, mask=bg_mask)
        # ------------------------------------------------------------------------------------------------------
    
        # Creating a seperated background mask
        # ------------------------------------------------------------------------------------------------------
        
        # Converting the black background img to grayscale img
        gray_img = cv2.cvtColor(black_bg_img, cv2.COLOR_BGR2GRAY)
    
        # Minimizing noise
        gray_img = cv2.medianBlur(gray_img, 15)
        
        # Converting to black and white 
        bw_mask = cv2.threshold(gray_img, 10, 255, cv2.THRESH_BINARY_INV)[1]
    
        # saving the image        
        cv2.imwrite(r"sample.png", bw_mask)
        
        break
    
        # apply morphology to remove isolated extraneous noise
        kernel = np.ones((3,3), np.uint8)
        bw_mask = cv2.morphologyEx(bw_mask, cv2.MORPH_OPEN, kernel, iterations=4)
        bw_mask = cv2.morphologyEx(bw_mask, cv2.MORPH_CLOSE, kernel, iterations=4)
    
        # Finding the sure foreground area using distance transformation method
        dist_transform = cv2.distanceTransform(bw_mask, cv2.DIST_L2, 5)
        fg = cv2.threshold(dist_transform, 0.01*dist_transform.max(), 255, 0)[1]
        fg = np.uint8(fg)
        
        # Creating bounded particles 
        black_bg_bound = cv2.subtract(bw_mask, fg)
        
        black_bg_bound = fg.copy()
        
        # Converting the image form RGB to RGBA to get extra "alpha" channel  
        sep_mask = cv2.cvtColor(black_bg_bound, cv2.COLOR_BGR2BGRA)
        
        # apply the mask on the img with black background to get the seperated background mask
        sep_mask[:, :, 3][black_bg_bound == 255] = 0
        # ------------------------------------------------------------------------------------------------------
        

    
        sep_imgs.append(sep_img)
        sep_masks.append(sep_mask)

        cv2.imwrite(r'C:\Users\GPU\particle_size_project\Dataset\plastic particles\particles_color\waste_color{}.png'.format(idx+1), sep_img)
        cv2.imwrite(r'C:\Users\GPU\particle_size_project\Dataset\plastic particles\particles_mask\waste_mask{}.png'.format(idx+1), sep_mask)
    
        idx+=1
        
    return sep_imgs, sep_masks



imgs = glob.glob(r'C:\Users\GPU\particle_size_project\Dataset\plastic particles\particles_color\*.png')

sep_imgs, sep_masks = bg_separation(imgs)