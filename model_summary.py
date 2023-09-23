# -*- coding: utf-8 -*-
"""
Created on Mon Aug  7 18:41:54 2023

@author: GPU
"""

# Importing libraries
# --------------------------------------------------------------------
import os
os.environ["KMP_DUPLICATE_LIB_OK"]="TRUE"
import ssl
ssl._create_default_https_context = ssl._create_unverified_context
import torch
from torchsummary import summary
import segmentation_models_pytorch as smp 

# Initial setup 
# --------------------------------------------------------------------
device = "cuda" if torch.cuda.is_available() else "cpu"
model_summary = True

# Params
# --------------------------------------------------------------------
img_height = 736
img_width = 1280
num_channels = 1
unet = True
unetplus=False
linknet=False
manet=False

# Model Summary 
# --------------------------------------------------------------------
if __name__ == "__main__":
    
    # aux_params=dict(
    # pooling='avg',             # one of 'avg', 'max'
    # dropout=0.5,               # dropout ratio, default is None
    # activation='sigmoid',      # activation function, default is None
    # classes=1)                 # define number of output labels

    if unet:
        model = smp.Unet(encoder_name='timm-regnety_040', encoder_depth=5,
                          encoder_weights='imagenet', decoder_use_batchnorm=False,
                          decoder_channels=(256, 128, 64, 32, 16),
                          decoder_attention_type=None, in_channels=1,
                          classes=1, activation=None, aux_params=None).to(device)
    
    if unetplus:
        model = smp.UnetPlusPlus(encoder_name='vgg13', encoder_depth=5,
                      encoder_weights='imagenet', decoder_use_batchnorm=True,
                      decoder_channels=(256, 128, 64, 32, 16), decoder_attention_type=None,
                      in_channels=1, classes=1, activation=None, aux_params=None).to(device)
    
    if linknet:
        model = smp.Linknet(encoder_name='timm-regnetx_040', encoder_depth=5,
                        encoder_weights='imagenet', decoder_use_batchnorm=True,
                        in_channels=1, classes=1, activation=None, aux_params=None).to(device)
    
    if manet:
        model = smp.MAnet(encoder_name='vgg13', encoder_depth=5, encoder_weights='imagenet',
                          decoder_use_batchnorm=True, decoder_channels=(256, 128, 64, 32, 16),
                          decoder_pab_channels=64, in_channels=1,
                          classes=1, activation=None, aux_params=None).to(device)
        
    if model_summary:
        # Printout the model summary 
        summary = summary(model, (num_channels, img_height, img_width))