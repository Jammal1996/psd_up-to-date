# -*- coding: utf-8 -*-
"""
Created on Thu Jul 20 01:43:41 2023

@author: GPU
"""

# Importing libraries
# --------------------------------------------------------------------
import os
os.environ["KMP_DUPLICATE_LIB_OK"]="TRUE"
import ssl
ssl._create_default_https_context = ssl._create_unverified_context
import time
import numpy as np
import torch
from torchsummary import summary
import albumentations as A
from dataset_w import SegmentationDataSet
from transformations_w import ComposeDouble, FunctionWrapperDouble, create_dense_target, normalize_01, AlbuSeg2d
from skimage.transform import resize
from torch.utils.data import DataLoader
from trainer import Trainer
from visual import plot_training
import pickle
from segmentation_models_pytorch.losses import DiceLoss
import segmentation_models_pytorch as smp 

# Initial setup 
# --------------------------------------------------------------------
device = "cuda" if torch.cuda.is_available() else "cpu"
resume_train = False
load_history = False
model_summary = False
init_param = False
load_model = False
train = True
save_history = True
save_model = True
plot = True
pin_memory = True
# --------------------------------------------------------------------

# Params
# --------------------------------------------------------------------
img_height = 736
img_width = 1280
lr = 1e-6
batch_size = 1
num_workers = 0
num_channels = 1
overlap = 0
scale = "wood_500"
# --------------------------------------------------------------------

# Directories
# --------------------------------------------------------------------
# getting the current directory
curr_dir = os.getcwd()

# getting parent directories 
parent_dir_1 = os.path.dirname(curr_dir)

# the folder path to save the model
saved_model = os.path.join(curr_dir, scale + "\models")

# the folder to save the history
saved_history = os.path.join(curr_dir, scale + "\history")

title = "(w1,ln,vgg13_bn,adam,dice,d=500,c=50,t=0,lr=1e-6,e=50)"

# history files dir
train_loss_history = os.path.join(saved_history, "t_losses_" + title +".pkl")
valid_loss_history = os.path.join(saved_history, "v_losses_" + title +".pkl")
iou_score_history = os.path.join(saved_history, "iou_score_" + title +".pkl")
pixel_accuracy_history = os.path.join(saved_history, "accuracy_" + title +".pkl")
f1_score_history = os.path.join(saved_history, "f1score_" + title +".pkl")
precision_history = os.path.join(saved_history, "precision_" + title +".pkl")
recall_history = os.path.join(saved_history, "recall_" + title +".pkl")

# The path where the model is saved
saving_model = os.path.join(saved_model, title + ".pth.tar")

if resume_train:
    title = title + ",e=50)"
    train_loss_history = os.path.join(saved_history, "t_losses_" + title +".pkl")
    valid_loss_history = os.path.join(saved_history, "v_losses_" + title +".pkl")
    iou_score_history = os.path.join(saved_history, "iou_score_" + title +".pkl")
    pixel_accuracy_history = os.path.join(saved_history, "accuracy_" + title +".pkl")

# Image paths
train_dir = r"C:\Users\GPU\particle_size_project\Dataset\wood_500\train\Images\*.png"
train_maskdir = r"C:\Users\GPU\particle_size_project\Dataset\wood_500\train\Masks\*.png"
val_dir = r"C:\Users\GPU\particle_size_project\Dataset\wood_500\valid\Images\*.png"
val_maskdir = r"C:\Users\GPU\particle_size_project\Dataset\wood_500\valid\Masks\*.png"
# --------------------------------------------------------------------

if __name__ == "__main__":
    
    # loading history
    if load_history:
        training_losses = pickle.load(open(train_loss_history, "rb"))
        validation_losses = pickle.load(open(valid_loss_history, "rb"))
        iou_score = pickle.load(open(iou_score_history, "rb"))
        pixel_accuracy = pickle.load(open(pixel_accuracy_history, "rb"))
    
    # Initiate the model
    # model = UNet(in_channels=1,
    #               out_channels=1,
    #               n_blocks=5,
    #               start_filters=128,
    #               activation='relu',
    #               normalization= "batch",
    #               conv_mode='same',
    #               dim=2).to(device)
    
    # model = smp.Unet(encoder_name='timm-regnetx_160', encoder_depth=5,
    #                   encoder_weights='imagenet', decoder_use_batchnorm=True,
    #                   decoder_channels=(512, 256, 128, 64, 32),
    #                   decoder_attention_type=None, in_channels=1,
    #                   classes=1, activation=None, aux_params=None).to(device)

    # model = smp.UnetPlusPlus(encoder_name='timm-resnest14d', encoder_depth=5,
    #               encoder_weights='imagenet', decoder_use_batchnorm=True,
    #               decoder_channels=(512, 256, 128, 64, 32), decoder_attention_type="scse",
    #               in_channels=1, classes=1, activation=None, aux_params=None).to(device)

    # model = smp.MAnet(encoder_name='timm-regnetx_160', encoder_depth=5, encoder_weights='imagenet',
    #                   decoder_use_batchnorm=True, decoder_channels=(512, 256, 128, 64, 32),
    #                   decoder_pab_channels=64, in_channels=1,
    #                   classes=1, activation=None, aux_params=None).to(device)
    
    model = smp.Linknet(encoder_name='vgg13_bn', encoder_depth=5,
                        encoder_weights='imagenet', decoder_use_batchnorm=True,
                        in_channels=num_channels, classes=1, activation=None, aux_params=None).to(device)
    
    if model_summary:
        # Printout the model summary 
        summary = summary(model, (num_channels, img_height, img_width))
    
    # Initializing the parameters 
    if init_param:
        model.initialize_parameters
        print("Parameters successfully initialized !")
    
    # optimizer
    optimizer = torch.optim.Adam(model.parameters(), lr = lr)
    
    # Choosing the loss function 
    criterion = DiceLoss(mode="binary")
    
    if load_model:
        print("Loading Checkpoint.........")
        model = torch.load(saving_model)
    
    # Training Phase
    if train:
        
        # Loading Data 
        print("Loading Data.........")
        train_transform = ComposeDouble([
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
            AlbuSeg2d(A.HorizontalFlip(p=0.6)),
            AlbuSeg2d(A.VerticalFlip(p=0.4)),
            AlbuSeg2d(A.Rotate(limit=25, p=0.3)),
            FunctionWrapperDouble(create_dense_target, input=False, target=True),
            FunctionWrapperDouble(np.moveaxis, input=True, target=False, source=-1, destination=0),
            FunctionWrapperDouble(normalize_01)
        ])
        
        # validation transformations
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
        
        train_dataset = SegmentationDataSet(imgs_dir=train_dir, masks_dir=train_maskdir, transform=train_transform)
        val_dataset = SegmentationDataSet(imgs_dir=val_dir, masks_dir=val_maskdir, transform=validate_transform)
        
        
        train_loader = DataLoader(train_dataset, batch_size=batch_size, num_workers=num_workers, pin_memory=pin_memory, shuffle=True)
        valid_loader = DataLoader(val_dataset, batch_size=batch_size, num_workers=num_workers, pin_memory=pin_memory, shuffle=True)
        print("Loading is Successful !")
        
        # Start training 
        print("Start Training...")
        start_training = time.time()
        
        # Calling the trainer function
        trainer = Trainer(model=model,
                      device=device,
                      criterion=criterion,
                      optimizer=optimizer,
                      training_DataLoader=train_loader,
                      validation_DataLoader=valid_loader,
                      epochs=50,
                      epoch=0,
                      notebook=False)
        
        # training phase 
        training_losses, validation_losses, iou_score, pixel_accuracy, f1_score, precision, recall = trainer.run_trainer()
        
        # Stop training 
        end_training = time.time()
        training_time = (end_training - start_training) / 3600
        print("Stop Training...")
        
        # Saving model
        if save_model:
            print("Saving Checkpoint.........")
            torch.save(model, saving_model)
        
        # saving history 
        if save_history:
            with open(train_loss_history, 'wb') as f:
                pickle.dump(training_losses, f)
            with open(valid_loss_history, 'wb') as f:
                pickle.dump(validation_losses, f)
            with open(iou_score_history, 'wb') as f:
                pickle.dump(iou_score, f)
            with open(pixel_accuracy_history, 'wb') as f:
                pickle.dump(pixel_accuracy, f)
            with open(f1_score_history, 'wb') as f:
                pickle.dump(f1_score, f)
            with open(precision_history, 'wb') as f:
                pickle.dump(precision, f)
            with open(recall_history, 'wb') as f:
                pickle.dump(recall, f)
        
        # Display training time 
        print(f"training_time: {training_time:.2f} hour")
        
    # Plotting training/validation graphs 
    if plot:
        plot_training(title, training_losses, validation_losses, iou_score, pixel_accuracy, f1_score, precision, recall, scale)
        