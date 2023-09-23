# -*- coding: utf-8 -*-
"""
Created on Wed Oct 26 12:06:19 2022

@author: ranad
"""

import numpy as np
import cv2
import matplotlib.pyplot as plt
import torch
import torchvision
from tqdm import tqdm 

def plot_training(title,
                  training_losses,
                  validation_losses,
                  iou_score,
                  accuracy,
                  f1_score,
                  precision,
                  recall,
                  scale
                  ):
    """
    Returns a loss plot with training loss and validation loss, and dice score plot
    """

    import matplotlib.pyplot as plt
    
    # create figure (will only create new window if needed)
    plt.figure()
    
    list_len = len(training_losses)
    x_range = list(range(1, list_len + 1))  # number of x values
    
    linestyle_original = '-'
    color_original_train = 'red'
    color_original_valid = 'green'
    color_original_iou = 'blue'
    color_original_f1 = 'brown'
    color_original_accuracy = 'orange'
    color_original_precision = "yellow"
    color_original_recall = "yellow"
    alpha = 1.0
    
    plt.plot(x_range, training_losses, linestyle_original, color=color_original_train, label='Training',
                      alpha=alpha)
    plt.plot(x_range, validation_losses, linestyle_original, color=color_original_valid, label='Validation',
                      alpha=alpha)

    plt.title('T & V loss '+ title)
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    
    plt.legend(loc='best')
    
    # Saving figure
    plt.savefig(scale + '\T&Vloss' + title + '.png', dpi=400, bbox_inches = 'tight')
    
    # Show the plot in non-blocking mode
    plt.show()
    
    plt.close()
    
    if iou_score is not None:
        # create figure (will only create new window if needed)
        plt.figure()
        
        plt.plot(x_range, iou_score, linestyle_original, color=color_original_iou, label='iou score',
                          alpha=alpha)
        
        plt.title('IOU Score '+ title)
        plt.xlabel('Epoch')
        plt.ylabel('iou score')
        
        plt.legend(loc='best')
        
        # Saving figure
        plt.savefig(scale + '\IOU Score' + title + '.png', dpi=400, bbox_inches = 'tight')
        
        # Show the plot in non-blocking mode
        plt.show()
        
        plt.close()
        
    if f1_score is not None:
        # create figure (will only create new window if needed)
        plt.figure()
        
        plt.plot(x_range, f1_score, linestyle_original, color=color_original_f1, label='f1 score',
                          alpha=alpha)
        
        plt.title('F1 Score '+ title)
        plt.xlabel('Epoch')
        plt.ylabel('f1 score')
        
        plt.legend(loc='best')
        
        # Saving figure
        plt.savefig(scale + '\F1 Score' + title + '.png', dpi=400, bbox_inches = 'tight')
        
        # Show the plot in non-blocking mode
        plt.show()
        
        plt.close()
    
    if accuracy is not None:
        
        # create figure (will only create new window if needed)
        plt.figure()
        
        plt.plot(x_range, accuracy, linestyle_original, color=color_original_accuracy, label='accuracy',
                          alpha=alpha)
        
        plt.title('Pixel Accuracy '+ title)
        plt.xlabel('Epoch')
        plt.ylabel('Accuracy')
        
        plt.legend(loc='best')
        
        # Saving figure
        plt.savefig(scale + '\Accuracy' + title + '.png', dpi=400, bbox_inches = 'tight')
        
        # Finally block main thread until all plots are closed
        plt.show()
        
    if precision is not None:
        
        # create figure (will only create new window if needed)
        plt.figure()
        
        plt.plot(x_range, precision, linestyle_original, color=color_original_precision, label='precision',
                          alpha=alpha)
        
        plt.title('precision '+ title)
        plt.xlabel('Epoch')
        plt.ylabel('Precision')
        
        plt.legend(loc='best')
        
        # Saving figure
        plt.savefig(scale + '\Precision' + title + '.png', dpi=400, bbox_inches = 'tight')
        
        # Finally block main thread until all plots are closed
        plt.show()
        
    if recall is not None:
        
        # create figure (will only create new window if needed)
        plt.figure()
        
        plt.plot(x_range, recall, linestyle_original, color=color_original_recall, label='recall',
                          alpha=alpha)
        
        plt.title('recall '+ title)
        plt.xlabel('Epoch')
        plt.ylabel('Recall')
        
        plt.legend(loc='best')
        
        # Saving figure
        plt.savefig(scale + '\Recall' + title + '.png', dpi=400, bbox_inches = 'tight')
        
        # Finally block main thread until all plots are closed
        plt.show()
    
def plot_testing(title,
              test_losses,
              iou_score,
              accuracy,
              f1_score
              ):
    """
    Returns a loss plot with training loss and validation loss, and dice score plot
    """

    import matplotlib.pyplot as plt
    
    # create figure (will only create new window if needed)
    plt.figure()
    
    list_len = len(test_losses)
    x_range = list(range(1, list_len + 1))  # number of x values
    
    linestyle_original = '-'
    color_original_train = 'red'
    color_original_iou = 'blue'
    color_original_f1 = 'brown'
    color_original_accuracy = 'orange'
    alpha = 1.0
    
    plt.plot(x_range, test_losses, linestyle_original, color=color_original_train, label='Testing',
                      alpha=alpha)
    
    plt.title('Test Loss '+ title)
    plt.xlabel('Batch')
    plt.ylabel('Loss')
    
    plt.legend(loc='best')
    
    # Saving figure
    plt.savefig('2.0x\Test loss' + title + '.png', dpi=400)
    
    # Show the plot in non-blocking mode
    plt.show()
    
    plt.close()
    
    if iou_score is not None:
        # create figure (will only create new window if needed)
        plt.figure()
        
        plt.plot(x_range, iou_score, linestyle_original, color=color_original_iou, label='iou score',
                          alpha=alpha)
        
        plt.title('IOU Score '+ title)
        plt.xlabel('Batch')
        plt.ylabel('iou score')
        
        plt.legend(loc='best')
        
        # Saving figure
        plt.savefig('2.0x\IOU Score' + title + '.png', dpi=400)
        
        # Show the plot in non-blocking mode
        plt.show()
        
        plt.close()
        
    if f1_score is not None:
        # create figure (will only create new window if needed)
        plt.figure()
        
        plt.plot(x_range, f1_score, linestyle_original, color=color_original_f1, label='f1 score',
                          alpha=alpha)
        
        plt.title('F1 Score '+ title)
        plt.xlabel('Batch')
        plt.ylabel('f1 score')
        
        plt.legend(loc='best')
        
        # Saving figure
        plt.savefig('2.0x\F1 Score' + title + '.png', dpi=400)
        
        # Show the plot in non-blocking mode
        plt.show()
        
        plt.close()
    
    if accuracy is not None:
        
        # create figure (will only create new window if needed)
        plt.figure()
        
        plt.plot(x_range, accuracy, linestyle_original, color=color_original_accuracy, label='accuracy',
                          alpha=alpha)
        
        plt.title('Pixel Accuracy '+ title)
        plt.xlabel('Batch')
        plt.ylabel('Accuracy')
        
        plt.legend(loc='best')
        
        # Saving figure
        plt.savefig('2.0x\Accuracy' + title + '.png', dpi=400)
        
        # Finally block main thread until all plots are closed
        plt.show()
        
def save_predictions_as_imgs_bw(model,loader):
    # save predictions as images in black and white 
    device = "cpu"
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model.eval()
    total_preds = []
    num = 2
    for idx, (x, y) in tqdm(enumerate(loader), desc = "saving predictions as b/w img", unit = " images"):
        x = x.to(device=device)
        with torch.no_grad():
            pred = torch.sigmoid(model(x))
            pred = (pred > 0.5).float()
            pred = pred.cpu().numpy()
            total_preds.append(pred)
            img = torch.tensor(pred)
            img = img * 255
            torchvision.utils.save_image(img, r"C:\Users\GPU\particle_size_project\particle_size_distribution\non-machine\samples\labeled-wood-b&w{}.png".format(num))
    return total_preds

def save_predictions_as_colored_labels(model, loader):
    device = "cpu"
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model.eval()
    total_preds = []
    num = 1
    for idx, (x, y) in tqdm(enumerate(loader), desc = "saving predictions as colored labels", unit = " images"):
        x = x.to(device=device)
        with torch.no_grad():
            pred = torch.sigmoid(model(x))
            pred = (pred > 0.5).float()
            pred = pred.cpu().numpy()
            pred = np.reshape(pred, (736, 1280))
            total_preds.append(pred)
            img = (pred * 255).astype(np.uint8)
            # img = cv2.threshold(img, 0, 255, cv2.THRESH_BINARY)
            # img = cv2.threshold(img, 0, 255, cv2.THRESH_BINARY_INV)
            # img = cv2.threshold(img, 0, 255, cv2.THRESH_BINARY_INV)[1]
            # img = cv2.threshold(img, 0, 255, cv2.THRESH_BINARY)
            # img = cv2.threshold(img, 0, 255,
            #  	cv2.THRESH_BINARY_INV | cv2.THRESH_OTSU)[1]
            cv2.imwrite(r"C:\Users\GPU\particle_size_project\particle_size_distribution\thesis\machine\wood-b&w-app4.png", img)
            # img = cv2.threshold(img, 0, 255,
            #  	cv2.THRESH_BINARY_INV | cv2.THRESH_OTSU)[1]
            contours, hierarchy = cv2.findContours(img.astype(np.uint8), cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
            n_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(img.astype(np.uint8), connectivity=4)
            colors = np.random.randint(0, 255, size=(n_labels, 3), dtype=np.uint8)
            colors[0] = [0, 0, 0]  # for cosmetic reason we want the background black
            false_colors = colors[labels]
            cv2.imwrite(r"C:\Users\GPU\particle_size_project\particle_size_distribution\thesis\machine/wood-colored-app4.png", false_colors)
    return total_preds

def particle_analysis(stitched_images):
    for image in stitched_images:
        # Estimating the particle size distribution 
        size_0_25 = []
        size_25_50 = []
        size_50_100 = []
        size_100_250 = []
        size_250_500 = []
        size_500_1000 = []
        size_1000_2000 = []
        size_2000_4000 = []
        size_4000 = []
        image = (image * 255).astype(np.uint8)
        # image_invr = cv2.threshold(image, 15, 255, cv2.THRESH_BINARY_INV)[1]
        # image = cv2.threshold(image, 15, 255, cv2.THRESH_BINARY)
        n_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(image, connectivity=4)
        areas = stats[1:, cv2.CC_STAT_AREA]
        areas = list(areas)
        for particle in areas:
            if particle < 25:
                size_0_25.append(particle)
            if particle >= 25 and particle < 50:
                size_25_50.append(particle)
            if particle >= 50 and particle < 100:
                size_50_100.append(particle)
            if particle >= 100 and particle < 250:
                size_100_250.append(particle)
            if particle >= 250 and particle < 500:
                size_250_500.append(particle)
            if particle >= 500 and particle < 1000:
                size_500_1000.append(particle)
            if particle >= 1000 and particle < 2000:
                size_1000_2000.append(particle)
            if particle >= 2000 and particle < 4000:
                size_2000_4000.append(particle)
            if particle > 4000:
                size_4000.append(particle)
        x_axis = ['P < 25', '25 =< P < 50', '50 =< P < 100', '100 =< P < 250', '250 =< P < 500',
                  '500 =< P < 1000', '1000 =< P < 2000',
                  '2000 =< P < 4000', 'P > 4000']
        y_axis = [len(size_0_25), len(size_25_50), len(size_50_100), len(size_100_250), len(size_250_500),
                  len(size_500_1000), len(size_1000_2000), len(size_2000_4000),
                  len(size_4000)]
        plt.barh(x_axis, y_axis)
        plt.title('Particle size P (pixels) vs num of particles')
        plt.ylabel('Particle size (pixels)')
        plt.xlabel('num of particles')
        plt.savefig('C:/Users/GPU\particle_size_project/particle_size_distribution/thesis/machine/dis-wood-app4.png',dpi=400, bbox_inches = 'tight')
