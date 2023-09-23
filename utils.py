# -*- coding: utf-8 -*-
"""
Created on Wed Jan 25 14:07:44 2023

@author: ranad
"""

import torch
from tqdm import tqdm 
import segmentation_models_pytorch as smp

def testing(model: torch.nn.Module,
            device: torch.device,
            criterion: torch.nn.Module,
            test_DataLoader: torch.utils.data.Dataset):
    
    model.eval()  # evaluation mode
    test_losses = []  # accumulate the losses here
    pixel_accs = []
    iou_scores = []
    f1_scores = []
    precision_scores = []
    recall_scores = []
    
    with tqdm(test_DataLoader, unit="batch") as tepoch:
        for i, (x, y) in enumerate(tepoch):
            tepoch.set_description("<Testing>")
            img, target = x.to(device), y.to(device)  # send to device (GPU or CPU)
            target = torch.unsqueeze(target, dim=1)
            with torch.no_grad():
                pred = model(img)
                loss = criterion(pred, target)
                loss_value = loss.item()
                # pred = torch.sigmoid(pred)
                tp, fp, fn, tn = smp.metrics.get_stats(pred, target, mode='binary', threshold=0.5)
                iou_score = smp.metrics.iou_score(tp, fp, fn, tn, reduction="micro")
                f1_score = smp.metrics.f1_score(tp, fp, fn, tn, reduction="micro")
                pixel_acc = smp.metrics.accuracy(tp, fp, fn, tn, reduction="macro")
                recall = smp.metrics.recall(tp, fp, fn, tn, reduction="macro-imagewise")
                precision = smp.metrics.precision(tp, fp, fn, tn, reduction="macro-imagewise")
                # pixel_acc, dice, precision, _, recall = self.binary_metrics._calculate_overlap_metrics(target, pred)
                test_losses.append(loss_value)
                pixel_accs.append(pixel_acc.item())
                iou_scores.append(iou_score.item())
                f1_scores.append(f1_score.item())
                precision_scores.append(precision.item())
                recall_scores.append(recall.item())
            
            tepoch.set_postfix(loss=loss.item(), 
                               pixel_acc = pixel_acc.item(),
                               iou = iou_score.item(),
                               f1 = f1_score.item(),
                               precision = precision.item(),
                               recall = recall.item())  # update progressbar
            
    return test_losses, pixel_accs, iou_scores, f1_scores