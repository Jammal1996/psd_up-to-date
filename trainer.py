# -*- coding: utf-8 -*-
"""
Created on Sat Dec 10 15:09:28 2022

@author: ranad
"""

import torch
import numpy as np
from tqdm import tqdm
from metrics import BinaryMetrics
import segmentation_models_pytorch as smp

class Trainer:
    def __init__(self,
                 model: torch.nn.Module,
                 device: torch.device,
                 criterion: torch.nn.Module,
                 optimizer: torch.optim.Optimizer,
                 training_DataLoader: torch.utils.data.Dataset,
                 validation_DataLoader: torch.utils.data.Dataset = None,
                 epochs: int = 100,
                 epoch: int = 0,
                 notebook: bool = False
                 ):
        
        self.model = model
        self.criterion = criterion
        self.optimizer = optimizer
        self.training_DataLoader = training_DataLoader
        self.validation_DataLoader = validation_DataLoader
        self.device = device
        self.epochs = epochs
        self.epoch = epoch
        self.notebook = notebook
        self.binary_metrics = BinaryMetrics()
        
        self.training_loss = []
        self.validation_loss = []
        self.pixel_accuracy = []
        self.iou_score = []
        self.f1_score = []
        self.precision_scores = []
        self.recall_scores = []
        
    def run_trainer(self):
        
        for epoch in range(self.epochs):
            """Training block"""
            self._train()
            
            """Validation block"""
            if self.validation_DataLoader is not None:
                self._validate()
            
            """Epoch counter"""
            self.epoch += 1  # epoch counter
            
        return self.training_loss, self.validation_loss, self.iou_score, self.pixel_accuracy, self.f1_score, self.precision_scores, self.recall_scores
        
    def _train(self):
        
        self.model.train()  # train mode
        train_losses = []  # accumulate the losses here
        
        with tqdm(self.training_DataLoader, unit=" batch") as tepoch:
            for i, (x, y) in enumerate(tepoch):
                tepoch.set_description(f"<Training> Epoch {self.epoch+1}")
                img, target = x.to(self.device), y.to(self.device)  # send to device (GPU or CPU)
                target = torch.unsqueeze(target, dim=1)
                self.optimizer.zero_grad()  # zerograd the parameters
                pred = self.model(img)  # one forward pass
                loss = self.criterion(pred, target)  # calculate loss
                loss_value = loss.item()
                train_losses.append(loss_value)
                loss.backward()  # one backward pass
                self.optimizer.step()  # update the parameters
                
                tepoch.set_postfix(loss=loss.item())  # update progressbar
        
        self.training_loss.append(np.mean(train_losses))
        
    def _validate(self):
        
        from tqdm import tqdm
        
        self.model.eval()  # evaluation mode
        valid_losses = []  # accumulate the losses here
        pixel_accs = []
        iou_scores = []
        f1_scores = []
        precision_scores = []
        recall_scores = []
        
        with tqdm(self.validation_DataLoader, unit=" batch") as tepoch:
            for i, (x, y) in enumerate(tepoch):
                tepoch.set_description(f"<Validating> Epoch {self.epoch+1}")
                img, target = x.to(self.device), y.to(self.device)  # send to device (GPU or CPU)
                target = torch.unsqueeze(target, dim=1)
                with torch.no_grad():
                    pred = self.model(img)
                    loss = self.criterion(pred, target)
                    loss_value = loss.item()
                    # pred = torch.sigmoid(pred)
                    tp, fp, fn, tn = smp.metrics.get_stats(pred, target, mode='binary', threshold=0.5)
                    iou_score = smp.metrics.iou_score(tp, fp, fn, tn, reduction="micro")
                    f1_score = smp.metrics.f1_score(tp, fp, fn, tn, reduction="micro")
                    pixel_acc = smp.metrics.accuracy(tp, fp, fn, tn, reduction="macro")
                    recall = smp.metrics.recall(tp, fp, fn, tn, reduction="macro-imagewise")
                    precision = smp.metrics.precision(tp, fp, fn, tn, reduction="macro-imagewise")
                    # pixel_acc, dice, precision, _, recall = self.binary_metrics._calculate_overlap_metrics(target, pred)
                    valid_losses.append(loss_value)
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
        
        self.validation_loss.append(np.mean(valid_losses))
        self.iou_score.append(np.mean(iou_scores))
        self.pixel_accuracy.append(np.mean(pixel_accs))
        self.f1_score.append(np.mean(f1_scores))
        self.precision_scores.append(np.mean(precision_scores))
        self.recall_scores.append(np.mean(recall_scores))