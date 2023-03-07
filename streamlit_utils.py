import os
import cv2     
import math   
import torch
import shutil
import statistics
import pandas as pd
import numpy as np 
from glob import glob
from tqdm import tqdm
import streamlit as st
import torchvision.models as models
from statistics import mode
from IPython.display import Video
from collections import Counter
import torch.nn.functional as F
import torch.nn as nn
from torch.optim import lr_scheduler 
from torchvision.datasets.utils import download_url
from torchvision.datasets import ImageFolder
from torch.utils.data import DataLoader
import torchvision.transforms as tt
from torch.utils.data import random_split
from torchvision.utils import make_grid
import matplotlib.pyplot as plt   

def get_default_device():
    """Pick GPU if available, else CPU"""
    if torch.cuda.is_available():
        return torch.device('cuda')
    else:
        return torch.device('cpu')
    
def to_device(data, device):
    """Move tensor(s) to chosen device"""
    if isinstance(data, (list,tuple)):
        return [to_device(x, device) for x in data]
    return data.to(device, non_blocking=True)

class DeviceDataLoader():
    """Wrap a dataloader to move data to a device"""
    def __init__(self, dl, device):
        self.dl = dl
        self.device = device
        
    def __iter__(self):
        """Yield a batch of data after moving it to device"""
        for b in self.dl: 
            yield to_device(b, self.device)

    def __len__(self):
        """Number of batches"""
        return len(self.dl)
    
class ImageClassificationBase(nn.Module):
    def training_step(self, batch):
        images, labels = batch 
        out = self(images)                  # Generate predictions
        loss = F.cross_entropy(out, labels) # Calculate loss
        return loss
    
    def validation_step(self, batch):
        images, labels = batch 
        out = self(images)                    # Generate predictions
        loss = F.cross_entropy(out, labels)   # Calculate loss
        acc = accuracy(out, labels)           # Calculate accuracy
        return {'val_loss': loss.detach(), 'val_acc': acc}
        
    def validation_epoch_end(self, outputs):
        batch_losses = [x['val_loss'] for x in outputs]
        epoch_loss = torch.stack(batch_losses).mean()   # Combine losses
        batch_accs = [x['val_acc'] for x in outputs]
        epoch_acc = torch.stack(batch_accs).mean()      # Combine accuracies
        return {'val_loss': epoch_loss.item(), 'val_acc': epoch_acc.item()}
    
    def epoch_end(self, epoch, result):
        print("Epoch [{}], last_lr: {:.5f}, train_loss: {:.4f}, val_loss: {:.4f}, val_acc: {:.4f}".format(
            epoch, result['lrs'][-1], result['train_loss'], result['val_loss'], result['val_acc']))
        
class VideoResnet(ImageClassificationBase):
    def __init__(self):
        super().__init__()
        # Use a pretrained model
        self.network = models.resnet18(pretrained=True)
        # Replace last layer
        num_ftrs = self.network.fc.in_features
        self.network.fc = nn.Sequential(
                          nn.Linear(num_ftrs, 256),
                          nn.ReLU(),
                          nn.Dropout(0.4),
                          nn.Linear(256, 101))
    
    def forward(self, xb):
        return torch.sigmoid(self.network(xb))
    
    def freeze(self):
        # To freeze the residual layers
        for param in self.network.parameters():
            param.require_grad = False
        for param in self.network.fc.parameters():
            param.require_grad = True
    
    def unfreeze(self):
        # Unfreeze all layers
        for param in self.network.parameters():
            param.require_grad = True
            
def predict_image(img, model):
    # Convert to a batch of 1
    xb = to_device(img.unsqueeze(0), device)
    # Get predictions from model
    yb = model(xb)
    # Pick index with highest probability
    _, preds  = torch.max(yb, dim=1)
    # Retrieve the class label|
    return train_ds.classes[preds[0].item()]

def predictVideoClass(video_path, confidences = False):
    video = cv2.VideoCapture(video_path)
    predictions = []
    odd_count = 0
    frame_count = 0
    while video.isOpened():
        ret, frame = video.read()
        frame_count=frame_count+1
        
        if not ret:
            break

        if frame_count % 2 == 0:
            continue
    
        convert_tensor = tt.ToTensor()
        tensor_frame = convert_tensor(frame)
    
        with torch.no_grad():
            pred = predict_image(tensor_frame, model)
        predictions.append(pred)
        
        if frame_count == 5:
            video_frame = tensor_frame
            
    plt.imshow(video_frame.permute(1, 2, 0))
    print('No. of frames predicted:', frame_count//2, '; Actual:', frame_count)
    print('Predicted Class: ', mode(predictions), '\n')
    
    if confidences:
        counts = Counter(predictions)
        total_items = len(predictions)
        
        for item, count in counts.items():
            percentage = (count / total_items) * 100
            print(f"{item}: {percentage:.2f}%")