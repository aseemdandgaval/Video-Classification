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
from statistics import mode
import torchvision.models as models
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
from sklearn.model_selection import train_test_split


from streamlit_utils import get_default_device, to_device
from streamlit_utils import DeviceDataLoader, ImageClassificationBase, VideoResnet

device = get_default_device()
model = to_device(VideoResnet(), device)
model.load_state_dict(torch.load('resnet18-60epoch.pth'))
model.eval()

def predict_image(img, model):
    xb = to_device(img.unsqueeze(0), device)
    yb = model(xb)
    _, preds  = torch.max(yb, dim=1)
    classes = classes = os.listdir('uc/UCF101/UCF-101/')
    return classes[preds[0].item()]

def predictVideoClass(video_path):
    video = cv2.VideoCapture(video_path)
    predictions = []
    odd_count = 0
    frame_count = 0
    while video.isOpened():
        ret, frame = video.read()
        frame_count = frame_count+1
        
        if not ret:
            break

        if frame_count % 2 == 0:
            continue
        if frame_count == 5:
            video_frame = tensor_frame
    
        convert_tensor = tt.ToTensor()
        tensor_frame = convert_tensor(frame)
    
        with torch.no_grad():
            pred = predict_image(tensor_frame, model)
        predictions.append(pred)
        
    return frame_count, mode(predictions), predictions, video_frame
    
def main():
    torch.cuda.empty_cache()
    st.title("Video Classification Web App")
    
    classes = os.listdir('uc/UCF101/UCF-101/')
    video_class = st.selectbox('Select the class:', classes)
    
    video_files = os.listdir('uc/UCF101/UCF-101/' + str(video_class))
    video_option = st.selectbox('Select the video:', video_files)
    video_path = 'uc/UCF101/UCF-101/' + str(video_class) + '/' + str(video_option)
    
    if st.button('Run Model'):
        frame_count, predicted_class, predictions, video_frame = predictVideoClass(video_path)

        plt.imshow(video_frame.permute(1, 2, 0))
        plt.savefig('streamlit_images/' + str(video_class) + '_' + str(video_option.split('.')[0]) + '.jpg')
        st.image('streamlit_images/' + str(video_class) + '_' + str(video_option.split('.')[0]) + '.jpg')
        
        st.write('No. of frames predicted:  ', frame_count//2, ' ;  Actual Frames:  ', frame_count)
        st.write('Predicted Class: ', predicted_class)
        st.write("")
        
        counts = Counter(predictions)
        total_items = len(predictions)
        for item, count in counts.items():
            percentage = (count / total_items) * 100
            st.write(f"{item}:  {percentage:.2f}%")
    
if __name__ == '__main__':
    main()