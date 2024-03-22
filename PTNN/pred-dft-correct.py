# -*- coding: utf-8 -*-
"""
Created on Sun Oct  8 10:05:54 2023

@author: dell
"""

import os
import torch
import torchvision
from torch import nn
from d2l import torch as d2l
import cv2 as cv
import torchvision.transforms as transforms
import numpy as np
import random
from torch.utils import data
import scipy.io as scio
import models

import time
time_start = time.time()
###### load models file
#net = torch.load("D:/crss-fcc.pth")
#net = torch.load("D:/crss-bcc.pth")
#net = torch.load("D:/crss-hcp.pth")
#net = torch.load("D:/crss-fcc-small.pth")
net = torch.load("D:/crss-total.pth")
#net = torch.load("D:/FT/FT-cu-S.pth")
#net = torch.load("D:/FT/FT-cu-L.pth")
#net = torch.load("D:/FT/FT-al-L.pth")
#net = torch.load("D:/FT/FT-fe-L.pth")
#net = torch.load("D:/FT/FT-ti-L.pth")

##### load DFT calculated gamma surface
##data_path = "D:/crss/gsf_dft_potential_al"
#data_path = "D:/crss/gsf_dft_potential_fe"
data_path = "D:/crss/gsf_dft_potential_ti"
#data_path = "C:/Users/dell/Desktop/DFT_gamma_cu"
#data_path = "D:/crss/gsf_dft_potential_cu"

#pre-processing
form = transforms.ToPILImage()
resize = transforms.Resize([224, 224])
to_tensor = transforms.ToTensor()
normalize = transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])

#loading images
files_path = os.listdir(data_path)
data_X = torch.zeros([len(files_path),3,224,224],dtype=torch.float)
count = 0
for img_file in files_path:
    img = cv.imread(os.path.join(data_path, img_file))
    img = form(img)
    img = resize(img)
    img = to_tensor(img)
    img = normalize(img)
    data_X[count,:,:,:] = img
    count +=1
    print('image' + ' ' + img_file + ' ' + 'has been loaded')

data_X = data_X.to(torch.float32)

##### load DFT calculated materials parameters
#data_features_path = "C:/Users/dell/Desktop/DFT_fea_cu.txt"
#data_features_path = "C:/Users/dell/Desktop/DFT_fea_cu_total.txt"
#data_features_path = "C:/Users/dell/Desktop/DFT_fea_al.txt"
##data_features_path = "C:/Users/dell/Desktop/DFT_fea_al_total.txt"
#data_features_path = "C:/Users/dell/Desktop/DFT_fea_fe.txt"
#data_features_path = "C:/Users/dell/Desktop/DFT_fea_fe_total.txt"
#data_features_path = "C:/Users/dell/Desktop/DFT_fea_ti.txt"
data_features_path = "C:/Users/dell/Desktop/DFT_fea_ti_total.txt"
data_features = np.loadtxt(data_features_path)
data_features = torch.from_numpy(data_features)
data_features = data_features.to(torch.float32)
#train_features = data_features
test_features = data_features
test_features = test_features.unsqueeze(0)
#test_features = test_features
#data_label_path = "D:/newdesktop/Desktop/crss/crss4.txt"
#data_label_path = "C:/Users/dell/Desktop/crss_fcc_cut.txt"
#data_Y = np.loadtxt(data_label_path)
#data_Y = torch.from_numpy(data_Y)
#data_Y = data_Y.unsqueeze(1)

#data_Y = data_Y.to(torch.float32)
#data_Y2 = data_Y[index]
#data_features2 = data_features[index]
#data_X2 = data_X[index]

res = net(data_X,test_features)
#tar = data_Y2
print(res)
time_end = time.time()  # record time
time_sum = time_end - time_start  # unit: s
print(time_sum)
#np.savetxt('D:/newdesktop/Desktop/crss/resfcccut-1-1-0.txt',res.detach().numpy())
#np.savetxt('D:/newdesktop/Desktop/crss/tar_fcc_cut.txt',tar)