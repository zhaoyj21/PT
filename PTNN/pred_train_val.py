# -*- coding: utf-8 -*-
"""
Created on Mon Aug 28 16:13:32 2023

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


#net = torch.load("D:/crss_models7/crss-fcc-3-0.pth")
#net = torch.load("D:/crss_models_fcc_cut1/crss-fcc-cut-1-0.pth")
#net = torch.load("D:/crss_models_bcc_cut1/crss-bcc-cut-2-0.pth")
net = torch.load("D:/crss_models_hcp_cut2/crss-hcp-cut-0-0.pth")

#data_path = "D:/newdesktop/Desktop/crss/data_gsf4"
#index = np.array([0,1,2,4,5,7,8,10,12,13,15,16,17,18,20,22,3,6,9,11,14,19,21,23])
#index = np.array([0,1,4,5,8,10,2,3,6,7,11,12])
#index = np.array([0,1,4,6,9,10,12,2,3,5,7,8,11,13])
index = np.array([2,3,5,6,8,11,0,1,4,7,9,10])
#index = np.array([0,1,4,7,8,10,12,13,14,16,2,3,5,6,9,11,15])
#index = np.array([0,2,5,7,9,11,13,15,16,17,1,3,4,6,8,10,12,14,18,19])
#index = np.array([0,1,2,4,5,7,8,10,12,13,15,16,17,18,20,22,3,6,9,11,14,19,21,23])

data_path = "C:/Users/dell/Desktop/crss/gsf_hcp_1010_cut"

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

#data_features_path = "D:/newdesktop/Desktop/crss/features_a_C_e5.txt"
data_features_path = "C:/Users/dell/Desktop/crss/features_hcp_1010_cut.txt"
data_features = np.loadtxt(data_features_path)
data_features = torch.from_numpy(data_features)
data_features = data_features.to(torch.float32)
train_features = data_features[index[:9]]
test_features = data_features[index[9:]]

#data_label_path = "D:/newdesktop/Desktop/crss/crss4.txt"
data_label_path = "C:/Users/dell/Desktop/crss/crss_hcp_1010_cut.txt"
data_Y = np.loadtxt(data_label_path)
data_Y = torch.from_numpy(data_Y)
data_Y = data_Y.unsqueeze(1)

data_Y = data_Y.to(torch.float32)
data_Y2 = data_Y[index]
data_features2 = data_features[index]
data_X2 = data_X[index]

res = net(data_X2,data_features2)
tar = data_Y2

np.savetxt('D:/newdesktop/Desktop/crss/reshcpnew.txt',res.detach().numpy())
np.savetxt('D:/newdesktop/Desktop/crss/tar_hcp_new.txt',tar)