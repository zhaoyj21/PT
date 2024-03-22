# -*- coding: utf-8 -*-

import os
import torch
from torch import nn
import cv2 as cv
import torchvision.transforms as transforms
import numpy as np
import torch.utils.data as Data

#net = torch.load("D:/crss-fcc.pth")
net = torch.load("D:/crss-fcc-small.pth")
#net = torch.load("D:/crss-bcc.pth")
#net = torch.load("D:/crss-hcp.pth")
for param in net.parameters():
    param.requires_grad = False
    
net.fc3 = nn.Linear(32,1)

loss_func = nn.MSELoss()
optimizer = torch.optim.Adam([
    {'params':net.fc3.parameters()}
], lr=0.001)

device = torch.device("cpu")
data_path = "C:/Users/dell/Desktop/FT"


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
    #print('image' + ' ' + img_file + ' ' + 'has been loaded')

train_X = data_X

data_features_path = "C:/Users/dell/Desktop/features_mlff_cu_al.txt"
#data_features_path = "C:/Users/dell/Desktop/features_mlff_al.txt"
#data_features_path = "C:/Users/dell/Desktop/features_mlff_fe.txt"
#data_features_path = "C:/Users/dell/Desktop/features_mlff_ti.txt"
data_features = np.loadtxt(data_features_path)
data_features = torch.from_numpy(data_features)

data_features = data_features.to(torch.float32)
train_features = data_features.reshape(1,-1)

#data_label_path = "C:/Users/dell/Desktop/crss_cu_al.txt"
#data_label_path = "C:/Users/dell/Desktop/crss_al.txt"
#data_label_path = "C:/Users/dell/Desktop/crss_fe.txt"
#data_label_path = "C:/Users/dell/Desktop/crss_ti.txt"
data_label_path = "C:/Users/dell/Desktop/crss_cu_S.txt"
data_Y = np.loadtxt(data_label_path)
data_Y = torch.from_numpy(data_Y)
data_Y = data_Y.reshape(1,1)
#data_Y = data_Y.unsqueeze(1)

data_Y = data_Y.to(torch.float32)

train_Y = data_Y

torch_dataset = Data.TensorDataset(train_X,train_features,train_Y)
train_loader = Data.DataLoader(torch_dataset, batch_size = 1, shuffle=True)
EPOCH = 200

for epoch in range(EPOCH):
    net.train()
    for batch_idx, data in enumerate(train_loader):
        x1,x2, y= data
        x1=x1.to(device)
        x2=x2.to(device)
        y=y.to(device)
        optimizer.zero_grad()
        y_hat=net(x1,x2)
        loss = loss_func(y_hat, y)
        loss.backward()
        optimizer.step()
        print ('Train Epoch: {}\t Loss: {:.6f}'.format(epoch,loss.item()))

path = 'D:/FT/FT-cu-S.pth'
torch.save(net, path)