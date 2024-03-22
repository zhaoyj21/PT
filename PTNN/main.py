#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from scipy import io
import copy

import utils
import train_model
import eval_model
import model_config

import os
import torch
import cv2 as cv
import torchvision.transforms as transforms
import numpy as np 

index = np.array([0,1,2,4,5,7,8,10,12,13,15,16,17,18,20,22,3,6,9,11,14,19,21,23])
data_path = "C:/Users/dell/Desktop/gsf_fcc"

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

train_X = data_X[index[:16]]
test_X = data_X[index[16:]]

data_features_path = "C:/Users/dell/Desktop/features_fcc.txt"
data_features = np.loadtxt(data_features_path)
data_features = torch.from_numpy(data_features)

data_features = data_features.to(torch.float32)
train_features = data_features[index[:16]]
test_features = data_features[index[16:]]

data_label_path = "C:/Users/dell/Desktop/crss_fcc.txt"
data_Y = np.loadtxt(data_label_path)
data_Y = torch.from_numpy(data_Y)
data_Y = data_Y.unsqueeze(1)
data_Y = data_Y.to(torch.float32)

train_Y = data_Y[index[:16]]
test_Y = data_Y[index[16:]]

para_name = 'max_epoch'
configs = model_config.set_config(para_name)
num_weight = utils.get_weight_num(configs, para_name)
result = utils.result(para_name, configs[para_name], num_weight, configs['iter_time'])

for i,hyper_para in enumerate(configs[para_name]):
    for k in range(configs['iter_time']):
        config = copy.deepcopy(configs)
        config[para_name] = hyper_para
        config['train_size'] = int( config['train_size'] / ( 1 - config['rho'] ) )
        model = train_model.train(config, train_X, train_features, train_Y, test_X, test_features, test_Y )
        path = 'D:/' +'crss_models_fcc/'+'crss-fcc-cut-'+str(i)+'-'+str(k)+'.pth'
        torch.save(model, path)
        if config['evaluation'] == True:
            result = eval_model.eval_model(model = model,
                                config = config,
                                train_x = train_X,
                                train_features = train_features,
                                train_y = train_Y,
                                test_x = test_X,
                                test_features = test_features,
                                test_y = test_Y,
                                result = result,
                                index = (i, k, num_weight))
        
pre_text = 'crss_fcc_1_' + para_name
io.savemat('./Visualization/{}_result.mat'.format(pre_text),{'{}_result'.format(pre_text):result})

