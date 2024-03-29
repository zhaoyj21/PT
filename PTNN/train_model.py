#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import torch
#import numpy as np
import torch.utils.data as Data
import torch.nn as nn
import models
import utils

def set_model(config):
    #net_size = config['net_size']
    #s = config['s']
    #d = config['d']
    model = models.MINet()
    #model = utils.init_weight(model,s)
    return model
    
def train(config,train_x,train_features,train_y,test_x,test_features,test_y):
    LR = config['alpha']
    batch_size = config['B']
    EPOCH = config['max_epoch']
    #test_size = config['test_size']
    #sample_num  = config['train_size']
    #mis_label_prob = config['rho']

    beta = config['beta']
    stop_loss = config['stop_loss']
    #sample_holder = config['sample_holder']
    add_regulization = config['regulization']
    
    #train_x,train_y,test_x,test_y = data.sub_set_task(sample_holder,sample_num)
    #train_x,train_y,identity = data.mislabel(train_x,train_y,mis_label_prob = mis_label_prob)
    
    #correct_range = int(len(train_y)*(1-mis_label_prob))
    #correct_train_x = train_x[0:correct_range,:]
    #correct_train_y = train_y[0:correct_range]    
    #test_x = test_x[0:test_size,:]
    #test_y = test_y[0:test_size]
    
    model = set_model(config)
    params_1x = [param for name, param in model.named_parameters()
             if name not in ["finetune_net.fc.weight", "finetune_net.fc.bias", "fc1.weight","fc1.bias","fc2.weight","fc2.bias","fc3.weight","fc3.bias"]]
    optimizer = torch.optim.SGD([{'params': params_1x},
                            {'params': model.finetune_net.fc.parameters(),
                             'lr': LR * 10},
                          {'params': model.fc1.parameters(),
                             'lr': LR * 10},
                          {'params': model.fc2.parameters(),
                             'lr': LR * 10},
                          {'params': model.fc3.parameters(),
                             'lr': LR * 10}],
                             lr=LR, weight_decay=0.001)

    #optimizer = torch.optim.SGD(model.parameters(),lr=LR)
    loss_func = nn.MSELoss()
    
    torch_dataset = Data.TensorDataset(train_x,train_features,train_y)
    train_loader = Data.DataLoader(torch_dataset, batch_size = batch_size, shuffle=True)
    
    train_loss_holder = []
    #correct_train_accuracy_holder = []
    for epoch in range(EPOCH):
        model.train()
        for step, (b_x,b_x_f,b_y) in enumerate(train_loader):
            out_put = model(b_x,b_x_f)
            if add_regulization:
                if epoch < 200:
                    loss = loss_func(out_put,b_y) + beta * utils.regulization(model,0)
                else:
                    loss = loss_func(out_put,b_y)
            else:
                loss = loss_func(out_put,b_y)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            print(f'epoch {epoch + 1}, train_loss {loss:f}')
    
   
            
        model.eval()
        all_loss = utils.cal_loss(model,train_x,train_features,train_y)
        val_loss = utils.cal_loss(model,test_x,test_features,test_y)
        train_loss_holder.append(all_loss)
        print(f'test_loss {val_loss}')
                    
        #correct_train_accuracy = utils.predict_accuracy(model,data_x = correct_train_x,data_y = correct_train_y)
        #correct_train_accuracy_holder.append(correct_train_accuracy)
        
        # test_accuracy = utils.predict_accuracy(model,data_x = test_x,data_y = test_y)
        # test_loss = utils.cal_loss(model, test_x, test_y)
        # if (epoch%1 == 0):
        #     print('Epoch is |',epoch,
        #           'train loss is |',all_loss,
        #           'test loss is |',test_loss,
        #           'train accuracy is |',correct_train_accuracy,
        #           'test accuracy is |',test_accuracy)
            
        #if all_loss < stop_loss:
            #print('cut')
            #break
    return model