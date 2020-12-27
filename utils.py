# -*- coding: utf-8 -*-
"""
Created on Tue Sep  3 14:33:01 2019

@author: User
"""

import torch

def use_optimizer(model,config):
    if config['optimizer']=='adam':
        optimizer=torch.optim.Adam(model.parameters(),lr=config['adam_lr'],weight_decay=config['l2_regularization'])        
    return optimizer    

def save_checkpoint(model,model_dir):
    torch.save(model.state_dict(),model_dir)