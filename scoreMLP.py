# -*- coding: utf-8 -*-
"""
Created on Wed Nov  6 17:48:43 2019

@author: User
"""

import torch

        
class Pair_score_MLP(torch.nn.Module):
    def __init__(self,config):
        super(Pair_score_MLP,self).__init__()
        self.fc_layers=torch.nn.ModuleList()
        for idx, (in_size,out_size) in enumerate(zip(config['score_mlp_layers'][:-1],config['score_mlp_layers'][1:])):
            self.fc_layers.append(torch.nn.Linear(in_size,out_size))
            
    def forward(self,context,embeddings):     
        vector=torch.cat([context,embeddings],dim=-1)
        for idx in range(len(self.fc_layers)):
            vector=self.fc_layers[idx](vector)
            vector=torch.nn.ReLU()(vector)  
        return vector     
    
class CNN_score_MLP(torch.nn.Module):
    def __init__(self,config):
        super(CNN_score_MLP,self).__init__()
        self.fc_layers=torch.nn.ModuleList()
        for idx, (in_size,out_size) in enumerate(zip(config['score_mlp_layers'][:-1],config['score_mlp_layers'][1:])):
            self.fc_layers.append(torch.nn.Linear(in_size,out_size))
            
    def forward(self,context,embeddings):     
        vector=torch.cat([context,embeddings],dim=-1)
        for idx in range(len(self.fc_layers)):
            vector=self.fc_layers[idx](vector)
            vector=torch.nn.ReLU()(vector)  
        return vector  
    
    
class RNN_score_MLP(torch.nn.Module):
    def __init__(self,config):
        super(RNN_score_MLP,self).__init__()
        self.fc_layers=torch.nn.ModuleList()
        for idx, (in_size,out_size) in enumerate(zip(config['score_mlp_layers'][:-1],config['score_mlp_layers'][1:])):
            self.fc_layers.append(torch.nn.Linear(in_size,out_size))
            
    def forward(self,context,embeddings):     
        vector=torch.cat([context,embeddings],dim=-1)
        for idx in range(len(self.fc_layers)):
            vector=self.fc_layers[idx](vector)
            vector=torch.nn.ReLU()(vector)  
        return vector  