# -*- coding: utf-8 -*-
"""
Created on Tue Sep  3 23:48:14 2019

@author: User
"""
import torch

class Pair_Shop_MLP(torch.nn.Module):
    def __init__(self,config):
        super(Pair_Shop_MLP,self).__init__()
        self.fc_layers=torch.nn.ModuleList()
        for idx, (in_size,out_size) in enumerate(zip(config['shop_mlp_layers'][:-1],config['shop_mlp_layers'][1:])):
            self.fc_layers.append(torch.nn.Linear(in_size,out_size))
            
    def forward(self,shops_embedding,distances_embedding,ratings,comments,prices_embedding,group):     
        ratings=ratings.view(ratings.shape[0],ratings.shape[1],1)
        comments=comments.view(comments.shape[0],comments.shape[1],1)
        group=group.view(group.shape[0],group.shape[1],1)
        vector=torch.cat([shops_embedding,distances_embedding,ratings,comments,prices_embedding,group],dim=-1)
        for idx in range(len(self.fc_layers)):
            vector=self.fc_layers[idx](vector)
            vector=torch.nn.ReLU()(vector)  
        return vector 
    
class CNN_Shop_MLP(torch.nn.Module):
    def __init__(self,config):
        super(CNN_Shop_MLP,self).__init__()
        self.fc_layers=torch.nn.ModuleList()
        for idx, (in_size,out_size) in enumerate(zip(config['shop_mlp_layers'][:-1],config['shop_mlp_layers'][1:])):
            self.fc_layers.append(torch.nn.Linear(in_size,out_size))
            
    def forward(self,shops_embedding,distances_embedding,ratings,comments,prices_embedding,group):     
        ratings=ratings.view(ratings.shape[0],ratings.shape[1],1)
        comments=comments.view(comments.shape[0],comments.shape[1],1)
        group=group.view(group.shape[0],group.shape[1],1)
        vector=torch.cat([shops_embedding,distances_embedding,ratings,comments,prices_embedding,group],dim=-1)
        for idx in range(len(self.fc_layers)):
            vector=self.fc_layers[idx](vector)
            vector=torch.nn.ReLU()(vector)  
        return vector 
    
class RNN_Shop_MLP(torch.nn.Module):
    def __init__(self,config):
        super(RNN_Shop_MLP,self).__init__()
        self.fc_layers=torch.nn.ModuleList()
        for idx, (in_size,out_size) in enumerate(zip(config['shop_mlp_layers'][:-1],config['shop_mlp_layers'][1:])):
            self.fc_layers.append(torch.nn.Linear(in_size,out_size))
            
    def forward(self,shops_embedding,distances_embedding,ratings,comments,prices_embedding,group):     
        ratings=ratings.view(ratings.shape[0],ratings.shape[1],1)
        comments=comments.view(comments.shape[0],comments.shape[1],1)
        group=group.view(group.shape[0],group.shape[1],1)
        vector=torch.cat([shops_embedding,distances_embedding,ratings,comments,prices_embedding,group],dim=-1)
        for idx in range(len(self.fc_layers)):
            vector=self.fc_layers[idx](vector)
            vector=torch.nn.ReLU()(vector)  
        return vector 
    