# -*- coding: utf-8 -*-
"""
Created on Wed Nov  6 17:48:43 2019

@author: User
"""

import torch

class GNN_CNN_MLP(torch.nn.Module):
    def __init__(self,config):
        super(GNN_CNN_MLP,self).__init__()
        self.layer1=torch.nn.Linear(16,16)
        self.layer2=torch.nn.Linear(16,16)
        self.layer3=torch.nn.Linear(16,16)
        
    def forward(self,hidden): 
# =============================================================================
#         hidden=hidden.expand(types.size()[0],-1,-1) # (n_types,batch_size,hidden_size)
#         hidden=hidden.transpose(0,1) # (batch_size,n_types,hidden_size)
#         types=types.expand(hidden.size()[0],-1,-1) # (batch_size,n_types,hidden_size)
# =============================================================================
        vector1=torch.cat([hidden],dim=-1)

        vector2=torch.nn.ReLU()(self.layer1(vector1))
        vector3=torch.nn.ReLU()(self.layer2(vector2))
        GNN_CNN_result=torch.nn.ReLU()(self.layer3(vector3))
        return GNN_CNN_result
        