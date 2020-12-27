# -*- coding: utf-8 -*-
"""
Created on Fri Sep  6 19:51:25 2019

@author: User
"""

import torch

class Combine_MLP(torch.nn.Module):
    def __init__(self,config):
        super(Combine_MLP,self).__init__()
        weights=torch.tensor([[0.33],[0.33],[0.33]])
        self.weight=torch.autograd.Variable(weights,requires_grad=True).cuda()
        self.fc_layers=torch.nn.ModuleList()
        for idx, (in_size,out_size) in enumerate(zip(config['combine_mlp_layers'][:-1],config['combine_mlp_layers'][1:])):
            self.fc_layers.append(torch.nn.Linear(in_size,out_size))
            
    def forward(self,Pair_result,CNN_result,RNN_result): 
# =============================================================================
#         vector=torch.stack((general_context,distance_context,type_context),dim=-1)
#         vector=vector.matmul(self.weight).squeeze(-1)
# =============================================================================
        vector=torch.cat([Pair_result,CNN_result,RNN_result],dim=-1)
        for idx in range(len(self.fc_layers)):
            vector=self.fc_layers[idx](vector)
            vector=torch.nn.ReLU()(vector)
            
        return vector 