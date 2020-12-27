# -*- coding: utf-8 -*-
"""
Created on Wed Nov  6 17:48:43 2019

@author: User
"""

import torch

class GMF(torch.nn.Module):
    def __init__(self,config):
        super(GMF,self).__init__()
        
    def forward(self,hidden,types): 
# =============================================================================
#         hidden=hidden.expand(types.size()[0],-1,-1) # (n_types,batch_size,hidden_size)
#         hidden=hidden.transpose(0,1) # (batch_size,n_types,hidden_size)
#         types=types.expand(hidden.size()[0],-1,-1) # (batch_size,n_types,hidden_size)
# =============================================================================
        element_product=torch.mul(hidden,types)

        
        return element_product