# -*- coding: utf-8 -*-
"""
Created on Fri Aug 30 23:40:49 2019

@author: User
"""
import torch
import torch.nn.functional as F
from engine import Engine


class GNN(torch.nn.Module):
    def __init__(self,config):
        super(GNN,self).__init__()
        self.config=config
        self.step=config['GNNStep']
        self.hidden_size=config['shop_mlp_layers'][-1]
        self.gate_size=3*self.hidden_size
        
        self.w_ih = torch.nn.Parameter(torch.Tensor(self.gate_size, self.hidden_size))
        self.w_hh = torch.nn.Parameter(torch.Tensor(self.gate_size, self.hidden_size))
        self.b_ih = torch.nn.Parameter(torch.Tensor(self.gate_size))
        self.b_hh = torch.nn.Parameter(torch.Tensor(self.gate_size))
        self.b_iah = torch.nn.Parameter(torch.Tensor(self.hidden_size))
        self.b_oah = torch.nn.Parameter(torch.Tensor(self.hidden_size))

        self.linear_edge_in = torch.nn.Linear(self.hidden_size, self.hidden_size, bias=True)
        self.linear_edge_f = torch.nn.Linear(self.hidden_size, self.hidden_size, bias=True)
        
    def GNNCell(self,hidden):    #hidden (batch_size, n_shops, hidden_size)
        inputs=self.linear_edge_in(hidden)
        gi=F.linear(inputs,self.w_ih,self.b_ih)
        gh=F.linear(hidden,self.w_hh,self.b_hh)
        i_r,i_i,i_n=gi.chunk(3,2)
        h_r,h_i,h_n=gh.chunk(3,2)
        
        resetgate=torch.sigmoid(i_r+h_r)
        inputgate=torch.sigmoid(i_i+h_i)
        
        newgate=torch.tanh(i_n+resetgate*h_n)
        hy=newgate+inputgate*(hidden-newgate)
        return hy
    
    def forward(self,hidden):  
        for i in range(self.step):
            hidden=self.GNNCell(hidden)
        return hidden    
