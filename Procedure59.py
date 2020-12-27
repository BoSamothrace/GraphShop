# -*- coding: utf-8 -*-
"""
Created on Fri Oct  4 21:36:11 2019

@author: User
"""

import torch
from engine import Engine
from ShopMLP import Pair_Shop_MLP,CNN_Shop_MLP,RNN_Shop_MLP
from CombineMLP import Combine_MLP
#from ScoreMLP import Score_MLP
from scoreMLP import Pair_score_MLP, CNN_score_MLP, RNN_score_MLP

import numpy as np

class Procedure(torch.nn.Module):
    def __init__(self,config):
        super(Procedure,self).__init__()
        self.config=config
        
        # type-based
        self.Pair_embedding_types=torch.nn.Embedding(config['n_types']+1,
                                                config['type_hidden_size'],
                                                padding_idx=config['n_types'])
        
        self.embedding_Pair_distances=torch.nn.Embedding(config['n_distances']+1,
                                                    config['distance_hidden_size'],
                                                    padding_idx=config['n_distances'])
        self.embedding_Pair_prices=torch.nn.Embedding(config['n_prices']+1,
                                                 config['price_hidden_size'],
                                                 padding_idx=config['n_prices'])
        self.Pair_shop_mlp=Pair_Shop_MLP(config)
        self.Pair_score_mlp=Pair_score_MLP(config)
        
        
        #CNN
        self.CNN_embedding_types=torch.nn.Embedding(config['n_types']+1,
                                                config['type_hidden_size'],
                                                padding_idx=config['n_types'])
        self.embedding_CNN_distances=torch.nn.Embedding(config['n_distances']+1,
                                                    config['distance_hidden_size'],
                                                    padding_idx=config['n_distances'])
        self.embedding_CNN_prices=torch.nn.Embedding(config['n_prices']+1,
                                                 config['price_hidden_size'],
                                                 padding_idx=config['n_prices'])
        self.CNN_shop_mlp=CNN_Shop_MLP(config)
        self.CNN_score_mlp=CNN_score_MLP(config)
        
        self.typeConv=torch.nn.Sequential(
                torch.nn.Conv2d(1,1,3,1,1),
                torch.nn.ReLU(),
                torch.nn.Conv2d(1,1,3,1,1),
                torch.nn.ReLU()
                )
        
        
        
        #RNN
        self.RNN_embedding_types=torch.nn.Embedding(config['n_types']+1,
                                                config['type_hidden_size'],
                                                padding_idx=config['n_types'])
        
        self.embedding_RNN_prices=torch.nn.Embedding(config['n_prices']+1,
                                                 config['price_hidden_size'],
                                                 padding_idx=config['n_prices'])
        
        self.embedding_RNN_distances=torch.nn.Embedding(config['n_distances']+1,
                                                    config['distance_hidden_size'],
                                                    padding_idx=config['n_distances'])
        self.RNN_Shop_mlp=RNN_Shop_MLP(config)
        self.RNN_score_mlp=RNN_score_MLP(config)
        
        self.distanceLSTM=torch.nn.LSTM(
                input_size=16,
                hidden_size=16,
                num_layers=3,
                batch_first=True
                )
        
        
        self.combine_mlp=Combine_MLP(config)
        
        
        
        if torch.cuda.is_available():
            self.Pair_embedding_types=self.Pair_embedding_types.cuda()
            self.embedding_Pair_distances=self.embedding_Pair_distances.cuda()
            self.embedding_Pair_prices=self.embedding_Pair_prices.cuda()
            self.pair_shop_mlp=self.Pair_shop_mlp.cuda()
            self.Pair_score_mlp=self.Pair_score_mlp.cuda()
            
            
            self.CNN_embedding_types=self.CNN_embedding_types.cuda()
            self.embedding_CNN_distances=self.embedding_CNN_distances.cuda()
            self.embedding_CNN_prices=self.embedding_CNN_prices.cuda()
            self.CNN_shop_mlp=self.CNN_shop_mlp.cuda()
            self.CNN_score_mlp=self.CNN_score_mlp.cuda()
            self.typeConv=self.typeConv.cuda()
            
            
            self.RNN_embedding_types=self.RNN_embedding_types.cuda()
            self.embedding_RNN_prices=self.embedding_RNN_prices.cuda()
            self.embedding_RNN_distances=self.embedding_RNN_distances.cuda()
            self.RNN_shop_mlp=self.RNN_Shop_mlp.cuda()
            self.RNN_score_mlp=self.RNN_score_mlp.cuda()
            self.distanceLSTM=self.distanceLSTM.cuda()
            

            self.combine_mlp=self.combine_mlp.cuda()

        
        
        
    def forward(self,types,regions,ratings,prices,lengthes,neighbortypes,
                neighbordistances,neighborratings,neighborcomments,
                neighborprices,neighborgroups): 
        
        #Pair based aggregation
        Pair_type_embedding=self.Pair_embedding_types(types)
        Pair_neighbortypes_embedding=self.Pair_embedding_types(neighbortypes)   #(batch_size, n_shops, shop_hidden_size)
        
        Pair_neighbordistances_embedding=self.embedding_Pair_distances(neighbordistances)
        Pair_neighborprices_embedding=self.embedding_Pair_prices(neighborprices)
        
        pair_shops_hidden=self.pair_shop_mlp(Pair_neighbortypes_embedding,Pair_neighbordistances_embedding,neighborratings,
                                  neighborcomments,Pair_neighborprices_embedding,neighborgroups) #(10,15,16)
        
        aggregator=[]
        for i,j in enumerate(lengthes):
            if j ==0:
                aggregator.append(torch.FloatTensor(np.zeros((1,pair_shops_hidden.shape[-1]))).cuda())
            else:
                aggregator.append(torch.mean(pair_shops_hidden[i:i+1,:j,:],1,False)) 
        pair_context=torch.cat(aggregator)   #(batch_size,hidden_size) (10,16)
        
        Pair_result=self.Pair_score_mlp(pair_context,Pair_type_embedding)

        #GNN+CNN
       
        CNN_type_embedding=self.CNN_embedding_types(types)
        CNN_neighbortypes_embedding=self.CNN_embedding_types(neighbortypes)   #(batch_size, n_shops, shop_hidden_size)
        
        CNN_neighbordistances_embedding=self.embedding_CNN_distances(neighbordistances)
        CNN_neighborprices_embedding=self.embedding_CNN_prices(neighborprices)
        
        CNN_shops_hidden=self.CNN_shop_mlp(CNN_neighbortypes_embedding,CNN_neighbordistances_embedding,neighborratings,
                                  neighborcomments,CNN_neighborprices_embedding,neighborgroups) #(10,15,16)
        
        type_based_matrix=self.g_type_based_matrix(CNN_shops_hidden,neighbortypes) #(10,126,16)
        if torch.cuda.is_available():
            type_based_matrix=type_based_matrix.cuda()
        type_matrix=type_based_matrix.view(type_based_matrix.shape[0],1,type_based_matrix.shape[1],
                                           type_based_matrix.shape[2])
        type_context_1=self.typeConv(type_matrix).squeeze() #(10,126,16)
        type_context_2=torch.empty(type_context_1.shape[0],type_context_1.shape[-1]).cuda() #(10,16)
        for i,t in enumerate(types):
            type_context_2[i]=type_context_1[i][t.item()]  
        #type_context=type_context.view(type_context.size(0),-1) #15位 #(10,15)
        #type_context_3=self.GNN_CNN_MLP(type_context_2)
        CNN_result=self.CNN_score_mlp(type_context_2,CNN_type_embedding)
        
        
        #GNN+RNN
        RNN_type_embedding=self.RNN_embedding_types(types)
        
        RNN_neighbortypes_embedding=self.RNN_embedding_types(neighbortypes)   #(batch_size, n_shops, shop_hidden_size)
        RNN_neighborprices_embedding=self.embedding_RNN_prices(neighborprices)
        RNN_neighbordistances_embedding=self.embedding_RNN_distances(neighbordistances)
        
        rnn_neighbors_hidden=self.RNN_Shop_mlp(RNN_neighbortypes_embedding,RNN_neighbordistances_embedding,neighborratings,
                                  neighborcomments,RNN_neighborprices_embedding,neighborgroups)
        if torch.cuda.is_available():
            rnn_neighbors_hidden=rnn_neighbors_hidden.cuda()  
        distance_based_matrix=self.g_distance_based_matrix(rnn_neighbors_hidden,neighbordistances) #(10,6,16)
        #distance_context1=distance_based_matrix.reshape(distance_based_matrix.shape[0],-1)
        
        distance_based_context,(h_n,h_c)=self.distanceLSTM(distance_based_matrix)
        distance_context2=distance_based_context[:,-1,:] #(10,16)
        RNN_result=self.RNN_score_mlp(distance_context2,RNN_type_embedding)
        
        
        
        final_scores=self.combine_mlp(Pair_result,CNN_result,RNN_result)
        
        scores=final_scores.squeeze(1)

        return scores

    def g_type_based_matrix(self,shops_hidden,neighbortypes):
        '''
            shops_hidden: tensor (batch_size,n_neighbors,hidden_size)
            neighbortypes: tensor (batch_size,n_neighbors)
        '''
        neighbors_types=neighbortypes.cpu()
        batch_size,n_types=shops_hidden.shape[0],self.config['n_types']
        matrix=torch.tensor(np.zeros((batch_size,n_types,16))).float()

        
        for i in range(batch_size):
            for j in range(n_types):
                index=np.where(neighbors_types[i]==j)
                vector1=shops_hidden[i][index]
                if len(vector1)!=0:
                    matrix[i][j]=torch.sum(vector1).float()
                    
        return matrix # tensor (batch_size,n_types,32)    

    def g_distance_based_matrix(self,rnn_neighbors_hidden,neighbordistances):
        
        n_distances=1
        neighbordistances=neighbordistances.cpu().data.numpy()
        batch_size=neighbordistances.shape[0]
        matrix=torch.tensor(np.zeros((batch_size,n_distances,16))).float()
    
        vector=rnn_neighbors_hidden
        try: 
            lastVector=vector[0][0]
        except:
            lastVector=torch.zeros(16)
        for i in range(batch_size):
            for k,j in zip(range(n_distances),reversed(range(n_distances))):
                index=np.where(neighbordistances[i]==j)
                vector1=vector[i][index]
                if len(vector1)!=0:
                    lastVector=torch.mean(vector1).float()
                    matrix[i][k]=lastVector
                else:
                    matrix[i][k]=lastVector
                    
        return matrix.cuda() # tensor (batch_size,time_step,Rnninput_size)  
    
class ProcedureEngine(Engine):
    def __init__(self,config):
        self.model=Procedure(config)
        super(ProcedureEngine,self).__init__(config)
        
        #use pretrained
        model_dict=self.model.state_dict()
        
        pair=torch.load('checkpoints/GNN60_Epoch3_valmse4.2323_testmse4.0734_ndcg0.7604.model')
        pretrained_pair={k:v for k,v in pair.items() if k in model_dict}
        model_dict.update(pretrained_pair)
        
        cnn=torch.load('checkpoints/GNN61_Epoch12_valmse4.8727_testmse4.8142_ndcg0.7532.model')
        pretrained_cnn={k:v for k,v in cnn.items() if k in model_dict}
        model_dict.update(pretrained_cnn)
        
        rnn=torch.load('checkpoints/GNN62_Epoch2_valmse4.5752_testmse4.4591_ndcg0.7548.model')
        pretrained_rnn={k:v for k,v in rnn.items() if k in model_dict}
        model_dict.update(pretrained_rnn)
        
        self.model.load_state_dict(model_dict)
        
# =============================================================================
#         for i,p in enumerate(self.model.parameters()):
#             if i < 49 or i> 56 :
#                 p.requires_grad = False
# =============================================================================
        
        
        
        
        
        
        