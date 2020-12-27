# -*- coding: utf-8 -*-
"""
Created on Fri Aug 30 23:13:18 2019

@author: User
"""
import torch
from utils import use_optimizer,save_checkpoint
from metrics import Metrics
import numpy as np
import pandas as pd
from sklearn.metrics import ndcg_score 

class Engine(object):
    def __init__(self,config):
        self.config=config
        self._metron=Metrics(1)
        self.opt=use_optimizer(self.model,config)
        self.crit=torch.nn.MSELoss()
        torch.autograd.set_detect_anomaly(True)
        
        if torch.cuda.is_available():
            self.crit=self.crit.cuda()
        
    def train_single_batch(self,types,regions,targets,ratings,prices,lengthes,neighbortypes,
                neighbordistances,neighborratings,neighborcomments,
                neighborprices,neighborgroups):
        
        if torch.cuda.is_available():
            types=types.cuda()
            regions=regions.cuda()
            targets=targets.cuda()
            ratings=ratings.cuda()
            prices=prices.cuda()
            neighbortypes=neighbortypes.cuda()
            neighbordistances=neighbordistances.cuda()
            neighborratings=neighborratings.cuda()
            neighborcomments=neighborcomments.cuda()
            neighborprices=neighborprices.cuda()
            neighborgroups=neighborgroups.cuda()
        
        self.opt.zero_grad()
        scores=self.model(types,regions,ratings,prices,lengthes,neighbortypes,
                neighbordistances,neighborratings,neighborcomments,
                neighborprices,neighborgroups) #(batch_size)
        loss=self.crit(scores,targets)
        loss.backward()
        self.opt.step()
        loss=loss.item()
        return loss

    
    def train_an_epoch(self,sample_generator,epoch_id):
        self.model.train()
        total_loss=0
        batches=sample_generator.generate_train_batch(self.config['batch_size'])
        for batch_id, batch in enumerate(batches):
            if len(batch)==1:
                continue
            (types,regions,targets,ratings,prices,lengthes,neighbortypes,
                neighbordistances,neighborratings,neighborcomments,
                neighborprices,neighborgroups)=sample_generator.get_train_batch(batch)
            loss=self.train_single_batch(types,regions,targets,ratings,prices,lengthes,neighbortypes,
                neighbordistances,neighborratings,neighborcomments,
                neighborprices,neighborgroups)
            print('[Training Epoch{}] Batch {}, loss {}'.format(epoch_id,
                  batch_id,loss))
            total_loss+=loss
            
    def evaluate(self,sample_generator,epoch_id):
        loss_fn=torch.nn.MSELoss()
        
        
        val_data=sample_generator.generate_val_batch(self.config['test_size'])
        
        (types,regions,targets,ratings,prices,lengthes,neighbortypes,
                neighbordistances,neighborratings,neighborcomments,
                neighborprices,neighborgroups)=sample_generator.get_train_batch(val_data)
        
        if torch.cuda.is_available():
            types=types.cuda()
            regions=regions.cuda()
            targets=targets.cuda()
            ratings=ratings.cuda()
            prices=prices.cuda()
            neighbortypes=neighbortypes.cuda()
            neighbordistances=neighbordistances.cuda()
            neighborratings=neighborratings.cuda()
            neighborcomments=neighborcomments.cuda()
            neighborprices=neighborprices.cuda()
            neighborgroups=neighborgroups.cuda()
        
        val_scores=self.model(types,regions,ratings,prices,lengthes,neighbortypes,
                neighbordistances,neighborratings,neighborcomments,
                neighborprices,neighborgroups)
        
        val_mse=loss_fn(val_scores,targets)
        ########################################################################
        
        evaluate_data=sample_generator.generate_test_batch(self.config['test_size'])
        
        (types,regions,targets,ratings,prices,lengthes,neighbortypes,
                neighbordistances,neighborratings,neighborcomments,
                neighborprices,neighborgroups)=sample_generator.get_train_batch(evaluate_data)
        
        if torch.cuda.is_available():
            types=types.cuda()
            regions=regions.cuda()
            targets=targets.cuda()
            ratings=ratings.cuda()
            prices=prices.cuda()
            neighbortypes=neighbortypes.cuda()
            neighbordistances=neighbordistances.cuda()
            neighborratings=neighborratings.cuda()
            neighborcomments=neighborcomments.cuda()
            neighborprices=neighborprices.cuda()
            neighborgroups=neighborgroups.cuda()
        
        test_scores=self.model(types,regions,ratings,prices,lengthes,neighbortypes,
                neighbordistances,neighborratings,neighborcomments,
                neighborprices,neighborgroups)
        
        test_mse=loss_fn(test_scores,targets)
        
        result=pd.DataFrame({'regions':regions.tolist(),'targets':targets.tolist(),
                             'scores':test_scores.tolist()})
        regionList=list(result['regions'].value_counts().index)
        ndcg=[]
        for iregion in regionList:
            if len(result[result['regions']==iregion]['targets'].values) > 1:
                ndcg.append(ndcg_score(result[result['regions']==iregion]['targets'].values.reshape((1,-1)),
                                   result[result['regions']==iregion]['scores'].values.reshape((1,-1))))
        ndcg=np.mean(ndcg)
        
        return val_mse,test_mse,ndcg
    
    def save(self,alias,epoch_id,val_mse,test_mse,ndcg):
        model_dir=self.config['model_dir'].format(alias,epoch_id,val_mse,test_mse,ndcg)
        save_checkpoint(self.model,model_dir)