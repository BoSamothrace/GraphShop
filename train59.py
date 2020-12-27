# -*- coding: utf-8 -*-
"""
Created on Wed Aug 28 22:36:29 2019

@author: User
"""
import pandas as pd
from data import SampleGenerator,gen_neg
from Procedure59 import ProcedureEngine
import codecs

config={'alias':'GNN59',
        'model_dir':'checkpoints/{}_Epoch{}_valmse{:.4f}_testmse{:.4f}_ndcg{:.4f}.model',
        'num_epoch':200,
        'batch_size':10,
        'n_types':4,
        'n_shops':7329,
        'n_distances':8,
        'n_prices':5,
        'n_regions':29,
        'optimizer':'adam',
        'adam_lr':1e-3,
        'l2_regularization':0,
        'type_hidden_size':16,
        'distance_hidden_size':8,
        'price_hidden_size':5,
        'region_hidden_size':16,
        'shop_mlp_layers':[32,64,16],
        'combine_mlp_layers':[24,24,64,128,128,64,16,8,1],
        'score_mlp_layers':[32,64,8],
        'attn_model':'dot',
        'test_size':500,
        'GNNStep':3}


data_set='b_fengtai'
total_dir='data/'+data_set+'.csv'
train_dir='data/'+data_set+'_train.csv'
val_dir='data/'+data_set+'_val.csv'
test_dir='data/'+data_set+'_test.csv'
neighbor_dir='data/'+data_set+'_neighbors.txt'

total_set=pd.read_csv(total_dir)
train_set=pd.read_csv(train_dir) 
val_set=pd.read_csv(val_dir)
test_set=pd.read_csv(test_dir)

f=open(neighbor_dir,'r')
Neighbors=f.read()
Neighbors=eval(Neighbors)
f.close()

# =============================================================================
# #total_dir='n_haidian_7329_600.csv'
# total_dir='chRegionNumber3.csv'
# total_set=pd.read_csv(total_dir)
# train_dir='chTrain_set.csv'
# train_set=pd.read_csv(train_dir)
# test_dir='chTest_set.csv'
# test_set=pd.read_csv(test_dir)
# 
# f=open('chNeighbors.txt','r')
# Neighbors=f.read()
# Neighbors=eval(Neighbors)
# f.close()
# =============================================================================


fr = open("type2index.txt",'r+',encoding='utf-8')
type2index = eval(fr.read())   #读取的str转换为字典
fr.close()
types=list(type2index.keys())
config['n_types']=len(types)
#types=list(total_set['subType'].value_counts().index)
total_set,train_set,val_set,test_set=total_set[total_set['subType'].isin(types)],train_set[train_set['subType'].isin(types)],val_set[val_set['subType'].isin(types)],test_set[test_set['subType'].isin(types)]

# region2Index for embedding    
regions=set(total_set['region_number'])
config['n_regions']=len(regions)
region2Index={}
for index,region in enumerate(regions):
    region2Index[region]=index    




f=codecs.open('haidian_7329_600_shopID2NegativeTypes.txt','r','utf-8')
shopID2NegativeTypes=f.read()
shopID2NegativeTypes=eval(shopID2NegativeTypes)
f.close()
for shop in shopID2NegativeTypes.keys():
    shopID2NegativeTypes[shop]=[type_ for type_ in shopID2NegativeTypes[shop
                        ] if type_ in types]



sample_generator=SampleGenerator(config,total_set,Neighbors,train_set,val_set,test_set,shopID2NegativeTypes,type2index,region2Index)
engine=ProcedureEngine(config)

for epoch in range(config['num_epoch']):
    print('Epoch{}starts !'.format(epoch))
    print('_'*80)

    engine.train_an_epoch(sample_generator,epoch)
    val_mse,test_mse,ndcg=engine.evaluate(sample_generator,epoch)
    engine.save(config['alias'],epoch,val_mse,test_mse,ndcg)
    