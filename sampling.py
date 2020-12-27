# -*- coding: utf-8 -*-
"""
Created on Wed Dec 25 22:08:39 2019

@author: User
"""

import pandas as pd
import numpy as np


def undersampling(data_set):
    '''
    data_set ['shopID', 'rating', 'comments', 'price', 'rating1', 'rating2',
       'rating3', 'group', 'type', 'subType', 'region', 'BaiduID',
       'coordinate', 'lat', 'lng', 'i_lat', 'i_lng', 'region_number']
    '''
    types=list(data_set['subType'].value_counts().index)
    new_set=[]
    n=data_set['subType'].value_counts().values.min()
    
    for atype in types:
        new_set.append(data_set[data_set['subType']==atype].sample(n))
    
    new_set=pd.concat(new_set)
    
    return new_set