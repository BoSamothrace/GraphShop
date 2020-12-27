# -*- coding: utf-8 -*-
"""
Created on Thu Sep 12 20:50:40 2019

@author: User
"""

import math
import pandas as pd

class Metrics(object):
    def __init__(self,top_k):
        self._top_k=top_k
        self._subjects=None
    
    @property    
    def top_k(self):
        return self._top_k
    
    @top_k.setter
    def top_k(self,top_k):
        self._top_k=top_k
        
    @property
    def subjects(self):
        return self._subjects
    
    @subjects.setter
    def subjects(self,subjects):
        assert isinstance(subjects,list)   
        targets=subjects[0]
        scores=subjects[1]
        full=pd.DataFrame({'targets':targets,
                           'scores':scores})
        full['predict']=full['scores'].apply(lambda x: x.index(max(x)))
        
        self._subjects=full
        
    def cal_hit_ratio(self):
        full=self._subjects
        correct=full[full['targets']==full['predict']]
        return len(correct)*1.0/len(full)
        
        
        