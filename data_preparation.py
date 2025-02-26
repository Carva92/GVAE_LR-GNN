#!/usr/bin/env python
# coding: utf-8

# In[7]:


import pandas as pd
pd.options.mode.chained_assignment = None
from sklearn.metrics.pairwise import cosine_similarity
import warnings
import platform


def e_distance(conditions_features, patients):
    from sklearn.metrics.pairwise import euclidean_distances

    
    dist = euclidean_distances(conditions_features, conditions_features)
    #similarity_p = cosine_similarity(conditions_features)
    index = list(conditions_features.index.values)
    ids = patients['RRID'].tolist()
    warnings.filterwarnings('ignore')
    df = []
    for i in range(len(index)):  
        for j in range(1+i,len(index)): 
            temp = {'PatientX':ids[i],'PatientY':ids[j],'Euclidean_Dist':dist[i,j]}   
            df.append(temp)

    df = pd.DataFrame(df)
    df = df.sort_values(by=['PatientX', 'Euclidean_Dist'],ascending = [True, False])
    df = df.reset_index()
    del df['index']
    
    return df





