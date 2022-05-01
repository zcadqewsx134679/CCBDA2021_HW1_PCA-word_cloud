# -*- coding: utf-8 -*-
"""
Created on Sat Apr  3 19:40:09 2021

@author: USER
"""
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler

def myPCA(datapath,target=2):
    df = pd.read_csv(datapath,encoding="cp437")
    sample_X = []
    sample_Y = []
    sample_list = {'1':50,'2':50,'10':50,'15':50,'17':50,'19':50,'20':50,'22':50,'23':50,'24':50,'25':50,'26':50,'27':50,'28':50,'29':50,'43':50}

    for i in range(df.shape[0]):
        try:
            if df["category_id"].loc[i] in sample_list and sample_list[df["category_id"].loc[i]] > 0:
                sample_X.append([float(df["views"].loc[i]),
                                 float(df["likes"].loc[i]),
                                 float(df["dislikes"].loc[i]),
                                 float(df["comment_count"].loc[i]),
                                 bool(df["comments_disabled"].loc[i]),
                                 bool(df["ratings_disabled"].loc[i]),
                                 bool(df["video_error_or_removed"].loc[i])
                                 ] )
                sample_Y.append( str(df["category_id"].loc[i]) )
                sample_list[df["category_id"].loc[i]] -= 1
        except:
            pass
    
    X = np.array(sample_X)
    Y = np.array(sample_Y)
    
    X_scaled = StandardScaler().fit_transform(X)
    features = X_scaled.T
    cov_matrix = np.cov(features)
    values, vectors = np.linalg.eig(cov_matrix)
    explained_variances = []
    for i in range(len(values)):
        explained_variances.append(values[i] / np.sum(values))
    
    projected = np.zeros((len(sample_X),target))
    for i in range(target):
        projected[:,i] = X_scaled.dot(vectors.T [i])

    res = pd.DataFrame(projected,columns= ['PC'+str(i+1) for i in range(target)])
    
    res ['Y'] = Y

    return res