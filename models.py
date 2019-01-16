# -*- coding: utf-8 -*-
"""
Created on Mon Dec  4 08:52:43 2017

@author: E154709
"""

"""
AI claim scoring model files
"""

# Load modules
import numpy as np
import pandas as pd
import copy
import os
from collections import Counter
from sklearn.metrics import mean_squared_error, classification_report, accuracy_score
from sklearn.utils.validation import column_or_1d
import matplotlib.pylab as plt
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import auc, roc_auc_score, roc_curve, precision_recall_curve
from scipy import interp
from sklearn import ensemble
from sklearn.model_selection import GridSearchCV,ShuffleSplit, cross_val_predict
from sklearn.model_selection import learning_curve
from sklearn.calibration import calibration_curve
from sklearn import tree
import graphviz 
import itertools

from matplotlib.pyplot import subplots, show
from sklearn import preprocessing
from sklearn.decomposition import PCA
from sklearn.preprocessing import LabelEncoder

# models
class label_encoder(object):
    def fit_pd(self,df,cols=[]):
        '''
        fit all columns in the df or specific list. 
        generate a dict:
        {feature1:{label1:1,label2:2}, feature2:{label1:1,label2:2}...}
        '''
        if len(cols) == 0:
            cols = df.columns
        self.class_index = {}
        for f in cols:
            uf = df[f].unique()
            self.class_index[f] = {}
            index = 1
            for item in uf:
                self.class_index[f][item] = index
                index += 1
    
    def fit_transform_pd(self,df,cols=[]):
        '''
        fit all columns in the df or specific list and return an update dataframe.
        '''
        if len(cols) == 0:
            cols = df.columns
        newdf = copy.deepcopy(df)
        self.class_index = {}
        for f in cols:
            uf = df[f].unique()
            self.class_index[f] = {}
            index = 1
            for item in uf:
                self.class_index[f][item] = index
                index += 1
                
            newdf[f] = df[f].apply(lambda d: self.update_label(f,d))
        return newdf
    
    def transform_pd(self,df,cols=[]):
        '''
        transform all columns in the df or specific list from lable to index, return an update dataframe.
        '''
        newdf = copy.deepcopy(df)
        if len(cols) == 0:
            cols = df.columns
        for f in cols:
            if f in self.class_index:
                newdf[f] = df[f].apply(lambda d: self.update_label(f,d))
        return newdf
                
    def update_label(self,f,x):
        '''
        update the label to index, if not found in the dict, add and update the dict.
        '''
        try:
            return self.class_index[f][x]
        except:
            self.class_index[f][x] = max(self.class_index[f].values())+1
            return self.class_index[f][x]

class plotPCA(object):
    """
    define a class to normalize the dataFrame, perform PCA and plot PCA
    
     Parameters
    ----------
    X : {array-like, sparse matrix} of shape [n_samples, n_features]
        Training vectors, where n_samples is the number of samples
        and n_features is the number of features.

    Returns
    -------
    self : object

    Returns an instance of self.
    """
    def __init__(self, X):
        self.X = X
    
    def normalize_X(self):
        """
        Normalize the given data frame to a standardized zero mean and deviation
    
        """
        xvalue = self.values 
        self.normX_scaler = preprocessing.StandardScaler().fit(xvalue) 
        xScaled = self.normX_scaler.transform(xvalue)    
        self.normX = pd.DataFrame(xScaled)
        
        self.normX.index = self.index
        self.normX.columns = self.columns
        return self  
    
    def minmax_X(self):
        """
        Normalize the given data frame to a min/max
        
        """
        xvalue = self.values 
        self.normX_scaler = preprocessing.MinMaxScaler().fit(xvalue)  
        xScaled = self.normX_scaler.transform(xvalue)    
        self.normX = pd.DataFrame(xScaled)
        
        self.normX.index = self.index
        self.normX.columns = self.columns
        return self 
    
    def do_PCA(self):
        """
        conduct PCA on data
        """
        n_comps = len(self.normX.columns)
        self.pca = PCA(n_components = n_comps)
        self.dpca = self.pca.fit(self.normX).transform(self.normX)
        return self
    
    # plot PCA results    
    def pca_summary(self, out=True):
        from IPython.display import display
        names = ["PC"+str(i) for i in range(1, len(self.pca.explained_variance_ratio_)+1)]
        a = list(np.std(self.pca.transform(self.normX), axis=0))
        b = list(self.pca.explained_variance_ratio_)
        c = [np.sum(self.pca.explained_variance_ratio_[:i]) for i in range(1, len(self.pca.explained_variance_ratio_)+1)]
        columns = pd.MultiIndex.from_tuples([("sdev", "Standard deviation"), ("varprop", "Proportion of Variance"), ("cumprop", "Cumulative Proportion")])
        self.summary = pd.DataFrame(list(zip(a, b, c)), index=names, columns=columns)
  #      ioptPC, optPC = i, item for item in enumerate(c) if item > thred
        if out:
            print("Importance of components:")
            display(self.summary)
   #         print("Number of components %s greater than threshold:  %d" %ioptPC %optPC)
        return self


    def plot_pca_var(self):
        pcaVariance =self.pca.explained_variance_
        tot = sum(pcaVariance)
        ncomp = len(pcaVariance)
        var_exp = [(i/tot) for i in sorted(pcaVariance, reverse = True)]
        
        cum_var_exp = np.cumsum(var_exp)
        
        fig, ax = subplots(figsize=(10,4))
        ax.bar(range(1, ncomp+1), var_exp, alpha =0.5, align='center',
                label = 'individual explained variance')
        ax.step(range(1, ncomp+1), cum_var_exp, where = 'mid',
                 label = 'cumulative explained variance')
        ax.set_ylabel('Explained variance ratio')
        ax.set_xlabel('Principal components')
        ax.legend(loc='best')
        show()

    
    def plot_PCA(self, y, tag):
        """
        plot two PCs
        """
        target_names = ['Pass', 'Fail']
        # Percentage of variance explained for each components
        print('explained variance ratio (first two components): %s'
              % str(self.pca.explained_variance_ratio_))

        fig, ax = subplots(figsize=(10,4))
        colors = ['navy', 'darkorange']
        lw = 2

        for color, i, target_name in zip(colors, [0, 1], target_names):
            plt.scatter(self.dpca[y == i, 0], self.dpca[y == i, 1], color=color, alpha=.8, lw=lw,
                        label=target_name)
        
        ax.legend(loc='best', shadow=False, scatterpoints=1)
        ax.set_title('PCA of Multivariate Dataset: %s' %tag)
        ax.set_ylabel('PC2')
        ax.set_xlabel('PC1')
        show()
   