from sklearn.decomposition import FactorAnalysis
from sklearn.cluster import KMeans
import numpy as np

from FeatureSelection.FeatureSelector import FeatureSelector

class FeatureSelector_FA_Kmeans(FeatureSelector):
    def __init__(self) -> None:
        self.coef=None
        self.data=None
        self.fa_model=None
        self.kmeans_model=None
        self.metric_repre=None
        self.label2metric=None

    def fit(self,data,fa_components=None,kmeans_k=None):
        '''
        data: matrix, each row represents a configuration, each column represents metric value under different configs
        '''
        if fa_components is None:
            self.fa_model=FactorAnalysis()
        else:
            self.fa_model=FactorAnalysis(n_components=fa_components)

         
        if kmeans_k is None:
            kmeans_k=8
        
        self.kmeans_model=KMeans(n_clusters=kmeans_k)
        
        self.fa_model.fit(data)
        self.coef=self.fa_model.components_

        self.metric_repre=[]
        for i in range(len(data[0])):
            metric_i_repre=[]
            for coef in self.coef:
                metric_i_repre.append(coef[i])
            self.metric_repre.append(metric_i_repre)
        
        self.kmeans_model.fit(self.metric_repre)
        self.label2metric={}
        tmp=np.array(self.kmeans_model.labels_)
        for i in range(kmeans_k):
            metric_ids=np.where(tmp==i)[0]
            self.label2metric[i]=metric_ids
        return self
    
    def transform(self,Xnew:list):
        '''
        Select a subset of db states
        Xnew: 1-D array, represents the state of metric
        '''
        if self.label2metric is None:
            return Xnew
        output=[]
        for cluster in self.label2metric:
            if len(cluster)>0:
                output.append(Xnew[cluster[0]])
        return np.array(output)
