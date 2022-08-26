from math import log
import sys
sys.path.append("../")
from DBConnector.BaseExecutor import Executor
from sklearn.ensemble import RandomForestClassifier
import numpy as np

class TF_IDF():
    def __init__(self,vocabulary=None) -> None:
        if vocabulary is None:
            self.vocabulary=["select","insert","update","delete","as","and","like","min","where","from","is","not","null","in","or"]
        else:
            self.vocabulary=vocabulary
    
    def fit_transform(self,db:Executor,workload:list,num_of_levels:int):
        self.tf=[]
        self.idf=[1 for i in range(len(self.vocabulary))]
        total_count=[0 for i in range(len(self.vocabulary))]
        self.target=[]
        for query in workload:
            query_tf=[0 for i in range(len(self.vocabulary))]
            small_query=query.lower()
            for i in range(len(self.vocabulary)):
                cnt=small_query.count(self.vocabulary[i])
                if cnt>0:
                    self.idf[i]+=1
                    total_count[i]+=cnt
                    query_tf[i]+=cnt
            self.tf.append(query_tf)
            latency,_=db.run_job(1,[query])
            self.target.append(latency)
        for i in range(len(self.vocabulary)):
            self.idf[i]=log(len(workload)/self.idf[i])
        for i in range(len(self.tf)):
            for j in range(len(self.vocabulary)):
                if total_count[j]>0:
                    self.tf[i][j]=self.tf[i][j]*self.idf[j]/total_count[j]
            self.target[i]=log(self.target[i])
           
        min_target=min(self.target)
        max_target=max(self.target)
        granularity=(max_target-min_target)/num_of_levels
        self.level=[]
        for i in range(len(workload)):
            self.level.append(int((self.target[i]-min_target)/granularity))
        self.model=RandomForestClassifier()
        self.model.fit(self.tf,self.level)
        self.proba=[]
        for i in range(len(workload)):
            self.proba.append(self.model.predict_proba([self.tf[i]])[0])
        self.workload_embedding=np.array([0.0 for i in range(len(self.proba[0]))])
        for pro in self.proba:
            print(pro)
            self.workload_embedding+=np.array(pro)
        return self.workload_embedding
