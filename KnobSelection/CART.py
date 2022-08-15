import sys
sys.path.append("../")
from KnobSelection.Plackett_Burman import *
from DBConnector.BaseExecutor import *
from config import *
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import StandardScaler
import numpy as np
import pickle

class CART():
    def __init__(self,LHS_N:int,rf_nestimator=100,max_depth=None,max_leaf_nodes=None) -> None:
        self.n=LHS_N
        self.n_estimator=rf_nestimator
        self.max_depth=max_depth
        self.max_leaf_nodes=max_leaf_nodes
        self.argsort=None

    def fit(self,db:Executor,knob_names:list,knob_max:list,knob_min:list,knob_type:list,workload:list):
        self.knob_names=knob_names
        knob_granularity=[]
        for i in range(len(knob_names)):
            knob_granularity.append((knob_max[i]-knob_min[i])/self.n)
        x=[]
        y=[]
        for i in range(self.n):
            knob_value=[]
            for j in range(len(knob_names)):
                knob_value.append(knob_min[j]+knob_granularity[j]*i)
            db.change_knob(knob_names,knob_value,knob_type)
            print("Change Knob: "+str(knob_value))
            thread_num=db.get_max_thread_num()
            latency,throughput=db.run_job(thread_num,workload)
            print("Latency: "+str(latency))
            print("Throughput: "+str(throughput))
            x.append(knob_value)
            y.append(latency)
        scaler=StandardScaler()
        x=scaler.fit_transform(x)
        model=RandomForestRegressor(n_estimators=self.n_estimator,max_depth=self.max_depth,max_leaf_nodes=self.max_leaf_nodes)
        model.fit(x,y)
        y_mean=0
        for y_i in y:
            y_mean+=y_i
        y_mean=y_mean/len(y)
        total_coef=[0 for i in range(len(knob_names))]
        for regressor in model.estimators_:
            y_pred=regressor.predict(x)
            ssr=0
            sse=0
            for i in range(len(y)):
                ssr+=(y_pred[i]-y_mean)**2
                sse+=(y_pred[i]-y[i])**2
            r_square=ssr/(ssr+sse)
            tree=regressor.tree_
            feature=tree.feature
            for i in range(len(feature)):
                if feature[i]>=0:
                    total_coef[feature[i]]-=r_square
        self.model=model
        self.total_coef=total_coef
        self.argsort=np.argsort(total_coef)
        return self
    
    def rank_knob(self,knob_values):
        if self.argsort is None:
            raise Exception("Please fit the model first")
        
        if len(knob_values)!=len(self.knob_names):
            raise Exception("Input knob value should be consistent with knob names")
        
        return [self.knob_names[i] for i in self.argsort],[knob_values[i] for i in self.argsort]
    
    def get_top_rank(self):
        if self.argsort is None:
            raise Exception("Fit the model first")
        return [self.knob_names[i] for i in self.argsort]
