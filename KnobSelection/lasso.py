import numpy as np
from sklearn.linear_model import lasso_path
from sklearn.preprocessing import StandardScaler

class Lasso():
    def __init__(self,knobs):
        self.alpha=None
        self.coef=None
        self.ranking=None
        self.knobs_names=knobs
    
    def fit(self,knob_values,metrics):
        if knob_values is None or metrics is None:
            raise Exception("Input should not be None")

        if len(knob_values[0])!=len(self.knobs_names):
            raise Exception("Input knob value should be consistent with knob names")

        scaler=StandardScaler()
        knob_values=scaler.fit_transform(knob_values)
        metric_scaler=StandardScaler()
        metrics=metric_scaler.fit_transform(metrics)
        self.alpha,self.coef,_=lasso_path(knob_values,metrics)
        self.ranking=[]
        self.feature_steps=[[] for i in range(len(self.knobs_names))]
        for target_coef_path in self.coef:
            for i,feature_coef_path in enumerate(target_coef_path):
                step=0
                for value_on_path in feature_coef_path:
                    if value_on_path==0:
                        step=step+1
                    else:
                        break
                self.feature_steps[i].append(step)
        self.ranking=np.array([np.mean(step_list) for step_list in self.feature_steps])
        return self

    def rank_knob(self,knob_values):
        if self.ranking is None:
            raise Exception("Please fit the model first")
        
        if len(knob_values)!=len(self.ranking):
            raise Exception("Input knob value should be consistent with knob names")
        
        idx=np.argsort(self.ranking)
        return [self.knobs_names[i] for i in idx],[knob_values[i] for i in idx]
    
    def get_top_rank(self):
        idx=np.argsort(self.ranking)
        return [self.knobs_names[i] for i in idx]
