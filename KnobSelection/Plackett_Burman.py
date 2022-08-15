from math import floor
import random
import numpy as np
import sys
sys.path.append("../")
from DBConnector.BaseExecutor import *
class Plackett_Burman():
    def __init__(self) -> None:
        pass
    
    def fit(self,db:Executor,knob_names:list,knob_min:list,knob_max:list,knob_type:list,workload:list,foldover=True,sampling_rate=-1):
        self.knob_names=knob_names
        n=len(knob_names)
        x=int(floor(n/4)+1)*4
        assert x%4==0
        if sampling_rate<=0:
            sample_workload=workload
        else:
            num_of_sample=int(sampling_rate*len(workload))
            sample_workload=random.sample(workload,num_of_sample)

        init=None
        if x==4:
            init=[1,-1,-1]
        elif x==8:
            init=[1,1,-1,-1,-1,-1,1]
        elif x==12:
            init=[1,1,1,1,-1,-1,-1,1,-1,-1,-1]
        elif x==16:
            init=[1,1,1,-1,-1,-1,-1,-1,-1,-1,-1,1,1,1,1]
        elif x==20:
            init=[1,1,1,-1,1,-1,1,-1,-1,-1,-1,1,1,-1,-1,1,-1,-1,1]
        elif x==24:
            init=[1,1,1,-1,1,-1,1,1,-1,-1,1,1,-1,-1,1,-1,1,-1,-1,-1,-1,-1,1]
        elif x==28:
            init=[1,1,1,-1,-1,1,1,1,-1,1,1,-1,-1,1,1,1,1,-1,-1,-1,-1,-1,-1,-1,-1,1,1]
        elif x==32:
            init=[1,1,1,1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,1,1,1,1,1,1,1,1,1,1,1]
        else:
            raise Exception("number of knobs is too large, hasn't been implemented yet")
        
        workload_ranking_record=[]
        for query in sample_workload:
            rb_effect=[0 for i in range(len(knob_names))]
            db.change_knob(knob_name=knob_names,knob_value=knob_max,knob_type=knob_type)
            latency,_=db.run_job(1,[query])
            for i in range(len(knob_names)):
                rb_effect[i]+=latency
            current_knob=init
            for i in range(x-1):
                current_knob_value=[]
                for j in range(len(knob_names)):
                    if current_knob[j]==1:
                        current_knob_value.append(knob_max[j])
                    else:
                        current_knob_value.append(knob_min[j])
                db.change_knob(knob_name=knob_names,knob_value=current_knob_value,knob_type=knob_type)
                print("Change Knob: "+str(current_knob_value))
                latency,_=db.run_job(1,[query])
                print("Latency: "+str(latency))
                for j in range(len(knob_names)):
                    if current_knob[j]==1:
                        rb_effect[j]+=latency
                    else:
                        rb_effect[j]-=latency
                tmp=current_knob[0]
                for j in range(len(current_knob)-1):
                    current_knob[j]=current_knob[j+1]
                current_knob[len(current_knob)-1]=tmp

            if foldover:
                db.change_knob(knob_names,knob_min,knob_type)
                latency,_=db.run_job(1,[query])
                for i in range(len(knob_names)):
                    rb_effect[i]-=latency
                current_knob=init
                for i in range(x-1):
                    current_knob_value=[]
                    for j in range(len(knob_names)):
                        if current_knob[j]==1:
                            current_knob_value.append(knob_min[j])
                        else:
                            current_knob_value.append(knob_max[j])
                    db.change_knob(knob_names,current_knob_value,knob_type)
                    print("Change Knob: "+str(current_knob_value))
                    latency,_=db.run_job(1,[query])
                    print("Latency: "+str(latency))
                    for j in range(len(knob_names)):
                        if current_knob[j]==1:
                            rb_effect[j]-=latency
                        else:
                            rb_effect[j]+=latency
                    tmp=current_knob[0]
                    for j in range(len(current_knob)-1):
                        current_knob[j]=current_knob[j+1]
                    current_knob[len(current_knob)-1]=tmp
            
            max_effect=max(rb_effect)
            for i in range(len(knob_names)):
                rb_effect[i]=round(rb_effect[i]/max_effect,1)
            
            argsort=np.argsort(rb_effect)
            rb_effect_np=np.array(rb_effect)
            sorted_array=rb_effect_np[argsort]
            current_rank=1
            rank=[]
            not_recorded_start=0
            for i in range(1,len(knob_names)):
                if sorted_array[i]!=sorted_array[i-1]:
                    for j in range(not_recorded_start,i):
                        rank.append(current_rank)
                    not_recorded_start=i
                current_rank+=1
            for j in range(not_recorded_start,len(knob_names)):
                rank.append(current_rank)
            rank=np.array(rank)
            workload_ranking_record.append(rank[argsort])

        ranking_mean=[]
        for i in range(len(knob_names)):
            total_rank=0
            for ranking in workload_ranking_record:
                total_rank+=ranking[i]
            total_rank=total_rank/len(workload_ranking_record)
            ranking_mean.append(total_rank)
        
        self.argsort=np.argsort(ranking_mean)
        return self
    
    def rank_knob(self,knob_values):
        if self.argsort is None:
            raise Exception("Please fit the model first")
        
        if len(knob_values)!=len(self.knob_names):
            raise Exception("Input knob value should be consistent with knob names")
        
        return [self.knob_names[i] for i in self.argsort],[knob_values[i] for i in self.argsort]
    
    def get_top_rank(self):
        return [self.knob_names[i] for i in self.argsort]
