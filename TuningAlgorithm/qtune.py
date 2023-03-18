import os
import sys
sys.path.append("../")
import torch
import torch.nn as nn
from torch.optim import Adam
import numpy as np
import time
from collections import deque
from config import *
from DBConnector.BaseExecutor import *
from FeatureSelection.QueryEmbed import *

from .ddpg import ActorCritic,Actor,Critic

class QTune():
    def __init__(self,db_connector:Executor,workload:list,selected_knob_config=non_restart_knob_config,latency_weight=9,throughput_weight=1,logger=None) -> None:
        self.db=db_connector
        self.workload=workload
        self.latency_weight=latency_weight
        self.throughput_weight=throughput_weight
        if logger is not None:
            self.logger=logger
        else:
            self.logger=sys.stdout

        knobs=selected_knob_config.keys()
        knob_info=self.db.get_knob_min_max(knobs)
        self.knob_names=[]
        self.knob_min=[]
        self.knob_max=[]
        self.knob_granularity=[]
        self.knob_type=[]
        for key in knob_info:
            self.knob_names.append(key)
            if "min" in selected_knob_config[key]:
                self.knob_min.append(selected_knob_config[key]["min"])
            else:
                self.knob_min.append(knob_info[key]['min'])
            if "max" in selected_knob_config[key]:
                self.knob_max.append(selected_knob_config[key]["max"])
            else:
                self.knob_max.append(knob_info[key]['max'])
            if "granularity" in selected_knob_config[key]:
                self.knob_granularity.append(selected_knob_config[key]["granularity"])
            else:
                self.knob_granularity.append(knob_info[key]['granularity'])
            self.knob_type.append(knob_info[key]['type'])
        self.knob_min=np.array(self.knob_min)
        self.knob_max=np.array(self.knob_max)
        self.knob_granularity=np.array(self.knob_granularity)

        self.action_num=len(self.knob_names)
        db_state=self.db.get_db_state()
        if self.feature_selector is None:
            self.state_num=len(db_state)
        else:
            self.state_num=len(self.feature_selector.transform(db_state))
        self.model=ActorCritic(self.state_num,self.action_num)
    
    def run_workload(self):
        self.db.restart_db(remote_config["port"],remote_config["user"],remote_config["password"])
        thread_num=self.db.get_max_thread_num()
        db_state=self.db.get_db_state()
        latency,throughput=self.db.run_job(thread_num,self.workload)
        time.sleep(0.2)
        after_db_state=self.db.get_db_state()
        new_state=np.array(after_db_state)-np.array(db_state)
        if self.feature_selector is not None:
            new_state=self.feature_selector.transform(new_state)
        return latency,throughput,new_state
    
    def take_action(self,action):
        knob_value=np.round((self.knob_max-self.knob_min)/self.knob_granularity*action)*self.knob_granularity+self.knob_min
        self.logger.write("Knob value: "+str(knob_value)+"\n")
        self.db.change_restart_knob(self.knob_names,knob_value,self.knob_type)
        self.db.restart_db(remote_config["port"],remote_config["user"],remote_config["password"])

    def train(self,total_epoch,epoch_steps,save_epoch_interval,save_folder):
        for epoch in range(total_epoch): 
            self.logger.write("Epoch: "+str(epoch)+"\n")
            self.db.reset_restart_knob(self.knob_names)
            self.db.restart_db(remote_config["port"],remote_config["user"],remote_config["password"])
            self.init_latency,self.init_throughput,current_state=self.run_workload()
            self.last_latency=self.init_latency
            self.last_throughput=self.init_throughput

            for step in range(epoch_steps):
                self.logger.write("Current state: "+str(current_state)+"\n")
                action=self.model.act(current_state)
                self.take_action(action)
                latency,throughput,new_state=self.run_workload()
                self.logger.write("Latency: "+str(round(latency,4))+"\nThroughput: "+str(round(throughput,4))+"\n")
                reward=self.calculate_reward(latency,throughput)
                self.model.remember(current_state,action,reward,new_state)
                self.model.train()
                current_state=new_state
                self.logger.write("\n")
            
            if ((epoch+1)%save_epoch_interval)==0:
                torch.save(self.model.actor_model.state_dict(),os.path.join(save_folder,"actor_epoch_{}.pkl".format(epoch)))
                torch.save(self.model.critic_model.state_dict(),os.path.join(save_folder,"critic_epoch_{}.pkl".format(epoch)))
    
    def calculate_reward(self,latency,throughput):
        dl_0=(self.init_latency-latency)/self.init_latency
        dt_0=(throughput-self.init_throughput)/self.init_throughput
        dl_last=(self.last_latency-latency)/self.last_latency
        dt_last=(throughput-self.last_throughput)/self.last_throughput
        r_l=0
        if dl_0>0:
            r_l=((1+dl_last)**2-1)*abs(1+dl_0)
        else:
            r_l=-(((1-dl_last)**2-1)*abs(1-dl_0))
        
        r_t=0
        if dt_0>0:
            r_t=((1+dt_last)**2-1)*abs(1+dt_0)
        else:
            r_t=-(((1-dt_last)**2-1)*abs(1-dt_0))
        
        reward=(self.throughput_weight*r_t+self.latency_weight*r_l)/(self.throughput_weight+self.latency_weight)
        self.last_latency=latency
        self.last_throughput=throughput
        return reward
