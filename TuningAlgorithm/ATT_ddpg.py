import os
import sys
import torch
import torch.nn as nn
from torch.optim import Adam
import numpy as np
import time
from collections import deque
from sklearn.preprocessing import StandardScaler
from config import *
from DBConnector.BaseExecutor import *
from FeatureSelection.FeatureSelector import *

global_logger=Logger("../log/training_log_sigmoidfc3.log")
yGrad=torch.zeros(1,1)

def extract_grad(xVar):
    global yGrad
    yGrad=xVar

class AttentionState(nn.Module):
    def __init__(self,workload_dim,state_dim,hidden_state_dim) -> None:
        super().__init__()
        self.state_dim=state_dim
        self.mlp_model_list=[nn.Sequential(nn.Linear(workload_dim+1,hidden_state_dim),nn.Tanh(),nn.Linear(hidden_state_dim,1),nn.Sigmoid()) for i in range(state_dim)]
        self.softmax=nn.Softmax()

    def forward(self,workload_character:list,state:list):
        # workload_character & state: 1-D array, one sample each time
        # data=[] 
        # for i in range(self.state_dim):
        #     data.append(np.append(workload_character,state[i]))
        # data=np.array(data)
        # data=torch.FloatTensor(data)
        # print(data)
        # alpha=[self.mlp_model_list[i](data[i]) for i in range(self.state_dim)]
        # print(alpha)
        # alpha=self.softmax(torch.Tensor(alpha))
        # print(alpha)
        # return alpha.mul(torch.Tensor(state))

        # workload_character & state: 2-D array: (num_samples, character dimension)
        # print(len(state))
        # print(state)
        data=[] #(state_dim,sample_num,workload_dim+1)
        for i in range(self.state_dim):
            sample_data=[]
            for j in range(len(state)):
                sample_data.append(np.append(workload_character[j],state[j][i]))
            data.append(sample_data)
        data=np.array(data)
        # global_logger.write("data:\n")
        # global_logger.write(str(data)+"\n")
        data=torch.FloatTensor(data)
        alpha=[self.mlp_model_list[i](data[i]) for i in range(self.state_dim)]
        # global_logger.write("Alpha:\n")
        # global_logger.write(str(alpha)+"\n")
        # print(len(alpha))
        # for sub_alpha in alpha:
        #     print(sub_alpha.shape)
        alpha=torch.cat(alpha,dim=0).reshape(len(workload_character),self.state_dim)
        # print(alpha.shape)
        alpha=self.softmax(alpha)
        # global_logger.write("Alpha after Softmax:\n")
        # global_logger.write(str(alpha)+"\n")
        # print(alpha.shape)
        result=alpha.mul(torch.Tensor(state))
        # global_logger.write("Attention Module Results:\n")
        # global_logger.write(str(result)+"\n")
        # print(result.shape)
        return result
        
class Actor(nn.Module):
    def __init__(self,observation_dim,action_dim,workload_dim) -> None:
        super().__init__()
        self.attention_model=AttentionState(workload_dim,observation_dim,32)
        # self.fc1=nn.Linear(observation_dim,128)
        self.fc1=nn.Linear(observation_dim,64)
        # self.bn1=nn.BatchNorm1d(128)
        # self.fc2=nn.Linear(128,64)
        self.dropout=nn.Dropout(0.3)
        self.fc3=nn.Linear(64,action_dim)


    def forward(self,observation,workload_character):
        observation=self.attention_model(workload_character,observation)
        a=self.fc1(observation)
        # global_logger.write("actor fc1 result:\n")
        # global_logger.write(str(a)+"\n")
        a=torch.relu(a)
        # global_logger.write("actor relu after fc1:\n")
        # global_logger.write(str(a)+"\n")
        # a=self.bn1(a)
        # global_logger.write("actor batch normalization:\n")
        # global_logger.write(str(a)+"\n")
        # a=self.fc2(a)
        # global_logger.write("actor fc2 results:\n")
        # global_logger.write(str(a)+"\n")
        # a=torch.relu(a)
        # global_logger.write("actor relu after fc2:\n")
        # global_logger.write(str(a)+"\n")
        a=self.dropout(a)
        global_logger.write("actor dropout:\n")
        global_logger.write(str(a)+"\n")
        a=self.fc3(a)
        global_logger.write("actor fc3 results:\n")
        global_logger.write(str(a)+"\n")
        a=torch.sigmoid(a)
        global_logger.write("actor final(sigmoid) results:\n")
        global_logger.write(str(a)+"\n")
        a.register_hook(extract_grad)
        return a

class Critic(nn.Module):
    def __init__(self,observation_dim,action_dim) -> None:
        super().__init__()
        # self.observation_normalize=nn.BatchNorm1d(observation_dim)
        self.observation_fc=nn.Linear(observation_dim,128)
        self.action_fc=nn.Linear(action_dim,128)
        self.merge_fc1=nn.Linear(128,256)
        self.bn1=nn.BatchNorm1d(256)
        self.merge_fc2=nn.Linear(256,64)
        self.dropout=nn.Dropout(0.3)
        self.bn2=nn.BatchNorm1d(64)
        self.output=nn.Linear(64,1)

    def forward(self,observation,action):
        # global_logger.write("Critic input observation:\n")
        # global_logger.write(str(observation)+"\n")
        # global_logger.write("Critic input action:\n")
        # global_logger.write(str(action)+"\n")
        # observation=self.observation_normalize(observation)
        # global_logger.write("Critic observation normalize:\n")
        # global_logger.write(str(observation)+"\n")
        observation_out=self.observation_fc(observation)
        # global_logger.write("Critic observation fc:\n")
        # global_logger.write(str(observation_out)+"\n")
        action_out=self.action_fc(action)
        # global_logger.write("Critic action fc:\n")
        # global_logger.write(str(action_out)+"\n")
        a=torch.add(observation_out,action_out)
        # global_logger.write("Critic add observation and action:\n")
        # global_logger.write(str(a)+"\n")
        a=self.merge_fc1(a)
        # global_logger.write("Critic merge fc1:\n")
        # global_logger.write(str(a)+"\n")
        a=torch.relu(a)
        # global_logger.write("Critic relu after merge fc1:\n")
        # global_logger.write(str(a)+"\n")
        # a=torch.tanh(a)
        a=self.bn1(a)
        # global_logger.write("Critic after batch normalization 1:\n")
        # global_logger.write(str(a)+"\n")
        a=self.merge_fc2(a)
        # global_logger.write("Critic merge fc2:\n")
        # global_logger.write(str(a)+"\n")
        a=torch.tanh(a)
        # global_logger.write("Critic after tanh:\n")
        # global_logger.write(str(a)+"\n")
        a=self.dropout(a)
        # global_logger.write("Critic after dropout:\n")
        # global_logger.write(str(a)+"\n")
        a=self.bn2(a)
        # global_logger.write("Critic after batch normalization 2:\n")
        # global_logger.write(str(a)+"\n")
        a=self.output(a)
        # global_logger.write("Critic last fc, final result:\n")
        # global_logger.write(str(a)+"\n")
        return a

class ActorCritic:
    def __init__(self,state_num,action_num,workload_dim,learning_rate=0.5, train_min_size=64, size_mem=2000, size_predict_mem=2000):

        self.learning_rate = learning_rate  # 0.001
        self.train_min_size = train_min_size

        self.gamma = .095
        self.tau = .125
        self.timestamp = int(time.time())

        self.mem_capacity=size_mem
        self.memory = deque(maxlen=size_mem)
        self.mem_predicted = deque(maxlen=size_predict_mem)

        self.actor_model=Actor(state_num,action_num,workload_dim)
        self.target_actor_model=Actor(state_num,action_num,workload_dim)
        self.actor_optimizer=Adam(self.actor_model.parameters(),lr=learning_rate)

        self.critic_model=Critic(state_num,action_num)
        self.target_critic_model=Critic(state_num,action_num)
        self.critic_optimizer=Adam(self.critic_model.parameters(),lr=learning_rate)

        self.mse_loss=nn.MSELoss()

        # if torch.cuda.is_available():
        #     self.actor_model.cuda()
        #     self.target_actor_model.cuda()
        #     self.critic_model.cuda()
        #     self.target_critic_model.cuda()

    def remember(self,cur_state, action, reward, new_state,workload_character):
        self.memory.append([cur_state, action, reward, new_state,workload_character])

    def sampling(self,batch_size):
        indices=np.random.choice(len(self.memory)-1,size=batch_size-1).tolist()
        tmp=np.array(list(self.memory))
        return tmp[indices]

    def set_train_mode(self):
        self.actor_model=self.actor_model.train()
        # self.actor_model.bn1.train()
        self.actor_model.dropout.train()
        self.target_actor_model=self.target_actor_model.train()
        # self.target_actor_model.bn1.train()
        self.target_actor_model.dropout.train()
        self.critic_model=self.critic_model.train()
        # self.critic_model.observation_normalize.train()
        self.critic_model.bn1.train()
        self.critic_model.bn2.train()
        self.critic_model.dropout.train()
        self.target_critic_model=self.target_critic_model.train()
        # self.target_critic_model.observation_normalize.train()
        self.target_critic_model.bn1.train()
        self.target_critic_model.bn2.train()
        self.target_critic_model.dropout.train()

    def set_eval_mode(self):
        self.actor_model=self.actor_model.eval()
        # self.actor_model.bn1.eval()
        self.actor_model.dropout.eval()
        self.target_actor_model=self.target_actor_model.eval()
        # self.target_actor_model.bn1.eval()
        self.target_actor_model.dropout.eval()
        self.critic_model=self.critic_model.eval()
        # self.critic_model.observation_normalize.eval()
        self.critic_model.bn1.eval()
        self.critic_model.bn2.eval()
        self.critic_model.dropout.eval()
        self.target_critic_model=self.target_critic_model.eval()
        # self.target_critic_model.observation_normalize.eval()
        self.target_critic_model.bn1.eval()
        self.target_critic_model.bn2.eval()
        self.target_critic_model.dropout.eval()

    def train(self):
        self.batch_size = self.train_min_size  # 32
        if len(self.memory) < self.batch_size:
            return
        self.set_train_mode()
        samples=self.sampling(self.batch_size)
        samples=np.append(samples,[self.memory[-1]],axis=0)
        
        a_layers=self.target_actor_model.named_children()
        c_layers=self.target_critic_model.named_children()
        for al in a_layers:
            try:
                al[1].weight.data.mul_((1-self.tau))
                al[1].weight.data.add_(self.tau * self.actor_model.state_dict()[al[0]+'.weight'])
            except:
                pass
            try:
                al[1].bias.data.mul_((1-self.tau))
                al[1].bias.data.add_(self.tau * self.actor_model.state_dict()[al[0]+'.bias'])
            except:
                pass
        
        for cl in c_layers:
            try:
                cl[1].weight.data.mul_((1-self.tau))
                cl[1].weight.data.add_(self.tau * self.critic_model.state_dict()[cl[0]+'.weight'])
            except:
                pass
            try:
                cl[1].bias.data.mul_((1-self.tau))
                cl[1].bias.data.add_(self.tau * self.critic_model.state_dict()[cl[0]+'.bias'])
            except:
                pass

        
        current_state_sample=torch.FloatTensor(np.array([s[0] for s in samples]))
        next_state_sample=torch.FloatTensor(np.array([s[3] for s in samples]))
        action_sample_input=torch.FloatTensor(np.array([s[1] for s in samples]))
        rewards=torch.FloatTensor(np.array([s[2] for s in samples]))
        workload_input=torch.FloatTensor(np.array([s[4] for s in samples]))
        
        # if torch.cuda.is_available():
        #     current_state_sample.cuda()
        #     next_state_sample.cuda()
        #     action_sample_input.cuda()
        #     rewards.cuda()

        ## Train actor
        global_logger.write("Train actor:\n")
        self.actor_optimizer.zero_grad()
        a=self.actor_model(current_state_sample,workload_input)
        q=self.critic_model(current_state_sample,a)
        a_loss=-torch.mean(q)
        a_loss.backward(retain_graph=True)
        for name, param in self.critic_model.named_parameters():
            global_logger.write("-->name:"+str(name)+"\n")
            global_logger.write("-->grad_requirs:"+str(param.requires_grad)+"\n")
            global_logger.write("-->grad_value:\n")
            global_logger.write(str(param.grad)+"\n")
            # print('-->name:', name, '-->grad_requirs:',param.requires_grad, ' -->grad_value:',param.grad)
        for name, param in self.actor_model.named_parameters():
            # print('-->name:', name, '-->grad_requirs:',param.requires_grad, ' -->grad_value:',param.grad)
            global_logger.write("-->name:"+str(name)+"\n")
            global_logger.write("-->grad_requirs:"+str(param.requires_grad)+"\n")
            global_logger.write("-->grad_value:\n")
            global_logger.write(str(param.grad)+"\n")
        global_logger.write("dL/dy:\n")
        global_logger.write(str(yGrad))
        global_logger.write("\n")
        self.actor_optimizer.step()

        ##Train critic
        global_logger.write("Train critic:\n")
        a_=self.target_actor_model(next_state_sample,workload_input)
        q_=self.target_critic_model(next_state_sample,a_)
        q_target=rewards+self.gamma*q_
        q_eval=self.critic_model(current_state_sample,action_sample_input)
        td_error=self.mse_loss(q_target,q_eval)
        self.critic_optimizer.zero_grad()
        td_error.backward()
        for name, param in self.critic_model.named_parameters():
            global_logger.write("-->name:"+str(name)+"\n")
            global_logger.write("-->grad_requirs:"+str(param.requires_grad)+"\n")
            global_logger.write("-->grad_value:\n")
            global_logger.write(str(param.grad)+"\n")
        self.critic_optimizer.step()
        global_logger.write("\n\n")

    def act(self, cur_state,workload_embedding):
        '''
        cur_state: predicted next state
        '''
        self.set_eval_mode()
        # cur_state=torch.FloatTensor([cur_state])
        cur_state=[cur_state]
        # if torch.cuda.is_available():
        #     cur_state.cuda()
        action = self.actor_model(cur_state,[workload_embedding]).cpu().detach().numpy()[0]

        return action

    def judge(self,cur_state,action)->float:
        self.set_eval_mode()
        cur_state=torch.FloatTensor([cur_state])
        action=torch.FloatTensor([action])
        # if torch.cuda.is_available():
        #     cur_state.cuda()
        #     action.cuda()
        reward=self.critic_model(cur_state,action).cpu().detach().numpy()[0][0]
        return reward

class ATT_DDPG():
    def __init__(self,db_connector:Executor,feature_selector:FeatureSelector,workload:list,workload_embed_dim=38,selected_knob_config=nonrestart_knob_config,latency_weight=9,throughput_weight=1,mu=0.5,rho=0.5,logger=None) -> None:
        self.db=db_connector
        self.workload=workload
        self.feature_selector=feature_selector
        self.latency_weight=latency_weight
        self.throughput_weight=throughput_weight
        self.mu=mu
        self.rho=rho
        if logger is not None:
            self.logger=logger
        else:
            self.logger=sys.stdout

        knobs=list(selected_knob_config.keys())
        knob_info=self.db.get_knob_min_max(knobs)
        self.knob_names,self.knob_min,self.knob_max,self.knob_granularity,self.knob_type=modifyKnobConfig(knob_info,selected_knob_config)
        self.logger.write(str(self.knob_names)+"\n")
        # self.knob_names=[]
        # self.knob_min=[]
        # self.knob_max=[]
        # self.knob_granularity=[]
        # self.knob_type=[]
        # for key in knob_info:
        #     self.knob_names.append(key)
        #     if "min" in selected_knob_config[key]:
        #         self.knob_min.append(selected_knob_config[key]["min"])
        #     else:
        #         self.knob_min.append(knob_info[key]['min'])
        #     if "max" in selected_knob_config[key]:
        #         self.knob_max.append(selected_knob_config[key]["max"])
        #     else:
        #         self.knob_max.append(knob_info[key]['max'])
        #     if "granularity" in selected_knob_config[key]:
        #         self.knob_granularity.append(selected_knob_config[key]["granularity"])
        #     else:
        #         self.knob_granularity.append(knob_info[key]['granularity'])
        #     self.knob_type.append(knob_info[key]['type'])
        # self.knob_min=np.array(self.knob_min)
        # self.knob_max=np.array(self.knob_max)
        # self.knob_granularity=np.array(self.knob_granularity)

        self.action_num=len(self.knob_names)
        db_state=self.db.get_db_state()
        if self.feature_selector is None:
            self.state_num=len(db_state)
        else:
            self.state_num=len(self.feature_selector.transform(db_state))
        self.model=ActorCritic(self.state_num,self.action_num,workload_dim=workload_embed_dim)
        self.state_scaler=StandardScaler()
    
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
        for i in range(len(knob_value)):
            if knob_value[i]<self.knob_min[i]:
                knob_value[i]=self.knob_min[i]
            elif knob_value[i]>self.knob_max[i]:
                knob_value[i]=self.knob_max[i]
        self.logger.write("Knob value: "+str(knob_value)+"\n")
        self.db.change_restart_knob(self.knob_names,knob_value,self.knob_type)
        self.db.restart_db(remote_config["port"],remote_config["user"],remote_config["password"])

    def train(self,workload_embedding,total_epoch,epoch_steps,save_epoch_interval,save_folder,init_random_step=10):
        self.db.reset_restart_knob(self.knob_names)
        self.db.restart_db(remote_config["port"],remote_config["user"],remote_config["password"])
        self.init_latency,self.init_throughput,current_state=self.run_workload()
        self.last_latency=self.init_latency
        self.last_throughput=self.init_throughput
        self.logger.write("Init Data Collecting:\n")
        scaler_training_data=[]
        scaler_training_data.append(current_state)
        random_record=[]
        for step in range(init_random_step):
            self.logger.write("Current state: "+str(current_state)+"\n")
            action=np.random.random(size=self.action_num)
            self.take_action(action)
            latency,throughput,new_state=self.run_workload()
            self.logger.write("Latency: "+str(round(latency,4))+"\nThroughput: "+str(round(throughput,4))+"\n")
            reward=self.calculate_reward(latency,throughput)
            # self.model.remember(current_state,action,reward,new_state,workload_embedding)
            scaler_training_data.append(new_state)
            random_record.append([current_state,action,reward,new_state])
            current_state=new_state
            self.logger.write("\n")
        self.state_scaler.fit(scaler_training_data)
        for s in random_record:
            self.model.remember(self.state_scaler.transform(np.array(s[0]).reshape(1,-1))[0],s[1],s[2],self.state_scaler.transform(np.array(s[3]).reshape(1,-1))[0],workload_embedding)
        for epoch in range(total_epoch): 
            self.logger.write("Epoch: "+str(epoch)+"\n")
            self.db.reset_restart_knob(self.knob_names)
            self.db.restart_db(remote_config["port"],remote_config["user"],remote_config["password"])
            self.init_latency,self.init_throughput,current_state=self.run_workload()
            self.last_latency=self.init_latency
            self.last_throughput=self.init_throughput
            current_state=self.state_scaler.transform(np.array(current_state).reshape(1,-1))[0]

            for step in range(epoch_steps):
                global_logger.write("Step: "+str(step)+"\n")
                self.logger.write("Current state: "+str(current_state)+"\n")
                action=self.model.act(current_state,workload_embedding)
                self.take_action(action)
                latency,throughput,new_state=self.run_workload()
                new_state=self.state_scaler.transform(np.array(new_state).reshape(1,-1))[0]
                self.logger.write("Latency: "+str(round(latency,4))+"\nThroughput: "+str(round(throughput,4))+"\n")
                reward=self.calculate_reward(latency,throughput)
                self.model.remember(current_state,action,reward,new_state,workload_embedding)
                self.model.train()
                current_state=new_state
                self.logger.write("\n")
            
            if (epoch+1)%save_epoch_interval==0:
                torch.save(self.model.actor_model.state_dict(),os.path.join(save_folder,"actor_epoch_{}.pkl".format(epoch)))
                torch.save(self.model.critic_model.state_dict(),os.path.join(save_folder,"critic_epoch_{}.pkl".format(epoch)))
    
    def calculate_reward(self,latency,throughput):
        dl_0=(self.init_latency-latency)/self.init_latency
        dt_0=(throughput-self.init_throughput)/self.init_throughput
        dl_last=(self.last_latency-latency)/self.last_latency
        dt_last=(throughput-self.last_throughput)/self.last_throughput
        r_l=0
        if dl_0>0 and dl_last>0:
            r_l=((1+dl_0)**2-1)*abs(1+dl_last)
        elif dl_0>0 and dl_last>-self.rho:
            r_l=self.mu*((1+dl_0)**2-1)*abs(1+dl_last)
        elif dl_0>0:
            r_l=0
        else:
            r_l=-(((1-dl_0)**2-1)*abs(1+dl_last))
        
        r_t=0
        if dt_0>0 and dt_last>0:
            r_t=((1+dt_0)**2-1)*abs(1+dt_last)
        elif dt_0>0 and dt_last>-self.rho:
            r_t=self.mu*((1+dt_0)**2-1)*abs(1+dt_last)
        elif dt_0>0:
            r_t=0
        else:
            r_t=-(((1-dl_0)**2-1)*abs(1-dl_last))
        
        reward=(self.throughput_weight*r_t+self.latency_weight*r_l)/(self.throughput_weight+self.latency_weight)
        self.last_latency=latency
        self.last_throughput=throughput
        return reward
