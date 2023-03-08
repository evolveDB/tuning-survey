import sys
sys.path.append("../")
from DBConnector.BaseExecutor import Executor
from config import *
import torch
import torch.nn as nn
from torch.optim import SGD
from torch.utils.data import Dataset,DataLoader
from torch.autograd import Variable,grad
import gpytorch
import time

class TrainDataSet(Dataset):
    def __init__(self,x,y) -> None:
        super().__init__()
        self.x=x
        self.y=y
    
    def __len__(self):
        return len(self.y)
    
    def __getitem__(self, index):
        return torch.FloatTensor(self.x[index]),torch.FloatTensor(self.y[index])

class ExactGPModel(gpytorch.models.ExactGP):
    def __init__(self, train_inputs, train_targets, likelihood):
        super(ExactGPModel, self).__init__(train_inputs, train_targets, likelihood)
        self.mean_module = gpytorch.means.ConstantMean()
        self.covar_module = gpytorch.kernels.ScaleKernel(gpytorch.kernels.RBFKernel())
    
    def forward(self,x):
        mean_x = self.mean_module(x)
        covar_x = self.covar_module(x)
        return gpytorch.distributions.MultivariateNormal(mean_x, covar_x)

class GPR():
    def __init__(self,feature_num=10,knob_num=5):
        self.feature_num=feature_num
        self.knob_num=knob_num
        self.likelihood=gpytorch.likelihoods.GaussianLikelihood()
        self.model=None
    
    def train(self,db:Executor,workload:list,train_epoch=100,lr=0.01,batch_size=32,save_interval=20,save_path="../TuningAlgorithm/model/gpr/",knob_config=nonrestart_knob_config,train_data_size=100):
        self.train_data_size=train_data_size
        knob_names=list(knob_config.keys())
        knob_info=db.get_knob_min_max(knob_names)
        knob_names,knob_min,knob_max,knob_granularity,knob_type=modifyKnobConfig(knob_info,knob_config)
        
        knob_data=[]
        scalered_knob_data=[]
        metric_data=[]
        latency_data=[]
        
        for i in range(self.train_data_size):
            print(i)
            action=np.random.random(size=len(knob_info))
            scalered_knob_data.append(action)
            knob_value=np.round((knob_max-knob_min)/knob_granularity*action)*knob_granularity+knob_min
            print(knob_value)
            db.change_restart_knob(knob_names,knob_value,knob_type)
            db.restart_db(remote_config["port"],remote_config["user"],remote_config["password"])
            thread_num=db.get_max_thread_num()
            before_state=db.get_db_state()
            l,t=db.run_job(thread_num,workload)
            print("Latency: "+str(l))
            print("Throughput: "+str(t))
            time.sleep(0.1)
            after_state=db.get_db_state()
            state=np.array(after_state)-np.array(before_state)
            knob_data.append(knob_value)
            metric_data.append(state)
            latency_data.append(l)
            # print(state)
            # print()
        scalered_knob_data=torch.FloatTensor(scalered_knob_data)
        latency_data=torch.FloatTensor(latency_data)
        self.model=ExactGPModel(scalered_knob_data, latency_data, self.likelihood)
        self.model.train()
        self.likelihood.train()
        self.optimizer=torch.optim.Adam(self.model.parameters(),lr=lr)
        mll = gpytorch.mlls.ExactMarginalLogLikelihood(self.likelihood, self.model)
        # dataset=TrainDataSet(scalered_knob_data,latency_data)
        # dataloader=DataLoader(dataset,batch_size=batch_size,shuffle=True)
        
        for epoch in range(train_epoch):
            epoch_loss=0
            self.optimizer.zero_grad()
            y_pred=self.model(scalered_knob_data)
            loss=-mll(y_pred,latency_data)
            loss=loss.mean()
            loss.backward()
            epoch_loss=loss.item()
            self.optimizer.step()
            # for batch_idx,data in enumerate(dataloader):
            #     self.optimizer.zero_grad()
            #     x,y=data
            #     y_pred=self.model(x)
            #     loss=-mll(y_pred,y)
            #     loss=loss.mean()
            #     loss.backward()
            #     epoch_loss+=loss.item()
            #     self.optimizer.step()
            print("Epoch "+str(epoch)+", Loss: "+str(epoch_loss))
            
        print("Finish Training DNN")
        self.knob_names=knob_names
        self.knob_min=knob_min
        self.knob_max=knob_max
        self.knob_granularity=knob_granularity
        self.knob_type=knob_type
        self.scalered_knob_data=scalered_knob_data
        self.knob_data=knob_data
        self.metric_data=metric_data
        self.latency_data=latency_data
    
    def recommand(self,x_start,recommand_epoch=100,lr=0.05,explore=False):
        if x_start is None:
            x_start=torch.ones(size=(1,len(self.knob_names)))
            for i in range(len(x_start[0])):
                x_start[0][i]=x_start[0][i]/2
        x_start=Variable(x_start,requires_grad=True)
        x_start=nn.Parameter(x_start,requires_grad=True)
        x_optimizer=SGD([x_start],lr)
        tmp_model=ExactGPModel(self.scalered_knob_data, self.latency_data, self.likelihood)
        tmp_model.load_state_dict(self.model.state_dict())
        if explore:
            with torch.no_grad():
                for param in tmp_model.parameters():
                    param.add_(torch.randn(param.shape)*0.1)
        results=[]
        self.likelihood.eval()
        tmp_model.eval()
        for epoch in range(recommand_epoch):
            x_optimizer.zero_grad()
            y_pred=tmp_model(x_start)
            print([x_start,y_pred.loc])
            # target_y=torch.zeros(y_pred.shape)
            # loss=loss_function(y_pred,target_y)
            y_pred.loc.backward(retain_graph=True)
            print(x_start.grad)
            x_optimizer.step()
        results.append([x_start.clone(),y_pred.loc])
        
        for i in range(len(results)):
            results[i][0]=results[i][0][0].detach().numpy()
            results[i][1]=results[i][1].detach().numpy()
            res_knob_value=[]
            for j in range(len(results[i][0])):
                tmp=np.round((self.knob_max[j]-self.knob_min[j])/self.knob_granularity[j]*results[i][0][j])*self.knob_granularity[j]+self.knob_min[j]
                res_knob_value.append(max(self.knob_min[j],min(self.knob_max[j],tmp)))
            results[i].append(res_knob_value)
        
        return results