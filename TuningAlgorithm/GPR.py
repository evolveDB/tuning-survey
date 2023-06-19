import sys
sys.path.append("../")
import time
import gpytorch
from torch.autograd import Variable
from torch.utils.data import Dataset
from torch.optim import SGD
import torch.nn as nn
import torch
from config import *
from DBConnector.BaseExecutor import Executor
import pickle


class TrainDataSet(Dataset):
    def __init__(self, x, y) -> None:
        super().__init__()
        self.x = x
        self.y = y

    def __len__(self):
        return len(self.y)

    def __getitem__(self, index):
        return torch.FloatTensor(
            self.x[index]), torch.FloatTensor(
            self.y[index])


class ExactGPModel(gpytorch.models.ExactGP):
    def __init__(self, train_inputs, train_targets, likelihood):
        super(
            ExactGPModel,
            self).__init__(
            train_inputs,
            train_targets,
            likelihood)
        self.mean_module = gpytorch.means.ConstantMean()
        self.covar_module = gpytorch.kernels.ScaleKernel(
            gpytorch.kernels.RBFKernel())

    def forward(self, x):
        mean_x = self.mean_module(x)
        covar_x = self.covar_module(x)
        return gpytorch.distributions.MultivariateNormal(mean_x, covar_x)


class GPR():
    def __init__(self, feature_num=10, knob_num=5, logger=None):
        self.feature_num = feature_num
        self.knob_num = knob_num
        self.likelihood = gpytorch.likelihoods.GaussianLikelihood()
        self.model = None
        if logger is not None:
            self.logger = logger
        else:
            self.logger = sys.stdout

    def train(
            self,
            db: Executor,
            train_epoch=100,
            lr=0.01,
            batch_size=32,
            save_interval=20,
            save_path="../TuningAlgorithm/model/gpr/",
            knob_config=knob_config,
            train_data_size=100):
        self.train_data_size = train_data_size
        knob_names = list(knob_config.keys())
        knob_info = db.get_knob_min_max(knob_names)
        knob_names, knob_min, knob_max, knob_granularity, knob_type,_ = modifyKnobConfig(
            knob_info, knob_config)

        knob_data = []
        scalered_knob_data = []
        metric_data = []
        latency_data = []
        sample_dataset={}
        sample_latency=[]
        sample_throughput=[]

        for i in range(self.train_data_size):
            action = np.random.random(size=len(knob_info))
            scalered_knob_data.append(action)
            knob_value = np.round(
                (knob_max - knob_min) / knob_granularity * action) * knob_granularity + knob_min
            db.change_knob(knob_names, knob_value.astype(int))
            self.logger.write("Change Knob: " + str(knob_value) + "\n")
            before_state = db.get_db_state()
            l, t = db.run_tpcc()
            self.logger.write("Latency: " + str(l) + "\n")
            self.logger.write("Throughput: " + str(t) + "\n")
            time.sleep(0.1)
            after_state = db.get_db_state()
            state = np.array(after_state) - np.array(before_state)
            knob_data.append(knob_value)
            metric_data.append(state)
            latency_data.append(l)
            sample_latency.append(l)
            sample_throughput.append(t)
            self.logger.write("Performance: " + str(l) + "\n\n")
            latency_data.append(l)
            
        sample_dataset["scalered_knob_data"]=scalered_knob_data
        sample_dataset["latency_label"]=sample_latency
        sample_dataset["throughput_label"]=sample_throughput
        f=open("../data/sample_dataset_tpch.bin","wb")
        pickle.dump(sample_dataset,f)
        f.close()

        # f=open("../data/sample_dataset_tpcc_dnn_0528.bin",'rb')
        # saved_dataset=pickle.load(f)
        # f.close()
        # scalered_knob_data = saved_dataset["scalered_knob_data"]
        # latency_data = [ l for l in saved_dataset["latency_label"]]
        
        self.logger.write("Finish collecting data\n\n\n")
        scalered_knob_data = torch.FloatTensor(scalered_knob_data)
        latency_data = torch.FloatTensor(latency_data)
        
        self.model.train()
        self.likelihood.train()
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=lr)
        mll = gpytorch.mlls.ExactMarginalLogLikelihood(
            self.likelihood, self.model)


        for epoch in range(train_epoch):
            epoch_loss = 0
            self.optimizer.zero_grad()
            y_pred = self.model(scalered_knob_data)
            loss = -mll(y_pred, latency_data)
            loss = loss.mean()
            loss.backward()
            epoch_loss = loss.item()
            self.optimizer.step()
            self.logger.write("Epoch " + str(epoch) + ", Loss: " + str(epoch_loss) + "\n")

        self.logger.write("Finish Training GPR\n\n\n")
        self.knob_names = knob_names
        self.knob_min = knob_min
        self.knob_max = knob_max
        self.knob_granularity = knob_granularity
        self.knob_type = knob_type
        self.scalered_knob_data = scalered_knob_data
        self.knob_data = knob_data
        self.metric_data = metric_data
        self.latency_data = latency_data

    def recommend(self, x_start, recommand_epoch=100, lr=0.05, explore=False):
        if x_start is None:
            x_start = torch.ones(size=(1, len(self.knob_names)))
            for i in range(len(x_start[0])):
                x_start[0][i] = x_start[0][i] / 2
        x_start = Variable(x_start, requires_grad=True)
        x_start = nn.Parameter(x_start, requires_grad=True)
        x_optimizer = SGD([x_start], lr)
        tmp_model = ExactGPModel(
            self.scalered_knob_data,
            self.latency_data,
            self.likelihood)
        tmp_model.load_state_dict(self.model.state_dict())
        if explore:
            with torch.no_grad():
                for param in tmp_model.parameters():
                    param.add_(torch.randn(param.shape) * 0.1)
        results = []
        self.likelihood.eval()
        tmp_model.eval()
        for epoch in range(recommand_epoch):
            x_optimizer.zero_grad()
            y_pred = tmp_model(x_start)
            print([x_start, y_pred.loc])
            # target_y=torch.zeros(y_pred.shape)
            # loss=loss_function(y_pred,target_y)
            y_pred.loc.backward(retain_graph=True)
            print(x_start.grad)
            x_optimizer.step()
        results.append([x_start.clone(), y_pred.loc])

        for i in range(len(results)):
            results[i][0] = results[i][0][0].detach().numpy()
            results[i][1] = results[i][1].detach().numpy()
            res_knob_value = []
            for j in range(len(results[i][0])):
                tmp = np.round((self.knob_max[j] - self.knob_min[j]) / self.knob_granularity[j]
                               * results[i][0][j]) * self.knob_granularity[j] + self.knob_min[j]
                res_knob_value.append(
                    max(self.knob_min[j], min(self.knob_max[j], tmp)))
            results[i].append(res_knob_value)

        return results
    
