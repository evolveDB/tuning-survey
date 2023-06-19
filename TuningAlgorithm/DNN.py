import sys
sys.path.append("../")
import time
from torch.autograd import Variable
from torch.utils.data import Dataset, DataLoader
from torch.optim import Adam, SGD
import torch.nn.functional as F
import torch.nn as nn
import torch
from KnobSelection.lasso import Lasso
from FeatureSelection.FA_Kmeans import FeatureSelector_FA_Kmeans
from config import *
from DBConnector.BaseExecutor import Executor
import pickle
from random import sample as rsample


class Net(nn.Module):
    def __init__(self, input_dim, output_dim) -> None:
        super().__init__()
        self.fc1 = nn.Linear(input_dim, 64)
        self.fc2 = nn.Linear(64, 64)
        self.dropout = nn.Dropout(0.5)
        self.fc3 = nn.Linear(64, output_dim)

    def forward(self, feature):
        a = self.fc1(feature)
        a = F.relu(a)
        a = self.dropout(a)
        a = self.fc2(a)
        a = F.relu(a)
        a = self.fc3(a)
        a = F.relu(a)
        return a


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


class DNN_Tune():
    def __init__(
            self,
            train_data_size=100,
            feature_num=10,
            selected_knob_num=10,
            logger=None) -> None:
        self.train_data_size = train_data_size
        self.feature_num = feature_num
        self.selected_knob_num = selected_knob_num
        self.dnn_model = None
        if logger is not None:
            self.logger = logger
        else:
            self.logger = sys.stdout

    def recommend(
            self,
            x_start=None,
            recommend_epoch=100,
            lr=0.05,
            explore=False):
        if x_start is None:
            x_start = torch.ones(size=(1, len(self.knob_names)))
            for i in range(len(x_start[0])):
                x_start[0][i] = x_start[0][i] / 2
        x_start = Variable(x_start, requires_grad=True)
        x_start = nn.Parameter(x_start, requires_grad=True)
        x_optimizer = SGD([x_start], lr)
        loss_function = nn.MSELoss()
        if explore:
            with torch.no_grad():
                for param in self.dnn_model.parameters():
                    param.add_(torch.randn(param.shape) * 0.1)
        results = []
        for epoch in range(recommend_epoch):
            x_optimizer.zero_grad()
            y_pred = self.dnn_model(x_start)
            print([x_start, y_pred])
            results.append([x_start.clone(), y_pred])
            y_pred.backward(retain_graph=True)
            x_optimizer.step()

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

    def train(
            self,
            db: Executor,
            knob_names,
            knob_min,
            knob_max,
            knob_granularity,
            knob_type,
            train_epoch=100,
            lr=0.01,
            batch_size=32):
        knob_data = []
        scalered_knob_data = []
        metric_data = []
        latency_data = []
        sample_dataset={}
        sample_latency=[]
        sample_throughput=[]
  
        for i in range(self.train_data_size):
            action = np.random.random(size=len(knob_names))
            knob_value = np.round(
                (knob_max - knob_min) / knob_granularity * action) * knob_granularity + knob_min
            scalered_knob_data.append(action)
            
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
            sample_latency.append(l)
            sample_throughput.append(t)
            self.logger.write("Performance: " + str(l) + "\n\n")
            latency_data.append([l])

        sample_dataset["scalered_knob_data"]=scalered_knob_data
        sample_dataset["latency_label"]=sample_latency
        sample_dataset["throughput_label"]=sample_throughput
        f=open("../data/mydds_sample_dataset_tpcc_dnn_0529.bin","wb")
        pickle.dump(sample_dataset,f)
        f.close()

        # f=open("../data/sample_dataset_sysro.bin","rb")
        # saved_dataset=pickle.load(f)
        # f.close()
        # scalered_knob_data = saved_dataset["scalered_knob_data"]
        # latency_data = [ [l] for l in saved_dataset["latency_label"]]

        self.logger.write("Finish collecting data\n\n\n")
        print("training process start!")
        new_metric_data = []

        optimizer = Adam(self.dnn_model.parameters(), lr)
        loss_function = nn.MSELoss()
        dataset = TrainDataSet(scalered_knob_data, latency_data)
        dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

        for epoch in range(train_epoch):
            epoch_loss = 0
            for batch_idx, data in enumerate(dataloader):
                optimizer.zero_grad()
                x, y = data
                y_pred = self.dnn_model(x)
                loss = loss_function(y, y_pred)
                loss.backward()
                epoch_loss += loss.item()
                optimizer.step()
            self.logger.write("Epoch " + str(epoch) + ", Loss: " + str(epoch_loss)+"\n")

        self.logger.write("Finish Training DNN\n\n\n")

        self.knob_names = knob_names
        self.knob_min = knob_min
        self.knob_max = knob_max
        self.knob_granularity = knob_granularity
        self.knob_type = knob_type
        self.scalered_knob_data = scalered_knob_data
        self.knob_data = knob_data
        self.metric_data = new_metric_data
        self.latency_data = latency_data
