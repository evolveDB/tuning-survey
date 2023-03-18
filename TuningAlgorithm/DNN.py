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
            selected_knob_num=10) -> None:
        self.train_data_size = train_data_size
        self.feature_num = feature_num
        self.selected_knob_num = selected_knob_num
        self.dnn_model = None

    def train(
            self,
            db: Executor,
            workload: list,
            train_epoch=100,
            lr=0.01,
            batch_size=32,
            save_interval=20,
            save_path="../TuningAlgorithm/model/dnn/",
            knob_config=nonrestart_knob_config):
        knob_names = list(knob_config.keys())
        knob_info = db.get_knob_min_max(knob_names)
        knob_names, knob_min, knob_max, knob_granularity, knob_type = modifyKnobConfig(
            knob_info, knob_config)
        self.feature_selector = FeatureSelector_FA_Kmeans()
        self.knob_selector = Lasso(knobs=knob_names)

        knob_data = []
        scalered_knob_data = []
        metric_data = []
        latency_data = []

        for i in range(self.train_data_size):
            print(i)
            action = np.random.random(size=len(knob_info))
            scalered_knob_data.append(action)
            knob_value = np.round(
                (knob_max - knob_min) / knob_granularity * action) * knob_granularity + knob_min
            # print(knob_value)
            db.change_restart_knob(knob_names, knob_value, knob_type)
            db.restart_db(
                remote_config["port"],
                remote_config["user"],
                remote_config["password"])
            thread_num = db.get_max_thread_num()
            before_state = db.get_db_state()
            l, t = db.run_job(thread_num, workload)
            # print("Latency: "+str(l))
            # print("Throughput: "+str(t))
            time.sleep(0.1)
            after_state = db.get_db_state()
            state = np.array(after_state) - np.array(before_state)
            knob_data.append(knob_value)
            metric_data.append(state)
            latency_data.append([l])
            # print(state)
            # print()

        print("Finish collecting data")
        db.reset_restart_knob(knob_names)
        self.feature_selector.fit(metric_data, kmeans_k=self.feature_num)
        new_metric_data = []
        for metric in metric_data:
            new_metric_data.append(self.feature_selector.transform(metric))
        self.knob_selector.fit(knob_data, new_metric_data)
        ranked_knob_names = None
        new_knob_value = []
        for knob_value in scalered_knob_data:
            ranked_knob_names, new_knob = self.knob_selector.rank_knob(
                knob_value)
            new_knob_value.append(new_knob[:self.selected_knob_num])
        ranked_knob_names = ranked_knob_names[:self.selected_knob_num]
        _, knob_min = self.knob_selector.rank_knob(knob_min)
        _, knob_max = self.knob_selector.rank_knob(knob_max)
        _, knob_granularity = self.knob_selector.rank_knob(knob_granularity)
        _, knob_type = self.knob_selector.rank_knob(knob_type)

        # self.latency_scaler=StandardScaler()
        # latency_data=self.latency_scaler.fit_transform(latency_data)

        self.dnn_model = Net(len(new_knob_value[0]), 1)
        optimizer = Adam(self.dnn_model.parameters(), lr)
        loss_function = nn.MSELoss()
        dataset = TrainDataSet(new_knob_value, latency_data)
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
            print("Epoch " + str(epoch) + ", Loss: " + str(epoch_loss))

        print("Finish Training DNN")

        self.knob_names = ranked_knob_names
        self.knob_min = knob_min[:self.selected_knob_num]
        self.knob_max = knob_max[:self.selected_knob_num]
        self.knob_granularity = knob_granularity[:self.selected_knob_num]
        self.knob_type = knob_type[:self.selected_knob_num]
        self.scalered_knob_data = scalered_knob_data
        self.knob_data = knob_data
        self.metric_data = new_metric_data
        self.latency_data = latency_data

    def recommand(
            self,
            x_start=None,
            recommand_epoch=100,
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
        for epoch in range(recommand_epoch):
            x_optimizer.zero_grad()
            y_pred = self.dnn_model(x_start)
            print([x_start, y_pred])
            results.append([x_start.clone(), y_pred])
            # target_y=torch.zeros(y_pred.shape)
            # loss=loss_function(y_pred,target_y)
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
            workload: list,
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
        self.feature_selector = FeatureSelector_FA_Kmeans()

        for i in range(self.train_data_size):
            # print(i)
            action = np.random.random(size=len(knob_names))
            scalered_knob_data.append(action)
            knob_value = np.round(
                (knob_max - knob_min) / knob_granularity * action) * knob_granularity + knob_min
            # print(knob_value)
            db.change_restart_knob(knob_names, knob_value, knob_type)
            db.restart_db(
                remote_config["port"],
                remote_config["user"],
                remote_config["password"])
            thread_num = db.get_max_thread_num()
            before_state = db.get_db_state()
            l, t = db.run_job(thread_num, workload)
            # print("Latency: "+str(l))
            # print("Throughput: "+str(t))
            time.sleep(0.1)
            after_state = db.get_db_state()
            state = np.array(after_state) - np.array(before_state)
            knob_data.append(knob_value)
            metric_data.append(state)
            latency_data.append([l])
            # print(state)
            # print()

        print("Finish collecting data")
        db.reset_restart_knob(knob_names)
        self.feature_selector.fit(metric_data, kmeans_k=self.feature_num)
        new_metric_data = []
        for metric in metric_data:
            new_metric_data.append(self.feature_selector.transform(metric))

        # self.latency_scaler=StandardScaler()
        # latency_data=self.latency_scaler.fit_transform(latency_data)

        self.dnn_model = Net(len(scalered_knob_data[0]), 1)
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
            print("Epoch " + str(epoch) + ", Loss: " + str(epoch_loss))

        print("Finish Training DNN")

        self.knob_names = knob_names
        self.knob_min = knob_min[:self.selected_knob_num]
        self.knob_max = knob_max[:self.selected_knob_num]
        self.knob_granularity = knob_granularity[:self.selected_knob_num]
        self.knob_type = knob_type[:self.selected_knob_num]
        self.scalered_knob_data = scalered_knob_data
        self.knob_data = knob_data
        self.metric_data = new_metric_data
        self.latency_data = latency_data
