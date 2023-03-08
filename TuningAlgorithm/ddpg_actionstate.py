import os
import sys
import torch
import torch.nn as nn
from torch.optim import Adam
import time
from collections import deque
from sklearn.preprocessing import StandardScaler
from config import *
from DBConnector.BaseExecutor import *
from FeatureSelection.FeatureSelector import *


class Actor(nn.Module):
    def __init__(self, observation_dim, action_dim) -> None:
        super().__init__()
        self.fc1 = nn.Linear(observation_dim, 32)
        self.action_fc = nn.Linear(action_dim, 32)
        self.bn1 = nn.BatchNorm1d(32)
        self.fc2 = nn.Linear(32, 32)
        self.bn2 = nn.BatchNorm1d(32)
        self.dropout = nn.Dropout(0.15)
        self.fc3 = nn.Linear(32, action_dim)

    def forward(self, observation, action):
        a = torch.relu(self.fc1(observation)) * 0.5
        a = self.bn1(a)
        action_out = torch.relu(self.action_fc(action)) * 0.5
        b = torch.add(a, action_out)
        b = self.fc2(b) * 0.3
        b = self.bn2(b)
        b = torch.tanh(b)
        b = self.dropout(b)
        b = self.fc3(b) * 0.1
        return torch.tanh(b)


class Critic(nn.Module):
    def __init__(self, observation_dim, action_dim) -> None:
        super().__init__()
        self.observation_fc = nn.Linear(observation_dim, 32)
        self.action_fc = nn.Linear(action_dim, 32)
        self.merge_fc1 = nn.Linear(32, 32)
        self.bn1 = nn.BatchNorm1d(32)
        self.merge_fc2 = nn.Linear(32, 32)
        self.dropout = nn.Dropout(0.15)
        self.bn2 = nn.BatchNorm1d(32)
        self.output = nn.Linear(32, 1)

    def forward(self, observation, action):
        observation_out = self.observation_fc(observation)
        action_out = self.action_fc(action)
        a = torch.add(observation_out, action_out)
        a = self.merge_fc1(a)
        a = self.bn1(a)
        a = torch.tanh(self.merge_fc2(a))
        a = self.dropout(a)
        a = self.bn2(a)
        return self.output(a)


class ActorCritic:
    def __init__(
            self,
            state_num,
            action_num,
            learning_rate=0.1,
            train_min_size=32,
            size_mem=2000,
            size_predict_mem=2000):

        self.learning_rate = learning_rate  # 0.001
        self.train_min_size = train_min_size

        self.gamma = .095
        self.tau = .125
        self.timestamp = int(time.time())

        self.mem_capacity = size_mem
        self.memory = deque(maxlen=size_mem)
        self.mem_predicted = deque(maxlen=size_predict_mem)

        self.actor_model = Actor(state_num, action_num)
        self.target_actor_model = Actor(state_num, action_num)
        self.actor_optimizer = Adam(
            self.actor_model.parameters(),
            lr=learning_rate)

        self.critic_model = Critic(state_num, action_num)
        self.target_critic_model = Critic(state_num, action_num)
        self.critic_optimizer = Adam(
            self.critic_model.parameters(),
            lr=learning_rate)

        self.mse_loss = nn.MSELoss()

        # if torch.cuda.is_available():
        #     self.actor_model.cuda()
        #     self.target_actor_model.cuda()
        #     self.critic_model.cuda()
        #     self.target_critic_model.cuda()

    # ========================================================================= #
    #                               Model Training                              #
    # ========================================================================= #

    def remember(self, cur_state, action, reward, new_state):
        self.memory.append([cur_state, action, reward, new_state])

    def sampling(self, batch_size):
        indices = np.random.choice(
            len(self.memory) - 1, size=batch_size - 1).tolist()
        tmp = np.array(list(self.memory))
        return tmp[indices]

    def set_train_mode(self):
        self.actor_model = self.actor_model.train()
        self.actor_model.bn1.train()
        self.actor_model.bn2.train()
        self.actor_model.dropout.train()
        self.target_actor_model = self.target_actor_model.train()
        self.target_actor_model.bn1.train()
        self.target_actor_model.bn2.train()
        self.target_actor_model.dropout.train()
        self.critic_model = self.critic_model.train()
        self.critic_model.bn1.train()
        self.critic_model.bn2.train()
        self.critic_model.dropout.train()
        self.target_critic_model = self.target_critic_model.train()
        self.target_critic_model.bn1.train()
        self.target_critic_model.bn2.train()
        self.target_critic_model.dropout.train()

    def set_eval_mode(self):
        self.actor_model = self.actor_model.eval()
        self.actor_model.bn1.eval()
        self.actor_model.bn2.eval()
        self.actor_model.dropout.eval()
        self.target_actor_model = self.target_actor_model.eval()
        self.target_actor_model.bn1.eval()
        self.target_actor_model.bn2.eval()
        self.target_actor_model.dropout.eval()
        self.critic_model = self.critic_model.eval()
        self.critic_model.bn1.eval()
        self.critic_model.bn2.eval()
        self.critic_model.dropout.eval()
        self.target_critic_model = self.target_critic_model.eval()
        self.target_critic_model.bn1.eval()
        self.target_critic_model.bn2.eval()
        self.target_critic_model.dropout.eval()

    def train(self):
        self.batch_size = self.train_min_size  # 32
        if len(self.memory) < self.batch_size:
            return
        self.set_train_mode()
        samples = self.sampling(self.batch_size)
        samples = np.append(samples, [self.memory[-1]], axis=0)

        a_layers = self.target_actor_model.named_children()
        c_layers = self.target_critic_model.named_children()
        for al in a_layers:
            try:
                al[1].weight.data.mul_((1 - self.tau))
                al[1].weight.data.add_(
                    self.tau *
                    self.actor_model.state_dict()[
                        al[0] +
                        '.weight'])
            except BaseException:
                pass
            try:
                al[1].bias.data.mul_((1 - self.tau))
                al[1].bias.data.add_(
                    self.tau *
                    self.actor_model.state_dict()[
                        al[0] +
                        '.bias'])
            except BaseException:
                pass

        for cl in c_layers:
            try:
                cl[1].weight.data.mul_((1 - self.tau))
                cl[1].weight.data.add_(
                    self.tau *
                    self.critic_model.state_dict()[
                        cl[0] +
                        '.weight'])
            except BaseException:
                pass
            try:
                cl[1].bias.data.mul_((1 - self.tau))
                cl[1].bias.data.add_(
                    self.tau *
                    self.critic_model.state_dict()[
                        cl[0] +
                        '.bias'])
            except BaseException:
                pass

        current_state_sample = torch.FloatTensor(
            np.array([s[0] for s in samples]))
        next_state_sample = torch.FloatTensor(
            np.array([s[3] for s in samples]))
        action_sample_input = torch.FloatTensor(
            np.array([s[1] for s in samples]))
        rewards = torch.FloatTensor(np.array([s[2] for s in samples]))

        # if torch.cuda.is_available():
        #     current_state_sample.cuda()
        #     next_state_sample.cuda()
        #     action_sample_input.cuda()
        #     rewards.cuda()

        # Train actor
        a = self.actor_model(current_state_sample, action_sample_input)
        action = torch.add(a, action_sample_input)
        q = self.critic_model(current_state_sample, action)
        a_loss = -torch.mean(q)
        self.actor_optimizer.zero_grad()
        a_loss.backward(retain_graph=True)
        self.actor_optimizer.step()

        # Train critic
        a1 = self.actor_model(current_state_sample, action_sample_input)
        action1 = torch.add(a1, action_sample_input)
        a_ = self.target_actor_model(next_state_sample, action1)
        next_state_action = torch.add(a_, action1)
        q_ = self.target_critic_model(next_state_sample, next_state_action)
        rewards = rewards.reshape(-1, 1)
        q_target = rewards + self.gamma * q_
        q_eval = self.critic_model(current_state_sample, action1)
        td_error = self.mse_loss(q_target, q_eval)
        self.critic_optimizer.zero_grad()
        td_error.backward(retain_graph=True)
        self.critic_optimizer.step()

    # ========================================================================= #
    #                              Model Predictions                            #
    # ========================================================================= #

    def act(self, cur_state, cur_action):
        '''
        cur_state: predicted next state
        '''
        self.set_eval_mode()
        cur_state = torch.FloatTensor([cur_state])
        cur_action = torch.FloatTensor([cur_action])
        # if torch.cuda.is_available():
        #     cur_state.cuda()
        action = self.actor_model(
            cur_state, cur_action).cpu().detach().numpy()[0]

        return action

    def judge(self, cur_state, action) -> float:
        self.set_eval_mode()
        cur_state = torch.FloatTensor([cur_state])
        action = torch.FloatTensor([action])
        # if torch.cuda.is_available():
        #     cur_state.cuda()
        #     action.cuda()
        reward = self.critic_model(
            cur_state, action).cpu().detach().numpy()[0][0]
        return reward


class DDPG_Algorithm():
    def __init__(
            self,
            db_connector: Executor,
            feature_selector: FeatureSelector,
            workload: list,
            selected_knob_config=non_restart_knob_config,
            latency_weight=1,
            throughput_weight=9,
            logger=None) -> None:
        self.db = db_connector
        self.workload = workload
        self.feature_selector = feature_selector
        self.latency_weight = latency_weight
        self.throughput_weight = throughput_weight
        self.random_prob = 0.8
        self.decay = 0.99
        if logger is not None:
            self.logger = logger
        else:
            self.logger = sys.stdout

        knobs = selected_knob_config.keys()
        knob_info = self.db.get_knob_min_max(knobs)
        self.knob_names = []
        self.knob_min = []
        self.knob_max = []
        self.knob_granularity = []
        self.knob_type = []
        for key in selected_knob_config:
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
                self.knob_granularity.append(
                    selected_knob_config[key]["granularity"])
            else:
                self.knob_granularity.append(knob_info[key]['granularity'])
            self.knob_type.append(knob_info[key]['type'])
        self.knob_min = np.array(self.knob_min)
        self.knob_max = np.array(self.knob_max)
        self.knob_granularity = np.array(self.knob_granularity)

        self.action_num = len(self.knob_names)
        self.nonrestart_knob_num = len(non_restart_knob_config)
        db_state = self.db.get_db_state()
        if self.feature_selector is None:
            self.state_num = len(db_state)
        else:
            self.state_num = len(self.feature_selector.transform(db_state))
        self.model = ActorCritic(self.state_num, self.action_num)

    def run_workload(self):
        self.db.restart_db(
            remote_config["port"],
            remote_config["user"],
            remote_config["password"])
        thread_num = self.db.get_max_thread_num()
        db_state = self.db.get_db_state()
        latency, throughput = self.db.run_job(thread_num, self.workload)
        time.sleep(0.2)
        after_db_state = self.db.get_db_state()
        new_state = np.array(after_db_state) - np.array(db_state)
        if self.feature_selector is not None:
            new_state = self.feature_selector.transform(new_state)
        return latency, throughput, new_state

    def take_action(self, action):
        knob_value = np.round((self.knob_max - self.knob_min) /
                              self.knob_granularity * action) * self.knob_granularity + self.knob_min
        knob_value = np.minimum(
            np.maximum(
                knob_value,
                self.knob_min),
            self.knob_max)
        self.logger.write("Knob value: " + str(knob_value) + "\n")
        # print(self.knob_names[:self.nonrestart_knob_num])
        # print(knob_value[:self.nonrestart_knob_num])
        # self.db.change_knob(self.knob_names[:self.nonrestart_knob_num], knob_value[:self.nonrestart_knob_num], self.knob_type[:self.nonrestart_knob_num])
        # if self.nonrestart_knob_num<self.action_num:
        # self.db.change_restart_knob(self.knob_names[self.nonrestart_knob_num:],knob_value[self.nonrestart_knob_num:],self.knob_type[self.nonrestart_knob_num:])
        self.db.change_restart_knob(
            self.knob_names, knob_value, self.knob_type)
        self.db.restart_db(
            remote_config["port"],
            remote_config["user"],
            remote_config["password"])

    def train(
            self,
            total_epoch,
            epoch_steps,
            save_epoch_interval,
            save_folder):
        start_time = time.time()
        torch.autograd.set_detect_anomaly(True)
        self.data = []
        # self.db.reset_knob(self.knob_names[:self.nonrestart_knob_num])
        # if self.nonrestart_knob_num<self.action_num:
        self.db.reset_restart_knob(self.knob_names)
        self.db.restart_db(
            remote_config["port"],
            remote_config["user"],
            remote_config["password"])
        self.init_latency, self.init_throughput, current_state = self.run_workload()
        self.last_latency = self.init_latency
        self.last_throughput = self.init_throughput
        self.logger.write("Init Data Collecting:\n")
        scaler_training_data = []
        scaler_training_data.append(current_state)
        random_record = []
        epoch_data = {}
        epoch_data['init_latency'] = self.init_latency
        epoch_data['init_throughput'] = self.init_throughput
        trial = []
        for step in range(50):
            self.logger.write("Current state: " + str(current_state) + "\n")
            action = np.random.random(size=self.action_num)
            self.take_action(action)
            latency, throughput, new_state = self.run_workload()
            self.logger.write("Latency: " +
                              str(round(latency, 4)) +
                              "\nThroughput: " +
                              str(round(throughput, 4)) +
                              "\n")
            reward = self.calculate_reward(latency, throughput)
            trial.append({'latency': latency,
                          'throughput': throughput,
                          'action': action,
                          'current_state': current_state,
                          'new_state': new_state,
                          'reward': reward})
            # self.model.remember(current_state,action,reward,new_state,workload_embedding)
            scaler_training_data.append(new_state)
            random_record.append([current_state, action, reward, new_state])
            current_state = new_state
            self.logger.write("\n")
        epoch_data['trial'] = trial
        self.data.append(epoch_data)
        self.state_scaler = StandardScaler()
        self.state_scaler.fit(scaler_training_data)
        for s in random_record:
            self.model.remember(
                self.state_scaler.transform(
                    np.array(
                        s[0]).reshape(
                        1, -1))[0], s[1], s[2], self.state_scaler.transform(
                    np.array(
                        s[3]).reshape(
                            1, -1))[0])
        self.logger.write("")
        for epoch in range(total_epoch):
            epoch_data = {}
            self.logger.write("Epoch: " + str(epoch) + "\n")
            epoch_total_reward = 0
            # self.db.reset_knob(self.knob_names[:self.nonrestart_knob_num])
            # if self.nonrestart_knob_num<self.action_num:
            self.db.reset_restart_knob(self.knob_names)
            self.db.restart_db(
                remote_config["port"],
                remote_config["user"],
                remote_config["password"])
            current_action = np.random.random(size=self.action_num)
            self.take_action(current_action)
            self.init_latency, self.init_throughput, current_state = self.run_workload()
            self.last_latency = self.init_latency
            self.last_throughput = self.init_throughput
            epoch_data['init_latency'] = self.init_latency
            epoch_data['init_throughput'] = self.init_throughput
            epoch_data['init_action'] = current_action
            trial = []
            current_state = self.state_scaler.transform(
                np.array(current_state).reshape(1, -1))[0]
            for step in range(epoch_steps):
                self.logger.write(
                    "Current state: " + str(current_state) + "\n")
                # if np.random.random(1)[0]>self.random_prob:
                delta_action = self.model.act(current_state, current_action)
                # else:
                #     action=np.random.random(size=(1,len(self.knob_names)))[0]
                next_action = current_action + delta_action
                self.take_action(next_action)
                latency, throughput, new_state = self.run_workload()
                new_state = self.state_scaler.transform(
                    np.array(new_state).reshape(1, -1))[0]
                self.logger.write("Latency: " +
                                  str(round(latency, 4)) +
                                  "\nThroughput: " +
                                  str(round(throughput, 4)) +
                                  "\n")
                reward = self.calculate_reward(latency, throughput)
                epoch_total_reward += reward
                trial.append({'latency': latency,
                              'throughput': throughput,
                              'current_action': current_action,
                              'delta_action': delta_action,
                              'current_state': current_state,
                              'new_state': new_state,
                              'reward': reward})
                self.logger.write(
                    "Step reward: " +
                    str(reward) +
                    ", Epoch Total Reward: " +
                    str(epoch_total_reward) +
                    "\n")
                self.model.remember(current_state, action, reward, new_state)
                self.model.train()
                current_state = new_state
                current_action = next_action
                self.logger.write("\n")
            epoch_data['trial'] = trial
            self.data.append(epoch_data)
            if ((epoch + 1) % save_epoch_interval) == 0:
                torch.save(
                    self.model.actor_model.state_dict(),
                    os.path.join(
                        save_folder,
                        "actor_epoch_{}.pkl".format(epoch)))
                torch.save(
                    self.model.critic_model.state_dict(),
                    os.path.join(
                        save_folder,
                        "critic_epoch_{}.pkl".format(epoch)))
            epoch_end_time = time.time()
            self.logger.write("Epoch " +
                              str(epoch) +
                              " finish, total time: " +
                              str(epoch_end_time -
                                  start_time) +
                              "s\n")
            self.random_prob = self.random_prob * self.decay

    def calculate_reward(self, latency, throughput):
        dl_0 = (self.init_latency - latency) / max(self.init_latency, latency)
        dt_0 = (throughput - self.init_throughput) / \
            max(self.init_throughput, throughput)
        dl_last = (self.last_latency - latency) / \
            max(self.last_latency, latency)
        dt_last = (throughput - self.last_throughput) / \
            max(self.last_throughput, throughput)
        r_l = 0
        if dl_0 > 0:
            r_l = ((1 + dl_last)**2 - 1) * abs(1 + dl_0)
        else:
            r_l = -(((1 - dl_last)**2 - 1) * abs(1 - dl_0))

        r_t = 0
        if dt_0 > 0:
            r_t = ((1 + dt_last)**2 - 1) * abs(1 + dt_0)
        else:
            r_t = -(((1 - dt_last)**2 - 1) * abs(1 - dt_0))

        reward = (self.throughput_weight * r_t + self.latency_weight *
                  r_l) / (self.throughput_weight + self.latency_weight)
        self.last_latency = latency
        self.last_throughput = throughput
        return reward
