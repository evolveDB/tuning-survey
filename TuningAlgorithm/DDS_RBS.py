import sys

sys.path.append("../")
from DBConnector.BaseExecutor import Executor
from config import *
import random
import numpy as np


class DDS_RBS_Algorithm():
    def __init__(self, num_iteration: int, lhs_n: int, logger=None) -> None:
        self.T = num_iteration
        self.N = lhs_n
        if logger is not None:
            self.logger = logger
        else:
            self.logger = sys.stdout

    def default_latency_throughput(self, db_connector: Executor, workload: list, selected_knob_config=knob_config):
        knob_names = list(selected_knob_config.keys())
        knob_value = []
        for knob_name in knob_names:
            knob_value.append(db_connector.get_db_knob_value(knob_name))
        self.logger.write("Default Knob: " + str(knob_value) + "\n")
        # thread_num = db_connector.get_max_thread_num()
        latency, throughput = db_connector.run_tpcc()
        self.logger.write("Latency: " + str(latency) + "\n")
        self.logger.write("Throughput: " + str(throughput) + "\n\n")

    def train(self, db_connector: Executor, workload: list, selected_knob_config=knob_config):
        knob_names = list(selected_knob_config.keys())
        knob_info = db_connector.get_knob_min_max(knob_names)
        knob_names, knob_min, knob_max, _, knob_type = modifyKnobConfig(knob_info, selected_knob_config)
        print(knob_names)
        global_bsf = None  # list (knob_values:list,latency:float)
        for ite in range(self.T):
            # reset the search space to the entire search space
            current_max = knob_max.copy()
            current_min = knob_min.copy()
            step = 0
            step_bsf = None  # list (knob_values:list,latency:float)
            self.logger.write("Iteration: " + str(ite) + "\n")
            self.logger.flush()

            while True:  # for each iteration of RBS, keep search until it converges
                # randomly initiate the search space
                step += 1
                sample = []
                current_granularity = []
                for i in range(len(knob_names)):
                    sample.append(random.sample(range(self.N), self.N))
                    current_granularity.append((int(current_max[i]) - int(current_min[i])) / self.N)

                # run workload to find the best point
                result_list = []
                knob_value_record = []
                for i in range(self.N):
                    knob_value = []
                    for j in range(len(knob_names)):
                        knob_value.append(
                            max(int(knob_min[j]), int(current_min[j]) + int(current_granularity[j]) * sample[j][i]))
                    knob_value_record.append(knob_value)
                    db_connector.change_conf_konb(knob_names, knob_value, knob_type)
                    db_connector.restart_db()
                    self.logger.write("Change Knob: " + str(knob_value) + "\n")
                    self.logger.flush()
                    thread_num = db_connector.get_max_thread_num()
                    latency, throughput = db_connector.run_tpcc()
                    self.logger.write("Latency: " + str(latency) + "\n")
                    self.logger.write("Throughput: " + str(throughput) + "\n\n")
                    self.logger.flush()
                    result_list.append(latency)
                bsf_point_idx = np.argmin(result_list)
                interval_id = []
                for i in range(len(knob_names)):
                    interval_id.append(sample[i][bsf_point_idx])

                # check: break the while loop
                if step_bsf is not None and result_list[bsf_point_idx] >= step_bsf[1]:
                    self.logger.write("Iteration result: latency=" + str(result_list[bsf_point_idx]))
                    step_bsf = [knob_value_record[bsf_point_idx], result_list[bsf_point_idx]]
                    if global_bsf is None or step_bsf[1] < global_bsf[1]:
                        self.logger.write("Update global best: latency=" + str(step_bsf[1]))
                        global_bsf = step_bsf.copy()
                    self.logger.flush()
                    break
                elif step_bsf is None:
                    step_bsf = [knob_value_record[bsf_point_idx], result_list[bsf_point_idx]]

                # update the search space
                for i in range(len(knob_names)):
                    current_min[i] = max(int(knob_min[i]),
                                         int(current_min[i]) + (int(interval_id[i]) - 1) * int(current_granularity[i]))
                    current_max[i] = min(int(knob_max[i]),
                                         int(current_min[i]) + (int(interval_id[i]) + 1) * int(current_granularity[i]))

            self.logger.write("End of this Iteration\n\n")
            self.logger.flush()

        return global_bsf
