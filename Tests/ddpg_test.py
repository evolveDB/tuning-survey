import sys
sys.path.append("../")
from TuningAlgorithm.ddpg_actionstate import *
from DBConnector.MysqlExecutor import *
from config import *
import os
import pickle

if __name__ == '__main__':
    WORKLOAD = "../Workload/TestWorkload/train_table3_size4M_sample.txt"
    MODEL_SAVE_FOLDER = "../TuningAlgorithm/model/ddpg/sysbench_ro_mysql_actionstate_allknob/"
    FEATURE_SELECTOR_MODEL = None
    LOGGER_PATH = "../log/ddpg_sysbench_ro_4M_mysql_allknob_actionstate_restart_1007.log"
    TRAIN_EPOCH = 100
    EPOCH_TOTAL_STEPS = 50
    SAVE_EPOCH_INTERVAL = 5

    # f=open(WORKLOAD,'rb')
    # workload=pickle.load(f)
    # f.close()

    f = open(WORKLOAD, 'r')
    workload_lines = f.readlines()
    workload = [i.strip("\n") for i in workload_lines]
    f.close()

    feature_selector = None
    if FEATURE_SELECTOR_MODEL is not None and os.path.exists(
            FEATURE_SELECTOR_MODEL):
        f = open(FEATURE_SELECTOR_MODEL, 'rb')
        feature_selector = pickle.load(f)
        f.close()

    logger = Logger(LOGGER_PATH)
    # db=PostgresExecutor(ip=db_config["host"],port=db_config["port"],user=db_config["user"],password=db_config["password"],database=db_config["dbname"])
    db = MysqlExecutor(
        ip=db_config["host"],
        port=db_config["port"],
        user=db_config["user"],
        password=db_config["password"],
        database=db_config["dbname"])
    logger.write("Initialize:\n")
    knob_config = nonrestart_knob_config.copy()
    knob_names = list(knob_config.keys())
    db.reset_knob(knob_names)
    logger.write("Reset non restart:\n")
    knob_config.update(restart_knob_config)
    knob_names = list(knob_config.keys())
    db.reset_restart_knob(knob_names)
    logger.write("Reset restart:\n")
    db.restart_db(
        remote_config["port"],
        remote_config["user"],
        remote_config["password"])
    logger.write("Restarted:\n")
    knob_info = db.get_knob_min_max(knob_names)
    knob_names, knob_min, knob_max, knob_granularity, knob_type = modifyKnobConfig(
        knob_info, knob_config)
    logger.write("Got knob info\n")
    thread_num = db.get_max_thread_num()
    logger.write("Start run workload\n")
    latency, throughput = db.run_job(thread_num, workload)
    logger.write(
        "Default: latency=" +
        str(latency) +
        ", throughput=" +
        str(throughput) +
        "\n")

    model = DDPG_Algorithm(
        db,
        feature_selector,
        workload,
        selected_knob_config=knob_config,
        logger=logger)
    model.train(
        TRAIN_EPOCH,
        EPOCH_TOTAL_STEPS,
        SAVE_EPOCH_INTERVAL,
        MODEL_SAVE_FOLDER)
    db.reset_restart_knob(knob_names)
    db.restart_db(
        remote_config["port"],
        remote_config["user"],
        remote_config["password"])
