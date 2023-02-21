import sys

sys.path.append("../")
from TuningAlgorithm.ddpg import *
from DBConnector.PostgresExecutor import *
from FeatureSelection.FeatureSelector import *
from config import *
import os
import pickle

if __name__ == '__main__':
    WORKLOAD = "../Workload/TestWorkload/workload.bin"
    MODEL_SAVE_FOLDER = "../TuningAlgorithm/model/"
    FEATURE_SELECTOR_MODEL = None
    LOGGER_PATH = "../log/ddpg_job_pg_2.log"
    TRAIN_EPOCH = 300
    EPOCH_TOTAL_STEPS = 500
    SAVE_EPOCH_INTERVAL = 30

    f = open(WORKLOAD, 'rb')
    workload = pickle.load(f)
    f.close()
    feature_selector = None
    if FEATURE_SELECTOR_MODEL is not None and os.path.exists(FEATURE_SELECTOR_MODEL):
        f = open(FEATURE_SELECTOR_MODEL, 'rb')
        feature_selector = pickle.load(f)
        f.close()

    logger = Logger(LOGGER_PATH)
    db = PostgresExecutor(ip=db_config["host"], port=db_config["port"], user=db_config["user"],
                          password=db_config["password"], database=db_config["dbname"])
    model = DDPG_Algorithm(db, feature_selector, workload, logger=logger)
    model.train(TRAIN_EPOCH, EPOCH_TOTAL_STEPS, SAVE_EPOCH_INTERVAL, MODEL_SAVE_FOLDER)
