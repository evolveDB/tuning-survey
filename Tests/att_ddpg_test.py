import sys
sys.path.append("../")
from TuningAlgorithm.ATT_ddpg import *
from DBConnector.PostgresExecutor import *
from FeatureSelection.QueryEmbed import *
from config import *
import os
import pickle
import numpy as np

if __name__=='__main__':
    WORKLOAD="../Workload/TestWorkload/workload.bin"
    MODEL_SAVE_FOLDER="../TuningAlgorithm/model/att_ddpg/"
    FEATURE_SELECTOR_MODEL=None
    LOGGER_PATH="../log/att_ddpg_job_pg_0906_sigmoidtest.log"
    TRAIN_EPOCH=10
    EPOCH_TOTAL_STEPS=15
    SAVE_EPOCH_INTERVAL=10

    f=open(WORKLOAD,'rb')
    workload=pickle.load(f)
    f.close()
    feature_selector=None
    if FEATURE_SELECTOR_MODEL is not None and os.path.exists(FEATURE_SELECTOR_MODEL):
        f=open(FEATURE_SELECTOR_MODEL,'rb')
        feature_selector=pickle.load(f)
        f.close()

    logger=Logger(LOGGER_PATH)
    db=PostgresExecutor(ip=db_config["host"],port=db_config["port"],user=db_config["user"],password=db_config["password"],database=db_config["dbname"])

    knob_config=non_restart_knob_config
    knob_names=list(knob_config.keys())
    db.reset_knob(knob_names)
    knob_config.update(restart_knob_config)
    knob_names=list(knob_config.keys())
    db.reset_restart_knob(knob_names)
    db.restart_db(remote_config["port"],remote_config["user"],remote_config["password"])

    workload_encoder=QueryEmbed()
    query_embeddings=workload_encoder.fit_transform(workload)
    print(query_embeddings[0])
    print(query_embeddings[1])
    workload_embedding=[0]*len(query_embeddings[0])
    workload_embedding=np.array(workload_embedding)
    for query_embed in query_embeddings:
        workload_embedding+=np.array(query_embed)
    workload_embedding=workload_embedding/len(query_embeddings)
    print(workload_embedding)
    knob_info=db.get_knob_min_max(knob_names)
    knob_names,knob_min,knob_max,knob_granularity,knob_type=modifyKnobConfig(knob_info,knob_config)
    thread_num=db.get_max_thread_num()
    latency,throughput=db.run_job(thread_num,workload)
    logger.write("Default: latency="+str(latency)+", throughput="+str(throughput)+"\n")

    model=ATT_DDPG(db,None,workload,logger=logger)
    model.train(workload_embedding=workload_embedding,total_epoch=TRAIN_EPOCH,epoch_steps=EPOCH_TOTAL_STEPS,save_epoch_interval=SAVE_EPOCH_INTERVAL,save_folder=MODEL_SAVE_FOLDER)
    


