import sys
sys.path.append("../")
from TuningAlgorithm.DDS_RBS import DDS_RBS_Algorithm
from DBConnector.PostgresExecutor import PostgresExecutor
from config import *
import pickle

if __name__=="__main__":
    WORKLOAD_PATH="../Workload/TestWorkload/workload.bin"
    NUM_ITERATION=10
    LHS_N=10
    f=open(WORKLOAD_PATH,"rb")
    workload=pickle.load(f)
    f.close()

    logger=open("../log/dds_rbs_job_pg_20220822.log",'w')
    db=PostgresExecutor(ip=db_config["host"],port=db_config["port"],user=db_config["user"],password=db_config["password"],database=db_config["dbname"])
    model=DDS_RBS_Algorithm(NUM_ITERATION,LHS_N,logger)

    model.train(db,workload)
    logger.close()