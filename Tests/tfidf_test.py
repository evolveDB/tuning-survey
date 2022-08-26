import sys
sys.path.append("../")
from FeatureSelection.TFIDF import TF_IDF
from DBConnector.PostgresExecutor import PostgresExecutor
from config import *
import pickle

if __name__=='__main__':
    WORKLOAD_PATH="../Workload/TestWorkload/workload.bin"
    NUM_OF_LEVELS=10
    f=open(WORKLOAD_PATH,"rb")
    workload=pickle.load(f)
    f.close()
    db=PostgresExecutor(ip=db_config["host"],port=db_config["port"],user=db_config["user"],password=db_config["password"],database=db_config["dbname"])

    model=TF_IDF()
    workload_embed=model.fit_transform(db,workload,NUM_OF_LEVELS)
    print(workload_embed)
