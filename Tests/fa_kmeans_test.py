import sys
sys.path.append("../")
from FeatureSelection.FA_Kmeans import *
from DBConnector.PostgresExecutor import *
from config import *
import pickle

if __name__=='__main__':
    TRAIN_DATA_SIZE=500
    WORKLOAD_PATH="../Workload/workload.txt"
    f=open(WORKLOAD_PATH,'rb')
    # workload=f.readlines()
    workload=pickle.load(f)
    f.close()
    # workload=[ i.strip("\n") for i in workload]
    knob_names=[]
    knob_min=[]
    knob_max=[]
    knob_granularity=[]
    knob_type=[]
    for key in knob_config:
        knob_names.append(key)
        knob_min.append(knob_config[key]['min'])
        knob_max.append(knob_config[key]['max'])
        knob_granularity.append(knob_config[key]['granularity'])
        if 'type' in knob_config[key]:
            knob_type.append(knob_config[key]['type'])
        else:
            knob_type.append(None)
    knob_min=np.array(knob_min)
    knob_max=np.array(knob_max)
    knob_granularity=np.array(knob_granularity)

    # db=PostgresExecutor(ip=db_config["host"],port=db_config["port"],user=db_config["user"],password=db_config["password"],database=db_config["dbname"])
    # data=[]
    # for i in range(TRAIN_DATA_SIZE):
    #     action=np.random.random(size=len(knob_config))
    #     knob_value=np.round((knob_max-knob_min)/knob_granularity*action)*knob_granularity+knob_min
    #     print(knob_value)
    #     db.change_knob(knob_names,knob_value,knob_type)
    #     thread_num=db.get_max_thread_num()
    #     before_state=db.get_db_state()
    #     db.run_job(thread_num,workload)
    #     time.sleep(0.1)
    #     after_state=db.get_db_state()
    #     state=np.array(after_state)-np.array(before_state)
    #     data.append(state)
    #     print(state)
    #     print()
    # f=open("data_file.pkl",'wb')
    # pickle.dump(data,f)
    # f.close()

    f=open("../data_file.pkl","rb")
    data=pickle.load(f)
    f.close()
    model=FeatureSelector_FA_Kmeans()
    model.fit(data,fa_components=4,kmeans_k=4)
    # f=open("fa_kmeans_file.pkl",'wb')
    # pickle.dump(model,f)
    # f.close()
    # print(model.coef)
    print(model.label2metric)

##{0: array([6], dtype=int64), 1: array([4], dtype=int64), 2: array([], dtype=int64), 3: array([3], dtype=int64), 4: array([5], dtype=int64), 5: array([1], dtype=int64), 6: array([2], dtype=int64), 7: array([ 0,  7,  8,  9, 10, 11, 12, 13], dtype=int64)}
## [         0        219          4   39685086  162365463  310863700
##  212222769          0          0          0          0       1040
## 1300178371          0]