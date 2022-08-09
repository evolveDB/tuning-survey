import sys
sys.path.append("../")
from KnobSelection.lasso import *
from DBConnector.PostgresExecutor import *
from config import *
import pickle
import os

if __name__=='__main__':
    WORKLOAD="../Workload/TestWorkload/workload.bin"
    TRAIN_DATA_SIZE=50
    DATA_PATH="../data/lasso_train_data.pkl"
    MODEL_SAVE_PATH="../KnobSelection/model/lasso_model.pkl"
    f=open(WORKLOAD,'rb')
    workload=pickle.load(f)
    f.close()

    knob_data=[]
    metric_data=[]
    if not os.path.exists(DATA_PATH):
        knobs=knob_config.keys()
        db=PostgresExecutor(ip=db_config["host"],port=db_config["port"],user=db_config["user"],password=db_config["password"],database=db_config["dbname"])
        knob_info=db.get_knob_min_max(knobs)

        knob_names=[]
        knob_min=[]
        knob_max=[]
        knob_granularity=[]
        knob_type=[]
        for key in knob_info:
            knob_names.append(key)
            knob_min.append(knob_info[key]['min'])
            knob_max.append(knob_info[key]['max'])
            knob_granularity.append(knob_info[key]['granularity'])
            if 'type' in knob_config[key]:
                knob_type.append(knob_info[key]['type'])
            else:
                knob_type.append(None)
        knob_min=np.array(knob_min)
        knob_max=np.array(knob_max)
        knob_max[14]=7680
        knob_granularity=np.array(knob_granularity)
        
        for i in range(TRAIN_DATA_SIZE):
            action=np.random.random(size=len(knob_info))
            knob_value=np.round((knob_max-knob_min)/knob_granularity*action)*knob_granularity+knob_min
            print(knob_value)
            db.change_knob(knob_names,knob_value,knob_type)
            thread_num=db.get_max_thread_num()
            before_state=db.get_db_state()
            db.run_job(thread_num,workload)
            time.sleep(0.1)
            after_state=db.get_db_state()
            state=np.array(after_state)-np.array(before_state)
            knob_data.append(knob_value)
            metric_data.append(state)
            print(state)
            print()
        
        data={"knob":knob_data,"metric":metric_data}
        f=open(DATA_PATH,'wb')
        pickle.dump(data,f)
        f.close()
    else:
        knobs=knob_config.keys()
        knob_names=list(knobs)
        f=open(DATA_PATH,'rb')
        data=pickle.load(f)
        f.close()
        knob_data=data["knob"]
        metric_data=data["metric"]
    
    model=Lasso(knob_names)
    model.fit(knob_data,metric_data)
    f=open(MODEL_SAVE_PATH,'wb')
    pickle.dump(model,f)
    f.close()
    print(model.get_top_rank())
