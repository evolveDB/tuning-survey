import sys
sys.path.append("../")
from KnobSelection.lasso import *
from DBConnector.MysqlExecutor import *
from config import *
import pickle
import os
import time

if __name__=='__main__':
    WORKLOAD="../Workload/TestWorkload/job_mysql.bin"
    TRAIN_DATA_SIZE=50
    DATA_PATH="../data/lasso_train_data_mysql_0908.pkl"
    MODEL_SAVE_PATH="../KnobSelection/model/lasso_model_mysql_0908.pkl"
    f=open(WORKLOAD,'rb')
    workload=pickle.load(f)
    f.close()
    start_time=time.time()
    knob_config=nonrestart_knob_config
    knob_config.update(restart_knob_config)
    scalered_knob_data=[]
    knob_data=[]
    metric_data=[]
    latency_data=[]
    throughput_data=[]
    if not os.path.exists(DATA_PATH):
        knobs=knob_config.keys()
        db=MysqlExecutor(ip=db_config["host"],port=db_config["port"],user=db_config["user"],password=db_config["password"],database=db_config["dbname"])
        knob_info=db.get_knob_min_max(knobs)

        knob_names=[]
        knob_min=[]
        knob_max=[]
        knob_granularity=[]
        knob_type=[]
        for key in knob_info:
            knob_names.append(key)
            if "min" in knob_config[key]:
                knob_min.append(knob_config[key]["min"])
            else:
                knob_min.append(knob_info[key]['min'])
            if "max" in knob_config[key]:
                knob_max.append(knob_config[key]["max"])
            else:
                knob_max.append(knob_info[key]['max'])
            if "granularity" in knob_config[key]:
                knob_granularity.append(knob_config[key]["granularity"])
            else:
                knob_granularity.append(knob_info[key]['granularity'])
            knob_type.append(knob_info[key]['type'])
        knob_min=np.array(knob_min)
        knob_max=np.array(knob_max)
        knob_granularity=np.array(knob_granularity)
        db.reset_restart_knob(knob_names)
        db.restart_db(remote_config["port"],remote_config["user"],remote_config["password"])
        print("Start Collecting Data")
        print(knob_names)


        for i in range(TRAIN_DATA_SIZE):
            action=np.random.random(size=len(knob_info))
            scalered_knob_data.append(action)
            knob_value=np.round((knob_max-knob_min)/knob_granularity*action)*knob_granularity+knob_min
            print(knob_value)
            db.change_restart_knob(knob_names,knob_value,knob_type)
            db.restart_db(remote_config["port"],remote_config["user"],remote_config["password"])
            thread_num=db.get_max_thread_num()
            before_state=db.get_db_state()
            l,t=db.run_job(thread_num,workload)
            latency_data.append(l)
            throughput_data.append(t)
            print("Latency: "+str(l))
            print("Throughput: "+str(t))
            time.sleep(0.1)
            after_state=db.get_db_state()
            state=np.array(after_state)-np.array(before_state)
            knob_data.append(knob_value)
            metric_data.append(state)
            print(state)
            print()
        
        data={"knob":knob_data,"metric":metric_data,"scalered_knob":scalered_knob_data,"latency":latency_data,"throughput":throughput_data}
        f=open(DATA_PATH,'wb')
        pickle.dump(data,f)
        f.close()
        db.reset_knob(knob_names)
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
    end_time=time.time()
    f=open(MODEL_SAVE_PATH,'wb')
    pickle.dump(model,f)
    f.close()
    f=open("../log/lasso_result_0908_mysql.log",'a')
    f.write(str(model.get_top_rank()))
    f.write("Training time:"+str(end_time-start_time)+"\n")
    f.close()
    print(model.get_top_rank())

#['commit_siblings', 'effective_cache_size', 'vacuum_multixact_freeze_min_age', 
# 'vacuum_cost_limit', 'max_stack_depth', 'vacuum_freeze_table_age', 'join_collapse_limit', 'maintenance_work_mem', 
# 'geqo_selection_bias', 'work_mem', 'vacuum_cost_delay', 'geqo_threshold', 'deadlock_timeout', 'vacuum_cost_page_miss', 
# 'geqo_pool_size', 'vacuum_freeze_min_age', 'geqo_generations', 'vacuum_multixact_freeze_table_age', 'statement_timeout', 
# 'geqo_effort', 'vacuum_cost_page_dirty', 'from_collapse_limit', 'effective_io_concurrency', 'vacuum_cost_page_hit', 'default_statistics_target', 
# 'temp_file_limit', 'commit_delay', 'temp_buffers']