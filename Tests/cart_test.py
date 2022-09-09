import sys
sys.path.append("../")
from KnobSelection.CART import *
from DBConnector.MysqlExecutor import *
from config import *
import pickle

if __name__=='__main__':
    WORKLOAD_PATH="../Workload/TestWorkload/job_mysql.bin"
    MODEL_SAVE_FILE="../KnobSelection/model/cart_0905_mysql_2.pkl"
    NUM_TRAIL=5
    LHS_N=10
    N_ESTIMATOR=100
    MAX_DEPTH=3

    # f=open(WORKLOAD_PATH,"r")
    # workload=f.readlines()
    # f.close()
    # for i in range(len(workload)):
    #     workload[i]=workload[i].strip("\n")
    f=open(WORKLOAD_PATH,"rb")
    workload=pickle.load(f)
    f.close()
    start_time=time.time()
    knob_config=nonrestart_knob_config
    knob_config.update(restart_knob_config)
    db=MysqlExecutor(ip=db_config["host"],port=db_config["port"],user=db_config["user"],password=db_config["password"],database=db_config["dbname"])
    knob_names=list(knob_config.keys())
    knob_info=db.get_knob_min_max(knob_names)
    # knob_names=[]
    # knob_min=[]
    # knob_max=[]
    # knob_granularity=[]
    # knob_type=[]
    # for key in knob_info:
    #     knob_names.append(key)
    #     if "min" in knob_config[key]:
    #         knob_min.append(knob_config[key]["min"])
    #     else:
    #         knob_min.append(knob_info[key]['min'])
    #     if "max" in knob_config[key]:
    #         knob_max.append(knob_config[key]["max"])
    #     else:
    #         knob_max.append(knob_info[key]['max'])
    #     if "granularity" in knob_config[key]:
    #         knob_granularity.append(knob_config[key]["granularity"])
    #     else:
    #         knob_granularity.append(knob_info[key]['granularity'])
    #     knob_type.append(knob_info[key]["type"])
    # knob_min=np.array(knob_min)
    # knob_max=np.array(knob_max)
    # knob_granularity=np.array(knob_granularity)
    knob_names,knob_min,knob_max,knob_granularity,knob_type=modifyKnobConfig(knob_info,knob_config)
    print(knob_names)
    db.reset_restart_knob(knob_names)
    db.restart_db(remote_config["port"],remote_config["user"],remote_config["password"])
    thread_num=db.get_max_thread_num()
    latency,throughput=db.run_job(thread_num,workload)
    print("Default: latency=%s, throughput=%s"%(str(latency),str(throughput)))

    model=CART(LHS_N,N_ESTIMATOR,MAX_DEPTH,num_of_trials=NUM_TRAIL)
    model.fit(db,knob_names,knob_max,knob_min,knob_type,workload)
    end_time=time.time()
    f=open(MODEL_SAVE_FILE,'wb')
    pickle.dump(model,f)
    f.close()
    db.reset_restart_knob(knob_names)
    f=open("../log/cart_result_0905_mysql_server.log",'w')
    f.write(str(model.get_top_rank()))
    f.write("\n")
    f.write("Training time:"+str(end_time-start_time)+"\n")
    f.close()
    print(model.get_top_rank())
    #['temp_file_limit', 'vacuum_cost_page_miss', 'from_collapse_limit', 'default_statistics_target', 'temp_buffers', 'geqo_effort', 'maintenance_work_mem', 'geqo_generations', 'vacuum_multixact_freeze_table_age', 'effective_cache_size', 'join_collapse_limit', 'commit_delay', 'geqo_pool_size', 'statement_timeout', 'vacuum_freeze_min_age', 'work_mem', 'geqo_threshold', 'effective_io_concurrency', 'commit_siblings', 'vacuum_multixact_freeze_min_age', 'deadlock_timeout', 'max_stack_depth', 'vacuum_cost_limit', 'vacuum_cost_page_dirty', 'vacuum_cost_delay', 'vacuum_cost_page_hit', 'geqo_selection_bias', 'vacuum_freeze_table_age']

#['vacuum_cost_delay', 'geqo_effort', 'geqo_selection_bias', 'work_mem', 'vacuum_cost_limit', 'maintenance_work_mem', 'vacuum_freeze_min_age', 'effective_io_concurrency', 'vacuum_multixact_freeze_min_age', 'vacuum_cost_page_hit', 'commit_delay', 'vacuum_freeze_table_age', 'commit_siblings', 'geqo_generations', 'geqo_pool_size', 'vacuum_multixact_freeze_table_age', 'effective_cache_size', 'default_statistics_target', 'join_collapse_limit', 'temp_file_limit', 'geqo_threshold', 'deadlock_timeout', 'max_stack_depth', 'vacuum_cost_page_miss', 'vacuum_cost_page_dirty', 'statement_timeout', 'from_collapse_limit', 'temp_buffers']