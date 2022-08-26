import sys
sys.path.append("../")
from KnobSelection.Plackett_Burman import *
from DBConnector.PostgresExecutor import *
from config import *
import pickle

if __name__=='__main__':
    WORKLOAD_FILE="../Workload/TestWorkload/workload.bin"
    MODEL_SAVE_FILE="../KnobSelection/model/PB_model.pkl"
    f=open(WORKLOAD_FILE,'rb')
    workload=pickle.load(f)
    f.close()
    
    db=PostgresExecutor(ip=db_config["host"],port=db_config["port"],user=db_config["user"],password=db_config["password"],database=db_config["dbname"])
    model=Plackett_Burman()
    knob_names=list(knob_config.keys())
    knob_info=db.get_knob_min_max(knob_names)
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
    db.reset_knob(knob_names)
    model.fit(db,knob_names,knob_min,knob_max,knob_type,workload,False,0.1)
    f=open(MODEL_SAVE_FILE,'wb')
    pickle.dump(model,f)
    f.close()
    db.reset_knob(knob_names)
    f=open("../log/p_and_b_result.log",'w')
    f.write(str(model.get_top_rank()))
    f.close()
    print(model.get_top_rank())
    # print(model.rank_knob(knob_min))

#['geqo_selection_bias', 'geqo_pool_size', 'vacuum_cost_limit', 'from_collapse_limit', 'statement_timeout', 'join_collapse_limit', 'vacuum_cost_page_hit', 'geqo_threshold', 'max_stack_depth', 'geqo_effort', 'vacuum_multixact_freeze_min_age', 'vacuum_cost_delay', 'vacuum_multixact_freeze_table_age', 'temp_file_limit', 'temp_buffers', 'geqo_generations', 'commit_siblings', 'default_statistics_target', 'work_mem', 'deadlock_timeout', 'maintenance_work_mem', 'vacuum_freeze_min_age', 'vacuum_freeze_table_age', 'commit_delay', 'effective_cache_size', 'vacuum_cost_page_miss', 'effective_io_concurrency', 'vacuum_cost_page_dirty']
#(['geqo_selection_bias', 'geqo_pool_size', 'vacuum_cost_limit', 'from_collapse_limit', 'statement_timeout', 'join_collapse_limit', 'vacuum_cost_page_hit', 'geqo_threshold', 'max_stack_depth', 'geqo_effort', 'vacuum_multixact_freeze_min_age', 'vacuum_cost_delay', 'vacuum_multixact_freeze_table_age', 'temp_file_limit', 'temp_buffers', 'geqo_generations', 'commit_siblings', 'default_statistics_target', 'work_mem', 'deadlock_timeout', 'maintenance_work_mem', 'vacuum_freeze_min_age', 'vacuum_freeze_table_age', 'commit_delay', 'effective_cache_size', 'vacuum_cost_page_miss', 'effective_io_concurrency', 'vacuum_cost_page_dirty'], [1.5, 0.0, 1.0, 1.0, 0.0, 1.0, 0.0, 2.0, 100.0, 1.0, 0.0, 0.0, 0.0, -1.0, 100.0, 0.0, 0.0, 1.0, 64.0, 1.0, 1024.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0])