import sys
sys.path.append("../")
from TuningAlgorithm.GPR import *
from DBConnector.MysqlExecutor import *
from config import *
import pickle
import time

if __name__=='__main__':
    # WORKLOAD="../Workload/TestWorkload/job_mysql.bin"
    WORKLOAD="../Workload/TestWorkload/train_table3_size4M_sample.txt"
    TRAIN_DATA_SIZE=200
    TRAIN_EPOCH=100
    RECOMMAND_EPOCH=4
    FEATURE_NUM=12
    # KNOB_NUM=5

    # f=open(WORKLOAD,'rb')
    # workload=pickle.load(f)
    # f.close()
    f=open(WORKLOAD,'r')
    workload_lines=f.readlines()
    workload=[i.strip("\n") for i in workload_lines]
    f.close()

    # db=PostgresExecutor(ip=db_config["host"],port=db_config["port"],user=db_config["user"],password=db_config["password"],database=db_config["dbname"])
    db=MysqlExecutor(ip=db_config["host"],port=db_config["port"],user=db_config["user"],password=db_config["password"],database=db_config["dbname"])
    knob_config=non_restart_knob_config
    knob_names=list(knob_config.keys())
    db.reset_knob(knob_names)
    knob_config.update(restart_knob_config)
    knob_names=list(knob_config.keys())
    db.reset_restart_knob(knob_names)
    db.restart_db(remote_config["port"],remote_config["user"],remote_config["password"])
    knob_info=db.get_knob_min_max(knob_names)
    knob_names,knob_min,knob_max,knob_granularity,knob_type=modifyKnobConfig(knob_info,knob_config)
    # thread_num=db.get_max_thread_num()
    # latency,throughput=db.run_job(thread_num,workload)
    # print("Default: latency="+str(latency)+", throughput="+str(throughput))
    start_time=time.time()
    gpr_model=GPR(FEATURE_NUM,len(knob_names))
    gpr_model.train(db,workload,train_epoch=TRAIN_EPOCH,lr=0.05,train_data_size=TRAIN_DATA_SIZE)
    # f=open("../data/gpr_sysbench_ro_allknob_0924_fromserver.pkl",'rb')
    # gpr_model=pickle.load(f)
    # f.close()
    end_time=time.time()
    duration=end_time-start_time
    # f=open("../data/gpr_job_pandbtop5_mysql_1014.pkl","wb")
    # pickle.dump(gpr_model,f)
    # f.close()
    import torch
    
    x_start=torch.rand(size=[1,len(gpr_model.knob_names)])
    db.reset_restart_knob(gpr_model.knob_names)
    db.restart_db(remote_config["port"],remote_config["user"],remote_config["password"])
    results=gpr_model.recommand(x_start,RECOMMAND_EPOCH,0.1,explore=False)
    # results2=gpr_model.recommand(x_start,RECOMMAND_EPOCH,0.1,explore=True)
    # results=results+results2
    for i in range(5):
        x_start=torch.rand(size=[1,len(gpr_model.knob_names)])
        db.reset_restart_knob(gpr_model.knob_names)
        db.restart_db(remote_config["port"],remote_config["user"],remote_config["password"])
        res=gpr_model.recommand(x_start,RECOMMAND_EPOCH,0.1,explore=False)
        results=results+res
    for i in range(5):
        x_start=torch.rand(size=[1,len(gpr_model.knob_names)])
        db.reset_restart_knob(gpr_model.knob_names)
        db.restart_db(remote_config["port"],remote_config["user"],remote_config["password"])
        res=gpr_model.recommand(x_start,RECOMMAND_EPOCH,0.1,explore=True)
        results=results+res
    f=open("../data/gpr_results_sysbench_ro_mysql_allknob_local_1023.pkl",'wb')
    pickle.dump(results,f)
    f.close()
    # f=open("../data/dnn_results_3.pkl","rb")
    # results=pickle.load(f)
    # f.close()
    # for res in results:
    #     print(res)
    f=open("../log/gpr_test_sysbench_ro_mysql_allknob_local_1023.log","a")
    # f.write("Default: latency="+str(latency)+", throughput="+str(throughput)+"\n")
    f.write(str(gpr_model.knob_names)+"\n")
    f.write("Training time: "+str(duration)+"\n")
    for res in results:
        action, pred_latency,knob_value=res
        db.change_restart_knob(gpr_model.knob_names,knob_value,gpr_model.knob_type)
        db.restart_db(remote_config["port"],remote_config["user"],remote_config["password"])
        thread_num=db.get_max_thread_num()
        latency,throughput=db.run_job(thread_num,workload)
        f.write("Knob value: "+str(knob_value)+", Latency: "+str(latency)+", Predicted Latency: "+str(pred_latency)+", Throughput: "+str(throughput)+"\n")
        f.flush()
    f.close()
    db.reset_restart_knob(knob_names)
    db.restart_db(remote_config["port"],remote_config["user"],remote_config["password"])