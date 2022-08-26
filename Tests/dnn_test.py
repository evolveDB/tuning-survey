import sys
sys.path.append("../")
from TuningAlgorithm.DNN import *
from DBConnector.PostgresExecutor import *
from config import *
import pickle

if __name__=='__main__':
    WORKLOAD="../Workload/TestWorkload/workload.bin"
    TRAIN_DATA_SIZE=200
    TRAIN_EPOCH=100
    RECOMMAND_EPOCH=20
    FEATURE_NUM=12
    KNOB_NUM=12

    f=open(WORKLOAD,'rb')
    workload=pickle.load(f)
    f.close()

    db=PostgresExecutor(ip=db_config["host"],port=db_config["port"],user=db_config["user"],password=db_config["password"],database=db_config["dbname"])
    knob_names=list(knob_config.keys())
    knob_info=db.get_knob_min_max(knob_names)
    knob_names,knob_min,knob_max,knob_granularity,knob_type=modifyKnobConfig(knob_info,knob_config)
    thread_num=db.get_max_thread_num()
    latency,throughput=db.run_job(thread_num,workload)

    dnn_model=DNN_Tune(TRAIN_DATA_SIZE,FEATURE_NUM,KNOB_NUM)
    dnn_model.train(db,workload,TRAIN_EPOCH,0.05)
    f=open("dnn_model.pkl","wb")
    pickle.dump(dnn_model,f)
    f.close()
    import torch
    x_start=torch.randn(size=[1,KNOB_NUM])
    results=dnn_model.recommand(x_start,RECOMMAND_EPOCH,0.01,explore=False)
    results2=dnn_model.recommand(x_start,RECOMMAND_EPOCH,0.01,explore=True)
    results=results+results2
    f=open("dnn_results.pkl",'wb')
    pickle.dump(results,f)
    f.close()
    f=open("../log/dnn_test.log","w")
    f.write(str(dnn_model.knob_names)+"\n")
    for res in results:
        action, pred_latency,knob_value=res
        db.change_knob(dnn_model.knob_names,knob_value,dnn_model.knob_type)
        db.restart_db(remote_config["port"],remote_config["user"],remote_config["password"])
        thread_num=db.get_max_thread_num()
        latency,throughput=db.run_job(thread_num,workload)
        f.write("Knob value: "+str(knob_value)+", Latency: "+str(latency)+", Predicted Latency: "+str(pred_latency)+", Throughput: "+str(throughput)+"\n")
        f.flush()
    f.close()