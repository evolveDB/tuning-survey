import sys
sys.path.append("../")
from TuningAlgorithm.DNN import *
from DBConnector.PostgresExecutor import *
from config import *
import pickle

if __name__=='__main__':
    WORKLOAD="../Workload/TestWorkload/workload.bin"
    TRAIN_DATA_SIZE=100
    TRAIN_EPOCH=100
    RECOMMAND_EPOCH=20
    FEATURE_NUM=12
    KNOB_NUM=5

    f=open(WORKLOAD,'rb')
    workload=pickle.load(f)
    f.close()

    db=PostgresExecutor(ip=db_config["host"],port=db_config["port"],user=db_config["user"],password=db_config["password"],database=db_config["dbname"])
    knob_config=non_restart_knob_config
    knob_names=list(knob_config.keys())
    db.reset_knob(knob_names)
    knob_config.update(restart_knob_config)
    knob_names=list(knob_config.keys())
    db.reset_restart_knob(knob_names)
    db.restart_db(remote_config["port"],remote_config["user"],remote_config["password"])
    knob_info=db.get_knob_min_max(knob_names)
    knob_names,knob_min,knob_max,knob_granularity,knob_type=modifyKnobConfig(knob_info,knob_config)
    thread_num=db.get_max_thread_num()
    latency,throughput=db.run_job(thread_num,workload)

    dnn_model=DNN_Tune(TRAIN_DATA_SIZE,FEATURE_NUM,KNOB_NUM)
    dnn_model.train(db,workload,knob_names=knob_names,knob_min=knob_min,knob_max=knob_max,knob_granularity=knob_granularity,knob_type=knob_type,train_epoch=TRAIN_EPOCH,lr=0.05)
    f=open("../data/dnn_model_0905_restart.pkl","wb")
    pickle.dump(dnn_model,f)
    f.close()
    # f=open("../data/dnn_model_0827.pkl","rb")
    # dnn_model=pickle.load(f)
    # f.close()
    import torch
    # fc1=list(dnn_model.dnn_model.fc1.named_parameters())
    # fc2=list(dnn_model.dnn_model.fc2.named_parameters())
    # fc3=list(dnn_model.dnn_model.fc3.named_parameters())

    # for param in fc1:
    #     print(param)
    # for param in fc2:
    #     print(param)
    # for param in fc3:
    #     print(param)
    

    x_start=torch.rand(size=[1,len(knob_names)])
    db.reset_restart_knob(dnn_model.knob_names)
    db.restart_db(remote_config["port"],remote_config["user"],remote_config["password"])
    results=dnn_model.recommand(x_start,RECOMMAND_EPOCH,0.1,explore=False)
    results2=dnn_model.recommand(x_start,RECOMMAND_EPOCH,0.1,explore=True)
    results=results+results2
    f=open("../data/dnn_results_0905_restart.pkl",'wb')
    pickle.dump(results,f)
    f.close()
    # f=open("../data/dnn_results_3.pkl","rb")
    # results=pickle.load(f)
    # f.close()
    # for res in results:
    #     print(res)
    f=open("../log/dnn_test_0905_pb_top5.log","a")
    # f.write("Default: latency="+str(latency)+", throughput="+str(throughput)+"\n")
    f.write(str(dnn_model.knob_names)+"\n")
    for res in results:
        action, pred_latency,knob_value=res
        db.change_restart_knob(dnn_model.knob_names,knob_value,dnn_model.knob_type)
        db.restart_db(remote_config["port"],remote_config["user"],remote_config["password"])
        thread_num=db.get_max_thread_num()
        latency,throughput=db.run_job(thread_num,workload)
        f.write("Knob value: "+str(knob_value)+", Latency: "+str(latency)+", Predicted Latency: "+str(pred_latency)+", Throughput: "+str(throughput)+"\n")
        f.flush()
    f.close()