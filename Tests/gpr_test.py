import sys
sys.path.append("../")
from TuningAlgorithm.GPR import *
from DBConnector.MysqlExecutor import *
from config import *
import pickle
import time

if __name__=='__main__':
    TRAIN_DATA_SIZE=200
    TRAIN_EPOCH=100
    RECOMMEND_EPOCH=20
    FEATURE_NUM=12
    # KNOB_NUM=5
    PATH_PREFIX="fth2c_t_gpr_0530"

    logger = Logger("../log/{}.log".format(PATH_PREFIX))
    db=MysqlExecutor(ip=db_config["host"],port=db_config["port"],user=db_config["user"],
                     password=db_config["password"],database=db_config["dbname"],remote_user=remote_config["user"],
                       remote_port=remote_config["port"], remote_password=remote_config["password"])
    knob_config=non_restart_knob_config
    knob_names=list(knob_config.keys())
    db.reset_knob(knob_names)
    knob_config.update(restart_knob_config)
    knob_names=list(knob_config.keys())
    db.reset_restart_knob(knob_names)
    # db.restart_db()
    knob_info=db.get_knob_min_max(knob_names)
    knob_names,knob_min,knob_max,knob_granularity,knob_type,_=modifyKnobConfig(knob_info,knob_config)
    
    latency,throughput=db.run_tpcc()
    # latency,throughput=0,0
    logger.write("Default: latency="+str(latency)+", throughput="+str(throughput)+"\n")
    start_time=time.time()
    gpr_model=GPR(FEATURE_NUM,len(knob_names),logger)
    gpr_model.init_latency = latency
    gpr_model.init_throughput = throughput
    f=open("../data/t_tpch_gpr_0530_model.pkl",'rb')
    gpr_model=pickle.load(f)
    f.close()    
    gpr_model.train(db,train_epoch=TRAIN_EPOCH,lr=0.05,knob_config=knob_config,train_data_size=TRAIN_DATA_SIZE)
    f=open("../data/{}_model.pkl".format(PATH_PREFIX),"wb")
    pickle.dump(gpr_model,f)
    f.close()
    end_time=time.time()
    duration=end_time-start_time
    gpr_model.knob_names=knob_names
    logger.write("gpr_model.knob_names "+str(gpr_model.knob_names)+"\n")
    logger.write("Training time: "+str(duration)+"\n")
    print("training is over")


    import torch
    
    x_start=torch.rand(size=[1,len(gpr_model.knob_names)])
    db.reset_knob(gpr_model.knob_names)
    db.reset_restart_knob(gpr_model.knob_names)
    # db.restart_db()
    results=gpr_model.recommend(x_start,RECOMMEND_EPOCH,0.1,explore=False)
    results2=gpr_model.recommend(x_start,RECOMMEND_EPOCH,0.1,explore=True)
    results=results+results2
    for i in range(3):
        x_start=torch.rand(size=[1,len(gpr_model.knob_names)])
        db.reset_restart_knob(gpr_model.knob_names)
        res=gpr_model.recommend(x_start,RECOMMEND_EPOCH,0.1,explore=False)
        results=results+res
    for i in range(3):
        x_start=torch.rand(size=[1,len(gpr_model.knob_names)])
        db.reset_restart_knob(gpr_model.knob_names)
        res=gpr_model.recommend(x_start,RECOMMEND_EPOCH,0.1,explore=True)
        results=results+res
    f=open("../data/{}_results.pkl".format(PATH_PREFIX),'wb')
    pickle.dump(results,f)
    f.close()
    # f=open("../data/l_tpcc_gpr_0530_results.pkl","rb")
    # results=pickle.load(f)
    # f.close()
    # for res in results:
    #     print(res)
    print("testing starts")
    for res in results:
        action, pred_latency,knob_value=res
        t_knob_value = [int(i) for i in knob_value]
        db.change_knob(gpr_model.knob_names,t_knob_value)
        latency,throughput=db.run_tpcc()
        # logger.write("Knob value: "+str(knob_value)+", Latency: "+str(latency)+", Predicted Latency: "+str(pred_latency)+", Throughput: "+str(throughput)+"\n")
        logger.write("Knob value: "+str(knob_value)+", Latency: "+str(latency)+", Predicted Performance: "+str(pred_latency)+", Throughput: "+str(throughput)+"\n")

    # db.reset_restart_knob(knob_names)
    # db.restart_db()