import sys
sys.path.append("../")
from TuningAlgorithm.DNN import *
from DBConnector.MysqlExecutor import *
from config import *
import pickle
import time

if __name__=='__main__':
    TRAIN_DATA_SIZE=100
    TRAIN_EPOCH=100
    RECOMMEND_EPOCH=20
    FEATURE_NUM=12
    KNOB_NUM=5
    PATH_PREFIX="ftc2s_l_dnn_0530"

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
    logger.write(str(knob_names)+"\n")
    logger.write("Default: latency="+str(latency)+", throughput="+str(throughput)+"\n")
    logger.write("train_lr={},train_epoch={},recommend_lr={},recommend_epoch={}\n\n".format(0.05,TRAIN_EPOCH,0.1,RECOMMEND_EPOCH))

    start_time=time.time()
    dnn_model=DNN_Tune(TRAIN_DATA_SIZE,FEATURE_NUM,KNOB_NUM,logger)

    f=open("../data/dnn_model_0408_norestart.pkl","rb")
    dnn_model=pickle.load(f)
    f.close()

    dnn_model.init_latency=latency
    dnn_model.init_throughput=throughput
    dnn_model.train_data_size=TRAIN_DATA_SIZE
    dnn_model.logger=logger

    dnn_model.train(db,knob_names=knob_names,knob_min=knob_min,knob_max=knob_max,knob_granularity=knob_granularity,knob_type=knob_type,
                    train_epoch=TRAIN_EPOCH,lr=0.05)
    # f=open("../data/dnn_model_0408_norestart.pkl","wb")
    end_time=time.time()
    duration=end_time-start_time
    f=open("../data/{}_model.pkl".format(PATH_PREFIX),"wb")
    pickle.dump(dnn_model,f)
    f.close()

    import torch
    # dnn_model.knob_min = knob_min
    # dnn_model.knob_max = knob_max
    # dnn_model.knob_granularity = knob_granularity
    # dnn_model.knob_type = knob_type
    # f=open("../data/dnn_model_0408_nonrestart.pkl","wb")
    # pickle.dump(dnn_model,f)
    # f.close()
    db.reset_knob(dnn_model.knob_names)
    db.reset_restart_knob(dnn_model.knob_names)
    # db.restart_db()

    x_start=torch.rand(size=[1,len(knob_names)])
    results=dnn_model.recommend(x_start,RECOMMEND_EPOCH,0.1,explore=False)
    results2=dnn_model.recommend(x_start,RECOMMEND_EPOCH,0.1,explore=True)
    results=results+results2
    f=open("../data/{}_results.pkl".format(PATH_PREFIX),'wb')
    pickle.dump(results,f)
    f.close()

    logger.write("dnn_model.knob_names "+str(dnn_model.knob_names)+"\n")
    logger.write("Training time: "+str(duration)+"\n")
    for res in results:
        action, pred_latency,knob_value=res
        t_knob_value = [int(i) for i in knob_value]
        db.change_knob(dnn_model.knob_names,t_knob_value)
        latency,throughput=db.run_tpcc()
        # logger.write("Knob value: "+str(knob_value)+", Latency: "+str(latency)+", Predicted Latency: "+str(pred_latency)+", Throughput: "+str(throughput)+"\n")
        logger.write("Knob value: "+str(knob_value)+", Latency: "+str(latency)+", Predicted Performance: "+str(pred_latency)+", Throughput: "+str(throughput)+"\n")