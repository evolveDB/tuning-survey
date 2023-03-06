import sys

sys.path.append("../")
from TuningAlgorithm.DDS_RBS import DDS_RBS_Algorithm
from DBConnector.MysqlExecutor import MysqlExecutor
from config import *

if __name__ == "__main__":
    NUM_ITERATION = 10
    LHS_N = 20
    # WORKLOAD_PATH = "../../slow_queries/order_can_run_sqls.txt"
    # f = open(WORKLOAD_PATH)
    # lines = f.readlines()  # 调用文件的 readline()方法
    # content = ""
    # for line in lines:
    #     content += line
    # content = content.replace('\n', ' ')
    # content = content.replace('  ', ' ')
    # workload = content.split(';')
    # workload.extend(workload)
    # print(sys.getsizeof(workload))
    # f.close()
    # print('workload:', len(workload))

    logger = open("../log/dds_rbs_order_mysql.log", 'w+')
    db = MysqlExecutor(ip=db_config["host"], port=db_config["port"], user=db_config["user"],
                       password=db_config["password"], database=db_config["dbname"], remote_user=remote_config["user"],
                       remote_port=remote_config["port"], remote_password=remote_config["password"])
    model = DDS_RBS_Algorithm(NUM_ITERATION, LHS_N, logger)

    model.default_latency_throughput(db, [])
    model.train(db)
    logger.close()