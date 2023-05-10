import sys

sys.path.append("../")
from DBConnector.MysqlExecutor import MysqlExecutor
from config import *


def default_latency_throughput(logger, db_connector: MysqlExecutor, workload: list, selected_knob_config=knob_config):
    knob_names = list(selected_knob_config.keys())
    knob_value = []
    for knob_name in knob_names:
        knob_value.append(db_connector.get_db_knob_value(knob_name))
    logger.write("Default Knob: " + str(knob_value) + "\n")
    thread_num = db_connector.get_max_thread_num()
    latency, throughput = db_connector.run_job(thread_num, workload)
    logger.write("Latency: " + str(latency) + "\n")
    logger.write("Throughput: " + str(throughput) + "\n\n")
    logger.flush()


if __name__ == "__main__":
    logger = open("../log/dds_rbs_job_mysql.log", 'w+')
    WORKLOAD_PATH = "../Workload/TestWorkload/test.txt"
    f = open(WORKLOAD_PATH)
    lines = f.readlines()  # 调用文件的 readline()方法
    content = ""
    for line in lines:
        content += line
    content = content.replace('\n', ' ')
    content = content.replace('  ', ' ')
    workload = content.split(';')
    f.close()
    db = MysqlExecutor(ip=db_config["host"], port=db_config["port"], user=db_config["user"],
                       password=db_config["password"], database=db_config["dbname"])

    for i in range(10):
        default_latency_throughput(db, workload)
