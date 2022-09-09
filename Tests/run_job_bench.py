import sys
sys.path.append("../")
from config import *
from DBConnector.PostgresExecutor import *
import pickle

workload_path="../Workload/TestWorkload/workload.bin"
f=open(workload_path,"rb")
workload=pickle.load(f)
f.close()

db=PostgresExecutor(ip=db_config["host"],port=db_config["port"],user=db_config["user"],password=db_config["password"],database=db_config["dbname"])
# values=[1024,4096,32768,33554432,1073709056,2147483647]

# db.restart_db(remote_config["port"],remote_config["user"],remote_config["password"])
thread_num=db.get_max_thread_num()
l,t=db.run_job(thread_num,workload)
print(l)
print(t)

# results=[[] for i in range(len(workload))]
# for value in values:
#     db.change_knob(["maintenance_work_mem"],[value],["integer"])
#     print("Change maintenance_work_mem: "+str(value))
#     db.restart_db(remote_config["port"],remote_config["user"],remote_config["password"])
#     for i in range(len(workload)):
#         query=workload[i]
#         l,t=db.run_job(1,[query])
#         results[i].append(l)
#         print("Query "+str(i)+", Latency "+str(l))

# f=open("../log/maintenance_work_mem_test_2.log",'w')
# for i in range(len(workload)):
#     f.write("Query "+str(i)+", Result: "+str(results[i])+"\n")
# f.close()


# results=[]
# f=open("../log/effective_cache_size_test_from1_to8192_interval32.log",'w')
# for value in range(1,8192,32):
#     db.change_knob(["effective_cache_size"],[value],["integer"])
#     print("Change effective_cache_size: "+str(value))
#     db.restart_db(remote_config["port"],remote_config["user"],remote_config["password"])
#     l,t=db.run_job(100,workload)
#     results.append([value,l,t])
#     print("Value: "+str(value)+", Latency: "+str(l)+", Throughput: "+str(t))
#     f.write("Value: "+str(value)+", Latency: "+str(l)+", Throughput: "+str(t)+"\n")
#     f.flush()
# f.close()