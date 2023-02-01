import sys
sys.path.append("../")
from config import *
from DBConnector.MysqlExecutor import *
import pickle

workload_path="../Workload/TestWorkload/train_table3_size4M_sample.txt"
# f=open(workload_path,"rb")
# workload=pickle.load(f)
# f.close()

f=open(workload_path,'r')
workload_lines=f.readlines()
workload=[i.strip("\n") for i in workload_lines]
f.close()

# db=PostgresExecutor(ip=db_config["host"],port=db_config["port"],user=db_config["user"],password=db_config["password"],database=db_config["dbname"])
db=MysqlExecutor(ip=db_config["host"],port=db_config["port"],user=db_config["user"],password=db_config["password"],database=db_config["dbname"])
# values=[1024,4096,32768,33554432,1073709056,2147483647]

# db.restart_db(remote_config["port"],remote_config["user"],remote_config["password"])
# knob_names=['binlog_cache_size', 'innodb_buffer_pool_size', 'innodb_concurrency_tickets', 'innodb_io_capacity', 'innodb_thread_concurrency', 'join_buffer_size', 'max_connections', 'max_length_for_sort_data', 'max_prepared_stmt_count', 'max_sp_recursion_depth', 'max_user_connections', 'preload_buffer_size', 'read_buffer_size', 'sort_buffer_size', 'table_open_cache']
knob_names=['max_connections', 'join_buffer_size', 'preload_buffer_size', 'sort_buffer_size', 'binlog_cache_size', 'max_length_for_sort_data', 'max_prepared_stmt_count', 'max_sp_recursion_depth', 'max_user_connections', 'read_buffer_size', 'innodb_io_capacity', 'innodb_buffer_pool_size', 'table_open_cache', 'innodb_thread_concurrency', 'innodb_concurrency_tickets']
# knob_value=[4294967295, 389529207.0, 1, 2522666025.0, 1000, 128, 1, 4, 0, 255, 0, 1024, 337584564.0, 32768, 1]
knob_value=[1.00000000e+00,1.28000000e+02,2.23390473e+08,3.27680000e+04,4.09600000e+03,8.38860800e+06,4.19430400e+06,0.00000000e+00, 4.29496730e+09,3.31559731e+08,3.31559731e+08,3.31559731e+08, 5.24288000e+05,0.00000000e+00,1.00000000e+00] 
knob_value=[int(i) for i in knob_value]
knob_type=["integer" for i in range(len(knob_names))]
db.change_knob(knob_names, knob_value, knob_type)
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