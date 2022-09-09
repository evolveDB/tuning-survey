from unicodedata import numeric
from unittest import result
import psycopg2
from dbutils.pooled_db import PooledDB
from .BaseExecutor import *
import queue
import time
import paramiko

class PostgresExecutor(Executor):
    def __init__(self,ip,port,user,password,database):
        self.ip=ip
        self.port=int(port)
        self.user=user
        self.password=password
        self.database=database
        self.pool=None
        self.total_latency=0
        self.success_query=0
        self.lock=threading.Lock()
    
    def get_connection(self):
        conn=None
        flag=True
        while flag:
            flag=False
            try:
                conn=psycopg2.connect(host=self.ip,user=self.user,password=self.password,database=self.database,port=self.port)
            except:
                flag=True
        return conn

    def change_knob(self, knob_name, knob_value,knob_type):
        if len(knob_name)!=len(knob_value):
            raise Exception("len(knob_name) should be equal to len(knob_value)")
        conn=self.get_connection()
        cur=conn.cursor()
        for i in range(len(knob_name)):
            if knob_type[i]=="integer":
                sql="alter database "+str(self.database)+" set "+str(knob_name[i])+"="+str(int(knob_value[i]))
            else:
                sql="alter database "+str(self.database)+" set "+str(knob_name[i])+"="+str(knob_value[i])
            cur.execute(sql)
        cur.close()
        conn.commit()
        conn.close()
    
    def change_restart_knob(self,knob_name,knob_value,knob_type):
        if len(knob_name)!=len(knob_value):
            raise Exception("len(knob_name) should be equal to len(knob_value)")
        conn=self.get_connection()
        old_isolation_level = conn.isolation_level
        conn.set_isolation_level(0)
        cur=conn.cursor()
        for i in range(len(knob_name)):
            if knob_type[i]=="integer":
                sql="alter system set "+str(knob_name[i])+"="+str(int(knob_value[i]))
            else:
                sql="alter system set "+str(knob_name[i])+"="+str(knob_value[i])
            cur.execute(sql)
        cur.close()
        conn.set_isolation_level(old_isolation_level)
        conn.close()

    def reset_knob(self, knob_name: list):
        conn=self.get_connection()
        cur=conn.cursor()
        for knob in knob_name:
            sql="alter database "+str(self.database)+" reset "+str(knob)
            cur.execute(sql)
        cur.close()
        conn.commit()
        conn.close()
    
    def reset_restart_knob(self,knob_name:list):
        conn=self.get_connection()
        old_isolation_level = conn.isolation_level
        conn.set_isolation_level(0)
        cur=conn.cursor()
        for knob in knob_name:
            sql="alter system reset "+str(knob)
            cur.execute(sql)
        cur.close()
        conn.set_isolation_level(old_isolation_level)
        conn.close()

    def run_job(self, thread_num, workload: list):
        self.pool=PooledDB(
            creator=psycopg2,  # 使用链接数据库的模块
            maxconnections=thread_num,  # 连接池允许的最大连接数，0和None表示不限制连接数
            mincached=0,  # 初始化时，链接池中至少创建的空闲的链接，0表示不创建
            maxcached=0,  # 链接池中最多闲置的链接，0和None不限制
            maxshared=0,
            blocking=True,  # 连接池中如果没有可用连接后，是否阻塞等待。True，等待；False，不等待然后报错
            maxusage=None,  # 一个链接最多被重复使用的次数，None表示无限制
            setsession=[],  # 开始会话前执行的命令列表。
            ping=0, # ping 服务端，检查是否服务可用。
            host=self.ip,
            port=int(self.port),
            user=self.user,
            password=self.password,
            database=self.database
        )
        self.success_query=0
        self.total_latency=0
        main_queue = queue.Queue(maxsize=0)
        p = Producer("Producer Query", main_queue, workload)
        p.setDaemon(True)
        p.start()
        t_consumer = []
        for i in range(thread_num):
            c = Consumer(i, main_queue,self.consumer_process)
            c.setDaemon(True)
            c.start()
            t_consumer.append(c)
        p.join()
        start = time.time()
        main_queue.join()
        self.pool.close()
        run_time = round(time.time() - start, 1)
        avg_lat = self.total_latency / (self.success_query+1e-5)
        avg_qps = self.success_query / (run_time+1e-5)
        return avg_lat,avg_qps

    
    def execute_query_with_pool(self,sql):
        conn=None
        try:
            conn = self.pool.connection()
            cursor = conn.cursor()
            cursor.execute(sql)
            cursor.close()
            conn.commit()
            return True
        except Exception as error:
            if conn is not None:
                conn.close()
            print("Postgres execute: " + str(error))
            error_msg=str(error)
            if "invalid" in error_msg:
                print(sql)
                exit()
            return False

    def consumer_process(self,task_key):
        query = task_key.split('~#~')[1]
        if query:
            success=False
            while not success:
                start = time.time()
                result=self.execute_query_with_pool(query)
                end = time.time()
                interval = end - start

                success=result
                self.lock.acquire()
                if result:
                    self.success_query+=1
                self.total_latency+=interval
                self.lock.release()
                # break

    def get_db_state(self):
        conn=self.get_connection()
        cur=conn.cursor()
        sql="select * from pg_stat_database where datname='"+str(self.database)+"'"
        cur.execute(sql)
        result=cur.fetchall()
        state_list=[]
        for i,s in enumerate(result[0]):
            if i>1 and i<16:
                state_list.append(s)
        cur.close()
        conn.close()
        return state_list

    def get_max_thread_num(self):
        try:
            conn=self.get_connection()
            cur=conn.cursor()
            sql="show max_connections;"
            cur.execute(sql)
            result=cur.fetchall()
            cur.close()
            conn.close()
            return int(result[0][0])-1
        except Exception as error:
            if conn is not None:
                conn.close()
            print("Postgres execute: " + str(error))
            return self.get_max_thread_num()

    def get_knob_min_max(self, knob_names:list)->dict:
        result={}
        knob_tuple=str(tuple(knob_names))
        conn=self.get_connection()
        cur=conn.cursor()
        sql="select name, min_val, max_val,vartype from pg_settings where name in "+knob_tuple
        cur.execute(sql)
        sql_result=cur.fetchall()
        for row in sql_result:
            dic={}
            if row[3]=="integer":
                dic["min"]=int(row[1])
                dic["max"]=int(row[2])
                dic["granularity"]=1
                dic["type"]="integer"
            elif row[3]=="real":
                dic["min"]=float(row[1])
                dic["max"]=float(row[2])
                dic["granularity"]=max((dic["max"]-dic["min"])/1000000,0.001)
                dic["type"]="real"
            result[row[0]]=dic
        cur.close()
        conn.close()
        return result
    
    def restart_db(self,remote_port,remote_user,remote_password):
        ssh=paramiko.SSHClient()
        ssh.set_missing_host_key_policy(paramiko.AutoAddPolicy())
        ssh.connect(hostname=self.ip,port=int(remote_port),username=remote_user,password=remote_password)
        stdin,stdout,stderr=ssh.exec_command("sudo -S service postgresql restart")
        time.sleep(0.1)
        stdin.write(remote_password+"\n")
        stdin.flush()
        ssh.close()
        time.sleep(2)