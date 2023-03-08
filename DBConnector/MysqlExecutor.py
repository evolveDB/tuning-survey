import pymysql
from dbutils.pooled_db import PooledDB
from .BaseExecutor import *
import queue
import time
import paramiko
import os
from config import *


class MysqlExecutor(Executor):
    def __init__(
            self,
            ip,
            port,
            user,
            password,
            database,
            remote_port,
            remote_user,
            remote_password):
        self.ip = ip
        self.port = int(port)
        self.user = user
        self.password = password
        self.database = database
        self.remote_port = remote_port
        self.remote_user = remote_user
        self.remote_password = remote_password
        self.pool = None
        self.total_latency = 0
        self.success_query = 0
        self.lock = threading.Lock()
        self.success_latencies = []
        self.success_queries = []

    def _get_connection(self, database):
        conn = None
        flag = True
        while flag:
            flag = False
            try:
                conn = pymysql.connect(
                    host=self.ip,
                    user=self.user,
                    password=self.password,
                    database=database,
                    port=self.port)
            except BaseException:
                flag = True
        return conn

    def change_knob(self, knob_name, knob_value):
        if len(knob_name) != len(knob_value):
            raise Exception(
                "len(knob_name) should be equal to len(knob_value)")
        non_restart_knob_names = []
        non_restart_knob_values = []
        restart_knob_names = []
        restart_knob_values = []

        for i in range(len(knob_name) - 1):
            if knob_name[i] in non_restart_knob_config_names:
                non_restart_knob_names.append(knob_name[i])
                non_restart_knob_values.append(knob_value[i])
            elif knob_name[i] in restart_knob_config_names:
                restart_knob_names.append(knob_name[i])
                restart_knob_values.append(knob_value[i])

        if len(restart_knob_names) > 0:
            self._change_restart_knob(restart_knob_names, restart_knob_values)
            self.restart_db()

        if len(non_restart_knob_names) > 0:
            self._change_non_restart_knob(
                non_restart_knob_names, non_restart_knob_values)

    def _change_non_restart_knob(self, knob_name, knob_value):
        if len(knob_name) != len(knob_value):
            raise Exception(
                "len(knob_name) should be equal to len(knob_value)")
        conn = self._get_connection(self.database)
        cur = conn.cursor()
        for i in range(len(knob_name)):
            sql = "set global " + \
                  str(knob_name[i]) + "=" + str(int(knob_value[i])) + ";"
            cur.execute(sql)
        cur.close()
        conn.commit()
        conn.close()

    def _change_restart_knob(self, knob_name, knob_value):
        if len(knob_name) != len(knob_value):
            raise Exception(
                "len(knob_name) should be equal to len(knob_value)")
        for i in range(len(knob_name)):
            item_name = str(knob_name[i])
            item_value = str(knob_value[i])
            stdout, stderr = self.ssh_exec_command(
                'sudo -S ' + "sed -i -E 's/^#?({} = )\\S+/{} = {}/' /etc/my.cnf".format(
                    item_name, item_name, item_value))
            print("change_restart_knob_konb_stderr:", stderr)
            print("change_restart_knob_konb_stdout:", stdout)

    def reset_knob(self, knob_name: list):
        conn = self._get_connection(self.database)
        cur = conn.cursor()
        for knob in knob_name:
            sql = 'set @@SESSION.' + str(knob) + '=DEFAULT;'
            cur.execute(sql)
            sql = 'set @@GLOBAL.' + knob + '=DEFAULT;'
            cur.execute(sql)
        cur.close()
        conn.commit()
        conn.close()

    def reset_restart_knob(self, knob_name: list):
        conn = self._get_connection(self.database)
        cur = conn.cursor()
        for i in range(len(knob_name)):
            sql = "set persist " + str(knob_name[i]) + "=DEFAULT;"
        cur.execute(sql)
        cur.close()
        conn.commit()
        conn.close()

    def run_job(self, thread_num, workload: list):
        self.pool = PooledDB(
            creator=pymysql,  # 使用链接数据库的模块
            maxconnections=thread_num,  # 连接池允许的最大连接数，0和None表示不限制连接数
            mincached=0,  # 初始化时，链接池中至少创建的空闲的链接，0表示不创建
            maxcached=0,  # 链接池中最多闲置的链接，0和None不限制
            maxshared=0,
            blocking=True,  # 连接池中如果没有可用连接后，是否阻塞等待。True，等待；False，不等待然后报错
            maxusage=None,  # 一个链接最多被重复使用的次数，None表示无限制
            setsession=[],  # 开始会话前执行的命令列表。
            ping=0,  # ping MySQL服务端，检查是否服务可用。
            host=self.ip,
            port=int(self.port),
            user=self.user,
            password=self.password,
            database=self.database,
            charset='utf8'
        )
        self.success_query = 0
        self.total_latency = 0
        main_queue = queue.Queue(maxsize=0)
        p = Producer("Producer Query", main_queue, workload)
        p.setDaemon(True)
        p.start()
        t_consumer = []
        for i in range(thread_num):
            c = Consumer(i, main_queue, self.consumer_process)
            c.setDaemon(True)
            c.start()
            t_consumer.append(c)
        p.join()
        start = time.time()
        main_queue.join()
        self.pool.close()
        run_time = round(time.time() - start, 1)
        avg_lat = self.total_latency / self.success_query
        avg_qps = self.success_query / (run_time + 1e-5)
        max_value = 0
        for i in range(len(self.success_latencies)):
            if self.success_latencies[i] > max_value:
                max_value = self.success_latencies[i]
        return avg_lat, avg_qps

    def run_tpcc(self):
        run_command = './tpcc.lua --mysql-host={} --mysql-password={}  --mysql-user={} --mysql-db={} --time=60' \
                      ' --threads=64 --report-interval=10 --tables=2 --scale=50' \
                      ' --db-driver=mysql run'.format(self.ip, self.password, self.user, self.database)
        out = os.popen(run_command)
        latency = 0
        throughput = 0
        for line in out.readlines():
            if 'transactions:' in line or 'avg:' in line:
                a = line.split()
                if 'transactions:' == a[0]:
                    throughput = float(a[2][1:])
                    print('T: ' + str(throughput))
                if 'avg:' == a[0]:
                    latency = float(a[1][:2])
                    print('latency: ' + str(latency))
        print("执行完毕")
        if throughput == 0:
            return self.run_tpcc()
        return latency, throughput

    def execute_query_with_pool(self, sql):
        if len(sql) < 10:
            return False
        conn = None
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
            print("mysql execute: " + str(error))
            return False

    def consumer_process(self, task_key):
        query = task_key.split('~#~')[1]
        if query:
            start = time.time()
            result = self.execute_query_with_pool(query)
            end = time.time()
            interval = end - start

            if result:
                self.lock.acquire()
                self.success_query += 1
                self.total_latency += interval
                self.success_latencies.append(interval)
                self.success_queries.append(query)
                self.lock.release()

    def get_db_state(self):
        state_list = []
        conn = self._get_connection(self.database)
        cur = conn.cursor()
        sql = "SELECT count FROM INFORMATION_SCHEMA.INNODB_METRICS where status='enabled'"
        cur.execute(sql)
        result = cur.fetchall()
        for s in result:
            state_list.append(s['count'])
        cur.close()
        conn.close()
        return state_list

    def get_db_knob_value(self, knob_name):
        conn = self._get_connection(self.database)
        cur = conn.cursor()
        sql = "show variables like '{}';".format(knob_name)
        cur.execute(sql)
        result = cur.fetchall()
        cur.close()
        conn.close()
        return result[0][1]

    def get_max_thread_num(self):
        return 800
        # return self.get_db_knob_value('max_connections')

    def get_knob_min_max(self, knob_names) -> dict:
        result = {}
        knob_tuple = str(tuple(knob_names))
        conn = self._get_connection("performance_schema")
        cur = conn.cursor()
        sql = "select VARIABLE_NAME, MIN_VALUE, MAX_VALUE from variables_info where VARIABLE_NAME in " + knob_tuple
        cur.execute(sql)
        sql_result = cur.fetchall()
        for row in sql_result:
            dic = {}
            dic["min"] = int(row[1])
            dic["max"] = int(row[2])
            dic["granularity"] = 1
            dic["type"] = "integer"
            result[row[0]] = dic
        cur.close()
        conn.close()
        return result

    def ssh_exec_command(self, command):
        ssh = paramiko.SSHClient()
        ssh.set_missing_host_key_policy(paramiko.AutoAddPolicy())
        ssh.connect(
            hostname=self.ip,
            port=int(self.remote_port),
            username=self.remote_user,
            password=self.remote_password)
        stdin, stdout, stderr = ssh.exec_command(command, get_pty=True)
        time.sleep(0.1)
        stdin.write(self.remote_password + "\n")
        stdin.flush()
        stdout_readlines = stdout.readlines()
        stderr_read = str(stderr.read(), encoding='utf-8')
        ssh.close()
        return stdout_readlines, stderr_read

    def restart_db(self):
        print('数据库重启中')
        time.sleep(30)
        stdout_readlines, stderr_read = self.ssh_exec_command(
            'sudo systemctl restart mysqld.service')
        print("restart_db_stderr:", stderr_read)
        print("restart_db_stdout:", stdout_readlines)
        print('数据库重启成功')
