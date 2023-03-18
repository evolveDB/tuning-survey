import re
import subprocess

def run_sysbench(script):

    # script='/usr/share/sysbench/tests/include/oltp_legacy/oltp.lua'
    host='127.0.0.1'
    user='root'
    password='TestUser_20221222'
    mode='complex'
    count=6
    size=10000
    threads=10
    time=300 # 300s
    interval=10 # 10s

    cmd_prefix = ("sysbench {} --mysql-host={} --mysql-user={} --mysql-password={} --oltp-test-mode={} " 
                  + "--oltp-tables-count={} --oltp-table-size={} --threads={} --time={} --report-interval={} "
        ).format(script, host, user, password, mode, count, size, threads, time, interval)


    cmd_prepare = cmd_prefix + "prepare"
    prepare_code = subprocess.call(cmd_prepare,shell=True)

    cmd_run = cmd_prefix + "run"
    run_result = subprocess.getoutput(cmd_run)

    transactions_line=re.findall(r"transactions:(.*?)\n",run_result)[0]
    thr = transactions_line.split()[0]
    avg_line=re.findall(r"avg:(.*?)\n",run_result)[0]
    lat=avg_line.split()[0]

    cmd_cleanup = cmd_prefix + "cleanup"
    cleanup_code = subprocess.call(cmd_cleanup,shell=True)

    return float(lat),float(thr)

if __name__ == "__main__":
    l,t = run_sysbench()
    print(l,t)