import sys
sys.path.append("../")
import psycopg2
from config import *

vector_dict = {'ProjectSet': 0,
               'ModifyTable': 1,

               'Append': 2, 'Merge Append': 3,

               'Recursive Union': 4,
               'BitmapAnd': 5, 'BitmapOr': 6,
               'Nested Loop': 7,
               'Merge Join': 8, 'Hash Join': 9, 'Seq Scan': 10, 'Sample Scan': 11,
               'Gather': 12, 'Gather Merge': 13,
               'Index Scan': 14, 'Index Only Scan': 15, 'Bitmap Index Scan': 16, 'Bitmap Heap Scan': 17, 'Tid Scan': 18,
               'Subquery Scan': 19, 'Function Scan': 20, 'Table Function Scan': 21, 'Values Scan': 22, 'CTE Scan': 23,
               'Named Tuplestore Scan': 24, 'WorkTable Scan': 25, 'Foreign Scan': 26, 'Custom Scan': 27,
               'Materialize': 28,
               'Sort': 29,
               'Group': 30,
               'Aggregate': 31,
               'WindowAgg': 32,
               'Unique': 33,
               'SetOp': 34,
               'LockRows': 35,
               'Limit': 36,
               'Hash': 37}

class QueryEmbed():
    def __init__(self) -> None:
        try:
            conn=psycopg2.connect(host=db_config["host"],port=int(db_config["port"]),user=db_config["user"],password=db_config["password"],database=db_config["dbname"])
            conn.close()
        except:
            raise Exception("PostgreSQL config may be not correct, cannot connect to database")
    
    def search_plan(self, dict_p,sql_vector):
        op_name = dict_p['Node Type']
        op_id = vector_dict[op_name]
        # print("### Op %d" % op_id)
        cost = int(dict_p['Total Cost']) - int(dict_p['Startup Cost'])
        sql_vector[op_id] += cost

        if 'Plans' in dict_p.keys():
            for d in dict_p['Plans']:
                self.search_plan(d,sql_vector)
        return sql_vector
    
    def fit_transform(self,workload:list):
        self.query_embedding=[]
        for query in workload:
            conn=psycopg2.connect(host=db_config["host"],port=int(db_config["port"]),user=db_config["user"],password=db_config["password"],database=db_config["dbname"])
            cur=conn.cursor()
            sql="explain (format json) "+query
            cur.execute(sql)
            res=cur
            for s in res:
                plan=s[0][0]['Plan']
                vec=[0] * len(vector_dict)
                self.query_embedding.append(self.search_plan(plan,vec))
        return self.query_embedding

                
