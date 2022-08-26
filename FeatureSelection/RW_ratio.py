class RWRatio():
    def __init__(self) -> None:
        pass
    
    def fit(self,workload:list):
        read_num=0
        write_num=0
        for query in workload:
            q=query.lower()
            if 'select' in q:
                read_num+=1
            elif 'update' in q or 'insert' in q or 'update' in q:
                write_num+=1
        return write_num/(read_num+write_num)