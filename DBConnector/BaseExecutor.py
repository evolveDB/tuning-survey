import abc
import queue
import threading


class Producer(threading.Thread):
    def __init__(self, name, queue, workload):
        self.__name = name
        self.__queue = queue
        self.workload = workload
        super(Producer, self).__init__()

    def run(self):
        for index, query in enumerate(self.workload):
            self.__queue.put(str(index) + "~#~" + query)

class Consumer(threading.Thread):
    def __init__(self, name, queue_ins,method):
        self.__name = name
        self.__queue = queue_ins
        self.__method=method
        super(Consumer, self).__init__()

    def run(self):
        while not self.__queue.empty():
            query = self.__queue.get()
            try:
                self.__method(query)
            finally:
                self.__queue.task_done()

class Executor(metaclass=abc.ABCMeta):

    def __init__(self) -> None:
        pass

    @abc.abstractmethod
    def change_knob(self,knob_name:list,knob_value:list,knob_type:list):
        pass

    @abc.abstractmethod
    def reset_knob(self,knob_name:list):
        pass

    @abc.abstractmethod
    def run_job(self,thread_num,workload:list):
        pass

    @abc.abstractmethod
    def get_db_state(self):
        pass

    @abc.abstractmethod
    def get_max_thread_num(self):
        pass

    @abc.abstractmethod
    def get_knob_min_max(self,knob_names)->dict:
        pass

    