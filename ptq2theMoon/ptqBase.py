from ptq2theMoon import *
from abc import ABCMeta, abstractmethod

'''
The interface to realize normal post-training quantization 
'''
class PtqBase(metaclass=ABCMeta):
    @abstractmethod
    def ptq(self, batchsize: int, model_name: str, ) -> BaseGraph:
        pass

    @abstractmethod
    def report(self, type:str, graph: BaseGraph):
        pass

    @abstractmethod
    def evaluation(self, val_dir: str, graph: BaseGraph, batchsize: int):
        pass

    @abstractmethod
    def profiler_table(self, out_path: str, model_name: str = None, graph: BaseGraph = None, verbose=True):
        pass
