from ppq import BaseGraph

from ptqBase import PtqBase

'''
Post training quantization for CNN models
'''
class PTQForCNN(PtqBase):
    def ptq(self, model_name: str) -> BaseGraph:
        pass
