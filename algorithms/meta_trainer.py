import sys
sys.path.append(sys.path[0].replace('algorithms', ''))

class MetaTrainer(object):
    def __init__(self, args) -> None:
        self.args = args
        
        self.data_obj = None
        
        self.global_model = None
        self.models_dict = {}
        self.optimizers_dict = {}
        self.schedulers_dict = {}
        self.warmup_schedulers_dict = {}
        
        self.log_dir = None
        self.save_dir = None
        self.log_file = None
        self.log_ten = None
    
    def initialize(self):
        '''实现模型初始化 数据集初始化 log初始化'''
        NotImplementedError
    
    def site_train(self):
        '''单个节点的模型训练'''
        NotImplementedError
        
    def site_val(self):
        '''单个节点的模型验证'''
        NotImplementedError
        
    def val(self):
        '''整体模型验证'''
        NotImplementedError
    
    def save_chackpoint(self):
        '''保存模型'''
        NotImplementedError
        
    def run(self):
        '''整体流程控制'''
        NotImplementedError        


