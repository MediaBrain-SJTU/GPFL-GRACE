import sys
sys.path.append(sys.path[0].replace('algorithms', ''))
from algorithms.fedavg import FedAvg_Trainer, FedAvg_Prostate_Trainer, FedAvg_ISIC_Trainer



class FedPer_Trainer(FedAvg_Trainer):
    def broadcast(self, models_dict=None, site_list=None, param_dict=None):
        if models_dict is None:
            models_dict = self.models_dict
        if site_list is None:
            site_list = self.domain_list
        if param_dict is None:
            param_dict = self.global_model.state_dict()
        pop_keys = []
        for key in param_dict.keys():
            if self.args.model == 'prostate_unet':
                if 'conv.' in key:
                    pop_keys.append(key)
            elif self.args.model == 'isic_b0':
                if '_fc.' in key:
                    pop_keys.append(key)
            else:
                if 'fc_class.' in key:
                    pop_keys.append(key)
        for key in pop_keys:
            param_dict.pop(key)
            
        for site_name in site_list:
            models_dict[site_name].load_state_dict(param_dict, strict=False)
        return models_dict
    
    
class FedPer_ISIC_Trainer(FedAvg_ISIC_Trainer,FedPer_Trainer):
    def broadcast(self, models_dict=None, site_list=None, param_dict=None):
        return FedPer_Trainer.broadcast(self, models_dict, site_list, param_dict)

class FedPer_Prostate_Trainer(FedAvg_Prostate_Trainer,FedPer_Trainer):
    def broadcast(self, models_dict=None, site_list=None, param_dict=None):
        return FedPer_Trainer.broadcast(self, models_dict, site_list, param_dict) 
    
