import sys
sys.path.append(sys.path[0].replace('algorithms', ''))


from algorithms.fedavg import FedAvg_Trainer, Dict_Add, Dict_weight, FedAvg_Prostate_Trainer
from utils.classification_metric import Balance_Classification

def remove_bn(model_dict):
    new_dict = {}
    for k,v in model_dict.items():
        if 'bn' not in k:
            new_dict[k] = v
    return new_dict

class FedBN_Trainer(FedAvg_Trainer):
    def aggregation(self, model_dict=None, weight_dict=None, site_list=None):
        if weight_dict is None:
            weight_dict = self.weight_dict
        if site_list is None:
            site_list = self.train_domain_list
        if model_dict is None:
            model_dict = self.models_dict
            
        new_model_dict = None
        
        for model_name in weight_dict.keys():
            model = model_dict[model_name]
            model_state_dict = model.state_dict()
            if new_model_dict is None:
                new_model_dict = Dict_weight(model_state_dict, weight_dict[model_name])
            else:
                new_model_dict = Dict_Add(new_model_dict, Dict_weight(model_state_dict, weight_dict[model_name]))
        
        self.global_model.load_state_dict(new_model_dict, strict=False)
        self.broadcast()
        return new_model_dict

    def broadcast(self, models_dict=None, site_list=None, param_dict=None):
        if models_dict is None:
            models_dict = self.models_dict
        if site_list is None:
            site_list = self.domain_list
        if param_dict is None:
            param_dict = self.global_model.state_dict()
            
        param_dict = remove_bn(param_dict)
        for site_name in site_list:
            models_dict[site_name].load_state_dict(param_dict, strict=False)
        return models_dict
    
    def val(self, n_round, is_global=False, is_test=False, data_type='val'):
        results_dict = {}
        for site_name in self.train_domain_list:
            if is_global:
                results_dict[site_name] = self.site_evaluation(n_round, site_name, data_type=data_type, model=self.models_dict[site_name], prefix=f'global_{data_type}')
            else:
                results_dict[site_name] = self.site_evaluation(n_round, site_name, data_type=data_type, model=self.models_dict[site_name], prefix=f'local_{data_type}')
        
        val_acc = self.get_val_acc(results_dict, item_name='acc')
        results_dict['avg_val_acc'] = val_acc
        self.log_file.info(f'Round: {n_round:3d} | {"global" if is_global else "local"} Val Acc: {val_acc*100:.2f}%')
        
        if is_test:
            results_dict[self.test_domain] = self.site_evaluation(n_round, self.test_domain, data_type='test', model=self.global_model, prefix='global_test')
        
        return results_dict
    
    

class FedBN_ISIC_Trainer(FedBN_Trainer):
    def initialize(self):
        super().initialize()
        self.metric = Balance_Classification()


class FedBN_Prostate_Trainer(FedAvg_Prostate_Trainer, FedBN_Trainer):
    def aggregation(self, model_dict=None, weight_dict=None, site_list=None):
        return FedBN_Trainer.aggregation(self, model_dict, weight_dict, site_list)

    def broadcast(self, models_dict=None, site_list=None, param_dict=None):
        return FedBN_Trainer.broadcast(self, models_dict, site_list, param_dict)
    
    def val(self, n_round, is_global=False, is_test=False, data_type='val'):
        results_dict = {}
        for site_name in self.train_domain_list:
            if is_global:
                results_dict[site_name] = self.site_evaluation(n_round, site_name, data_type=data_type, model=self.models_dict[site_name], prefix=f'global_{data_type}')
            else:
                results_dict[site_name] = self.site_evaluation(n_round, site_name, data_type=data_type, model=self.models_dict[site_name], prefix=f'local_{data_type}')
        
        val_acc = self.get_val_acc(results_dict, item_name='dice')
        results_dict['avg_val_dice'] = val_acc
        self.log_file.info(f'Round: {n_round:3d} | {"global" if is_global else "local"} Val Dice: {val_acc*100:.2f}%')
        
        if is_test:
            results_dict[self.test_domain] = self.site_evaluation(n_round, self.test_domain, data_type='total', model=self.global_model, prefix='global_total')
        
        return results_dict
    
