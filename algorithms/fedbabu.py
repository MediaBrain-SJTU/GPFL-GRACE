import sys
sys.path.append(sys.path[0].replace('algorithms', ''))
import torch
from algorithms.fedavg import FedAvg_Trainer, FedAvg_Prostate_Trainer, FedAvg_ISIC_Trainer

class FedBABU_Trainer(FedAvg_Trainer):
    def get_optimier(self, domain_name):
        if self.args.model == 'prostate_unet':
            key_name = 'conv.'
        elif self.args.model == 'isic_b0':
            key_name = '_fc.'
        else:
            key_name = 'fc_class.'
        for k,v in self.global_model.named_parameters():
            if key_name in k:
                v.requires_grad = False
        for k,v in self.models_dict[domain_name].named_parameters():
            if key_name in k:
                v.requires_grad = False
        
        if self.args.optimizer == 'sgd':
            self.optimizers_dict[domain_name] = torch.optim.SGD(filter(lambda p: p.requires_grad, self.models_dict[domain_name].parameters()), lr=self.args.lr, momentum=0.9, weight_decay=5e-4)
        elif self.args.optimizer == 'adamw':
            self.optimizers_dict[domain_name] = torch.optim.AdamW(filter(lambda p: p.requires_grad, self.models_dict[domain_name].parameters()), lr=self.args.lr) # 默认是1e-3
        elif self.args.optimizer == 'adam':
            self.optimizers_dict[domain_name] = torch.optim.Adam(filter(lambda p: p.requires_grad, self.models_dict[domain_name].parameters()), lr=self.args.lr) # isic默认是5e-4 bs64
        elif self.args.optimizer == 'adam_amsgrad':
            self.optimizers_dict[domain_name] = torch.optim.Adam(filter(lambda p: p.requires_grad, self.models_dict[domain_name].parameters()), lr=self.args.lr, weight_decay=3e-5, amsgrad=True)

        
class FedBABU_ISIC_Trainer(FedAvg_ISIC_Trainer, FedBABU_Trainer):
    def get_optimier(self, domain_name):
        FedBABU_Trainer.get_optimier(self, domain_name)
        
class FedBABU_Prostate_Trainer(FedAvg_Prostate_Trainer, FedBABU_Trainer):
    def get_optimier(self, domain_name):
        FedBABU_Trainer.get_optimier(self, domain_name)