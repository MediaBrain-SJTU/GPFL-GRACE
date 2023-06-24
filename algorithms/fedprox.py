import sys
sys.path.append(sys.path[0].replace('algorithms', ''))
from algorithms.fedavg import FedAvg_Trainer, FedAvg_Prostate_Trainer, Balance_Classification
import torch
from networks.get_network import GetNetwork
from networks.FedOptimizer.FedProx import FedProx, FedProx_Adam

class FedProx_Trainer(FedAvg_Trainer):
    def get_log_name(self, args, log_num, file_name='train'):
        return super().get_log_name(args, log_num, file_name) + '-mu' + str(self.mu)
    
    def initialize(self):
        self.mu = self.args.mu
        super().initialize()
    
    def get_optimier(self, domain_name):
        if self.args.optimizer == 'adam':
            self.optimizers_dict[domain_name] = FedProx_Adam(self.models_dict[domain_name].parameters(), lr=self.args.lr, mu=self.mu)
        else:
            self.optimizers_dict[domain_name] = FedProx(self.models_dict[domain_name].parameters(), lr=self.args.lr, momentum=0.9, weight_decay=5e-4, mu=self.mu)
    
    def get_model(self, pretrained=True):
        self.global_model, self.feature_level = GetNetwork(self.args, self.num_classes, pretrained)
        self.global_model.to(self.device)
        self.criterion = torch.nn.CrossEntropyLoss().to(self.device)
        
        for domain_name in self.domain_list:
            self.models_dict[domain_name], _ = GetNetwork(self.args, self.num_classes, pretrained)
            self.models_dict[domain_name].to(self.device)
            self.get_optimier(domain_name)
            if self.args.lr_policy == 'step':
                self.schedulers_dict[domain_name] = torch.optim.lr_scheduler.StepLR(self.optimizers_dict[domain_name], step_size=self.args.local_epochs * self.args.rounds, gamma=0.1)

    def aggregation(self, model_dict=None, weight_dict=None, site_list=None):
        new_weight_dict = super().aggregation(model_dict, weight_dict, site_list)
        for domain_name in self.train_domain_list:
            self.optimizers_dict[domain_name].update_old_init(self.global_model.parameters())
        return new_weight_dict
        

class FedProx_ISIC_Trainer(FedProx_Trainer):
    def initialize(self):
        super().initialize()
        self.metric = Balance_Classification()

class FedProx_Prostate_Trainer(FedAvg_Prostate_Trainer):
    def get_log_name(self, args, log_num, file_name='train'):
        return super().get_log_name(args, log_num, file_name) + '-mu' + str(self.mu)
    
    def initialize(self):
        self.mu = self.args.mu
        super().initialize()
    
    def get_optimier(self, domain_name):
        if self.args.optimizer == 'adam':
            self.optimizers_dict[domain_name] = FedProx_Adam(self.models_dict[domain_name].parameters(), lr=self.args.lr, mu=self.mu)
        else:
            self.optimizers_dict[domain_name] = FedProx(self.models_dict[domain_name].parameters(), lr=self.args.lr, momentum=0.9, weight_decay=5e-4, mu=self.mu)
            
    def aggregation(self, model_dict=None, weight_dict=None, site_list=None):
        new_weight_dict = super().aggregation(model_dict, weight_dict, site_list)
        for domain_name in self.train_domain_list:
            self.optimizers_dict[domain_name].update_old_init(self.global_model.parameters())
        return new_weight_dict






