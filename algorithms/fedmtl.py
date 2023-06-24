import sys

from utils.classification_metric import Balance_Classification, Segmentation2D
from utils.segmentation_metric import SegmentationFundus
sys.path.append(sys.path[0].replace('algorithms', ''))
from algorithms.fedavg import FedAvg_Trainer, FedAvg_Prostate_Trainer
import torch
import copy
import math

def flatten(model):
    state_dict = model.state_dict()
    keys = state_dict.keys()
    W = [state_dict[key].flatten() for key in keys if 'amp_norm' not in key]
    return torch.cat(W)

class FedMTL_Trainer(FedAvg_Trainer):
    def initialize(self):
        super().initialize()
        self.join_clients = len(self.train_domain_list)
        
        self.dim = len(flatten(self.global_model))
        self.W_glob = torch.zeros((self.dim, self.join_clients), device=self.device)
        self.device = self.device

        I = torch.ones((self.join_clients, self.join_clients))
        i = torch.ones((self.join_clients, 1))
        omega = (I - 1 / self.join_clients * i.mm(i.T)) ** 2
        self.omega = omega.to(self.device)
        
        self.W_glob_dict = {domain_name:None for domain_name in self.train_domain_list}
        self.omega_dict = {domain_name:None for domain_name in self.train_domain_list}
    
    def broadcast(self, models_dict=None, site_list=None, param_dict=None):
        if models_dict is None:
            models_dict = self.models_dict
        if site_list is None:
            site_list = self.domain_list
        if param_dict is None:
            param_dict = self.global_model.state_dict()
    
        for domain_name in site_list:
            self.W_glob_dict[domain_name] = copy.deepcopy(self.W_glob)
            self.omega_dict[domain_name] = torch.sqrt(self.omega[0][0])
            
    def aggregation(self, model_dict=None, weight_dict=None, site_list=None):
        self.W_glob = torch.zeros((self.dim, self.join_clients), device=self.device)
        for idx, domain_name in enumerate(self.train_domain_list):
            self.W_glob[:, idx] = flatten(self.models_dict[domain_name])

    
    def step_train(self, site_name, model, imgs, labels):
        output = model(imgs)
        loss = self.criterion(output, labels)
        if site_name in self.train_domain_list:
            W_glob = self.W_glob_dict[site_name]
            omega = self.omega_dict[site_name]
            
            W_glob[:, self.train_domain_list.index(site_name)] = flatten(model)
            loss_regularizer = 0
            loss_regularizer += W_glob.norm() ** 2
            loss_regularizer += torch.sum(torch.sum((W_glob*omega), 1)**2)
            f = (int)(math.log10(W_glob.shape[0])+1) + 1
            loss_regularizer *= 10 ** (-f)

            loss += loss_regularizer
        return loss, output
    

class FedMTL_ISIC_Trainer(FedMTL_Trainer):
    def initialize(self):
        super().initialize()
        self.metric = Balance_Classification()


class FedMTL_Prostate_Trainer(FedAvg_Prostate_Trainer, FedMTL_Trainer):
    def initialize(self):
        FedMTL_Trainer.initialize(self)
        self.metric = Segmentation2D()

    def broadcast(self, models_dict=None, site_list=None, param_dict=None):
        return FedMTL_Trainer.broadcast(self, models_dict, site_list, param_dict)
    
    def aggregation(self, model_dict=None, weight_dict=None, site_list=None):
        return FedMTL_Trainer.aggregation(self, model_dict, weight_dict, site_list)
    
    def step_train(self, site_name, model, imgs, labels):
        return FedMTL_Trainer.step_train(self, site_name, model, imgs, labels)
    
