

import sys
sys.path.append(sys.path[0].replace('algorithms', ''))
from data.a_distance import collect_feature, calculate, collect_feature_unet
from algorithms.fedavg import FedAvg_ISIC_Trainer, FedAvg_Trainer, FedAvg_Prostate_Trainer
from utils.classification_metric import Balance_Classification
import numpy as np



class Fed_Base_A_Distance_Trainer(FedAvg_Trainer):
    def initialize(self):
        super().initialize()
        self.a_distance_dict = {}
    
    def calc_a_distance(self, site_name, global_model, local_model, data_type='train', training_epochs=1):
        avg_A_distance = []
        for i in range(10):
            global_feature = collect_feature(self.dataloader_dict[site_name][data_type], global_model, self.device)
            local_feature = collect_feature(self.dataloader_dict[site_name][data_type], local_model, self.device)
            A_distance = calculate(global_feature, local_feature, self.device, progress=False, training_epochs=training_epochs, optim_type='sgd')
            avg_A_distance.append(A_distance.cpu().numpy())
        return np.mean(avg_A_distance)
    
    def site_train(self, n_round, site_name, data_type='train', model=None):
        super().site_train(n_round, site_name, data_type, model)
        A_distance = self.calc_a_distance(site_name, self.global_model, self.models_dict[site_name], data_type)
        self.log_file.info(f'Round:{n_round:4d} | site: {site_name} | A Distance on {data_type}: {A_distance:.4f}')
        self.log_ten.add_scalar(f'{site_name}/{data_type}/A_distance', A_distance, n_round)
        self.a_distance_dict[n_round][site_name] = A_distance
    
    def train(self, n_round, data_type='train', model=None):
        self.a_distance_dict[n_round] = {}
        super().train(n_round, data_type, model)
        avg_a_distance = sum(self.a_distance_dict[n_round].values()) / len(self.a_distance_dict[n_round])
        self.log_file.info(f'Round:{n_round:4d} | Average A Distance on {data_type}: {avg_a_distance:.4f}')
        self.log_ten.add_scalar(f'Average/{data_type}/A_distance', avg_a_distance, n_round)

class Fed_Base_A_Distance_ISIC_Trainer(Fed_Base_A_Distance_Trainer):
    def initialize(self):
        super().initialize()
        self.metric = Balance_Classification()

class Fed_Base_A_Distance_Prostate_Trainer(FedAvg_Prostate_Trainer):
    def initialize(self):
        super().initialize()
        self.a_distance_dict = {}
    
    def calc_a_distance(self, site_name, global_model, local_model, data_type='train', training_epochs=1):
        avg_A_distance = []
        for i in range(10):
            global_feature = collect_feature_unet(self.dataloader_dict[site_name][data_type], global_model, self.device)
            local_feature = collect_feature_unet(self.dataloader_dict[site_name][data_type], local_model, self.device)
            A_distance = calculate(global_feature, local_feature, self.device, progress=False, training_epochs=training_epochs, optim_type='adam')
            avg_A_distance.append(A_distance.cpu().numpy())
        return np.mean(avg_A_distance)
    
    def site_train(self, n_round, site_name, data_type='train', model=None):
        super().site_train(n_round, site_name, data_type, model)
        A_distance = self.calc_a_distance(site_name, self.global_model, self.models_dict[site_name], data_type)
        self.log_file.info(f'Round:{n_round:4d} | site: {site_name} | A Distance on {data_type}: {A_distance:.4f}')
        self.log_ten.add_scalar(f'{site_name}/{data_type}/A_distance', A_distance, n_round)
        self.a_distance_dict[n_round][site_name] = A_distance
    
    def train(self, n_round, data_type='train', model=None):
        self.a_distance_dict[n_round] = {}
        super().train(n_round, data_type, model)
        avg_a_distance = sum(self.a_distance_dict[n_round].values()) / len(self.a_distance_dict[n_round])
        self.log_file.info(f'Round:{n_round:4d} | Average A Distance on {data_type}: {avg_a_distance:.4f}')
        self.log_ten.add_scalar(f'Average/{data_type}/A_distance', avg_a_distance, n_round)
        


