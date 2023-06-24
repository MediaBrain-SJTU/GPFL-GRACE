

import sys
sys.path.append(sys.path[0].replace('utils', ''))
from data.a_distance import  calculate, collect_feature_unet
import torch
import numpy as np
from algorithms.fed_distance import Fed_Base_A_Distance_Prostate_Trainer, Fed_Base_A_Distance_ISIC_Trainer

# Base_Trainer = Fed_Base_A_Distance_ISIC_Trainer
Base_Trainer = Fed_Base_A_Distance_Prostate_Trainer

class Test_Trainer(Base_Trainer):
    def __init__(self, log_name, model_name):
        self.load_checkpoint(log_name, model_name)
        self.args.display = True
        super().__init__(self.args)

    def load_checkpoint(self, log_name=None, model_name=None):
        self.load_log_name = log_name
        self.load_dir = f'/nvme/zhangruipeng/logs/{self.load_log_name}/checkpoints/global_{model_name}.pth'
        self.checkpoint = torch.load(self.load_dir, map_location='cuda')
        self.load_n_round = self.checkpoint['n_round']
        self.args = self.checkpoint['args']        
    
    # 取多次的平均值
    def calc_a_distance(self, site_name, global_model, local_model, data_type='train', training_epochs=1):
        # 计算global model和local model的A_distance
        avg_A_distance = []
        for i in range(10):
            global_feature = collect_feature_unet(self.dataloader_dict[site_name][data_type], global_model, self.device)
            local_feature = collect_feature_unet(self.dataloader_dict[site_name][data_type], local_model, self.device)
            A_distance = calculate(global_feature, local_feature, self.device, progress=False, training_epochs=training_epochs, optim_type='adam')
            avg_A_distance.append(A_distance.cpu().numpy())
        # avg_A_distance.sort()
        print(avg_A_distance)
        return np.min(avg_A_distance)
    
    def run(self):
        print(self.load_dir)
        self.global_model.load_state_dict(self.checkpoint['model'])
        self.broadcast()
        self.train(self.load_n_round+1)
        n_round = 1
        avg_a_distance = []
        for site_name in self.train_domain_list:
            A_distance = self.calc_a_distance(site_name, self.global_model, self.models_dict[site_name], 'val', n_round)
            self.log_file.info(f'Round:{n_round:4d} | site: {site_name} | A Distance on train: {A_distance:.4f}')
            avg_a_distance.append(A_distance)
        self.log_file.info(f'Round:{n_round:4d} | Average A Distance on train: {np.mean(avg_a_distance):.4f} +- {np.std(avg_a_distance):.4f}')

        n_round = 5
        avg_a_distance = []
        for site_name in self.train_domain_list:
            A_distance = self.calc_a_distance(site_name, self.global_model, self.models_dict[site_name], 'val', n_round)
            self.log_file.info(f'Round:{n_round:4d} | site: {site_name} | A Distance on train: {A_distance:.4f}')
            avg_a_distance.append(A_distance)
        self.log_file.info(f'Round:{n_round:4d} | Average A Distance on train: {np.mean(avg_a_distance):.4f} +- {np.std(avg_a_distance):.4f}')
        
        n_round = 10
        avg_a_distance = []
        for site_name in self.train_domain_list:
            A_distance = self.calc_a_distance(site_name, self.global_model, self.models_dict[site_name], 'val', n_round)
            self.log_file.info(f'Round:{n_round:4d} | site: {site_name} | A Distance on train: {A_distance:.4f}')
            avg_a_distance.append(A_distance)
        self.log_file.info(f'Round:{n_round:4d} | Average A Distance on train: {np.mean(avg_a_distance):.4f} +- {np.std(avg_a_distance):.4f}')
        
        
if __name__ == '__main__':
    # 首先根据新的a distance计算方式 得到best model 和 last model的a distance
    log_name_list = [
        ]
    model_name_list = ['model_best',
                    #    'last_checkpoint',
                       ]
    for model_name in model_name_list:
        for log_name in log_name_list:
            trainer = Test_Trainer(log_name, model_name)
            trainer.run()









