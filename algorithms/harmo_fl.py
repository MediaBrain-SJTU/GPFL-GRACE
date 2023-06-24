import sys

from utils.classification_metric import Balance_Classification
sys.path.append(sys.path[0].replace('algorithms', ''))
from algorithms.fedavg import FedAvg_Prostate_Trainer, FedAvg_Trainer
from networks.FedOptimizer.HarmoFL import WPOptim
import torch

class harmoFL_Prostate_Trainer(FedAvg_Prostate_Trainer):
    def initialize(self):
        super().initialize()
        for domain_name in self.train_domain_list:
            if self.args.dataset == 'prostate':
                self.optimizers_dict[domain_name] = WPOptim(params=self.models_dict[domain_name].parameters(), base_optimizer=torch.optim.Adam, lr=self.args.lr, alpha=self.args.harmo_alpha, weight_decay=1e-4)
    
    def get_log_name(self, args, log_num, file_name='train'):
        return super().get_log_name(args, log_num, file_name) + f'_harmo{self.args.harmo_alpha}'
    
    def site_epoch_train(self, epoch, site_name, data_type='train', model=None):
        if model is None:
            model = self.models_dict[site_name]
        model.train()
        optimizer = self.optimizers_dict[site_name]
        dataloader = self.dataloader_dict[site_name][data_type]
        scheduler = self.schedulers_dict[site_name]
        for i, data_list in enumerate(dataloader):
            if len(data_list) == 3:
                imgs, labels, _ = data_list
            else:
                imgs, labels = data_list
            imgs, labels = imgs.to(self.device), labels.to(self.device)
            
            optimizer.zero_grad()
            loss, output = self.step_train(site_name, model, imgs, labels)
            loss.backward()
            optimizer.generate_delta(zero_grad=True)
            loss, output = self.step_train(site_name, model, imgs, labels)
            loss.backward()
            optimizer.step(zero_grad=True)
            self.log_ten.add_scalar(f'{site_name}_train_loss', loss.item(), epoch*len(dataloader)+i)
            self.metric.update(output, labels)
        
        self.log_ten.add_scalar(f'{site_name}_train_dice', self.metric.results()['dice'], epoch)
        scheduler.step()


class harmoFL_Trainer(FedAvg_Trainer):
    def initialize(self):
        super().initialize()
        for domain_name in self.train_domain_list:
            self.optimizers_dict[domain_name] = WPOptim(params=self.models_dict[domain_name].parameters(), base_optimizer=torch.optim.SGD, lr=self.args.lr, alpha=self.args.harmo_alpha, weight_decay=5e-4)
    
    def get_log_name(self, args, log_num, file_name='train'):
        return super().get_log_name(args, log_num, file_name) + f'_harmo{self.args.harmo_alpha}'
    
    def site_epoch_train(self, epoch, site_name, data_type='train', model=None):
        if model is None:
            model = self.models_dict[site_name]
        model.train()
        optimizer = self.optimizers_dict[site_name]
        dataloader = self.dataloader_dict[site_name][data_type]
        scheduler = self.schedulers_dict[site_name]
        for i, data_list in enumerate(dataloader):
            if len(data_list) == 3:
                imgs, labels, _ = data_list
            else:
                imgs, labels = data_list
            imgs, labels = imgs.to(self.device), labels.to(self.device)
            
            optimizer.zero_grad()
            loss, output = self.step_train(site_name, model, imgs, labels)
            loss.backward()
            optimizer.generate_delta(zero_grad=True)
            loss, output = self.step_train(site_name, model, imgs, labels)
            loss.backward()
            optimizer.step(zero_grad=True)
            self.log_ten.add_scalar(f'{site_name}_train_loss', loss.item(), epoch*len(dataloader)+i)
            self.metric.update(output, labels)
        
        self.log_ten.add_scalar(f'{site_name}_train_acc', self.metric.results()['acc'], epoch)
        scheduler.step()

class harmoFL_ISIC_Trainer(harmoFL_Trainer):
    def initialize(self):
        super().initialize()
        self.metric = Balance_Classification()
