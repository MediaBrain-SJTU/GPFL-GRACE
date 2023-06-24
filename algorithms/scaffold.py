import sys
sys.path.append(sys.path[0].replace('algorithms', ''))
import torch
from utils.classification_metric import Balance_Classification
from algorithms.fedavg import FedAvg_Trainer, FedAvg_Prostate_Trainer
from networks.FedOptimizer.Scaffold import GenZeroParamList, Scaffold, ListMinus, UpdateLocalControl, UpdateServerControl, Scaffold_Adam
from networks.get_network import GetNetwork


class Scaffold_Trainer(FedAvg_Trainer):
    def __init__(self, args) -> None:
        self.ci_dict = {}
        self.c = None
        super().__init__(args)
        
    def get_optimizer(self, domain_name):
        if self.args.optimizer == 'adam':
            self.optimizers_dict[domain_name] = Scaffold_Adam(self.models_dict[domain_name].parameters(), lr=self.args.lr)
        else:
            self.optimizers_dict[domain_name] = Scaffold(self.models_dict[domain_name].parameters(), lr=self.args.lr, momentum=0.9, weight_decay=5e-4)
        
    
    def get_model(self, pretrained=True):
        self.global_model, self.feature_level = GetNetwork(self.args, self.num_classes, pretrained)
        self.global_model.to(self.device)
        self.get_criterion()
        self.c = GenZeroParamList(self.global_model)
        
        for domain_name in self.domain_list:
            self.models_dict[domain_name], _ = GetNetwork(self.args, self.num_classes, pretrained)
            self.models_dict[domain_name].to(self.device)
            self.ci_dict[domain_name] = GenZeroParamList(self.models_dict[domain_name])
            self.get_optimizer(domain_name)
            if self.args.lr_policy == 'step':
                self.schedulers_dict[domain_name] = torch.optim.lr_scheduler.StepLR(self.optimizers_dict[domain_name], step_size=self.args.local_epochs * self.args.rounds, gamma=0.1)
    
    def site_epoch_train(self, epoch, site_name, c_ci, data_type='train', model=None):
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
            output = model(imgs)
            loss = self.criterion(output, labels)
            loss.backward()
            optimizer.step(c_ci) # scaffold改变的地方
            self.log_ten.add_scalar(f'{site_name}_train_loss', loss.item(), epoch*len(dataloader)+i)
            self.metric.update(output, labels)
        
        self.log_ten.add_scalar(f'{site_name}_train_acc', self.metric.results()['acc'], epoch)
        scheduler.step()
       
    def site_train(self, n_round, site_name, data_type='train', model=None):
        c_ci = ListMinus(self.c, self.ci_dict[site_name])
        K = len(self.dataloader_dict[site_name]['train']) * self.local_epochs
        for local_epoch in range(self.local_epochs):
            epoch = n_round * self.local_epochs + local_epoch
            self.site_epoch_train(epoch, site_name, c_ci, data_type, model)
        self.ci_dict[site_name] = UpdateLocalControl(self.c, self.ci_dict[site_name], self.global_model,  self.models_dict[site_name], K)
    
    def train(self, n_round, data_type='train', model=None):
        super().train(n_round, data_type, model)
        self.c = UpdateServerControl(self.c, self.ci_dict, self.weight_dict)
        


class Scaffold_ISIC_Trainer(Scaffold_Trainer):
    def initialize(self):
        super().initialize()
        self.metric = Balance_Classification()


class Scaffold_Prostate_Trainer(FedAvg_Prostate_Trainer, Scaffold_Trainer):
    def __init__(self, args) -> None:
        self.ci_dict = {}
        self.c = None
        super().__init__(args)
        
    def get_model(self, pretrained=True):
        '''继承自Scaffold_Trainer类'''
        Scaffold_Trainer.get_model(self, pretrained)

    def site_train(self, n_round, site_name, data_type='train', model=None):
        '''继承自Scaffold_Trainer类'''
        Scaffold_Trainer.site_train(self, n_round, site_name, data_type, model)
    
    def train(self, n_round, data_type='train', model=None):
        '''继承自Scaffold_Trainer类'''
        Scaffold_Trainer.train(self, n_round, data_type, model)
        
    def site_epoch_train(self, epoch, site_name, c_ci, data_type='train', model=None):
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
            output = model(imgs)
            loss = self.criterion(output, labels)
            loss.backward()
            optimizer.step(c_ci) # scaffold改变的地方
            self.log_ten.add_scalar(f'{site_name}_train_loss', loss.item(), epoch*len(dataloader)+i)
            self.metric.update(output, labels)
        
        self.log_ten.add_scalar(f'{site_name}_train_dice', self.metric.results()['dice'], epoch)
        scheduler.step()
        