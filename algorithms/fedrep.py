import sys
sys.path.append(sys.path[0].replace('algorithms', ''))
from utils.classification_metric import Balance_Classification
import torch
from algorithms.fedrod import FedRoD_Trainer, FedRoD_Prostate_Trainer
import copy
class FedRep_Trainer(FedRoD_Trainer):
    '''直接舍弃global model的head 只用local head 舍弃test domain上的效果'''
    def get_model(self, pretrained=True):
        super().get_model(pretrained)
        self.local_heads_dict = {}
        self.local_optimizers_dict = {}
        for domain_name in self.train_domain_list:
            self.get_load_head(domain_name)
            self.optimizers_dict[domain_name] = torch.optim.SGD(self.models_dict[domain_name].parameters(), lr=self.args.lr, momentum=0.9, weight_decay=5e-4)
            self.local_optimizers_dict[domain_name] = torch.optim.SGD(self.local_heads_dict[domain_name].parameters(), lr=self.args.lr, momentum=0.9, weight_decay=5e-4)
            
    def step_train(self, site_name, model, imgs, labels):
        output, x_feature = model(imgs, feature_out=True)
        local_output = self.local_heads_dict[site_name](x_feature)
        loss = self.criterion(local_output, labels)
        return loss, local_output
    
    def site_epoch_train(self, epoch, site_name, data_type='train', model=None):
        if model is None:
            model = self.models_dict[site_name]
        model.train()
        
        dataloader = self.dataloader_dict[site_name][data_type]
        scheduler = self.schedulers_dict[site_name]
        
        for i, data_list in enumerate(dataloader):
            if len(data_list) == 3:
                imgs, labels, _ = data_list
            else:
                imgs, labels = data_list
            imgs, labels = imgs.to(self.device), labels.to(self.device)
            
            loss, output = self.step_train(site_name, model, imgs, labels)
            if i%11==10: # 先训10次feature extract 再训一次classifier
                optimizer = self.local_optimizers_dict[site_name]
            else:
                optimizer = self.optimizers_dict[site_name]
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            self.log_ten.add_scalar(f'{site_name}_train_loss', loss.item(), epoch*len(dataloader)+i)
            self.metric.update(output, labels)
        
        self.log_ten.add_scalar(f'{site_name}_train_acc', self.metric.results()['acc'], epoch)
        scheduler.step()
        
    def site_evaluation(self, n_round, site_name, data_type='val', model=None, prefix='after_fed'):
        '''针对local model和global model使用不同的inference方法'''
        if model is None:
            model = self.models_dict[site_name]
        model.eval()
        dataloader = self.dataloader_dict[site_name][data_type]
        with torch.no_grad():
            for i, data_list in enumerate(dataloader):
                if len(data_list) == 3:
                    imgs, labels, _ = data_list
                else:
                    imgs, labels = data_list
                imgs = imgs.to(self.device)
                if 'global' in prefix: # 只测试global model
                    output = model(imgs)
                    if isinstance(output, tuple):
                        output = output[0]
                else:
                    output_global, x_feature = model(imgs, feature_out=True)
                    output_local = self.local_heads_dict[site_name](x_feature)
                    if isinstance(output_local, tuple):
                        output_local = output_local[0]
                    output = output_local
                
                self.metric.update(output, labels)
        results_dict = self.metric.results()
        self.log_ten.add_scalar(f'{prefix}_{site_name}_loss', results_dict['loss'], n_round)
        self.log_ten.add_scalar(f'{prefix}_{site_name}_acc', results_dict['acc'], n_round)
        self.log_file.info(f'{prefix} Round: {n_round:3d} | Domain: {site_name} | loss: {results_dict["loss"]:.4f} | Acc: {results_dict["acc"]*100:.2f}%')
        return results_dict


class FedRep_ISIC_Trainer(FedRep_Trainer):
    def initialize(self):
        super().initialize()
        self.metric = Balance_Classification()
        
        
class FedRep_Prostate_Trainer(FedRoD_Prostate_Trainer):
    '''直接舍弃global model的head 只用local head 舍弃test domain上的效果'''
    def get_model(self, pretrained=True):
        super().get_model(pretrained)
        self.local_heads_dict = {}
        self.local_optimizers_dict = {}
        for domain_name in self.train_domain_list:
            self.get_load_head(domain_name)
            self.optimizers_dict[domain_name] = torch.optim.Adam(self.models_dict[domain_name].parameters(), lr=self.args.lr)
            self.local_optimizers_dict[domain_name] = torch.optim.Adam(self.local_heads_dict[domain_name].parameters(), lr=self.args.lr)
            
    def step_train(self, site_name, model, imgs, labels):
        output, x_feature = model(imgs, feature_out=True)
        local_output = self.local_heads_dict[site_name](x_feature)
        loss = self.criterion(local_output, labels)
        return loss, local_output
    
    def site_epoch_train(self, epoch, site_name, data_type='train', model=None):
        if model is None:
            model = self.models_dict[site_name]
        model.train()
        
        dataloader = self.dataloader_dict[site_name][data_type]
        scheduler = self.schedulers_dict[site_name]
        
        for i, data_list in enumerate(dataloader):
            if len(data_list) == 3:
                imgs, labels, _ = data_list
            else:
                imgs, labels = data_list
            imgs, labels = imgs.to(self.device), labels.to(self.device)
            
            loss, output = self.step_train(site_name, model, imgs, labels)
            if i%11==10: # 先训10次feature extract 再训一次classifier
                optimizer = self.local_optimizers_dict[site_name]
            else:
                optimizer = self.optimizers_dict[site_name]
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            self.log_ten.add_scalar(f'{site_name}_train_loss', loss.item(), epoch*len(dataloader)+i)
            self.metric.update(output, labels)
        
        self.log_ten.add_scalar(f'{site_name}_train_dice', self.metric.results()['dice'], epoch)
        scheduler.step()

    def site_evaluation(self, n_round, site_name, data_type='val', model=None, prefix='after_fed'):
        '''针对local model和global model使用不同的inference方法'''
        if model is None:
            model = self.models_dict[site_name]
        model.eval()
        dataloader = self.dataloader_dict[site_name][data_type]
        with torch.no_grad():
            for i, data_list in enumerate(dataloader):
                if len(data_list) == 3:
                    imgs, labels, _ = data_list
                else:
                    imgs, labels = data_list
                imgs, labels = imgs.to(self.device), labels.to(self.device)
                if 'global' in prefix: # 只测试global model
                    output = model(imgs)
                    if isinstance(output, tuple):
                        output = output[0]
                else:
                    _, x_feature = model(imgs, feature_out=True)
                    output_local = self.local_heads_dict[site_name](x_feature)
                    if isinstance(output_local, tuple):
                        output_local = output_local[0]
                    output = output_local
                
                self.metric.update(output, labels)
        results_dict = self.metric.results()
        self.log_ten.add_scalar(f'{prefix}_{site_name}_loss', results_dict['loss'], n_round)
        self.log_ten.add_scalar(f'{prefix}_{site_name}_dice', results_dict['dice'], n_round)
        self.log_file.info(f'{prefix} Round: {n_round:3d} | Domain: {site_name} | loss: {results_dict["loss"]:.4f} | Dice: {results_dict["dice"]*100:.2f}%')
        return results_dict