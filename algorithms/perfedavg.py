import sys
from utils.classification_metric import Balance_Classification
sys.path.append(sys.path[0].replace('algorithms', ''))
import copy
from algorithms.fedavg import FedAvg_Prostate_Trainer, FedAvg_Trainer
from networks.FedOptimizer.PerFedAvg import PerAvgOptimizer, PerAvg_Adam



class PerFedAvg_Trainer(FedAvg_Trainer):
    def get_optimier(self, domain_name):
        if self.args.optimizer == 'adam':
            self.optimizers_dict[domain_name] = PerAvg_Adam(self.models_dict[domain_name].parameters(), lr=self.args.lr)
        else:
            self.optimizers_dict[domain_name] = PerAvgOptimizer(self.models_dict[domain_name].parameters(), lr=self.args.lr)
        
    def site_epoch_train(self, epoch, site_name, data_type='train', model=None):
        if model is None:
            model = self.models_dict[site_name]
        model.train()
        optimizer = self.optimizers_dict[site_name]
        dataloader = self.dataloader_dict[site_name][data_type]
        scheduler = self.schedulers_dict[site_name]
        for i, data_list in enumerate(dataloader):
            temp_model = copy.deepcopy(list(model.parameters()))
            if len(data_list) == 3:
                imgs, labels, _ = data_list
            else:
                imgs, labels = data_list
            imgs, labels = imgs.to(self.device), labels.to(self.device)
            # 将数据分为两部分
            imgs_1, labels_1 = imgs[:len(imgs)//2], labels[:len(imgs)//2]
            imgs_2, labels_2 = imgs[len(imgs)//2:], labels[len(imgs)//2:]
            # step 1
            loss, output = self.step_train(site_name, model, imgs_1, labels_1)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            # step 2
            optimizer.zero_grad()
            loss, output = self.step_train(site_name, model, imgs_2, labels_2)
            loss.backward()
            for old_param, new_param in zip(model.parameters(), temp_model):
                    old_param.data = new_param.data.clone()
            optimizer.step(beta=self.args.lr)
            
            self.log_ten.add_scalar(f'{site_name}_train_loss', loss.item(), epoch*len(dataloader)+i)
            self.metric.update(output, labels_2)
        
        self.log_ten.add_scalar(f'{site_name}_train_acc', self.metric.results()['acc'], epoch)
        scheduler.step()


class PerFedAvg_ISIC_Trainer(PerFedAvg_Trainer):
    def initialize(self):
        super().initialize()
        self.metric = Balance_Classification()


class PerFedAvg_Prostate_Trainer(FedAvg_Prostate_Trainer):
    '''local训练两步 acc改为dice'''
    def get_optimier(self, domain_name):
        if self.args.optimizer == 'adam':
            self.optimizers_dict[domain_name] = PerAvg_Adam(self.models_dict[domain_name].parameters(), lr=self.args.lr)
        else:
            self.optimizers_dict[domain_name] = PerAvgOptimizer(self.models_dict[domain_name].parameters(), lr=self.args.lr)
        
    def site_epoch_train(self, epoch, site_name, data_type='train', model=None):
        if model is None:
            model = self.models_dict[site_name]
        model.train()
        optimizer = self.optimizers_dict[site_name]
        dataloader = self.dataloader_dict[site_name][data_type]
        scheduler = self.schedulers_dict[site_name]
        for i, data_list in enumerate(dataloader):
            temp_model = copy.deepcopy(list(model.parameters()))
            if len(data_list) == 3:
                imgs, labels, _ = data_list
            else:
                imgs, labels = data_list
            imgs, labels = imgs.to(self.device), labels.to(self.device)
            # 将数据分为两部分
            imgs_1, labels_1 = imgs[:len(imgs)//2], labels[:len(imgs)//2]
            imgs_2, labels_2 = imgs[len(imgs)//2:], labels[len(imgs)//2:]
            # step 1
            loss, output = self.step_train(site_name, model, imgs_1, labels_1)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            # step 2
            optimizer.zero_grad()
            loss, output = self.step_train(site_name, model, imgs_2, labels_2)
            loss.backward()
            for old_param, new_param in zip(model.parameters(), temp_model):
                    old_param.data = new_param.data.clone()
            optimizer.step(beta=self.args.lr)
            
            self.log_ten.add_scalar(f'{site_name}_train_loss', loss.item(), epoch*len(dataloader)+i)
            self.metric.update(output, labels_2)
        
        self.log_ten.add_scalar(f'{site_name}_train_dice', self.metric.results()['dice'], epoch)
        scheduler.step()
        
