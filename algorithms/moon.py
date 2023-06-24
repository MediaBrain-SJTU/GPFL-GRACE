
import sys
sys.path.append(sys.path[0].replace('algorithms', ''))
from algorithms.fedavg import FedAvg_Trainer, FedAvg_Prostate_Trainer, FedAvg_Fundus_Trainer
from utils.classification_metric import Balance_Classification
import torch
import torch.nn.functional as F
import copy
from networks.FedOptimizer.PerFedAvg import PerAvgOptimizer, PerAvg_Adam
from algorithms.grace_server import GRACE_Server_Trainer, GRACE_Server_Prostate_Trainer
from algorithms.fedrod import BalancedSoftmax

class MOON_Trainer(FedAvg_Trainer):
    def initialize(self):
        self.pre_models_dict = {}
        self.cos = torch.nn.CosineSimilarity(dim=-1)
        self.con_criteria = torch.nn.CrossEntropyLoss().cuda()
        self.temperature = 0.5
        super().initialize()
        
    def get_model(self, pretrained=True):
        super().get_model(pretrained)
        for domain_name in self.domain_list:
            self.pre_models_dict[domain_name] = copy.deepcopy(self.models_dict[domain_name])

    def get_log_name(self, args, log_num, file_name='train'):
        return super().get_log_name(args, log_num, file_name) + f'_typeMOON_warmup{self.args.align_warmup}_coral_weight{self.args.align_weight}'
    
    def step_train(self, site_name, model, imgs, labels):
        pre_local_model = self.pre_models_dict[site_name]
        global_model = self.global_model
        
        output, local_feature = model(imgs, feature_out=True)
        _, global_feature = global_model(imgs, feature_out=True)
        _, pre_local_feature = pre_local_model(imgs, feature_out=True)
        
        local_feature = local_feature.view(local_feature.shape[0], -1)
        global_feature = global_feature.view(global_feature.shape[0], -1)
        pre_local_feature = pre_local_feature.view(pre_local_feature.shape[0], -1)
        
        positive = self.cos(local_feature, global_feature)
        negative = self.cos(local_feature, pre_local_feature)
        logits = torch.cat([positive.reshape(-1, 1), negative.reshape(-1, 1)], dim=1) / self.temperature
        con_labels = torch.zeros(imgs.size(0), dtype=torch.long).to(self.device)
        
        sup_loss = self.criterion(output, labels)
        con_loss = self.con_criteria(logits, con_labels)
        
        loss = sup_loss + self.args.align_weight * con_loss
        return loss, output
    
    def site_train(self, n_round, site_name, data_type='train', model=None):
        for local_epoch in range(self.local_epochs):
            epoch = n_round * self.local_epochs + local_epoch
            self.site_epoch_train(epoch, site_name, data_type, model)
            self.pre_models_dict[site_name] = copy.deepcopy(self.models_dict[site_name]) # 更新pre_local_model

class MOON_ISIC_Trainer(MOON_Trainer):
    def initialize(self):
        super().initialize()
        self.metric = Balance_Classification()

class MOON_Prostate_Trainer(FedAvg_Prostate_Trainer, MOON_Trainer):
    def initialize(self):
        self.pre_models_dict = {}
        self.cos = torch.nn.CosineSimilarity(dim=-1)
        self.temperature = 0.5 # paper中给出的默认结果
        super().initialize()
        
    def get_model(self, pretrained=True):
        super().get_model(pretrained)
        for domain_name in self.domain_list:
            self.pre_models_dict[domain_name] = copy.deepcopy(self.models_dict[domain_name])

    def get_log_name(self, args, log_num, file_name='train'):
        return MOON_Trainer.get_log_name(self, args, log_num, file_name)
    
    def step_train(self, site_name, model, imgs, labels):
        return MOON_Trainer.step_train(self, site_name, model, imgs, labels)
    
    def site_train(self, n_round, site_name, data_type='train', model=None):
        for local_epoch in range(self.local_epochs):
            epoch = n_round * self.local_epochs + local_epoch
            self.site_epoch_train(epoch, site_name, data_type, model)
            self.pre_models_dict[site_name] = copy.deepcopy(self.models_dict[site_name]) # 更新pre_local_model
            
            
class MOON_Fundus_Trainer(FedAvg_Fundus_Trainer, MOON_Trainer):
    def initialize(self):
        self.pre_models_dict = {}
        self.cos = torch.nn.CosineSimilarity(dim=-1)
        self.con_criteria = torch.nn.CrossEntropyLoss().cuda()
        self.temperature = 0.5
        FedAvg_Fundus_Trainer.initialize(self)
        
    def get_model(self, pretrained=True):
        FedAvg_Fundus_Trainer.get_model(self, pretrained)
        for domain_name in self.domain_list:
            self.pre_models_dict[domain_name] = copy.deepcopy(self.models_dict[domain_name])

    def get_log_name(self, args, log_num, file_name='train'):
        return MOON_Trainer.get_log_name(self, args, log_num, file_name)
    
    def step_train(self, site_name, model, imgs, labels):
        return MOON_Trainer.step_train(self, site_name, model, imgs, labels)
    
    def site_train(self, n_round, site_name, data_type='train', model=None):
        for local_epoch in range(self.local_epochs):
            epoch = n_round * self.local_epochs + local_epoch
            self.site_epoch_train(epoch, site_name, data_type, model)
            self.pre_models_dict[site_name] = copy.deepcopy(self.models_dict[site_name]) # 更新pre_local_model


class GRACE_MOON_Trainer(MOON_Trainer):
    def get_optimier(self, domain_name):
        if self.args.optimizer == 'adam':
            self.optimizers_dict[domain_name] = PerAvg_Adam(self.models_dict[domain_name].parameters(), lr=self.args.lr)
        else:
            self.optimizers_dict[domain_name] = PerAvgOptimizer(self.models_dict[domain_name].parameters(), lr=self.args.lr)

    def step_train(self, site_name, model, imgs, labels):
        output = model(imgs)
        loss = self.criterion(output, labels)
        return loss, output
    
    def step_align_train(self, site_name, model, imgs, labels):
        pre_local_model = self.pre_models_dict[site_name]
        global_model = self.global_model
        
        output, local_feature = model(imgs, feature_out=True)
        _, global_feature = global_model(imgs, feature_out=True)
        _, pre_local_feature = pre_local_model(imgs, feature_out=True)
        
        local_feature = local_feature.view(local_feature.shape[0], -1)
        global_feature = global_feature.view(global_feature.shape[0], -1)
        pre_local_feature = pre_local_feature.view(pre_local_feature.shape[0], -1)
        
        positive = self.cos(local_feature, global_feature)
        negative = self.cos(local_feature, pre_local_feature)
        logits = torch.cat([positive.reshape(-1, 1), negative.reshape(-1, 1)], dim=1) / self.temperature
        con_labels = torch.zeros(imgs.size(0), dtype=torch.long).to(self.device)
        
        sup_loss = self.criterion(output, labels)
        con_loss = self.con_criteria(logits, con_labels)
        
        loss = sup_loss + self.args.align_weight * con_loss
        return loss, output
    
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
            loss, output = self.step_align_train(site_name, model, imgs_2, labels_2)
            loss.backward()
            for old_param, new_param in zip(model.parameters(), temp_model):
                    old_param.data = new_param.data.clone()
            optimizer.step(beta=self.args.lr)
            
            self.log_ten.add_scalar(f'{site_name}_train_loss', loss.item(), epoch*len(dataloader)+i)
            self.metric.update(output, labels_2)
        
        self.log_ten.add_scalar(f'{site_name}_train_acc', self.metric.results()['acc'], epoch)
        scheduler.step()
        
    
class GRACE_MOON_Prostate_Trainer(MOON_Prostate_Trainer):
    def get_optimier(self, domain_name):
        if self.args.optimizer == 'adam':
            self.optimizers_dict[domain_name] = PerAvg_Adam(self.models_dict[domain_name].parameters(), lr=self.args.lr)
        else:
            self.optimizers_dict[domain_name] = PerAvgOptimizer(self.models_dict[domain_name].parameters(), lr=self.args.lr)

    def step_train(self, site_name, model, imgs, labels):
        output = model(imgs)
        loss = self.criterion(output, labels)
        return loss, output
    
    def step_align_train(self, site_name, model, imgs, labels):
        pre_local_model = self.pre_models_dict[site_name]
        global_model = self.global_model
        
        output, local_feature = model(imgs, feature_out=True)
        _, global_feature = global_model(imgs, feature_out=True)
        _, pre_local_feature = pre_local_model(imgs, feature_out=True)
        
        local_feature = local_feature.view(local_feature.shape[0], -1)
        global_feature = global_feature.view(global_feature.shape[0], -1)
        pre_local_feature = pre_local_feature.view(pre_local_feature.shape[0], -1)
        
        positive = self.cos(local_feature, global_feature)
        negative = self.cos(local_feature, pre_local_feature)
        logits = torch.cat([positive.reshape(-1, 1), negative.reshape(-1, 1)], dim=1) / self.temperature
        con_labels = torch.zeros(imgs.size(0), dtype=torch.long).to(self.device)
        
        sup_loss = self.criterion(output, labels)
        con_loss = self.con_criteria(logits, con_labels)
        
        loss = sup_loss + self.args.align_weight * con_loss
        return loss, output
    
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
            imgs_1, labels_1 = imgs[:len(imgs)//2], labels[:len(imgs)//2]
            imgs_2, labels_2 = imgs[len(imgs)//2:], labels[len(imgs)//2:]
            # step 1
            loss, output = self.step_train(site_name, model, imgs_1, labels_1)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            # step 2
            optimizer.zero_grad()
            loss, output = self.step_align_train(site_name, model, imgs_2, labels_2)
            loss.backward()
            for old_param, new_param in zip(model.parameters(), temp_model):
                    old_param.data = new_param.data.clone()
            optimizer.step(beta=self.args.lr)
            
            self.log_ten.add_scalar(f'{site_name}_train_loss', loss.item(), epoch*len(dataloader)+i)
            self.metric.update(output, labels_2)
        
        self.log_ten.add_scalar(f'{site_name}_train_acc', self.metric.results()['acc'], epoch)
        scheduler.step()


class GRACE_MOON_ISIC_BSM_Trainer(GRACE_MOON_Trainer):
    def get_class_count(self, domain_name):
        class_list = [0 for i in range(self.num_classes)]
        dataset = self.dataset_dict[domain_name]['train']
        for i in range(len(dataset)):
            single_output = dataset[i]
            label = single_output[1]
            class_list[label] += 1
        return class_list
    
    def initialize(self):
        super().initialize()
        self.global_criterion = {}
        for domain_name in self.train_domain_list:
            class_count_list = self.get_class_count(domain_name)
            print(domain_name, class_count_list)
            self.global_criterion[domain_name] = BalancedSoftmax(class_count_list).to(self.device)

    def step_train(self, site_name, model, imgs, labels):
        output = model(imgs)
        loss = self.global_criterion[site_name](output, labels)
        return loss, output



class GRACE_GM_MOON_Trainer(GRACE_MOON_Trainer, GRACE_Server_Trainer):
    def initialize(self):
        super().initialize()
        self.server_momentum_item = None
        if self.args.gm_type == 'total':
            self.calc_cos = self.func_total_cos
        elif self.args.gm_type == 'split':
            self.calc_cos = self.func_split_cos
        elif self.args.gm_type == 'reweight':
            self.calc_cos = self.func_reweight_cos
            
    def aggregation(self, model_dict=None, weight_dict=None, site_list=None):
        return GRACE_Server_Trainer.aggregation(self, model_dict, weight_dict, site_list)
    
class GRACE_GM_MOON_ISIC_BSM_Trainer(GRACE_MOON_ISIC_BSM_Trainer, GRACE_Server_Trainer):
    def initialize(self):
        super().initialize()
        self.server_momentum_item = None
        if self.args.gm_type == 'total':
            self.calc_cos = self.func_total_cos
        elif self.args.gm_type == 'split':
            self.calc_cos = self.func_split_cos
        elif self.args.gm_type == 'reweight':
            self.calc_cos = self.func_reweight_cos
            
    def aggregation(self, model_dict=None, weight_dict=None, site_list=None):
        return GRACE_Server_Trainer.aggregation(self, model_dict, weight_dict, site_list)    

class GRACE_GM_MOON_Prostate_Trainer(GRACE_MOON_Prostate_Trainer, GRACE_Server_Prostate_Trainer):
    def initialize(self):
        super().initialize()
        self.server_momentum_item = None
        if self.args.gm_type == 'total':
            self.calc_cos = self.func_total_cos
        elif self.args.gm_type == 'split':
            self.calc_cos = self.func_split_cos
        elif self.args.gm_type == 'reweight':
            self.calc_cos = self.func_reweight_cos
            
    def aggregation(self, model_dict=None, weight_dict=None, site_list=None):
        return GRACE_Server_Prostate_Trainer.aggregation(self, model_dict, weight_dict, site_list)

