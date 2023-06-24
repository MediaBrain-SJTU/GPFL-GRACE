
import sys
sys.path.append(sys.path[0].replace('algorithms', ''))
from utils.classification_metric import Balance_Classification, Segmentation2D
from algorithms.fedavg import FedAvg_Prostate_Trainer, FedAvg_Trainer, Dict_weight, Dict_Add
from networks.FedOptimizer.PerFedMe import pFedMeOptimizer, pFedMeAdam
import copy

class PerFedMe_Trainer(FedAvg_Trainer):
    def get_optimier(self, domain_name):
        if self.args.optimizer == 'adam':
            self.optimizers_dict[domain_name] = pFedMeAdam(self.models_dict[domain_name].parameters(), lr=self.args.personal_lr, lamda=self.args.lamda)
        else:
            self.optimizers_dict[domain_name] = pFedMeOptimizer(self.models_dict[domain_name].parameters(), lr=self.args.personal_lr, lamda=self.args.lamda)

    def get_log_name(self, args, log_num, file_name='train'):
        return super().get_log_name(args, log_num, file_name) + f'_lamda{args.lamda}_p_lr{args.personal_lr}_K{args.K}'
    
    def initialize(self):
        super().initialize()
        self.local_params_dict = {}
        self.personalized_params_dict = {}
        
        self.lamda = self.args.lamda
        self.lr = self.args.lr
        self.p_lr = self.args.personal_lr # 0.1 
        self.beta = self.args.per_beta # 1.0 0.1 4.0 
        
        for domain_name in self.train_domain_list:
            self.local_params_dict[domain_name] = copy.deepcopy(list(self.models_dict[domain_name].parameters())) # 已经在cuda上了
            self.personalized_params_dict[domain_name] = copy.deepcopy(list(self.models_dict[domain_name].parameters()))
    
    def site_train(self, n_round, site_name, data_type='train', model=None):
        for local_epoch in range(self.local_epochs):
            epoch = n_round * self.local_epochs + local_epoch
            self.site_epoch_train(epoch, site_name, data_type, model)
        if model == None:
            model = self.models_dict[site_name]
        self.update_parameters(model, self.local_params_dict[site_name])
    
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
            
            for j in range(self.args.K):
                loss, output = self.step_train(site_name, model, imgs, labels)
                optimizer.zero_grad()
                loss.backward()
                self.personalized_params_dict[site_name], _ = optimizer.step(self.local_params_dict[site_name], self.device)
            
            for new_param, localweight in zip(self.personalized_params_dict[site_name], self.local_params_dict[site_name]):
                    localweight.data = localweight.data - self.lamda * self.lr * (localweight.data - new_param.data)
                    
            self.log_ten.add_scalar(f'{site_name}_train_loss', loss.item(), epoch*len(dataloader)+i)
            self.metric.update(output, labels)
            
        
        self.log_ten.add_scalar(f'{site_name}_train_acc', self.metric.results()['acc'], epoch)
        scheduler.step()
        
    
    def update_parameters(self, model, new_params):
        for param, new_param in zip(model.parameters(), new_params):
            param.data = new_param.data.clone()
    
    def aggregation(self, model_dict=None, weight_dict=None, site_list=None):
        '''参数聚合'''
        if weight_dict is None:
            weight_dict = self.weight_dict
        if site_list is None:
            site_list = self.train_domain_list
        if model_dict is None:
            model_dict = self.models_dict
            
        new_model_dict = None
        pre_global_model = copy.deepcopy(list(self.global_model.parameters()))
        for model_name in weight_dict.keys():
            model = model_dict[model_name]
            model_state_dict = model.state_dict()
            if new_model_dict is None:
                # 第一个节点
                new_model_dict = Dict_weight(model_state_dict, weight_dict[model_name])
            else:
                # 其他节点
                new_model_dict = Dict_Add(new_model_dict, Dict_weight(model_state_dict, weight_dict[model_name]))
        self.global_model.load_state_dict(new_model_dict) #
        for pre_param, param in zip(pre_global_model, self.global_model.parameters()):
            param.data = (1 - self.beta)*pre_param.data + self.beta*param.data
        
        return new_model_dict
            
class PerFedMe_ISIC_Trainer(PerFedMe_Trainer):
    def initialize(self):
        super().initialize()
        self.metric = Balance_Classification()
        
        
class PerFedMe_Prostate_Trainer(FedAvg_Prostate_Trainer, PerFedMe_Trainer):
    def initialize(self):
        PerFedMe_Trainer.initialize(self)
        self.metric = Segmentation2D()
        
    def get_optimier(self, domain_name):
        return PerFedMe_Trainer.get_optimier(self, domain_name)
    
    def get_log_name(self, args, log_num, file_name='train'):
        return PerFedMe_Trainer.get_log_name(self, args, log_num, file_name)
    
    def update_parameters(self, model, new_params):
        return PerFedMe_Trainer.update_parameters(self, model, new_params)
    
    def aggregation(self, model_dict=None, weight_dict=None, site_list=None):
        return PerFedMe_Trainer.aggregation(self, model_dict, weight_dict, site_list)
    
    def site_train(self, n_round, site_name, data_type='train', model=None):
        return PerFedMe_Trainer.site_train(self, n_round, site_name, data_type, model)
    
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
            
            # 本地重复多次训练
            for j in range(self.args.K):
                loss, output = self.step_train(site_name, model, imgs, labels)
                optimizer.zero_grad()
                loss.backward()
                self.personalized_params_dict[site_name], _ = optimizer.step(self.local_params_dict[site_name], self.device)
        
            for new_param, localweight in zip(self.personalized_params_dict[site_name], self.local_params_dict[site_name]):
                    localweight.data = localweight.data - self.lamda * self.lr * (localweight.data - new_param.data)
                    
            self.log_ten.add_scalar(f'{site_name}_train_loss', loss.item(), epoch*len(dataloader)+i)
            self.metric.update(output, labels)
        
        self.log_ten.add_scalar(f'{site_name}_train_dice', self.metric.results()['dice'], epoch)
        scheduler.step()
        
        