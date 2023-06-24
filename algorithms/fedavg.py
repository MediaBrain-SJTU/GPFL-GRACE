import sys
sys.path.append(sys.path[0].replace('algorithms', ''))
import os
import torch
import time
import shutil
import numpy as np
from torch.utils.tensorboard.writer import SummaryWriter
from scipy.stats import pearsonr
from utils.classification_metric import Classification, Balance_Classification, Segmentation2D
from utils.log_utils import Get_Logger
from algorithms.meta_trainer import MetaTrainer
from data import *
from networks.get_network import GetNetwork
from configs.default import log_count_path
import pytorch_warmup as warmup

def Dict_weight(dict_in, weight_in):
    for k,v in dict_in.items():
        dict_in[k] = weight_in*v
    return dict_in
    
def Dict_Add(dict1, dict2):
    for k,v in dict1.items():
        dict1[k] = v + dict2[k]
    return dict1

def Dict_Minus(dict1, dict2):
    for k,v in dict1.items():
        dict1[k] = v - dict2[k]
    return dict1

def Dict_multiplay(dict1, dict2):
    for k,v in dict1.items():
        dict1[k] = v * dict2[k]
    return dict1


def weight_clip(weight_dict):
    new_total_weight = 0.0
    max_weight = 2./len(weight_dict)
    for key_name in weight_dict.keys():
        weight_dict[key_name] = np.clip(weight_dict[key_name], 0.0, max_weight)
        new_total_weight += weight_dict[key_name]
    
    for key_name in weight_dict.keys():
        weight_dict[key_name] /= new_total_weight
    
    return weight_dict


class FedAvg_Trainer(MetaTrainer):
    def __init__(self, args) -> None:
        super().__init__(args)
        self.initialize()
        self.log_file.info(self.args)
        self.log_file.info(f'{self.trainer_name} initialized')
    
    def get_log_name(self, args, log_num, file_name='train'):
        start_time = time.strftime('%Y-%m-%d-%H-%M',time.localtime())
        log_name = f"{start_time}-{file_name}-{args.dataset}-{args.model}-{args.test_domain}"\
            + f"-{args.lr}-bs{args.batch_size}-r{args.rounds}-l{args.local_epochs}"\
            +f"-{args.note}"
        return log_name
    
    def get_log_dir(self, args, file_name='train', tensorboard_subdir=True, log_path=log_count_path):
        log_num = 0
        log_name = self.get_log_name(args=args, log_num=log_num, file_name=file_name)
        log_dir = log_path + log_name + '/'
        os.makedirs(log_dir)
        if tensorboard_subdir:
            tensorboard_dir = log_dir + '/tensorboard/'
            os.makedirs(tensorboard_dir)
            return log_dir, tensorboard_dir
        else:
            return log_dir
    
    def get_log(self, file_name=None):
        if file_name is None:
            file_name = self.trainer_name
        log_dir, tensorboard_dir = self.get_log_dir(self.args, file_name=file_name)
        self.log_dir = log_dir
        self.save_dir = log_dir + 'checkpoints/'
        os.makedirs(self.save_dir)
        self.tensorboard_dir = tensorboard_dir
        self.log_ten = SummaryWriter(log_dir=tensorboard_dir)
        self.log_file = Get_Logger(file_name=log_dir + 'train.log', display=self.args.display)
       
    def get_data_obj(self):
        if self.args.dataset == 'isic':
            self.data_obj = Isic2019_FedDG(test_domain=self.args.test_domain, batch_size=self.args.batch_size)
            self.num_classes = 8
            self.domain_list = self.data_obj.domain_list.copy()
            
        elif self.args.dataset == 'prostate':
            self.data_obj = Prostate_FedDG(test_domain=self.args.test_domain, batch_size=self.args.batch_size)
            self.num_classes = 2
            self.domain_list = self.data_obj.domain_list.copy()
        
        
    def get_criterion(self):
        if self.args.dataset == 'isic':
            self.criterion = ISIC_BaselineLoss().to(self.device)
        elif self.args.dataset == 'prostate':
            self.criterion = JointLoss().to(self.device)
        else:
            self.criterion = torch.nn.CrossEntropyLoss().to(self.device)
    
    def get_metric(self):
        if self.args.dataset == 'isic':
            self.metric = Balance_Classification()
        elif self.args.dataset == 'prostate':
            self.metric = Segmentation2D()
        else:
            self.metric = Classification()
            
    def get_optimier(self, domain_name):
        if self.args.optimizer == 'sgd':
            self.optimizers_dict[domain_name] = torch.optim.SGD(self.models_dict[domain_name].parameters(), lr=self.args.lr, momentum=0.9, weight_decay=5e-4)
        elif self.args.optimizer == 'adamw':
            self.optimizers_dict[domain_name] = torch.optim.AdamW(self.models_dict[domain_name].parameters(), lr=self.args.lr) # 默认是1e-3
        elif self.args.optimizer == 'adam':
            self.optimizers_dict[domain_name] = torch.optim.Adam(self.models_dict[domain_name].parameters(), lr=self.args.lr) # isic默认是5e-4 bs64
        elif self.args.optimizer == 'adam_amsgrad':
            self.optimizers_dict[domain_name] = torch.optim.Adam(self.models_dict[domain_name].parameters(), lr=self.args.lr, weight_decay=3e-5, amsgrad=True)

    def get_scheduler(self, domain_name):
        if self.args.lr_policy == 'step':
            self.schedulers_dict[domain_name] = torch.optim.lr_scheduler.StepLR(self.optimizers_dict[domain_name], step_size=self.args.local_epochs * self.args.rounds, gamma=0.1)
        elif self.args.lr_policy == 'step_ixi':
            self.schedulers_dict[domain_name] = torch.optim.lr_scheduler.StepLR(self.optimizers_dict[domain_name], step_size=10, gamma=0.95, last_epoch=-1)
        elif self.args.lr_policy == 'step_isic':
            milestones=[3, 5, 7, 9, 11, 13, 15, 17]
            fed_milestones = [int(i * self.args.rounds) / 20 for i in milestones]
            self.schedulers_dict[domain_name] = torch.optim.lr_scheduler.MultiStepLR(self.optimizers_dict[domain_name], milestones=fed_milestones, gamma=0.5)    
        elif self.args.lr_policy == 'reduce_kits19':
            self.schedulers_dict[domain_name] = torch.optim.lr_scheduler.ReduceLROnPlateau(self.optimizers_dict[domain_name], mode='min', factor=0.2, patience=30, verbose=True, threshold=1e-3, threshold_mode='abs')
        elif self.args.lr_policy == 'cos':
            self.schedulers_dict[domain_name] = torch.optim.lr_scheduler.CosineAnnealingLR(self.optimizers_dict[domain_name], T_max=self.args.local_epochs * self.args.rounds, eta_min=0)
        
        if self.args.lr_warmup:
            self.warmup_schedulers_dict[domain_name] = warmup.LinearWarmup(self.optimizers_dict[domain_name], warmup_period=10*self.args.local_epochs)
            
            
    def get_model(self, pretrained=True):
        self.global_model, self.feature_level = GetNetwork(self.args, self.num_classes, pretrained)
        self.global_model.to(self.device)
        self.get_criterion()
        
        for domain_name in self.domain_list:
            self.models_dict[domain_name], _ = GetNetwork(self.args, self.num_classes, pretrained)
            self.models_dict[domain_name].to(self.device)
            
            self.get_optimier(domain_name)
            self.get_scheduler(domain_name)
            
            
    def Cal_Weight_Dict(self, site_list=None):
        dataset_dict = self.dataset_dict
        self.domain_count = {}
        if site_list is None:
            site_list = self.train_domain_list
        if site_list is None:
            site_list = list(dataset_dict.keys())
        weight_dict = {}
        total_len = 0
        if 'cifar' in self.args.dataset:
            for site_name in site_list:
                weight_dict[site_name] = 1./len(site_list)
            return weight_dict
        for site_name in site_list:
            total_len += len(dataset_dict[site_name]['train'])
        for site_name in site_list:
            site_len = len(dataset_dict[site_name]['train'])
            self.domain_count[site_name] = site_len
            weight_dict[site_name] = site_len/total_len
        return weight_dict
    
    def initialize(self):
        self.metric = Classification()
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.test_domain = self.args.test_domain
        self.local_epochs = self.args.local_epochs
        self.rounds = self.args.rounds
        self.best_acc = 0.
        self.trainer_name = self.__class__.__name__
        
        self.get_log()
        
        
        self.get_data_obj()
        self.dataloader_dict, self.dataset_dict = self.data_obj.GetData()
        
        self.weight_dict = self.Cal_Weight_Dict(site_list=self.data_obj.train_domain_list)
        self.train_domain_list = self.data_obj.train_domain_list
       
        self.get_model(self.args.pretrain)
        self.get_metric()
        
    def aggregation(self, model_dict=None, weight_dict=None, site_list=None):
        if weight_dict is None:
            weight_dict = self.weight_dict
        if site_list is None:
            site_list = self.train_domain_list
        if model_dict is None:
            model_dict = self.models_dict
            
        new_model_dict = None
        
        for model_name in weight_dict.keys():
            model = model_dict[model_name]
            model_state_dict = model.state_dict()
            if new_model_dict is None:
                new_model_dict = Dict_weight(model_state_dict, weight_dict[model_name])
            else:
                new_model_dict = Dict_Add(new_model_dict, Dict_weight(model_state_dict, weight_dict[model_name]))
        
        self.global_model.load_state_dict(new_model_dict)
        return new_model_dict
    
    def broadcast(self, models_dict=None, site_list=None, param_dict=None):
        if models_dict is None:
            models_dict = self.models_dict
        if site_list is None:
            site_list = self.domain_list
        if param_dict is None:
            param_dict = self.global_model.state_dict()
        for site_name in site_list:
            models_dict[site_name].load_state_dict(param_dict)
        return models_dict
        
    def get_val_acc(self, results_dict, item_name='acc'):
        val_acc = 0.
        for domain_name in self.train_domain_list:
            val_acc += self.weight_dict[domain_name] * results_dict[domain_name][item_name]
        return val_acc
    
    def save_checkpoint(self, n_round, model, results_dict, best_acc, save_dir, is_best=False, prefix=''):
        state = {
        'args': self.args,
        'n_round': n_round,
        'model': model.state_dict(),
        'results': results_dict,
        'best_acc': best_acc,
        'weight_dict': self.weight_dict,
        }
        torch.save(state, save_dir + prefix + 'last_checkpoint.pth')
        if is_best:
            shutil.copyfile(save_dir + prefix + 'last_checkpoint.pth', save_dir + prefix + 'model_best.pth')
    
    def step_train(self, site_name, model, imgs, labels):
        output = model(imgs)
        loss = self.criterion(output, labels)
        return loss, output
    
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
            
            loss, output = self.step_train(site_name, model, imgs, labels)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            self.log_ten.add_scalar(f'{site_name}_train_loss', loss.item(), epoch*len(dataloader)+i)
            self.metric.update(output, labels)
        
        self.log_ten.add_scalar(f'{site_name}_train_acc', self.metric.results()['acc'], epoch)
        if self.args.lr_warmup:
            with self.warmup_schedulers_dict[site_name].dampening():
                scheduler.step()
        scheduler.step()
    
    def site_train(self, n_round, site_name, data_type='train', model=None):
        for local_epoch in range(self.local_epochs):
            epoch = n_round * self.local_epochs + local_epoch
            self.site_epoch_train(epoch, site_name, data_type, model)
            
    def train(self, n_round, data_type='train', model=None):
        for site_name in self.train_domain_list:
            self.site_train(n_round, site_name, data_type, model)

    def site_evaluation(self, n_round, site_name, data_type='val', model=None, prefix='after_fed'):
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
                output = model(imgs)
                if isinstance(output, tuple):
                    output = output[0]
                self.metric.update(output, labels)
        results_dict = self.metric.results()
        self.log_ten.add_scalar(f'{prefix}_{site_name}_loss', results_dict['loss'], n_round)
        self.log_ten.add_scalar(f'{prefix}_{site_name}_acc', results_dict['acc'], n_round)
        self.log_file.info(f'{prefix} Round: {n_round:3d} | Domain: {site_name} | loss: {results_dict["loss"]:.4f} | Acc: {results_dict["acc"]*100:.2f}%')

        return results_dict
    
    def val(self, n_round, is_global=False, is_test=False, data_type='val'):
        results_dict = {}
        for site_name in self.train_domain_list:
            if is_global:
                results_dict[site_name] = self.site_evaluation(n_round, site_name, data_type='val', model=self.global_model, prefix='global_val')
            else:
                results_dict[site_name] = self.site_evaluation(n_round, site_name, data_type='val', model=self.models_dict[site_name], prefix='local_val')

        val_acc = self.get_val_acc(results_dict)
        results_dict['avg_val_acc'] = val_acc
        self.log_file.info(f'Round: {n_round:3d} | {"global" if is_global else "local"} Val Acc: {val_acc*100:.2f}%')
        
        if is_test:
            results_dict[self.test_domain] = self.site_evaluation(n_round, self.test_domain, data_type='test', model=self.global_model, prefix='global_test')
        
        return results_dict
    
    def get_p_fairness(self, local_results, global_results):
        x = []
        y = []
        for domain_name in self.train_domain_list:
            x.append(local_results[domain_name]['acc'])
            y.append(global_results[domain_name]['acc'])
        return pearsonr(x, y)[0]
    
    def run(self):
        for i in range(self.rounds):
            self.current_round = i
            self.broadcast()
            
            self.train(i)
            local_results = self.val(i, is_global=False, is_test=False)
            
            # fedavg
            self.aggregation(self.models_dict, self.weight_dict, self.train_domain_list)
            
            global_results = self.val(i, is_global=True, is_test=True)
            val_acc = global_results['avg_val_acc']
            
            # save checkpoint
            is_best = val_acc > self.best_acc
            self.best_acc = max(val_acc, self.best_acc)
            self.save_checkpoint(i, self.global_model, {'global':global_results, 'local':local_results}, global_results[self.test_domain]['acc'], self.save_dir, is_best, prefix='global_')

            p_corr = self.get_p_fairness(local_results, global_results)
            self.log_file.info(f'Round: {i:3d} | p_corr: {p_corr:.4f}')
            self.log_ten.add_scalar('p_corr', p_corr, i)
            if is_best:
                self.log_file.info(f'Round {i} Get Best Val Acc: {self.best_acc*100.:.2f}%')


class FedAvg_Seg2D_Trainer(FedAvg_Trainer):
    def get_p_fairness(self, local_results, global_results):
        x = []
        y = []
        for domain_name in self.train_domain_list:
            x.append(local_results[domain_name]['dice'])
            y.append(global_results[domain_name]['dice'])
        return pearsonr(x, y)[0]
    
    def site_evaluation(self, n_round, site_name, data_type='val', model=None, prefix='after_fed'):
        if model is None:
            model = self.models_dict[site_name]
        model.eval()
        dataloader = self.dataloader_dict[site_name][data_type]
        total_loss = .0
        with torch.no_grad():
            for i, data_list in enumerate(dataloader):
                if len(data_list) == 3:
                    imgs, labels, _ = data_list
                else:
                    imgs, labels = data_list
                    
                imgs, labels = imgs.to(self.device), labels.to(self.device)
                loss, output = self.step_train(site_name, model, imgs, labels)
                total_loss += loss.item()
                if isinstance(output, tuple):
                    output = output[0]
                self.metric.update(output, labels)
        
        total_loss /= len(dataloader)
        
        results_dict = self.metric.results()
        self.log_ten.add_scalar(f'{prefix}_{site_name}_loss', results_dict['loss'], n_round)
        self.log_ten.add_scalar(f'{prefix}_{site_name}_dice', results_dict['dice'], n_round)
        self.log_file.info(f'{prefix} Round: {n_round:3d} | Domain: {site_name} | loss: {results_dict["loss"]:.4f} | Dice: {results_dict["dice"]*100:.2f}%')

        return results_dict
    
    def val(self, n_round, is_global=False, is_test=False, data_type='val'):
        results_dict = {}
        for site_name in self.train_domain_list:
            if is_global:
                results_dict[site_name] = self.site_evaluation(n_round, site_name, data_type=data_type, model=self.global_model, prefix=f'global_{data_type}')
            else:
                results_dict[site_name] = self.site_evaluation(n_round, site_name, data_type=data_type, model=self.models_dict[site_name], prefix=f'local_{data_type}')
        
        val_acc = self.get_val_acc(results_dict, item_name='dice')
        results_dict['avg_val_dice'] = val_acc
        self.log_file.info(f'Round: {n_round:3d} | {"global" if is_global else "local"} Val Dice: {val_acc*100:.2f}%')
        
        if is_test:
            results_dict[self.test_domain] = self.site_evaluation(n_round, self.test_domain, data_type='test', model=self.global_model, prefix='global_test')
        
        return results_dict
    
    def run(self):
        for i in range(self.rounds):
            self.current_round = i
            self.broadcast() 
            
            self.train(i)
            local_results = self.val(i, is_global=False, is_test=False)
            
            # fedavg
            self.aggregation(self.models_dict, self.weight_dict, self.train_domain_list)
            
            global_results = self.val(i, is_global=True, is_test=True)
            val_acc = global_results['avg_val_dice']
            
            # save checkpoint
            is_best = val_acc > self.best_acc
            self.best_acc = max(val_acc, self.best_acc)
            self.save_checkpoint(i, self.global_model, {'global':global_results, 'local':local_results}, global_results[self.test_domain]['dice'], self.save_dir, is_best, prefix='global_')

            p_corr = self.get_p_fairness(local_results, global_results)
            self.log_file.info(f'Round: {i:3d} | p_corr: {p_corr:.4f}')
            self.log_ten.add_scalar('p_corr', p_corr, i)
            if is_best:
                self.log_file.info(f'Round {i} Get Best Val Dice: {self.best_acc*100.:.2f}%')

class FedAvg_Prostate_Trainer(FedAvg_Seg2D_Trainer):
    def get_val_acc(self, results_dict, item_name='acc'):
        val_acc = 0.
        for domain_name in self.train_domain_list:
            val_acc += results_dict[domain_name][item_name]
        return val_acc/len(self.train_domain_list)
    
    def val(self, n_round, is_global=False, is_test=False, data_type='val'):
        results_dict = {}
        for site_name in self.train_domain_list:
            if is_global:
                results_dict[site_name] = self.site_evaluation(n_round, site_name, data_type=data_type, model=self.global_model, prefix=f'global_{data_type}')
            else:
                results_dict[site_name] = self.site_evaluation(n_round, site_name, data_type=data_type, model=self.models_dict[site_name], prefix=f'local_{data_type}')
        
        val_acc = self.get_val_acc(results_dict, item_name='dice')
        results_dict['avg_val_dice'] = val_acc
        self.log_file.info(f'Round: {n_round:3d} | {"global" if is_global else "local"} Val Dice: {val_acc*100:.2f}%')
        
        if is_test:
            results_dict[self.test_domain] = self.site_evaluation(n_round, self.test_domain, data_type='total', model=self.global_model, prefix='global_total')

        return results_dict


class FedAvg_ISIC_Trainer(FedAvg_Trainer):
    def initialize(self):
        super().initialize()
        self.metric = Balance_Classification()


