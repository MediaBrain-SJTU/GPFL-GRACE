import sys
sys.path.append(sys.path[0].replace('algorithms', ''))
import torch
from algorithms.fedavg import FedAvg_Trainer, FedAvg_Prostate_Trainer, FedAvg_Fundus_Trainer
from utils.classification_metric import  Balance_Classification
import torch.nn.functional as F
from torch.nn.modules.loss import _Loss
import copy
from networks.get_network import GetNetwork

class BalancedSoftmax(_Loss):
    """
    Balanced Softmax Loss
    """
    def __init__(self, sample_per_class):
        super(BalancedSoftmax, self).__init__()
        sample_per_class = torch.tensor(sample_per_class)
        self.sample_per_class = sample_per_class

    def forward(self, input, label, reduction='mean'):
        return balanced_softmax_loss(label, input, self.sample_per_class, reduction)
        
def balanced_softmax_loss(labels, logits, sample_per_class, reduction):
    """Compute the Balanced Softmax Loss between `logits` and the ground truth `labels`.
    Args:
      labels: A int tensor of size [batch].
      logits: A float tensor of size [batch, no_of_classes].
      sample_per_class: A int tensor of size [no of classes].
      reduction: string. One of "none", "mean", "sum"
    Returns:
      loss: A float tensor. Balanced Softmax Loss.
    """
    spc = sample_per_class.type_as(logits)
    spc = spc.unsqueeze(0).expand(logits.shape[0], -1)
    logits = logits + spc.log()
    loss = F.cross_entropy(input=logits, target=labels, reduction=reduction)
    return loss   

class FedRoD_OnlyHead_Trainer(FedAvg_Trainer):
    def get_load_head(self, domain_name):
        if self.args.model == 'isic_b0':
            self.local_heads_dict[domain_name] = copy.deepcopy(self.models_dict[domain_name].base_model._fc)
        else:
            self.local_heads_dict[domain_name] = copy.deepcopy(self.models_dict[domain_name].fc_class)
    
    def get_model(self, pretrained=True):
        super().get_model(pretrained)
        self.local_heads_dict = {}
        for domain_name in self.train_domain_list:
            self.get_load_head(domain_name)
            self.optimizers_dict[domain_name] = torch.optim.SGD(
                [{'params':self.models_dict[domain_name].parameters()},
                 {'params':self.local_heads_dict[domain_name].parameters()}],
                lr=self.args.lr, momentum=0.9, weight_decay=5e-4)

    def step_train(self, site_name, model, imgs, labels):
        output, x_feature = model(imgs, feature_out=True)
        local_output = self.local_heads_dict[site_name](x_feature)
        global_loss = self.criterion(output, labels)
        local_loss = self.criterion(local_output+output, labels)
        loss = global_loss + local_loss
        return loss, output

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
                if 'global' in prefix:
                    output = model(imgs)
                    if isinstance(output, tuple):
                        output = output[0]
                else:
                    output_global, x_feature = model(imgs, feature_out=True)
                    output_local = self.local_heads_dict[site_name](x_feature)
                    if isinstance(output_local, tuple):
                        output_local = output_local[0]
                    output = output_global + output_local
                
                self.metric.update(output, labels)
        results_dict = self.metric.results()
        self.log_ten.add_scalar(f'{prefix}_{site_name}_loss', results_dict['loss'], n_round)
        self.log_ten.add_scalar(f'{prefix}_{site_name}_acc', results_dict['acc'], n_round)
        self.log_file.info(f'{prefix} Round: {n_round:3d} | Domain: {site_name} | loss: {results_dict["loss"]:.4f} | Acc: {results_dict["acc"]*100:.2f}%')
        return results_dict

 

class FedRoD_Trainer(FedRoD_OnlyHead_Trainer):
    def get_class_count(self, domain_name):
        class_list = [0 for i in range(self.num_classes)]
        dataset = self.dataset_dict[domain_name]['train']
        for i in range(len(dataset)):
            single_output = dataset[i]
            label = single_output[1]
            class_list[label] += 1
        return class_list
    
    def get_model(self, pretrained=True):
        super().get_model(pretrained)
        self.global_criterion = {}
        for domain_name in self.train_domain_list:
            class_count_list = self.get_class_count(domain_name)
            print(domain_name, class_count_list)
            self.global_criterion[domain_name] = BalancedSoftmax(class_count_list).to(self.device)
        
    def step_train(self, site_name, model, imgs, labels):
        output, x_feature = model(imgs, feature_out=True)
        local_output = self.local_heads_dict[site_name](x_feature)
        global_loss = self.global_criterion[site_name](output, labels)
        local_loss = self.criterion(local_output+output, labels)
        loss = global_loss + local_loss
        return loss, output



class FedRoD_ISIC_Trainer(FedRoD_Trainer):
    def initialize(self):
        super().initialize()
        self.metric = Balance_Classification()
        

class FedRoD_Prostate_Trainer(FedAvg_Prostate_Trainer):
    def get_load_head(self, domain_name):
        if self.args.model == 'prostate_unet':
            self.local_heads_dict[domain_name] = copy.deepcopy(self.models_dict[domain_name].conv)
        else:
            self.local_heads_dict[domain_name] = copy.deepcopy(self.models_dict[domain_name].fc_class)
    
    def get_model(self, pretrained=True):
        self.local_heads_dict = {}
        self.global_model, self.feature_level = GetNetwork(self.args, self.num_classes, pretrained)
        self.global_model.to(self.device)
        self.get_criterion()
        
        for domain_name in self.domain_list:
            self.models_dict[domain_name], _ = GetNetwork(self.args, self.num_classes, pretrained)
            self.models_dict[domain_name].to(self.device)
            self.get_load_head(domain_name)
            self.get_optimier(domain_name)
            self.get_scheduler(domain_name)
            
    
    def get_optimier(self, domain_name):
        if self.args.optimizer == 'adam':
            self.optimizers_dict[domain_name] = torch.optim.Adam(
                [{'params':self.models_dict[domain_name].parameters()},
                 {'params':self.local_heads_dict[domain_name].parameters()}],
                lr=self.args.lr)
        else:
            self.optimizers_dict[domain_name] = torch.optim.SGD(
                [{'params':self.models_dict[domain_name].parameters()},
                 {'params':self.local_heads_dict[domain_name].parameters()}],
                lr=self.args.lr, momentum=0.9, weight_decay=5e-4)
            
    def step_train(self, site_name, model, imgs, labels):
        output, x_feature = model(imgs, feature_out=True)
        local_output = self.local_heads_dict[site_name](x_feature)
        global_loss = self.criterion(output, labels)
        local_loss = self.criterion((local_output+output)/2., labels)
        loss = global_loss + local_loss
        return loss, output

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
                imgs, labels = imgs.to(self.device), labels.to(self.device)
                if 'global' in prefix: # 只测试global model
                    output = model(imgs)
                    if isinstance(output, tuple):
                        output = output[0]
                else:
                    output_global, x_feature = model(imgs, feature_out=True)
                    output_local = self.local_heads_dict[site_name](x_feature)
                    if isinstance(output_local, tuple):
                        output_local = output_local[0]
                    output = output_global + output_local
                
                self.metric.update(output, labels)
        results_dict = self.metric.results()
        self.log_ten.add_scalar(f'{prefix}_{site_name}_loss', results_dict['loss'], n_round)
        self.log_ten.add_scalar(f'{prefix}_{site_name}_dice', results_dict['dice'], n_round)
        self.log_file.info(f'{prefix} Round: {n_round:3d} | Domain: {site_name} | loss: {results_dict["loss"]:.4f} | Dice: {results_dict["dice"]*100:.2f}%')
        return results_dict


