import sys
sys.path.append(sys.path[0].replace('algorithms', ''))
from algorithms.fedavg import FedAvg_Trainer, FedAvg_Prostate_Trainer, Dict_Add, Dict_weight
from utils.classification_metric import Balance_Classification
import torch
import numpy as np

class GRACE_Server_Trainer(FedAvg_Trainer):
    def initialize(self):
        super().initialize()
        self.server_momentum_item = None
        if self.args.gm_type == 'total':
            self.calc_cos = self.func_total_cos 
        elif self.args.gm_type == 'split':
            self.calc_cos = self.func_split_cos
        elif self.args.gm_type == 'reweight':
            self.calc_cos = self.func_reweight_cos
            
    def get_log_name(self, args, log_num, file_name='train'):
        return super().get_log_name(args, log_num, file_name) + '_gm_' + args.gm_type  + '_slr_' + str(args.server_lr)
    
    def func_total_cos(self, grad1, grad2):
        cos = torch.nn.CosineSimilarity(dim=0, eps=1e-6)
        return cos(flatten_tensors(grad1), flatten_tensors(grad2)).detach().cpu().numpy()
    
    def func_split_cos(self, grad1, grad2):
        cos_list = get_cos_similarities(grad1, grad2)
        return np.mean(cos_list)
    
    def func_reweight_cos(self, grad1, grad2, reweight_list=None):
        cos_list = get_cos_similarities(grad1, grad2)
        return np.mean(reweight_cos_similarity(cos_list, reweight_list))
    
    def aggregation(self, model_dict=None, weight_dict=None, site_list=None):
        if weight_dict is None:
            weight_dict = self.weight_dict
        if site_list is None:
            site_list = self.train_domain_list
        if model_dict is None:
            model_dict = self.models_dict
        gradients_dict = {}
        for site_name in self.train_domain_list:
            gradients_dict[site_name] = get_gradient(self.global_model, model_dict[site_name]) # 输入需要是一个list

        new_weight_dict = {}
        cos_dict = {}
        for site_name in self.train_domain_list:
            avg_cos_sim = 0
            cos_sim_list = []
            for site_name2 in self.train_domain_list:
                if site_name == site_name2:
                    continue
                else:
                    cos_sim = self.calc_cos(gradients_dict[site_name], gradients_dict[site_name2])
                    avg_cos_sim += cos_sim
                    cos_sim_list.append(cos_sim)
            avg_cos_sim /= (len(self.train_domain_list) - 1)
            cos_dict[site_name] = avg_cos_sim
            
            self.log_ten.add_scalar(f'cos_sim_{site_name}', avg_cos_sim, self.current_round) # 记录cosine similarity
            self.log_file.info(f'{site_name} {np.mean(cos_sim_list):.4f} / {np.std(cos_sim_list):.4f} {cos_sim_list}')
        min_cos = min(cos_dict.values())
        if min_cos < 0:
            cos_dict = {k: v - min_cos + 1e-3 for k, v in cos_dict.items()} # 确保所有的都是大于1e-3的
        max_cos = max(cos_dict.values())
        if max_cos > 1:
            cos_dict = {k: v / max_cos for k, v in cos_dict.items()} # 确保都是小于1的
        
        
        for site_name in self.train_domain_list:
            cos_weight = cos_dict[site_name]
            new_weight_dict[site_name] = cos_weight*weight_dict[site_name]
        sum_weight = np.sum(list(new_weight_dict.values()))
        for site_name in self.train_domain_list:
            new_weight_dict[site_name] /= sum_weight  # 进行归一化操作
            
        self.log_file.info(f'Weight Dict: {weight_dict} | Cos Dict: {cos_dict} | New weight dict: {new_weight_dict}')
        
        self.update_global_model_with_gradient(gradients_dict, new_weight_dict)
          
        new_bn_dict = None
        for model_name in new_weight_dict.keys():
            model = model_dict[model_name]
            model_state_dict = get_bn_dict(model)
            if new_bn_dict is None:
                new_bn_dict = Dict_weight(model_state_dict, new_weight_dict[model_name])
            else:
                new_bn_dict = Dict_Add(new_bn_dict, Dict_weight(model_state_dict, new_weight_dict[model_name]))
        
        self.global_model.load_state_dict(new_bn_dict, strict=False)
        return self.global_model.state_dict()
    
    
    def update_global_model_with_gradient(self, gradient_dict, weight_dict=None):
        if weight_dict is None:
            weight_dict = self.weight_dict

        for i, params in enumerate(self.global_model.parameters()):
            avg_gradient = None
            for site_name in self.train_domain_list:
                avg_gradient = weight_dict[site_name]*gradient_dict[site_name][i] if avg_gradient is None else avg_gradient + weight_dict[site_name]*gradient_dict[site_name][i]
            params.data.sub_(self.args.server_lr * avg_gradient)

        return self.global_model.state_dict()


class GRACE_Server_ISIC_Trainer(GRACE_Server_Trainer):
    def initialize(self):
        super().initialize()
        self.metric = Balance_Classification()


class GRACE_Server_Prostate_Trainer(FedAvg_Prostate_Trainer, GRACE_Server_Trainer):
    def get_log_name(self, args, log_num, file_name='train'):
        return GRACE_Server_Trainer.get_log_name(self, args, log_num, file_name)
    
    def initialize(self):
        super().initialize()
        self.server_momentum_item = None
        if self.args.gm_type == 'total':
            self.calc_cos = self.func_total_cos # 计算cosine similarity的函数
        elif self.args.gm_type == 'split':
            self.calc_cos = self.func_split_cos
        elif self.args.gm_type == 'reweight':
            self.calc_cos = self.func_reweight_cos
            
    def aggregation(self, model_dict=None, weight_dict=None, site_list=None):
        return GRACE_Server_Trainer.aggregation(self, model_dict, weight_dict, site_list)

########## 对梯度的基本操作 ##########
def get_gradient(global_model, local_model):
    '''
    计算从global model出发到local model更新了的gradient
    global_model - local_model = gradient
    返回一个list
    '''
    list_global_params = list(global_model.parameters())
    list_local_params = list(local_model.parameters())
    gradient = []
    for global_params, local_params in zip(list_global_params, list_local_params):
        gradient.append(global_params - local_params)
    return gradient

def get_bn_dict(model):
    model_dict = model.state_dict()
    bn_dict = {}
    for key in model_dict.keys():
        if 'bn' in key:
            bn_dict[key] = model_dict[key]
    return bn_dict

def get_cos_similarities(grad_list1, grad_list2):
    '''分段进行cosine similarity的计算 返回的是numpy array的list'''
    cos_list = []
    cos = torch.nn.CosineSimilarity(dim=0, eps=1e-6)
    for grad1, grad2 in zip(grad_list1, grad_list2):
        cos_list.append(cos(flatten_tensors(grad1), flatten_tensors(grad2)).detach().cpu().numpy())
    return cos_list

def reweight_cos_similarity(cos_list, reweight_list=None):
    '''对cosine similarity进行reweight 默认采取前段重要后段不重要的方式 即对浅层的权重调整为1 深层的逐渐调整为0 可以考虑为非线性方式'''
    if reweight_list is None:
        reweight_list = np.linspace(1.,0.,len(cos_list))
    # 保持总和一致 检查reweight list的平均值是否为1
    avg_reweight = np.mean(reweight_list)
    for cos_sim, reweight in zip(cos_list, reweight_list):
        cos_sim *= (reweight/avg_reweight)
    return cos_list

def flatten_tensors(tensors):
    """
    Reference: https://github.com/facebookresearch/stochastic_gradient_push

    Flatten dense tensors into a contiguous 1D buffer. Assume tensors are of
    same dense type.
    Since inputs are dense, the resulting tensor will be a concatenated 1D
    buffer. Element-wise operation on this buffer will be equivalent to
    operating individually.
    Arguments:
        tensors (Iterable[Tensor]): dense tensors to flatten.
    Returns:
        A 1D buffer containing input tensors.
    """
    if len(tensors) == 1:
        return tensors[0].view(-1).clone()
    flat = torch.cat([t.view(-1) for t in tensors], dim=0)
    return flat


def unflatten_tensors(flat, tensors):
    """
    Reference: https://github.com/facebookresearch/stochastic_gradient_push

    View a flat buffer using the sizes of tensors. Assume that tensors are of
    same dense type, and that flat is given by flatten_dense_tensors.
    Arguments:
        flat (Tensor): flattened dense tensors to unflatten.
        tensors (Iterable[Tensor]): dense tensors whose sizes will be used to
            unflatten flat.
    Returns:
        Unflattened dense tensors with sizes same as tensors and values from
        flat.
    """
    outputs = []
    offset = 0
    for tensor in tensors:
        numel = tensor.numel()
        outputs.append(flat.narrow(0, offset, numel).view_as(tensor))
        offset += numel
    return tuple(outputs)






