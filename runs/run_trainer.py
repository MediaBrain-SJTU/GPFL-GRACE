import sys
sys.path.append(sys.path[0].replace('runs', ''))
import warnings
warnings.filterwarnings("ignore")
import argparse
import algorithms
import os
algorithm_names = sorted(name for name in algorithms.__dict__ if 'Trainer' in name and callable(algorithms.__dict__[name]))
import time
def get_args():
    parser = argparse.ArgumentParser()
    '''实验基础设定'''
    parser.add_argument("--algorithm", type=str, default='FedAvg_Trainer', choices=algorithm_names, help='Name of the algorithms')
    parser.add_argument("--dataset", type=str, default='pacs', help='Name of dataset')
    parser.add_argument("--model", type=str, default='resnet18', help='model name')
    parser.add_argument("--optimizer", type=str, default='sgd', help='optimizer name')
    parser.add_argument("--test_domain", type=str, default='p', help='the domain name for testing')
    parser.add_argument("--client_opt", type=str, default='sgd', choices=['sgd', 'adam'], help='the client optimization method')
    parser.add_argument("--server_opt", type=str, default='sgd', choices=['sgd', 'adam', 'adagrad', 'yogi'], help='the server optimization method')
    
    '''数据集相关'''
    parser.add_argument('--beta', help='the beta of dirichlet', type=float, default=0.5)
    parser.add_argument('--client_num', help='the number of split clients', type=int, default=10)
    
    parser.add_argument('--split_num', help='the number of split clients', type=int, default=10)
    parser.add_argument('--sample_num', help='the number of smapled clients during training', type=int, default=10)

    '''训练相关超参数'''
    parser.add_argument('--fa_aug_prob', help='The augmentation rate of FA', type=float, default=0.5)
    parser.add_argument('--mu', help='the mu of 0.1', type=float, default=0.1)
    parser.add_argument('--batch_size', help='batch_size', type=int, default=16)
    parser.add_argument('--local_epochs', help='epochs number', type=int, default=5)
    parser.add_argument('--adg_local_epochs', help='epochs number', type=int, default=7)
    parser.add_argument('--rounds', help='epochs number', type=int, default=40)
    parser.add_argument('--lr', help='learning rate', type=float, default=0.001)
    parser.add_argument('--lr_adv', help='learning rate for generator and discriminator', type=float, default=0.0007)
    parser.add_argument("--lr_policy", type=str, default='step', help="learning rate scheduler policy")
    parser.add_argument('--lr_warmup', help='learning rate warm up', action='store_true')
    parser.add_argument('--step_size', help='rate weight step', type=float, default=0.2)
    parser.add_argument('--step_decay', help='decay of step size', action='store_true')
    parser.add_argument("--fair", type=str, default='acc', choices=['acc', 'loss', 'cls_loss', 'sofr_loss'],
                        help="the fairness metric for FedAvg")
    parser.add_argument('--pretrain', help='load pretrain parameters for model', action='store_false')
    # FedSR相关
    parser.add_argument('--alpha_cmi', help='the loss weight of cmi', type=float, default=1e-3)
    parser.add_argument('--alpha_l2r', help='the loss weight of l2r', type=float, default=0.1)
    
    # FedVIb相关
    parser.add_argument('--alpha_vib', help='the loss weight of vib', type=float, default=0.1)
    parser.add_argument('--var', help='var', type=float, default=1.0)
    parser.add_argument('--var_ema', help='the momentunm of var', type=float, default=0.99)
    
    # FedSOFR相关
    parser.add_argument('--aug_type', help='augmentation type of SOFR', type=str, default='FA')
    parser.add_argument('--rampup', help='the round of rampup', type=int, default=10)

    # SWA相关
    parser.add_argument('--swa_c', help='the freq of swa', type=int, default=1)
    parser.add_argument('--swa_start', help='the start ratio of swa', type=float, default=None)
    
    # Fairness相关
    parser.add_argument('--local_converage_epochs', help='local converage epoch 在正式训练前local client上训练收敛所需要的epoch/FedAvg后FT需要的epoch数', type=int, default=30)
    
    # ARFL相关
    parser.add_argument('--arfl_gamma', help='the gamma of ARFL algorithm, 默认和数据集一样大', type=float, default=10000)
    
    # mean teacher
    parser.add_argument('--ema_momentum', help='the momentum weight of ema model', type=float, default=0.995)
    
    # HamoFL相关
    parser.add_argument('--hamo_alpha', help='HamoFL的优化扰动', type=float, default=0.05)

    # pFedMe 相关
    parser.add_argument('--lamda', help='pFedMe的约束项', type=float, default=15.0)
    parser.add_argument('--personal_lr', help='pFedMe的本地训练lr', type=float, default=1e-3)
    parser.add_argument('--K', help='Number of personalized training steps for pFedMe', type=int, default=5)
    parser.add_argument("--per_beta", type=float, default=1.0,
                        help="Average moving parameter for pFedMe, Second learning rate of Per-FedAvg, \
                        or L1 regularization weight of FedTransfer")
    
    # FedFOMO 相关
    parser.add_argument('--M', help='Number of send models for FedFOMO', type=int, default=5)
    
    # ELCFS 相关
    parser.add_argument('--clip_value', type=float,  default=100, help='maximum epoch number to train')
    parser.add_argument('--meta_step_size', type=float,  default=1e-3, help='maximum epoch number to train')
    
    # KD 相关
    parser.add_argument('--kd_temp', help='KD的蒸馏温度', type=float, default=1.0)
    parser.add_argument('--kd_warmup', help='global model作为teacher的warmup', type=int, default=10)
    parser.add_argument('--kd_weight', help='KD loss的权重', type=float, default=1.0)
    
    # fedalign 相关
    parser.add_argument('--align_weight', help='fedalign loss的权重', type=float, default=1.0)
    parser.add_argument('--align_warmup', help='fedalign的warmup', type=int, default=0)
    parser.add_argument('--align_type', help='fedalign的类型', type=str, default='CORAL')
    
    # gradient match 相关
    parser.add_argument('--gm_type', help='gradient match 计算cos similarity的方式', type=str, default='total')
    
    # TTDA type
    parser.add_argument('--ttda_type', help='TTDA type', type=str, default='tent', choices=['tent', 'dsbn', 'ttt'])
    
    # server momentum & 梯度更新
    parser.add_argument('--server_momentum', help='server momentum', type=float, default=0.9)
    parser.add_argument('--server_lr', help='server learning rate', type=float, default=1.0)
    
    # MixFeature相关
    parser.add_argument('--mix_type', help='mix feature tyle', type=str, default='mixstyle', choices=['mixstyle', 'dsu', 'csu'])
    parser.add_argument('--self_mix', help='self mix', action='store_true') # 如果使用self mix 则不使用global model的特征统计量作为mix的基础
    
    # Prostate Unet结构相关
    parser.add_argument('--unet_norm', help='unet norm', type=str, default='no', choices=['bn', 'in', 'no'])
    
    
    '''实验结果记录'''
    parser.add_argument('--note', help='note of experimental settings', type=str, default='fedavg')
    parser.add_argument('--display', help='display in controller', action='store_true') # 默认false 即不展示
    return parser.parse_args()


def main():
    args = get_args()
    trainer = getattr(algorithms, args.algorithm)(args)
    trainer.run()
    time.sleep(60)

if __name__ == '__main__':
    main()