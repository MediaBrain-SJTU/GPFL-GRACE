'''
2022.1.9
所有和DG相关的数据集 dataset dataloader之类的处理汇总
'''
from .isic2019_dataset import Isic2019_FedDG, ISIC_BaselineLoss
from .prostate_dataset import Prostate_FedDG, JointLoss, DiceLoss
