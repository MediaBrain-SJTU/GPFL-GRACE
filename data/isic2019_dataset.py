import sys
sys.path.append(sys.path[0].replace('data', ''))
from configs.default import isic2019_path
from data.flamby_fed_isic2019 import FedIsic2019
from data.meta_dataset import GetDataLoaderDict, dataloader_kwargs
import torch
from torch.nn import functional as F
from torch.nn.modules.loss import _Loss

class ISIC_BaselineLoss(_Loss):
    """Weighted focal loss
    See this [link](https://amaarora.github.io/2020/06/29/FocalLoss.html) for
    a good explanation
    Attributes
    ----------
    alpha: torch.tensor of size 8, class weights
    gamma: torch.tensor of size 1, positive float, for gamma = 0 focal loss is
    the same as CE loss, increases gamma reduces the loss for the "hard to classify
    examples"
    """

    def __init__(self, alpha=torch.tensor([5.5813, 2.0472, 7.0204, 26.1194, 9.5369, 101.0707, 92.5224, 38.3443]), gamma=2.):
        super(ISIC_BaselineLoss, self).__init__()
        self.alpha = alpha.to(torch.float)
        self.gamma = gamma

    def forward(self, inputs, targets):
        """Weighted focal loss function
        Parameters
        ----------
        inputs : torch.tensor of size 8, logits output by the model (pre-softmax)
        targets : torch.tensor of size 1, int between 0 and 7, groundtruth class
        """
        targets = targets.view(-1, 1).type_as(inputs)
        logpt = F.log_softmax(inputs, dim=1)
        logpt = logpt.gather(1, targets.long())
        logpt = logpt.view(-1)
        pt = logpt.exp()
        self.alpha = self.alpha.to(targets.device)
        at = self.alpha.gather(0, targets.data.view(-1).long())
        logpt = logpt * at
        loss = -1 * (1 - pt) ** self.gamma * logpt

        return loss.mean()
    
class Isic2019_FedDG(object):
    def __init__(self, test_domain=0, root_path=isic2019_path, batch_size=2) -> None:
        self.domain_list = [str(i) for i in range(6)]
        self.root_path = root_path
        self.test_domain = str(test_domain) if isinstance(test_domain, int) else test_domain
        self.train_domain_list = [i for i in self.domain_list if i != self.test_domain]
        self.batch_size = batch_size
        self.site_dataset_dict = {}
        self.site_dataloader_dict = {}
        for domain_name in self.domain_list:
            self.site_dataset_dict[domain_name] = self.get_site_dataset(domain_name)
            self.site_dataloader_dict[domain_name] = GetDataLoaderDict(self.site_dataset_dict[domain_name], batch_size=self.batch_size)
        total_train_list = []
        for domain_name in self.domain_list:
            total_train_list.append(self.site_dataset_dict[domain_name]['train'])
        self.total_train_dataset = torch.utils.data.ConcatDataset(total_train_list)
        self.total_train_dataloader = torch.utils.data.DataLoader(self.total_train_dataset, batch_size=self.batch_size, shuffle=True, **dataloader_kwargs)
        
        self.test_dataset = self.site_dataset_dict[self.test_domain]['test']

    def get_site_dataset(self, domain_name):
        domain_name = int(domain_name)
        train_dataset = FedIsic2019(data_path=self.root_path, center=domain_name, train=True, pooled=False)
        val_dataset = FedIsic2019(data_path=self.root_path, center=domain_name, train=False, pooled=False)
        test_dataset = FedIsic2019(data_path=self.root_path, center=domain_name, train=False, pooled=False)
        return {'train': train_dataset, 'val': val_dataset, 'test': test_dataset}
    
    def GetData(self):
        return self.site_dataloader_dict, self.site_dataset_dict


if __name__ == '__main__':
    data_obj = Isic2019_FedDG(test_domain=3, batch_size=64)
    for site_name in data_obj.domain_list:
        print(site_name, len(data_obj.site_dataset_dict[site_name]['train']), len(data_obj.site_dataset_dict[site_name]['val']), len(data_obj.site_dataset_dict[site_name]['test']))
        for batch_data in data_obj.site_dataloader_dict[site_name]['train']:
            imgs, labels = batch_data
            print(imgs.size(), labels.size())
            break