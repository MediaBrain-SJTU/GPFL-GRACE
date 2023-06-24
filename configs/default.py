import os
dataloader_kwargs = {'num_workers': 2, 'pin_memory': True}
prostate_domain_list = ['ISBI', 'ISBI_1.5', 'I2CVB', 'UCL', 'BIDMC', 'HK'] # follow the sort of domain A B C D E F
isic2019_domain_list = [str(i) for i in range(6)]
isic2019_path = 'put your isic2019 path here'
prostate_path = 'put your prostate path here'





