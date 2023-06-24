'''
A RUNMC ISBI
B BMC ISBI_1.5
C HCRUDB I2CVB
D UCL
E BIDMC
F HK
'''
import sys, os
sys.path.append(sys.path[0].replace('data', ''))
import numpy as np
from configs.default import prostate_path, prostate_domain_list, dataloader_kwargs
from torch.utils.data import Dataset
import SimpleITK as sitk
import random
import cv2
import torch
import torchvision.transforms as transforms
import torch.nn as nn
import torch.nn.functional as F
from scipy import ndimage

class DiceLoss(nn.Module):
    def __init__(self, smooth=1.0, activation='sigmoid'):
        super(DiceLoss, self).__init__()
        self.smooth = smooth
        self.activation = activation

    def dice_coef(self, pred, gt):
        softmax_pred = torch.nn.functional.softmax(pred, dim=1)
        seg_pred = torch.argmax(softmax_pred, dim=1)
        all_dice = 0
        gt = gt.squeeze(dim=1)
        batch_size = gt.shape[0]
        num_class = softmax_pred.shape[1]
        for i in range(num_class):

            each_pred = torch.zeros_like(seg_pred)
            each_pred[seg_pred==i] = 1

            each_gt = torch.zeros_like(gt)
            each_gt[gt==i] = 1            

        
            intersection = torch.sum((each_pred * each_gt).view(batch_size, -1), dim=1)
            
            union = each_pred.view(batch_size,-1).sum(1) + each_gt.view(batch_size,-1).sum(1)
            dice = (2. *  intersection )/ (union + 1e-5)
         
            all_dice += torch.mean(dice)
 
        return all_dice * 1.0 / num_class


    def forward(self, pred, gt):
        sigmoid_pred = F.softmax(pred,dim=1)

        batch_size = gt.shape[0]
        num_class = sigmoid_pred.shape[1]
    
        bg = torch.zeros_like(gt)
        bg[gt==0] = 1
        label1 = torch.zeros_like(gt)
        label1[gt==1] = 1
        label2 = torch.zeros_like(gt)
        label2[gt == 2] = 1
        label = torch.cat([bg, label1, label2], dim=1)
        
        loss = 0
        smooth = 1e-5

        for i in range(num_class):
            intersect = torch.sum(sigmoid_pred[:, i, ...] * label[:, i, ...])
            z_sum = torch.sum(sigmoid_pred[:, i, ...] )
            y_sum = torch.sum(label[:, i, ...] )
            loss += (2 * intersect + smooth) / (z_sum + y_sum + smooth)
        loss = 1 - loss * 1.0 / num_class
        return loss

class JointLoss(nn.Module):
    def __init__(self):
        super(JointLoss, self).__init__()
        self.ce = nn.CrossEntropyLoss()
        self.dice = DiceLoss()

    def forward(self, pred, gt):
        ce =  self.ce(pred, gt.squeeze(axis=1).long())
        return (ce + self.dice(pred, gt))/2

def GetDataLoaderDict(dataset_dict, batch_size, dataloader_kwargs=dataloader_kwargs):
    dataloader_dict = {}
    for dataset_name in dataset_dict.keys():
        if 'train' in dataset_name:
            dataloader_dict[dataset_name] = torch.utils.data.DataLoader(dataset_dict[dataset_name], batch_size=batch_size, shuffle=True, drop_last=True, **dataloader_kwargs)
        else:
            dataloader_dict[dataset_name] = torch.utils.data.DataLoader(dataset_dict[dataset_name], batch_size=batch_size, shuffle=False, drop_last=False, **dataloader_kwargs)
    return dataloader_dict

class Prostate_FedDG(object):
    def __init__(self, root_path=prostate_path, test_domain='BIDMC', batch_size=16, balanced=True) -> None:
        self.root_path = root_path
        self.test_domain = test_domain
        self.domain_list = prostate_domain_list
        self.batch_size = batch_size
        self.balance = balanced
        
        self.train_domain_list = self.domain_list.copy()
        self.train_domain_list.remove(self.test_domain)
        
        self.site_dataset_dict = {}
        self.site_dataloader_dict = {}
        self.transform = transforms.Compose([
            transforms.ToTensor(),
        ])
        
        for domain_name in self.domain_list:
            self.site_dataset_dict[domain_name] = self.get_single_site(domain_name)
        
        if self.balance:
            min_data_len = min([len(self.site_dataset_dict[domain_name]['train']) for domain_name in self.domain_list])
            for domain_name in self.domain_list:
                self.site_dataset_dict[domain_name]['train'] = torch.utils.data.Subset(self.site_dataset_dict[domain_name]['train'], list(range(int(min_data_len))))
        
        for domain_name in self.domain_list:
            self.site_dataloader_dict[domain_name] = GetDataLoaderDict(self.site_dataset_dict[domain_name], self.batch_size)
        
        self.test_dataset = self.site_dataset_dict[self.test_domain]['total']
    
    def get_single_site(self, domain_name):
        train_dataset = Prostate_SingleSite(domain_name, self.root_path, 'train', self.transform)
        val_dataset = Prostate_SingleSite(domain_name, self.root_path, 'val', self.transform)
        test_dataset = Prostate_SingleSite(domain_name, self.root_path, 'test', self.transform)
        total_dataset = Prostate_SingleSite(domain_name, self.root_path, 'total', self.transform)
        return {'train': train_dataset, 'val': val_dataset, 'test': test_dataset, 'total': total_dataset}

    def GetData(self):
        return self.site_dataloader_dict, self.site_dataset_dict




def convert_from_nii_to_png(img):
    high = np.quantile(img,0.99)
    low = np.min(img)
    img = np.where(img > high, high, img)
    lungwin = np.array([low * 1., high * 1.])
    newimg = (img - lungwin[0]) / (lungwin[1] - lungwin[0])  
    newimg = (newimg * 255).astype(np.uint8)
    return newimg

class Prostate_SingleSite(Dataset):
    def __init__(self, site, root_path=prostate_path, split='train', transform=None):
        channels = {'BIDMC':3, 'HK':3, 'I2CVB':3, 'ISBI':3, 'ISBI_1.5':3, 'UCL':3}
        self.root_path = root_path
        assert site in list(channels.keys())
        self.split = split
        
        
        images, labels = [], []
        sitedir = os.path.join(self.root_path, site)

        ossitedir = np.load(f"{self.root_path}/{site}-dir.npy").tolist()

        for sample in ossitedir:
            sampledir = os.path.join(sitedir, sample)
            if os.path.getsize(sampledir) < 1024 * 1024 and sampledir.endswith("nii.gz"):
                imgdir = os.path.join(sitedir, sample[:6] + ".nii.gz")
                label_v = sitk.ReadImage(sampledir)
                image_v = sitk.ReadImage(imgdir)
                label_v = sitk.GetArrayFromImage(label_v)
                label_v[label_v > 1] = 1
                image_v = sitk.GetArrayFromImage(image_v)
                image_v = convert_from_nii_to_png(image_v)

                for i in range(1, label_v.shape[0] - 1):
                    label = np.array(label_v[i, :, :])
                    if (np.all(label == 0)):
                        continue
                    image = np.array(image_v[i-1:i+2, :, :])
                    image = np.transpose(image,(1,2,0))
                    
                    labels.append(label)
                    images.append(image)
        labels = np.array(labels).astype(int)
        images = np.array(images)

        index = np.load(f"{self.root_path}/{site}-index.npy").tolist()

        labels = labels[index]
        images = images[index]

        trainlen = 0.8 * len(labels) * 0.8
        vallen = 0.8 * len(labels) - trainlen
        testlen = 0.2 * len(labels)
        if(split=='train'):
            self.images, self.labels = images[:int(trainlen)], labels[:int(trainlen)]

        elif(split=='val'):
            self.images, self.labels = images[int(trainlen):int(trainlen + vallen)], labels[int(trainlen):int(trainlen + vallen)]
        elif(split=='test'):
            self.images, self.labels = images[int(trainlen + vallen):], labels[int(trainlen + vallen):]
        elif(split=='total'):
            self.images, self.labels = images, labels
            
        self.transform = transform
        self.channels = channels[site]
        self.labels = self.labels.astype(np.int32).squeeze()

    def __len__(self):
        return self.images.shape[0]

    def __getitem__(self, idx):
        image = self.images[idx]
        label = self.labels[idx]
        contour, bg = _get_coutour_sample(np.expand_dims(label, axis=-1))
        contour_dict = {'contour': contour, 'bg': bg}
        if self.transform is not None:
            if self.split == 'train':
                R1 = RandomRotate90()
                image, label = R1(image, label)
                R2 = RandomFlip()
                image, label = R2(image, label)

            image = np.transpose(image,(2, 0, 1))
            
            image = torch.Tensor(image)
            
            label = self.transform(label)
        
        return image, label, contour_dict

def _get_coutour_sample(y_true):
    mask = np.expand_dims(y_true[..., 0], axis=2)

    erosion = ndimage.binary_erosion(mask[..., 0], iterations=1).astype(mask.dtype)
    dilation = ndimage.binary_dilation(mask[..., 0], iterations=5).astype(mask.dtype)
    contour = np.expand_dims(mask[..., 0] - erosion, axis = 2)
    bg = np.expand_dims(dilation - mask[..., 0], axis = 2)

    return [contour.transpose(2, 0, 1), bg.transpose(2, 0, 1)]

class RandomRotate90:
    def __init__(self, prob=1.0):
        self.prob = prob

    def __call__(self, img, mask=None):
        if random.random() < self.prob:
            factor = random.randint(0, 4)
            img = np.rot90(img, factor)
            if mask is not None:
                mask = np.rot90(mask, factor)
        return img.copy(), mask.copy()

class RandomFlip:
    def __init__(self, prob=0.75):
        self.prob = prob

    def __call__(self, img, mask=None):
        if random.random() < self.prob:
            d = random.randint(-1, 1)
            img = cv2.flip(img, d)
            if mask is not None:
                mask = cv2.flip(mask, d)

        return  img, mask

if __name__=='__main__':
    data_obj = Prostate_FedDG(test_domain='ISBI_1.5', batch_size=16, balanced=True)
    dataloader_dict, dataset_dict = data_obj.GetData()
    for domain_name in data_obj.domain_list:
        print(domain_name, len(dataset_dict[domain_name]['train']), len(dataset_dict[domain_name]['val']), len(dataset_dict[domain_name]['test']))
        
    dataloader = dataloader_dict['HK']['train']
    for i, batch_data in enumerate(dataloader):
        imgs, labels, contour_dict = batch_data
        print(contour_dict)
        