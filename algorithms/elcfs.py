import sys
sys.path.append(sys.path[0].replace('algorithms', ''))
from algorithms.fedavg import FedAvg_Trainer, FedAvg_Prostate_Trainer
from utils.classification_metric import Balance_Classification
import torch
import numpy as np
from collections import OrderedDict
from pytorch_metric_learning import losses
from data.Fourier_Aug import Shuffle_Batch_Data, Batch_FFT2_Amp_MixUp, Combine_AmplitudeANDPhase
import copy
from algorithms.perfedavg import PerFedAvg_Trainer
import random

class FA_FedAvg_Trainer(FedAvg_Trainer):
    def get_log_name(self, args, log_num, file_name='train'):
        return super().get_log_name(args, log_num, file_name) + '_fa' + str(self.fa_aug_prob)
    
    def initialize(self):
        self.fa_aug_prob = self.args.fa_aug_prob
        super().initialize()
        
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
            
            shuffle_imgs = Shuffle_Batch_Data(imgs)
            imgs = Batch_FFT2_Amp_MixUp(imgs, shuffle_imgs)
            
            optimizer.zero_grad()
            output = model(imgs)
            loss = self.criterion(output, labels)
            loss.backward()
            optimizer.step()
            self.log_ten.add_scalar(f'{site_name}_train_loss', loss.item(), epoch*len(dataloader)+i)
            self.metric.update(output, labels)
        
        self.log_ten.add_scalar(f'{site_name}_train_acc', self.metric.results()['acc'], epoch)
        scheduler.step()


class ELCFS_ISIC_Trainer(FA_FedAvg_Trainer):
    def initialize(self):
        super().initialize()
        self.metric = Balance_Classification()
        
def extract_contour_embedding(contour_list, embeddings):

    average_embeddings_list = []
    for contour in contour_list:
        contour_embeddings = contour * embeddings
        average_embeddings = torch.sum(contour_embeddings, (-1,-2))/torch.sum(contour, (-1,-2))
        average_embeddings_list.append(average_embeddings)
    return average_embeddings_list

def load_fast_weight(model, fast_weight):
    new_state_dict = OrderedDict()
    model_state_dict = model.state_dict()
    for k, v in model_state_dict.items():
        if 'amp' in k:
            new_state_dict[k] = model_state_dict[k]
        else:
            new_state_dict[k] = fast_weight[k]
    model.load_state_dict(new_state_dict)

class ELCFS_Prostate_Trainer(FedAvg_Prostate_Trainer): # 严格按照FedDG-ELCFS的方式实现
    def site_epoch_train(self, epoch, site_name, data_type='train', model=None):
        dataloader = self.dataloader_dict[site_name][data_type]
        model = self.models_dict[site_name]
        optimizer = self.optimizers_dict[site_name]
        clip_value = self.args.clip_value
        meta_step_size = self.args.meta_step_size
        temperature = 0.05
        cont_loss_func = losses.NTXentLoss(temperature)
        for i_batch, sampled_batch in enumerate(dataloader):
            # obtain training data
            imgs, labels, dist_contour = sampled_batch
            imgs, labels = imgs.to(self.device), labels.to(self.device)
            contour, bg = dist_contour['contour'], dist_contour['bg']

            shuffle_imgs = Shuffle_Batch_Data(imgs)
            aug_imgs = Batch_FFT2_Amp_MixUp(imgs, shuffle_imgs)

            contour, bg = contour.to(self.device), bg.to(self.device)

            outputs_soft_inner, embedding_inner = model(imgs, feature_out=True)
            loss_inner = self.criterion(outputs_soft_inner, labels)
            outputs_soft_outer_1, embedding_outer = model(aug_imgs, feature_out=True) #alpha
            loss_outer_1_dice = self.criterion(outputs_soft_outer_1, labels)

            inner_ct_em, inner_bg_em = \
                extract_contour_embedding([contour, bg], embedding_inner)
            outer_ct_em, outer_bg_em = \
                extract_contour_embedding([contour, bg], embedding_outer)

            ct_em = torch.cat((inner_ct_em, outer_ct_em), 0)
            bg_em = torch.cat((inner_bg_em, outer_bg_em), 0)

            disc_em = torch.cat((ct_em, bg_em), 0)
            label = np.concatenate([np.ones(ct_em.shape[0]), np.zeros(bg_em.shape[0])])
            label = torch.from_numpy(label)

            cont_loss = cont_loss_func(disc_em, label)
            loss_outer = loss_outer_1_dice + cont_loss * 0.1

            total_loss = loss_inner + loss_outer 

            optimizer.zero_grad()
            total_loss.backward()
            optimizer.step()


class FedDG_ELCFS_Trainer(PerFedAvg_Trainer):
    def initialize(self):
        super().initialize()
        self.get_fourier_amg()
        
    def get_fourier_amg(self):
        self.fourier_amg_dict = {}
        for domain_name in self.train_domain_list:
            self.fourier_amg_dict[domain_name] = [] # 是一个list
            dataloader = self.dataloader_dict[domain_name]['train']
            for i, data_list in enumerate(dataloader):
                if len(data_list) == 3:
                    imgs, labels, _ = data_list
                else:
                    imgs, labels = data_list
                imgs, labels = imgs.to(self.device), labels.to(self.device)
                fourier_amg = torch.abs(torch.fft.fftshift(torch.fft.fft2(imgs)))
                # 裁取中心区域 0.01的范围
                h_size, w_size = fourier_amg.shape[-2:]
                h_size = int(h_size * 0.1)
                w_size = int(w_size * 0.1)
                fourier_amg = fourier_amg[:,:,fourier_amg.shape[2]//2 - h_size:fourier_amg.shape[2]//2+h_size+1,fourier_amg.shape[3]//2-w_size:fourier_amg.shape[3]//2+w_size+1]
                for j in range(fourier_amg.shape[0]):
                    self.fourier_amg_dict[domain_name].append(fourier_amg[j])
        print('get fourier amg done!')
    
    def fourier_aug(self, img, site_name):
        selected_domain_list = self.train_domain_list.copy()
        selected_domain_list.remove(site_name)
        random_site = random.choice(selected_domain_list)
        random_fourier_amg = random.sample(self.fourier_amg_dict[random_site], int(img.shape[0]))
        random_fourier_amg = torch.stack(random_fourier_amg, dim=0)
        h_size, w_size = img.shape[-2:]
        h_size = int(h_size * 0.1)
        w_size = int(w_size * 0.1)
        new_amg = torch.abs(torch.fft.fftshift(torch.fft.fft2(img)))
        phase = torch.angle(torch.fft.fft2(img))
        random_lambda = torch.rand(img.shape[0], 1, 1, 1).to(self.device)
        new_amg[:, :,img.shape[2]//2 - h_size:img.shape[2]//2+h_size+1,img.shape[3]//2-w_size:img.shape[3]//2+w_size+1] = random_lambda * random_fourier_amg + (1-random_lambda) * new_amg[:, :,img.shape[2]//2 - h_size:img.shape[2]//2+h_size+1,img.shape[3]//2-w_size:img.shape[3]//2+w_size+1]
        new_fft = Combine_AmplitudeANDPhase(torch.fft.ifftshift(new_amg), phase)
        new_img = torch.real(torch.fft.ifft2(new_fft))
        return new_img
    
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
            # 将数据分为两部分
            # step 1
            loss, output = self.step_train(site_name, model, imgs_1, labels_1)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            # step 2
            new_imgs = self.fourier_aug(imgs_2, site_name)
            optimizer.zero_grad()
            loss, output = self.step_train(site_name, model, new_imgs, labels_2)
            loss.backward()
            for old_param, new_param in zip(model.parameters(), temp_model):
                    old_param.data = new_param.data.clone()
            optimizer.step(beta=self.args.lr)
            
            self.log_ten.add_scalar(f'{site_name}_train_loss', loss.item(), epoch*len(dataloader)+i)
            self.metric.update(output, labels_2)
        
        self.log_ten.add_scalar(f'{site_name}_train_acc', self.metric.results()['acc'], epoch)
        scheduler.step()
    

class FedDG_FA_Trainer(FedAvg_Trainer):
    def initialize(self):
        super().initialize()
        self.get_fourier_amg()
        
    def get_fourier_amg(self):
        self.fourier_amg_list = []
        for domain_name in self.train_domain_list:
            dataloader = self.dataloader_dict[domain_name]['train']
            for i, data_list in enumerate(dataloader):
                if len(data_list) == 3:
                    imgs, labels, _ = data_list
                else:
                    imgs, labels = data_list
                imgs, labels = imgs.to(self.device), labels.to(self.device)
                fourier_amg = torch.abs(torch.fft.fftshift(torch.fft.fft2(imgs)))
                h_size, w_size = fourier_amg.shape[-2:]
                h_size = int(h_size * 0.1)
                w_size = int(w_size * 0.1)
                fourier_amg = fourier_amg[:,:,fourier_amg.shape[2]//2 - h_size:fourier_amg.shape[2]//2+h_size+1,fourier_amg.shape[3]//2-w_size:fourier_amg.shape[3]//2+w_size+1]
                for j in range(fourier_amg.shape[0]):
                    self.fourier_amg_list.append(fourier_amg[j])
        print('get fourier amg done!')
        self.h_size = h_size
        self.w_size = w_size
    
    def fourier_aug(self, img, site_name):
        '''
        根据site_name选取不同的site的幅值信息
        '''
        random_fourier_amg = random.sample(self.fourier_amg_list, int(img.shape[0]))
        random_fourier_amg = torch.stack(random_fourier_amg, dim=0)
        h_size, w_size = self.h_size, self.w_size
        fft_img = torch.fft.fft2(img)
        new_amg = torch.abs(torch.fft.fftshift(fft_img))
        phase = torch.angle(fft_img)
        random_lambda = torch.rand(img.shape[0], 1, 1, 1).to(self.device)
        new_amg[:, :,img.shape[2]//2 - h_size:img.shape[2]//2+h_size+1,img.shape[3]//2-w_size:img.shape[3]//2+w_size+1] = random_lambda * random_fourier_amg + (1-random_lambda) * new_amg[:, :,img.shape[2]//2 - h_size:img.shape[2]//2+h_size+1,img.shape[3]//2-w_size:img.shape[3]//2+w_size+1]
        new_fft = Combine_AmplitudeANDPhase(torch.fft.ifftshift(new_amg), phase)
        new_img = torch.real(torch.fft.ifft2(new_fft))
        return new_img
    
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
            imgs = self.fourier_aug(imgs, site_name)
            loss, output = self.step_train(site_name, model, imgs, labels)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            self.log_ten.add_scalar(f'{site_name}_train_loss', loss.item(), epoch*len(dataloader)+i)
            self.metric.update(output, labels)
        
        self.log_ten.add_scalar(f'{site_name}_train_acc', self.metric.results()['acc'], epoch)
        scheduler.step()
        