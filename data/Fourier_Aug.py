
import torch
import numpy as np
import random

def Shuffle_Batch_Data(data_in):
    len_total = len(data_in)
    idx_list = list(range(len_total))
    random.shuffle(idx_list)
    return data_in[idx_list]

def FFT2_Amp_MixUp(data_original, data_aug, lamda):
    fft_data_original = torch.fft.fft2(data_original)
    fft_data_aug = torch.fft.fft2(data_aug)
    
    aug_amp = lamda*torch.abs(fft_data_original) + (1-lamda)*torch.abs(fft_data_aug)
    fft_mixup_data = torch.mul(aug_amp, torch.exp(1j*torch.angle(fft_data_original)))
    return torch.real(torch.fft.ifft2(fft_mixup_data))

def Combine_AmplitudeANDPhase(amp, phe):
    return torch.mul(amp, torch.exp(1j*phe))

def FFT_Exchange_Amplitude(domain_data1, domain_data2):
    fft_domain1 = torch.fft.fft2(domain_data1)
    fft_domain2 = torch.fft.fft2(domain_data2)
    lamda1 = torch.rand(1)/2 + 0.5 # [0.5, 1.0]
    lamda2 = torch.rand(1)/2 + 0.5
    lamda1, lamda2 = lamda1.to(domain_data1.device), lamda2.to(domain_data2.device)
    cross_amp_domain1 = lamda1*torch.abs(fft_domain2) + (1-lamda1)*torch.abs(fft_domain1)
    cross_amp_domain2 = lamda2*torch.abs(fft_domain1) + (1-lamda2)*torch.abs(fft_domain2)
    cross_domain1 = Combine_AmplitudeANDPhase(cross_amp_domain1, torch.angle(fft_domain1))
    cross_domain2 = Combine_AmplitudeANDPhase(cross_amp_domain2, torch.angle(fft_domain2))
    return torch.real(torch.fft.ifft2(cross_domain1)), torch.real(torch.fft.ifft2(cross_domain2))

def Batch_FFT2_Amp_MixUp(data_original, data_aug, p=0.5):
    apply_p = np.random.rand()
    if apply_p<=p:
        lamda_vector = np.random.rand(data_original.size(0))
        for i in range(data_original.size(0)):
            data_original[i] = FFT2_Amp_MixUp(data_original[i], data_aug[i], lamda_vector[i])
        return data_original
    else:
        return data_original

def GetPatchIndex(h,w,alpha=0.01):
    return [h//2 + 1 - int(alpha*h + 1), h//2 + 1 + int(alpha*h + 1) ,  w//2 + 1 - int(alpha*w + 1), w//2 + 1 + int(alpha*w + 1)]

def GetHW(batch_data):
    data_size = batch_data.size()
    if len(data_size) == 3:
        h,w,c = data_size[1], data_size[2], data_size[0]
    elif len(data_size) == 4:
        h,w,c = data_size[2], data_size[3], data_size[1]
    return h, w, c

def GetAmpPatch(batch_data, alpha=0.01):
    data_size = batch_data.size()
    h,w, _ = GetHW(batch_data)
        
    batch_fft_data = torch.fft.fft2(batch_data)
    batch_fft_data = torch.abs(torch.fft.fftshift(batch_fft_data))
    patch_index = GetPatchIndex(h, w, alpha=alpha)
    chosed_data = batch_fft_data[:,:, patch_index[0]:patch_index[1], patch_index[2]:patch_index[3]]
    return chosed_data

def Vectorize(batch_data):
    data_size = batch_data.size()
    if len(data_size) == 3:
        return batch_data.view(-1)
    elif len(data_size) == 4:
        return batch_data.reshape(data_size[0], -1)

def UnVectorize(batch_data, h, w, c):
    data_size = batch_data.size()
    if len(data_size) == 1 and h*w*c == data_size[0]:
        return batch_data.view(c, h, w)
    elif len(data_size) == 2 and h*w*c == data_size[1]:
        return batch_data.view(-1, c, h, w)

def MixAmpPatch(data_original, amp_patch, lamda=0.1, alpha=0.01):
    patch_size = amp_patch.size()
    data_size = data_original.size()
    h,w,c = GetHW(data_original)
    patch_index = GetPatchIndex(h, w)
    data_original_fft = torch.fft.fft2(data_original)
    data_original_fft = torch.fft.fftshift(data_original_fft)
    data_original_fft_amp = torch.abs(data_original_fft)

    if len(patch_size) == 3 and len(data_size)==3:
        data_original_fft_amp[:,:,patch_index[0]:patch_index[1], patch_index[2]:patch_index[3]] = \
            (1-lamda) * data_original_fft_amp[:,patch_index[0]:patch_index[1], patch_index[2]:patch_index[3]] + lamda * amp_patch
    elif len(patch_size) == 1 and len(data_size)==3:
        amp_patch = UnVectorize(amp_patch, 2*int(h*alpha+1), 2*int(w*alpha+1), c)
        data_original_fft_amp[:,:,patch_index[0]:patch_index[1], patch_index[2]:patch_index[3]] = \
            (1-lamda) * data_original_fft_amp[:,patch_index[0]:patch_index[1], patch_index[2]:patch_index[3]] + lamda * amp_patch
    
    new_data_fft = torch.mul(data_original_fft_amp, torch.exp(1j*torch.angle(data_original_fft)))
    
    new_data_fft = torch.fft.ifftshift(new_data_fft)
    new_data = torch.real(torch.fft.ifft2(new_data_fft))
    
    return new_data


def MixBatchAmpPatch(data_original, amp_patch, lamda=0.1, alpha=0.01):
    patch_size = amp_patch.size()
    data_size = data_original.size()
    h,w,c = GetHW(data_original)
    patch_index = GetPatchIndex(h, w)
    data_original_fft = torch.fft.fft2(data_original)
    data_original_fft = torch.fft.fftshift(data_original_fft)
    data_original_fft_amp = torch.abs(data_original_fft)

    if len(patch_size) == 4 and patch_size[0] == data_size[0]:
        data_original_fft_amp[:,:,patch_index[0]:patch_index[1], patch_index[2]:patch_index[3]] = \
            (1-lamda) * data_original_fft_amp[:,:,patch_index[0]:patch_index[1], patch_index[2]:patch_index[3]] + lamda * amp_patch
    elif len(patch_size) == 2 and patch_size[0] == data_size[0]:
        amp_patch = UnVectorize(amp_patch, 2*int(h*alpha+1), 2*int(w*alpha+1), c)
        data_original_fft_amp[:,:,patch_index[0]:patch_index[1], patch_index[2]:patch_index[3]] = \
            (1-lamda) * data_original_fft_amp[:,:,patch_index[0]:patch_index[1], patch_index[2]:patch_index[3]] + lamda * amp_patch
    
    new_data_fft = torch.mul(data_original_fft_amp, torch.exp(1j*torch.angle(data_original_fft)))
    
    new_data_fft = torch.fft.ifftshift(new_data_fft)
    new_data = torch.real(torch.fft.ifft2(new_data_fft))
    
    return new_data

