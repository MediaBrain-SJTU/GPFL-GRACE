'''
get model专用
'''
import sys
sys.path.append(sys.path[0].replace('networks', ''))
from networks.isic_model import MyEfficientNet
from networks.prostate_model import UNet

def GetNetwork(args, num_classes, pretrained=True, **kwargs):
    if args.model == 'isic_b0':
        model = MyEfficientNet(pretrained=pretrained, num_classes=8)
        feature_level = 1280
    elif args.model == 'prostate_unet':
        model = UNet(input_shape=[3, 384, 384], norm_type=args.unet_norm)
        feature_level = 32 #*384*384
    
    else:
        raise ValueError("The model is not support")

    return model, feature_level


if __name__ == '__main__':
    import torch
    model = MyEfficientNet(pretrained=False, num_classes=8)
    # model = UNet(input_shape=[3, 224, 224])
    imgs = torch.zeros(size=(10, 3, 224, 224))
    output, features = model(imgs, feature_out=True)
    print(features.shape)
    
    