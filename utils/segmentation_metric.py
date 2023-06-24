import sys
sys.path.append(sys.path[0].replace('utils', ''))
from data import DiceLoss
import torch.nn.functional as F
import torch
import numpy as np
from scipy import ndimage
from medpy import metric
    
class Segmentation(object):
    def __init__(self):
        self.init()
    
    
    def init(self):
        self.pred_list = []
        self.label_list = []
        self.dice_list = []
        self.loss = 0.
    
    def update(self, pred, label):
        pred = pred.cpu().detach().numpy()
        label = label.cpu().detach().numpy()
        self.pred_list.append(pred)
        self.label_list.append(label)
        
    def results(self):
        y_true_final = np.concatenate(self.label_list)
        y_pred_final = np.concatenate(self.pred_list)
        dice_value  = self.calc_dice(y_pred_final, y_true_final)
        self.init()
        return {'dice': dice_value, 'loss': 1. - dice_value, 'acc': dice_value}
        
    def calc_dice(self, y_pred, y_true):
        if len(y_pred.shape) == 4:
            SPATIAL_DIMENSIONS = 2, 3
        elif len(y_pred.shape) == 5:
            SPATIAL_DIMENSIONS = 2, 3, 4
        intersection = (y_pred * y_true).sum(axis=SPATIAL_DIMENSIONS)
        union = (0.5 * (y_pred + y_true)).sum(axis=SPATIAL_DIMENSIONS)
        dice = intersection / (union + 1.0e-7)
        # If both inputs are empty the dice coefficient should be equal 1
        dice[union == 0] = 1
        
        return np.mean(dice)


class Segmentation2D(Segmentation):
    def __init__(self):
        self.init()
        self.loss_func = DiceLoss().dice_coef
    
    
    def init(self):
        self.pred_list = []
        self.label_list = []
        self.dice_list = []
        self.loss = 0.
        self.samples = 0
    
    def update(self, pred, label):
        self.dice_list.append(self.loss_func(pred, label).item()*pred.shape[0])
        self.samples += pred.shape[0]
        
        
    def results(self):
        dice_score = np.sum(self.dice_list) / self.samples
        self.init()
        return {'dice': dice_score, 'loss': 1. - dice_score, 'acc': dice_score}


class NewSegmentation2D(Segmentation2D):
    def update(self, pred, label):
        for i in range(pred.shape[0]):
            self.dice_list.append(self.loss_func(pred[i].unsqueeze(0), label[i].unsqueeze(0)).item())
        
        self.samples += pred.shape[0]


softmax_helper = lambda x: F.softmax(x, 1)
class Segmentation3D(object):
    def __init__(self):
        self.init()
    
    def init(self):
        self.pred_list = []
        self.label_list = []
        self.dice_list = [] 
        self.dice_tk_list = [] # kidney
        self.dice_tu_list = [] # tumor
    
    def update(self, pred, label):
        pred = pred.detach().cpu()
        preds_softmax = softmax_helper(pred)
        preds = preds_softmax.argmax(1)
        label = label.detach().cpu()
        dice_score, dice_tk, dice_tu = self.metric(preds, label)
        print(dice_score, dice_tk, dice_tu)
        self.dice_list.append(dice_score)
        self.dice_tk_list.append(dice_tk)
        self.dice_tu_list.append(dice_tu)
        self.pred_list.append(pred)
        self.label_list.append(label)
    
    def results(self):
        dice_value = np.mean(self.dice_list)
        dice_tk_value = np.mean(self.dice_tk_list)
        dice_tu_value = np.mean(self.dice_tu_list)
        self.init()
        return {'dice': dice_value, 'loss': 1. - dice_value, 'acc': dice_value, 'dice_tk': dice_tk_value, 'dice_tu': dice_tu_value}
    
    def metric(self, predictions, gt):
        gt = gt.float()
        predictions = predictions.float()
        tk_pd = torch.gt(predictions, 0)
        tk_gt = torch.gt(gt, 0)
        tk_dice, denom, num = self.Dice_coef(tk_pd.float(), tk_gt.float())  # Composite
        tu_dice, denom, num = self.Dice_coef((predictions == 2).float(), (gt == 2).float())

        return (tk_dice+tu_dice)/2, tk_dice, tu_dice
    
    def Dice_coef(self, output, target, eps=1e-5):  # dice score used for evaluation
        target = target.float()
        num = 2 * torch.sum(torch.min(output, target))
        den = torch.sum(output+target) + eps
        return num / den, den, num
        
def _eval_haus(pred_y, gt_y):
    '''
    :param pred: whole brain prediction
    :param gt: whole
    :param detail:
    :return: a list, indicating Dice of each class for one case
    '''
    haus = []

    for cls in range(0,2):

        gt = gt_y[0, cls, ...]
        pred = pred_y[0, cls, ...]
        # def calculate_metric_percase(pred, gt):
        #     dice = metric.binary.dc(pred, gt)
        #     jc = metric.binary.jc(pred, gt)
        #     hd = metric.binary.hd95(pred, gt)
        #     asd = metric.binary.asd(pred, gt)

        # hausdorff_distance_filter = sitk.HausdorffDistanceImageFilter()
        # hausdorff_distance_filter.Execute(gt_i, pred_i)
        # print (gt.shape)
        try:
            haus_cls = metric.binary.hd95(pred, gt)
        except:
            haus_cls = 100.
        haus.append(haus_cls)


    return haus



def _eval_dice(gt_y, pred_y):

    class_map = {  # a map used for mapping label value to its name, used for output
        "0": "disc",
        "1": "cup"
    }

    dice = []

    for cls in range(0,2):

        gt = gt_y[:, cls, ...]
        pred = pred_y[:, cls, ...]


        dice_this = 2*np.sum(gt*pred)/(np.sum(gt)+np.sum(pred) + 1e-8)
        dice.append(dice_this)

    return dice

def _connectivity_region_analysis(mask):
    s = [[0,1,0],
         [1,1,1],
         [0,1,0]]
    label_im, nb_labels = ndimage.label(mask)#, structure=s)

    sizes = ndimage.sum(mask, label_im, range(nb_labels + 1))

    # plt.imshow(label_im)        
    label_im[label_im != np.argmax(sizes)] = 0
    label_im[label_im == np.argmax(sizes)] = 1

    return label_im

def _eval_average_surface_distances(reference, result, voxelspacing=None, connectivity=1):
    """
    The distances between the surface voxel of binary objects in result and their
    nearest partner surface voxel of a binary object in reference.
    """
    return metric.binary.asd(result, reference)

    
    
    