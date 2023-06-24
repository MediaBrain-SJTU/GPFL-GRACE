import sys
sys.path.append(sys.path[0].replace('utils', ''))
from data import DiceLoss
import torch.nn.functional as F
import torch
import numpy as np
from sklearn.metrics import roc_auc_score, accuracy_score, precision_score, recall_score, precision_recall_curve, f1_score, roc_curve, average_precision_score, auc, balanced_accuracy_score


class Classification(object):
    def __init__(self):
        self.init()
    
    
    def init(self):
        self.pred_list = []
        self.label_list = []
        self.correct_count = 0
        self.total_count = 0
        self.loss = 0
    
    def update(self, pred, label, easy_model=False):
        pred = pred.cpu()
        label = label.cpu()
        
        if easy_model:
            pass
        else:
            loss = F.cross_entropy(pred, label).item() * len(label)
            self.loss += loss
            pred = pred.data.max(1)[1]
        self.pred_list.extend(pred.numpy())
        self.label_list.extend(label.numpy())
        self.correct_count += pred.eq(label.data.view_as(pred)).sum()
        self.total_count += len(label)
            
    def results(self):
        result_dict = {}
        result_dict['acc'] = float(self.correct_count) / float(self.total_count)
        result_dict['loss'] = float(self.loss) / float(self.total_count)
        self.init()
        return result_dict

class Balance_Classification(object):
    def __init__(self):
        self.init()
    
    def init(self):
        self.pred_list = []
        self.label_list = []
        self.loss_list = []
        
    def update(self, pred, label):
        pred = pred.cpu().detach().numpy()
        label = label.cpu().detach().numpy()
        self.pred_list.append(pred)
        self.label_list.append(label)
        

    def results(self):
        y_true_final = np.concatenate(self.label_list)
        y_pred_final = np.concatenate(self.pred_list)
        y_true_final = y_true_final.reshape(-1)
        y_pred_final = np.argmax(y_pred_final, axis=1)
        acc_value  = balanced_accuracy_score(y_true_final, y_pred_final)
        self.init()
        return {'loss': 1. - acc_value, 'acc': acc_value}
    
class BCE_Classification():
    def __init__(self, num_classes):
        self.init()
        self.num_classes = num_classes
        
    def init(self):
        self.pred_list = []
        self.label_list = []
    
    def update(self, pred, label):
        pred = pred.cpu().detach()
        label = label.cpu().detach()
        pred = torch.sigmoid(pred)
        self.pred_list.append(pred.numpy())
        self.label_list.append(label.numpy())
        
    def results(self):
        result_matrix = np.vstack(self.pred_list)
        label_matrix = np.vstack(self.label_list)
        AUROC_list = []
        AVERAGE_PRECISION_list = []
        ACC_list = []
        PRECISION_list = []
        RECALL_list = []
        F1_list = []
        
        for t in range(self.num_classes):
            y_pred_list = result_matrix[~np.isnan(label_matrix[:,t]), t]
            label_list = label_matrix[~np.isnan(label_matrix[:,t]), t]
            if len(label_list) > 0:
                label_pred = [round(i) for i in y_pred_list]
                # 计算acc 需要pred和label都是int
                acc = accuracy_score(y_pred=label_pred, y_true=label_list)
                precision = precision_score(y_pred=label_pred, y_true=label_list)
                recall = recall_score(y_pred=label_pred, y_true=label_list)
                F1 = f1_score(y_pred=label_pred, y_true=label_list)
                auroc = roc_auc_score(label_list, y_pred_list)
                ap = average_precision_score(y_true=label_list, y_score=y_pred_list)
                
                ACC_list.append(acc)
                PRECISION_list.append(precision)
                RECALL_list.append(recall)
                F1_list.append(F1)
                AUROC_list.append(auroc)
                AVERAGE_PRECISION_list.append(ap)
        results_dict = {'auc_list': AUROC_list, 'auc': np.mean(AUROC_list), 
                        'ap_list': AVERAGE_PRECISION_list, 'ap': np.mean(AVERAGE_PRECISION_list), 
                        'acc_list': ACC_list, 'acc': np.mean(ACC_list), 
                        'precision_list': PRECISION_list, 'precision': np.mean(PRECISION_list), 
                        'recall_list': RECALL_list, 'recall': np.mean(RECALL_list), 
                        'f1_list': F1_list, 'f1': np.mean(F1_list)}
        self.init()
        return results_dict


class Classification_class_level(Classification):
    def __init__(self, num_classes=10):
        super(Classification_class_level, self).__init__()
        self.num_classes = num_classes
    def results(self):
        result_dict = {}
        result_dict['acc'] = float(self.correct_count) / float(self.total_count)
        result_dict['loss'] = float(self.loss) / float(self.total_count)
        result_dict['class_acc'] = {}
        class_level_acc = 0
        self.pred_list = np.array(self.pred_list)
        self.label_list = np.array(self.label_list)
        for i in range(self.num_classes):
            label_i_mask = self.label_list == i
            if np.sum(label_i_mask) == 0:
                class_acc = 1.0
            else:
                class_acc = np.sum(self.pred_list[label_i_mask]==i) / np.sum(label_i_mask)
            class_level_acc += class_acc
            result_dict['class_acc'][i] = class_acc
        class_level_acc /= self.num_classes
        result_dict['class_level_acc'] = class_level_acc
        self.init()
        return result_dict
        
    

class Classification_Dist(object):
    def __init__(self):
        self.init()
    
    
    def init(self):
        self.pred_list = []
        self.label_list = []
        self.loss_list = []
        self.correct_count = 0
        self.total_count = 0
        self.loss = 0
    
    def update(self, pred, label, easy_model=False):
        pred = pred.cpu()
        label = label.cpu()
        
        if easy_model:
            pass
        else:
            loss = F.cross_entropy(pred, label).item() * len(label)
            self.loss += loss
            self.loss_list.append(loss)
            pred = pred.data.max(1)[1]
        self.pred_list.extend(pred.numpy())
        self.label_list.extend(label.numpy())
        self.correct_count += pred.eq(label.data.view_as(pred)).sum()
        self.total_count += len(label)
            
    def results(self):
        result_dict = {}
        result_dict['acc'] = float(self.correct_count) / float(self.total_count)
        result_dict['loss'] = float(self.loss) / float(self.total_count)
        self.init()
        return result_dict
    
    
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


softmax_helper = lambda x: F.softmax(x, 1)
class Segmentation3D(object):
    def __init__(self):
        self.init()
    
    def init(self):
        self.pred_list = []
        self.label_list = []
        self.dice_list = [] 
        self.dice_tk_list = []
        self.dice_tu_list = []
    
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
    
    def Dice_coef(self, output, target, eps=1e-5):
        target = target.float()
        num = 2 * torch.sum(torch.min(output, target))
        den = torch.sum(output+target) + eps
        return num / den, den, num
        