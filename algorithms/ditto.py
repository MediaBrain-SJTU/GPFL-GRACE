import sys
sys.path.append(sys.path[0].replace('algorithms', ''))
from utils.classification_metric import Balance_Classification
from algorithms.fedprox import FedProx_Trainer, FedProx_Prostate_Trainer, FedProx_Fundus_Trainer



class Ditto_Trainer(FedProx_Trainer):
    def run(self):
        for i in range(self.rounds):
            self.train(i)
            local_results = self.val(i, is_global=False, is_test=False) # local model在训练后的表现
            
            # fedavg
            self.aggregation(self.models_dict, self.weight_dict, self.train_domain_list)
            
            global_results = self.val(i, is_global=True, is_test=True) # global model在训练前的表现
            val_acc = global_results['avg_val_acc']
            
            # save checkpoint
            is_best = val_acc > self.best_acc
            self.best_acc = max(val_acc, self.best_acc)
            # save
            self.save_checkpoint(i, self.global_model, {'global':global_results, 'local':local_results}, global_results[self.test_domain]['acc'], self.save_dir, is_best, prefix='global_')
            for domain_name in self.train_domain_list:
                self.save_checkpoint(i, self.models_dict[domain_name], {'global':global_results, 'local':local_results}, global_results[domain_name]['acc'], self.save_dir, is_best, prefix=f'{domain_name}_')
            
            p_corr = self.get_p_fairness(local_results, global_results)
            self.log_file.info(f'Round: {i:3d} | p_corr: {p_corr:.4f}')
            self.log_ten.add_scalar('p_corr', p_corr, i)
            if is_best:
                self.log_file.info(f'Round {i} Get Best Val Acc: {self.best_acc*100.:.2f}%')


class Ditto_ISIC_Trainer(Ditto_Trainer):
    def initialize(self):
        super().initialize()
        self.metric = Balance_Classification()
        

class Ditto_Prostate_Trainer(FedProx_Prostate_Trainer):
    def run(self):
        for i in range(self.rounds):
            self.train(i)
            local_results = self.val(i, is_global=False, is_test=False) # local model在训练后的表现
            
            # fedavg
            self.aggregation(self.models_dict, self.weight_dict, self.train_domain_list)
            
            global_results = self.val(i, is_global=True, is_test=True) # global model在训练前的表现
            val_acc = global_results['avg_val_dice']
            
            # save checkpoint
            is_best = val_acc > self.best_acc
            self.best_acc = max(val_acc, self.best_acc)
            # save
            self.save_checkpoint(i, self.global_model, {'global':global_results, 'local':local_results}, global_results[self.test_domain]['dice'], self.save_dir, is_best, prefix='global_')
            for domain_name in self.train_domain_list:
                self.save_checkpoint(i, self.models_dict[domain_name], {'global':global_results, 'local':local_results}, global_results[domain_name]['dice'], self.save_dir, is_best, prefix=f'{domain_name}_')
            
            p_corr = self.get_p_fairness(local_results, global_results)
            self.log_file.info(f'Round: {i:3d} | p_corr: {p_corr:.4f}')
            self.log_ten.add_scalar('p_corr', p_corr, i)
            if is_best:
                self.log_file.info(f'Round {i} Get Best Val Dice: {self.best_acc*100.:.2f}%')
