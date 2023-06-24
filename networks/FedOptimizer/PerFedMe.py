from torch.optim import Optimizer, Adam
import torch
class pFedMeOptimizer(Optimizer):
    def __init__(self, params, lr=0.01, lamda=0.1, mu=0.001):
        if lr < 0.0:
            raise ValueError("Invalid learning rate: {}".format(lr))
        defaults = dict(lr=lr, lamda=lamda, mu=mu)
        super(pFedMeOptimizer, self).__init__(params, defaults)

    def step(self, local_model, device):
        group = None
        weight_update = local_model.copy()
        for group in self.param_groups:
            for p, localweight in zip(group['params'], weight_update):
                localweight = localweight.to(device)
                p.data = p.data - group['lr'] * (p.grad.data + group['lamda'] * (p.data - localweight.data) + group['mu'] * p.data)

        return group['params']
    

class pFedMeAdam(Adam):
    def __init__(self, params, lr=1e-3, betas=(0.9, 0.999), eps=1e-8,
                 weight_decay=0, amsgrad=False, *, maximize: bool = False,lamda=0.1, mu=0.001):
        super().__init__(params, lr, betas, eps, weight_decay, amsgrad, maximize=maximize)
        self.lamda = lamda
        self.mu = mu

    def step(self, local_model, device):
        group = None
        weight_update = local_model.copy()
        for group in self.param_groups:
            for p, localweight in zip(group['params'], weight_update):
                localweight = localweight.to(device)
                # approximate local model
                p.data = p.data - group['lr'] * (self.lamda * (p.data - localweight.data) + self.mu * p.data)
        loss = super().step()
        return group['params'], loss



if __name__ == '__main__':
    import torchvision
    import copy
    model = torchvision.models.resnet18().cuda()
    local_model = copy.deepcopy(list(model.parameters()))
    optimizer = pFedMeAdam(model.parameters(), lr=0.01)
    import torch
    img = torch.zeros(size=[8,3,224,224]).cuda()
    outputs = model(img)
    loss = torch.norm(outputs)
    loss.backward()
    optimizer.step(local_model=local_model, device='cuda')
    