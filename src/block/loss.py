import torch
import torch.nn as nn,torch.nn.functional as F
import torch.optim as optim

class ContrastiveLoss(torch.nn.Module):
    """
    Contrastive loss function.
    """
    def __init__(self, margin=2.0):
        super(ContrastiveLoss, self).__init__()
        self.margin = margin

    def forward(self, y_pred, y):
        '''
        y=1 if x1 and x2 match else 0
        '''
        distance =1.0-y_pred
        # distance :(batch,1)
        loss_contrastive = torch.mean(y*torch.pow(distance, 2)\
                           +(1-y)* torch.pow(torch.clamp(self.margin \
                            - distance, min=0.0), 2))
        return loss_contrastive

class FocalLoss(nn.Module):
    '''
    input : pred (shape :b,type:float),target(shape:b,type:float)
    output: out(shape:b,type:float)
    '''
    def __init__(self, gamma=2, logits = True,size_average=True):
        super(FocalLoss, self).__init__()
        self.gamma = gamma
        self.size_average = size_average
        self.logits = logits
        print('focal loss')

    def forward(self, pred, target):
        probs = torch.sigmoid(pred)
        pt = probs.clamp(min=0.0001,max=0.999)
        loss = - (1 - pt) ** self.gamma * target * torch.log(pt) - pt ** self.gamma * (1 - target) * torch.log(1 - pt)
        return torch.mean(loss)
        
 
