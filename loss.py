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

