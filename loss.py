
from torch.utils.data import *
from torchvision import transforms, utils
from torch import nn
from torch import optim
from torch.autograd import Variable
import torch.nn.functional as F
import numpy as np
import pywt
import torch
def normalize(x):
    norm = x.norm(dim=1, p=2, keepdim=True)
    x = x.div(norm.expand_as(x)+0.00000001)
    return x

def pairwise_distance(x, y,need_norm=False):
    x=normalize(x)
    y=normalize(y)
    n = x.size(0)
    n2=y.size(0)
    dist = torch.pow(x, 2).sum(dim=1, keepdim=True)
    dist = dist.expand(n, n2)
    dist2 = torch.pow(y, 2).sum(dim=1, keepdim=True)
    dist2 = dist2.expand(n2, n)
    dist=dist+dist2.t()

    dist=dist-2 * torch.mm(x, y.t())
    dist=dist*(dist>0).float()+0.00000001
    dist=torch.sqrt(dist)

    return dist 

class EasyLoss(nn.Module):
    def __init__(self, alpha=1, beta=0, margin=0.5, hard_mining=None,  **kwargs):
        super(EasyLoss, self).__init__()
        self.margin = margin
        self.alpha = alpha
        self.beta = beta
        self.hard_mining = hard_mining

    def forward(self, inputs, targets):
        n = inputs.size(0)
        n_views=inputs.size(1)
        #sim_mat = pairwise_distance(inputs[:,0,:])
        sim_mat=pairwise_distance(inputs[:,0,:], inputs[:,0,:])
        for i in range(n_views):
            if i>0:
                sim_mat=sim_mat+pairwise_distance(inputs[:,i,:], inputs[:,0,:])

        eyes_ = Variable(torch.eye(n, n)).cuda()
        pos_mask = targets.expand(n, n).eq(targets.expand(n, n).t()).float()
        neg_mask = eyes_.eq(eyes_).float() - pos_mask
        pos_mask = pos_mask - eyes_.eq(1).float()
        posd=torch.masked_select(sim_mat,pos_mask==1)
        negd=torch.masked_select(sim_mat,neg_mask==1)
        posd=torch.sum(posd)
        negd=torch.sum(negd)
        loss=(posd-negd)/(n*(n-1))

        return loss