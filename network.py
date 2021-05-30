from torch.utils.data import *
from torchvision import transforms, utils
from torch import nn
from torch import optim
from torch.autograd import Variable
import torch.nn.functional as F
import numpy as np
import pywt
import torch

class FCN(nn.Module):
    def __init__(self,n_fe=2,n_hidden=64):
        super().__init__()

        self.n_hidden=n_hidden
        self.n_fe=n_fe

        self.norm1_0=nn.BatchNorm1d(64)
        self.norm2_0=nn.BatchNorm1d(64)
        self.norm3_0=nn.BatchNorm1d(64)

        self.conv1_0= nn.Conv1d(n_fe, 64, kernel_size=3, stride=1, padding=4)
        self.conv2_0= nn.Conv1d(64, 64, kernel_size=3, stride=1, padding=2)
        self.conv3_0= nn.Conv1d(64, n_hidden, kernel_size=3, stride=1, padding=1)

    def forward(self, xb):
        xb_single=F.relu(self.norm1_0(self.conv1_0(xb)))
        xb_single=F.relu(self.norm2_0(self.conv2_0(xb_single)))
        xb_single=F.relu(self.norm3_0(self.conv3_0(xb_single)))
        xb_single0=torch.mean(xb_single,2)
        return xb_single0




class FeatureLearning(nn.Module):
    def __init__(self,n_fe=2, n_c=4):
        super().__init__()
        self.backbone = FCN(n_fe)
        self.linear=nn.Linear(64,n_c)

    def forward(self, xb):
        xb = self.backbone(xb)
        xb = F.softmax(self.linear(xb))
        return xb

class MetricLeaning(nn.Module):
    def __init__(self,n_fe=2,n_view=12,n_hidden=64,n_w=58):
        super().__init__()

        self.n_w = n_w
        self.n_fe = n_fe
        self.n_view = n_view
        self.netlist = nn.ModuleList()
        self.n_hidden = n_hidden
        for i in range(self.n_view):
            self.netlist.append(FCN(n_fe))
        self.mahD=nn.ParameterList([nn.Parameter(torch.randn(n_w, n_hidden)) for i in range(n_view)])

    def forward(self, xb):
        num_samples=xb.shape[0]
        afters= torch.randn(num_samples, self.n_view,self.n_w).cuda()
        befores= torch.randn(num_samples, self.n_view,self.n_hidden).cuda()
        for i in range(self.n_view):
            xbi =self.netlist[i](xb[:,:,i,:])
            befores[:,i,:]=xbi
        for i in range(self.n_view):
            xxx = self.mahD[i].mm(befores[:,i,:].t()).t()
            afters[:,i,:]=xxx

        return afters,befores