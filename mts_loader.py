from torch.utils.data import *
from torchvision import transforms, utils
from torch import nn
import torch
from torch import optim
from torch.autograd import Variable
import torch.nn.functional as F
import numpy as np
import pywt
import pickle as cp


def wave_trans(data, EMDSIZE,w='db4'):
    """Decompose and plot a signal S.
    S = An + Dn + Dn-1 + ... + D1
    """
    mode = pywt.Modes.smooth
    w = pywt.Wavelet(w)#选取小波函数
    a = data
    ca = []#近似分量
    cd = []#细节分量
    for i in range(EMDSIZE-1):
        (a, d) = pywt.dwt(a, w, mode)
        ca.append(a)
        cd.append(d)
    rec_a = []
    rec_d = []
    for i, coeff in enumerate(ca):
        coeff_list = [coeff, None] + [None] * i
        rec_a.append(pywt.waverec(coeff_list, w))#重构

    for i, coeff in enumerate(cd):
        coeff_list = [None, coeff] + [None] * i
        rec_d.append(pywt.waverec(coeff_list, w))

    rec_d.append(rec_a[EMDSIZE-2])
    maxN,maxL=rec_a[EMDSIZE-2].shape

    results=np.zeros((EMDSIZE,maxN,maxL))
    for i in range(EMDSIZE):
        results[i,:,0:rec_d[i].shape[1]]=rec_d[i]
    results=results.transpose(1,0,2)

    return results
    
def to_device(data, device):
    if isinstance(data, (list,tuple)):
        return [to_device(x, device) for x in data]
    return data.to(device, non_blocking=True)
class DeviceDataLoader():
    def __init__(self,dl,device):
        self.dl=dl
        self.device=device

    def __iter__(self):
        for b in self.dl:
            yield to_device(b,self.device)
    def __len__(self):
        return len(self.dl)

class MTSDataset(Dataset):
    def __init__(self, file_name, selectindex, EMDSIZE,wave = 'db4'):
        x, y, _ = cp.load(open('MTS/' + file_name + '.p', 'rb'), encoding='bytes')

        x = x[selectindex, :, :]
        y = y[selectindex]

        n_samples = x.shape[0]
        n_views = x.shape[2]
        n_length = x.shape[1]

        x = x.transpose(0, 2, 1)
        x = x.reshape((-1, n_length))

        x = wave_trans(x, EMDSIZE,wave).reshape((n_samples, n_views, EMDSIZE, -1))

        y = y - min(y)

        self.y = torch.from_numpy(y).long()
        self.x = torch.from_numpy(x).float()

    def __len__(self):
        return len(self.x)

    def __getitem__(self, idx):
        return self.x[idx], self.y[idx]