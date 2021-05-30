from __future__ import print_function, division
import os
import time

from sklearn.model_selection import RepeatedStratifiedKFold
import warnings
import pywt
import torch
from mts_loader import *
from network import *
from loss import *


mydevice = torch.device("cuda:0")



def adjust_learning_rate(optimizer, epoch, args):
    """Sets the learning rate to the initial LR decayed by 10 every 30 epochs"""
    if epoch != 0 and epoch % args == 0:
        for param_group in optimizer.param_groups:
            param_group['lr'] *= 0.1


def fit_cls(epochs, model, loss_func, opt, train_dl, valid_dl, n_v, val =True):
    for epoch in range(epochs):
        model.train()
        for xb, yb in train_dl:
            output = model(xb[:, :, n_v, :])
            loss = loss_func(output, yb)
        if opt is not None:
            loss.backward()
            opt.step()
            opt.zero_grad()




def loss_batch_metric(model, loss_func, xb, yb, optimizer=None, lamda=1):
    loss_func_cls = F.cross_entropy
    optimizer.zero_grad()

    inputs, _ = model(xb)
    l1 = loss_func(inputs, yb)
    metricLearn_dict = model.state_dict()
    n_views = inputs.size(1)
    nw, nh = metricLearn_dict['mahD.0'].size()
    initW = torch.zeros(nw, nh * n_views).to(mydevice)
    for i in range(n_views):
        initW[:, i * nh:(i + 1) * nh] = metricLearn_dict['mahD.' + str(i)]

    initW = initW.t()
    D = torch.zeros(n_views * nh, n_views * nh).to(mydevice)
    for i in range(n_views):
        for j in range(n_views):
            x = torch.eye(nh).to(mydevice)
            if i == j:
                D[i * nh:(i + 1) * nh, i * nh:(i + 1) * nh] = (x * (n_views - 1))
            else:
                D[i * nh:(i + 1) * nh, j * nh:(j + 1) * nh] = -1 * x

    l3 = torch.trace(torch.mm(torch.mm(initW.t(), D), initW))

    loss = l1 + l3 * lamda
    loss.backward()
    optimizer.step()

    return loss.item()


def evaluate(trainx, trainy, testx, testy, model):

    test_output, _ = model(testx)
    n_views = test_output.size(1)
    simmat = pairwise_distance(test_output[:, 0, :], trainx[:, 0, :])
    for i in range(n_views):
        if i > 0:
            simmat = simmat + pairwise_distance(test_output[:, i, :], trainx[:, i, :])

    equ = torch.argmin(simmat, dim=1)
    preds = trainy[equ]
    rights = torch.sum(preds == testy)

    return rights


def fit_metric(epochs, model, loss_func, opt, train_dl, valid_dl, lamda):
    for epoch in range(epochs):
        with torch.no_grad():
            running_corrects = 0
            trainlist_x = list()
            trainlist_y = list()
            for xb, yb in train_dl:
                train_output, _ = model(xb)
                trainlist_x.append(train_output)
                trainlist_y.append(yb)

            trainlist_x = torch.cat(trainlist_x)
            trainlist_y = torch.cat(trainlist_y)
            c_right = 0
            best_rights = 0
            for xb, yb in valid_dl:
                c_right = c_right + evaluate(trainlist_x, trainlist_y, xb, yb, model)
                best_rights = best_rights + xb.size(0)

        print("Epoch:\t",epoch,"\tAccuracy:\t" ,c_right.cpu().numpy()/best_rights)
        model.train()
        for xb, yb in train_dl:
            loss_batch_metric(model, loss_func, xb, yb, opt, lamda)
        model.eval()





def metric_init(model, train_dl, n_w=58, neednorm=True, lamda=0.1):
    with torch.no_grad():
        trainlist_x = list()
        trainlist_y = list()
        emb_x = list()
        c = 0
        for xb, yb in train_dl:
            c += 1
            _, ttt = model(xb)
            emb_x.append(ttt)
            trainlist_y.append(yb)

        emb_x = torch.cat(emb_x)
        targets = torch.cat(trainlist_y)

        n = emb_x.size(0)
        n_views = emb_x.size(1)
        n_features = emb_x.size(2)

        eyes_ = torch.eye(n, n).to(mydevice)
        pos_mask = targets.expand(n, n).eq(targets.expand(n, n).t()).to(mydevice)
        pos_mask = pos_mask.float()
        neg_mask = eyes_.eq(eyes_).to(mydevice).float() - pos_mask
        pos_mask = pos_mask - eyes_.eq(1).to(mydevice).float()

        mask = torch.zeros(n_views * n_features, n_views * n_features).to(mydevice)
        ONE = torch.ones(n_features, n_features).to(mydevice)
        for i in range(n_views):
            mask[i * n_features:(i + 1) * n_features, i * n_features:(i + 1) * n_features] = ONE
        mask = mask.to(mydevice)
        P = torch.zeros(n_views * n_features, n_views * n_features).to(mydevice)


        for ite_i in range(n):
            for ite_j in range(ite_i):
                if ite_j != ite_i:
                    xi = emb_x[ite_i, :, :].reshape(n_views * n_features, 1)
                    xj = emb_x[ite_j, :, :].reshape(n_views * n_features, 1)
                    pij = torch.trace(torch.mm(xi - xj, (xi - xj).t()))
                    pij = mask * pij
                    if pos_mask[ite_i, ite_j] == 1:
                        P = P + pij
                    else:
                        P = P - pij


        P = P / (n * (n - 1) / 2)
        COV = torch.zeros(n_views * n_features, n_views * n_features).to(mydevice)
        for i in range(n_views):
            cov_single_view = torch.mm(emb_x[:, i, :].t(), emb_x[:, i, :]) / n
            COV[i * n_features:(i + 1) * n_features, i * n_features:(i + 1) * n_features] = cov_single_view.to(
                mydevice) + torch.eye(n_features).to(mydevice)

        D = torch.zeros(n_views * n_features, n_views * n_features).to(mydevice)

        for i in range(n_views):
            for j in range(n_views):
                x = torch.eye(n_features).to(mydevice)
                if i == j:
                    D[i * n_features:(i + 1) * n_features, i * n_features:(i + 1) * n_features] = (
                                x.to(mydevice) * (n_views - 1))
                else:
                    D[i * n_features:(i + 1) * n_features, j * n_features:(j + 1) * n_features] = -1 * x.to(mydevice)

        FINAL = torch.mm(
            torch.inverse(COV), P + lamda * D)

        LD, LV = torch.eig(FINAL, eigenvectors=True)
        LD = torch.sort(LD[:, 0])[1]
        LD = LD[0:n_w]
        LV_unnormalized = torch.index_select(LV, 1, LD)
        V_normalized = torch.zeros_like(LV_unnormalized)

        for i in range(n_w):
            lv_un = LV_unnormalized[:, i].view(-1, 1)
            vvv = lv_un * np.sqrt(n_views) / torch.sqrt(torch.mm(torch.mm(lv_un.t(), COV), lv_un) + 0.000001)
            V_normalized[:, i] = vvv.view(-1)
        return V_normalized


def one_cv(curti, train_index, test_index, n_w=58, lamda=0.1, lrm=0.001, lrf=0.01, batch_size=256, n_out=15,
           epochsf=1, epochsm=200, n_hidden=64,wave = 'db4'):
    # data prepare
    trainD = MTSDataset(datasetlist[curti], train_index, EMDSIZE,wave )
    testD = MTSDataset(datasetlist[curti], test_index, EMDSIZE,wave)
    batch_size = batch_size
    batch_sizeT = batch_size
    if trainD.__len__() < batch_size:
        batch_size = trainD.__len__()
    if testD.__len__() < batch_sizeT:
        batch_sizeT = testD.__len__()
    train_dl = DataLoader(trainD, batch_size=batch_size,
                          shuffle=True, num_workers=0, drop_last=True )
    valid_dl = DataLoader(testD, batch_size=batch_sizeT,
                          shuffle=False, num_workers=0, drop_last=False)
    train_dl = DeviceDataLoader(train_dl, mydevice)
    valid_dl = DeviceDataLoader(valid_dl, mydevice)


    metricLearningModel = MetricLeaning(n_fe=n_features_list[curti], n_view=EMDSIZE).to(mydevice)

    loss_func_feature = F.cross_entropy
    modellist = list()
    import torch
    for iter_view in range(EMDSIZE):
        featureModel = FeatureLearning(n_fe=n_features_list[curti], n_c=n_class_list[curti]).to(mydevice)

        opt_feature = optim.Adam(featureModel.parameters(), lr=lrf)
        fit_cls(epochsf, featureModel, loss_func_feature, opt_feature, train_dl, valid_dl, iter_view)

        modellist.append(featureModel)


    metricLearn_dict = metricLearningModel.state_dict()

    for i in range(EMDSIZE):
        statedict = modellist[i].state_dict()
        for key in metricLearn_dict:
            if not 'mahD' in key:
                metricLearn_dict[key] = statedict['backbone.' + key[len('netlist.') + 2:]]

    initW = metric_init(metricLearningModel, train_dl, n_w, lamda=lamda)
    metricLearningModel.to(mydevice)
    for i in range(EMDSIZE):
        metricLearn_dict['mahD.' + str(i)] = initW[i * n_hidden:(i + 1) * n_hidden, :].t()
    metricLearningModel.load_state_dict(metricLearn_dict)

    opt_metric = optim.Adam(metricLearningModel.parameters(), lr=lrm, weight_decay=0.9)
    loss_func_metric = EasyLoss()

    fit_metric(epochsm, metricLearningModel, loss_func_metric, opt_metric, train_dl, valid_dl, lamda)




datasetlist = [ 'JV', 'Libras', ]
n_features_list = [ 12, 2, ]
n_class_list = [9, 15, ]


EMDSIZE = 3
if __name__ == '__main__':
    for kkk in range(0,  3):
        rskf = RepeatedStratifiedKFold(n_splits=10, n_repeats=10, random_state=0)
        x, y, _ = cp.load(open('MTS/' + datasetlist[kkk] + '.p', 'rb'), encoding='bytes')
        x = x.reshape(x.shape[0], -1)
        for train_index, test_index in rskf.split(x, y):
            one_cv(curti=kkk, train_index=train_index, test_index=test_index,  lrm=0.001,wave = 'db4')

