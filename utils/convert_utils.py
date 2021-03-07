import os
import errno
import sys
import time
import math
from collections import OrderedDict
from utils.common import *
import torch
import torch.nn as nn
import torch.nn.init as init

import torchvision.transforms as transforms
import torchvision.datasets as datasets
import torch.utils.data as data

import ast
import json

import numpy as np

# from common import *

import matplotlib
import matplotlib.pyplot as plt

from evaluate import eval_net


def load_model(net,model_path,file_name):
    # Load checkpoint.
    file_path = os.path.join(model_path, file_name)
    if not os.path.exists(file_path):
        raise FileNotFoundError(errno.ENOENT, os.strerror(errno.ENOENT), file_path)

    print('==> Resuming from checkpoint..')
    checkpoint = torch.load(file_path, map_location=lambda storage, loc: storage)
    net.load_state_dict(checkpoint['net'])
    # best_acc = checkpoint['acc']
    start_epoch = checkpoint['epoch']
    # print (net, start_epoch)
    return checkpoint, net


def validate(model,dev_loader,Craters, device='cuda:0'):
    loss = eval_net(model,dev_loader,Craters['dev'],device)
    # print('DeepUnet data val_loss:', val_loss)
    return loss


def has_bn(net):
    for m in net.modules():
        if type(m) == nn.BatchNorm2d:
            return True
    return False


def merge_bn(model, model_nobn):
    "merges bn params with those of the previous layer"
    "works for the layer pattern: conv->bn only"

    # Serialize the original model
    name_to_type = serialize_model(model)
    key_list = list(name_to_type.keys())

    # Serialize the nobn model
    name_to_type_nobn = serialize_model(model_nobn)
    conv_names = []
    for k, v in name_to_type_nobn.items():
        if type(v) == nn.Conv2d or type(v) == nn.Linear:
            conv_names.append(k)

    nobn_num = 0
    layer_num = 0
    for i, n in enumerate(key_list):
        # i=54 OutConv报错
        # if isinstance(name_to_type[n], nn.Conv2d) and \
        #         isinstance(name_to_type[key_list[i + 1]], nn.BatchNorm2d):
        if (i+1 < len(key_list)):
            if  isinstance(name_to_type[key_list[i + 1]], nn.BatchNorm2d)  and isinstance(name_to_type[n], nn.Conv2d):
                conv_layer = name_to_type[n]
                bn_layer = name_to_type[key_list[i + 1]]
                new_wts, new_bias = adjust_weights(conv_layer, bn_layer)

                nobn_name = conv_names[layer_num]
                conv_layer_nobn = name_to_type_nobn[nobn_name]

                conv_layer_nobn.weight.data = new_wts
                if conv_layer_nobn.bias is not None:
                    conv_layer_nobn.bias.data = new_bias

                layer_num += 1

            elif isinstance(name_to_type[n], nn.Conv2d) or \
                    isinstance(name_to_type[n], nn.Linear):
                layer = name_to_type[n]

                nobn_name = conv_names[layer_num]

                layer_nobn = name_to_type_nobn[nobn_name]
                layer_nobn.weight.data = layer.weight.data.clone()
                if layer.bias is not None and layer_nobn.bias is not None:
                    layer_nobn.bias.data = layer.bias.data.clone()

                layer_num += 1

    return model_nobn



def adjust_weights(wt_layer, bn_layer):
    num_out_channels = wt_layer.weight.size()[0]

    bias = torch.zeros(num_out_channels)
    wt_layer_bias = torch.zeros(num_out_channels)
    if wt_layer.bias is not None:
        wt_layer_bias = wt_layer.bias

    wt_cap = torch.zeros(wt_layer.weight.size())
    for i in range(num_out_channels):
        beta, gamma = 0, 1

        if bn_layer.weight is not None:
            gamma = bn_layer.weight[i]
        if bn_layer.bias is not None:
            beta = bn_layer.bias[i]

        sigma = bn_layer.running_var[i]
        mu = bn_layer.running_mean[i]
        eps = bn_layer.eps
        scale_fac = gamma / torch.sqrt(eps+sigma)
        wt_cap[i,:,:,:] = wt_layer.weight[i,:,:,:]*scale_fac
        bias[i] = (wt_layer_bias[i]-mu)*scale_fac + beta
    return (wt_cap, bias)


def save_model(net, state, model_path, file_name):
    assert os.path.isdir(model_path), 'Error: no {} directory found!'.format(model_path)
    file_path = os.path.join(model_path, file_name)
    print('Saving..')
    state['net'] = net.state_dict()
    torch.save(state, file_path)



def compute_thresholds(net,dataloader,out_dir,percentile=99.9,device='cuda:0',spiking=True):
    n_train = 5000
    relues = []
    relus = []
    relu_names = []
    ftr_zeros_dict = {}
    for k,v in net.named_modules():
        if isinstance( v, nn.Conv2d) or \
            isinstance(v, nn.Linear) or \
            isinstance(v,nn.MaxPool2d) or \
            isinstance(v, nn.AdaptiveAvgPool2d) or \
            isinstance(v, nn.AvgPool2d):
            relus.append(v)
            relu_names.append(k)
            ftr_zeros_dict[k] = 0

    hooks = [Hook(layer) for layer in relus]
    print('number of spike layers with thresholds: {}'.format(len(hooks)))

    bs = 4
    dim = 256
    mask_type = torch.float32 if net.n_classes == 1 else torch.long
    net.eval()
    test_loss = 0
    correct = 0
    total = 0
    criterion = nn.BCEWithLogitsLoss()
    # 20 testdate 数据集大小
    acts = np.zeros((len(hooks)+1, n_train))

    with torch.no_grad():
        for n,batch in enumerate(dataloader):
            imgs, true_masks = batch[0], batch[1]
            imgs = torch.as_tensor(imgs)
            imgs = torch.reshape(imgs, (bs, 1, dim, dim))
            imgs = imgs.to(device=device, dtype=torch.float32)
            true_masks = torch.as_tensor(true_masks)
            true_masks = true_masks.to(device=device, dtype=mask_type)

            outputs = net(imgs)
            outputs = torch.reshape(outputs, (bs, dim, dim))
            loss = criterion(outputs, true_masks)

            test_loss += loss.item()
            batch_size = true_masks.size(0)

            img_max = np.amax(imgs.cpu().numpy(), axis=(1,2,3))
            acts[0,n*batch_size:(n+1)*batch_size] = img_max

            for i, hook in enumerate(hooks):
                if len(hook.output.size()) >2:
                    acts[i+1][n*batch_size:(n+1)*batch_size] = np.amax(hook.output.cpu().numpy(),axis=(1,2,3))

                else:
                    acts[i + 1][n * batch_size:(n + 1) * batch_size] = np.amax(hook.output.cpu().numpy(), axis=1)

    max_val = np.percentile(acts, percentile, axis=1)
    print('{}th percentile of max activations: {}'.format(percentile, max_val))

    if spiking:
        thresholds = torch.zeros(len(max_val) - 1)
        for i in range(len(thresholds)):
            thresholds[i] = max_val[i + 1] / max_val[i]
        np.savetxt(os.path.join(out_dir, 'thresholds.txt'), thresholds, fmt='%.5f')
        print('thresholds: ', thresholds)
        filenm = 'max_acts.txt'
    else:
        filenm = 'max_acts_{}.txt'.format(percentile)

    np.savetxt(os.path.join(out_dir, filenm), max_val, fmt='%.5f')
