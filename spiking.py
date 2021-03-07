import numpy as np
import torch.nn as nn
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm

from unet.unet_model import UNet
from utils import *
from utils.common import *
import math
import os
import json
import matplotlib
import matplotlib.pyplot as plt

from utils.convert_utils import load_model

plt.style.use('ggplot')
import pandas as pd
from unet.unet_model_spiking import *
from unet.spiking_activation import spikeRelu


def createSpikingModel(net, thresholds, max_acts, device='cuda:0'):

    ''' check if model has BN layers '''
    for m in net.modules():
        if isinstance(m,nn.BatchNorm2d):
            print('model UNET has BN layers. Can\'t spikify. Exiting...')
            exit()

    clamp_slope = 0
    reset = 'to-threshold'
    unity_vth = True

    num = 0
    num_to_type = {}
    for name, module in net.named_modules():
        #print(name)
        if type(module) == nn.Conv2d or type(module) == nn.Linear or \
            type(module) == nn.AvgPool2d or type(module) == nn.AdaptiveAvgPool2d\
                or type(module) == nn.MaxPool2d:
            num_to_type[num] = module

            if unity_vth and (type(module) == nn.Conv2d or type(module) == nn.Linear):
                thresholds[num] = 1

            num += 1

    spike_net = None


    spike_net = vgg16_nobn_spike(thresholds, device, clamp_slope, reset, n_channels=1, num_classes=1)


####### copy and adjust weights #######
    if unity_vth:
    # when all vth is normalized to 1 wts normalized by max_acts
        j = 0
        layer_num = 0
        for nm, spk_layer in spike_net.named_modules():
            if isinstance(spk_layer, torch.nn.Conv2d) or isinstance(spk_layer, nn.Linear) or \
                isinstance(spk_layer, nn.AvgPool2d) or isinstance(spk_layer, nn.AdaptiveAvgPool2d) or \
                isinstance(spk_layer,nn.MaxPool2d):

                if isinstance(spk_layer, torch.nn.Conv2d) or isinstance(spk_layer, nn.Linear):
                    L = num_to_type[layer_num]

                    scale = max_acts[j] / max_acts[j+1]
                    spk_layer.weight = torch.nn.Parameter(L.weight * scale)

                    if spk_layer.bias is not None:
                        temp_b = L.bias / max_acts[j+1]
                        spk_layer.bias = torch.nn.Parameter(temp_b)

                    #print('{}, max act indices: {},{}'.format(nm, j, j+1))

                layer_num += 1
                j += 1

    else:
        ## the following works for thresholds = max_acts and same weights
        ## as original ANN
        num = 0
        for name, module in spike_net.named_modules():
            if isinstance(module, torch.nn.Conv2d) or isinstance(module, nn.Linear) or \
                isinstance(module, nn.AvgPool2d) or isinstance(module, nn.AdaptiveAvgPool2d) or\
                    isinstance(module,nn.MaxPool2d):

                if isinstance(module, nn.Conv2d) or isinstance(module, nn.Linear):
                    module.weight.data = num_to_type[num].weight.data.clone()
                    if num_to_type[num].bias is not None and module.bias is not None:
                        temp = num_to_type[num].bias.data.clone()
                        module.bias.data = torch.nn.Parameter(temp / max_acts[num])
                num += 1

    ####### copy and adjust weights (ends) #######
    return spike_net


def create_buffers(net, img_size, device='cuda:0', B=1):
    name_to_type = serialize_model(net)
    key_list = list(name_to_type.keys())
    relus = []
    for i in range(len(key_list)):
        if i < len(key_list)-1 and \
            isinstance(name_to_type[key_list[i]], nn.Conv2d) and \
                (isinstance(name_to_type[key_list[i+1]], nn.ReLU) or \
                isinstance(name_to_type[key_list[i+1]], nn.ReLU6)):
            relus.append(name_to_type[key_list[i+1]])

        elif i < len(key_list)-1 and \
            isinstance(name_to_type[key_list[i]], nn.Linear) and \
                (isinstance(name_to_type[key_list[i+1]],nn.ReLU) or \
                isinstance(name_to_type[key_list[i+1]], nn.ReLU6)):
            relus.append(name_to_type[key_list[i+1]])

        elif isinstance(name_to_type[key_list[i]], nn.Linear) or \
                isinstance(name_to_type[key_list[i]], nn.MaxPool2d) or \
                isinstance(name_to_type[key_list[i]], nn.AdaptiveAvgPool2d):
            relus.append(name_to_type[key_list[i]])

    hooks = [Hook(layer) for layer in relus]
    mats = create_mats(net, img_size, hooks, device, B)

    return hooks, mats


# rand_val corresponds to vmem
# in_val/max_in_val corresponds to threshold
def condition(rand_val, in_val, abs_max_val, MFR=1):
    if rand_val <= (abs(in_val) / abs_max_val) * MFR:
        return (np.sign(in_val))
    else:
        return 0


def poisson_spikes(pixel_vals, MFR):
    " MFR = maximum firing rate "
    " Use when GPU is short of memory. Slower implementation"
    out_spikes = np.zeros(pixel_vals.shape)
    for b in range(pixel_vals.shape[0]):
        random_inputs = np.random.rand(pixel_vals.shape[1],pixel_vals.shape[2], pixel_vals.shape[3])
        single_img = pixel_vals[b,:,:,:]
        max_val = np.amax(abs(single_img)) # note: shouldn't this be max(abs(single_img)) ??
        vfunc = np.vectorize(condition)
        out_spikes[b,:,:,:] = vfunc(random_inputs, single_img, max_val, MFR)
    return out_spikes


def create_mats(net, img_size, hooks, device='cuda:0', B=1):
    inp_size = [B]
    inp_size = inp_size + list(img_size[1:])

    inp = torch.zeros(inp_size).to(device)
    outputs = net(inp.float())

    mats = []
    mats.append(torch.zeros(inp_size).to(device))
    for h in hooks:
        shape = h.output.size()
        if len(shape) > 2:
            curr_shape = [B, shape[1], shape[2], shape[3]]
        else:
            curr_shape = [B, shape[1]]
        temp = torch.zeros(curr_shape).to(device)
        mats.append(temp)

    return mats


def create_spike_buffers(net, img_size, device='cuda:0', B=1):
    relus = []
    for m in net.modules():
        if isinstance(m, spikeRelu):
            relus.append(m)

    hooks = [Hook(layer) for layer in relus]
    mats = create_mats(net, img_size, hooks, device, B)

    return hooks, mats


def simulate_spike_model(net, test_loader, thresholds, max_acts,img_size, sbi, model_partial, device='cuda:0'):

    out_dir =  'C:\\Users\\FMSII\\Desktop\\SUnet_DeepMoon\\checkpoints'

    net = net.to(device)
    batch_size = 4
    class_num = 1
    clamp_slope = 0
    time_window = 10 # 300时间太长
    numBatches = 1
    hybrid = False
    split_layer = 36
    save_activation = True
    save_correlation = True
    write_to_file = True
    MFR = 1
    num_classes = 1
    plot_mean_var = False
    # net.eval()

    spike_net = createSpikingModel(net, thresholds, max_acts, device)
    print('spike_net.eval()')
    spike_net.eval()
    # if plot_mean_var:
    #     plot_mean_var(net, spike_net, out_dir)

    buffers = None
    hooks = None
    num_layers = None
    layers = []
    if save_activation or save_correlation:
        hooks, buffers = create_buffers(net, img_size, device, batch_size)
        num_layers = len(buffers)
        image_corr = np.zeros(numBatches*batch_size)
        layer_corr = np.zeros((num_layers, numBatches*batch_size))

    dim = 256
    bs = 4
    total_correct = 0
    expected_correct = 0
    combined_model_correct = 0
    total_images = 0
    batch_num = 0
    snn_model_accuracy_list = []
    ann_expected_accuracy_list = []
    confusion_matrix = np.zeros([num_classes,num_classes], int)
    out_spikes_t_b_c = torch.zeros(time_window, 4,256,256)
    spike_buffers = None
    epoch_loss = 0
    global_step = 0
    writer = SummaryWriter('./log/spike')
    n_test = 500
    with tqdm(total=n_test, desc='Test spiking round', unit='batch',leave=False) as pbar:
        with torch.no_grad():
            for data in test_loader:

                print ('\n\n------------ inferring batch {} -------------'.format(batch_num))


                # perform inference on the original model to compare
                images = data[0]
                images = torch.as_tensor(images)

                images = torch.reshape(images, (bs, 1, dim, dim))
                images = images.to(device)
                net = net.to(device)
                output_org = net(images)
                true_masks = data[1]
                true_masks = torch.as_tensor(true_masks)
                true_masks = torch.reshape(true_masks, (bs, 1, dim, dim))
                true_masks = true_masks.to(device)
                criterion = nn.BCEWithLogitsLoss()
                ann_loss = criterion(output_org, true_masks)


                # create the spiking model
                spike_net = createSpikingModel(net, thresholds, max_acts, device)
                #sanity_check(net, spike_net, max_acts)
                spike_net = spike_net.to(device)
                spike_net.eval()

                for t in range(time_window):
                    # inp = images[1].cpu().numpy()
                    # inp = np.clip(inp, 0, 1).squeeze()
                    # plt.imshow(inp)
                    # plt.show()
                    # convert image pixels to spiking inputs
                    img = images.cpu().numpy()
                    spikes = poisson_spikes(img, MFR)

                    # supply random inputs
                    spikes = torch.from_numpy(spikes)


                    out_spikes = None


                        # supplying spiking inputs to the spiking model
                    spikes = spikes.to(device)  # 4 1 256 256
                    out_spikes = spike_net(spikes.float()).squeeze()

                    ann_pred = net(images)

                    out_spikes_t_b_c[t,:,:,:] = out_spikes

                total_spikes_b_c = torch.zeros((batch_size, dim,dim))
                for b in range(batch_size):
                    total_spikes_per_input = torch.zeros((dim,dim))
                    for t in range(time_window):
                        total_spikes_per_input += out_spikes_t_b_c[t,b,:,:]
                    #print ("total spikes per output: {}".format(total_spikes_per_input ))
                    total_spikes_b_c[b,:] = total_spikes_per_input
                    total_spikes_b_c[b,:] = total_spikes_per_input / time_window # note the change
                    total_spikes_b_c = total_spikes_b_c.to(device)

                    ann_loss = criterion(ann_pred,true_masks)
                    spike_loss = criterion(total_spikes_b_c,true_masks.squeeze())
                    epoch_loss += spike_loss.item()
                    writer.add_scalar('spike_loss/test', spike_loss.item(), global_step)
                    writer.add_scalar('ann_loss/test', ann_loss.item(), global_step)
                    pbar.set_postfix(**{'loss (batch)': spike_loss.item()})

                    global_step += 1
                    if global_step % (n_test // (1 * bs)) == 0:
                        for tag, value in net.named_parameters():
                            tag = tag.replace('.', '/')
                            writer.add_histogram('weights/' + tag, value.data.cpu().numpy(), global_step)
                            # writer.add_histogram('grads/' + tag, value.grad.data.cpu().numpy(), global_step)

                        writer.add_images('images', images, global_step)
                        writer.add_images('masks/true', true_masks, global_step)
                        writer.add_images('masks/ann_pred', torch.sigmoid(ann_pred)  > 0.5, global_step)
                        writer.add_images('masks/snn_pred', torch.sigmoid(total_spikes_b_c).unsqueeze(dim=1) > 0.5, global_step)




import ast
def plot_correlations(corr, out_dir, config, class_num=-1):
    num_layers,num_samples = corr.shape
    num_imgs = config.getint('num_imgs')
    if num_imgs < 0:
        num_imgs = num_samples

    layer_nums = ast.literal_eval(config['layer_nums'])
    layers = [int(i) for i in layer_nums]
    plot_layers = []
    for i in layers:
        if i <= num_layers:
            plot_layers.append(i)

    csv_dict = {}
    plt.figure()
    #Plot correlations
    for l in plot_layers:
        plt.plot(corr[l][0:num_imgs])
        csv_dict[str(l)] = corr[l][0:num_imgs]
    leg = [str(i) for i in plot_layers]

    plt.legend(leg)
    plt.grid()
    print('Plotting correlations..')
    plt.savefig(os.path.join(out_dir, 'correlation'+str(class_num)+'.png'), bbox_inches='tight')

    #### write the correlation values to a csv file
    pd.DataFrame(csv_dict).to_csv(os.path.join(out_dir, 'correlation'+str(class_num)+'.csv'))


from collections import Counter
def plot_histogram(container, max_acts, spike_config, out_dir, class_num=-1):
    print('[INFO] Plotting histogram of spiking activity..')
    time_window = spike_config.getint('time_window')
    acts = [container[key] for key in container]
    i = 0
    import csv
    with open(os.path.join(out_dir,'spike_hist'+str(class_num)+'.csv'), mode='w') as fl:
        writer = csv.writer(fl, delimiter=',')
        writer.writerow(['0 spikes', '1-10 spikes', '11-50 spikes', '> 50 spikes', 'total spikes'])
        arr = np.zeros((len(acts), 4))
        for a in acts:
            if i > 200:
                break
            a = (acts[i] / max_acts[i]) * time_window
            a = a.flatten()
            recounted = Counter(abs(a))
            sort_rcnt = {}
            for r in sorted(recounted):
                sort_rcnt[r] = recounted[r]
            x = [k for k in sort_rcnt.keys()]
            y = [v for v in sort_rcnt.values()]
            inactive = 0
            ten = 0
            twenty = 0
            rest = 0
            for k,v in sort_rcnt.items():
                if k == 0:
                    inactive += v
                if k > 0 and k <= 10:
                    ten += v
                elif k > 10 and k <= 50:
                    twenty += v
                elif k > 50:
                    rest += v
            arr[i, 0] = inactive
            arr[i, 1] = ten
            arr[i, 2] = twenty
            arr[i, 3] = rest
            writer.writerow([inactive, ten, twenty, rest, sum(y)])

            print('\nlayer {}'.format(i))
            print('# of inactive neurons: {}'.format(inactive))
            print('# neurons with 1-10 spikes: {}'.format(ten))
            print('# neurons with 11-50 spikes: {}'.format(twenty))
            print('# neurons with > 50 spikes: {}'.format(rest))
            i += 1




def plot_activity(container, max_acts, out_dir, class_num=-1):
    acts = [container[key] for key in container]

    plt.figure()
    i = 0
    mean, std = [], []
    for a in acts:
        a = a / max_acts[i]
        #print(a.shape)
        if i == 0:
            b = np.zeros(a.shape)
            b[np.nonzero(a)] = abs(a[np.nonzero(a)])
            std.append(np.std(b))
            mean.append(np.mean(b))
        else:
            std.append(np.std(a))
            mean.append(np.mean(a))

        i += 1
    layer_num = [i for i in range(len(container))]

    plt.plot(layer_num[1:], mean[1:], 'ro', label='mean')
    plt.plot(layer_num[1:], std[1:], 'c^', label='std')

    plt.title('Layer-wise average spiking activity percentage')
    plt.xlabel('layer number')
    plt.ylabel('ratio')
    plt.legend()
    plt.grid()
    print('Plotting activity..')
    plt.savefig(os.path.join(out_dir, 'activity'+str(class_num)+'.png'), bbox_inches='tight')

    #### write the mean spike activity factors to a csv file
    pd.DataFrame(np.asarray(mean)).to_csv(os.path.join(out_dir, 'activity'+str(class_num)+'.csv'))



def plot_mean_var(net, spike_net, out_dir):
    "wt, bias mean and var of the original model"

    wt_mean, wt_var = [], []
    bias_mean, bias_var = [], []
    layer_num, layer_num_b = [], []
    i = 1
    for m in net.modules():
        if type(m) == nn.Conv2d or type(m) == nn.Linear:
            with torch.no_grad():
                layer_num.append(i)
                wt_mean.append(m.weight.mean().cpu().numpy())
                wt_var.append(m.weight.var().cpu().numpy())
                if m.bias is not None:
                    layer_num_b.append(i)
                    bias_mean.append(m.bias.mean().cpu().numpy())
                    bias_var.append(m.bias.var().cpu().numpy())
                i += 1

    "wt, bias mean and var of the spiking model"
    wt_mean_s, wt_var_s = [], []
    bias_mean_s, bias_var_s = [], []
    layer_num_s, layer_num_b_s = [], []
    i = 1
    for m in spike_net.modules():
        if type(m) == nn.Conv2d or type(m) == nn.Linear:
            with torch.no_grad():
                layer_num_s.append(i)
                wt_mean_s.append(m.weight.mean().cpu().numpy())
                wt_var_s.append(m.weight.var().cpu().numpy())
                if m.bias is not None:
                    layer_num_b_s.append(i)
                    bias_mean_s.append(m.bias.mean().cpu().numpy())
                    bias_var_s.append(m.bias.var().cpu().numpy())
                i += 1

    plt.figure()
    plt.subplot(2, 2, 1)
    plt.plot(layer_num, wt_mean, 'ro', label='mean')
    plt.plot(layer_num, wt_var, 'c^', label='variance')
    plt.title('original model weights')
    plt.legend()

    plt.subplot(2, 2, 2)
    if len(bias_mean) > 0:
        plt.plot(layer_num_b, bias_mean, 'go', label='mean')
        plt.plot(layer_num_b, bias_var, 'b^', label='variance')
        plt.title('original model biases')
        plt.legend()

    plt.subplot(2, 2, 3)
    plt.plot(layer_num, wt_mean_s, 'ro', label='mean')
    plt.plot(layer_num, wt_var_s, 'c^', label='variance')
    plt.title('spike model weights')
    plt.legend()
    plt.xlabel('layer number')

    plt.subplot(2, 2, 4)
    if len(bias_mean_s) > 0:
        plt.plot(layer_num_b_s, bias_mean_s, 'go', label='mean')
        plt.plot(layer_num_b_s, bias_var_s, 'b^', label='variance')
        plt.title('spike model biases')
        plt.legend()
        plt.xlabel('layer number')
    plt.grid()

    plt.savefig(os.path.join(out_dir, 'mean_var.png'), bbox_inches='tight')

