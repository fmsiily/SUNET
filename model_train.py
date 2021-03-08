import os
import numpy as np
import logging
import torch
import matplotlib
import matplotlib.pyplot as plt
import torch.nn as nn
from torch import optim
from torch.autograd import Variable

from dice_loss import DiceLoss
from unet.unet_model import UNet
import torch
import torch.nn as nn
from torch import optim
from tqdm import tqdm
from torch.utils.tensorboard import SummaryWriter
from torch.utils.data import DataLoader, random_split
from  processsing import *
from unet.unet_model_nobn import UNet_nobn
from utils.dataset import MyDataset
from utils.template_match_target import *
from evaluate import *

dir_checkpoint = 'checkpoints/'


def get_metrics(data, craters, dim, model, beta=1, device='cuda:0', bs=1):
    """Function that prints pertinent metrics at the end of each epoch.

    Parameters
    ----------
    data : hdf5
        Input images.
    craters : hdf5
        Pandas arrays of human-counted crater data.
    dim : int
        Dimension of input images (assumes square).
    model : keras model object
        Keras model
    beta : int, optional
        Beta value when calculating F-beta score. Defaults to 1.
    """
    device = 'cuda:0'

    MP = {}
    # Directory of train/dev/test image and crater hdf5 files.
    MP['dir'] = 'E:/dataset/'

    # Image width/height, assuming square images.
    MP['dim'] = 256

    # Batch size: smaller values = less memory but less accurate gradient estimate
    MP['bs'] = 2

    # Number of training epochs.
    MP['epochs'] = 4

    MP['n_dev'] = 10
    # MP['n_test'] = 5000
    device = 'cuda:0'
    n_samples = MP['n_dev']
    dim, nb_epoch= MP['dim'], MP['epochs']

    dev_dataset = MyDataset(data['dev'][0], data['dev'][1])
    dev_loader = DataLoader(dataset=dev_dataset,  # 加载的数据集（Dataset对象）
                              batch_size=4,  # 一个批量大小
                              shuffle=True,  # 是否打乱数据顺序
                              num_workers=0)  # 使用多进程加载的进程数，0代表不使用多进程（win系统建议改成0）

    bs = 5
    epoch_loss = 0
    global_step = 0
    X, Y = dev_loader[0], dev_loader[1]
    X = torch.tensor(X)
    X = torch.reshape(X, (bs, 1, dim, dim))
    Y = torch.tensor(Y)
    Y = torch.reshape(Y, (bs, 1, dim, dim))
    X = X.to(device)
    Y = Y.to(device)
    model = model.to(device)
    preds = model(X)
    criterion = nn.BCEWithLogitsLoss()
    loss = criterion(preds, Y)
    print('Loss/val', loss.item())
    preds = torch.reshape(preds, (bs, dim, dim))
    preds = preds.cuda().data.cpu().numpy()


    # Get csvs of human-counted craters
    csvs = []
    minrad, maxrad, cutrad, n_csvs = 3, 50, 0.8, len(X)
    diam = 'Diameter (pix)'
    for i in range(n_csvs):
        csv = craters[get_id(i)]
        # remove small/large/half craters
        csv = csv[(csv[diam] < 2 * maxrad) & (csv[diam] > 2 * minrad)]
        csv = csv[(csv['x'] + cutrad * csv[diam] / 2 <= dim)]
        csv = csv[(csv['y'] + cutrad * csv[diam] / 2 <= dim)]
        csv = csv[(csv['x'] - cutrad * csv[diam] / 2 > 0)]
        csv = csv[(csv['y'] - cutrad * csv[diam] / 2 > 0)]
        if len(csv) < 3:  # Exclude csvs with few craters
            csvs.append([-1])
        else:
            csv_coords = np.asarray((csv['x'], csv['y'], csv[diam] / 2)).T
            csvs.append(csv_coords)

    # Calculate custom metrics
    print("")
    print("*********Custom Loss*********")
    recall, precision, fscore = [], [], []
    frac_new, frac_new2, maxrad = [], [], []
    err_lo, err_la, err_r = [], [], []
    frac_duplicates = []
    # preds = model.predict(X)

    for i in range(n_csvs):
        if len(csvs[i]) < 3:
            continue
        (N_match, N_csv, N_detect, maxr,
         elo, ela, er, frac_dupes) = template_match_t2c(preds[i], csvs[i],
                                                        rmv_oor_csvs=0)
        if N_match > 0:
            p = float(N_match) / float(N_match + (N_detect - N_match))
            r = float(N_match) / float(N_csv)
            f = (1 + beta ** 2) * (r * p) / (p * beta ** 2 + r)
            diff = float(N_detect - N_match)
            fn = diff / (float(N_detect) + diff)
            fn2 = diff / (float(N_csv) + diff)
            recall.append(r)
            precision.append(p)
            fscore.append(f)
            frac_new.append(fn)
            frac_new2.append(fn2)
            maxrad.append(maxr)
            err_lo.append(elo)
            err_la.append(ela)
            err_r.append(er)
            frac_duplicates.append(frac_dupes)
        else:
            print("skipping iteration %d,N_csv=%d,N_detect=%d,N_match=%d" %
                  (i, N_csv, N_detect, N_match))

    if len(recall) > 3:
        print("mean and std of N_match/N_csv (recall) = %f, %f" %
              (np.mean(recall), np.std(recall)))
        print("""mean and std of N_match/(N_match + (N_detect-N_match))
              (precision) = %f, %f""" % (np.mean(precision), np.std(precision)))
        print("mean and std of F_%d score = %f, %f" %
              (beta, np.mean(fscore), np.std(fscore)))
        print("""mean and std of (N_detect - N_match)/N_detect (fraction
              of craters that are new) = %f, %f""" %
              (np.mean(frac_new), np.std(frac_new)))
        print("""mean and std of (N_detect - N_match)/N_csv (fraction of
              "craters that are new, 2) = %f, %f""" %
              (np.mean(frac_new2), np.std(frac_new2)))
        print("median and IQR fractional longitude diff = %f, 25:%f, 75:%f" %
              (np.median(err_lo), np.percentile(err_lo, 25),
               np.percentile(err_lo, 75)))
        print("median and IQR fractional latitude diff = %f, 25:%f, 75:%f" %
              (np.median(err_la), np.percentile(err_la, 25),
               np.percentile(err_la, 75)))
        print("median and IQR fractional radius diff = %f, 25:%f, 75:%f" %
              (np.median(err_r), np.percentile(err_r, 25),
               np.percentile(err_r, 75)))
        print("mean and std of frac_duplicates: %f, %f" %
              (np.mean(frac_duplicates), np.std(frac_duplicates)))
        print("""mean and std of maximum detected pixel radius in an image =
              %f, %f""" % (np.mean(maxrad), np.std(maxrad)))
        print("""absolute maximum detected pixel radius over all images =
              %f""" % np.max(maxrad))
        print("")


def train_and_test_model(Data, Craters, MP, i_MP,device='cuda:0'):
    # Static params
    dim, nb_epoch, bs = MP['dim'], MP['epochs'], MP['bs']

    # Iterating params
    FL = get_param_i(MP['filter_length'], i_MP)
    learn_rate = get_param_i(MP['lr'], i_MP)
    n_filters = get_param_i(MP['n_filters'], i_MP)
    init = get_param_i(MP['init'], i_MP)
    lmbda = get_param_i(MP['lambda'], i_MP)
    drop = get_param_i(MP['dropout'], i_MP)

    train_dataset = MyDataset(Data['train'][0], Data['train'][1])
    train_loader = DataLoader(dataset=train_dataset,  # 加载的数据集（Dataset对象）
                              batch_size=4,  # 一个批量大小
                              shuffle=True,  # 是否打乱数据顺序
                              num_workers=0)  # 使用多进程加载的进程数，0代表不使用多进程（win系统建议改成0）

    dev_dataset = MyDataset(Data['dev'][0], Data['dev'][1])
    dev_loader = DataLoader(dataset=dev_dataset,
                              batch_size=4,
                              shuffle=True,
                              num_workers=0)
    test_dataset = MyDataset(Data['dev'][0], Data['dev'][1])
    test_loader = DataLoader(dataset=test_dataset,
                              batch_size=4,
                              shuffle=True,
                              num_workers=0)

    writer = SummaryWriter('./log/ann')
    epoch_loss = 0
    global_step = 0
    n_classes = 1
    # model = UNet(n_channels=1, n_classes=1, bilinear=True)
    model = UNet_nobn(n_channels=1, n_classes=1, bilinear=True)
    optimizer = optim.Adam(model.parameters(), lr=learn_rate)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min' if n_classes > 1 else 'max', patience=2)
    criterion = nn.BCEWithLogitsLoss()
    # criterion = DiceLoss()
    device = 'cuda:0'
    # Main loop
    n_samples = MP['n_train']

    for nb in range(nb_epoch):
        model.train()
        num_train = 0
        circu = 2
        with tqdm(total=n_samples*circu, desc=f'Epoch {nb + 1}/{nb_epoch}', unit='img') as pbar:
            for steps_per_epoch in range(circu):  # 100效果尚可
                for batch in train_loader:
                    imgs = batch[0]
                    imgs = torch.as_tensor(imgs)
                    model = model.to(device)
                    imgs = torch.reshape(imgs,(bs,1,dim,dim))
                    imgs = imgs.to(device)

                    true_masks = batch[1]
                    true_masks = torch.as_tensor(true_masks)
                    true_masks = torch.reshape(true_masks,(bs,1,dim,dim))
                    true_masks = true_masks.to(device)
                    masks_pred = model(imgs)
                    num_train += 1
                    # 改成dice_loss
                    loss = criterion(masks_pred, true_masks)
                    # pred = torch.sigmoid(masks_pred)
                    # pred = (pred > 0.5).float()

                    # epoch_loss += loss.item()
                    writer.add_scalar('Loss/train', loss.item(), global_step)
                    pbar.set_postfix(**{'loss (batch)': loss.item()})

                    optimizer.zero_grad()
                    loss.backward()
                    optimizer.step()
                    nn.utils.clip_grad_value_(model.parameters(), 0.1)


                    pbar.update(imgs.shape[0])
                    global_step += 1
                    if global_step % (n_samples // (10 * bs)) == 0:
                        for tag, value in model.named_parameters():
                            tag = tag.replace('.', '/')
                            writer.add_histogram('weights/' + tag, value.data.cpu().numpy(), global_step)
                            writer.add_histogram('grads/' + tag, value.grad.data.cpu().numpy(), global_step)

                        # val_score = eval_net(model,dev_loader,Craters['dev'],device,bs,dim ,writer)
                        # scheduler.step(val_score)
                        # writer.add_scalar('Loss/dev', val_score, global_step)
                        writer.add_scalar('learning_rate', optimizer.param_groups[0]['lr'], global_step)


                        writer.add_images('images', imgs, global_step)
                        writer.add_images('masks/true', true_masks, global_step)
                        writer.add_images('masks/pred', torch.sigmoid(masks_pred) > 0.5, global_step)

        if MP['save_models'] == 1:
            # model.save(MP['save_dir'])
            state = {'net': model.state_dict(),  'epoch': nb_epoch}
            torch.save(state,dir_checkpoint + 'unet_deepmoon.pth')
            # logging.info(f'Checkpoint {nb + 1} saved !')

    # get_metrics(Data['dev'], Craters['dev'], dim, model, device,bs,writer)



    print("###################################")
    print("##########END_OF_RUN_INFO##########")
    print("""learning_rate=%e, batch_size=%d, filter_length=%e, n_epoch=%d
          n_train=%d, img_dimensions=%d, init=%s, n_filters=%d, lambda=%e
          dropout=%f""" % (learn_rate, bs, FL, nb_epoch, MP['n_train'],
                           MP['dim'], init, n_filters, lmbda, drop))

    # get_metrics(test_loader, Craters['test'], dim, model)
    test_score = eval_net(model, test_loader, Craters['test'], device)
    scheduler.step(test_score)
    writer.add_scalar('Loss/test', test_score, global_step)

    writer.close()
    print("###################################")
    print("###################################")






def get_param_i(param, i):
    """Gets correct parameter for iteration i.

    Parameters
    ----------
    param : list
        List of model hyperparameters to be iterated over.
    i : integer
        Hyperparameter iteration.

    Returns
    -------
    Correct hyperparameter for iteration i.
    """
    if len(param) > i:
        return param[i]
    else:
        return param[0]