import configparser
import numpy as np
import pandas as pd
import h5py
from processsing import *
from model_train import *
from unet.unet_model_nobn import UNet_nobn

from utils.convert_utils import *
from utils.dataset import *


def get_ann_models(MP):
    for i in range(MP['N_runs']):
        train_and_test_model(Data, Craters, MP, i,device)


if __name__ == '__main__':

    device = 'cuda:0'
    MP = {}
    # Directory of train/dev/test image and crater hdf5 files.
    # MP['dir'] = 'E:/dataset/'
    MP['dir'] = '/home/zkk/SUNET/dataset'
    # Image width/height, assuming square images.
    MP['dim'] = 256
    dir = MP['dir']

    # Batch size: smaller values = less memory but less accurate gradient estimate
    MP['bs'] = 4

    # Number of training epochs.
    MP['epochs'] = 1

    # Number of train/valid/test samples, needs to be a multiple of batch size.
    # MP['n_train'] = 30000
    MP['n_train'] = 5000
    # MP['n_dev'] = 5000
    MP['n_dev'] = 200
    # MP['n_test'] = 5000
    MP['n_test'] = 500

    # Save model (binary flag) and directory.
    MP['save_models'] = 1
    # MP['save_dir'] = 'C:\\Users\\FMSII\\PycharmProjects\\Unet_DeepMoon\\fmsi.h5'


    # Model Parameters (to potentially iterate over, keep in lists).
    MP['N_runs'] = 1  # Number of runs
    MP['filter_length'] = [3]  # Filter length
    MP['lr'] = [0.0001]  # Learning rate
    MP['n_filters'] = [112]  # Number of filters
    MP['init'] = ['he_normal']  # Weight initialization
    MP['lambda'] = [1e-6]  # Weight regularization
    MP['dropout'] = [0.15]  # Dropout fraction

    n_train, n_dev, n_test = MP['n_train'], MP['n_dev'], MP['n_test']
    bs, dim = MP['bs'], MP['dim']
    # Load data
    print("-----------------Load data-----------")
    train = h5py.File('%strain_images.hdf5' % dir, 'r')
    dev = h5py.File('%sdev_images.hdf5' % dir, 'r')
    test = h5py.File('%stest_images.hdf5' % dir, 'r')
    Data = {
        'train': [train['input_images'][:n_train].astype('float32'),
                  train['target_masks'][:n_train].astype('float32')],
        'dev': [dev['input_images'][:n_dev].astype('float32'),
                dev['target_masks'][:n_dev].astype('float32')],
        'test': [test['input_images'][:n_test].astype('float32'),
                 test['target_masks'][:n_test].astype('float32')]
    }
    train.close()
    dev.close()
    test.close()

    # Rescale, normalize, add extra dim
    print('-----------------Preprocess data----------')
    preprocess(Data)

    # Load ground-truth craters
    print('-----------------Load craters-----------------')
    Craters = {
        'train': pd.HDFStore('%strain_craters.hdf5' % dir, 'r'),
        'dev': pd.HDFStore('%sdev_craters.hdf5' % dir, 'r'),
        'test': pd.HDFStore('%stest_craters.hdf5' % dir, 'r')
    }

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


    "train unet model"
    train_ann_model = True
    if train_ann_model == True:
        print("……train ann unet model……")
        get_ann_models(MP)


    print("……load ann unet model……")
    "load model"
    net = UNet_nobn(n_channels=1, n_classes=1, bilinear=True)
    # net = UNet(n_channels=1, n_classes=1, bilinear=True)
    net = net.to(device)
    model_path = '/home/zkk/SUNET/checkpoints'
    # model_path = 'C:\\Users\\FMSII\\Desktop\\SUnet_DeepMoon\\checkpoints'
    file_name = 'unet_deepmoon_nobn.pth'
    state, net = load_model(net,model_path,file_name)
    net = net.to(device)

    "validate original model"
    val_loss = validate(net, dev_loader, Craters, device='cuda:0')
    print('DeepUnet_ANN val_loss_nobn:', val_loss)

    new_net = None
    remove_bn = True
    # if remove_bn:
    #     if has_bn(net):
    #         # only delete BatchNorm2d()
    #         new_net = UNet_nobn(n_channels=1, n_classes=1, bilinear=True)
    #         new_net = new_net.to(device)
    #         # print(new_net)
    #
    #         new_net = merge_bn(net, new_net)
    #
    #         new_net = new_net.to(device)
    #         print('Validating model after delete BN layers...')
    #         val_loss_nobn =  validate(new_net, dev_loader, Craters, device='cuda:0')
    #         print('DeepUnet_ANN val_loss_nobn:', val_loss_nobn)
    #         save_model(new_net, state, model_path, 'nobn_' + file_name)
    #     else:
    #         print('model has no BN layer')

    # #use    the model     without    bn
    # use_nobn = True
    # if use_nobn:
    #     net = new_net
    #
    # "compute thresholds"
    # percentile = 99.9
    # compute_thresholds(net, train_loader, model_path, percentile, device)
    #
    # "convert ann to snn"
    # from spiking import createSpikingModel
    #
    # thresholds = np.loadtxt(os.path.join(model_path, 'thresholds.txt'))
    # max_acts = np.loadtxt(os.path.join(model_path, 'max_acts.txt'))
    #
    # spike_net = createSpikingModel(net,torch.from_numpy(thresholds), max_acts, device)
    # print('spike_net.state_dict().keys():', spike_net.state_dict().keys())
    # print('spike_net:', spike_net)
    #
    #
    # "simulate snn"
    # from spiking import simulate_spike_model
    #
    # thresholds = np.loadtxt(os.path.join(model_path, 'thresholds.txt'))
    # max_acts = np.loadtxt(os.path.join(model_path, 'max_acts.txt'))
    #
    # thresholds = torch.from_numpy(thresholds).to(device)
    # max_acts = torch.from_numpy(max_acts).to(device)
    #
    # sbi, model_partial = None, None
    # img_size = (-1,1,256,256)
    # simulate_spike_model(net,test_loader, thresholds.float(),max_acts, img_size, sbi, model_partial, device)
