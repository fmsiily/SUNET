import torch
import torch.nn.functional as F
from tqdm import tqdm
from utils.template_match_target import *
from dice_loss import dice_coeff
from  processsing import *
import torch.nn as nn


def eval_net(net, loader,craters , device):
    """Evaluation without the densecrf with the dice coefficient"""
    bs = 4
    dim = 256
    net.eval()
    mask_type = torch.float32 if net.n_classes == 1 else torch.long
    n_val = len(loader)  # the number of batch
    tot = 0
    global_step = 0
    beta = 1
    with tqdm(total=n_val, desc='Validation round', unit='batch', leave=False) as pbar:
        for batch in loader:
            imgs, true_masks = batch[0], batch[1]
            imgs = torch.as_tensor(imgs)
            imgs = torch.reshape(imgs, (bs, 1, dim, dim))
            imgs = imgs.to(device=device, dtype=torch.float32)
            true_masks = torch.as_tensor(true_masks)
            true_masks = true_masks.to(device=device, dtype=mask_type)

            with torch.no_grad():
                mask_pred = net(imgs)
                mask_pred = torch.reshape(mask_pred, (bs, dim, dim))
                criterion = nn.BCEWithLogitsLoss()
                loss = criterion(mask_pred, true_masks)
                pbar.set_postfix(**{'loss (batch)': loss.item()})
                pbar.update(imgs.shape[0])
                pred = torch.sigmoid(mask_pred)
                pred = (pred > 0.5).float()
                # tot += dice_coeff(pred, true_masks).item()
                # Get csvs of human-counted craters
                # csvs = []
                # preds = mask_pred.cuda().data.cpu().numpy()
                #
                # minrad, maxrad, cutrad, n_csvs = 3, 50, 0.8, len(imgs)
                # diam = 'Diameter (pix)'
                # for i in range(n_csvs):
                #     csv = craters[get_id(i)]
                #     # remove small/large/half craters
                #     csv = csv[(csv[diam] < 2 * maxrad) & (csv[diam] > 2 * minrad)]
                #     csv = csv[(csv['x'] + cutrad * csv[diam] / 2 <= dim)]
                #     csv = csv[(csv['y'] + cutrad * csv[diam] / 2 <= dim)]
                #     csv = csv[(csv['x'] - cutrad * csv[diam] / 2 > 0)]
                #     csv = csv[(csv['y'] - cutrad * csv[diam] / 2 > 0)]
                #     if len(csv) < 3:  # Exclude csvs with few craters
                #         csvs.append([-1])
                #     else:
                #         csv_coords = np.asarray((csv['x'], csv['y'], csv[diam] / 2)).T
                #         csvs.append(csv_coords)
                #
                # # Calculate custom metrics
                # print("")
                # print("*********Custom Loss*********")
                # recall, precision, fscore = [], [], []
                # frac_new, frac_new2, maxrad = [], [], []
                # err_lo, err_la, err_r = [], [], []
                # frac_duplicates = []
                # # preds = model.predict(X)
                #
                # for i in range(n_csvs):
                #     if len(csvs[i]) < 3:
                #         continue
                #     (N_match, N_csv, N_detect, maxr,
                #      elo, ela, er, frac_dupes) = template_match_t2c(preds[i], csvs[i],
                #                                                     rmv_oor_csvs=0)
                #     if N_match > 0:
                #         p = float(N_match) / float(N_match + (N_detect - N_match))
                #         r = float(N_match) / float(N_csv)
                #         f = (1 + beta ** 2) * (r * p) / (p * beta ** 2 + r)
                #         diff = float(N_detect - N_match)
                #         fn = diff / (float(N_detect) + diff)
                #         fn2 = diff / (float(N_csv) + diff)
                #         recall.append(r)
                #         precision.append(p)
                #         fscore.append(f)
                #         frac_new.append(fn)
                #         frac_new2.append(fn2)
                #         maxrad.append(maxr)
                #         err_lo.append(elo)
                #         err_la.append(ela)
                #         err_r.append(er)
                #         frac_duplicates.append(frac_dupes)
                #     else:
                #         print("skipping iteration %d,N_csv=%d,N_detect=%d,N_match=%d" %
                #               (i, N_csv, N_detect, N_match))
                #
                # if len(recall) > 3:
                #     print("mean and std of N_match/N_csv (recall) = %f, %f" %
                #           (np.mean(recall), np.std(recall)))
                #     print("""mean and std of N_match/(N_match + (N_detect-N_match))
                #               (precision) = %f, %f""" % (np.mean(precision), np.std(precision)))
                #     print("mean and std of F_%d score = %f, %f" %
                #           (beta, np.mean(fscore), np.std(fscore)))
                #     print("""mean and std of (N_detect - N_match)/N_detect (fraction
                #               of craters that are new) = %f, %f""" %
                #           (np.mean(frac_new), np.std(frac_new)))
                #     print("""mean and std of (N_detect - N_match)/N_csv (fraction of
                #               "craters that are new, 2) = %f, %f""" %
                #           (np.mean(frac_new2), np.std(frac_new2)))
                #     print("median and IQR fractional longitude diff = %f, 25:%f, 75:%f" %
                #           (np.median(err_lo), np.percentile(err_lo, 25),
                #            np.percentile(err_lo, 75)))
                #     print("median and IQR fractional latitude diff = %f, 25:%f, 75:%f" %
                #           (np.median(err_la), np.percentile(err_la, 25),
                #            np.percentile(err_la, 75)))
                #     print("median and IQR fractional radius diff = %f, 25:%f, 75:%f" %
                #           (np.median(err_r), np.percentile(err_r, 25),
                #            np.percentile(err_r, 75)))
                #     print("mean and std of frac_duplicates: %f, %f" %
                #           (np.mean(frac_duplicates), np.std(frac_duplicates)))
                #     print("""mean and std of maximum detected pixel radius in an image =
                #               %f, %f""" % (np.mean(maxrad), np.std(maxrad)))
                #     print("""absolute maximum detected pixel radius over all images =
                #               %f""" % np.max(maxrad))
                #     print("")
                #

    net.eval()
    return loss
