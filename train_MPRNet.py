import os
from config import Config 
opt = Config('training.yml')

gpus = ','.join([str(i) for i in opt.GPU])
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = gpus

import torch
torch.backends.cudnn.benchmark = True

import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader

import random
import time
import numpy as np

import utils
from data_RGB import get_training_data, get_validation_data
from dataset_RGB import read_bin
# from Denoising.net.original import Predictor

# from Denoising.net.deNoise.DeNoise_HDC import HDC_Net
# from Denoising.net.deNoise.HDC_2 import HDC_Net
from Denoising.net.deNoise.HDC_3 import HDC_Net

# from MPRNet import MPRNet
from deNoise.HDC_MPRnet import MPRNet
# import net
# from net.layers import get_norm_layer
from sklearn.model_selection import train_test_split
import losses
from warmup_scheduler import GradualWarmupScheduler
from tqdm import tqdm
from pdb import set_trace as stx
def meg_score(samples_pred, samples_gt):
    # print("pred:{},gt:{}".format(samples_pred.shape,samples_gt.shape))
    means = samples_gt.mean(dim=(1, 2))
    weight = (1 / means) ** 0.5
    diff = torch.abs(samples_pred - samples_gt).mean(dim=(1, 2))
    diff = diff * weight
    score = diff.mean()

    score = torch.log10(100 / score) * 5
    return score
    # print('score', score)
if __name__ == "__main__":
    ######### Set Seeds ###########
    random.seed(1234)
    np.random.seed(1234)
    torch.manual_seed(1234)
    torch.cuda.manual_seed_all(1234)

    start_epoch = 1
    mode = opt.MODEL.MODE
    session = opt.MODEL.SESSION

    result_dir = os.path.join(opt.TRAINING.SAVE_DIR, mode, 'results', session)
    model_dir  = os.path.join(opt.TRAINING.SAVE_DIR, mode, 'models',  session)

    utils.mkdir(result_dir)
    utils.mkdir(model_dir)

    train_dir = opt.TRAINING.TRAIN_DIR
    val_dir   = opt.TRAINING.VAL_DIR

    ######### Model ###########
    # model_restoration = getattr(net, "Unet")
    # model_restoration = model_restoration(inplanes=1, num_classes=1, width=48, norm_layer=get_norm_layer("group"), deep_supervision=False)
    # model_restoration = HDC_Net()
    model_restoration = MPRNet()
    model_restoration.cuda()

    device_ids = [i for i in range(torch.cuda.device_count())]
    if torch.cuda.device_count() > 1:
      print("\n\nLet's use", torch.cuda.device_count(), "GPUs!\n\n")


    new_lr = opt.OPTIM.LR_INITIAL

    optimizer = optim.Adam(model_restoration.parameters(), lr=new_lr, betas=(0.9, 0.999),eps=1e-8, weight_decay=1e-8)


    ######### Scheduler ###########
    warmup_epochs = 3
    scheduler_cosine = optim.lr_scheduler.CosineAnnealingLR(optimizer, opt.OPTIM.NUM_EPOCHS-warmup_epochs+40, eta_min=opt.OPTIM.LR_MIN)
    scheduler = GradualWarmupScheduler(optimizer, multiplier=1, total_epoch=warmup_epochs, after_scheduler=scheduler_cosine)
    scheduler.step()

    ######### Resume ###########
    if opt.TRAINING.RESUME:
        path_chk_rest    = utils.get_last_path(model_dir, '_latest.pth')
        utils.load_checkpoint(model_restoration,path_chk_rest)
        start_epoch = utils.load_start_epoch(path_chk_rest) + 1
        utils.load_optim(optimizer, path_chk_rest)

        for i in range(1, start_epoch):
            scheduler.step()
        new_lr = scheduler.get_lr()[0]
        print('------------------------------------------------------------------------------')
        print("==> Resuming Training with learning rate:", new_lr)
        print('------------------------------------------------------------------------------')

    if len(device_ids)>1:
        model_restoration = nn.DataParallel(model_restoration, device_ids = device_ids)

    ######### Loss ###########
    criterion = losses.CharbonnierLoss()

    img_paths, mask_paths = read_bin(train_dir)

    train_img_paths, val_img_paths, train_mask_paths, val_mask_paths = \
                    train_test_split(img_paths, mask_paths, train_size=0.8, test_size=0.2, random_state=39)

    ######### DataLoaders ###########
    train_dataset = get_training_data(train_img_paths,train_mask_paths, {'patch_size':opt.TRAINING.TRAIN_PS})
    train_loader = DataLoader(dataset=train_dataset, batch_size=opt.OPTIM.BATCH_SIZE, shuffle=True, num_workers=16, drop_last=False, pin_memory=True)

    val_dataset = get_validation_data(val_img_paths,val_mask_paths, {'patch_size':opt.TRAINING.VAL_PS})
    val_loader = DataLoader(dataset=val_dataset, batch_size=1, shuffle=False, num_workers=8, drop_last=False, pin_memory=True)

    print('===> Start Epoch {} End Epoch {}'.format(start_epoch,opt.OPTIM.NUM_EPOCHS + 1))
    print('===> Loading datasets')

    best_psnr = 0
    best_epoch = 0
    best_iter = 0

    eval_now = len(train_loader)//3 - 1
    print(f"\nEval after every {eval_now} Iterations !!!\n")
    mixup = utils.MixUp_AUG()

    for epoch in range(start_epoch, opt.OPTIM.NUM_EPOCHS + 1):
        epoch_start_time = time.time()
        epoch_loss = 0
        train_id = 1

        model_restoration.train()
        for i, data in enumerate(tqdm(train_loader), 0):

            # zero_grad
            for param in model_restoration.parameters():
                param.grad = None

            target = data[0].cuda()
            input_ = data[1].cuda()

            if epoch>5:
                target, input_ = mixup.aug(target, input_)
            # print(input_.shape)
            restored = model_restoration(input_)

            # Compute loss at each stage
            loss = 0
            for output in restored:
                loss += (criterion(output, target))
            # loss = np.sum([criterion(restored[j], target) for j in range(len(restored))])
            loss = loss / len(restored)
            loss.backward()
            optimizer.step()
            epoch_loss +=loss.item()

            #### Evaluation ####
            if i%eval_now==0 and i>0 and (epoch in [1,25,45] or epoch>60):
                model_restoration.eval()
                psnr_val_rgb = []
                sc = []
                for ii, data_val in enumerate((val_loader), 0):
                    target = data_val[0].cuda()
                    input_ = data_val[1].cuda()

                    with torch.no_grad():
                        restored = model_restoration(input_)
                    restored = restored[0]

                    for res,tar in zip(restored,target):
                        # print("res:{},tar:{}".format(res.shape, tar.shape))
                        sco = meg_score(res, tar)
                        sc.append(sco)
                        psnr_val_rgb.append(utils.torchPSNR(res, tar))
                psnr_val_rgb  = torch.stack(psnr_val_rgb).mean().item()
                s = torch.stack(sc).mean().item()
                if psnr_val_rgb > best_psnr:
                    best_psnr = psnr_val_rgb
                    best_epoch = epoch
                    best_iter = i
                    torch.save({'epoch': epoch,
                                'state_dict': model_restoration.state_dict(),
                                'optimizer' : optimizer.state_dict()
                                }, os.path.join(model_dir,"model_best.pth"))

                print("[epoch %d it %d PSNR: %.4f the Score:%.4f--- best_epoch %d best_iter %d Best_PSNR %.4f]" %
                       (epoch, i, psnr_val_rgb, s, best_epoch, best_iter, best_psnr))

                torch.save({'epoch': epoch,
                            'state_dict': model_restoration.state_dict(),
                            'optimizer' : optimizer.state_dict()
                            }, os.path.join(model_dir,f"model_epoch_{epoch}.pth"))

                model_restoration.train()

        scheduler.step()

        print("------------------------------------------------------------------")
        print("Epoch: {}\tTime: {:.4f}\tLoss: {:.4f}\tLearningRate {:.6f}".format(epoch, time.time()-epoch_start_time, epoch_loss / (i+1), scheduler.get_lr()[0]))
        print("------------------------------------------------------------------")

        torch.save({'epoch': epoch,
                    'state_dict': model_restoration.state_dict(),
                    'optimizer' : optimizer.state_dict()
                    }, os.path.join(model_dir,"model_latest.pth"))

